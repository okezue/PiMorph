import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from pimorph.complex.halfedge import FaceKind
from pimorph.complex3d import (
    CellKind3D,
    LineKind3D,
    VertexKind3D,
    boundary_matrices3d,
    euler_characteristic3d,
    extract_complex3d,
    neighbor_counts,
    per_cell_euler,
    surface_boundary_matrices,
    surface_complex_from_labels,
    validate3d,
    validate_surface,
)
from pimorph.complex3d.invariants3d import contact_multiplicity


def voronoi_labels3d(n_seeds: int, shape=(48, 48, 48), seed: int = 0, gaps: int = 0, gap_size: int = 3) -> np.ndarray:
    """Voronoi-like 3-D tessellation via watershed on the seed distance transform."""
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.integers(0, s, n_seeds) for s in shape], axis=1)
    pts = np.unique(pts, axis=0)
    markers = np.zeros(shape, dtype=np.int32)
    markers[pts[:, 0], pts[:, 1], pts[:, 2]] = np.arange(1, pts.shape[0] + 1)
    dist = ndi.distance_transform_edt(markers == 0)
    lab = watershed(dist, markers).astype(np.int32)
    for _ in range(gaps):
        c = [rng.integers(gap_size, s - gap_size) for s in shape]
        lab[c[0] - gap_size : c[0] + gap_size, c[1] - gap_size : c[1] + gap_size, c[2] - gap_size : c[2] + gap_size] = 0
    return lab


def hollow_cylinder(
    n_sectors: int = 6, size: int = 40, height: int = 12, r_in: float = 7.0, r_out: float = 15.0, lumen_label=None
):
    """Angular sectors of an annulus extruded along z. The lumen (r < r_in) gets
    ``lumen_label`` when given, else stays background and merges with the outer cell
    through the two open ends."""
    zz, yy, xx = np.mgrid[:height, :size, :size]
    cy = cx = (size - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)
    ang = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    lab = np.zeros((height, size, size), dtype=np.int32)
    ring = (r >= r_in) & (r < r_out)
    lab[ring] = (ang[ring] / (2 * np.pi) * n_sectors).astype(np.int32) + 1
    if lumen_label is not None:
        lab[r < r_in] = lumen_label
    return lab


def _assert_exact(cx):
    rep = validate3d(cx)
    assert rep.b1b2_zero and rep.b2b3_zero, rep.messages
    assert rep.every_interface_two_cells and rep.loops_match_boundary and rep.every_line_has_interface
    B1, B2, B3 = boundary_matrices3d(cx)
    assert (B1 @ B2).nnz == 0 or np.all((B1 @ B2).toarray() == 0)
    assert (B2 @ B3).nnz == 0 or np.all((B2 @ B3).toarray() == 0)
    return rep


def _assert_valid(cx):
    rep = _assert_exact(cx)
    assert rep.ok, rep.messages
    assert rep.euler_residual == 0
    return rep


# ----------------------------------------------------------------- hand cases
def test_two_cubes_side_by_side():
    lab = np.zeros((6, 6, 10), dtype=np.int32)
    lab[1:5, 1:5, 1:5] = 1
    lab[1:5, 1:5, 5:9] = 2
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_cells == 2 and rep.n_gaps == 0 and cx.n_cells == 3
    assert cx.n_interfaces == 3
    cell_cell = [f for f in range(cx.n_interfaces) if np.all(cx.cell_kind[cx.interface_cells[f]] == CellKind3D.CELL)]
    assert len(cell_cell) == 1 and cx.interface_area[cell_cell[0]] == 16
    # the single triple line is the closed square around the shared face: one artificial vertex
    assert cx.n_lines == 1 and cx.n_vertices == 1
    assert cx.line_tail[0] == cx.line_head[0] and cx.vertex_kind[0] == VertexKind3D.ARTIFICIAL
    assert cx.line_length[0] == 16
    assert euler_characteristic3d(cx, include_outer=True) == 0
    for c in cx.cell_cells:
        assert per_cell_euler(cx, int(c)) == 2
    assert neighbor_counts(cx).tolist()[:2] == [1, 1]


def test_block_of_eight_cubes():
    lab = np.zeros((8, 8, 8), dtype=np.int32)
    k = 1
    for i in range(2):
        for j in range(2):
            for m in range(2):
                lab[1 + 3 * i : 4 + 3 * i, 1 + 3 * j : 4 + 3 * j, 1 + 3 * m : 4 + 3 * m] = k
                k += 1
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_cells == 8 and cx.n_cells == 9
    interior = [f for f in range(cx.n_interfaces) if np.all(cx.cell_kind[cx.interface_cells[f]] == CellKind3D.CELL)]
    assert len(interior) == 12 and cx.n_interfaces == 20
    # centre quadruple point of degree 6 (three axes, both directions), six face-centre points of degree 5
    degs = cx.vertex_degrees()
    assert rep.degree_histogram == {5: 6, 6: 1}
    centre = int(np.flatnonzero(degs == 6)[0])
    assert np.allclose(cx.vertex_xyz[centre], (3.5, 3.5, 3.5))
    assert rep.n_quadruple_points == 7 and cx.n_lines == 18
    assert cx.n_vertices - cx.n_lines + cx.n_interfaces - cx.n_cells == 0
    assert all(per_cell_euler(cx, int(c)) == 2 for c in cx.cell_cells)
    assert neighbor_counts(cx)[cx.cell_cells].tolist() == [3] * 8


def test_sphere_inside_cube_artificial_arc():
    lab = np.zeros((16, 16, 16), dtype=np.int32)
    lab[2:14, 2:14, 2:14] = 1
    zz, yy, xx = np.mgrid[:16, :16, :16]
    lab[(zz - 7.5) ** 2 + (yy - 7.5) ** 2 + (xx - 7.5) ** 2 < 9] = 2
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    # two closed interfaces, each with one artificial arc and two artificial vertices
    assert cx.n_interfaces == 2 and cx.n_lines == 2 and cx.n_vertices == 4
    assert rep.n_artificial_lines == 2 and rep.n_artificial_vertices == 4
    assert np.all(cx.line_kind == LineKind3D.ARTIFICIAL)
    assert all(b.shape[0] == 0 for b in cx.interface_boundary)  # zero boundary chains
    assert all(len(loops) == 1 and len(loops[0]) == 2 for loops in cx.interface_loops)
    assert rep.skeleton_components == 2
    inner = int(np.flatnonzero(cx.cell_label == 2)[0])
    outer_cube = int(np.flatnonzero(cx.cell_label == 1)[0])
    assert per_cell_euler(cx, inner) == 2
    assert per_cell_euler(cx, outer_cube) == 4  # ball with one cavity
    assert not cx.cell_touches_outer(inner) and cx.cell_touches_outer(outer_cube)


def test_empty_and_single_label():
    cx = extract_complex3d(np.zeros((4, 5, 6), dtype=np.int32))
    rep = _assert_valid(cx)
    assert cx.n_cells == 1 and cx.cell_kind[0] == CellKind3D.OUTER and cx.n_interfaces == 0
    cx = extract_complex3d(np.ones((4, 5, 6), dtype=np.int32))
    rep = _assert_valid(cx)
    assert rep.n_cells == 1 and cx.n_interfaces == 1 and cx.interface_area[0] == 2 * (20 + 24 + 30)
    assert per_cell_euler(cx, 0) == 2


def test_gap_and_split_label():
    lab = np.ones((10, 10, 10), dtype=np.int32)
    lab[:, :, 5:] = 2
    lab[4:6, 4:6, 4:6] = 0  # enclosed background straddling the 1|2 contact: a gap 3-cell
    lab[0:2, 0:2, 0:2] = 3
    lab[7:9, 7:9, 7:9] = 3  # same label, two 6-components
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_gaps == 1 and rep.n_cells == 4
    assert cx.provenance["n_split_labels"] == 1
    gap = int(cx.gap_cells[0])
    assert cx.cell_voxels[gap] == 8 and per_cell_euler(cx, gap) == 2 and per_cell_euler(cx, gap, exact=True) == 2
    assert sorted(cx.cell_neighbors(gap).tolist()) == sorted(np.flatnonzero(np.isin(cx.cell_label, [1, 2])).tolist())
    # two annular interfaces: the 1|2 contact around the gap, and the 1|outer surface around the corner block
    assert rep.n_extra_loops == 2 and rep.skeleton_components == 2 and rep.n_artificial_lines == 1
    # cell 2 encloses the second label-3 block: its boundary is two spheres (coarse and exact agree)
    for c in cx.cell_cells:
        expect = 4 if cx.cell_label[c] == 2 else 2
        assert per_cell_euler(cx, int(c)) == per_cell_euler(cx, int(c), exact=True) == expect


def test_interface_faces_encoding_and_centroid():
    lab = np.zeros((5, 5, 8), dtype=np.int32)
    lab[1:4, 1:4, 1:4] = 1
    lab[1:4, 1:4, 4:7] = 2
    cx = extract_complex3d(lab)
    f = int(cx.interfaces_between(0, 1)[0])
    faces = cx.interface_faces(f)
    assert faces.shape == (9, 4) and np.all(faces[:, 0] == 2)  # x-normal faces
    assert np.all(faces[:, 1:] % 2 == np.array([1, 1, 0]))
    assert np.allclose(cx.interface_centroid[f], (2.0, 2.0, 3.5))
    assert contact_multiplicity(cx).tolist() == [1, 1, 1]


def test_annular_contact_and_double_contact():
    lab = np.zeros((6, 10, 6), dtype=np.int32)
    lab[1:5, 1:5, 1:5] = 1
    lab[1:5, 5:9, 1:5] = 2
    lab[2:4, 4:6, 2:4] = 0  # gap enclosed by both cells, punching a hole through their contact
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_gaps == 1
    shared = cx.interfaces_between(0, 1)
    assert shared.size == 1 and len(cx.interface_loops[int(shared[0])]) == 2  # annular contact
    assert rep.n_extra_loops == 1 and rep.euler_residual == 0

    lab[1:5, 4:6, 2:4] = 0  # the channel now reaches the border background: two separate contacts
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_gaps == 0
    shared = cx.interfaces_between(0, 1)
    assert shared.size == 2 and np.all(contact_multiplicity(cx)[shared] == 2)
    assert neighbor_counts(cx)[[0, 1]].tolist() == [1, 1]
    # the union of the two ball cells is a solid torus: the occupied set has chi 0
    assert rep.euler_occupied_set == 0 and rep.n_extra_loops == 2
    assert all(per_cell_euler(cx, c) == per_cell_euler(cx, c, exact=True) == 2 for c in (0, 1))


# --------------------------------------------------------------- tessellation
@pytest.mark.parametrize("seed", [0, 2, 3])
def test_random_voronoi_with_gaps(seed):
    lab = voronoi_labels3d(40, shape=(48, 48, 48), seed=seed, gaps=3)
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert rep.n_cells >= 35 and rep.n_gaps >= 1
    for c in range(cx.n_cells - 1):  # cells and gaps
        assert per_cell_euler(cx, c) == 2
        assert per_cell_euler(cx, c, exact=True) == 2
    assert rep.n_quadruple_points > 0 and rep.n_triple_lines > rep.n_quadruple_points
    assert rep.euler_occupied_set == 1 and rep.n_islands == 1
    nc = neighbor_counts(cx)[cx.cell_cells]
    assert nc.min() >= 1 and nc.mean() > 4


def test_voronoi_seed_with_tunnel_cell_is_flagged():
    """Watershed seed 1 produces one cell with a tunnel (a solid torus). The exact
    boundary Euler characteristic is 0 there and the global residual is nonzero,
    while the boundary operators stay exact."""
    lab = voronoi_labels3d(40, shape=(48, 48, 48), seed=1, gaps=3)
    cx = extract_complex3d(lab)
    rep = _assert_exact(cx)
    exact = np.array([per_cell_euler(cx, c, exact=True) for c in range(cx.n_cells - 1)])
    assert np.sum(exact == 0) == 1 and np.all(np.isin(exact, [0, 2]))
    assert not rep.ok and rep.euler_residual != 0


def test_voronoi_counts_and_polylines():
    lab = voronoi_labels3d(30, shape=(32, 32, 32), seed=3)
    cx = extract_complex3d(lab)
    _assert_valid(cx)
    for e in range(cx.n_lines):
        p = cx.line_polyline[e]
        assert p.shape == (cx.line_length[e] + 1, 3)
        assert np.allclose(p[0], cx.vertex_xyz[cx.line_tail[e]]) and np.allclose(p[-1], cx.vertex_xyz[cx.line_head[e]])
        assert np.all(np.abs(np.diff(p, axis=0)).sum(axis=1) == 1)
    assert cx.interface_face_ptr[-1] == cx.face_dg.shape[0] == cx.interface_area.sum()


# ------------------------------------------------------------------ property
@settings(max_examples=60, deadline=None)
@given(
    hnp.arrays(
        dtype=np.int32,
        shape=st.tuples(st.integers(3, 8), st.integers(3, 8), st.integers(3, 8)),
        elements=st.integers(0, 3),
    )
)
def test_property_boundary_operators_exact(lab):
    """Random volumes contain pinches, checkerboard edges, handles and cavities; the
    boundary operators must stay exact regardless. The Euler check is reported but
    only asserted through its own consistency (it flags handles legitimately)."""
    cx = extract_complex3d(lab)
    rep = _assert_exact(cx)
    assert rep.n_cells + rep.n_gaps + 1 == cx.n_cells
    assert cx.interface_area.sum() == cx.provenance["n_voxel_faces"]
    assert np.all(cx.line_length >= 1)
    assert np.all(cx.vertex_degrees() >= 1)


def test_torus_cell_is_flagged_by_euler_only():
    lab = np.zeros((6, 9, 9), dtype=np.int32)
    lab[1:5, 1:8, 1:8] = 1
    lab[1:5, 3:6, 3:6] = 0  # tunnel through the whole height: a solid torus touching the border background
    cx = extract_complex3d(lab)
    rep = _assert_exact(cx)
    assert rep.n_gaps == 0 and rep.n_cells == 1
    # the closed torus interface carries a slit and is counted as a sphere by the
    # coarse complex; the exact voxel count sees the handle
    assert per_cell_euler(cx, 0) == 2 and per_cell_euler(cx, 0, exact=True) == 0
    assert rep.euler_occupied_set == 0
    assert not rep.ok and rep.euler_residual == 1


def test_corner_touching_cubes_are_separate_islands():
    lab = np.zeros((6, 6, 6), dtype=np.int32)
    lab[1:3, 1:3, 1:3] = 1
    lab[3:5, 3:5, 3:5] = 2
    cx = extract_complex3d(lab)
    rep = _assert_valid(cx)
    assert cx.n_interfaces == 2 and rep.n_artificial_lines == 2
    assert rep.skeleton_components == 2 and rep.n_islands == 1 and rep.euler_occupied_set == 1
    assert cx.n_vertices - cx.n_lines + cx.n_interfaces - cx.n_cells == 1


# -------------------------------------------------------------------- surface
def test_surface_hollow_cylinder_explicit_lumen():
    n = 6
    lab = hollow_cylinder(n_sectors=n, lumen_label=99)
    sc = surface_complex_from_labels(lab, lumen_label=99)
    rep = validate_surface(sc)
    assert rep.ok, rep.messages
    assert len(sc.cell_faces) == n and sc.n_faces == n + 1
    B1, B2 = surface_boundary_matrices(sc)
    assert (B1 @ B2).nnz == 0
    # cylinder sheet: chi 0; n radial edges plus 2n rim arcs; 2n trivalent vertices
    assert rep.euler_characteristic == 0
    assert sc.n_edges == 3 * n and sc.n_vertices == 2 * n
    assert rep.degree_histogram == {3: 2 * n}
    assert rep.n_boundary_edges == 2 * n
    interior = [e for e in range(sc.n_edges) if e not in set(sc.boundary_edges().tolist())]
    assert len(interior) == n
    for e in interior:
        a, b = sc.edge_faces[e]
        assert sc.face_kind[a] == FaceKind.CELL and sc.face_kind[b] == FaceKind.CELL and a != b
    for f in sc.cell_faces:
        assert (
            sorted(sc.face_neighbors(int(f))) == sorted(set(sc.face_neighbors(int(f))))
            and len(sc.face_neighbors(int(f))) == 2
        )
    assert sc.vertex_xyz.shape == (2 * n, 3)
    assert set(np.round(sc.vertex_xyz[:, 0], 1).tolist()) == {-0.5, 11.5}  # both open ends


def test_surface_hollow_cylinder_lumen_merged_with_outer():
    n = 5
    lab = hollow_cylinder(n_sectors=n)
    sc = surface_complex_from_labels(lab)
    rep = validate_surface(sc)
    assert rep.ok, rep.messages
    assert len(sc.cell_faces) == n
    # each sector's outer interface is a band whose two boundary loops are the closed
    # side-contact rims: n loop edges, one artificial vertex each, no free rim
    assert sc.n_edges == n and rep.n_boundary_edges == 0
    assert np.all(sc.edge_tail == sc.edge_head)
    assert all(len(sc.face_loops[int(f)]) == 2 for f in sc.cell_faces)
    assert sc.face_boundary[sc.outer_face].shape[0] == 0
