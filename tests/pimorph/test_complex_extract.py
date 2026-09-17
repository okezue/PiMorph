import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pimorph.complex import (
    FaceKind,
    HalfEdgeComplex,
    VertexKind,
    boundary_matrices,
    clique_vertex_report,
    defect_law_residual,
    euler_characteristic,
    extract_complex,
    to_multigraph,
    to_simple_graph,
    validate,
    weaire_sum_rule_residual,
)
from pimorph.complex.geometry import (
    face_area,
    face_perimeter,
    polyline_length,
    smooth_complex,
)

from .conftest import voronoi_labels


def _assert_valid(cx: HalfEdgeComplex):
    rep = validate(cx)
    assert rep.ok, rep.messages
    assert rep.b1b2_zero
    assert rep.euler_residual == 0
    assert defect_law_residual(cx) == 0
    assert abs(weaire_sum_rule_residual(cx)) < 1e-9
    return rep


# ----------------------------------------------------------------- hand cases
def test_quadrants_topology(quadrants):
    cx = extract_complex(quadrants)
    rep = _assert_valid(cx)
    assert rep.n_cells == 4 and rep.n_gaps == 0 and cx.n_faces == 5
    assert cx.n_vertices == 5 and cx.n_edges == 8
    assert rep.degree_histogram == {3: 4, 4: 1}
    assert rep.n_boundary_edges == 4
    # the central vertex is incident to all four cells
    center = [v for v in range(cx.n_vertices) if cx.vertex_degree(v) == 4][0]
    assert cx.vertex_cell_set(center) == frozenset(range(4))
    # no 3-cliques in a 4-cycle adjacency
    rep_c = clique_vertex_report(cx)
    assert rep_c["n_3cliques"] == 0 and rep_c["n_tricellular_vertices"] == 1


def test_island_artificial_vertex(island):
    cx = extract_complex(island)
    rep = _assert_valid(cx)
    assert cx.n_vertices == 1 and cx.n_edges == 1 and cx.n_faces == 2
    assert rep.n_artificial_vertices == 1
    assert cx.vertex_kind[0] == VertexKind.ARTIFICIAL
    assert cx.edge_tail[0] == cx.edge_head[0]
    # loop polyline closes and is CCW around the cell (positive area)
    p = cx.edge_polyline[0]
    assert np.allclose(p[0], p[-1])
    assert face_area(cx, int(cx.cell_faces[0]), smoothed=False) == pytest.approx(16 * 16)


def test_empty_image():
    cx = extract_complex(np.zeros((7, 9), dtype=np.int32))
    rep = _assert_valid(cx)
    assert cx.n_faces == 1 and cx.face_kind[0] == FaceKind.OUTER
    assert rep.n_cells == 0 and cx.n_edges == 0


def test_hole_and_island(hole_and_island):
    cx = extract_complex(hole_and_island)
    rep = _assert_valid(cx)
    assert rep.n_cells == 2 and rep.n_gaps == 1
    big = [f for f in cx.cell_faces if cx.face_label[f] == 1][0]
    gap = int(cx.gap_faces[0])
    assert len(cx.face_loops[big]) == 2  # outer loop + hole loop
    assert len(cx.face_loops[gap]) == 2  # boundary with cell 1 + boundary with the island
    assert rep.skeleton_components == 3
    # hole loop is subtracted: polygon area equals pixel count
    assert face_area(cx, big, smoothed=False) == pytest.approx((26 * 26) - (10 * 10))
    assert face_area(cx, gap, smoothed=False) == pytest.approx((10 * 10) - (4 * 4))


def test_pinch_splits_label_and_makes_degree4_vertex(pinch):
    cx = extract_complex(pinch)
    rep = _assert_valid(cx)
    assert rep.n_cells == 4
    assert cx.provenance["n_split_labels"] == 1
    assert sorted(cx.face_label[cx.cell_faces].tolist()) == [1, 1, 2, 3]
    assert 4 in rep.degree_histogram


def test_double_contact_two_edges(double_contact):
    cx = extract_complex(double_contact)
    rep = _assert_valid(cx)
    a = [f for f in cx.cell_faces if cx.face_label[f] == 1][0]
    b = [f for f in cx.cell_faces if cx.face_label[f] == 2][0]
    assert cx.edges_between(a, b).size == 2
    assert rep.n_gaps == 1
    G = to_multigraph(cx)
    assert G.number_of_edges(a, b) == 2
    S = to_simple_graph(cx)
    assert S[a][b]["n_components"] == 2


def test_honeycomb_trivalent(honeycomb):
    cx = extract_complex(honeycomb)
    _assert_valid(cx)
    interior = [v for v in range(cx.n_vertices) if all(cx.face_kind[f] == FaceKind.CELL for f in cx.vertex_faces(v))]
    assert len(interior) > 20
    assert all(cx.vertex_degree(v) == 3 for v in interior)
    # interior cells are hexagons
    inner_cells = [f for f in cx.cell_faces if not cx.face_touches_outer(int(f))]
    sides = [cx.face_sides(int(f)) for f in inner_cells]
    assert len(inner_cells) > 10
    assert np.mean(sides) == pytest.approx(6.0, abs=0.05)
    # every 3-clique of interior cells is realized by a common vertex
    rep_c = clique_vertex_report(cx)
    assert rep_c["n_3cliques"] > 0
    assert rep_c["n_3cliques_without_common_vertex"] == 0


# --------------------------------------------------------------- orientation
def test_cell_loops_are_ccw_and_areas_match_pixels(voronoi_small):
    cx = extract_complex(voronoi_small)
    _assert_valid(cx)
    counts = np.bincount(voronoi_small.ravel())
    for f in cx.cell_faces:
        f = int(f)
        assert face_area(cx, f, smoothed=False) == pytest.approx(counts[cx.face_label[f]])


def test_left_face_convention(quadrants):
    cx = extract_complex(quadrants)
    # For every forward half-edge, the pixel just left of the first crack segment
    # must belong to edge_faces[e, 0].
    for e in range(cx.n_edges):
        p = cx.edge_polyline[e]
        mid = 0.5 * (p[0] + p[1])
        d = p[1] - p[0]
        left = mid + 0.5 * np.array([-d[1], d[0]])  # left normal on screen
        r, c = int(round(left[0])), int(round(left[1]))
        if 0 <= r < 20 and 0 <= c < 20:
            lab = quadrants[r, c]
            f = int(cx.edge_faces[e, 0])
            assert cx.face_label[f] == lab


# ---------------------------------------------------------------- geometry
def test_diagonal_arclength_bias_removed():
    lab = np.zeros((100, 100), dtype=np.int32)
    rr, cc = np.mgrid[:100, :100]
    lab[rr + cc < 100] = 1
    lab[rr + cc >= 100] = 2
    cx = extract_complex(lab)
    smooth_complex(cx, iterations=4)
    e = int(cx.cell_cell_edges()[0])
    true = 99 * np.sqrt(2)
    raw = polyline_length(cx.edge_polyline[e])
    sm = polyline_length(cx.edge_smooth[e])
    assert raw / true > 1.35  # staircase bias of pixel counting
    assert abs(sm / true - 1) < 0.03


def test_disk_perimeter_and_area():
    lab = np.zeros((200, 200), dtype=np.int32)
    rr, cc = np.mgrid[:200, :200]
    lab[(rr - 100) ** 2 + (cc - 100) ** 2 < 60**2] = 1
    cx = extract_complex(lab)
    smooth_complex(cx, iterations=4)
    f = int(cx.cell_faces[0])
    assert abs(face_perimeter(cx, f) / (2 * np.pi * 60) - 1) < 0.02
    assert abs(face_area(cx, f) / (np.pi * 60**2) - 1) < 0.01


def test_pixel_size_propagates():
    lab = np.zeros((10, 10), dtype=np.int32)
    lab[:, :5] = 1
    lab[:, 5:] = 2
    cx = extract_complex(lab, pixel_size_um=0.5)
    assert cx.pixel_size_um == 0.5
    assert cx.shape == (10, 10)


# ---------------------------------------------------------- serialization
def test_roundtrip_dict(voronoi_small):
    cx = extract_complex(voronoi_small, pixel_size_um=0.3)
    smooth_complex(cx)
    d = cx.to_dict()
    cx2 = HalfEdgeComplex.from_dict(d)
    assert validate(cx2).ok
    assert cx2.n_edges == cx.n_edges and cx2.n_vertices == cx.n_vertices
    assert np.allclose(cx2.vertex_xy, cx.vertex_xy)
    assert cx2.pixel_size_um == 0.3


# --------------------------------------------------------- incidence matrices
def test_boundary_matrices_shapes_and_signs(quadrants):
    cx = extract_complex(quadrants)
    B1, B2 = boundary_matrices(cx)
    assert B1.shape == (cx.n_vertices, cx.n_edges)
    assert B2.shape == (cx.n_edges, cx.n_faces)
    assert (B1 @ B2).nnz == 0
    # every edge column of B1 sums to zero (or is empty for a loop)
    assert np.all(np.asarray(B1.sum(axis=0)).ravel() == 0)
    # every edge appears in exactly two faces with opposite orientation
    assert np.all(np.asarray(abs(B2).sum(axis=1)).ravel() == 2)
    assert np.all(np.asarray(B2.sum(axis=1)).ravel() == 0)


def test_euler_without_outer_is_one_when_connected(honeycomb):
    cx = extract_complex(honeycomb)
    assert validate(cx).skeleton_components == 1
    assert euler_characteristic(cx, include_outer=False) == 1
    assert euler_characteristic(cx, include_outer=True) == 2


# ---------------------------------------------------------- property tests
@settings(max_examples=40, deadline=None)
@given(
    labels=hnp.arrays(
        dtype=np.int32,
        shape=st.tuples(st.integers(3, 14), st.integers(3, 14)),
        elements=st.integers(0, 4),
    )
)
def test_random_label_images_always_valid(labels):
    cx = extract_complex(labels)
    _assert_valid(cx)
    # face count = 4-connected components of nonzero labels + enclosed bg comps + 1
    from skimage.measure import label as cc

    n_cells = int(cc(labels, connectivity=1, background=0).max())
    assert int(cx.cell_faces.size) == n_cells
    # pixel areas add up
    counts = np.bincount(labels.ravel(), minlength=5)
    total_poly = sum(face_area(cx, int(f), smoothed=False) for f in cx.cell_faces)
    assert total_poly == pytest.approx(counts[1:].sum())


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(5, 60), gaps=st.integers(0, 4))
def test_random_voronoi_valid(seed, n, gaps):
    lab = voronoi_labels(n, shape=(96, 96), seed=seed, gaps=gaps)
    cx = extract_complex(lab)
    _assert_valid(cx)
