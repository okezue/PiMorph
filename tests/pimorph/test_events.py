import numpy as np
import pytest

from pimorph.complex import FaceKind, defect_law_residual, extract_complex, topological_charge, validate
from pimorph.complex.events import (
    EventPreconditionError,
    EventResult,
    contact_birth,
    contact_death,
    divide,
    extrude,
    nucleate_gap,
    reseal,
    rupture,
    t1_exchange,
)
from pimorph.complex.geometry import face_area, polyline_length

from .conftest import voronoi_labels


# ------------------------------------------------------------------ helpers
def _counts(cx):
    return cx.n_vertices, cx.n_edges, cx.n_faces


def _sides(cx):
    return {int(f): cx.face_sides(int(f)) for f in cx.cell_faces}


def _assert_event(res: EventResult, expected):
    assert isinstance(res, EventResult)
    assert res.report.ok, res.report.messages
    assert res.delta_vef == expected
    assert res.expected_delta_vef == expected
    assert defect_law_residual(res.cx) == 0
    assert validate(res.cx).ok
    assert "before" in res.participants and "after" in res.participants
    assert res.cx.provenance["events"][-1]["event"] == res.event


def _all_cells(cx, faces):
    return all(cx.face_kind[f] == FaceKind.CELL for f in faces)


def _generic_edges(cx, interior_only=True):
    """Cell-cell edges with trivalent endpoints and four distinct cell wedges."""
    out = []
    for e in cx.cell_cell_edges():
        e = int(e)
        u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
        if u == w or cx.vertex_degree(u) != 3 or cx.vertex_degree(w) != 3:
            continue
        faces = set(cx.vertex_faces(u)) | set(cx.vertex_faces(w))
        if len(faces) != 4 or not _all_cells(cx, faces):
            continue
        if interior_only and any(cx.face_touches_outer(f) for f in faces):
            continue
        out.append(e)
    return out


def _generic_edge(cx):
    edges = _generic_edges(cx)
    assert edges, "no generic T1 edge found"
    return edges[0]


def _interior_faces(cx):
    """Cells with one loop, trivalent corners and distinct non-outer neighbours."""
    out = []
    for f in cx.cell_faces:
        f = int(f)
        if cx.face_touches_outer(f) or len(cx.face_loops[f]) != 1:
            continue
        if not all(cx.vertex_degree(v) == 3 for v in cx.face_vertices(f)):
            continue
        nb = cx.face_neighbors(f, kinds=(FaceKind.CELL, FaceKind.GAP))
        if len(nb) != cx.face_sides(f) or len(set(nb)) != len(nb):
            continue
        out.append(f)
    return out


def _interior_face(cx):
    faces = _interior_faces(cx)
    assert faces, "no interior face found"
    return faces[0]


def _tricellular_vertex(cx):
    for v in range(cx.n_vertices):
        if cx.vertex_degree(v) != 3:
            continue
        faces = cx.vertex_faces(v)
        if len(set(faces)) == 3 and _all_cells(cx, faces) and not any(cx.face_touches_outer(f) for f in faces):
            return v
    raise AssertionError("no interior tricellular vertex found")


def _boundary_vertex(cx):
    outer = cx.outer_face
    for v in range(cx.n_vertices):
        if cx.vertex_degree(v) == 3 and outer in cx.vertex_faces(v):
            return v
    raise AssertionError("no trivalent boundary vertex found")


def _cell_edge_at_boundary_vertex(cx):
    """A cell-cell edge with one endpoint whose wedges include the outer face."""
    outer = cx.outer_face
    for e in cx.cell_cell_edges():
        e = int(e)
        if outer in cx.vertex_faces(int(cx.edge_tail[e])) or outer in cx.vertex_faces(int(cx.edge_head[e])):
            return e
    raise AssertionError("no cell-cell edge ending on the outer face")


def _cell_sides_by_id(cx, faces):
    return {int(f): cx.face_sides(int(f)) for f in faces}


@pytest.fixture
def honey(honeycomb):
    return extract_complex(honeycomb)


@pytest.fixture(scope="module")
def voronoi():
    return extract_complex(voronoi_labels(30, shape=(96, 96), seed=3))


# ---------------------------------------------------------------------- T1
def test_t1_exchange_honeycomb(honey):
    cx = honey
    e = _generic_edge(cx)
    A, B = (int(f) for f in cx.edge_faces[e])
    u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
    C, D = (set(cx.vertex_faces(u)) - {A, B}).pop(), (set(cx.vertex_faces(w)) - {A, B}).pop()
    assert cx.edges_between(A, B).size == 1 and cx.edges_between(C, D).size == 0
    sides0 = _sides(cx)
    charge0 = sum(topological_charge(cx).values())
    old_dir = cx.vertex_xy[w] - cx.vertex_xy[u]

    res = t1_exchange(cx, e, new_length=2.0)
    _assert_event(res, (0, 0, 0))
    new = res.cx
    assert new.edges_between(A, B).size == 0
    assert new.edges_between(C, D).size == 1
    assert int(new.edges_between(C, D)[0]) == e
    sides1 = _sides(new)
    assert sides1[A] == sides0[A] - 1 and sides1[B] == sides0[B] - 1
    assert sides1[C] == sides0[C] + 1 and sides1[D] == sides0[D] + 1
    assert sum(topological_charge(new).values()) == charge0
    # new edge: perpendicular, length new_length, centred on the old midpoint
    new_dir = new.vertex_xy[w] - new.vertex_xy[u]
    assert np.linalg.norm(new_dir) == pytest.approx(2.0)
    assert abs(np.dot(new_dir, old_dir)) < 1e-9
    assert np.allclose(0.5 * (new.vertex_xy[u] + new.vertex_xy[w]), 0.5 * (cx.vertex_xy[u] + cx.vertex_xy[w]))
    assert res.participants["before"]["cells_in_contact"] == [A, B]
    assert res.participants["after"]["cells_in_contact"] == [C, D]
    # polylines still start and end on their vertices
    for k in range(new.n_edges):
        assert np.allclose(new.edge_polyline[k][0], new.vertex_xy[new.edge_tail[k]])
        assert np.allclose(new.edge_polyline[k][-1], new.vertex_xy[new.edge_head[k]])


def test_t1_exchange_all_interior_edges_voronoi(voronoi):
    cx = voronoi
    edges = _generic_edges(cx)
    assert len(edges) >= 5
    for e in edges:
        res = t1_exchange(cx, e)
        _assert_event(res, (0, 0, 0))
    # input untouched
    assert validate(cx).ok


def test_t1_precondition_boundary_edge_raises(honey):
    cx = honey
    e = int(cx.boundary_edges()[0])
    with pytest.raises(EventPreconditionError):
        t1_exchange(cx, e)
    # a cell-cell edge whose endpoint touches the outer face is rejected as well
    e = _cell_edge_at_boundary_vertex(cx)
    with pytest.raises(EventPreconditionError):
        t1_exchange(cx, e)
    with pytest.raises(EventPreconditionError):
        t1_exchange(cx, cx.n_edges + 5)


def test_t1_inplace_and_copy_semantics(honey):
    cx = honey
    e = _generic_edge(cx)
    xy0 = cx.vertex_xy.copy()
    faces0 = cx.edge_faces.copy()
    res = t1_exchange(cx, e)
    assert res.cx is not cx
    assert np.array_equal(cx.vertex_xy, xy0) and np.array_equal(cx.edge_faces, faces0)
    res2 = t1_exchange(cx, e, inplace=True)
    assert res2.cx is cx
    assert np.array_equal(cx.edge_faces, res.cx.edge_faces)


# ------------------------------------------------------------------- divide
def test_divide_honeycomb(honey):
    cx = honey
    f = _interior_face(cx)
    hs = cx.face_loops[f][0]
    e1, e2 = hs[0] >> 1, hs[len(hs) // 2] >> 1
    area0 = face_area(cx, f, smoothed=False)
    label = int(cx.face_label[f])

    res = divide(cx, f, (e1, 0.5), (e2, 0.5))
    _assert_event(res, (2, 3, 1))
    new = res.cx
    f_new = res.participants["after"]["faces"][1]
    assert new.face_kind[f] == FaceKind.CELL and new.face_kind[f_new] == FaceKind.CELL
    assert int(new.face_label[f]) == label and int(new.face_label[f_new]) == label
    a1, a2 = face_area(new, f, smoothed=False), face_area(new, f_new, smoothed=False)
    assert a1 > 0 and a2 > 0
    assert a1 + a2 == pytest.approx(area0, rel=1e-2)
    chord = res.participants["after"]["chord"]
    assert set(int(x) for x in new.edge_faces[chord]) == {f, f_new}
    assert new.edges_between(f, f_new).size == 1
    assert new.face_sides(f) + new.face_sides(f_new) == cx.face_sides(f) + 4
    for v in res.participants["after"]["new_vertices"]:
        assert new.vertex_degree(v) == 3
    assert new.provenance["lineage"][-1] == {"event": "divide", "parent": f, "child": f_new}


def test_divide_voronoi_all_interior_faces(voronoi):
    cx = voronoi
    faces = _interior_faces(cx)
    assert faces
    for f in faces:
        hs = cx.face_loops[f][0]
        e1, e2 = hs[0] >> 1, hs[len(hs) // 2] >> 1
        res = divide(cx, f, (e1, 0.3), (e2, 0.7))
        _assert_event(res, (2, 3, 1))
        f_new = res.participants["after"]["faces"][1]
        total = face_area(res.cx, f, smoothed=False) + face_area(res.cx, f_new, smoothed=False)
        assert total == pytest.approx(face_area(cx, f, smoothed=False), rel=1e-2)


def test_divide_preconditions(honey):
    cx = honey
    f = _interior_face(cx)
    hs = cx.face_loops[f][0]
    e1, e2 = hs[0] >> 1, hs[1] >> 1
    with pytest.raises(EventPreconditionError):
        divide(cx, f, (e1, 0.5), (e1, 0.7))  # same edge twice
    with pytest.raises(EventPreconditionError):
        divide(cx, f, (e1, 0.0), (e2, 0.5))  # t not inside (0, 1)
    with pytest.raises(EventPreconditionError):
        divide(cx, cx.outer_face, (e1, 0.5), (e2, 0.5))  # not a cell
    other = [int(e) for e in cx.cell_cell_edges() if f not in cx.edge_faces[e]][0]
    with pytest.raises(EventPreconditionError):
        divide(cx, f, (e1, 0.5), (other, 0.5))  # edge not on the face


# ------------------------------------------------------------------ extrude
def test_extrude_honeycomb(honey):
    cx = honey
    f = _interior_face(cx)
    n = cx.face_sides(f)
    assert n == 6
    label = int(cx.face_label[f])
    neighbors = cx.face_neighbors(f, kinds=(FaceKind.CELL,))
    sides0 = _cell_sides_by_id(cx, neighbors)

    res = extrude(cx, f)
    _assert_event(res, (1 - n, -n, -1))
    new = res.cx
    assert label not in set(new.face_label.tolist())
    vc = res.participants["after"]["vertex"]
    assert new.vertex_degree(vc) == n
    assert set(new.vertex_faces(vc)) == set(res.participants["after"]["neighbors"])
    for g_old, g_new in zip(neighbors, res.participants["after"]["neighbors"]):
        assert new.face_sides(g_new) == sides0[g_old] - 1
        assert int(new.face_label[g_new]) == int(cx.face_label[g_old])
    assert np.allclose(new.vertex_xy[vc], res.cx.vertex_xy[vc])
    for k in range(new.n_edges):
        assert np.allclose(new.edge_polyline[k][0], new.vertex_xy[new.edge_tail[k]])
        assert np.allclose(new.edge_polyline[k][-1], new.vertex_xy[new.edge_head[k]])


def test_extrude_voronoi_all_interior_faces(voronoi):
    cx = voronoi
    faces = _interior_faces(cx)
    assert faces
    for f in faces:
        n = cx.face_sides(f)
        res = extrude(cx, f)
        _assert_event(res, (1 - n, -n, -1))


def test_extrude_precondition_outer_contact_raises(honey):
    cx = honey
    f = [int(f) for f in cx.cell_faces if cx.face_touches_outer(int(f))][0]
    with pytest.raises(EventPreconditionError):
        extrude(cx, f)
    with pytest.raises(EventPreconditionError):
        extrude(cx, cx.outer_face)


# ------------------------------------------------------------- nucleate_gap
def test_nucleate_gap_honeycomb(honey):
    cx = honey
    v = _tricellular_vertex(cx)
    cells = sorted(set(cx.vertex_faces(v)))
    sides0 = _cell_sides_by_id(cx, cells)
    n_gaps0 = cx.gap_faces.size

    res = nucleate_gap(cx, v, radius=1.5)
    _assert_event(res, (2, 3, 1))
    new = res.cx
    g = res.participants["after"]["gap"]
    assert new.face_kind[g] == FaceKind.GAP and int(new.face_label[g]) == -1
    assert new.gap_faces.size == n_gaps0 + 1
    assert new.face_sides(g) == 3
    assert sorted(new.face_neighbors(g, kinds=(FaceKind.CELL,))) == cells
    assert face_area(new, g, smoothed=False) > 0  # gap loop is CCW like every interior face
    for c in cells:
        assert new.face_sides(c) == sides0[c] + 1
    for pv in res.participants["after"]["vertices"]:
        assert new.vertex_degree(pv) == 3
        assert np.linalg.norm(new.vertex_xy[pv] - cx.vertex_xy[v]) == pytest.approx(1.5)


def test_nucleate_gap_precondition_boundary_vertex_raises(honey):
    cx = honey
    with pytest.raises(EventPreconditionError):
        nucleate_gap(cx, _boundary_vertex(cx), radius=1.0)
    v = _tricellular_vertex(cx)
    with pytest.raises(EventPreconditionError):
        nucleate_gap(cx, v, radius=1e6)  # longer than the incident edges


# ----------------------------------------------------------- rupture/reseal
def test_rupture_then_reseal_roundtrip(honey):
    cx = honey
    e = _generic_edge(cx)
    A, B = (int(f) for f in cx.edge_faces[e])
    u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
    C, D = (set(cx.vertex_faces(u)) - {A, B}).pop(), (set(cx.vertex_faces(w)) - {A, B}).pop()
    sides0 = _sides(cx)
    counts0 = _counts(cx)

    r1 = rupture(cx, e, width=1.0)
    _assert_event(r1, (2, 3, 1))
    mid = r1.cx
    g = r1.participants["after"]["gap"]
    assert mid.face_kind[g] == FaceKind.GAP
    assert mid.face_sides(g) == 4
    assert sorted(mid.face_neighbors(g, kinds=(FaceKind.CELL,))) == sorted([A, B, C, D])
    assert mid.edges_between(A, B).size == 0
    assert mid.edges_between(A, g).size == 1 and mid.edges_between(g, B).size == 1
    sides_mid = _sides(mid)
    assert sides_mid[A] == sides0[A] and sides_mid[B] == sides0[B]
    assert sides_mid[C] == sides0[C] + 1 and sides_mid[D] == sides0[D] + 1
    # long sides are offset copies of the old trace, width apart
    ea, eb = r1.participants["after"]["long_edges"]
    assert polyline_length(mid.edge_polyline[ea]) == pytest.approx(polyline_length(cx.edge_polyline[e]), rel=0.1)
    assert np.linalg.norm(mid.edge_polyline[ea][0] - mid.edge_polyline[eb][0]) == pytest.approx(1.0)
    for cap in r1.participants["after"]["cap_edges"]:
        assert polyline_length(mid.edge_polyline[cap]) == pytest.approx(1.0)

    r2 = reseal(mid, g)
    _assert_event(r2, (-2, -3, -1))
    back = r2.cx
    assert _counts(back) == counts0
    assert back.gap_faces.size == cx.gap_faces.size
    # face ids survive (the gap was appended last), so sides and adjacency compare directly
    assert _sides(back) == sides0
    assert back.edges_between(A, B).size == 1
    e_back = r2.participants["after"]["edge"]
    assert set(int(f) for f in back.edge_faces[e_back]) == {A, B}
    assert np.allclose(back.edge_polyline[e_back][0], cx.vertex_xy[u]) or np.allclose(
        back.edge_polyline[e_back][0], cx.vertex_xy[w]
    )
    assert sorted(back.vertex_degrees().tolist()) == sorted(cx.vertex_degrees().tolist())


def test_rupture_reseal_voronoi_edges(voronoi):
    cx = voronoi
    edges = _generic_edges(cx)
    assert edges
    for e in edges:
        r1 = rupture(cx, e, width=0.5)
        _assert_event(r1, (2, 3, 1))
        r2 = reseal(r1.cx, r1.participants["after"]["gap"])
        _assert_event(r2, (-2, -3, -1))
        assert _counts(r2.cx) == _counts(cx)
        assert _sides(r2.cx) == _sides(cx)


def test_rupture_and_reseal_preconditions(honey, voronoi_small):
    cx = honey
    with pytest.raises(EventPreconditionError):
        rupture(cx, int(cx.boundary_edges()[0]), width=1.0)
    with pytest.raises(EventPreconditionError):
        rupture(cx, _generic_edge(cx), width=0.0)
    with pytest.raises(EventPreconditionError):
        reseal(cx, _interior_face(cx))  # not a gap
    # extracted square gaps have four sides but their corners are the wrong shape for reseal
    cg = extract_complex(voronoi_small)
    for g in cg.gap_faces:
        g = int(g)
        if cg.face_sides(g) != 4:
            with pytest.raises(EventPreconditionError):
                reseal(cg, g)
            break


# ------------------------------------------------------------- birth/death
def test_contact_birth_then_death_roundtrip(quadrants):
    cx = extract_complex(quadrants)
    v = [v for v in range(cx.n_vertices) if cx.vertex_degree(v) == 4][0]
    cells = cx.vertex_cell_set(v)
    assert len(cells) == 4
    counts0 = _counts(cx)

    r1 = contact_birth(cx, v, new_length=2.0)
    _assert_event(r1, (1, 1, 0))
    mid = r1.cx
    e_new = r1.participants["after"]["edge"]
    v1, v2 = r1.participants["after"]["vertices"]
    assert mid.vertex_degree(v1) == 3 and mid.vertex_degree(v2) == 3
    assert {int(mid.edge_tail[e_new]), int(mid.edge_head[e_new])} == {v1, v2}
    assert np.linalg.norm(mid.vertex_xy[v1] - mid.vertex_xy[v2]) == pytest.approx(2.0)
    assert np.allclose(0.5 * (mid.vertex_xy[v1] + mid.vertex_xy[v2]), cx.vertex_xy[v])
    F3, F1 = r1.participants["after"]["cells_in_contact"]
    assert set(int(f) for f in mid.edge_faces[e_new]) == {F1, F3}
    assert mid.edges_between(F1, F3).size == 1
    assert mid.face_sides(F1) == cx.face_sides(F1) + 1 and mid.face_sides(F3) == cx.face_sides(F3) + 1
    assert 4 not in mid.vertex_degrees()

    r2 = contact_death(mid, e_new)
    _assert_event(r2, (-1, -1, 0))
    back = r2.cx
    assert _counts(back) == counts0
    vb = r2.participants["after"]["vertex"]
    assert back.vertex_degree(vb) == 4
    assert back.vertex_cell_set(vb) == cells
    assert np.allclose(back.vertex_xy[vb], cx.vertex_xy[v])
    assert back.edges_between(F1, F3).size == 0
    assert _sides(back) == _sides(cx)


def test_contact_death_then_birth_honeycomb(honey):
    cx = honey
    e = _generic_edge(cx)
    A, B = (int(f) for f in cx.edge_faces[e])
    sides0 = _sides(cx)

    r1 = contact_death(cx, e)
    _assert_event(r1, (-1, -1, 0))
    mid = r1.cx
    v = r1.participants["after"]["vertex"]
    assert mid.vertex_degree(v) == 4
    assert len(mid.vertex_cell_set(v)) == 4
    assert mid.vertex_degrees().tolist().count(4) == 1
    assert 4 in mid.vertex_degrees()
    # face ids are unchanged, A and B each lost a side
    assert mid.face_sides(A) == sides0[A] - 1 and mid.face_sides(B) == sides0[B] - 1

    # resolving the vertex again in either orientation gives a valid trivalent complex
    for start in range(4):
        r2 = contact_birth(mid, v, start=start)
        _assert_event(r2, (1, 1, 0))
        assert 4 not in r2.cx.vertex_degrees()
        assert _counts(r2.cx) == _counts(cx)


def test_contact_preconditions(honey):
    cx = honey
    outer = cx.outer_face
    v3 = _tricellular_vertex(cx)
    with pytest.raises(EventPreconditionError):
        contact_birth(cx, v3)  # degree 3, not 4
    with pytest.raises(EventPreconditionError):
        contact_death(cx, int(cx.boundary_edges()[0]))
    with pytest.raises(EventPreconditionError):
        contact_death(cx, _cell_edge_at_boundary_vertex(cx))
    assert outer not in cx.vertex_faces(v3)


# ----------------------------------------------------------- composability
def test_event_chain_keeps_validity_and_provenance(honey):
    cx = honey
    res = t1_exchange(cx, _generic_edge(cx))
    res = nucleate_gap(res.cx, _tricellular_vertex(res.cx), radius=1.0)
    f = _interior_face(res.cx)
    hs = res.cx.face_loops[f][0]
    res = divide(res.cx, f, (hs[0] >> 1, 0.4), (hs[len(hs) // 2] >> 1, 0.6))
    res = extrude(res.cx, _interior_face(res.cx))
    events = [ev["event"] for ev in res.cx.provenance["events"]]
    assert events == ["t1_exchange", "nucleate_gap", "divide", "extrude"]
    assert validate(res.cx).ok and defect_law_residual(res.cx) == 0
    # provenance stays serialisable
    d = res.cx.to_dict()
    import json

    json.dumps(d["provenance"])
