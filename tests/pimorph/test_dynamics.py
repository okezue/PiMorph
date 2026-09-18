"""Synthetic tests of the temporal layer: linking, tracking, event detection and
the exact (dV, dE, dF) bookkeeping. Ground truth is created with the admissible
rewrites of ``pimorph.complex.events`` and rasterized back to label images."""

import numpy as np
import pandas as pd
import pytest
from skimage.segmentation import expand_labels

from pimorph.complex import FaceKind, extract_complex, topological_charge
from pimorph.complex.events import contact_birth, divide, extrude, nucleate_gap, rupture, t1_exchange
from pimorph.complex.geometry import polyline_length
from pimorph.dynamics import (
    TopologySnapshot,
    admissibility_check,
    complex_to_labels,
    complexes_over_time,
    detect_events,
    detect_events_from_snapshots,
    division_detection_metrics,
    event_summary,
    link_frames,
    raster_face_labels,
    t1_metrics,
    track,
    tracking_metrics,
)

from .conftest import voronoi_labels

SHAPE = (96, 96)


# ------------------------------------------------------------------ helpers
def _all_cells(cx, faces):
    return all(cx.face_kind[f] == FaceKind.CELL for f in faces)


def _generic_edges(cx):
    out = []
    for e in cx.cell_cell_edges():
        e = int(e)
        u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
        if u == w or cx.vertex_degree(u) != 3 or cx.vertex_degree(w) != 3:
            continue
        faces = set(cx.vertex_faces(u)) | set(cx.vertex_faces(w))
        if len(faces) != 4 or not _all_cells(cx, faces):
            continue
        if any(cx.face_touches_outer(f) for f in faces):
            continue
        out.append(e)
    return out


def _interior_faces(cx):
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


def _tricellular_vertices(cx):
    out = []
    for v in range(cx.n_vertices):
        if cx.vertex_degree(v) != 3:
            continue
        faces = cx.vertex_faces(v)
        if len(set(faces)) == 3 and _all_cells(cx, faces) and not any(cx.face_touches_outer(f) for f in faces):
            out.append(v)
    return out


def _roundtrip(cx_a, res):
    labels_b = complex_to_labels(res.cx, SHAPE)
    cx_b = extract_complex(labels_b)
    events = detect_events(cx_a, cx_b)
    return cx_b, events, admissibility_check(cx_a, cx_b, events)


def _margin_tissue(shape=(120, 120), margin=12, seed=5, min_area=120):
    """Tessellation surrounded by background; cells below ``min_area`` px are merged away."""
    inner = voronoi_labels(20, (shape[0] - 2 * margin, shape[1] - 2 * margin), seed=seed)
    counts = np.bincount(inner.ravel())
    small = np.isin(inner, np.flatnonzero(counts < min_area))
    inner = np.where(small, 0, inner)
    inner = expand_labels(inner, distance=float(max(shape)))
    lab = np.zeros(shape, dtype=np.int64)
    lab[margin:-margin, margin:-margin] = inner
    return lab


@pytest.fixture(scope="module")
def labels_a():
    return voronoi_labels(30, SHAPE, seed=3)


@pytest.fixture(scope="module")
def cx_a(labels_a):
    return extract_complex(labels_a)


@pytest.fixture
def quadrants_cx(quadrants):
    return extract_complex(quadrants)


# ------------------------------------------------------------------ linking
def test_link_frames_identity(labels_a):
    link = link_frames(labels_a, labels_a)
    n = np.unique(labels_a[labels_a > 0]).size
    assert len(link.pairs) == n
    assert all(iou == pytest.approx(1.0) for _, _, iou in link.pairs)
    assert all(a == b for a, b, _ in link.pairs)
    assert link.unmatched_a == [] and link.unmatched_b == []
    assert link.candidates_split == {} and link.candidates_merge == {} and link.divisions == []


def test_link_frames_translation_keeps_matches():
    lab = _margin_tissue()
    shifted = np.zeros_like(lab)
    shifted[2:, 2:] = lab[:-2, :-2]
    link = link_frames(lab, shifted)
    ids = np.unique(lab[lab > 0])
    assert len(link.pairs) == ids.size
    assert all(a == b for a, b, _ in link.pairs)
    assert min(iou for _, _, iou in link.pairs) >= 0.5
    assert link.divisions == []


def test_link_frames_split_is_a_division(labels_a):
    lab_b = labels_a.copy()
    parent = 5
    rows = np.where(labels_a == parent)[0]
    cut = int(np.median(rows))
    lab_b[(labels_a == parent) & (np.arange(SHAPE[0])[:, None] >= cut)] = 99
    link = link_frames(labels_a, lab_b)
    assert link.candidates_split == {parent: [parent, 99]}
    assert link.divisions == [(parent, (parent, 99))]
    assert parent in link.unmatched_a and 99 in link.unmatched_b and parent in link.unmatched_b
    # the merge direction is symmetric
    back = link_frames(lab_b, labels_a)
    assert back.candidates_merge == {parent: [parent, 99]}
    assert back.divisions == []


# ----------------------------------------------------------------- tracking
def test_track_translating_tissue_keeps_ids_stable():
    lab = _margin_tissue()
    stack = []
    for t in range(5):
        fr = np.zeros_like(lab)
        fr[t:, t:] = lab[: lab.shape[0] - t, : lab.shape[1] - t]
        stack.append(fr)
    tracks = track(stack)
    assert tracks.n_frames == 5
    assert tracks.n_tracks == np.unique(lab[lab > 0]).size
    assert tracks.lineage == {}
    assert all(f == 0 for f in tracks.births.values())
    assert all(f == 4 for f in tracks.deaths.values())
    m = tracking_metrics(tracks, stack)
    assert m["n_id_switches"] == 0
    assert m["n_fragmentations"] == 0
    assert m["cell_matching_accuracy"] == pytest.approx(1.0)
    cxs = complexes_over_time(tracks)
    assert len(cxs) == 5
    for t in range(1, 5):
        events = detect_events(cxs[t - 1], cxs[t], frame=t - 1, lineage=tracks.lineage)
        assert events == []
        assert admissibility_check(cxs[t - 1], cxs[t], events)["fully_explained"]


def test_track_division_lineage_and_detection(labels_a, cx_a):
    lab_b = labels_a.copy()
    parent = 5
    rows = np.where(labels_a == parent)[0]
    cut = int(np.median(rows))
    lab_b[(labels_a == parent) & (np.arange(SHAPE[0])[:, None] >= cut)] = 99
    tracks = track([labels_a, lab_b])
    p = tracks.label_to_track[0][parent]
    kids = sorted(c for c, q in tracks.lineage.items() if q == p)
    assert len(kids) == 2
    assert tracks.deaths[p] == 0 and all(tracks.births[c] == 1 for c in kids)
    assert tracks.divisions() == [{"parent": p, "children": kids, "frame": 1}]
    cxs = complexes_over_time(tracks)
    events = detect_events(cxs[0], cxs[1], frame=0, lineage=tracks.lineage)
    kinds = [e.kind for e in events]
    assert kinds.count("division") == 1
    div = [e for e in events if e.kind == "division"][0]
    assert div.participants == {"parent": p, "children": kids}
    assert div.evidence["source"] == "lineage"
    assert set(kinds) <= {"division", "contact_birth", "contact_death"}
    check = admissibility_check(cxs[0], cxs[1], events)
    assert check["residual_dVEF"] == (0, 0, 0) and check["n_unexplained"] == 0
    metrics = division_detection_metrics(events, [{"parent": p, "children": kids, "frame": 1}])
    assert metrics["n_tp"] == 1 and metrics["f1"] == pytest.approx(1.0)


# --------------------------------------------------------------- rasterize
def test_complex_to_labels_roundtrip_exact(labels_a):
    assert np.array_equal(complex_to_labels(extract_complex(labels_a), SHAPE), labels_a)
    with_gaps = voronoi_labels(40, (128, 128), seed=1, gaps=3)
    assert np.array_equal(complex_to_labels(extract_complex(with_gaps), (128, 128)), with_gaps)


# ------------------------------------------------------------------- events
def test_no_change_gives_no_events(cx_a):
    events = detect_events(cx_a, cx_a)
    assert events == []
    check = admissibility_check(cx_a, cx_a, events)
    assert check["fully_explained"] and check["observed_dVEF"] == (0, 0, 0)
    assert check["charge_a"] == sum(topological_charge(cx_a).values())


def test_detect_t1_from_rewrite(cx_a):
    edges = _generic_edges(cx_a)
    assert len(edges) >= 10
    n_clean = 0
    for e in edges:
        A, B = (int(cx_a.face_label[f]) for f in cx_a.edge_faces[e])
        u, w = int(cx_a.edge_tail[e]), int(cx_a.edge_head[e])
        C = int(cx_a.face_label[(set(cx_a.vertex_faces(u)) - set(cx_a.edge_faces[e].tolist())).pop()])
        D = int(cx_a.face_label[(set(cx_a.vertex_faces(w)) - set(cx_a.edge_faces[e].tolist())).pop()])
        res = t1_exchange(cx_a, e, new_length=3.0)
        _, events, check = _roundtrip(cx_a, res)
        if not events:
            continue  # the 3 px edge did not survive rasterization; no topology change at all
        assert [ev.kind for ev in events] == ["t1"]
        ev = events[0]
        assert ev.participants["lost"] == sorted([A, B])
        assert ev.participants["gained"] == sorted([C, D])
        assert ev.evidence["charge_delta"] == 0 and ev.evidence["generic"]
        assert check["residual_dVEF"] == (0, 0, 0) and check["fully_explained"]
        assert check["charge_delta"] == 0 and check["n_t1_charge_conserved"] == 1
        n_clean += 1
    assert n_clean >= 0.8 * len(edges)


def test_detect_division_from_rewrite(cx_a):
    f = _interior_faces(cx_a)[0]
    hs = cx_a.face_loops[f][0]
    res = divide(cx_a, f, (hs[0] >> 1, 0.5), (hs[len(hs) // 2] >> 1, 0.5))
    parent = int(cx_a.face_label[f])
    child = int(raster_face_labels(res.cx)[res.participants["after"]["faces"][1]])
    assert child != parent
    _, events, check = _roundtrip(cx_a, res)
    assert [ev.kind for ev in events] == ["division"]
    assert events[0].participants == {"parent": parent, "children": sorted([parent, child])}
    assert events[0].expected_delta == (2, 3, 1)
    assert check["residual_dVEF"] == (0, 0, 0) and check["fully_explained"]


def test_detect_extrusion_from_rewrite(cx_a):
    faces = _interior_faces(cx_a)
    assert faces
    for f in faces:
        n = cx_a.face_sides(f)
        label = int(cx_a.face_label[f])
        neighbours = sorted(int(cx_a.face_label[g]) for g in cx_a.face_neighbors(f))
        res = extrude(cx_a, f)
        _, events, check = _roundtrip(cx_a, res)
        kinds = [ev.kind for ev in events]
        assert kinds.count("extrusion") == 1
        ev = events[kinds.index("extrusion")]
        assert ev.participants["cell"] == label and ev.participants["sides"] == n
        assert ev.participants["neighbours"] == neighbours
        assert ev.expected_delta == (1 - n, -n, -1)
        # an n-fold junction rasterizes into trivalent vertices joined by short edges
        assert set(kinds) <= {"extrusion", "contact_birth"}
        assert check["residual_dVEF"] == (0, 0, 0) and check["fully_explained"]


def test_detect_gap_nucleation_from_rewrite(cx_a):
    n_ok = 0
    for v in _tricellular_vertices(cx_a):
        if min(polyline_length(cx_a.edge_polyline[h >> 1]) for h in cx_a.vertex_out_half_edges(v)) <= 2.0:
            continue
        cells = sorted(int(cx_a.face_label[f]) for f in cx_a.vertex_cell_set(v))
        res = nucleate_gap(cx_a, v, radius=2.0)
        cx_b, events, check = _roundtrip(cx_a, res)
        assert cx_b.gap_faces.size == cx_a.gap_faces.size + 1
        assert [ev.kind for ev in events] == ["gap_nucleation"]
        assert events[0].participants["cells"] == cells and events[0].participants["sides"] == 3
        assert events[0].expected_delta == (2, 3, 1)
        assert check["residual_dVEF"] == (0, 0, 0) and check["fully_explained"]
        n_ok += 1
    assert n_ok >= 3


def test_detect_contact_birth_and_death(quadrants_cx):
    cx = quadrants_cx
    v = [v for v in range(cx.n_vertices) if cx.vertex_degree(v) == 4][0]
    for start in range(4):
        res = contact_birth(cx, v, new_length=3.0, start=start)
        F3, F1 = res.participants["after"]["cells_in_contact"]
        pair = sorted([int(cx.face_label[F1]), int(cx.face_label[F3])])
        labels_b = complex_to_labels(res.cx, (20, 20))
        cx_b = extract_complex(labels_b)
        forward = detect_events(cx, cx_b)
        assert [ev.kind for ev in forward] == ["contact_birth"]
        assert forward[0].participants["cells"] == pair
        assert admissibility_check(cx, cx_b, forward)["fully_explained"]
        backward = detect_events(cx_b, cx)
        assert [ev.kind for ev in backward] == ["contact_death"]
        assert backward[0].participants["cells"] == pair
        check = admissibility_check(cx_b, cx, backward)
        assert check["observed_dVEF"] == (-1, -1, 0) and check["fully_explained"]


def test_detect_rupture_and_reseal(cx_a):
    n_clean = 0
    n_total = 0
    for e in _generic_edges(cx_a):
        if polyline_length(cx_a.edge_polyline[e]) < 6:
            continue
        n_total += 1
        A, B = sorted(int(cx_a.face_label[f]) for f in cx_a.edge_faces[e])
        res = rupture(cx_a, e, width=1.5)
        cx_b, forward, check = _roundtrip(cx_a, res)
        backward = detect_events(cx_b, cx_a)
        check_back = admissibility_check(cx_b, cx_a, backward)
        if [ev.kind for ev in forward] != ["rupture"]:
            continue  # the offset traces swallowed a neighbouring junction when rasterized
        assert forward[0].participants["cells"] == [A, B]
        assert len(forward[0].participants["caps"]) == 2
        assert forward[0].expected_delta == (2, 3, 1)
        assert check["observed_dVEF"] == (2, 3, 1) and check["fully_explained"]
        assert [ev.kind for ev in backward] == ["reseal"]
        assert backward[0].participants["cells"] == [A, B]
        assert check_back["observed_dVEF"] == (-2, -3, -1) and check_back["fully_explained"]
        n_clean += 1
    assert n_total >= 8 and n_clean >= 0.7 * n_total


def test_death_to_gap_and_exit_bookkeeping(labels_a, cx_a):
    interior = int(cx_a.face_label[_interior_faces(cx_a)[0]])
    lab_b = labels_a.copy()
    lab_b[labels_a == interior] = 0
    cx_b = extract_complex(lab_b)
    events = detect_events(cx_a, cx_b)
    assert [ev.kind for ev in events] == ["death_to_gap"]
    assert events[0].participants["cell"] == interior and events[0].evidence["bounding_matches"]
    check = admissibility_check(cx_a, cx_b, events)
    assert check["observed_dVEF"] == (0, 0, 0) and check["fully_explained"]
    # a boundary cell leaving the field is not an admissible rewrite
    boundary = int(cx_a.face_label[[f for f in cx_a.cell_faces if cx_a.face_touches_outer(int(f))][0]])
    lab_c = labels_a.copy()
    lab_c[labels_a == boundary] = 0
    cx_c = extract_complex(lab_c)
    events = detect_events(cx_a, cx_c)
    assert "exit" in [ev.kind for ev in events]
    check = admissibility_check(cx_a, cx_c, events)
    assert check["n_unexplained"] >= 1 and not check["fully_explained"]


def test_event_summary_shapes(cx_a):
    assert event_summary([]).shape == (0, 1)
    edges = _generic_edges(cx_a)
    events = []
    for k, e in enumerate(edges[:3]):
        res = t1_exchange(cx_a, e, new_length=3.0)
        cx_b = extract_complex(complex_to_labels(res.cx, SHAPE))
        events += detect_events(cx_a, cx_b, frame=k)
    frames = sorted({ev.frame for ev in events})
    tab = event_summary(events)
    assert isinstance(tab, pd.DataFrame)
    assert list(tab["frame"]) == frames
    assert "t1" in tab.columns and tab.shape[0] == len(frames)
    assert int(tab.drop(columns="frame").to_numpy().sum()) == len(events)


def test_snapshot_t1_and_metrics():
    # quadrilateral p=1, q=2 in contact with r=3, s=4 both; the T1 swaps the diagonal
    contacts_a = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)}
    contacts_b = {(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)}
    sides_a = {1: 5, 2: 5, 3: 4, 4: 4}
    sides_b = {1: 4, 2: 4, 3: 5, 4: 5}
    sa = TopologySnapshot({1, 2, 3, 4}, contacts_a, set(), {}, sides_a, (10, 15, 6))
    sb = TopologySnapshot({1, 2, 3, 4}, contacts_b, set(), {}, sides_b, (10, 15, 6))
    events = detect_events_from_snapshots(sa, sb, frame=7)
    assert [ev.kind for ev in events] == ["t1"]
    assert events[0].participants["lost"] == [1, 2] and events[0].participants["gained"] == [3, 4]
    assert events[0].evidence["charge_delta"] == 0
    assert admissibility_check(sa, sb, events)["fully_explained"]
    gt = [{"lost": [1, 2], "gained": [3, 4], "frame": 7}]
    m = t1_metrics(events, gt)
    assert m["n_tp"] == 1 and m["f1"] == pytest.approx(1.0)
    assert t1_metrics(events, None)["n_pred"] == 1
    # divisions: CTC lineage rows {id: (start, end, parent)}
    lineage = {1: (0, 4, 0), 2: (5, 9, 1), 3: (5, 9, 1), 4: (0, 9, 0)}
    pred = [{"parent": 1, "children": [2, 3], "frame": 5}, {"parent": 4, "children": [7, 8], "frame": 3}]
    d = division_detection_metrics(pred, lineage)
    assert d["n_gt"] == 1 and d["n_pred"] == 2 and d["n_tp"] == 1
    assert d["precision"] == pytest.approx(0.5) and d["recall"] == pytest.approx(1.0)
