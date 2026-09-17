import numpy as np
import pytest

from pimorph.complex import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.complex.matching import (
    _cyclic_equal,
    face_id_image,
    match_edges,
    match_faces,
    match_vertices,
)

from .conftest import voronoi_labels


def _complex(lab):
    return smooth_complex(extract_complex(lab))


def _match_all(pred, ref, tol_px=3.0):
    cx_p, cx_r = _complex(pred), _complex(ref)
    fm = match_faces(pred, ref, cx_p, cx_r)
    em = match_edges(cx_p, cx_r, fm, tol_px=tol_px)
    vm = match_vertices(cx_p, cx_r, fm, dist_px=tol_px)
    return cx_p, cx_r, fm, em, vm


def _adjacent_label_pairs(lab):
    """(label_a, label_b, area_a, area_b) for every cell-cell contact."""
    cx = extract_complex(lab)
    areas = np.bincount(lab.ravel())
    out = []
    for e in cx.cell_cell_edges():
        a, b = (int(x) for x in cx.edge_faces[e])
        la, lb = int(cx.face_label[a]), int(cx.face_label[b])
        out.append((la, lb, int(areas[la]), int(areas[lb])))
    return out


def merge_balanced_pair(lab):
    """Relabel one cell into its neighbour, choosing the most equal-sized adjacent pair."""
    best = max(_adjacent_label_pairs(lab), key=lambda t: min(t[2], t[3]) / (t[2] + t[3]))
    la, lb = best[0], best[1]
    out = lab.copy()
    out[out == lb] = la
    return out, la, lb


def split_largest_cell(lab):
    """Give the right half (by median column) of the largest cell a new label."""
    areas = np.bincount(lab.ravel())
    big = int(np.argmax(areas[1:]) + 1)
    mask = lab == big
    cols = np.nonzero(mask)[1]
    med = int(np.median(cols))
    colgrid = np.arange(lab.shape[1])[None, :]
    out = lab.copy()
    out[mask & (colgrid >= med)] = lab.max() + 1
    return out, big


@pytest.fixture
def voronoi40():
    return voronoi_labels(40, (128, 128), seed=2)


# ---------------------------------------------------------------- identity
def test_identity_matches_everything(voronoi40):
    cx_p, cx_r, fm, em, vm = _match_all(voronoi40, voronoi40)
    assert len(fm.pairs) == cx_r.cell_faces.size
    assert not fm.unmatched_pred and not fm.unmatched_ref
    assert np.all(fm.iou == 1.0)
    assert fm.n_splits == 0 and fm.n_merges == 0
    assert len(em.pairs) == cx_r.cell_cell_edges().size
    assert em.pair_f1 == 1.0 and em.component_f1 == 1.0
    assert np.all(em.arclength_rel_error == 0.0)
    assert vm.precision == 1.0 and vm.recall == 1.0
    assert vm.n_pred_vertices > 20
    assert np.all(vm.distance_px == 0.0)
    assert vm.incident_set_accuracy == 1.0 and vm.cyclic_order_accuracy == 1.0


def test_honeycomb_identity_vertices(honeycomb):
    _, cx_r, fm, em, vm = _match_all(honeycomb, honeycomb)
    assert vm.n_ref_vertices > 20
    assert len(vm.pairs) == vm.n_ref_vertices
    assert vm.cyclic_order_accuracy == 1.0
    assert em.component_recall == 1.0


# ------------------------------------------------------------ perturbations
def test_merge_two_adjacent_cells(voronoi40):
    merged, la, lb = merge_balanced_pair(voronoi40)
    cx_p, cx_r, fm, em, vm = _match_all(merged, voronoi40)
    assert fm.n_merges == 1
    assert fm.n_splits == 0
    assert fm.n_pred_cells == fm.n_ref_cells - 1
    assert len(fm.unmatched_ref) == 1
    assert em.pair_recall < 1.0
    assert em.pair_precision > 0.9
    assert em.component_recall < 1.0
    # the two junctions on the erased contact vanished, remaining ones moved nowhere
    assert vm.n_pred_vertices == vm.n_ref_vertices - 2
    assert np.all(vm.distance_px == 0.0)
    assert vm.n_incident_set_mismatches > 0


def test_split_one_cell(voronoi40):
    split, big = split_largest_cell(voronoi40)
    cx_p, cx_r, fm, em, vm = _match_all(split, voronoi40)
    assert fm.n_splits == 1
    assert fm.n_merges == 0
    assert fm.n_pred_cells == fm.n_ref_cells + 1
    # the new contact between the halves has no reference counterpart
    assert em.pair_precision < 1.0
    assert em.pair_recall == 1.0


def test_shift_one_pixel(voronoi40):
    shifted = np.roll(voronoi40, 1, axis=0)
    shifted[0, :] = 0  # drop the wrapped row so this is a pure translation
    cx_p, cx_r, fm, em, vm = _match_all(shifted, voronoi40)
    assert len(fm.unmatched_ref) == 0
    assert fm.iou.min() > 0.5
    assert em.pair_f1 > 0.9 and em.component_f1 > 0.9
    assert 0.8 <= float(vm.distance_px.mean()) <= 1.5
    assert float(np.median(vm.distance_px)) == pytest.approx(1.0)
    assert vm.incident_set_accuracy > 0.9


# ---------------------------------------------------------- face id images
def test_face_id_image_fallback_for_split_labels(pinch):
    cx = extract_complex(pinch)
    assert cx.provenance["n_split_labels"] == 1
    fid = face_id_image(pinch, cx)
    assert fid.shape == pinch.shape
    assert set(np.unique(fid[fid >= 0]).tolist()) == set(cx.cell_faces.tolist())
    for f in cx.cell_faces:
        vals = np.unique(pinch[fid == f])
        assert vals.size == 1 and vals[0] == cx.face_label[f]
    assert np.all(fid[pinch == 0] == -1)
    fm = match_faces(pinch, pinch, cx, cx)
    assert len(fm.pairs) == 4 and fm.n_splits == 0


def test_face_id_image_lookup_for_unsplit_labels(voronoi40):
    cx = extract_complex(voronoi40)
    fid = face_id_image(voronoi40, cx)
    assert np.array_equal(cx.face_label[fid], voronoi40)


def test_face_id_image_shape_mismatch_raises(voronoi40):
    cx = extract_complex(voronoi40)
    with pytest.raises(ValueError):
        face_id_image(voronoi40[:64], cx)


# --------------------------------------------------- multiplicity and order
def test_component_level_sees_double_contact(double_contact):
    single = double_contact.copy()
    single[8:12, 10:20] = 2  # fill the gap: one contact component instead of two
    cx_p, cx_r, fm, em, vm = _match_all(single, double_contact)
    assert len(fm.pairs) == 2
    assert em.n_ref_edges == 2 and em.n_pred_edges == 1
    assert em.pair_precision == 1.0 and em.pair_recall == 1.0
    assert em.component_recall == 0.5
    assert len(em.unmatched_ref) == 1


def test_cyclic_equal_rotation_only():
    assert _cyclic_equal([1, 2, 3], [2, 3, 1])
    assert _cyclic_equal([1, 2, 3], [1, 2, 3])
    assert not _cyclic_equal([1, 2, 3], [1, 3, 2])  # reversed direction
    assert not _cyclic_equal([1, 2, 3], [1, 2])
    assert _cyclic_equal([], [])
