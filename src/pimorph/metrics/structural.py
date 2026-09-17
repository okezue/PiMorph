"""Structural segmentation metrics on the exact cell complex.

Pixel overlap scores barely move when a one-pixel border is lost and two cells
merge, yet the merge removes a face, several edges and changes the incident cell
sets of the surrounding vertices. The metrics here score those structural changes
directly, next to the usual instance and boundary scores for comparison.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from pimorph.complex.extract import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.complex.incidence import euler_residual, validate
from pimorph.complex.matching import FaceMatch, match_edges, match_faces, match_vertices


def _py(v: Any) -> Any:
    """Convert numpy scalars and bools to JSON-friendly Python types."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


def panoptic_quality(face_match: FaceMatch, thresh: float = 0.5) -> Dict[str, float]:
    """PQ = SQ * RQ with TP at IoU >= thresh, FP unmatched predicted, FN unmatched reference."""
    iou = face_match.iou_matrix()
    tp_iou = iou.data[iou.data >= thresh]
    tp = int(tp_iou.size)
    fp = face_match.n_pred_cells - tp
    fn = face_match.n_ref_cells - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    # two empty segmentations agree perfectly
    sq = float(tp_iou.mean()) if tp else (1.0 if denom == 0 else 0.0)
    rq = float(tp / denom) if denom > 0 else 1.0
    return {"pq": sq * rq, "sq": sq, "rq": rq, "n_tp": tp, "n_fp": int(fp), "n_fn": int(fn)}


def instance_ap(face_match: FaceMatch, thresh: float) -> float:
    """TP / (TP + FP + FN) at IoU >= thresh (DSB-2018 style instance average precision)."""
    tp = face_match.n_true_positives(thresh)
    denom = face_match.n_pred_cells + face_match.n_ref_cells - tp
    return float(tp / denom) if denom > 0 else 1.0


def variation_of_information(fid_pred: np.ndarray, fid_ref: np.ndarray) -> Tuple[float, float, float]:
    """(VI, H(ref | pred), H(pred | ref)) in bits over all pixels; background is one class.

    H(ref | pred) grows with false merges (under-segmentation), H(pred | ref) with
    false splits (over-segmentation).
    """
    p = fid_pred.ravel().astype(np.int64) + 1
    r = fid_ref.ravel().astype(np.int64) + 1
    n_r = int(r.max()) + 1
    code = p * n_r + r
    uniq, counts = np.unique(code, return_counts=True)
    pi = uniq // n_r
    ri = uniq % n_r
    pxy = counts / counts.sum()
    px = np.bincount(pi, weights=pxy)
    py = np.bincount(ri, weights=pxy)
    h_ref_given_pred = float(-np.sum(pxy * np.log2(pxy / px[pi])))
    h_pred_given_ref = float(-np.sum(pxy * np.log2(pxy / py[ri])))
    return h_ref_given_pred + h_pred_given_ref, h_ref_given_pred, h_pred_given_ref


def boundary_scores(labels_pred: np.ndarray, labels_ref: np.ndarray, tol_px: float = 3.0) -> Dict[str, float]:
    """Boundary precision/recall/F1 at ``tol_px`` and symmetric Hausdorff distances.

    Boundary pixels come from ``find_boundaries(mode="inner")``; a pixel counts as
    matched when its Euclidean distance to the other boundary set is <= tol_px.
    """
    bp = find_boundaries(labels_pred, mode="inner")
    br = find_boundaries(labels_ref, mode="inner")
    if not bp.any() and not br.any():
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "hausdorff_px": 0.0,
            "hausdorff95_px": 0.0,
        }
    if not bp.any() or not br.any():
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "hausdorff_px": float("inf"),
            "hausdorff95_px": float("inf"),
        }
    d_to_ref = ndi.distance_transform_edt(~br)
    d_to_pred = ndi.distance_transform_edt(~bp)
    dp = d_to_ref[bp]  # distance of each predicted boundary pixel to the reference boundary
    dr = d_to_pred[br]
    prec = float(np.mean(dp <= tol_px))
    rec = float(np.mean(dr <= tol_px))
    return {
        "boundary_precision": prec,
        "boundary_recall": rec,
        "boundary_f1": _f1(prec, rec),
        "hausdorff_px": float(max(dp.max(), dr.max())),
        "hausdorff95_px": float(max(np.percentile(dp, 95), np.percentile(dr, 95))),
    }


def structural_metrics(
    labels_pred: np.ndarray,
    labels_ref: np.ndarray,
    pixel_size_um: Optional[float] = None,
    tol_px: float = 3.0,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Flat, JSON-serializable dictionary of structural and pixel-level metrics.

    Both label images are converted to smoothed half-edge complexes; faces, edges
    and vertices are matched with ``pimorph.complex.matching``. Instance (PQ, AP),
    information (VI) and boundary (F1, Hausdorff) scores are added for comparison.
    """
    t0 = time.perf_counter()
    labels_pred = np.asarray(labels_pred)
    labels_ref = np.asarray(labels_ref)
    if labels_pred.shape != labels_ref.shape:
        raise ValueError("label images must have the same shape")

    cx_p = smooth_complex(extract_complex(labels_pred, pixel_size_um=pixel_size_um))
    cx_r = smooth_complex(extract_complex(labels_ref, pixel_size_um=pixel_size_um))
    fm = match_faces(labels_pred, labels_ref, cx_p, cx_r, iou_thresh=iou_thresh)
    em = match_edges(cx_p, cx_r, fm, tol_px=tol_px)
    vm = match_vertices(cx_p, cx_r, fm, dist_px=tol_px)

    out: Dict[str, Any] = {}
    out.update(fm.summary())
    out.update(em.summary())
    out.update(vm.summary())

    pq = panoptic_quality(fm, thresh=0.5)
    out["pq"] = pq["pq"]
    out["sq"] = pq["sq"]
    out["rq"] = pq["rq"]
    out["ap50"] = instance_ap(fm, 0.5)
    out["ap75"] = instance_ap(fm, 0.75)

    vi, vi_merge, vi_split = variation_of_information(fm.fid_pred, fm.fid_ref)
    out["vi"] = vi
    out["vi_merge"] = vi_merge  # H(ref | pred), under-segmentation
    out["vi_split"] = vi_split  # H(pred | ref), over-segmentation

    out.update(boundary_scores(labels_pred, labels_ref, tol_px=tol_px))

    rep_p = validate(cx_p)
    rep_r = validate(cx_r)
    out["euler_residual_pred"] = euler_residual(cx_p)
    out["euler_residual_ref"] = euler_residual(cx_r)
    out["valid_pred"] = bool(rep_p.ok)
    out["valid_ref"] = bool(rep_r.ok)
    out["validity_fraction"] = 1.0 if rep_p.ok else 0.0

    out["complex_edit_distance_approx"] = (
        len(fm.unmatched_pred)
        + len(fm.unmatched_ref)
        + len(em.unmatched_pred)
        + len(em.unmatched_ref)
        + vm.n_incident_set_mismatches
    )
    out["tol_px"] = float(tol_px)
    out["iou_thresh"] = float(iou_thresh)
    if pixel_size_um is not None:
        out["pixel_size_um"] = float(pixel_size_um)
    out["runtime_s"] = time.perf_counter() - t0
    return {k: _py(v) for k, v in out.items()}


def _label_pairs(labels: np.ndarray, min_contact_px: int) -> Set[Tuple[int, int]]:
    from endopigraph.interfaces import compute_interfaces

    df = compute_interfaces(labels, min_contact_px=min_contact_px).edges
    return {(int(i), int(j)) for i, j in zip(df["cell_i"].values, df["cell_j"].values)}


def _label_mapping(labels_pred: np.ndarray, labels_ref: np.ndarray, iou_thresh: float = 0.5) -> Dict[int, int]:
    """Predicted label -> reference label with pixel IoU >= iou_thresh (unique at that level)."""
    p = labels_pred.ravel().astype(np.int64)
    r = labels_ref.ravel().astype(np.int64)
    both = (p > 0) & (r > 0)
    n_r = int(r.max()) + 1
    code = p[both] * n_r + r[both]
    uniq, counts = np.unique(code, return_counts=True)
    pi, ri = uniq // n_r, uniq % n_r
    area_p = np.bincount(p, minlength=int(p.max()) + 1)
    area_r = np.bincount(r, minlength=n_r)
    iou = counts / (area_p[pi] + area_r[ri] - counts)
    keep = iou >= iou_thresh
    return {int(a): int(b) for a, b in zip(pi[keep], ri[keep])}


def legacy_adjacency_f1(labels_pred: np.ndarray, labels_ref: np.ndarray, min_contact_px: int = 10) -> Dict[str, Any]:
    """Adjacency precision/recall/F1 from 4-neighbour pixel contacts.

    Uses ``endopigraph.interfaces.compute_interfaces`` on both images, so numbers
    are directly comparable with the legacy benchmarks. Predicted labels are
    mapped to reference labels by pixel IoU >= 0.5; pairs whose labels have no
    counterpart count as false positives.
    """
    labels_pred = np.asarray(labels_pred)
    labels_ref = np.asarray(labels_ref)
    pred_pairs = _label_pairs(labels_pred, min_contact_px)
    ref_pairs = _label_pairs(labels_ref, min_contact_px)
    mapping = _label_mapping(labels_pred, labels_ref)
    mapped: Set[Tuple[int, int]] = set()
    for a, b in pred_pairs:
        if a in mapping and b in mapping:
            ra, rb = mapping[a], mapping[b]
            mapped.add((ra, rb) if ra <= rb else (rb, ra))
    tp = len(mapped & ref_pairs)
    fp = len(pred_pairs) - tp
    fn = len(ref_pairs) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(_f1(precision, recall)),
        "n_predicted": len(pred_pairs),
        "n_ground_truth": len(ref_pairs),
        "min_contact_px": int(min_contact_px),
    }


__all__ = [
    "boundary_scores",
    "instance_ap",
    "legacy_adjacency_f1",
    "panoptic_quality",
    "structural_metrics",
    "variation_of_information",
]
