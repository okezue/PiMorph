"""Structural matching of a predicted complex against a reference complex.

Both complexes must come from label images of the same shape. Faces are matched
by pixel overlap (Hungarian on IoU), cell-cell edges by geometric overlap of their
traces once the incident face pair is mapped through the face matching, and
vertices with at least three incident cells by position, followed by incident
cell set and cyclic order checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label

from .geometry import edge_arclength
from .halfedge import HalfEdgeComplex


# ----------------------------------------------------------------- helpers
def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


def _ratio(num: float, den: float, empty: float = 1.0) -> float:
    return float(empty) if den == 0 else float(num) / float(den)


def face_id_image(labels: np.ndarray, cx: HalfEdgeComplex) -> np.ndarray:
    """(H, W) int64 image holding the cell face id of each pixel, -1 elsewhere.

    With unsplit labels the map is a lookup through ``face_label``. When a label
    was split into several faces the components are re-derived with the same
    4-connected labelling used by ``extract_complex`` and the ordering is checked
    against ``face_label`` at one pixel per face.
    """
    labels = np.asarray(labels)
    if cx.shape is not None and tuple(cx.shape) != labels.shape:
        raise ValueError(f"label shape {labels.shape} differs from complex shape {cx.shape}")
    cells = cx.cell_faces
    unsplit = int(cx.provenance.get("n_split_labels", 0)) == 0 and np.unique(cx.face_label[cells]).size == cells.size
    if unsplit:
        lut = np.full(int(labels.max()) + 1 if labels.size else 1, -1, dtype=np.int64)
        lut[cx.face_label[cells]] = cells
        return lut[labels]
    cc = cc_label(labels, connectivity=1, background=0).astype(np.int64)
    fid = cc - 1
    ids, first = np.unique(fid.ravel(), return_index=True)
    ids, first = ids[ids >= 0], first[ids >= 0]
    if ids.size != cells.size or not np.array_equal(ids, cells):
        raise ValueError("connected-component face ids do not match the complex cell faces")
    if not np.array_equal(cx.face_label[ids], labels.ravel()[first]):
        raise ValueError("face_label disagrees with the re-derived component order")
    return fid


def _overlap_table(fid_pred: np.ndarray, fid_ref: np.ndarray, n_pred: int, n_ref: int) -> sparse.coo_matrix:
    """Sparse (n_pred, n_ref) pixel-intersection counts between cell faces."""
    both = (fid_pred >= 0) & (fid_ref >= 0)
    p = fid_pred[both]
    r = fid_ref[both]
    code = p * np.int64(max(n_ref, 1)) + r
    uniq, counts = np.unique(code, return_counts=True)
    rows = uniq // max(n_ref, 1)
    cols = uniq % max(n_ref, 1)
    return sparse.coo_matrix((counts.astype(np.float64), (rows, cols)), shape=(n_pred, n_ref))


def _cyclic_equal(a: Sequence[int], b: Sequence[int]) -> bool:
    """True when b is a rotation of a (same direction)."""
    n = len(a)
    if n != len(b):
        return False
    if n == 0:
        return True
    bb = list(b) + list(b)
    aa = list(a)
    return any(bb[k : k + n] == aa for k in range(n))


# ------------------------------------------------------------------- faces
@dataclass
class FaceMatch:
    pairs: List[Tuple[int, int]]  # (pred face, ref face)
    iou: np.ndarray  # per pair
    pred_to_ref: Dict[int, int]
    ref_to_pred: Dict[int, int]
    unmatched_pred: List[int]
    unmatched_ref: List[int]
    split_ref_faces: List[int]  # reference faces covered by >= 2 predicted faces
    merge_pred_faces: List[int]  # predicted faces covering >= 2 reference faces
    intersection: sparse.coo_matrix  # (n_pred, n_ref) pixel counts, cell faces only
    area_pred: np.ndarray  # pixel area per predicted face (all faces, 0 for non-cells)
    area_ref: np.ndarray
    n_pred_cells: int
    n_ref_cells: int
    iou_thresh: float
    fid_pred: np.ndarray = field(repr=False)
    fid_ref: np.ndarray = field(repr=False)

    @property
    def n_splits(self) -> int:
        return len(self.split_ref_faces)

    @property
    def n_merges(self) -> int:
        return len(self.merge_pred_faces)

    def iou_matrix(self) -> sparse.coo_matrix:
        """Sparse IoU with the same pattern as ``intersection``."""
        inter = self.intersection
        union = self.area_pred[inter.row] + self.area_ref[inter.col] - inter.data
        return sparse.coo_matrix((inter.data / union, (inter.row, inter.col)), shape=inter.shape)

    def n_true_positives(self, thresh: float) -> int:
        """Number of one-to-one matches with IoU >= thresh (Hungarian for thresh < 0.5)."""
        iou = self.iou_matrix()
        if thresh >= 0.5:
            # at IoU >= 0.5 a face has at most one partner, so no assignment is needed
            return int(np.count_nonzero(iou.data >= thresh))
        pairs, _ = _assign(iou, thresh)
        return len(pairs)

    def summary(self) -> Dict[str, float]:
        return {
            "n_pred_cells": int(self.n_pred_cells),
            "n_ref_cells": int(self.n_ref_cells),
            "n_matched_cells": len(self.pairs),
            "n_unmatched_pred_cells": len(self.unmatched_pred),
            "n_unmatched_ref_cells": len(self.unmatched_ref),
            "n_splits": int(self.n_splits),
            "n_merges": int(self.n_merges),
            "mean_matched_iou": float(self.iou.mean()) if self.iou.size else 0.0,
        }


def _assign(iou: sparse.coo_matrix, thresh: float) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Maximum-IoU assignment restricted to faces with any overlap, then thresholded."""
    if iou.nnz == 0:
        return [], np.zeros(0)
    rows = np.unique(iou.row)
    cols = np.unique(iou.col)
    ri = np.searchsorted(rows, iou.row)
    ci = np.searchsorted(cols, iou.col)
    dense = np.zeros((rows.size, cols.size), dtype=np.float64)
    dense[ri, ci] = iou.data
    a, b = linear_sum_assignment(dense, maximize=True)
    keep = dense[a, b] >= thresh
    pairs = [(int(rows[i]), int(cols[j])) for i, j in zip(a[keep], b[keep])]
    return pairs, dense[a, b][keep]


def match_faces(
    labels_pred: np.ndarray,
    labels_ref: np.ndarray,
    cx_pred: HalfEdgeComplex,
    cx_ref: HalfEdgeComplex,
    iou_thresh: float = 0.5,
    split_frac: float = 0.2,
) -> FaceMatch:
    """Match predicted cell faces to reference cell faces by pixel IoU.

    A reference face overlapped by >= 2 predicted faces, each covering at least
    ``split_frac`` of its area, is counted as a split; the symmetric case is a merge.
    """
    labels_pred = np.asarray(labels_pred)
    labels_ref = np.asarray(labels_ref)
    if labels_pred.shape != labels_ref.shape:
        raise ValueError("label images must have the same shape")
    fid_p = face_id_image(labels_pred, cx_pred)
    fid_r = face_id_image(labels_ref, cx_ref)
    n_p, n_r = cx_pred.n_faces, cx_ref.n_faces
    area_p = np.bincount(fid_p[fid_p >= 0], minlength=n_p).astype(np.float64)
    area_r = np.bincount(fid_r[fid_r >= 0], minlength=n_r).astype(np.float64)
    inter = _overlap_table(fid_p, fid_r, n_p, n_r)

    union = area_p[inter.row] + area_r[inter.col] - inter.data
    iou = sparse.coo_matrix((inter.data / union, (inter.row, inter.col)), shape=(n_p, n_r))
    pairs, iou_vals = _assign(iou, iou_thresh)
    p2r = {p: r for p, r in pairs}
    r2p = {r: p for p, r in pairs}
    cells_p = [int(f) for f in cx_pred.cell_faces]
    cells_r = [int(f) for f in cx_ref.cell_faces]

    # splits: fraction of the reference face covered by each predicted face
    frac_of_ref = inter.data / area_r[inter.col]
    big = frac_of_ref >= split_frac
    per_ref = np.bincount(inter.col[big], minlength=n_r)
    split_ref = [int(f) for f in np.flatnonzero(per_ref >= 2)]
    frac_of_pred = inter.data / area_p[inter.row]
    big = frac_of_pred >= split_frac
    per_pred = np.bincount(inter.row[big], minlength=n_p)
    merge_pred = [int(f) for f in np.flatnonzero(per_pred >= 2)]

    return FaceMatch(
        pairs=pairs,
        iou=np.asarray(iou_vals, dtype=np.float64),
        pred_to_ref=p2r,
        ref_to_pred=r2p,
        unmatched_pred=[f for f in cells_p if f not in p2r],
        unmatched_ref=[f for f in cells_r if f not in r2p],
        split_ref_faces=split_ref,
        merge_pred_faces=merge_pred,
        intersection=inter,
        area_pred=area_p,
        area_ref=area_r,
        n_pred_cells=len(cells_p),
        n_ref_cells=len(cells_r),
        iou_thresh=float(iou_thresh),
        fid_pred=fid_p,
        fid_ref=fid_r,
    )


# ------------------------------------------------------------------- edges
@dataclass
class EdgeMatch:
    pairs: List[Tuple[int, int]]  # (pred edge, ref edge)
    overlap_pred: np.ndarray  # fraction of pred points within tol of the ref trace
    overlap_ref: np.ndarray
    unmatched_pred: List[int]
    unmatched_ref: List[int]
    n_pred_edges: int
    n_ref_edges: int
    n_pred_pairs: int
    n_ref_pairs: int
    n_pair_tp: int
    arclength_pred: np.ndarray  # smoothed arclength per matched pair
    arclength_ref: np.ndarray
    tol_px: float

    @property
    def pair_precision(self) -> float:
        return _ratio(self.n_pair_tp, self.n_pred_pairs)

    @property
    def pair_recall(self) -> float:
        return _ratio(self.n_pair_tp, self.n_ref_pairs)

    @property
    def pair_f1(self) -> float:
        return _f1(self.pair_precision, self.pair_recall)

    @property
    def component_precision(self) -> float:
        return _ratio(len(self.pairs), self.n_pred_edges)

    @property
    def component_recall(self) -> float:
        return _ratio(len(self.pairs), self.n_ref_edges)

    @property
    def component_f1(self) -> float:
        return _f1(self.component_precision, self.component_recall)

    @property
    def arclength_rel_error(self) -> np.ndarray:
        """Signed relative error (pred - ref) / ref of the smoothed arclength."""
        ref = self.arclength_ref
        out = np.zeros_like(ref)
        ok = ref > 0
        out[ok] = (self.arclength_pred[ok] - ref[ok]) / ref[ok]
        return out

    def summary(self) -> Dict[str, float]:
        rel = self.arclength_rel_error
        has = rel.size > 0
        return {
            "n_pred_edges": int(self.n_pred_edges),
            "n_ref_edges": int(self.n_ref_edges),
            "n_matched_edges": len(self.pairs),
            "n_unmatched_pred_edges": len(self.unmatched_pred),
            "n_unmatched_ref_edges": len(self.unmatched_ref),
            "n_pred_pairs": int(self.n_pred_pairs),
            "n_ref_pairs": int(self.n_ref_pairs),
            "n_pair_tp": int(self.n_pair_tp),
            "adjacency_pair_precision": self.pair_precision,
            "adjacency_pair_recall": self.pair_recall,
            "adjacency_pair_f1": self.pair_f1,
            "adjacency_component_precision": self.component_precision,
            "adjacency_component_recall": self.component_recall,
            "adjacency_component_f1": self.component_f1,
            "arclength_rel_error_mean_abs": float(np.abs(rel).mean()) if has else 0.0,
            "arclength_rel_error_median_abs": float(np.median(np.abs(rel))) if has else 0.0,
            "arclength_rel_error_mean_signed": float(rel.mean()) if has else 0.0,
            "arclength_rel_error_max_abs": float(np.abs(rel).max()) if has else 0.0,
        }


def _trace_overlap(p: np.ndarray, q: np.ndarray, tree_q: cKDTree, tree_p: cKDTree, tol: float) -> Tuple[float, float]:
    dp, _ = tree_q.query(p, distance_upper_bound=tol + 1e-9)
    dq, _ = tree_p.query(q, distance_upper_bound=tol + 1e-9)
    return float(np.mean(dp <= tol)), float(np.mean(dq <= tol))


def match_edges(
    cx_pred: HalfEdgeComplex,
    cx_ref: HalfEdgeComplex,
    face_match: FaceMatch,
    tol_px: float = 3.0,
    min_overlap: float = 0.5,
) -> EdgeMatch:
    """Match cell-cell edges through the face matching and trace overlap.

    Candidate reference edges for a predicted edge are those between the mapped
    face pair. A candidate is accepted when at least ``min_overlap`` of the points
    of each trace lie within ``tol_px`` of the other trace; ties are resolved
    greedily by descending overlap, one-to-one.
    """
    p2r = face_match.pred_to_ref
    edges_p = [int(e) for e in cx_pred.cell_cell_edges()]
    edges_r = [int(e) for e in cx_ref.cell_cell_edges()]

    def pair_of(cx: HalfEdgeComplex, e: int) -> Tuple[int, int]:
        a, b = (int(x) for x in cx.edge_faces[e])
        return (a, b) if a <= b else (b, a)

    pairs_p = {pair_of(cx_pred, e) for e in edges_p}
    pairs_r = {pair_of(cx_ref, e) for e in edges_r}
    mapped = set()
    for a, b in pairs_p:
        if a in p2r and b in p2r:
            ra, rb = p2r[a], p2r[b]
            mapped.add((ra, rb) if ra <= rb else (rb, ra))
    n_pair_tp = len(mapped & pairs_r)

    trees_r: Dict[int, cKDTree] = {}
    trees_p: Dict[int, cKDTree] = {}

    def tree(cache: Dict[int, cKDTree], cx: HalfEdgeComplex, e: int) -> cKDTree:
        t = cache.get(e)
        if t is None:
            t = cKDTree(cx.edge_geometry(e))
            cache[e] = t
        return t

    cands: List[Tuple[float, float, float, int, int]] = []
    for e in edges_p:
        a, b = (int(x) for x in cx_pred.edge_faces[e])
        if a not in p2r or b not in p2r:
            continue
        ref_edges = cx_ref.edges_between(p2r[a], p2r[b])
        if ref_edges.size == 0:
            continue
        p = cx_pred.edge_geometry(e)
        tp = tree(trees_p, cx_pred, e)
        for er in ref_edges:
            er = int(er)
            q = cx_ref.edge_geometry(er)
            fp, fr = _trace_overlap(p, q, tree(trees_r, cx_ref, er), tp, tol_px)
            if fp >= min_overlap and fr >= min_overlap:
                cands.append((min(fp, fr), fp, fr, e, er))
    cands.sort(key=lambda t: (-t[0], -(t[1] + t[2]), t[3], t[4]))

    used_p: set = set()
    used_r: set = set()
    pairs: List[Tuple[int, int]] = []
    ov_p: List[float] = []
    ov_r: List[float] = []
    for _, fp, fr, e, er in cands:
        if e in used_p or er in used_r:
            continue
        used_p.add(e)
        used_r.add(er)
        pairs.append((e, er))
        ov_p.append(fp)
        ov_r.append(fr)

    return EdgeMatch(
        pairs=pairs,
        overlap_pred=np.asarray(ov_p, dtype=np.float64),
        overlap_ref=np.asarray(ov_r, dtype=np.float64),
        unmatched_pred=[e for e in edges_p if e not in used_p],
        unmatched_ref=[e for e in edges_r if e not in used_r],
        n_pred_edges=len(edges_p),
        n_ref_edges=len(edges_r),
        n_pred_pairs=len(pairs_p),
        n_ref_pairs=len(pairs_r),
        n_pair_tp=n_pair_tp,
        arclength_pred=np.asarray([edge_arclength(cx_pred, e) for e, _ in pairs], dtype=np.float64),
        arclength_ref=np.asarray([edge_arclength(cx_ref, er) for _, er in pairs], dtype=np.float64),
        tol_px=float(tol_px),
    )


# ---------------------------------------------------------------- vertices
@dataclass
class VertexMatch:
    pairs: List[Tuple[int, int]]  # (pred vertex, ref vertex)
    distance_px: np.ndarray
    set_agree: np.ndarray  # bool per pair
    order_agree: np.ndarray  # bool per pair (implies set_agree)
    unmatched_pred: List[int]
    unmatched_ref: List[int]
    n_pred_vertices: int
    n_ref_vertices: int
    dist_px: float
    pixel_size_um: Optional[float]

    @property
    def precision(self) -> float:
        return _ratio(len(self.pairs), self.n_pred_vertices)

    @property
    def recall(self) -> float:
        return _ratio(len(self.pairs), self.n_ref_vertices)

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def incident_set_accuracy(self) -> float:
        return _ratio(int(self.set_agree.sum()), self.set_agree.size)

    @property
    def cyclic_order_accuracy(self) -> float:
        return _ratio(int(self.order_agree.sum()), self.order_agree.size)

    @property
    def n_incident_set_mismatches(self) -> int:
        return int(self.set_agree.size - self.set_agree.sum())

    def summary(self) -> Dict[str, float]:
        d = self.distance_px
        has = d.size > 0
        out = {
            "n_pred_vertices": int(self.n_pred_vertices),
            "n_ref_vertices": int(self.n_ref_vertices),
            "n_matched_vertices": len(self.pairs),
            "vertex_precision": self.precision,
            "vertex_recall": self.recall,
            "vertex_f1": self.f1,
            "vertex_loc_error_mean_px": float(d.mean()) if has else 0.0,
            "vertex_loc_error_median_px": float(np.median(d)) if has else 0.0,
            "vertex_loc_error_p95_px": float(np.percentile(d, 95)) if has else 0.0,
            "vertex_incident_set_accuracy": self.incident_set_accuracy,
            "vertex_cyclic_order_accuracy": self.cyclic_order_accuracy,
            "n_vertex_incident_set_mismatches": self.n_incident_set_mismatches,
        }
        if self.pixel_size_um is not None:
            s = float(self.pixel_size_um)
            out["vertex_loc_error_mean_um"] = out["vertex_loc_error_mean_px"] * s
            out["vertex_loc_error_median_um"] = out["vertex_loc_error_median_px"] * s
            out["vertex_loc_error_p95_um"] = out["vertex_loc_error_p95_px"] * s
        return out


def _junction_vertices(cx: HalfEdgeComplex) -> np.ndarray:
    return np.array([v for v in range(cx.n_vertices) if len(cx.vertex_cell_set(v)) >= 3], dtype=np.int64)


def match_vertices(
    cx_pred: HalfEdgeComplex,
    cx_ref: HalfEdgeComplex,
    face_match: FaceMatch,
    dist_px: float = 3.0,
) -> VertexMatch:
    """Match junction vertices (>= 3 incident cells) by mutual nearest position.

    For each matched pair the incident cell set of the predicted vertex is mapped
    through the face matching and compared with the reference set; the cyclic
    order check requires the mapped cell cycle to be a rotation of the reference
    cycle in the same direction.
    """
    vp = _junction_vertices(cx_pred)
    vr = _junction_vertices(cx_ref)
    pairs: List[Tuple[int, int]] = []
    dists: List[float] = []
    if vp.size and vr.size:
        xy_p = cx_pred.vertex_xy[vp]
        xy_r = cx_ref.vertex_xy[vr]
        d_pr, nn_r = cKDTree(xy_r).query(xy_p)
        _, nn_p = cKDTree(xy_p).query(xy_r)
        for i in range(vp.size):
            j = int(nn_r[i])
            if int(nn_p[j]) == i and d_pr[i] <= dist_px:
                pairs.append((int(vp[i]), int(vr[j])))
                dists.append(float(d_pr[i]))

    p2r = face_match.pred_to_ref
    cell_kind_pred = set(int(f) for f in cx_pred.cell_faces)
    cell_kind_ref = set(int(f) for f in cx_ref.cell_faces)
    set_agree = np.zeros(len(pairs), dtype=bool)
    order_agree = np.zeros(len(pairs), dtype=bool)
    for k, (v, w) in enumerate(pairs):
        cyc_p = [int(f) for f in cx_pred.vertex_faces(v) if int(f) in cell_kind_pred]
        cyc_r = [int(f) for f in cx_ref.vertex_faces(w) if int(f) in cell_kind_ref]
        if any(f not in p2r for f in cyc_p):
            continue
        mapped = [p2r[f] for f in cyc_p]
        set_agree[k] = frozenset(mapped) == frozenset(cyc_r)
        order_agree[k] = set_agree[k] and _cyclic_equal(mapped, cyc_r)

    matched_p = {v for v, _ in pairs}
    matched_r = {w for _, w in pairs}
    px_um = cx_ref.pixel_size_um if cx_ref.pixel_size_um is not None else cx_pred.pixel_size_um
    return VertexMatch(
        pairs=pairs,
        distance_px=np.asarray(dists, dtype=np.float64),
        set_agree=set_agree,
        order_agree=order_agree,
        unmatched_pred=[int(v) for v in vp if int(v) not in matched_p],
        unmatched_ref=[int(v) for v in vr if int(v) not in matched_r],
        n_pred_vertices=int(vp.size),
        n_ref_vertices=int(vr.size),
        dist_px=float(dist_px),
        pixel_size_um=px_um,
    )


__all__ = [
    "EdgeMatch",
    "FaceMatch",
    "VertexMatch",
    "face_id_image",
    "match_edges",
    "match_faces",
    "match_vertices",
]
