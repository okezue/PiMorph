"""Constrained decoder: proposal maps -> legal embedded complex.

Representation validity is guaranteed by construction: any label image yields a
valid complex through ``extract_complex``. What the decoder decides is WHICH label
image, driven by seeds (one marker per candidate cell), the boundary map (watershed
elevation) and the gap map (pixels excluded from cells). Cells are labelled by their
seed index + 1 so that faces are comparable across hypotheses that share a seed list.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from ..complex.extract import extract_complex
from ..complex.geometry import smooth_complex
from ..complex.halfedge import HalfEdgeComplex
from .proposals import ProposalMaps


@dataclass
class DecoderParams:
    seed_threshold: float = 0.3  # keep seeds with score >= threshold
    seed_drop_frac: float = 0.0  # additionally drop this fraction of the weakest kept seeds
    boundary_gamma: float = 1.0  # elevation = boundary ** gamma (matters only when mixed with distance)
    boundary_smooth_sigma: float = 0.0  # Gaussian smoothing of the boundary map before flooding
    distance_mix: float = 0.0  # elevation = (1-a) * boundary + a * (distance to nearest seed / cell_radius)
    # elevation += vertex_weight * vertex probability: the flood fronts of three cells then
    # meet where the vertex head predicts the junction, instead of wherever the thick
    # boundary ridge happens to be lowest
    vertex_weight: float = 0.0
    cell_radius_px: float = 15.0
    compactness: float = 0.0
    gap_threshold: float = 0.7  # pixels with gap >= threshold are excluded from cells
    # pixels whose predicted signed distance is below -outside_px are outside every cell
    # and excluded too (None disables). The gap head only knows ENCLOSED background, so
    # without this the flood runs through open background and joins cells that never
    # touch (HAEC field 0005: 1,167 predicted multicellular vertices vs 81 true at None,
    # 89 at 0.0). Values >= 1 px carve slivers along real contacts in confluent tissue.
    outside_px: Optional[float] = 0.0
    min_cell_area_px: int = 60
    min_gap_area_px: int = 12
    fill_gaps: bool = False  # True ignores gap evidence entirely (no explicit gaps)
    smooth_iters: int = 4
    # vertex-consistency merges: a cell-cell edge whose mean boundary support is below
    # merge_boundary_max and whose end vertices have no vertex-head support above
    # merge_vertex_min is a false split; the two cells are merged (0 merges disables)
    merge_boundary_max: float = 0.0
    merge_vertex_min: float = 0.5
    merge_vertex_radius_px: float = 3.0
    max_merges: int = 0
    # nucleus-consistency merges (needs maps.meta["nucleus_points"]): a cell containing no
    # nuclear peak and smaller than nucleus_merge_max_area_frac x the median cell area is a
    # split fragment; it joins the neighbour across its weakest boundary
    nucleus_merge: bool = False
    nucleus_merge_max_area_frac: float = 1.0
    nucleus_merge_boundary_max: float = 1.0  # only across edges with mean boundary prob below this
    label: str = ""

    def with_(self, **kw) -> "DecoderParams":
        return replace(self, **kw)


@dataclass
class DecodeResult:
    labels: np.ndarray  # int32, 0 background; cell label = master seed index + 1
    cx: HalfEdgeComplex
    seed_ids: np.ndarray  # master seed indices used (label - 1)
    params: DecoderParams
    info: Dict = field(default_factory=dict)


def merge_small_regions(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Merge components below ``min_area`` into the neighbouring label sharing the
    longest 4-neighbour contact; components with no labelled neighbour become 0."""
    lab = labels.copy()
    counts = np.bincount(lab.ravel())
    small = np.flatnonzero(counts < min_area)
    small = small[small > 0]
    if small.size == 0:
        return lab
    # 4-neighbour contact counts between labels via pair codes
    a = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    b = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    m = a != b
    a, b = a[m], b[m]
    for s in small:
        sel_a = a == s
        sel_b = b == s
        neigh = np.concatenate([b[sel_a], a[sel_b]])
        neigh = neigh[neigh > 0]
        if neigh.size == 0:
            lab[lab == s] = 0
            continue
        tgt = int(np.bincount(neigh).argmax())
        lab[lab == s] = tgt
        a[a == s] = tgt
        b[b == s] = tgt
    return lab


def _fill_small_background(labels: np.ndarray, min_gap_area: int) -> np.ndarray:
    """Assign enclosed background components smaller than ``min_gap_area`` to the
    nearest cell pixel by pixel (they are segmentation debris or annotation seams,
    not gaps). Vectorized: label images with thousands of 1 px seams (FlyWing has
    about 10,000 per field) fill in well under a second."""
    bg = cc_label(labels == 0, connectivity=1)
    if bg.max() == 0:
        return labels
    counts = np.bincount(bg.ravel())
    border = np.zeros(bg.max() + 1, dtype=bool)
    for edge in (bg[0], bg[-1], bg[:, 0], bg[:, -1]):
        border[np.unique(edge)] = True
    small = (counts < min_gap_area) & ~border
    small[0] = False
    fill = small[bg]
    if not fill.any():
        return labels
    idx = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    lab = labels.copy()
    lab[fill] = labels[idx[0][fill], idx[1][fill]]
    return lab


def edge_support(cx: HalfEdgeComplex, prob: np.ndarray, e: int) -> float:
    """Mean of a probability map along the pixel path of edge ``e``."""
    H, W = prob.shape
    p = cx.edge_geometry(int(e))
    if p.shape[0] == 0:
        return 0.0
    pr = np.clip(np.round(p[:, 0]).astype(int), 0, H - 1)
    pc = np.clip(np.round(p[:, 1]).astype(int), 0, W - 1)
    return float(prob[pr, pc].mean())


def _local_max(prob: np.ndarray, rc, radius: float) -> float:
    H, W = prob.shape
    r, c = int(round(rc[0])), int(round(rc[1]))
    k = int(np.ceil(radius))
    win = prob[max(r - k, 0) : min(r + k + 1, H), max(c - k, 0) : min(c + k + 1, W)]
    return float(win.max()) if win.size else 0.0


def vertex_consistent_merges(
    labels: np.ndarray, cx: HalfEdgeComplex, maps: ProposalMaps, params: DecoderParams
) -> tuple[np.ndarray, int]:
    """Merge cell pairs across edges that the evidence does not support.

    A false split of one cell produces an edge with weak boundary probability whose
    two ends create vertices the vertex head does not predict. Such edges are
    removed by merging their two cells, weakest first, at most ``params.max_merges``
    per call and each cell at most once. The vertex test is skipped when the
    proposal has no vertex map.
    """
    cands = []
    for e in cx.cell_cell_edges():
        e = int(e)
        fa, fb = (int(f) for f in cx.edge_faces[e])
        a, b = int(cx.face_label[fa]), int(cx.face_label[fb])
        if a == b or a <= 0 or b <= 0:
            continue
        b_sup = edge_support(cx, maps.boundary, e)
        if b_sup >= params.merge_boundary_max:
            continue
        if maps.vertex is not None:
            ends = (int(cx.edge_tail[e]), int(cx.edge_head[e]))
            v_sup = max(_local_max(maps.vertex, cx.vertex_xy[v], params.merge_vertex_radius_px) for v in ends)
            if v_sup >= params.merge_vertex_min:
                continue
        cands.append((b_sup, a, b))
    cands.sort()
    lab = labels.copy()
    touched: set = set()
    n = 0
    for _, a, b in cands:
        if a in touched or b in touched:
            continue
        lab[lab == b] = a
        touched.update((a, b))
        n += 1
        if n >= params.max_merges:
            break
    return lab, n


def nucleus_consistency_merges(
    labels: np.ndarray, cx: HalfEdgeComplex, maps: ProposalMaps, params: DecoderParams
) -> tuple[np.ndarray, int]:
    """Merge cells that contain no nuclear peak into the neighbour across their weakest
    boundary. The one-nucleus-per-cell prior applied after decoding: split fragments
    have no nucleus, real cells almost always do. Only cells below
    ``nucleus_merge_max_area_frac`` x the median cell area qualify (a large cell with a
    dim nucleus is kept), and only edges below ``nucleus_merge_boundary_max`` mean
    boundary probability are crossed. Each cell merges at most once per call."""
    pts = maps.meta.get("nucleus_points")
    if pts is None:
        return labels, 0
    pts = np.asarray(pts).reshape(-1, 2)
    H, W = labels.shape
    counts = np.bincount(labels.ravel())
    n_nuc = np.zeros_like(counts)
    if len(pts):
        pr = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)
        owner = labels[pr, pc]
        n_nuc = np.bincount(owner[owner > 0], minlength=counts.size)
    areas = counts[1:][counts[1:] > 0]
    if areas.size == 0:
        return labels, 0
    max_area = params.nucleus_merge_max_area_frac * float(np.median(areas))
    # weakest edge per empty cell
    best: Dict[int, tuple] = {}
    for e in cx.cell_cell_edges():
        e = int(e)
        fa, fb = (int(f) for f in cx.edge_faces[e])
        a, b = int(cx.face_label[fa]), int(cx.face_label[fb])
        if a == b or a <= 0 or b <= 0:
            continue
        sup = None
        for cell, other in ((a, b), (b, a)):
            if cell >= counts.size or n_nuc[cell] > 0 or counts[cell] > max_area or counts[cell] == 0:
                continue
            if sup is None:
                sup = edge_support(cx, maps.boundary, e)
            if sup < params.nucleus_merge_boundary_max and (cell not in best or sup < best[cell][0]):
                best[cell] = (sup, other)
    lab = labels.copy()
    touched: set = set()
    n = 0
    for cell, (_, other) in sorted(best.items(), key=lambda kv: kv[1][0]):
        if cell in touched or other in touched:
            continue
        lab[lab == cell] = other
        touched.update((cell, other))
        n += 1
    return lab, n


class ConstrainedDecoder:
    def __init__(self, pixel_size_um: Optional[float] = None):
        self.pixel_size_um = pixel_size_um

    def select_seeds(self, maps: ProposalMaps, params: DecoderParams) -> np.ndarray:
        keep = np.flatnonzero(maps.seed_scores >= params.seed_threshold)
        if params.seed_drop_frac > 0 and keep.size:
            # seed_points are sorted strongest first, so the weakest are at the end
            n_drop = int(round(params.seed_drop_frac * keep.size))
            if n_drop:
                keep = keep[: keep.size - n_drop]
        return keep

    @staticmethod
    def elevation(maps: ProposalMaps, params: DecoderParams, markers: np.ndarray) -> np.ndarray:
        b = np.clip(maps.boundary, 0.0, 1.0).astype(np.float32)
        if params.boundary_smooth_sigma > 0:
            b = ndi.gaussian_filter(b, params.boundary_smooth_sigma)
        b = np.power(b, params.boundary_gamma)
        if params.vertex_weight > 0 and maps.vertex is not None:
            b = b + params.vertex_weight * np.clip(maps.vertex, 0.0, 1.0)
        if params.distance_mix > 0:
            d = ndi.distance_transform_edt(markers == 0) / max(params.cell_radius_px, 1.0)
            b = (1.0 - params.distance_mix) * b + params.distance_mix * np.clip(d, 0.0, 3.0) / 3.0
        return b.astype(np.float32)

    def decode(
        self,
        maps: ProposalMaps,
        params: DecoderParams,
        seed_ids: Optional[np.ndarray] = None,
        extra_seed_points: Optional[np.ndarray] = None,
    ) -> DecodeResult:
        H, W = maps.shape
        if seed_ids is None:
            seed_ids = self.select_seeds(maps, params)
        seed_ids = np.asarray(seed_ids, dtype=np.int64)
        markers = np.zeros((H, W), dtype=np.int32)
        pts = np.round(maps.seed_points[seed_ids]).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
        markers[pts[:, 0], pts[:, 1]] = seed_ids + 1
        n_master = int(maps.seed_points.shape[0])
        if extra_seed_points is not None and len(extra_seed_points):
            ep = np.round(np.asarray(extra_seed_points)).astype(int)
            for k, (r, c) in enumerate(ep):
                markers[np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)] = n_master + 1 + k

        elevation = self.elevation(maps, params, markers)
        mask = maps.tissue.copy()
        if not params.fill_gaps:
            mask &= maps.gap < params.gap_threshold
            if params.outside_px is not None and maps.distance is not None:
                mask &= maps.distance >= -float(params.outside_px)
            # never exclude a seed pixel
            mask[markers > 0] = True

        lab = watershed(elevation, markers=markers, mask=mask, compactness=params.compactness).astype(np.int32)
        lab = _fill_small_background(lab, params.min_gap_area_px)
        lab = merge_small_regions(lab, params.min_cell_area_px)

        provenance = {"decoder": "watershed", "params": params.__dict__, "proposal_source": maps.source}
        cx = extract_complex(lab, pixel_size_um=self.pixel_size_um, provenance=provenance)
        n_merged = 0
        if params.max_merges > 0:
            lab, n_merged = vertex_consistent_merges(lab, cx, maps, params)
            if n_merged:
                cx = extract_complex(lab, pixel_size_um=self.pixel_size_um, provenance=provenance)
        n_nuc_merged = 0
        if params.nucleus_merge:
            lab, n_nuc_merged = nucleus_consistency_merges(lab, cx, maps, params)
            if n_nuc_merged:
                cx = extract_complex(lab, pixel_size_um=self.pixel_size_um, provenance=provenance)
        smooth_complex(cx, iterations=params.smooth_iters)
        used = np.unique(lab[lab > 0]) - 1
        return DecodeResult(
            labels=lab,
            cx=cx,
            seed_ids=used.astype(np.int64),
            params=params,
            info={
                "n_seeds_selected": int(seed_ids.size),
                "n_cells": int(cx.cell_faces.size),
                "n_gaps": int(cx.gap_faces.size),
                "n_vertex_merges": int(n_merged),
                "n_nucleus_merges": int(n_nuc_merged),
            },
        )

    def merge_faces(self, result: DecodeResult, label_a: int, label_b: int) -> DecodeResult:
        """Local move: hypothesise that two seeds belong to one cell."""
        lab = result.labels.copy()
        lab[lab == label_b] = label_a
        cx = extract_complex(lab, pixel_size_um=self.pixel_size_um, provenance=dict(result.cx.provenance))
        smooth_complex(cx, iterations=result.params.smooth_iters)
        cx.provenance["move"] = {"merge": [int(label_a), int(label_b)]}
        used = np.unique(lab[lab > 0]) - 1
        return DecodeResult(
            labels=lab, cx=cx, seed_ids=used, params=result.params, info=dict(result.info, move="merge")
        )

    def split_face(
        self, maps: ProposalMaps, result: DecodeResult, label: int, new_seed_rc, new_label: int
    ) -> DecodeResult:
        """Local move: re-flood one cell from its own seed plus an extra seed, keeping
        everything else fixed. ``new_label`` must not collide with existing labels."""
        lab = result.labels.copy()
        region = lab == label
        if not region.any():
            return result
        H, W = lab.shape
        markers = np.zeros((H, W), dtype=np.int32)
        own = np.round(maps.seed_points[label - 1]).astype(int) if 0 <= label - 1 < len(maps.seed_points) else None
        if own is None or not region[np.clip(own[0], 0, H - 1), np.clip(own[1], 0, W - 1)]:
            # fall back to the farthest interior point of the region
            d = ndi.distance_transform_edt(region)
            own = np.unravel_index(int(np.argmax(d)), d.shape)
        r2, c2 = (int(np.clip(new_seed_rc[0], 0, H - 1)), int(np.clip(new_seed_rc[1], 0, W - 1)))
        if not region[r2, c2] or (r2, c2) == (int(own[0]), int(own[1])):
            return result
        markers[int(own[0]), int(own[1])] = label
        markers[r2, c2] = new_label
        elev = self.elevation(maps, result.params, markers)
        sub = watershed(elev, markers=markers, mask=region, compactness=result.params.compactness)
        lab[region] = sub[region]
        cx = extract_complex(lab, pixel_size_um=self.pixel_size_um, provenance=dict(result.cx.provenance))
        smooth_complex(cx, iterations=result.params.smooth_iters)
        cx.provenance["move"] = {"split": [int(label), int(new_label)]}
        used = np.unique(lab[lab > 0]) - 1
        return DecodeResult(
            labels=lab, cx=cx, seed_ids=used, params=result.params, info=dict(result.info, move="split")
        )
