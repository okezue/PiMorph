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
    cell_radius_px: float = 15.0
    compactness: float = 0.0
    gap_threshold: float = 0.7  # pixels with gap >= threshold are excluded from cells
    min_cell_area_px: int = 60
    min_gap_area_px: int = 12
    fill_gaps: bool = False  # True ignores gap evidence entirely (no explicit gaps)
    smooth_iters: int = 4
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
    dominant neighbouring cell (they are segmentation debris, not gaps)."""
    bg = cc_label(labels == 0, connectivity=1)
    if bg.max() == 0:
        return labels
    counts = np.bincount(bg.ravel())
    border = np.zeros(bg.max() + 1, dtype=bool)
    for edge in (bg[0], bg[-1], bg[:, 0], bg[:, -1]):
        border[np.unique(edge)] = True
    lab = labels.copy()
    small = [i for i in range(1, bg.max() + 1) if counts[i] < min_gap_area and not border[i]]
    for i in small:
        m = bg == i
        ring = ndi.binary_dilation(m, structure=np.ones((3, 3), bool)) & ~m
        vals = lab[ring]
        vals = vals[vals > 0]
        if vals.size:
            lab[m] = int(np.bincount(vals).argmax())
    return lab


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
            # never exclude a seed pixel
            mask[markers > 0] = True

        lab = watershed(elevation, markers=markers, mask=mask, compactness=params.compactness).astype(np.int32)
        lab = _fill_small_background(lab, params.min_gap_area_px)
        lab = merge_small_regions(lab, params.min_cell_area_px)

        cx = extract_complex(
            lab,
            pixel_size_um=self.pixel_size_um,
            provenance={"decoder": "watershed", "params": params.__dict__, "proposal_source": maps.source},
        )
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
