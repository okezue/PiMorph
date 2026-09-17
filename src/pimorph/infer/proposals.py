"""Proposal maps: dense per-pixel evidence consumed by the constrained decoder.

The same ``ProposalMaps`` container is produced by the classical filters here and by
the neural multi-head model (``pimorph.infer.neural``), so the decoder and the
posterior machinery are agnostic to where the evidence came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, sato, threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects


@dataclass
class ProposalMaps:
    boundary: np.ndarray  # (H,W) float32 in [0,1]
    seed: np.ndarray  # (H,W) float32 heatmap in [0,1]
    tissue: np.ndarray  # (H,W) bool
    gap: np.ndarray  # (H,W) float32 in [0,1]
    sigma: np.ndarray  # (H,W) float32 local noise scale (image units)
    seed_points: np.ndarray  # (N,2) float (row, col) candidate seeds, strongest first
    seed_scores: np.ndarray  # (N,) float in [0,1]
    vertex: Optional[np.ndarray] = None
    distance: Optional[np.ndarray] = None
    source: str = "classical"
    meta: Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.boundary.shape[0]), int(self.boundary.shape[1])


def robust_normalize(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.8) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def estimate_noise_sigma(img: np.ndarray, window: int = 32) -> np.ndarray:
    """Local noise scale from the MAD of a Laplacian-like residual.

    For white noise the Laplacian stencil [[0,1,0],[1,-4,1],[0,1,0]] has variance
    20 sigma^2, so the residual MAD is divided by 1.4826 * sqrt(20).
    """
    x = np.asarray(img, dtype=np.float32)
    lap = ndi.laplace(x)
    med = ndi.median_filter(np.abs(lap), size=window)
    return (med / (1.4826 * np.sqrt(20.0))).astype(np.float32)


def tissue_mask(
    img: np.ndarray, scale_px: float, seed_points: Optional[np.ndarray] = None, rel_thresh: float = 0.08
) -> np.ndarray:
    """Permissive foreground mask: blurred signal above a small fraction of its robust
    maximum, unioned with disks around seeds, holes filled.

    A confluent monolayer imaged with a junction marker is dark inside cells, so an
    Otsu split of blurred intensity would wrongly call sparse-junction regions
    background. Explicit gaps are the job of the gap map, not this mask.
    """
    x = robust_normalize(img)
    blurred = gaussian(x, sigma=scale_px, preserve_range=True)
    hi = np.percentile(blurred, 99.5)
    mask = blurred > rel_thresh * hi
    if seed_points is not None and len(seed_points):
        H, W = x.shape
        pts = np.round(np.asarray(seed_points)).astype(int)
        s = np.zeros((H, W), dtype=bool)
        s[np.clip(pts[:, 0], 0, H - 1), np.clip(pts[:, 1], 0, W - 1)] = True
        mask |= ndi.distance_transform_edt(~s) <= 2.0 * scale_px
    mask = ndi.binary_fill_holes(mask)
    npx = max(int(mask.size * 0.002), 500)
    mask = remove_small_objects(mask, max_size=npx - 1)
    mask = remove_small_holes(mask, max_size=npx - 1)
    if mask.mean() > 0.97:
        return np.ones(x.shape, dtype=bool)
    return mask


def estimate_ridge_width(img: np.ndarray) -> float:
    """Median full width (px) of bright ridges: 2x distance transform sampled on the
    skeleton of the Otsu-thresholded image."""
    from skimage.morphology import skeletonize

    x = robust_normalize(img)
    m = x > threshold_otsu(x)
    m = remove_small_objects(m, max_size=19)
    if m.sum() == 0:
        return 3.0
    dist = ndi.distance_transform_edt(m)
    sk = skeletonize(m)
    vals = dist[sk]
    if vals.size == 0:
        return 3.0
    return float(np.clip(2.0 * np.median(vals), 1.5, 20.0))


def estimate_nucleus_radius(nuclei: np.ndarray, min_area_px: int = 30) -> float:
    """Median equivalent radius of Otsu-thresholded nuclear blobs (px)."""
    from skimage.measure import label as cc_label

    n = robust_normalize(nuclei)
    ns = gaussian(n, sigma=1.0, preserve_range=True)
    m = ns > threshold_otsu(ns)
    m = remove_small_objects(m, max_size=min_area_px - 1)
    lab = cc_label(m, connectivity=1)
    if lab.max() == 0:
        return 6.0
    areas = np.bincount(lab.ravel())[1:]
    areas = areas[areas >= min_area_px]
    if areas.size == 0:
        return 6.0
    return float(np.sqrt(np.median(areas) / np.pi))


@dataclass
class ClassicalProposer:
    """Filter-based proposals.

    boundary: bright-ridge response (Sato) of the geometry channel mixed with its
    normalized intensity, rescaled to [0,1].
    seed: Gaussian bumps at nuclear blob maxima (LoG-smoothed nuclei channel); when
    no nuclei channel exists, maxima of the distance-to-boundary map are used.
    gap: dark, ridge-free pixels far from any seed.
    """

    ridge_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    intensity_weight: float = 0.4
    nucleus_radius_px: float = 6.0
    seed_min_distance_px: int = 6
    seed_rel_threshold: float = 0.05
    cell_radius_px: float = 15.0
    seed_heatmap_sigma_px: float = 3.0
    use_tissue_mask: bool = True
    # When True and a nuclei channel is given, nucleus_radius_px is estimated from the
    # data and the other scale parameters are derived from it.
    auto_scale: bool = True
    cell_to_nucleus_ratio: float = 2.5

    def _scaled(self, geometry: np.ndarray, nuclei: Optional[np.ndarray]) -> "ClassicalProposer":
        if not self.auto_scale or nuclei is None:
            return self
        r = estimate_nucleus_radius(nuclei)
        w = estimate_ridge_width(geometry)
        return ClassicalProposer(
            ridge_sigmas=tuple(float(s) for s in np.clip(np.array([0.35, 0.6, 1.0]) * w, 0.7, 6.0)),
            intensity_weight=self.intensity_weight,
            nucleus_radius_px=r,
            seed_min_distance_px=max(int(round(r)), 3),
            seed_rel_threshold=self.seed_rel_threshold,
            cell_radius_px=self.cell_to_nucleus_ratio * r,
            seed_heatmap_sigma_px=max(r / 2.0, 1.5),
            use_tissue_mask=self.use_tissue_mask,
            auto_scale=False,
            cell_to_nucleus_ratio=self.cell_to_nucleus_ratio,
        )

    def __call__(
        self,
        geometry: np.ndarray,
        nuclei: Optional[np.ndarray] = None,
        junction: Optional[np.ndarray] = None,
    ) -> ProposalMaps:
        if self.auto_scale and nuclei is not None:
            return self._scaled(geometry, nuclei)(geometry, nuclei, junction)
        g = robust_normalize(geometry)
        ridge = sato(g, sigmas=self.ridge_sigmas, black_ridges=False)
        ridge = robust_normalize(ridge, 0.0, 99.5)
        boundary = np.clip((1.0 - self.intensity_weight) * ridge + self.intensity_weight * g, 0.0, 1.0)
        boundary = robust_normalize(boundary, 0.0, 99.9).astype(np.float32)

        if nuclei is not None:
            n = robust_normalize(nuclei)
            ns = gaussian(n, sigma=self.nucleus_radius_px / 2.0, preserve_range=True)
            pts = peak_local_max(
                ns,
                min_distance=self.seed_min_distance_px,
                threshold_rel=self.seed_rel_threshold,
                exclude_border=False,
            )
            scores = ns[pts[:, 0], pts[:, 1]] if len(pts) else np.zeros(0)
        else:
            # distance-to-boundary maxima as a nucleus-free fallback
            bmask = boundary > 0.5
            dist = ndi.distance_transform_edt(~bmask)
            pts = peak_local_max(dist, min_distance=int(self.cell_radius_px * 0.6), exclude_border=False)
            scores = dist[pts[:, 0], pts[:, 1]] / max(dist.max(), 1e-6) if len(pts) else np.zeros(0)
        if len(pts):
            order = np.argsort(-scores)
            pts, scores = pts[order], scores[order]
            # normalize by a high percentile so one very bright nucleus does not push
            # ordinary nuclei below the seed threshold
            ref = float(np.percentile(scores, 90)) if scores.size >= 10 else float(scores.max())
            scores = np.clip(scores / max(ref, 1e-6), 0.0, 1.0)

        tissue = (
            tissue_mask(geometry, scale_px=self.cell_radius_px, seed_points=pts)
            if self.use_tissue_mask
            else np.ones(g.shape, dtype=bool)
        )
        if len(pts):
            inside = tissue[pts[:, 0], pts[:, 1]]
            pts, scores = pts[inside], scores[inside]
        seed = np.zeros(g.shape, dtype=np.float32)
        if len(pts):
            seed[pts[:, 0], pts[:, 1]] = scores
            seed = gaussian(seed, sigma=self.seed_heatmap_sigma_px, preserve_range=True)
            seed = (seed / max(float(seed.max()), 1e-6)).astype(np.float32)

        # gap evidence: far from seeds, dark, and without ridge support
        if len(pts):
            seed_img = np.zeros(g.shape, dtype=bool)
            seed_img[pts[:, 0], pts[:, 1]] = True
            dseed = ndi.distance_transform_edt(~seed_img)
        else:
            dseed = np.full(g.shape, 4 * self.cell_radius_px, dtype=np.float32)
        # elongated cells put their far edge well beyond one cell radius from the nucleus
        far = 1.0 / (1.0 + np.exp(-(dseed - 2.5 * self.cell_radius_px) / (0.5 * self.cell_radius_px)))
        gap = (far * (1.0 - boundary) * (1.0 - g)).astype(np.float32)
        gap[~tissue] = 1.0

        sigma = estimate_noise_sigma(geometry)

        return ProposalMaps(
            boundary=boundary,
            seed=seed,
            tissue=tissue,
            gap=gap,
            sigma=sigma,
            seed_points=pts.astype(np.float64).reshape(-1, 2),
            seed_scores=np.asarray(scores, dtype=np.float32).reshape(-1),
            source="classical",
            meta={
                "ridge_sigmas": list(self.ridge_sigmas),
                "ridge_width_px": float(estimate_ridge_width(geometry)),
                "nucleus_radius_px": float(self.nucleus_radius_px),
                "cell_radius_px": float(self.cell_radius_px),
                "nuclei_used": nuclei is not None,
                "n_seed_candidates": int(len(pts)),
            },
        )
