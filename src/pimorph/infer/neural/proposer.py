"""Neural proposer: checkpoint -> ``ProposalMaps`` with the classical proposer's call
signature, so the decoder and posterior code do not care which produced the maps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes, remove_small_objects

from ..proposals import ProposalMaps, estimate_nucleus_radius, estimate_ridge_width, robust_normalize
from .data import build_input
from .model import DISTANCE_SCALE, HEAD_INDEX, MultiHeadUNet
from .train import load_model, resolve_device


def nucleus_peaks(nuclei: np.ndarray, nucleus_radius_px: float, rel_threshold: float = 0.15) -> np.ndarray:
    """(N, 2) nuclear intensity peaks: smoothed at half the nucleus radius, at least one
    radius apart, above ``rel_threshold`` of the maximum (same rule as ClassicalProposer)."""
    n = robust_normalize(np.asarray(nuclei, dtype=np.float32))
    ns = gaussian(n, sigma=max(nucleus_radius_px / 2.0, 1.0), preserve_range=True)
    pts = peak_local_max(
        ns, min_distance=max(int(round(nucleus_radius_px)), 3), threshold_rel=rel_threshold, exclude_border=False
    )
    return np.asarray(pts, dtype=np.int64).reshape(-1, 2)


def _taper(n: int, overlap: int) -> np.ndarray:
    """1-D blending weight: cosine ramps of length ``overlap`` at both ends, 1 inside."""
    w = np.ones(n, dtype=np.float32)
    ramp = min(int(overlap), n // 2)
    if ramp > 0:
        t = (np.arange(ramp) + 0.5) / ramp
        edge = 0.5 * (1.0 - np.cos(np.pi * t))
        w[:ramp] = edge
        w[n - ramp :] = edge[::-1]
    return np.maximum(w, 1e-3)


def _starts(n: int, tile: int, stride: int) -> List[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] + tile < n:
        s.append(n - tile)
    return s


def _pad_to_multiple(x: np.ndarray, m: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = x.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect"), (ph, pw)


@torch.no_grad()
def tiled_predict(
    model: MultiHeadUNet,
    x: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Run ``model`` over an (C, H, W) input in overlapping tiles with cosine blending.
    The image is reflect-padded by ``overlap // 2`` so no true border pixel sits at a
    tile edge; tiles are padded to the network's spatial divisor."""
    device = device or next(model.parameters()).device
    C, H, W = x.shape
    pad = overlap // 2 if (H > tile or W > tile) else 0
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="reflect") if pad else x
    Hp, Wp = xp.shape[-2:]
    th, tw = min(tile, Hp), min(tile, Wp)
    stride = max(tile - overlap, 1)
    windows = [(r, c) for r in _starts(Hp, th, stride) for c in _starts(Wp, tw, stride)]
    weight2d = np.outer(_taper(th, overlap), _taper(tw, overlap)).astype(np.float32)
    out = np.zeros((model.out_channels, Hp, Wp), dtype=np.float32)
    acc = np.zeros((Hp, Wp), dtype=np.float32)
    m = model.divisor
    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        crops = [xp[:, r : r + th, c : c + tw] for r, c in batch]
        padded = [_pad_to_multiple(cr, m) for cr in crops]
        xb = torch.from_numpy(np.stack([p[0] for p in padded])).to(device)
        yb = model(xb).float().cpu().numpy()
        for (r, c), (_, (ph, pw)), yy in zip(batch, padded, yb):
            yy = yy[:, : yy.shape[1] - ph, : yy.shape[2] - pw]
            out[:, r : r + th, c : c + tw] += yy * weight2d
            acc[r : r + th, c : c + tw] += weight2d
    out /= np.maximum(acc, 1e-6)
    return out[:, pad : pad + H, pad : pad + W]


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))).astype(np.float32)


def _clean_tissue(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask)
    npx = max(int(mask.size * 0.002), 500)
    mask = remove_small_objects(mask, max_size=npx - 1)
    mask = remove_small_holes(mask, max_size=npx - 1)
    if mask.mean() > 0.97 or not mask.any():
        return np.ones(mask.shape, dtype=bool)
    return mask


class NeuralProposer:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        tile: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
        seed_rel_threshold: float = 0.05,
        cell_to_nucleus_ratio: float = 2.5,
        nucleus_seeds: bool = True,
        nucleus_seed_score: float = 0.5,
        tta: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.batch_size = int(batch_size)
        self.seed_rel_threshold = float(seed_rel_threshold)
        self.cell_to_nucleus_ratio = float(cell_to_nucleus_ratio)
        # nuclear peaks with no neural seed within 0.6 cell radii are added as seeds
        # (the one-nucleus-per-cell prior applied where the seed head is silent, e.g.
        # in regions of weak membrane signal that annotators still split by nuclei)
        self.nucleus_seeds = bool(nucleus_seeds)
        self.nucleus_seed_score = float(nucleus_seed_score)
        # test-time augmentation: average the raw heads over the 8 dihedral transforms
        self.tta = bool(tta)

    def predict_raw(
        self, geometry: np.ndarray, nuclei: Optional[np.ndarray] = None, junction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """(6, H, W) raw head outputs (logits / scaled regression) in HEADS order."""
        shape = np.asarray(geometry).shape
        x = build_input(geometry, nuclei, junction, shape, normalize=True)
        if not self.tta:
            return tiled_predict(self.model, x, self.tile, self.overlap, self.device, self.batch_size)
        acc = None
        for k in range(4):
            for flip in (False, True):
                xt = np.rot90(x, k, axes=(1, 2))
                if flip:
                    xt = xt[:, :, ::-1]
                out = tiled_predict(
                    self.model, np.ascontiguousarray(xt), self.tile, self.overlap, self.device, self.batch_size
                )
                if flip:
                    out = out[:, :, ::-1]
                out = np.rot90(out, -k, axes=(1, 2))
                acc = out.astype(np.float64) if acc is None else acc + out
        return (acc / 8.0).astype(np.float32)

    def __call__(
        self,
        geometry: np.ndarray,
        nuclei: Optional[np.ndarray] = None,
        junction: Optional[np.ndarray] = None,
        tissue: Optional[np.ndarray] = None,
    ) -> ProposalMaps:
        geometry = np.asarray(geometry, dtype=np.float32)
        raw = self.predict_raw(geometry, nuclei, junction)
        boundary = _sigmoid(raw[HEAD_INDEX["boundary"]])
        seed = _sigmoid(raw[HEAD_INDEX["seed"]])
        seed = (seed / max(float(seed.max()), 1e-6)).astype(np.float32)
        vertex = _sigmoid(raw[HEAD_INDEX["vertex"]])
        gap = _sigmoid(raw[HEAD_INDEX["gap"]])
        distance = (raw[HEAD_INDEX["distance"]] * DISTANCE_SCALE).astype(np.float32)
        sigma = (np.exp(np.clip(raw[HEAD_INDEX["log_sigma"]], -6.0, 6.0)) * DISTANCE_SCALE).astype(np.float32)

        # cell scale: nuclei when available, otherwise distance maxima at provisional seeds
        nucleus_radius = None
        if nuclei is not None:
            nucleus_radius = estimate_nucleus_radius(nuclei)
            cell_radius = self.cell_to_nucleus_ratio * nucleus_radius
        else:
            pts0 = peak_local_max(seed, min_distance=3, threshold_rel=self.seed_rel_threshold, exclude_border=False)
            d0 = distance[pts0[:, 0], pts0[:, 1]] if len(pts0) else np.zeros(0)
            d0 = d0[d0 > 0]
            cell_radius = float(np.clip(2.0 * np.median(d0), 3.0, 60.0)) if d0.size >= 3 else 15.0

        pts = peak_local_max(
            seed,
            min_distance=max(3, int(0.6 * cell_radius)),
            threshold_rel=self.seed_rel_threshold,
            exclude_border=False,
        )
        scores = seed[pts[:, 0], pts[:, 1]] if len(pts) else np.zeros(0, dtype=np.float32)
        if len(pts):
            order = np.argsort(-scores)
            pts, scores = pts[order], scores[order]
            ref = float(np.percentile(scores, 90)) if scores.size >= 10 else float(scores.max())
            scores = np.clip(scores / max(ref, 1e-6), 0.0, 1.0)

        if tissue is None:
            tissue = _clean_tissue((gap < 0.5) | (distance > 0))
        else:
            tissue = np.asarray(tissue, dtype=bool)
        if len(pts):
            inside = tissue[pts[:, 0], pts[:, 1]]
            pts, scores = pts[inside], scores[inside]

        n_nucleus_seeds = 0
        if self.nucleus_seeds and nuclei is not None and nucleus_radius is not None:
            npts = nucleus_peaks(nuclei, nucleus_radius)
            if len(npts):
                keep = tissue[npts[:, 0], npts[:, 1]] & (distance[npts[:, 0], npts[:, 1]] > 0)
                npts = npts[keep]
            if len(npts):
                if len(pts):
                    d, _ = cKDTree(pts).query(npts)
                    npts = npts[d > 0.6 * cell_radius]
                if len(npts):
                    pts = np.concatenate([pts, npts], axis=0) if len(pts) else npts
                    scores = np.concatenate([scores, np.full(len(npts), self.nucleus_seed_score, dtype=np.float32)])
                    order = np.argsort(-scores, kind="stable")
                    pts, scores = pts[order], scores[order]
                    n_nucleus_seeds = int(len(npts))

        meta = {
            "n_nucleus_seeds": n_nucleus_seeds,
            "cell_radius_px": float(cell_radius),
            "ridge_width_px": float(estimate_ridge_width(geometry)),
            "nuclei_used": nuclei is not None,
            "junction_used": junction is not None,
            "n_seed_candidates": int(len(pts)),
            "checkpoint": str(self.checkpoint_path),
            "tile": self.tile,
            "overlap": self.overlap,
            "device": str(self.device),
        }
        if nucleus_radius is not None:
            meta["nucleus_radius_px"] = float(nucleus_radius)
        return ProposalMaps(
            boundary=boundary,
            seed=seed,
            tissue=tissue,
            gap=gap,
            sigma=sigma,
            seed_points=np.asarray(pts, dtype=np.float64).reshape(-1, 2),
            seed_scores=np.asarray(scores, dtype=np.float32).reshape(-1),
            vertex=vertex,
            distance=distance,
            source="neural",
            meta=meta,
        )


__all__ = ["NeuralProposer", "tiled_predict"]
