"""Exact training targets from a label image and its complex, plus dataset writing.

Targets (all aligned to ``labels.shape``):

- ``boundary``: bool, ``skimage.segmentation.find_boundaries(labels, connectivity=1,
  mode="inner")``, i.e. nonzero pixels with a 4-neighbour of a different label
  (including background).
- ``signed_distance``: float32 Euclidean distance from the pixel centre to the nearest
  crack (label change between 4-adjacent pixels), positive inside cells, negative in
  background, clipped to +/- ``clip_px``. Boundary pixels are at +0.5.
- ``seed``: float32 Gaussian heatmap (peak 1) at nuclei or cell centroids.
- ``vertex``: float32 Gaussian heatmap at vertices with >= 3 incident cells.
- ``gap``: bool, enclosed background (gap faces of the complex).
- ``outer``: bool, background connected to the image border (outer face).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries

from ..complex import HalfEdgeComplex, extract_complex
from .render import RenderParams, junction_intensity_for_snr, render_channels
from .tissue import GAP_KINDS, SynthTissueParams, generate_tissue

CHANNEL_KEYS = ("junction", "membrane", "nuclei", "broken_mask")
TARGET_KEYS = ("boundary", "signed_distance", "seed", "vertex", "gap", "outer")


def crack_distance(labels: np.ndarray) -> np.ndarray:
    """Unsigned distance (px) from each pixel centre to the nearest crack, computed with
    one EDT on the doubled grid where cracks and crack corners are marked."""
    H, W = labels.shape
    G = np.zeros((2 * H + 1, 2 * W + 1), dtype=bool)
    v = labels[:, :-1] != labels[:, 1:]  # (H, W-1) vertical crack right of pixel (r, c)
    h = labels[:-1, :] != labels[1:, :]  # (H-1, W) horizontal crack below pixel (r, c)
    G[1::2, 2:-1:2] = v
    G[2:-1:2, 1::2] = h
    if H > 1 and W > 1:
        G[2:-1:2, 2:-1:2] = v[:-1, :] | v[1:, :] | h[:, :-1] | h[:, 1:]
    if not G.any():
        return np.full((H, W), np.inf, dtype=np.float64)
    d = ndi.distance_transform_edt(~G) / 2.0
    return d[1::2, 1::2]


def gaussian_heatmap(points_xy: np.ndarray, shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """Max-combined unit-peak Gaussians at subpixel (row, col) positions."""
    H, W = shape
    heat = np.zeros((H, W), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    rad = int(np.ceil(3 * sigma))
    for r, c in pts:
        r0, c0 = int(round(r)), int(round(c))
        rs = slice(max(r0 - rad, 0), min(r0 + rad + 1, H))
        cs = slice(max(c0 - rad, 0), min(c0 + rad + 1, W))
        if rs.start >= rs.stop or cs.start >= cs.stop:
            continue
        rr, cc = np.mgrid[rs, cs]
        g = np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma**2)).astype(np.float32)
        heat[rs, cs] = np.maximum(heat[rs, cs], g)
    return heat


def background_faces(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(gap, outer) masks using the complex's rule: 4-connected background components
    touching the border form the outer face, the rest are gaps."""
    bg = cc_label(labels == 0, connectivity=1)
    n_bg = int(bg.max())
    if n_bg == 0:
        z = np.zeros(labels.shape, dtype=bool)
        return z, z.copy()
    touch = np.zeros(n_bg + 1, dtype=bool)
    touch[np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))] = True
    touch[0] = False
    is_outer = touch[bg]
    is_gap = (bg > 0) & ~is_outer
    return is_gap, is_outer


def cell_centroids(labels: np.ndarray) -> np.ndarray:
    """(K, 2) centroids of labels 1..K (rows of NaN for absent labels)."""
    K = int(labels.max())
    if K == 0:
        return np.zeros((0, 2))
    flat = labels.ravel()
    rr, cc = np.mgrid[: labels.shape[0], : labels.shape[1]]
    n = np.bincount(flat, minlength=K + 1)[1:].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.bincount(flat, weights=rr.ravel(), minlength=K + 1)[1:] / n
        c = np.bincount(flat, weights=cc.ravel(), minlength=K + 1)[1:] / n
    out = np.stack([r, c], axis=1)
    return out[np.isfinite(out).all(axis=1)]


def tricellular_vertices(cx: HalfEdgeComplex) -> np.ndarray:
    """(M, 2) positions of vertices with at least three incident cells."""
    idx = [v for v in range(cx.n_vertices) if len(cx.vertex_cell_set(v)) >= 3]
    return cx.vertex_xy[idx].reshape(-1, 2)


def make_targets(
    labels: np.ndarray,
    cx: Optional[HalfEdgeComplex] = None,
    nuclei_xy: Optional[np.ndarray] = None,
    clip_px: float = 16.0,
    seed_sigma_px: float = 3.0,
    vertex_sigma_px: float = 2.0,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got {labels.shape}")
    if cx is None:
        cx = extract_complex(labels)
    boundary = find_boundaries(labels, connectivity=1, mode="inner")
    dist = np.minimum(crack_distance(labels), clip_px)
    signed = np.where(labels > 0, dist, -dist).astype(np.float32)
    seeds = np.asarray(nuclei_xy, dtype=np.float64).reshape(-1, 2) if nuclei_xy is not None else cell_centroids(labels)
    gap, outer = background_faces(labels)
    return {
        "boundary": boundary.astype(bool),
        "signed_distance": signed,
        "seed": gaussian_heatmap(seeds, labels.shape, seed_sigma_px),
        "vertex": gaussian_heatmap(tricellular_vertices(cx), labels.shape, vertex_sigma_px),
        "gap": gap,
        "outer": outer,
    }


# ---------------------------------------------------------------- samplers
def default_params_sampler(rng: np.random.Generator, shape: Tuple[int, int] = (512, 512)) -> SynthTissueParams:
    """Broad tissue prior: cell area 600 to 2500 px, elongation 1 to 3, gap fraction 0 to
    0.08 (30 percent of tiles gap-free), jitter 0 to 2.5 px."""
    H, W = shape
    cell_area = rng.uniform(600.0, 2500.0)
    gap_frac = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.0, 0.08))
    return SynthTissueParams(
        shape=(int(H), int(W)),
        n_cells=max(4, int(round(H * W / cell_area))),
        lloyd_iters=int(rng.integers(2, 5)),
        elongation=float(rng.uniform(1.0, 3.0)),
        flow_angle_deg=float(rng.uniform(0.0, 180.0)),
        gap_area_fraction=gap_frac,
        n_gaps=0,
        gap_kind=str(rng.choice(GAP_KINDS)),
        boundary_jitter_px=float(rng.uniform(0.0, 2.5)),
        jitter_scale_px=float(rng.uniform(8.0, 20.0)),
        binucleate_fraction=float(rng.uniform(0.0, 0.06)),
        anucleate_fraction=float(rng.uniform(0.0, 0.04)),
    )


def default_render_sampler(rng: np.random.Generator) -> RenderParams:
    """Broad optics prior: PSF sigma 0.8 to 2.5 px, junction width 1 to 3 px, SNR 3 to 30
    (log-uniform, via junction_intensity), broken fraction 0 to 0.4."""
    background = float(rng.uniform(2.0, 20.0))
    interior = float(rng.uniform(2.0, 25.0))
    read_noise = float(rng.uniform(1.0, 4.0))
    snr = float(np.exp(rng.uniform(np.log(3.0), np.log(30.0))))
    return RenderParams(
        psf_sigma_px=float(rng.uniform(0.8, 2.5)),
        junction_width_px=float(rng.uniform(1.0, 3.0)),
        junction_intensity=junction_intensity_for_snr(snr, background, interior, read_noise),
        interior_intensity=interior,
        background=background,
        gain=float(rng.uniform(0.8, 1.5)),
        read_noise_sigma=read_noise,
        photobleach_gradient=float(rng.uniform(0.0, 0.5)),
        broken_fraction=float(rng.uniform(0.0, 0.4)),
        broken_segment_len_px=float(rng.uniform(4.0, 14.0)),
        membrane_width_px=float(rng.uniform(0.8, 1.8)),
        membrane_intensity=float(rng.uniform(60.0, 250.0)),
        nucleus_radius_px=float(rng.uniform(3.5, 7.0)),
        nucleus_intensity=float(rng.uniform(60.0, 300.0)),
    )


# ------------------------------------------------------------------ dataset
def _flatten(prefix: str, obj: Any) -> Dict[str, Any]:
    out = {}
    for k, v in asdict(obj).items():
        key = f"{prefix}_{k}" if k in ("seed", "shape") else k
        out[key] = str(tuple(v)) if isinstance(v, (tuple, list)) else v
    return out


def make_dataset(
    n: int,
    out_dir: Union[str, Path],
    params_sampler: Optional[Callable[[np.random.Generator], SynthTissueParams]] = None,
    render_sampler: Optional[Callable[[np.random.Generator], RenderParams]] = None,
    seed: int = 0,
    target_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Write ``n`` tiles ``tile_XXXXX.npz`` (channels, targets, labels, nuclei_xy) and a
    ``manifest.csv`` to ``out_dir``. Per-tile tissue and render seeds are drawn from
    ``seed`` and override whatever the samplers set, so the run is reproducible."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    params_sampler = params_sampler or default_params_sampler
    render_sampler = render_sampler or default_render_sampler
    rows = []
    for i in range(int(n)):
        tp = replace(params_sampler(rng), seed=int(rng.integers(0, 2**31 - 1)))
        rp = replace(render_sampler(rng), seed=int(rng.integers(0, 2**31 - 1)))
        tissue = generate_tissue(tp)
        cx = extract_complex(tissue.labels)
        channels = render_channels(tissue, rp, cx=cx)
        targets = make_targets(tissue.labels, cx=cx, nuclei_xy=tissue.nuclei_xy, **(target_kwargs or {}))
        fname = f"tile_{i:05d}.npz"
        np.savez_compressed(
            out_dir / fname,
            **{k: channels[k] for k in CHANNEL_KEYS},
            **targets,
            labels=tissue.labels,
            nuclei_xy=tissue.nuclei_xy,
        )
        rp_used: RenderParams = channels["params"]
        row: Dict[str, Any] = {"file": fname, "index": i}
        row.update(_flatten("tissue", tp))
        row.update(_flatten("render", rp_used))
        row.update(
            {
                "n_cells_actual": tissue.n_cells,
                "n_gaps_actual": int(cx.gap_faces.size),
                "n_vertices": cx.n_vertices,
                "n_nuclei": int(tissue.nuclei_xy.shape[0]),
                "snr": float(
                    rp_used.junction_intensity
                    / np.sqrt(
                        rp_used.junction_intensity
                        + rp_used.background
                        + rp_used.interior_intensity
                        + rp_used.read_noise_sigma**2
                    )
                ),
            }
        )
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def load_tile(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


__all__: Sequence[str] = (
    "CHANNEL_KEYS",
    "TARGET_KEYS",
    "background_faces",
    "cell_centroids",
    "crack_distance",
    "default_params_sampler",
    "default_render_sampler",
    "gaussian_heatmap",
    "load_tile",
    "make_dataset",
    "make_targets",
    "tricellular_vertices",
)
