"""Forward model from a synthetic tissue to noisy fluorescence channels.

Channels (all float32, photon-count units, (H, W)):

- ``junction``: VE-cadherin-like signal rasterized along cell-cell edges of the exact
  complex with a given line width, randomly broken segments, weak interior haze.
- ``membrane``: independent pan-membrane marker on ALL face boundaries (cell-cell,
  cell-gap, cell-background), continuous, thinner.
- ``nuclei``: Gaussian blobs at the tissue's nuclei.

Every expected image is blurred with a Gaussian PSF, multiplied by a smooth
flat-field/bleaching ramp, Poisson sampled, scaled by ``gain`` and given Gaussian
read noise. ``broken_mask`` marks where junction signal was deliberately removed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

from ..complex import HalfEdgeComplex, extract_complex
from ..complex.geometry import smooth_complex
from .tissue import SynthTissue, smooth_random_field

Range = Union[float, Tuple[float, float]]


@dataclass
class RenderParams:
    """Optical and labelling parameters. ``psf_sigma_px`` and ``junction_width_px`` may
    be a scalar or a ``(lo, hi)`` range drawn uniformly per render from ``seed``.
    Widths are full band widths in pixels; intensities are expected photons per pixel
    at the centre of an unblurred structure."""

    psf_sigma_px: Range = 1.2
    junction_width_px: Range = 2.0
    junction_intensity: float = 200.0
    interior_intensity: float = 10.0
    background: float = 5.0
    gain: float = 1.0
    read_noise_sigma: float = 2.0
    photobleach_gradient: float = 0.3  # relative signal drop across the field along a random axis
    broken_fraction: float = 0.0  # fraction of cell-cell arclength with junction signal removed
    broken_segment_len_px: float = 8.0
    membrane_width_px: float = 1.2
    membrane_intensity: float = 120.0
    nucleus_radius_px: float = 5.0
    nucleus_intensity: float = 150.0
    sample_step_px: float = 0.25  # arclength spacing of polyline samples used for rasterization
    seed: int = 0


def _draw(value: Range, rng: np.random.Generator) -> float:
    if np.ndim(value) == 0:
        return float(value)
    lo, hi = value
    return float(rng.uniform(lo, hi))


def resolve_render_params(rp: RenderParams, rng: np.random.Generator) -> RenderParams:
    """Replace range-valued fields by concrete draws."""
    return replace(rp, psf_sigma_px=_draw(rp.psf_sigma_px, rng), junction_width_px=_draw(rp.junction_width_px, rng))


def junction_intensity_for_snr(
    snr: float, background: float = 5.0, interior_intensity: float = 10.0, read_noise_sigma: float = 2.0
) -> float:
    """Peak junction photons J such that J / sqrt(J + B + sigma_r^2) == snr, with
    B = background + interior haze."""
    b = float(background + interior_intensity + read_noise_sigma**2)
    s2 = float(snr) ** 2
    return 0.5 * (s2 + np.sqrt(s2**2 + 4 * s2 * b))


# --------------------------------------------------------------- rasterizing
def _sample_polyline(p: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` px of arclength along polyline ``p`` (endpoints included)."""
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))
    s = np.r_[0.0, np.cumsum(seg)]
    total = float(s[-1])
    if total <= 0:
        return p[:1].copy()
    n = max(int(np.ceil(total / step)), 1)
    t = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def _inside_pixels(pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Drop points lying on the image frame itself (outer-face edges along the border)."""
    H, W = shape
    eps = 1e-6
    rows_ok = (pts[:, 0] > -0.5 + eps) & (pts[:, 0] < H - 0.5 - eps)
    cols_ok = (pts[:, 1] > -0.5 + eps) & (pts[:, 1] < W - 0.5 - eps)
    return rows_ok & cols_ok


def point_distance_map(
    pts: np.ndarray,
    shape: Tuple[int, int],
    max_dist: float,
    flags: Optional[np.ndarray] = None,
    candidates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Distance (px) from each pixel centre to the nearest point (inf beyond ``max_dist``).
    With ``flags`` (bool per point) also return the flag of the nearest point (False
    where no point lies within ``max_dist``). ``candidates`` restricts the query to a
    pixel mask known to contain every pixel within ``max_dist`` of the points."""
    H, W = shape
    dist = np.full((H, W), np.inf)
    near = None if flags is None else np.zeros((H, W), dtype=bool)
    if pts.shape[0] == 0:
        return dist, near
    if candidates is None:
        rr, cc = np.mgrid[:H, :W]
        rr, cc = rr.ravel(), cc.ravel()
    else:
        rr, cc = np.nonzero(candidates)
    q = np.stack([rr, cc], axis=1).astype(np.float64)
    d, idx = cKDTree(pts).query(q, k=1, distance_upper_bound=max_dist)
    dist[rr, cc] = d
    if flags is not None:
        valid = idx < pts.shape[0]
        near[rr[valid], cc[valid]] = flags[idx[valid]]
    return dist, near


def _near_boundary(labels: np.ndarray, radius_px: float) -> np.ndarray:
    """Pixels within ``radius_px`` (plus slack for trace smoothing) of any crack."""
    thick = find_boundaries(labels, connectivity=1, mode="thick")
    return ndi.binary_dilation(thick, iterations=int(np.ceil(radius_px)) + 2)


def band_coverage(dist: np.ndarray, width_px: float) -> np.ndarray:
    """Antialiased band of full width ``width_px`` around distance-0: 1 inside, linear
    ramp over one pixel at the rim."""
    return np.clip(width_px / 2.0 + 0.5 - dist, 0.0, 1.0)


# ------------------------------------------------------------------ breaking
def _break_edges(
    samples: List[np.ndarray], rng: np.random.Generator, fraction: float, seg_len: float, step: float
) -> List[np.ndarray]:
    """Mark contiguous runs of samples as broken until ``fraction`` of total arclength is
    covered. Runs are placed on edges chosen proportionally to their length."""
    flags = [np.zeros(s.shape[0], dtype=bool) for s in samples]
    counts = np.array([s.shape[0] for s in samples], dtype=np.float64)
    total = counts.sum()
    if fraction <= 0 or total == 0:
        return flags
    target = fraction * total
    prob = counts / total
    broken = 0.0
    for _ in range(int(20 * total / max(seg_len / step, 1.0)) + 100):
        if broken >= target:
            break
        e = int(rng.choice(len(samples), p=prob))
        n = int(counts[e])
        run = max(int(round(seg_len * rng.uniform(0.5, 1.5) / step)), 1)
        start = int(rng.integers(-run // 2, n))  # may start before the edge so breaks can touch vertices
        lo, hi = max(start, 0), min(start + run, n)
        if hi <= lo:
            continue
        before = flags[e][lo:hi].sum()
        flags[e][lo:hi] = True
        broken += (hi - lo) - before
    return flags


# ----------------------------------------------------------------- photonics
def _low_frequency_field(rng: np.random.Generator, shape: Tuple[int, int], knots: int = 6) -> np.ndarray:
    """Unit-variance smooth field from bilinear upsampling of a coarse Gaussian knot grid."""
    H, W = shape
    coarse = ndi.gaussian_filter(rng.standard_normal((knots + 2, knots + 2)), sigma=0.8, mode="reflect")
    rr, cc = np.mgrid[:H, :W]
    coords = [rr * (knots + 1) / max(H - 1, 1), cc * (knots + 1) / max(W - 1, 1)]
    f = ndi.map_coordinates(coarse, coords, order=1, mode="nearest")
    s = f.std()
    return f / s if s > 0 else f


def _flat_field(rng: np.random.Generator, shape: Tuple[int, int], gradient: float) -> np.ndarray:
    """Multiplicative ramp 1 -> 1 - gradient along a random direction, times a gentle
    low-frequency illumination texture."""
    H, W = shape
    th = rng.uniform(0, 2 * np.pi)
    rr, cc = np.mgrid[:H, :W]
    proj = (rr - (H - 1) / 2) * np.sin(th) + (cc - (W - 1) / 2) * np.cos(th)
    t = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-9)
    ramp = 1.0 - gradient * t
    texture = 1.0 + 0.05 * _low_frequency_field(rng, shape)
    return np.clip(ramp * texture, 0.05, None)


def _acquire(expected: np.ndarray, rp: RenderParams, rng: np.random.Generator, flat: np.ndarray) -> np.ndarray:
    blurred = ndi.gaussian_filter(expected, sigma=float(rp.psf_sigma_px), mode="reflect") * flat
    photons = rng.poisson(np.clip(blurred, 0, None)).astype(np.float64) * rp.gain
    photons += rng.normal(0.0, rp.read_noise_sigma, size=expected.shape)
    return np.clip(photons, 0.0, None).astype(np.float32)


def _interior_haze(labels: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    if intensity <= 0:
        return np.zeros(labels.shape, dtype=np.float64)
    texture = 1.0 + 0.3 * smooth_random_field(rng, labels.shape, scale_px=4.0)
    return intensity * np.clip(texture, 0.2, None) * (labels > 0)


def _nuclei_expected(nuclei_xy: np.ndarray, shape: Tuple[int, int], rp: RenderParams, rng: np.random.Generator):
    H, W = shape
    out = np.zeros(shape, dtype=np.float64)
    sigma = max(rp.nucleus_radius_px / 2.0, 0.5)
    rad = int(np.ceil(3 * sigma))
    for r, c in nuclei_xy:
        amp = rp.nucleus_intensity * rng.uniform(0.7, 1.3)
        r0, c0 = int(round(r)), int(round(c))
        rs = slice(max(r0 - rad, 0), min(r0 + rad + 1, H))
        cs = slice(max(c0 - rad, 0), min(c0 + rad + 1, W))
        if rs.start >= rs.stop or cs.start >= cs.stop:
            continue
        rr, cc = np.mgrid[rs, cs]
        out[rs, cs] += amp * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma**2))
    return out


# ---------------------------------------------------------------------- main
def render_channels(
    tissue: SynthTissue, rp: RenderParams, cx: Optional[HalfEdgeComplex] = None
) -> Dict[str, Any]:
    """Render ``junction``, ``membrane`` and ``nuclei`` channels plus ``broken_mask``.

    Returns a dict with those four arrays and ``params`` (the resolved RenderParams).
    """
    rng = np.random.default_rng(rp.seed)
    rp = resolve_render_params(rp, rng)
    labels = tissue.labels
    shape = labels.shape
    step = float(rp.sample_step_px)
    if cx is None:
        cx = extract_complex(labels)
    if cx.edge_smooth is None:
        smooth_complex(cx)

    all_samples = [_sample_polyline(cx.edge_smooth[e], step) for e in range(cx.n_edges)]
    band_radius = float(rp.junction_width_px) / 2.0 + 0.5
    mem_radius = float(rp.membrane_width_px) / 2.0 + 1.0
    candidates = _near_boundary(labels, max(band_radius, mem_radius) + 0.5)

    # Junction: cell-cell edges only, with broken runs.
    cc_edges = [int(e) for e in cx.cell_cell_edges()]
    samples = [all_samples[e] for e in cc_edges]
    flags = _break_edges(samples, rng, float(rp.broken_fraction), float(rp.broken_segment_len_px), step)
    if samples:
        pts = np.concatenate(samples, axis=0)
        brk = np.concatenate(flags, axis=0)
        keep = _inside_pixels(pts, shape)
        pts, brk = pts[keep], brk[keep]
    else:
        pts, brk = np.zeros((0, 2)), np.zeros(0, dtype=bool)
    dist, near_broken = point_distance_map(pts, shape, band_radius + 0.5, flags=brk, candidates=candidates)
    broken_mask = (dist <= band_radius) & near_broken
    junction_band = band_coverage(dist, float(rp.junction_width_px)) * ~near_broken
    junction_expected = (
        rp.junction_intensity * junction_band + _interior_haze(labels, rp.interior_intensity, rng) + rp.background
    )

    # Membrane: every edge of the complex, continuous.
    if all_samples:
        all_pts = np.concatenate(all_samples, axis=0)
        all_pts = all_pts[_inside_pixels(all_pts, shape)]
    else:
        all_pts = np.zeros((0, 2))
    mem_dist, _ = point_distance_map(all_pts, shape, mem_radius, candidates=candidates)
    membrane_expected = (
        rp.membrane_intensity * band_coverage(mem_dist, float(rp.membrane_width_px))
        + _interior_haze(labels, 0.5 * rp.interior_intensity, rng)
        + rp.background
    )

    nuclei_expected = _nuclei_expected(tissue.nuclei_xy, shape, rp, rng) + rp.background

    out: Dict[str, Any] = {}
    for key, expected in (
        ("junction", junction_expected),
        ("membrane", membrane_expected),
        ("nuclei", nuclei_expected),
    ):
        flat = _flat_field(rng, shape, float(rp.photobleach_gradient))
        out[key] = _acquire(expected, rp, rng, flat)
    out["broken_mask"] = broken_mask.astype(bool)
    out["params"] = rp
    return out
