"""Synthetic endothelial sheets with exact labels, nuclei and gaps.

Pipeline: Poisson-disk seeds, Lloyd-relaxed Voronoi tessellation in an anisotropic
"flow frame" (cells come out elongated along the flow axis), smooth random boundary
jitter, 4-connectivity repair, gap carving at trivalent vertices or along cell-cell
edges, and nucleus placement. Coordinates are image (row, col).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label

from ..complex import FaceKind, HalfEdgeComplex, extract_complex
from ..complex.geometry import polyline_length, smooth_polyline

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
GAP_KINDS = ("tricellular", "bicellular", "mixed")


@dataclass
class SynthTissueParams:
    """Generation parameters. All randomness derives from ``seed``.

    ``elongation`` is the target cell aspect ratio along the flow axis (1 = isotropic).
    ``flow_angle_deg`` is measured from the column axis toward the row axis.
    ``gap_area_fraction`` is the target fraction of image area carved into gaps; when it
    is positive and ``n_gaps`` is 0 the count is derived from the mean gap radius. When
    both are positive the per-gap size follows from the area target (radius clipped to
    ``[gap_radius_px[0], 3 * gap_radius_px[1]]``). With ``gap_area_fraction == 0`` the
    radius is drawn uniformly from ``gap_radius_px``.
    """

    shape: Tuple[int, int] = (512, 512)
    n_cells: int = 200
    lloyd_iters: int = 3
    elongation: float = 1.0
    flow_angle_deg: float = 0.0
    gap_area_fraction: float = 0.0
    n_gaps: int = 0
    gap_kind: str = "tricellular"
    gap_radius_px: Tuple[float, float] = (2.0, 5.0)
    boundary_jitter_px: float = 0.0
    jitter_scale_px: float = 12.0
    nucleus_jitter: float = 0.15  # nucleus offset sigma as a fraction of sqrt(cell area)
    binucleate_fraction: float = 0.0
    anucleate_fraction: float = 0.0
    seed: int = 0


@dataclass
class SynthTissue:
    """``gap_xy`` and ``gap_kinds`` are in carve order, which differs from the face order
    of ``extract_complex(labels).gap_faces``; match them through the pixel at ``gap_xy``."""

    labels: np.ndarray  # (H, W) int32, 0 = background (gaps)
    nuclei_xy: np.ndarray  # (N, 2) float64 (row, col)
    nuclei_cell: np.ndarray  # (N,) int64 label of the cell containing each nucleus
    params: SynthTissueParams
    gap_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (G, 2) carve centers
    gap_kinds: List[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return int(self.labels.max())

    @property
    def n_gaps(self) -> int:
        return len(self.gap_kinds)


# ------------------------------------------------------------------ flow frame
class _FlowFrame:
    """Affine map to an isotropic frame: rotate so the flow axis is the first
    coordinate, then divide it by ``elongation``."""

    def __init__(self, shape: Tuple[int, int], elongation: float, angle_deg: float):
        th = np.deg2rad(angle_deg)
        self.d = np.array([np.sin(th), np.cos(th)])  # along flow (row, col)
        self.n = np.array([np.cos(th), -np.sin(th)])  # perpendicular
        self.center = np.array([(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0])
        self.e = float(elongation)

    def to_iso(self, rc: np.ndarray) -> np.ndarray:
        q = np.asarray(rc, dtype=np.float64) - self.center
        u = (q @ self.d) / self.e
        v = q @ self.n
        return np.stack([u, v], axis=1)


def _pixel_iso(frame: _FlowFrame, shape: Tuple[int, int]) -> np.ndarray:
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    return frame.to_iso(np.stack([rr.ravel(), cc.ravel()], axis=1))


def _sample_seeds(rng: np.random.Generator, frame: _FlowFrame, shape: Tuple[int, int], n: int) -> np.ndarray:
    """Poisson-disk-like seeds in the iso frame by best-of-k rejection."""
    H, W = shape
    attempts = 30
    area_iso = H * W / frame.e
    r_min = 0.7 * np.sqrt(area_iso / max(n, 1))
    seeds = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        cand = frame.to_iso(
            np.stack([rng.uniform(-0.5, H - 0.5, attempts), rng.uniform(-0.5, W - 0.5, attempts)], axis=1)
        )
        if i == 0:
            seeds[0] = cand[0]
            continue
        d = np.sqrt(((cand[:, None, :] - seeds[None, :i, :]) ** 2).sum(-1)).min(axis=1)
        good = np.flatnonzero(d >= r_min)
        seeds[i] = cand[good[0]] if good.size else cand[int(np.argmax(d))]
    return seeds


def _lloyd_voronoi(seeds: np.ndarray, pix_iso: np.ndarray, shape: Tuple[int, int], iters: int) -> np.ndarray:
    """Pixel Voronoi assignment in the iso frame with ``iters`` centroid updates."""
    seeds = seeds.copy()
    n = seeds.shape[0]
    idx = np.zeros(pix_iso.shape[0], dtype=np.int64)
    for it in range(iters + 1):
        _, idx = cKDTree(seeds).query(pix_iso, k=1)
        if it == iters:
            break
        counts = np.bincount(idx, minlength=n)
        su = np.bincount(idx, weights=pix_iso[:, 0], minlength=n)
        sv = np.bincount(idx, weights=pix_iso[:, 1], minlength=n)
        nz = counts > 0
        seeds[nz, 0] = su[nz] / counts[nz]
        seeds[nz, 1] = sv[nz] / counts[nz]
    return (idx.reshape(shape) + 1).astype(np.int32)


# -------------------------------------------------------------------- jitter
def smooth_random_field(rng: np.random.Generator, shape: Tuple[int, int], scale_px: float) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian random field with correlation length ``scale_px``."""
    f = ndi.gaussian_filter(rng.standard_normal(shape), sigma=scale_px, mode="reflect")
    s = f.std()
    return f / s if s > 0 else f


def _jitter_boundaries(labels: np.ndarray, rng: np.random.Generator, amp_px: float, scale_px: float) -> np.ndarray:
    H, W = labels.shape
    dr = amp_px * smooth_random_field(rng, (H, W), scale_px)
    dc = amp_px * smooth_random_field(rng, (H, W), scale_px)
    rr, cc = np.mgrid[:H, :W]
    return ndi.map_coordinates(labels, [rr + dr, cc + dc], order=0, mode="nearest").astype(np.int32)


def enforce_4_connected(labels: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Reassign every non-largest 4-connected fragment of each label to the majority
    label among its outside 4-neighbours. Terminates because each pass strictly
    reduces the component count."""
    lab = labels.copy()
    for _ in range(max_iter):
        comp = cc_label(lab, connectivity=1, background=0)
        n_comp = int(comp.max())
        if n_comp == 0:
            return lab
        flat_c, flat_l = comp.ravel(), lab.ravel()
        comp_label = np.zeros(n_comp + 1, dtype=np.int64)
        comp_label[flat_c] = flat_l
        comp_size = np.bincount(flat_c, minlength=n_comp + 1)
        ids = np.arange(1, n_comp + 1)
        order = np.lexsort((comp_size[1:], comp_label[1:]))
        sorted_labels = comp_label[1:][order]
        last = np.r_[sorted_labels[1:] != sorted_labels[:-1], True]
        keep = np.zeros(n_comp + 1, dtype=bool)
        keep[ids[order][last]] = True
        fragments = np.flatnonzero(~keep[1:]) + 1
        if fragments.size == 0:
            return lab
        objects = ndi.find_objects(comp)
        for cid in fragments:
            sl = objects[cid - 1]
            sl = (slice(max(sl[0].start - 1, 0), sl[0].stop + 1), slice(max(sl[1].start - 1, 0), sl[1].stop + 1))
            sub_c = comp[sl]
            sub_l = lab[sl]
            m = sub_c == cid
            ring = ndi.binary_dilation(m, _FOUR) & ~m
            nb = sub_l[ring]
            nb = nb[nb > 0]
            if nb.size == 0:
                sub_l[m] = 0
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            sub_l[m] = vals[int(np.argmax(cnt))]
    return lab


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    vals, inv = np.unique(labels, return_inverse=True)
    out = inv.reshape(labels.shape).astype(np.int32)
    if vals[0] != 0:
        out += 1
    return out


# ---------------------------------------------------------------------- gaps
def _resolve_gap_plan(params: SynthTissueParams) -> Tuple[int, Optional[float]]:
    """Return (n_gaps, target area per gap or None)."""
    H, W = params.shape
    lo, hi = params.gap_radius_px
    frac = float(params.gap_area_fraction)
    if frac <= 0:
        return int(params.n_gaps), None
    total = frac * H * W
    n = int(params.n_gaps)
    if n <= 0:
        mean_area = np.pi * (0.5 * (lo + hi)) ** 2
        n = max(1, int(np.ceil(total / mean_area)))
    return n, total / n


_Window = Tuple[slice, slice]


def _window(shape: Tuple[int, int], lo: np.ndarray, hi: np.ndarray) -> Optional[_Window]:
    """Window [lo, hi] padded by one pixel for the neighbour ring; None if the carve
    region would reach the 1-px border rim (the gap must stay enclosed)."""
    H, W = shape
    if lo[0] < 1 or lo[1] < 1 or hi[0] > H - 2 or hi[1] > W - 2:
        return None
    return (slice(int(lo[0]) - 1, int(hi[0]) + 2), slice(int(lo[1]) - 1, int(hi[1]) + 2))


def _disk_mask(shape: Tuple[int, int], center: np.ndarray, radius: float) -> Optional[Tuple[_Window, np.ndarray]]:
    """Window and local boolean disk of pixels within ``radius`` of ``center``."""
    lo = np.floor(np.asarray(center) - radius).astype(int)
    hi = np.ceil(np.asarray(center) + radius).astype(int)
    win = _window(shape, lo, hi)
    if win is None:
        return None
    rr, cc = np.mgrid[win[0], win[1]]
    return win, (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius**2


def _capsule_mask(shape: Tuple[int, int], pts: np.ndarray, half_width: float) -> Optional[Tuple[_Window, np.ndarray]]:
    """Window and local mask of pixels within ``half_width`` of a dense point set."""
    lo = np.floor(pts.min(axis=0) - half_width).astype(int)
    hi = np.ceil(pts.max(axis=0) + half_width).astype(int)
    win = _window(shape, lo, hi)
    if win is None:
        return None
    rr, cc = np.mgrid[win[0], win[1]]
    d, _ = cKDTree(pts).query(np.stack([rr.ravel(), cc.ravel()], axis=1), k=1)
    return win, (d <= half_width).reshape(rr.shape)


def _resample_polyline(p: np.ndarray, s_from: float, s_to: float, step: float = 0.25) -> np.ndarray:
    """Points of polyline ``p`` at arclengths in [s_from, s_to]."""
    seg = np.sqrt((np.diff(p, axis=0) ** 2).sum(axis=1))
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(s_from, s_to + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def _try_carve(
    labels: np.ndarray,
    win: _Window,
    mask: np.ndarray,
    min_neighbors: int,
    max_neighbors: int,
    bboxes: Sequence,
) -> bool:
    """Carve the local ``mask`` (inside window ``win``) to background in place.

    Accepted only if the mask is one 4-connected region of cell pixels, its 4-neighbour
    ring contains no background (so the new gap is a separate enclosed face; the window
    check already keeps it off the border), the ring holds an admissible number of
    distinct cells, and every affected cell stays non-empty and 4-connected.
    Restores ``labels`` on failure.
    """
    sub = labels[win]
    inside = sub[mask]
    if inside.size == 0 or (inside == 0).any():
        return False
    if int(cc_label(mask, connectivity=1).max()) != 1:
        return False
    ring = ndi.binary_dilation(mask, _FOUR) & ~mask
    ring_labels = sub[ring]
    if (ring_labels == 0).any():
        return False
    n_nb = np.unique(ring_labels).size
    if n_nb < min_neighbors or n_nb > max_neighbors:
        return False
    affected = np.unique(inside)
    sub[mask] = 0
    ok = True
    for lab in affected:
        sl = bboxes[int(lab) - 1]
        if sl is None or int(cc_label(labels[sl] == lab, connectivity=1).max()) != 1:
            ok = False
            break
    if not ok:
        sub[mask] = inside
    return ok


def _carve_gaps(
    labels: np.ndarray, params: SynthTissueParams, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n_gaps, area_per_gap = _resolve_gap_plan(params)
    if n_gaps <= 0:
        return labels, np.zeros((0, 2)), []
    shape = labels.shape
    lo, hi = params.gap_radius_px
    cx: HalfEdgeComplex = extract_complex(labels)

    tri_candidates = [
        v
        for v in range(cx.n_vertices)
        if len(cx.vertex_cell_set(v)) >= 3 and all(cx.face_kind[f] != FaceKind.OUTER for f in cx.vertex_faces(v))
    ]
    rng.shuffle(tri_candidates)
    bi_candidates = []
    for e in cx.cell_cell_edges():
        p = smooth_polyline(cx.edge_polyline[int(e)], iterations=4)
        if polyline_length(p) >= 8.0:
            bi_candidates.append(p)
    rng.shuffle(bi_candidates)

    if params.gap_kind == "mixed":
        kinds = [("tricellular", "bicellular")[k] for k in rng.integers(0, 2, n_gaps)]
    else:
        kinds = [params.gap_kind] * n_gaps

    # Bounding boxes only shrink under carving, so they stay valid for connectivity checks.
    bboxes = ndi.find_objects(labels)
    centers: List[np.ndarray] = []
    placed_kinds: List[str] = []
    labels = labels.copy()
    for kind in kinds:
        placed = False
        if kind == "tricellular":
            while tri_candidates and not placed:
                v = tri_candidates.pop()
                if area_per_gap is not None:
                    radius = float(np.clip(np.sqrt(area_per_gap / np.pi) * rng.uniform(0.85, 1.15), lo, 3 * hi))
                else:
                    radius = float(rng.uniform(lo, hi))
                got = _disk_mask(shape, cx.vertex_xy[v], radius)
                if got is None:
                    continue
                if _try_carve(labels, got[0], got[1], 3, 10**6, bboxes):
                    placed = True
                    centers.append(cx.vertex_xy[v].copy())
        else:
            while bi_candidates and not placed:
                p = bi_candidates.pop()
                total = polyline_length(p)
                hw = float(rng.uniform(1.0, 2.0))
                if area_per_gap is not None:
                    length = max(3.0, (area_per_gap - np.pi * hw**2) / (2 * hw))
                else:
                    length = float(rng.uniform(4.0, 10.0))
                margin = hw + 2.0
                length = min(length, total - 2 * margin)
                if length < 3.0:
                    continue
                s_lo, s_hi = margin + length / 2, total - margin - length / 2
                mid = float(rng.uniform(s_lo, s_hi)) if s_hi > s_lo else 0.5 * (s_lo + s_hi)
                pts = _resample_polyline(p, mid - length / 2, mid + length / 2)
                got = _capsule_mask(shape, pts, hw)
                if got is None:
                    continue
                if _try_carve(labels, got[0], got[1], 2, 2, bboxes):
                    placed = True
                    centers.append(pts[len(pts) // 2].copy())
        if placed:
            placed_kinds.append(kind)
    gap_xy = np.array(centers, dtype=np.float64).reshape(-1, 2)
    return labels, gap_xy, placed_kinds


# -------------------------------------------------------------------- nuclei
def _place_nuclei(
    labels: np.ndarray, rng: np.random.Generator, params: SynthTissueParams
) -> Tuple[np.ndarray, np.ndarray]:
    K = int(labels.max())
    if K == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=K + 1)[1:]
    rr, cc = np.mgrid[: labels.shape[0], : labels.shape[1]]
    safe = np.maximum(areas, 1)
    com = np.stack(
        [
            np.bincount(flat, weights=rr.ravel(), minlength=K + 1)[1:] / safe,
            np.bincount(flat, weights=cc.ravel(), minlength=K + 1)[1:] / safe,
        ],
        axis=1,
    )
    objects = ndi.find_objects(labels)
    u = rng.random(K)
    anucleate = u < params.anucleate_fraction
    binucleate = (~anucleate) & (u < params.anucleate_fraction + params.binucleate_fraction)
    pts: List[np.ndarray] = []
    owner: List[int] = []
    for k in range(K):
        if anucleate[k]:
            continue
        sl = objects[k]
        rs, cs = np.nonzero(labels[sl] == k + 1)
        cell_px = np.stack([rs + sl[0].start, cs + sl[1].start], axis=1).astype(np.float64)
        sigma = params.nucleus_jitter * np.sqrt(areas[k])
        min_sep = min(4.0, 0.3 * np.sqrt(areas[k]))
        placed: List[np.ndarray] = []
        n_want = 2 if binucleate[k] else 1
        for _ in range(n_want):
            best = None
            for _attempt in range(10):
                cand = np.asarray(com[k]) + rng.normal(0.0, sigma, 2)
                r0, c0 = int(round(cand[0])), int(round(cand[1]))
                if not (0 <= r0 < labels.shape[0] and 0 <= c0 < labels.shape[1] and labels[r0, c0] == k + 1):
                    cand = cell_px[int(np.argmin(((cell_px - cand) ** 2).sum(axis=1)))]
                best = cand
                if all(np.linalg.norm(cand - q) >= min_sep for q in placed):
                    break
            placed.append(best)
        for q in placed:
            pts.append(q)
            owner.append(k + 1)
    return np.array(pts, dtype=np.float64).reshape(-1, 2), np.array(owner, dtype=np.int64)


# ---------------------------------------------------------------------- main
def _check_params(p: SynthTissueParams) -> None:
    if len(p.shape) != 2 or p.shape[0] < 8 or p.shape[1] < 8:
        raise ValueError(f"shape must be 2-D and at least 8x8, got {p.shape}")
    if p.n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if p.elongation < 1.0:
        raise ValueError("elongation must be >= 1 (aspect ratio along the flow axis)")
    if p.gap_kind not in GAP_KINDS:
        raise ValueError(f"gap_kind must be one of {GAP_KINDS}, got {p.gap_kind!r}")
    if not (0.0 <= p.gap_area_fraction <= 0.5):
        raise ValueError("gap_area_fraction must be in [0, 0.5]")
    if p.binucleate_fraction + p.anucleate_fraction > 1.0:
        raise ValueError("binucleate_fraction + anucleate_fraction must be <= 1")


def generate_tissue(params: SynthTissueParams) -> SynthTissue:
    """Generate a synthetic tissue. Every nonzero label is a single 4-connected region
    and background occurs only inside carved gaps."""
    _check_params(params)
    rng = np.random.default_rng(params.seed)
    shape = (int(params.shape[0]), int(params.shape[1]))
    frame = _FlowFrame(shape, params.elongation, params.flow_angle_deg)
    pix_iso = _pixel_iso(frame, shape)
    seeds = _sample_seeds(rng, frame, shape, int(params.n_cells))
    labels = _lloyd_voronoi(seeds, pix_iso, shape, int(params.lloyd_iters))
    if params.boundary_jitter_px > 0:
        labels = _jitter_boundaries(labels, rng, float(params.boundary_jitter_px), float(params.jitter_scale_px))
    labels = enforce_4_connected(labels)
    labels = relabel_consecutive(labels)
    labels, gap_xy, gap_kinds = _carve_gaps(labels, params, rng)
    nuclei_xy, nuclei_cell = _place_nuclei(labels, rng, params)
    return SynthTissue(
        labels=labels.astype(np.int32),
        nuclei_xy=nuclei_xy,
        nuclei_cell=nuclei_cell,
        params=params,
        gap_xy=gap_xy,
        gap_kinds=gap_kinds,
    )
