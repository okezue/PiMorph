"""Oriented strips around interface curves.

For an edge e with curve gamma_e(s) (arclength s) and left unit normal n_e(s) the
strip is P_e(s, r, c) = I_c(gamma_e(s) + r * n_e(s)). Sampling is done with
``scipy.ndimage.map_coordinates`` in image (row, col) coordinates, pixel (r, c)
centered at (r, c).

Sign convention for the lateral offset r: positive r is the LEFT side of the
forward half-edge (tail -> head), i.e. it points into ``edge_faces[e, 0]``;
negative r is the right side (``edge_faces[e, 1]``). This matches
``pimorph.complex.geometry.polyline_normals``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from ..complex.halfedge import HalfEdgeComplex


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def resample_polyline(
    p: np.ndarray,
    step_px: float = 1.0,
    endpoints: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a polyline at uniform arclength spacing by linear interpolation.

    With ``endpoints=False`` (default) the S = max(1, round(L / step_px)) samples sit
    at the midpoints of equal arclength cells, so the two vertices, which are shared
    with edges of other faces, are never sampled. With ``endpoints=True`` the samples
    run from the first to the last point inclusive (S >= 2).

    Returns ``(points (S, 2), tangents (S, 2), normals (S, 2), s (S,))``. Tangents are
    unit vectors in the travel direction (central differences of the resampled
    points); normals are unit vectors to the left of travel, (-t_col, t_row).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    if step_px <= 0:
        raise ValueError("step_px must be positive")
    if p.shape[0] == 0:
        z = np.zeros((0, 2))
        return z, z.copy(), z.copy(), np.zeros(0)
    seg = np.diff(p, axis=0)
    seg_len = np.sqrt((seg**2).sum(axis=1)) if seg.size else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(cum[-1])

    if L <= 0 or p.shape[0] < 2:
        pts = np.repeat(p[:1], 2 if endpoints else 1, axis=0)
        t = np.zeros_like(pts)
        return pts, t, t.copy(), np.zeros(pts.shape[0])

    if endpoints:
        S = max(2, int(np.ceil(L / step_px)) + 1)
        s = np.linspace(0.0, L, S)
    else:
        S = max(1, int(round(L / step_px)))
        ds = L / S
        s = (np.arange(S) + 0.5) * ds

    rows = np.interp(s, cum, p[:, 0])
    cols = np.interp(s, cum, p[:, 1])
    pts = np.stack([rows, cols], axis=1)

    if S >= 2:
        t = np.zeros_like(pts)
        t[1:-1] = pts[2:] - pts[:-2]
        t[0] = pts[1] - pts[0]
        t[-1] = pts[-1] - pts[-2]
    else:
        # single sample: use the chord, or the first segment for a closed trace
        chord = p[-1] - p[0]
        t = (chord if np.linalg.norm(chord) > 0 else seg[0])[None, :].astype(np.float64)
    t = _unit(t)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return pts, t, n, s


def lateral_offsets(half_width_px: float, n_lateral: int) -> np.ndarray:
    """Signed lateral offsets r, symmetric in [-half_width_px, half_width_px]."""
    if n_lateral < 1:
        raise ValueError("n_lateral must be >= 1")
    if n_lateral == 1:
        return np.zeros(1)
    return np.linspace(-float(half_width_px), float(half_width_px), int(n_lateral))


def sample_strip(
    image: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    half_width_px: float = 3.0,
    n_lateral: int = 7,
    order: int = 1,
) -> np.ndarray:
    """Sample I(points + r * normals) for r in ``lateral_offsets``.

    ``image`` is (H, W) or (C, H, W). Returns (S, R) or (C, S, R) with R = n_lateral.
    Column index 0 is r = -half_width_px (right side), the last column is
    r = +half_width_px (left side, into ``edge_faces[e, 0]``). Out-of-image
    coordinates take the nearest edge value (``mode="nearest"``).
    """
    image = np.asarray(image)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 2)
    if points.shape != normals.shape:
        raise ValueError("points and normals must have the same shape")
    r = lateral_offsets(half_width_px, n_lateral)
    rows = points[:, 0:1] + r[None, :] * normals[:, 0:1]
    cols = points[:, 1:2] + r[None, :] * normals[:, 1:2]
    coords = np.stack([rows.ravel(), cols.ravel()], axis=0)
    S, R = rows.shape

    def _sample(plane: np.ndarray) -> np.ndarray:
        plane = plane.astype(np.float64, copy=False)
        return map_coordinates(plane, coords, order=order, mode="nearest").reshape(S, R)

    if image.ndim == 2:
        return _sample(image)
    if image.ndim == 3:
        out = np.empty((image.shape[0], S, R), dtype=np.float64)
        for c in range(image.shape[0]):
            out[c] = _sample(image[c])
        return out
    raise ValueError(f"image must be (H, W) or (C, H, W), got shape {image.shape}")


@dataclass
class Strip:
    """Strip samples of one edge together with the sampling geometry.

    ``values`` is (S, R) or (C, S, R); ``points``/``tangents``/``normals`` are (S, 2);
    ``s`` is the arclength of each sample; ``r`` the signed lateral offsets (R,),
    positive toward ``edge_faces[e, 0]``.
    """

    edge_id: int
    values: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    s: np.ndarray
    r: np.ndarray
    arclength_px: float

    @property
    def n_samples(self) -> int:
        return int(self.points.shape[0])

    @property
    def lateral_step_px(self) -> float:
        return float(self.r[1] - self.r[0]) if self.r.size > 1 else 1.0


def edge_strip(
    cx: HalfEdgeComplex,
    e: int,
    image: np.ndarray,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    order: int = 1,
    geometry: Optional[np.ndarray] = None,
) -> Strip:
    """Strip around edge ``e`` using ``cx.edge_geometry(e)`` (smoothed if present)."""
    p = cx.edge_geometry(int(e)) if geometry is None else np.asarray(geometry, dtype=np.float64)
    pts, t, n, s = resample_polyline(p, step_px=step_px)
    vals = sample_strip(image, pts, n, half_width_px=half_width_px, n_lateral=n_lateral, order=order)
    seg = np.diff(np.asarray(p, dtype=np.float64), axis=0)
    L = float(np.sqrt((seg**2).sum(axis=1)).sum()) if seg.size else 0.0
    return Strip(
        edge_id=int(e),
        values=vals,
        points=pts,
        tangents=t,
        normals=n,
        s=s,
        r=lateral_offsets(half_width_px, n_lateral),
        arclength_px=L,
    )
