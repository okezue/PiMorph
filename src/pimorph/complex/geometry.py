"""Subpixel geometry on the complex: smoothing, arclength, curvature, areas.

Crack traces are staircases. A diagonal boundary of true length L has crack length
sqrt(2) * L, which is the orientation bias of pixel-count contact lengths.
Endpoint-fixed Laplacian smoothing removes most of it while keeping vertices in
place, so arclengths and face perimeters become orientation-invariant to a few
percent.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .halfedge import FaceKind, HalfEdgeComplex


def smooth_polyline(p: np.ndarray, iterations: int = 4, lam: float = 0.5) -> np.ndarray:
    """Endpoint-fixed Laplacian smoothing of an open or closed polyline.

    For closed polylines (first == last point) the shared endpoint is kept fixed as
    well, which is where the artificial vertex sits.
    """
    q = np.asarray(p, dtype=np.float64).copy()
    n = q.shape[0]
    if n < 3 or iterations <= 0:
        return q
    for _ in range(iterations):
        lap = 0.5 * (q[:-2] + q[2:]) - q[1:-1]
        q[1:-1] += lam * lap
    return q


def polyline_length(p: np.ndarray) -> float:
    if p.shape[0] < 2:
        return 0.0
    d = np.diff(p, axis=0)
    return float(np.sqrt((d**2).sum(axis=1)).sum())


def polyline_curvature(p: np.ndarray) -> np.ndarray:
    """Discrete signed curvature at interior points (turning angle / mean segment length).
    Positive when turning counter-clockwise on screen."""
    n = p.shape[0]
    if n < 3:
        return np.zeros(n)
    # y-up frame for orientation
    xy = np.stack([p[:, 1], -p[:, 0]], axis=1)
    d1 = xy[1:-1] - xy[:-2]
    d2 = xy[2:] - xy[1:-1]
    l1 = np.linalg.norm(d1, axis=1)
    l2 = np.linalg.norm(d2, axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dot = (d1 * d2).sum(axis=1)
    ang = np.arctan2(cross, dot)
    seg = 0.5 * (l1 + l2)
    k = np.zeros(n)
    good = seg > 0
    k[1:-1][good] = ang[good] / seg[good]
    return k


def polyline_normals(p: np.ndarray) -> np.ndarray:
    """Unit normals pointing to the LEFT of the travel direction on screen.

    Returned in (row, col) components. For the forward half-edge this points into
    ``edge_faces[e, 0]``.
    """
    n = p.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    t = t / norm
    # travelling with tangent (drow, dcol); left on screen is (-dcol, drow)?  Check: facing
    # +row (down) left is +col -> tangent (1, 0) must map to (0, 1): (-dcol, drow) = (0, 1). Yes.
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def smooth_complex(cx: HalfEdgeComplex, iterations: int = 4, lam: float = 0.5) -> HalfEdgeComplex:
    """Fill ``cx.edge_smooth`` with endpoint-fixed smoothed traces (in place, returns cx)."""
    cx.edge_smooth = [smooth_polyline(p, iterations=iterations, lam=lam) for p in cx.edge_polyline]
    cx.provenance["smoothing"] = {"iterations": iterations, "lam": lam}
    return cx


def edge_arclength(cx: HalfEdgeComplex, e: int, smoothed: bool = True) -> float:
    p = cx.edge_smooth[e] if (smoothed and cx.edge_smooth is not None) else cx.edge_polyline[e]
    return polyline_length(p)


def edge_arclengths(cx: HalfEdgeComplex, smoothed: bool = True) -> np.ndarray:
    return np.array([edge_arclength(cx, e, smoothed) for e in range(cx.n_edges)], dtype=np.float64)


def _loop_points(cx: HalfEdgeComplex, loop: List[int], smoothed: bool) -> np.ndarray:
    pts = []
    for h in loop:
        e = h >> 1
        p = cx.edge_smooth[e] if (smoothed and cx.edge_smooth is not None) else cx.edge_polyline[e]
        pts.append(p[:-1] if (h & 1) == 0 else p[::-1][:-1])
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def loop_signed_area(pts: np.ndarray) -> float:
    """Shoelace in a y-up frame: positive for counter-clockwise on screen."""
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 1]
    y = -pts[:, 0]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def face_area(cx: HalfEdgeComplex, f: int, smoothed: bool = True) -> float:
    """Polygon area of the face (outer loop minus holes). Cell loops are CCW so the
    outer loop is positive and hole loops (traced with the face on the left) negative."""
    total = 0.0
    for loop in cx.face_loops[f]:
        total += loop_signed_area(_loop_points(cx, loop, smoothed))
    return float(total)


def face_perimeter(cx: HalfEdgeComplex, f: int, smoothed: bool = True, outer_loop_only: bool = False) -> float:
    loops = cx.face_loops[f][:1] if outer_loop_only else cx.face_loops[f]
    total = 0.0
    for loop in loops:
        pts = _loop_points(cx, loop, smoothed)
        total += polyline_length(np.vstack([pts, pts[:1]]))
    return float(total)


def face_centroid(cx: HalfEdgeComplex, f: int, smoothed: bool = True) -> Tuple[float, float]:
    """Area-weighted centroid of the outer loop polygon in (row, col)."""
    loops = cx.face_loops[f]
    if not loops:
        return (float("nan"), float("nan"))
    pts = _loop_points(cx, loops[0], smoothed)
    x = pts[:, 1]
    y = pts[:, 0]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-12:
        return (float(y.mean()), float(x.mean()))
    cx_ = float(((x + x1) * cross).sum() / (6 * a))
    cy_ = float(((y + y1) * cross).sum() / (6 * a))
    return (cy_, cx_)


def face_pixel_area(labels: np.ndarray, cx: HalfEdgeComplex) -> np.ndarray:
    """Exact pixel counts per face using the original label image and face_label."""
    out = np.zeros(cx.n_faces, dtype=np.int64)
    counts = np.bincount(labels.ravel())
    for f in cx.cell_faces:
        lab = int(cx.face_label[f])
        if lab < counts.size:
            out[f] = counts[lab]
    return out


def vertex_positions_um(cx: HalfEdgeComplex) -> Optional[np.ndarray]:
    if cx.pixel_size_um is None:
        return None
    return cx.vertex_xy * float(cx.pixel_size_um)


def refine_vertices(cx: HalfEdgeComplex, boundary_prob: np.ndarray, radius: int = 1) -> HalfEdgeComplex:
    """Move regular vertices to the boundary-probability-weighted centroid of a small
    window. Endpoints of the smoothed traces follow. Cyclic order is unchanged.
    """
    H, W = boundary_prob.shape
    new = cx.vertex_xy.copy()
    for v in range(cx.n_vertices):
        r, c = cx.vertex_xy[v]
        r0, c0 = int(round(r)), int(round(c))
        rs = slice(max(r0 - radius, 0), min(r0 + radius + 1, H))
        cs = slice(max(c0 - radius, 0), min(c0 + radius + 1, W))
        win = boundary_prob[rs, cs]
        if win.size == 0 or win.sum() <= 0:
            continue
        rr, cc = np.mgrid[rs, cs]
        w = win / win.sum()
        new[v] = [float((rr * w).sum()), float((cc * w).sum())]
    delta = new - cx.vertex_xy
    cx.vertex_xy = new
    # Only the smoothed traces follow the refined vertices; edge_polyline stays the
    # exact crack trace (and keeps driving the rotation system).
    if cx.edge_smooth is None:
        cx.edge_smooth = [p.copy() for p in cx.edge_polyline]
    for e in range(cx.n_edges):
        p = cx.edge_smooth[e].copy()
        p[0] += delta[cx.edge_tail[e]]
        p[-1] += delta[cx.edge_head[e]]
        cx.edge_smooth[e] = p
    return cx


def face_kind_name(k: int) -> str:
    return FaceKind(int(k)).name.lower()
