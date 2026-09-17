"""Admissible topology rewrites on the half-edge complex (blueprint section 7).

Every event checks its preconditions, performs a local combinatorial rewrite on a
copy of the complex (or in place), rebuilds the rotation system and face loops from
the edited geometry, validates the result and compares the observed change of
(V, E, F) with the expected one. Expected deltas, all with dV - dE + dF = 0:

    t1_exchange     ( 0,  0,  0)
    contact_death   (-1, -1,  0)   collapse an edge into a degree-4 vertex
    contact_birth   ( 1,  1,  0)   resolve a degree-4 vertex into a short edge
    divide          ( 2,  3,  1)
    extrude         (1-n, -n, -1)  remove an n-sided cell (T2)
    nucleate_gap    ( 2,  3,  1)
    rupture         ( 2,  3,  1)
    reseal          (-2, -3, -1)

Geometry after a rewrite is combinatorially consistent (the rotation system derived
from the polyline end segments reproduces the intended loops) but not guaranteed
to be free of self-intersections. Smoothed traces are dropped by every event, call
``pimorph.complex.geometry.smooth_complex`` again if they are needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .extract import rebuild_topology
from .geometry import face_centroid, polyline_length, polyline_normals
from .halfedge import FaceKind, HalfEdgeComplex, VertexKind
from .incidence import ValidationReport, validate
from .invariants import t1_charge_delta

_EPS = 1e-9
_TWO_PI = 2.0 * np.pi


class EventPreconditionError(ValueError):
    """The requested rewrite does not apply to this local configuration."""


class EventRewriteError(RuntimeError):
    """The rewrite produced an invalid complex or an unexpected (dV, dE, dF)."""


@dataclass
class EventResult:
    cx: HalfEdgeComplex
    event: str
    participants: Dict[str, Any]
    delta_vef: Tuple[int, int, int]
    expected_delta_vef: Tuple[int, int, int]
    report: ValidationReport


# --------------------------------------------------------------------- helpers
def _py(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays so provenance stays JSON friendly."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _require(cond: bool, event: str, msg: str) -> None:
    if not cond:
        raise EventPreconditionError(f"{event}: {msg}")


def _check_vertex(cx: HalfEdgeComplex, v: int, event: str) -> None:
    _require(0 <= int(v) < cx.n_vertices, event, f"vertex id {v} out of range (V={cx.n_vertices})")


def _check_edge(cx: HalfEdgeComplex, e: int, event: str) -> None:
    _require(0 <= int(e) < cx.n_edges, event, f"edge id {e} out of range (E={cx.n_edges})")


def _check_face(cx: HalfEdgeComplex, f: int, event: str) -> None:
    _require(0 <= int(f) < cx.n_faces, event, f"face id {f} out of range (F={cx.n_faces})")


def _copy_complex(cx: HalfEdgeComplex) -> HalfEdgeComplex:
    return HalfEdgeComplex(
        vertex_xy=cx.vertex_xy.copy(),
        vertex_kind=cx.vertex_kind.copy(),
        edge_tail=cx.edge_tail.copy(),
        edge_head=cx.edge_head.copy(),
        edge_faces=cx.edge_faces.copy(),
        edge_polyline=[p.copy() for p in cx.edge_polyline],
        face_kind=cx.face_kind.copy(),
        face_label=cx.face_label.copy(),
        face_loops=[[list(loop) for loop in loops] for loops in cx.face_loops],
        he_next=cx.he_next.copy(),
        pixel_size_um=cx.pixel_size_um,
        shape=cx.shape,
        edge_smooth=None,
        provenance=copy.deepcopy(cx.provenance),
    )


def _prepare(cx: HalfEdgeComplex, inplace: bool) -> HalfEdgeComplex:
    if inplace:
        cx.edge_smooth = None
        return cx
    return _copy_complex(cx)


def _dedupe(poly: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points; a polyline needs at least two distinct points."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 2:
        raise EventRewriteError("polyline degenerated to fewer than two points")
    keep = np.ones(poly.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(poly, axis=0), axis=1) > _EPS
    out = poly[keep]
    if out.shape[0] < 2:
        raise EventRewriteError("polyline degenerated to a single point")
    return out


def _add_vertex(cx: HalfEdgeComplex, xy: Sequence[float]) -> int:
    cx.vertex_xy = np.vstack([cx.vertex_xy, np.asarray(xy, dtype=np.float64).reshape(1, 2)])
    cx.vertex_kind = np.append(cx.vertex_kind, np.int8(VertexKind.REGULAR))
    return cx.n_vertices - 1


def _add_edge(cx: HalfEdgeComplex, tail: int, head: int, left: int, right: int, poly: np.ndarray) -> int:
    cx.edge_tail = np.append(cx.edge_tail, np.int64(tail))
    cx.edge_head = np.append(cx.edge_head, np.int64(head))
    cx.edge_faces = np.vstack([cx.edge_faces, np.array([[left, right]], dtype=np.int64)])
    cx.edge_polyline.append(_dedupe(poly))
    return cx.n_edges - 1


def _add_face(cx: HalfEdgeComplex, kind: FaceKind, label: int) -> int:
    cx.face_kind = np.append(cx.face_kind, np.int8(kind))
    cx.face_label = np.append(cx.face_label, np.int64(label))
    cx.face_loops.append([])
    return cx.n_faces - 1


def _set_endpoint(cx: HalfEdgeComplex, e: int, at_tail: bool, v: int) -> None:
    """Attach the tail or head of edge e to vertex v and move the polyline end onto it."""
    p = cx.edge_polyline[e].copy()
    if at_tail:
        cx.edge_tail[e] = v
        p[0] = cx.vertex_xy[v]
    else:
        cx.edge_head[e] = v
        p[-1] = cx.vertex_xy[v]
    cx.edge_polyline[e] = _dedupe(p)


def _reattach_origin(cx: HalfEdgeComplex, h: int, v: int) -> None:
    """Make v the origin of half-edge h (also used to sync a moved vertex coordinate)."""
    _set_endpoint(cx, h >> 1, (h & 1) == 0, v)


def _oriented_polyline(cx: HalfEdgeComplex, h: int) -> np.ndarray:
    p = cx.edge_polyline[h >> 1]
    return p if (h & 1) == 0 else p[::-1]


def _he_direction(cx: HalfEdgeComplex, h: int) -> np.ndarray:
    """Unit (drow, dcol) of the first segment of half-edge h leaving its origin."""
    p = _oriented_polyline(cx, h)
    d = p[1] - p[0]
    n = np.linalg.norm(d)
    if n <= _EPS:
        raise EventRewriteError(f"half-edge {h} has a zero-length first segment")
    return d / n


def _angle(d: Sequence[float]) -> float:
    """Angle in the y-up frame used by the rotation system (CCW on screen increases)."""
    return float(np.arctan2(-d[0], d[1]))


def _unit_from_angle(a: float) -> np.ndarray:
    return np.array([-np.sin(a), np.cos(a)])


def _left_normal(d: np.ndarray) -> np.ndarray:
    """Unit normal to the left of (drow, dcol) on screen; for a forward half-edge it points into edge_faces[e, 0]."""
    return np.array([-d[1], d[0]])


def _rotate_to(lst: List[int], h: int) -> List[int]:
    i = lst.index(h)
    return lst[i:] + lst[:i]


def _outgoing(cx: HalfEdgeComplex, v: int) -> List[int]:
    return [int(h) for h in cx.vertex_out_half_edges(int(v))]


def _locate(poly: np.ndarray, t: float) -> Tuple[int, np.ndarray]:
    """Segment index k and point at arclength fraction t of an open polyline."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= _EPS:
        raise EventPreconditionError("polyline has zero length")
    s = float(t) * total
    k = int(np.searchsorted(cum, s, side="right") - 1)
    k = min(max(k, 0), seg.size - 1)
    while seg[k] <= _EPS and k > 0:
        k -= 1
    frac = (s - cum[k]) / seg[k]
    return k, poly[k] + frac * (poly[k + 1] - poly[k])


def _polyline_point_at(poly: np.ndarray, t: float) -> np.ndarray:
    return _locate(np.asarray(poly, dtype=np.float64), t)[1]


def _split_polyline_at(poly: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split at arclength fraction t in (0, 1): (part ending at point, point, part starting at point)."""
    if not (0.0 < float(t) < 1.0):
        raise EventPreconditionError(f"split parameter t={t} must lie strictly inside (0, 1)")
    poly = np.asarray(poly, dtype=np.float64)
    k, pt = _locate(poly, t)
    if np.linalg.norm(pt - poly[0]) <= _EPS or np.linalg.norm(pt - poly[-1]) <= _EPS:
        raise EventPreconditionError("split point coincides with an edge endpoint")
    a = _dedupe(np.vstack([poly[: k + 1], pt[None, :]]))
    b = _dedupe(np.vstack([pt[None, :], poly[k + 1 :]]))
    return a, pt, b


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, float(cum[-1]), n)
    return np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], axis=1)


def _compact(
    cx: HalfEdgeComplex, keep_v: np.ndarray, keep_e: np.ndarray, keep_f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop masked-out vertices, edges and faces and renumber ids densely.

    Returns (vmap, emap, fmap) old -> new id arrays with -1 for removed entries.
    Raises if a kept edge still references a removed vertex or face.
    """
    vmap = np.full(cx.n_vertices, -1, dtype=np.int64)
    emap = np.full(cx.n_edges, -1, dtype=np.int64)
    fmap = np.full(cx.n_faces, -1, dtype=np.int64)
    vmap[keep_v] = np.arange(int(keep_v.sum()))
    emap[keep_e] = np.arange(int(keep_e.sum()))
    fmap[keep_f] = np.arange(int(keep_f.sum()))

    tail = vmap[cx.edge_tail[keep_e]]
    head = vmap[cx.edge_head[keep_e]]
    faces = fmap[cx.edge_faces[keep_e]]
    if np.any(tail < 0) or np.any(head < 0):
        raise EventRewriteError("a kept edge references a removed vertex")
    if np.any(faces < 0):
        raise EventRewriteError("a kept edge references a removed face")

    cx.vertex_xy = cx.vertex_xy[keep_v]
    cx.vertex_kind = cx.vertex_kind[keep_v]
    cx.edge_tail = tail
    cx.edge_head = head
    cx.edge_faces = faces
    cx.edge_polyline = [p for p, k in zip(cx.edge_polyline, keep_e) if k]
    if cx.edge_smooth is not None:
        cx.edge_smooth = [p for p, k in zip(cx.edge_smooth, keep_e) if k]
    cx.face_kind = cx.face_kind[keep_f]
    cx.face_label = cx.face_label[keep_f]
    cx.face_loops = [[] for _ in range(cx.n_faces)]
    cx.he_next = np.zeros(2 * cx.n_edges, dtype=np.int64)
    return vmap, emap, fmap


def _counts(cx: HalfEdgeComplex) -> Tuple[int, int, int]:
    return cx.n_vertices, cx.n_edges, cx.n_faces


def _finalize(
    new: HalfEdgeComplex,
    before: Tuple[int, int, int],
    event: str,
    participants: Dict[str, Any],
    expected: Tuple[int, int, int],
) -> EventResult:
    new.he_next = np.zeros(2 * new.n_edges, dtype=np.int64)
    new.face_loops = [[] for _ in range(new.n_faces)]
    try:
        rebuild_topology(new)
    except RuntimeError as exc:
        raise EventRewriteError(f"{event}: rewrite is inconsistent with the embedded geometry ({exc})") from exc
    report = validate(new)
    if not report.ok:
        raise EventRewriteError(f"{event}: result failed validation: {report.messages}")
    delta = (new.n_vertices - before[0], new.n_edges - before[1], new.n_faces - before[2])
    expected = tuple(int(x) for x in expected)
    if delta != expected:
        raise EventRewriteError(f"{event}: observed (dV, dE, dF) = {delta}, expected {expected}")
    participants = _py(participants)
    new.provenance.setdefault("events", []).append(
        {"event": event, "participants": participants, "delta_vef": list(delta)}
    )
    return EventResult(
        cx=new,
        event=event,
        participants=participants,
        delta_vef=delta,
        expected_delta_vef=expected,
        report=report,
    )


def _trivalent_cell_vertex(cx: HalfEdgeComplex, v: int, event: str) -> List[int]:
    """Outgoing half-edges (CCW) of a degree-3 vertex whose three wedges are all cells."""
    _check_vertex(cx, v, event)
    outs = _outgoing(cx, v)
    _require(len(outs) == 3, event, f"vertex {v} has degree {len(outs)}, need 3")
    for h in outs:
        f = cx.he_face(h)
        _require(
            cx.face_kind[f] == FaceKind.CELL,
            event,
            f"vertex {v} touches face {f} of kind {FaceKind(int(cx.face_kind[f])).name}, all wedges must be cells",
        )
    return outs


def _sides(cx: HalfEdgeComplex) -> Dict[int, int]:
    return {int(f): cx.face_sides(int(f)) for f in cx.cell_faces}


def _wedge(cx: HalfEdgeComplex, h_first: int, h_second: int) -> Tuple[float, float]:
    """(start angle, opening) of the CCW wedge from h_first to h_second at their common origin."""
    a0 = _angle(_he_direction(cx, h_first))
    a1 = _angle(_he_direction(cx, h_second))
    return a0, (a1 - a0) % _TWO_PI


def _inside_wedge(a: float, start: float, opening: float, margin: float = 1e-3) -> bool:
    rel = (a - start) % _TWO_PI
    return margin < rel < opening - margin


def _chord_polyline(cx: HalfEdgeComplex, P: np.ndarray, wedge_P, Q: np.ndarray, wedge_Q) -> np.ndarray:
    """Straight chord when it leaves both split points strictly inside the face
    wedge; otherwise a short stub along each wedge bisector keeps the rotation
    system consistent with a staircase trace."""
    d = Q - P
    dist = float(np.linalg.norm(d))
    if dist <= _EPS:
        raise EventPreconditionError("divide: the two split points coincide")
    if _inside_wedge(_angle(d), *wedge_P) and _inside_wedge(_angle(-d), *wedge_Q):
        return np.vstack([P, Q])
    delta = min(0.5, 0.25 * dist)
    n_P = _unit_from_angle(wedge_P[0] + 0.5 * wedge_P[1])
    n_Q = _unit_from_angle(wedge_Q[0] + 0.5 * wedge_Q[1])
    return _dedupe(np.vstack([P, P + delta * n_P, Q + delta * n_Q, Q]))


# ---------------------------------------------------------------------- events
def t1_exchange(cx: HalfEdgeComplex, e: int, new_length: float = 2.0, inplace: bool = False) -> EventResult:
    """T1 neighbour exchange across edge e. Expected (dV, dE, dF) = (0, 0, 0).

    Edge e separates cells A (left) and B (right); its tail and head are trivalent
    with cell-only wedges, C being the third cell at the tail and D the third cell at
    the head. Afterwards C and D share the new edge (which reuses id e), A and B lose
    one side each, C and D gain one. The new edge is perpendicular to the old one,
    has length ``new_length`` and is centred on the old midpoint. Vertex ids are kept:
    the tail moves onto the A side, the head onto the B side.
    """
    event = "t1_exchange"
    _check_edge(cx, e, event)
    e = int(e)
    _require(new_length > 0, event, "new_length must be positive")
    u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
    _require(u != w, event, f"edge {e} is a loop")
    A, B = (int(f) for f in cx.edge_faces[e])
    outs_u = _rotate_to(_trivalent_cell_vertex(cx, u, event), 2 * e)
    outs_w = _rotate_to(_trivalent_cell_vertex(cx, w, event), 2 * e + 1)
    _, x, y = outs_u  # x: C|A edge, y: B|C edge
    _, p, q = outs_w  # p: D|B edge, q: A|D edge
    C, D = cx.he_face(x), cx.he_face(p)
    _require(len({A, B, C, D}) == 4, event, f"cells around edge {e} are not distinct: {A, B, C, D}")
    _require(len({e, x >> 1, y >> 1, p >> 1, q >> 1}) == 5, event, f"edges around edge {e} are not distinct")

    n_before = _sides(cx)
    before = _counts(cx)
    xy_u, xy_w = cx.vertex_xy[u].copy(), cx.vertex_xy[w].copy()
    new = _prepare(cx, inplace)

    d = xy_w - xy_u
    length = float(np.linalg.norm(d))
    _require(length > _EPS, event, "edge endpoints coincide")
    d /= length
    n = _left_normal(d)  # into A
    mid = 0.5 * (xy_u + xy_w)
    new.vertex_xy[u] = mid + 0.5 * new_length * n
    new.vertex_xy[w] = mid - 0.5 * new_length * n
    # A's loop closes x^-1 -> q through the tail vertex, B's loop p^-1 -> y through the head.
    _reattach_origin(new, x, u)
    _reattach_origin(new, q, u)
    _reattach_origin(new, y, w)
    _reattach_origin(new, p, w)
    new.edge_faces[e] = [D, C]
    new.edge_polyline[e] = np.vstack([new.vertex_xy[u], new.vertex_xy[w]])

    participants = {
        "before": {
            "edge": e,
            "vertices": [u, w],
            "cells_in_contact": [A, B],
            "cells_apart": [C, D],
            "side_edges": {"x": x >> 1, "y": y >> 1, "p": p >> 1, "q": q >> 1},
        },
        "after": {"edge": e, "vertices": [u, w], "cells_in_contact": [C, D], "cells_apart": [A, B]},
    }
    res = _finalize(new, before, event, participants, (0, 0, 0))
    n_after = _sides(res.cx)
    if t1_charge_delta(n_before, n_after) != 0:
        raise EventRewriteError(f"{event}: total topological charge changed")
    for f, dn in ((A, -1), (B, -1), (C, 1), (D, 1)):
        if n_after[f] - n_before[f] != dn:
            raise EventRewriteError(f"{event}: face {f} changed by {n_after[f] - n_before[f]} sides, expected {dn}")
    return res


def contact_death(cx: HalfEdgeComplex, e: int, inplace: bool = False) -> EventResult:
    """Collapse edge e (both endpoints trivalent, cell-only wedges) into one degree-4
    vertex at its midpoint. Expected (dV, dE, dF) = (-1, -1, 0). The tail vertex id
    survives; the head vertex and edge e are removed and ids are compacted.
    """
    event = "contact_death"
    _check_edge(cx, e, event)
    e = int(e)
    u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
    _require(u != w, event, f"edge {e} is a loop")
    A, B = (int(f) for f in cx.edge_faces[e])
    _require(A != B, event, f"edge {e} has the same face on both sides")
    outs_u = _rotate_to(_trivalent_cell_vertex(cx, u, event), 2 * e)
    outs_w = _rotate_to(_trivalent_cell_vertex(cx, w, event), 2 * e + 1)
    _, x, y = outs_u
    _, p, q = outs_w
    _require(len({e, x >> 1, y >> 1, p >> 1, q >> 1}) == 5, event, f"edges around edge {e} are not distinct")
    C, D = cx.he_face(x), cx.he_face(p)

    before = _counts(cx)
    mid = 0.5 * (cx.vertex_xy[u] + cx.vertex_xy[w])
    new = _prepare(cx, inplace)
    new.vertex_xy[u] = mid
    for h in (x, y, p, q):
        _reattach_origin(new, h, u)
    keep_v = np.ones(new.n_vertices, dtype=bool)
    keep_e = np.ones(new.n_edges, dtype=bool)
    keep_f = np.ones(new.n_faces, dtype=bool)
    keep_v[w] = False
    keep_e[e] = False
    vmap, emap, _ = _compact(new, keep_v, keep_e, keep_f)

    participants = {
        "before": {"edge": e, "vertices": [u, w], "cells": [A, C, B, D]},
        "after": {
            "vertex": vmap[u],
            "edges_ccw": [emap[x >> 1], emap[y >> 1], emap[p >> 1], emap[q >> 1]],
            "cells": [A, C, B, D],
        },
    }
    return _finalize(new, before, event, participants, (-1, -1, 0))


def contact_birth(
    cx: HalfEdgeComplex, v: int, new_length: float = 2.0, start: int = 0, inplace: bool = False
) -> EventResult:
    """Resolve a degree-4 vertex with four distinct cell wedges into two trivalent
    vertices joined by a new edge of length ``new_length``. Expected (1, 1, 0).

    With outgoing half-edges h0..h3 in CCW order (rotated by ``start``), h0 and h1
    stay at v (moved along the bisector of their wedge) and h2, h3 move to the new
    vertex on the opposite side. The cells in the wedges h1-h2 and h3-h0 gain the
    new contact.
    """
    event = "contact_birth"
    _check_vertex(cx, v, event)
    v = int(v)
    _require(new_length > 0, event, "new_length must be positive")
    outs = _outgoing(cx, v)
    _require(len(outs) == 4, event, f"vertex {v} has degree {len(outs)}, need 4")
    k = int(start) % 4
    outs = outs[k:] + outs[:k]
    faces = [cx.he_face(h) for h in outs]
    for f in faces:
        _require(cx.face_kind[f] == FaceKind.CELL, event, f"vertex {v} touches non-cell face {f}")
    _require(len(set(faces)) == 4, event, f"cells around vertex {v} are not distinct: {faces}")
    _require(len({h >> 1 for h in outs}) == 4, event, f"edges at vertex {v} are not distinct")
    h0, h1, h2, h3 = outs
    F1, F3 = faces[1], faces[3]

    before = _counts(cx)
    n_before = _sides(cx)
    a0 = _angle(_he_direction(cx, h0))
    a1 = _angle(_he_direction(cx, h1))
    bis = _unit_from_angle(a0 + 0.5 * ((a1 - a0) % _TWO_PI))
    xy = cx.vertex_xy[v].copy()
    new = _prepare(cx, inplace)
    new.vertex_xy[v] = xy + 0.5 * new_length * bis
    v2 = _add_vertex(new, xy - 0.5 * new_length * bis)
    _reattach_origin(new, h0, v)
    _reattach_origin(new, h1, v)
    _reattach_origin(new, h2, v2)
    _reattach_origin(new, h3, v2)
    e_new = _add_edge(new, v, v2, F3, F1, np.vstack([new.vertex_xy[v], new.vertex_xy[v2]]))

    participants = {
        "before": {"vertex": v, "cells_ccw": faces, "edges_ccw": [h >> 1 for h in outs]},
        "after": {"edge": e_new, "vertices": [v, v2], "cells_in_contact": [F3, F1]},
    }
    res = _finalize(new, before, event, participants, (1, 1, 0))
    n_after = _sides(res.cx)
    for f in (F1, F3):
        if n_after[f] - n_before[f] != 1:
            raise EventRewriteError(f"{event}: face {f} did not gain exactly one side")
    return res


def divide(
    cx: HalfEdgeComplex, f: int, p: Tuple[int, float], q: Tuple[int, float], inplace: bool = False
) -> EventResult:
    """Split cell face f by a chord between points on two distinct edges of its outer
    loop. ``p`` and ``q`` are ``(edge_id, t)`` with t the arclength fraction along the
    edge polyline. Expected (dV, dE, dF) = (2, 3, 1).

    Each split edge keeps its id for the piece at its tail; the piece at its head is
    appended. Face f keeps the arc from q back to p; the arc from p to q goes to a new
    face appended with the same face_label. Hole loops of f stay with f. Provenance
    ``lineage`` records child -> parent.
    """
    event = "divide"
    _check_face(cx, f, event)
    f = int(f)
    _require(cx.face_kind[f] == FaceKind.CELL, event, f"face {f} is not a cell")
    _require(len(cx.face_loops[f]) >= 1, event, f"face {f} has no boundary loop")
    (e1, t1), (e2, t2) = (int(p[0]), float(p[1])), (int(q[0]), float(q[1]))
    _check_edge(cx, e1, event)
    _check_edge(cx, e2, event)
    _require(e1 != e2, event, "p and q must lie on distinct edges")
    loop = [int(h) for h in cx.face_loops[f][0]]
    loop_edges = {h >> 1 for h in loop}
    for e in (e1, e2):
        _require(e in loop_edges, event, f"edge {e} is not on the outer loop of face {f}")
        sides = [int(cx.edge_faces[e, 0]) == f, int(cx.edge_faces[e, 1]) == f]
        _require(sum(sides) == 1, event, f"edge {e} must have face {f} on exactly one side")
    for t in (t1, t2):
        _require(0.0 < t < 1.0, event, f"t={t} must lie strictly inside (0, 1)")

    before = _counts(cx)
    new = _prepare(cx, inplace)

    def split(e: int, t: float) -> Tuple[int, int, np.ndarray, Tuple[float, float]]:
        a, pt, b = _split_polyline_at(new.edge_polyline[e], t)
        vid = _add_vertex(new, pt)
        eb = _add_edge(new, vid, int(new.edge_head[e]), int(new.edge_faces[e, 0]), int(new.edge_faces[e, 1]), b)
        new.edge_head[e] = vid
        new.edge_polyline[e] = a
        # CCW wedge at the split point that belongs to f: from the half-edge leaving
        # along the loop direction to the one leaving against it.
        to_head, to_tail = 2 * eb, 2 * e + 1
        wedge = _wedge(new, to_head, to_tail) if int(new.edge_faces[e, 0]) == f else _wedge(new, to_tail, to_head)
        return vid, eb, pt, wedge

    vP, e1b, P, wedge_P = split(e1, t1)
    vQ, e2b, Q, wedge_Q = split(e2, t2)

    # Expand the loop through the split edges and mark where P and Q sit.
    expanded: List[int] = []
    pos: Dict[int, int] = {}
    for h in loop:
        eid = h >> 1
        if eid in (e1, e2):
            eb = e1b if eid == e1 else e2b
            seq = [2 * eid, 2 * eb] if (h & 1) == 0 else [2 * eb + 1, 2 * eid + 1]
            expanded.extend(seq)
            pos[eid] = len(expanded) - 1  # half-edge leaving the split point along the loop
        else:
            expanded.append(h)
    n = len(expanded)
    iP, iQ = pos[e1], pos[e2]
    arc1 = [expanded[(iP + k) % n] for k in range((iQ - iP) % n)]  # from P to Q, f on the left

    f_new = _add_face(new, FaceKind.CELL, int(cx.face_label[f]))
    for h in arc1:
        new.edge_faces[h >> 1, h & 1] = f_new
    chord = _add_edge(new, vP, vQ, f, f_new, _chord_polyline(new, P, wedge_P, Q, wedge_Q))
    new.provenance.setdefault("lineage", []).append({"event": event, "parent": f, "child": f_new})

    participants = {
        "before": {"face": f, "split_edges": [e1, e2], "t": [t1, t2]},
        "after": {
            "faces": [f, f_new],
            "new_vertices": [vP, vQ],
            "split_edges": [[e1, e1b], [e2, e2b]],
            "chord": chord,
        },
    }
    return _finalize(new, before, event, participants, (2, 3, 1))


def extrude(cx: HalfEdgeComplex, f: int, inplace: bool = False) -> EventResult:
    """T2: remove an n-sided cell face whose boundary is a single loop of interior
    edges with trivalent vertices. All boundary vertices merge into one vertex at
    the face centroid and every neighbour re-closes through it.
    Expected (dV, dE, dF) = (1 - n, -n, -1). Ids are compacted.
    """
    event = "extrude"
    _check_face(cx, f, event)
    f = int(f)
    _require(cx.face_kind[f] == FaceKind.CELL, event, f"face {f} is not a cell")
    _require(len(cx.face_loops[f]) == 1, event, f"face {f} must have exactly one boundary loop")
    hs = [int(h) for h in cx.face_loops[f][0]]
    n = len(hs)
    _require(n >= 3, event, f"face {f} has {n} sides, need at least 3")
    edges = [h >> 1 for h in hs]
    _require(len(set(edges)) == n, event, f"boundary of face {f} repeats an edge")
    verts = [cx.he_origin(h) for h in hs]
    _require(len(set(verts)) == n, event, f"boundary of face {f} repeats a vertex")
    outer = cx.outer_face
    neighbors = []
    for h in hs:
        g = cx.he_face(h ^ 1)
        _require(g != outer, event, f"face {f} touches the outer face")
        _require(g != f, event, f"face {f} has a bridge edge")
        neighbors.append(g)
    edge_set = set(edges)
    spokes = []
    for v in verts:
        outs = _outgoing(cx, v)
        _require(len(outs) == 3, event, f"boundary vertex {v} has degree {len(outs)}, need 3")
        cand = [h for h in outs if (h >> 1) not in edge_set]
        _require(len(cand) == 1, event, f"boundary vertex {v} has no unique outward edge")
        spokes.append(cand[0])
    c = np.asarray(face_centroid(cx, f, smoothed=False), dtype=np.float64)
    _require(bool(np.all(np.isfinite(c))), event, f"face {f} has no finite centroid")

    before = _counts(cx)
    new = _prepare(cx, inplace)
    vc = _add_vertex(new, c)
    for h in spokes:
        e = h >> 1
        poly = new.edge_polyline[e]
        # Keep the old junction as an interior trace point so the spoke leaves the
        # centroid exactly towards it (CCW order of spokes = loop order of vertices).
        if (h & 1) == 0:
            new.edge_tail[e] = vc
            poly = np.vstack([c[None, :], poly])
        else:
            new.edge_head[e] = vc
            poly = np.vstack([poly, c[None, :]])
        new.edge_polyline[e] = _dedupe(poly)
    keep_v = np.ones(new.n_vertices, dtype=bool)
    keep_e = np.ones(new.n_edges, dtype=bool)
    keep_f = np.ones(new.n_faces, dtype=bool)
    keep_v[verts] = False
    keep_e[edges] = False
    keep_f[f] = False
    vmap, emap, fmap = _compact(new, keep_v, keep_e, keep_f)

    participants = {
        "before": {
            "face": f,
            "sides": n,
            "label": int(cx.face_label[f]),
            "vertices": verts,
            "edges": edges,
            "neighbors": neighbors,
            "spokes": [h >> 1 for h in spokes],
        },
        "after": {
            "vertex": vmap[vc],
            "spokes": [emap[h >> 1] for h in spokes],
            "neighbors": [fmap[g] for g in neighbors],
        },
    }
    return _finalize(new, before, event, participants, (1 - n, -n, -1))


def nucleate_gap(cx: HalfEdgeComplex, v: int, radius: float = 1.0, inplace: bool = False) -> EventResult:
    """Replace a trivalent vertex with three distinct cell wedges by a triangular GAP
    face. New vertices sit at arclength ``radius`` from v along each incident edge,
    the incident edges are shortened to them and three new edges close the triangle.
    Expected (dV, dE, dF) = (2, 3, 1). The vertex id v is reused for the new vertex
    on the first incident edge (CCW order); the gap face is appended with label -1.
    """
    event = "nucleate_gap"
    outs = _trivalent_cell_vertex(cx, v, event)
    v = int(v)
    _require(radius > 0, event, "radius must be positive")
    faces = [cx.he_face(h) for h in outs]
    _require(len(set(faces)) == 3, event, f"cells around vertex {v} are not distinct: {faces}")
    edges = [h >> 1 for h in outs]
    _require(len(set(edges)) == 3, event, f"edges at vertex {v} are not distinct")
    for e in edges:
        _require(cx.edge_tail[e] != cx.edge_head[e], event, f"edge {e} is a loop")
        L = polyline_length(cx.edge_polyline[e])
        _require(L > radius, event, f"edge {e} has length {L:.3f} <= radius {radius}")

    before = _counts(cx)
    new = _prepare(cx, inplace)
    P: List[int] = []
    for i, h in enumerate(outs):
        e = h >> 1
        poly = new.edge_polyline[e]
        L = polyline_length(poly)
        if (h & 1) == 0:
            _, pt, rest = _split_polyline_at(poly, radius / L)
        else:
            rest, pt, _ = _split_polyline_at(poly, 1.0 - radius / L)
        if i == 0:
            vid = v
            new.vertex_xy[v] = pt
        else:
            vid = _add_vertex(new, pt)
        if (h & 1) == 0:
            new.edge_tail[e] = vid
        else:
            new.edge_head[e] = vid
        new.edge_polyline[e] = rest
        P.append(vid)
    g = _add_face(new, FaceKind.GAP, -1)
    # Triangle P0 -> P1 -> P2 is CCW (same order as the wedges), gap on the left.
    tri = []
    for i in range(3):
        a, b = P[i], P[(i + 1) % 3]
        tri.append(_add_edge(new, a, b, g, faces[i], np.vstack([new.vertex_xy[a], new.vertex_xy[b]])))

    participants = {
        "before": {"vertex": v, "cells_ccw": faces, "edges_ccw": edges},
        "after": {"gap": g, "vertices": P, "triangle_edges": tri, "shortened_edges": edges, "cells_ccw": faces},
    }
    return _finalize(new, before, event, participants, (2, 3, 1))


def rupture(cx: HalfEdgeComplex, e: int, width: float = 1.0, inplace: bool = False) -> EventResult:
    """Open a cell-cell edge e into an explicit GAP face of width ``width``.

    Edge e (cells A left, B right) with trivalent endpoints is replaced by two offset
    copies of its trace, one shifted by +width/2 towards A (keeps id e, faces [A, G])
    and one shifted by -width/2 towards B (faces [G, B]). Each endpoint splits into an
    A-side vertex (keeps the old id) and a B-side vertex joined by a straight cap edge,
    so the gap is a 4-sided face: two long sides and two caps. The third face at each
    endpoint (C at the tail, D at the head) gains the cap as a side.

    Euler count: vertices 2 -> 4 (+2), edges 1 -> 2 long + 2 caps (+3), faces +1 (G),
    so the expected (dV, dE, dF) = (2, 3, 1) with dV - dE + dF = 0.
    """
    event = "rupture"
    _check_edge(cx, e, event)
    e = int(e)
    _require(width > 0, event, "width must be positive")
    u, w = int(cx.edge_tail[e]), int(cx.edge_head[e])
    _require(u != w, event, f"edge {e} is a loop")
    A, B = (int(f) for f in cx.edge_faces[e])
    _require(cx.edge_is_cell_cell(e), event, f"edge {e} is not a cell-cell edge")
    _require(A != B, event, f"edge {e} has the same cell on both sides")
    outs_u = _outgoing(cx, u)
    outs_w = _outgoing(cx, w)
    _require(len(outs_u) == 3, event, f"tail vertex {u} has degree {len(outs_u)}, need 3")
    _require(len(outs_w) == 3, event, f"head vertex {w} has degree {len(outs_w)}, need 3")
    _, x, y = _rotate_to(outs_u, 2 * e)  # x on the A side, y on the B side
    _, p, q = _rotate_to(outs_w, 2 * e + 1)  # p on the B side, q on the A side
    C, D = cx.he_face(x), cx.he_face(p)
    outer = cx.outer_face
    _require(C != outer and D != outer, event, f"edge {e} ends on the outer face")

    before = _counts(cx)
    new = _prepare(cx, inplace)
    poly = new.edge_polyline[e]
    off = 0.5 * width * polyline_normals(poly)  # left normals point into A
    poly_a = poly + off
    poly_b = poly - off
    g = _add_face(new, FaceKind.GAP, -1)
    new.vertex_xy[u] = poly_a[0]
    new.vertex_xy[w] = poly_a[-1]
    u_b = _add_vertex(new, poly_b[0])
    w_b = _add_vertex(new, poly_b[-1])
    new.edge_faces[e] = [A, g]
    new.edge_polyline[e] = _dedupe(poly_a)
    e_b = _add_edge(new, u_b, w_b, g, B, poly_b)
    _reattach_origin(new, x, u)
    _reattach_origin(new, q, w)
    _reattach_origin(new, y, u_b)
    _reattach_origin(new, p, w_b)
    cap_u = _add_edge(new, u, u_b, g, C, np.vstack([new.vertex_xy[u], new.vertex_xy[u_b]]))
    cap_w = _add_edge(new, w, w_b, D, g, np.vstack([new.vertex_xy[w], new.vertex_xy[w_b]]))

    participants = {
        "before": {"edge": e, "vertices": [u, w], "cells": [A, B], "cap_faces": [C, D]},
        "after": {
            "gap": g,
            "long_edges": [e, e_b],
            "cap_edges": [cap_u, cap_w],
            "vertices_a_side": [u, w],
            "vertices_b_side": [u_b, w_b],
            "cells": [A, B],
            "cap_faces": [C, D],
        },
    }
    return _finalize(new, before, event, participants, (2, 3, 1))


def reseal(cx: HalfEdgeComplex, g: int, inplace: bool = False) -> EventResult:
    """Inverse of rupture: collapse a 4-sided GAP face g (two long sides across two
    distinct cells, two caps, trivalent corners) into a single cell-cell edge along
    the mean of the two long traces. The long pair is the pair of opposite sides with
    the larger total arclength. Expected (dV, dE, dF) = (-2, -3, -1). Ids are compacted.
    """
    event = "reseal"
    _check_face(cx, g, event)
    g = int(g)
    _require(cx.face_kind[g] == FaceKind.GAP, event, f"face {g} is not a gap")
    _require(len(cx.face_loops[g]) == 1, event, f"gap {g} must have exactly one boundary loop")
    hs = [int(h) for h in cx.face_loops[g][0]]
    _require(len(hs) == 4, event, f"gap {g} has {len(hs)} sides, need 4")
    _require(len({h >> 1 for h in hs}) == 4, event, f"gap {g} repeats an edge")
    corners = [cx.he_origin(h) for h in hs]
    _require(len(set(corners)) == 4, event, f"gap {g} repeats a vertex")
    for v in corners:
        _require(cx.vertex_degree(v) == 3, event, f"corner {v} has degree {cx.vertex_degree(v)}, need 3")
    lengths = [polyline_length(cx.edge_polyline[h >> 1]) for h in hs]
    if lengths[1] + lengths[3] > lengths[0] + lengths[2]:
        hs = hs[1:] + hs[:1]
    h0, h1, h2, h3 = hs
    across = [cx.he_face(h ^ 1) for h in hs]
    B, D, A, C = across  # long h0 across B, cap h1 across D, long h2 across A, cap h3 across C
    _require(
        cx.face_kind[A] == FaceKind.CELL and cx.face_kind[B] == FaceKind.CELL,
        event,
        f"long sides of gap {g} must be bounded by cells, got faces {A}, {B}",
    )
    _require(A != B, event, f"gap {g} is bounded by the same cell on both long sides")
    P0, P1, P2, P3 = (cx.he_origin(h) for h in hs)
    loop_edges = {h >> 1 for h in hs}
    ext = []
    for v in (P0, P1, P2, P3):
        cand = [h for h in _outgoing(cx, v) if (h >> 1) not in loop_edges]
        _require(len(cand) == 1, event, f"corner {v} has no unique external edge")
        ext.append(cand[0])

    before = _counts(cx)
    xy_u = 0.5 * (cx.vertex_xy[P0] + cx.vertex_xy[P3])
    xy_w = 0.5 * (cx.vertex_xy[P1] + cx.vertex_xy[P2])
    c0 = _oriented_polyline(cx, h0)  # P0 -> P1
    c2 = _oriented_polyline(cx, h2)[::-1]  # P3 -> P2
    n_pts = max(c0.shape[0], c2.shape[0], 2)
    mean = 0.5 * (_resample(c0, n_pts) + _resample(c2, n_pts))
    mean[0] = xy_u
    mean[-1] = xy_w
    new = _prepare(cx, inplace)
    new.vertex_xy[P0] = xy_u
    new.vertex_xy[P1] = xy_w
    e0 = h0 >> 1
    new.edge_tail[e0] = P0
    new.edge_head[e0] = P1
    new.edge_faces[e0] = [A, B]
    new.edge_polyline[e0] = _dedupe(mean)
    _reattach_origin(new, ext[0], P0)
    _reattach_origin(new, ext[3], P0)
    _reattach_origin(new, ext[1], P1)
    _reattach_origin(new, ext[2], P1)
    keep_v = np.ones(new.n_vertices, dtype=bool)
    keep_e = np.ones(new.n_edges, dtype=bool)
    keep_f = np.ones(new.n_faces, dtype=bool)
    keep_v[[P2, P3]] = False
    keep_e[[h1 >> 1, h2 >> 1, h3 >> 1]] = False
    keep_f[g] = False
    vmap, emap, fmap = _compact(new, keep_v, keep_e, keep_f)

    participants = {
        "before": {"gap": g, "corners": [P0, P1, P2, P3], "long_edges": [e0, h2 >> 1], "cap_edges": [h1 >> 1, h3 >> 1]},
        "after": {
            "edge": emap[e0],
            "vertices": [vmap[P0], vmap[P1]],
            "cells": [fmap[A], fmap[B]],
            "cap_faces": [fmap[C], fmap[D]],
        },
    }
    return _finalize(new, before, event, participants, (-2, -3, -1))


__all__ = [
    "EventPreconditionError",
    "EventResult",
    "EventRewriteError",
    "contact_birth",
    "contact_death",
    "divide",
    "extrude",
    "nucleate_gap",
    "reseal",
    "rupture",
    "t1_exchange",
]
