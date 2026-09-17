"""Half-edge cell complex container and topology queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class FaceKind(IntEnum):
    CELL = 0
    GAP = 1
    OUTER = 2


class VertexKind(IntEnum):
    REGULAR = 0
    # Inserted on a closed boundary loop that meets no other face boundary
    # (a cell fully enclosed by one other face). Degree 2 by construction.
    ARTIFICIAL = 1


@dataclass
class HalfEdgeComplex:
    """An embedded 2-complex.

    Arrays are indexed by vertex id, edge id, face id and half-edge id (``2*e + d``).
    ``edge_polyline[e]`` holds the ordered geometric trace from tail to head, first
    point equal to ``vertex_xy[tail]`` and last point equal to ``vertex_xy[head]``.
    ``face_loops[f]`` is a list of oriented boundary loops (lists of half-edge ids);
    the first loop is the outer loop for cell/gap faces, further loops are holes.
    """

    vertex_xy: np.ndarray  # (V, 2) float64 (row, col)
    vertex_kind: np.ndarray  # (V,) int8
    edge_tail: np.ndarray  # (E,) int64
    edge_head: np.ndarray  # (E,) int64
    edge_faces: np.ndarray  # (E, 2) int64: [face left of forward half-edge, face right]
    edge_polyline: List[np.ndarray]  # E arrays (n_k, 2), crack-corner trace (unsmoothed)
    face_kind: np.ndarray  # (F,) int8
    face_label: np.ndarray  # (F,) int64 original label; -1 for gap/outer
    face_loops: List[List[List[int]]]
    he_next: np.ndarray  # (2E,) int64
    pixel_size_um: Optional[float] = None
    shape: Optional[Tuple[int, int]] = None
    # Optional subpixel smoothed geometry, filled by pimorph.complex.geometry
    edge_smooth: Optional[List[np.ndarray]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ sizes
    @property
    def n_vertices(self) -> int:
        return int(self.vertex_xy.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_tail.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.face_kind.shape[0])

    @property
    def n_half_edges(self) -> int:
        return 2 * self.n_edges

    # ------------------------------------------------------------- half-edges
    @staticmethod
    def twin(h: int) -> int:
        return h ^ 1

    @staticmethod
    def edge_of(h: int) -> int:
        return h >> 1

    @staticmethod
    def is_forward(h: int) -> bool:
        return (h & 1) == 0

    def he_origin(self, h: int) -> int:
        e = h >> 1
        return int(self.edge_tail[e] if (h & 1) == 0 else self.edge_head[e])

    def he_target(self, h: int) -> int:
        return self.he_origin(h ^ 1)

    def he_face(self, h: int) -> int:
        e = h >> 1
        return int(self.edge_faces[e, h & 1])

    def he_face_array(self) -> np.ndarray:
        """(2E,) face id on the left of each half-edge."""
        return self.edge_faces.reshape(-1)

    # ---------------------------------------------------------------- faces
    def faces_of_kind(self, kind: FaceKind) -> np.ndarray:
        return np.flatnonzero(self.face_kind == int(kind))

    @property
    def cell_faces(self) -> np.ndarray:
        return self.faces_of_kind(FaceKind.CELL)

    @property
    def gap_faces(self) -> np.ndarray:
        return self.faces_of_kind(FaceKind.GAP)

    @property
    def outer_face(self) -> int:
        idx = self.faces_of_kind(FaceKind.OUTER)
        if idx.size != 1:
            raise ValueError(f"expected exactly one outer face, found {idx.size}")
        return int(idx[0])

    def face_half_edges(self, f: int) -> List[int]:
        return [h for loop in self.face_loops[f] for h in loop]

    def face_edges(self, f: int) -> List[int]:
        return [h >> 1 for h in self.face_half_edges(f)]

    def face_sides(self, f: int) -> int:
        """Number of edge occurrences around the face (n_f in the defect law)."""
        return len(self.face_half_edges(f))

    def face_vertices(self, f: int) -> List[int]:
        return [self.he_origin(h) for h in self.face_half_edges(f)]

    def face_neighbors(self, f: int, kinds: Sequence[FaceKind] = (FaceKind.CELL,)) -> List[int]:
        """Faces across each boundary edge (with multiplicity), filtered by kind."""
        want = {int(k) for k in kinds}
        out = []
        for h in self.face_half_edges(f):
            g = self.he_face(h ^ 1)
            if int(self.face_kind[g]) in want:
                out.append(g)
        return out

    def face_touches_outer(self, f: int) -> bool:
        outer = self.outer_face
        return any(self.he_face(h ^ 1) == outer for h in self.face_half_edges(f))

    # -------------------------------------------------------------- vertices
    def vertex_out_half_edges(self, v: int) -> List[int]:
        """Outgoing half-edges at v in cyclic (counter-clockwise on screen) order."""
        return list(self._rotation()[v])

    def vertex_degree(self, v: int) -> int:
        return len(self._rotation()[v])

    def vertex_degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_vertices, dtype=np.int64)
        np.add.at(deg, self.edge_tail, 1)
        np.add.at(deg, self.edge_head, 1)
        return deg

    def vertex_faces(self, v: int) -> List[int]:
        """Faces incident to v in cyclic order (face left of each outgoing half-edge)."""
        return [self.he_face(h) for h in self.vertex_out_half_edges(v)]

    def vertex_cell_set(self, v: int) -> frozenset:
        return frozenset(f for f in self.vertex_faces(v) if self.face_kind[f] == FaceKind.CELL)

    def _rotation(self) -> List[List[int]]:
        rot = getattr(self, "_rotation_cache", None)
        if rot is None:
            rot = build_rotation_system(self)
            object.__setattr__(self, "_rotation_cache", rot)
        return rot

    def invalidate_caches(self) -> None:
        if hasattr(self, "_rotation_cache"):
            object.__delattr__(self, "_rotation_cache")

    # ------------------------------------------------------------------ edges
    def edge_is_boundary(self, e: int) -> bool:
        outer = self.outer_face
        return bool(self.edge_faces[e, 0] == outer or self.edge_faces[e, 1] == outer)

    def edge_is_cell_cell(self, e: int) -> bool:
        a, b = self.edge_faces[e]
        return bool(self.face_kind[a] == FaceKind.CELL and self.face_kind[b] == FaceKind.CELL)

    def boundary_edges(self) -> np.ndarray:
        outer = self.outer_face
        return np.flatnonzero((self.edge_faces[:, 0] == outer) | (self.edge_faces[:, 1] == outer))

    def cell_cell_edges(self) -> np.ndarray:
        k = self.face_kind[self.edge_faces]
        return np.flatnonzero((k[:, 0] == FaceKind.CELL) & (k[:, 1] == FaceKind.CELL))

    def edges_between(self, f: int, g: int) -> np.ndarray:
        """All edges (connected contact components) shared by faces f and g."""
        a = (self.edge_faces[:, 0] == f) & (self.edge_faces[:, 1] == g)
        b = (self.edge_faces[:, 0] == g) & (self.edge_faces[:, 1] == f)
        return np.flatnonzero(a | b)

    def edge_geometry(self, e: int) -> np.ndarray:
        """Smoothed trace if available, else the crack trace."""
        if self.edge_smooth is not None:
            return self.edge_smooth[e]
        return self.edge_polyline[e]

    # ------------------------------------------------------------- iteration
    def iter_loop_half_edges(self, start: int) -> Iterator[int]:
        h = start
        n = 0
        while True:
            yield h
            h = int(self.he_next[h])
            n += 1
            if h == start:
                return
            if n > self.n_half_edges:
                raise RuntimeError("he_next does not close a loop")

    def skeleton_components(self) -> int:
        """Connected components of the 1-skeleton (vertices + edges)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if self.n_vertices == 0:
            return 0
        rows = np.concatenate([self.edge_tail, self.edge_head])
        cols = np.concatenate([self.edge_head, self.edge_tail])
        a = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(self.n_vertices, self.n_vertices))
        n, _ = connected_components(a, directed=False)
        return int(n)

    # ---------------------------------------------------------- (de)serialize
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertex_xy": self.vertex_xy.tolist(),
            "vertex_kind": self.vertex_kind.tolist(),
            "edge_tail": self.edge_tail.tolist(),
            "edge_head": self.edge_head.tolist(),
            "edge_faces": self.edge_faces.tolist(),
            "edge_polyline": [p.tolist() for p in self.edge_polyline],
            "edge_smooth": None if self.edge_smooth is None else [p.tolist() for p in self.edge_smooth],
            "face_kind": self.face_kind.tolist(),
            "face_label": self.face_label.tolist(),
            "face_loops": self.face_loops,
            "he_next": self.he_next.tolist(),
            "pixel_size_um": self.pixel_size_um,
            "shape": None if self.shape is None else list(self.shape),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HalfEdgeComplex":
        return cls(
            vertex_xy=np.asarray(d["vertex_xy"], dtype=np.float64).reshape(-1, 2),
            vertex_kind=np.asarray(d["vertex_kind"], dtype=np.int8),
            edge_tail=np.asarray(d["edge_tail"], dtype=np.int64),
            edge_head=np.asarray(d["edge_head"], dtype=np.int64),
            edge_faces=np.asarray(d["edge_faces"], dtype=np.int64).reshape(-1, 2),
            edge_polyline=[np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in d["edge_polyline"]],
            face_kind=np.asarray(d["face_kind"], dtype=np.int8),
            face_label=np.asarray(d["face_label"], dtype=np.int64),
            face_loops=[[list(map(int, loop)) for loop in loops] for loops in d["face_loops"]],
            he_next=np.asarray(d["he_next"], dtype=np.int64),
            pixel_size_um=d.get("pixel_size_um"),
            shape=None if d.get("shape") is None else tuple(d["shape"]),
            edge_smooth=None
            if d.get("edge_smooth") is None
            else [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in d["edge_smooth"]],
            provenance=dict(d.get("provenance", {})),
        )


def _outgoing_direction(cx: HalfEdgeComplex, h: int) -> Tuple[float, float]:
    """First segment direction (drow, dcol) of half-edge h leaving its origin."""
    e = h >> 1
    p = cx.edge_polyline[e]
    if (h & 1) == 0:
        a, b = p[0], p[1]
    else:
        a, b = p[-1], p[-2]
    return float(b[0] - a[0]), float(b[1] - a[1])


def build_rotation_system(cx: HalfEdgeComplex) -> List[List[int]]:
    """Outgoing half-edges per vertex sorted counter-clockwise on screen.

    Angles are measured in a y-up frame (x = col, y = -row) so that increasing angle
    is visually counter-clockwise.
    """
    rot: List[List[Tuple[float, int]]] = [[] for _ in range(cx.n_vertices)]
    for e in range(cx.n_edges):
        for d in (0, 1):
            h = 2 * e + d
            v = cx.he_origin(h)
            drow, dcol = _outgoing_direction(cx, h)
            ang = float(np.arctan2(-drow, dcol))
            rot[v].append((ang, h))
    out: List[List[int]] = []
    for lst in rot:
        lst.sort()
        out.append([h for _, h in lst])
    return out


def next_from_rotation(cx: HalfEdgeComplex, rot: List[List[int]]) -> np.ndarray:
    """he_next such that the face stays on the left: next(h) is the outgoing
    half-edge at head(h) immediately clockwise from twin(h)."""
    nxt = np.full(cx.n_half_edges, -1, dtype=np.int64)
    pos: Dict[int, Tuple[int, int]] = {}
    for v, lst in enumerate(rot):
        for i, h in enumerate(lst):
            pos[h] = (v, i)
    for h in range(cx.n_half_edges):
        t = h ^ 1
        v, i = pos[t]
        lst = rot[v]
        nxt[h] = lst[(i - 1) % len(lst)]
    return nxt
