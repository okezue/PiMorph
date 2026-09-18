"""Curved monolayers: the 2-complex of a cell sheet embedded in 3-D.

A monolayer on a vessel wall is read off the 3-D complex: the face of each cell is
its interface with the lumen 3-cell, the edges are the triple lines shared by two
such lumen interfaces (cell | cell | lumen), and the vertices are the 0-cells at
their ends. Faces are oriented with the sheet normal pointing out of the lumen,
so neighbouring faces induce opposite orientations on a shared edge. Edges with
a single sheet face (the free rim of the sheet) get the single outer face on
their other side, exactly like the outer face of the planar ``HalfEdgeComplex``;
its boundary chain is minus the sum of the sheet faces, so B1 @ B2 == 0 holds
identically.

The lumen is the outer 3-cell by default (background touching the volume
border) or, with ``lumen_label``, the single 3-cell carrying that label. A cell
with several lumen interfaces contributes its largest one (recorded in
``provenance``); cells without lumen contact are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from ..complex.halfedge import FaceKind
from .extract3d import CellComplex3D, CellKind3D, _euler_circuits, extract_complex3d
from .incidence3d import boundary_matrices3d


@dataclass
class SurfaceComplex:
    """A 2-complex embedded in 3-D with the combinatorics of ``HalfEdgeComplex``.

    Half-edge ``2 * e`` runs tail -> head; ``edge_faces[e] = (left, right)`` where the
    left face is the one whose oriented boundary traverses e forward. -1 marks a
    degenerate side (multiplicity other than one). ``face_loops[f]`` lists closed
    walks of half-edge ids; the last face is the outer face of the sheet.
    """

    vertex_xyz: np.ndarray  # (V, 3) float64 (z, y, x)
    vertex_kind: np.ndarray  # (V,) int8
    vertex_id3d: np.ndarray  # (V,) 0-cell id in the source complex
    edge_tail: np.ndarray  # (E,) int64
    edge_head: np.ndarray  # (E,) int64
    edge_id3d: np.ndarray  # (E,) 1-cell id in the source complex
    edge_length: np.ndarray  # (E,) voxel edges
    edge_polyline: List[np.ndarray]  # E arrays (n_k + 1, 3)
    edge_faces: np.ndarray  # (E, 2) int64
    face_kind: np.ndarray  # (F,) int8 FaceKind.CELL or FaceKind.OUTER
    face_label: np.ndarray  # (F,) int64 original label, -1 for outer
    face_cell3d: np.ndarray  # (F,) 3-cell id, -1 for outer
    face_interface3d: np.ndarray  # (F,) 2-cell id, -1 for outer
    face_area: np.ndarray  # (F,) voxel faces, 0 for outer
    face_boundary: List[np.ndarray]  # F arrays (m, 2) int64 (edge, coef)
    face_loops: List[List[List[int]]]
    lumen_cell: int
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.vertex_xyz.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_tail.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.face_kind.shape[0])

    @property
    def cell_faces(self) -> np.ndarray:
        return np.flatnonzero(self.face_kind == FaceKind.CELL)

    @property
    def outer_face(self) -> int:
        return self.n_faces - 1

    def vertex_degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_vertices, dtype=np.int64)
        np.add.at(deg, self.edge_tail, 1)
        np.add.at(deg, self.edge_head, 1)
        return deg

    def boundary_edges(self) -> np.ndarray:
        outer = self.outer_face
        return np.flatnonzero((self.edge_faces[:, 0] == outer) | (self.edge_faces[:, 1] == outer))

    def face_edges(self, f: int) -> np.ndarray:
        return np.unique(np.array([h >> 1 for loop in self.face_loops[f] for h in loop], dtype=np.int64))

    def face_sides(self, f: int) -> int:
        return sum(len(loop) for loop in self.face_loops[f])

    def face_neighbors(self, f: int) -> List[int]:
        """Cell faces across each boundary edge of f (with multiplicity)."""
        out = []
        for loop in self.face_loops[f]:
            for h in loop:
                a, b = self.edge_faces[h >> 1]
                g = int(b if a == f else a)
                if g >= 0 and self.face_kind[g] == FaceKind.CELL:
                    out.append(g)
        return out


def surface_boundary_matrices(sc: SurfaceComplex) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """(B1, B2) with the outer face included; B1 @ B2 == 0 identically."""
    V, E, F = sc.n_vertices, sc.n_edges, sc.n_faces
    e_idx = np.arange(E)
    B1 = sparse.coo_matrix(
        (
            np.concatenate([-np.ones(E), np.ones(E)]),
            (np.concatenate([sc.edge_tail, sc.edge_head]), np.concatenate([e_idx, e_idx])),
        ),
        shape=(V, E),
    ).tocsr()
    B1.sum_duplicates()
    B1.eliminate_zeros()
    r, c, v = [], [], []
    for f in range(F):
        b = sc.face_boundary[f]
        if b.shape[0]:
            r.append(b[:, 0])
            c.append(np.full(b.shape[0], f, dtype=np.int64))
            v.append(b[:, 1].astype(np.float64))
    if r:
        B2 = sparse.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(E, F)).tocsr()
    else:
        B2 = sparse.csr_matrix((E, F), dtype=np.float64)
    B2.sum_duplicates()
    B2.eliminate_zeros()
    return B1, B2


@dataclass
class SurfaceValidationReport:
    ok: bool
    b1b2_zero: bool
    every_edge_two_faces: bool
    loops_match_boundary: bool
    euler_characteristic: int
    n_vertices: int
    n_edges: int
    n_faces: int
    n_boundary_edges: int
    degree_histogram: Dict[int, int] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


def validate_surface(sc: SurfaceComplex) -> SurfaceValidationReport:
    msgs: List[str] = []
    B1, B2 = surface_boundary_matrices(sc)
    p = B1 @ B2
    p.eliminate_zeros()
    b1b2_zero = p.nnz == 0
    if not b1b2_zero:
        msgs.append(f"B1@B2 has {p.nnz} nonzeros")
    if sc.n_edges:
        counts = np.asarray(abs(B2).sum(axis=1)).ravel()
        two = bool(np.all(counts == 2)) and bool(np.all(sc.edge_faces >= 0))
    else:
        two = True
    if not two:
        msgs.append("an edge does not have exactly two face incidences")
    loops_match = True
    for f in range(sc.n_faces):
        acc: Dict[int, int] = {}
        for loop in sc.face_loops[f]:
            for h in loop:
                acc[h >> 1] = acc.get(h >> 1, 0) + (1 if (h & 1) == 0 else -1)
        acc = {e: c for e, c in acc.items() if c != 0}
        want = {int(e): int(c) for e, c in sc.face_boundary[f]}
        if acc != want:
            loops_match = False
            msgs.append(f"face {f}: loops do not reproduce the boundary chain")
            break
    chi = int(sc.n_vertices - sc.n_edges + (sc.n_faces - 1))
    degs = sc.vertex_degrees()
    hist = {int(d): int(n) for d, n in zip(*np.unique(degs, return_counts=True))} if sc.n_vertices else {}
    return SurfaceValidationReport(
        ok=b1b2_zero and two and loops_match,
        b1b2_zero=b1b2_zero,
        every_edge_two_faces=two,
        loops_match_boundary=loops_match,
        euler_characteristic=chi,
        n_vertices=sc.n_vertices,
        n_edges=sc.n_edges,
        n_faces=sc.n_faces,
        n_boundary_edges=int(sc.boundary_edges().size),
        degree_histogram=hist,
        messages=msgs,
    )


def _flip_loop(loop: List[int]) -> List[int]:
    return [h ^ 1 for h in reversed(loop)]


def surface_complex_from_labels(
    labels3d: np.ndarray, lumen_label: Optional[int] = None, cx: Optional[CellComplex3D] = None
) -> SurfaceComplex:
    """Sheet complex of a single-layer label volume (see module docstring)."""
    if cx is None:
        cx = extract_complex3d(labels3d)
    if lumen_label is None:
        lumen = cx.outer_cell
    else:
        ids = np.flatnonzero(cx.cell_label == lumen_label)
        if ids.size != 1:
            raise ValueError(f"lumen label {lumen_label} must form exactly one 6-connected component, found {ids.size}")
        lumen = int(ids[0])

    _, B2, B3 = boundary_matrices3d(cx)
    sigma = np.asarray(B3[:, lumen].todense()).ravel()  # +1 when the interface normal points out of the lumen

    sheet_cells: List[int] = []
    sheet_if: List[int] = []
    multi: List[int] = []
    skipped: List[int] = []
    for c in cx.cell_cells.tolist():
        if c == lumen:
            continue
        fs = cx.interfaces_between(c, lumen)
        if fs.size == 0:
            skipped.append(c)
            continue
        if fs.size > 1:
            multi.append(c)
            fs = fs[np.argsort(-cx.interface_area[fs], kind="stable")]
        sheet_cells.append(c)
        sheet_if.append(int(fs[0]))
    n_sheet = len(sheet_cells)

    # Edges: 1-cells with a nonzero coefficient on any sheet face.
    edge_ids = np.unique(
        np.concatenate([cx.interface_boundary[f][:, 0] for f in sheet_if] + [np.zeros(0, dtype=np.int64)])
    )
    e_new = {int(e): i for i, e in enumerate(edge_ids.tolist())}
    vert_ids = (
        np.unique(np.concatenate([cx.line_tail[edge_ids], cx.line_head[edge_ids]]))
        if edge_ids.size
        else np.zeros(0, dtype=np.int64)
    )
    v_new = {int(v): i for i, v in enumerate(vert_ids.tolist())}
    E = edge_ids.size
    edge_tail = np.array([v_new[int(v)] for v in cx.line_tail[edge_ids]], dtype=np.int64)
    edge_head = np.array([v_new[int(v)] for v in cx.line_head[edge_ids]], dtype=np.int64)

    face_boundary: List[np.ndarray] = []
    face_loops: List[List[List[int]]] = []
    total = np.zeros(E, dtype=np.int64)
    for f in sheet_if:
        s = int(sigma[f])
        b = cx.interface_boundary[f]
        rows = (
            np.stack([np.array([e_new[int(e)] for e in b[:, 0]], dtype=np.int64), s * b[:, 1]], axis=1)
            if b.shape[0]
            else np.zeros((0, 2), dtype=np.int64)
        )
        face_boundary.append(rows)
        np.add.at(total, rows[:, 0], rows[:, 1])
        loops = []
        for loop in cx.interface_loops[f]:
            mapped = [2 * e_new[h >> 1] + (h & 1) for h in loop]
            loops.append(mapped if s > 0 else _flip_loop(mapped))
        face_loops.append(loops)
    outer_rows = np.flatnonzero(total)
    outer_b = (
        np.stack([outer_rows, -total[outer_rows]], axis=1) if outer_rows.size else np.zeros((0, 2), dtype=np.int64)
    )
    face_boundary.append(outer_b)
    trav = []
    for e, cf in outer_b.tolist():
        trav.extend([(e, 0 if cf > 0 else 1)] * abs(cf))
    face_loops.append(_euler_circuits(trav, edge_tail, edge_head) if trav else [])
    F = n_sheet + 1

    edge_faces = np.full((E, 2), -1, dtype=np.int64)
    plus: List[List[int]] = [[] for _ in range(E)]
    minus: List[List[int]] = [[] for _ in range(E)]
    for f in range(F):
        for e, cf in face_boundary[f].tolist():
            (plus if cf > 0 else minus)[e].extend([f] * abs(cf))
    for e in range(E):
        if len(plus[e]) == 1 and len(minus[e]) == 1:
            edge_faces[e] = (plus[e][0], minus[e][0])
    degenerate = int(np.sum(edge_faces[:, 0] < 0))

    face_kind = np.concatenate(
        [np.full(n_sheet, FaceKind.CELL, dtype=np.int8), np.array([FaceKind.OUTER], dtype=np.int8)]
    )
    face_cell3d = np.array(sheet_cells + [-1], dtype=np.int64)
    face_if3d = np.array(sheet_if + [-1], dtype=np.int64)
    face_label = np.where(face_cell3d >= 0, cx.cell_label[np.maximum(face_cell3d, 0)], -1).astype(np.int64)
    face_area = np.where(face_if3d >= 0, cx.interface_area[np.maximum(face_if3d, 0)], 0).astype(np.int64)

    prov = {
        "lumen_cell": lumen,
        "lumen_is_outer": lumen == cx.outer_cell,
        "cells_with_multiple_lumen_interfaces": multi,
        "cells_without_lumen_contact": skipped,
        "n_degenerate_edges": degenerate,
        "source_n_cells": int(cx.cells_of_kind(CellKind3D.CELL).size),
    }
    return SurfaceComplex(
        vertex_xyz=cx.vertex_xyz[vert_ids].reshape(-1, 3),
        vertex_kind=cx.vertex_kind[vert_ids],
        vertex_id3d=vert_ids.astype(np.int64),
        edge_tail=edge_tail,
        edge_head=edge_head,
        edge_id3d=edge_ids.astype(np.int64),
        edge_length=cx.line_length[edge_ids],
        edge_polyline=[cx.line_polyline[int(e)] for e in edge_ids],
        edge_faces=edge_faces,
        face_kind=face_kind,
        face_label=face_label,
        face_cell3d=face_cell3d,
        face_interface3d=face_if3d,
        face_area=face_area,
        face_boundary=face_boundary,
        face_loops=face_loops,
        lumen_cell=lumen,
        provenance=prov,
    )
