"""Oriented boundary matrices and exact validity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from .halfedge import FaceKind, HalfEdgeComplex, VertexKind


def boundary_matrices(cx: HalfEdgeComplex, include_outer: bool = True) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return (B1, B2).

    B1 (V x E): -1 at the tail vertex, +1 at the head vertex of each edge.
    B2 (E x F): +1 when the forward half-edge lies on a loop of face f, -1 when the
    backward half-edge does. With the outer face included every edge has exactly two
    incident faces and B1 @ B2 == 0 must hold identically.
    """
    V, E, F = cx.n_vertices, cx.n_edges, cx.n_faces
    e_idx = np.arange(E)
    rows = np.concatenate([cx.edge_tail, cx.edge_head])
    cols = np.concatenate([e_idx, e_idx])
    vals = np.concatenate([-np.ones(E), np.ones(E)])
    B1 = sparse.coo_matrix((vals, (rows, cols)), shape=(V, E)).tocsr()
    B1.sum_duplicates()
    B1.eliminate_zeros()

    r2: List[int] = []
    c2: List[int] = []
    v2: List[float] = []
    for f in range(F):
        if not include_outer and cx.face_kind[f] == FaceKind.OUTER:
            continue
        for loop in cx.face_loops[f]:
            for h in loop:
                r2.append(h >> 1)
                c2.append(f)
                v2.append(1.0 if (h & 1) == 0 else -1.0)
    B2 = sparse.coo_matrix((v2, (r2, c2)), shape=(E, F)).tocsr()
    B2.sum_duplicates()
    B2.eliminate_zeros()
    return B1, B2


@dataclass
class ValidationReport:
    ok: bool
    b1b2_zero: bool
    every_edge_two_faces: bool
    loops_match_edge_faces: bool
    loops_cover_all_half_edges: bool
    vertex_links_ok: bool
    euler_residual: int
    n_vertices: int
    n_edges: int
    n_faces: int
    n_cells: int
    n_gaps: int
    n_boundary_edges: int
    n_artificial_vertices: int
    skeleton_components: int
    degree_histogram: Dict[int, int] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["degree_histogram"] = {int(k): int(v) for k, v in self.degree_histogram.items()}
        return d


def euler_residual(cx: HalfEdgeComplex) -> int:
    """V - E + F - (1 + c) with the outer face counted and c = skeleton components.

    A connected cellulation of the sphere has V - E + F = 2; each additional
    disconnected boundary component (an island inside a face) adds one.
    """
    if cx.n_edges == 0:
        return int(cx.n_vertices - cx.n_edges + cx.n_faces - 1)
    c = cx.skeleton_components()
    return int(cx.n_vertices - cx.n_edges + cx.n_faces - (1 + c))


def validate(cx: HalfEdgeComplex) -> ValidationReport:
    msgs: List[str] = []
    B1, B2 = boundary_matrices(cx, include_outer=True)
    prod = B1 @ B2
    prod.eliminate_zeros()
    b1b2_zero = prod.nnz == 0
    if not b1b2_zero:
        msgs.append(f"B1@B2 has {prod.nnz} nonzeros")

    counts = np.asarray(abs(B2).sum(axis=1)).ravel() if cx.n_edges else np.zeros(0)
    every_edge_two_faces = bool(np.all(counts == 2)) if cx.n_edges else True
    if not every_edge_two_faces:
        msgs.append(f"{int(np.sum(counts != 2))} edges do not have exactly two face incidences")

    he_face = cx.he_face_array()
    loops_match = True
    covered = np.zeros(cx.n_half_edges, dtype=bool)
    for f in range(cx.n_faces):
        for loop in cx.face_loops[f]:
            for h in loop:
                covered[h] = True
                if he_face[h] != f:
                    loops_match = False
    loops_cover = bool(covered.all())
    if not loops_match:
        msgs.append("a face loop contains a half-edge whose left face differs")
    if not loops_cover:
        msgs.append("some half-edges are not on any face loop")

    # Vertex link: the outgoing half-edges in cyclic order must chain faces
    # consistently (face left of h == face right of the next half-edge CCW).
    links_ok = True
    degs = cx.vertex_degrees()
    for v in range(cx.n_vertices):
        outs = cx.vertex_out_half_edges(v)
        k = len(outs)
        for i, h in enumerate(outs):
            h_next_ccw = outs[(i + 1) % k]
            if cx.he_face(h_next_ccw ^ 1) != cx.he_face(h):
                links_ok = False
                break
        if not links_ok:
            msgs.append(f"vertex {v} has an inconsistent link")
            break

    eul = euler_residual(cx)
    if eul != 0:
        msgs.append(f"Euler residual {eul}")

    hist = {int(d): int(n) for d, n in zip(*np.unique(degs, return_counts=True))} if cx.n_vertices else {}
    ok = b1b2_zero and every_edge_two_faces and loops_match and loops_cover and links_ok and eul == 0
    return ValidationReport(
        ok=ok,
        b1b2_zero=b1b2_zero,
        every_edge_two_faces=every_edge_two_faces,
        loops_match_edge_faces=loops_match,
        loops_cover_all_half_edges=loops_cover,
        vertex_links_ok=links_ok,
        euler_residual=eul,
        n_vertices=cx.n_vertices,
        n_edges=cx.n_edges,
        n_faces=cx.n_faces,
        n_cells=int(cx.cell_faces.size),
        n_gaps=int(cx.gap_faces.size),
        n_boundary_edges=int(cx.boundary_edges().size) if cx.n_edges else 0,
        n_artificial_vertices=int(np.sum(cx.vertex_kind == VertexKind.ARTIFICIAL)),
        skeleton_components=cx.skeleton_components() if cx.n_edges else 0,
        degree_histogram=hist,
        messages=msgs,
    )
