"""Oriented boundary matrices and exact validity checks for the 3-D complex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .extract3d import CellComplex3D, CellKind3D, LineKind3D, VertexKind3D


def boundary_matrices3d(cx: CellComplex3D) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Return (B1, B2, B3).

    B1 (V x E): -1 at the tail, +1 at the head of each 1-cell.
    B2 (E x F): signed number of traversals of 1-cell e by the oriented boundary of
    interface f (orientation by the right-hand rule with the interface normal).
    B3 (F x C): +1 for the 3-cell the normal points out of, -1 for the one it
    points into. B1 @ B2 == 0 and B2 @ B3 == 0 identically.
    """
    V, E, F, C = cx.n_vertices, cx.n_lines, cx.n_interfaces, cx.n_cells
    e_idx = np.arange(E)
    rows = np.concatenate([cx.line_tail, cx.line_head])
    cols = np.concatenate([e_idx, e_idx])
    vals = np.concatenate([-np.ones(E), np.ones(E)])
    B1 = sparse.coo_matrix((vals, (rows, cols)), shape=(V, E)).tocsr()
    B1.sum_duplicates()
    B1.eliminate_zeros()

    r2: List[np.ndarray] = []
    c2: List[np.ndarray] = []
    v2: List[np.ndarray] = []
    for f in range(F):
        b = cx.interface_boundary[f]
        if b.shape[0]:
            r2.append(b[:, 0])
            c2.append(np.full(b.shape[0], f, dtype=np.int64))
            v2.append(b[:, 1].astype(np.float64))
    if r2:
        B2 = sparse.coo_matrix((np.concatenate(v2), (np.concatenate(r2), np.concatenate(c2))), shape=(E, F)).tocsr()
    else:
        B2 = sparse.csr_matrix((E, F), dtype=np.float64)
    B2.sum_duplicates()
    B2.eliminate_zeros()

    f_idx = np.arange(F)
    r3 = np.concatenate([f_idx, f_idx])
    c3 = np.concatenate([cx.interface_cells[:, 0], cx.interface_cells[:, 1]])
    v3 = np.concatenate([np.ones(F), -np.ones(F)])
    B3 = sparse.coo_matrix((v3, (r3, c3)), shape=(F, C)).tocsr()
    B3.sum_duplicates()
    B3.eliminate_zeros()
    return B1, B2, B3


def skeleton_components3d(cx: CellComplex3D) -> int:
    """Connected components of the 2-skeleton: vertices, lines and interfaces linked
    by geometric incidence (a line touching an interface links them even when its
    boundary coefficient cancels), including artificial cells."""
    V, E, F = cx.n_vertices, cx.n_lines, cx.n_interfaces
    n = V + E + F
    if n == 0:
        return 0
    rows = [np.arange(E) + V, np.arange(E) + V]
    cols = [cx.line_tail, cx.line_head]
    for e in range(E):
        fs = cx.line_interfaces[e]
        if fs.size:
            rows.append(np.full(fs.size, V + e, dtype=np.int64))
            cols.append(fs + V + E)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    a = sparse.coo_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    return int(np.unique(lab[V + E :]).size) if F else 0


def occupied_set_euler(cx: CellComplex3D) -> Tuple[int, int]:
    """(chi, islands) of the closed voxel set S occupied by the bounded 3-cells.

    chi is the exact Euler characteristic of the cubical complex of S (all corners,
    edges, faces and voxels touching a non-outer voxel); islands is the number of
    26-connected components of S. Cheap and independent of the coarse complex."""
    m = np.pad(cx.cell_map != cx.outer_cell, 1)
    if not m.any():
        return 0, 0
    n_vox = int(np.count_nonzero(m))
    n_face = 0
    n_edge = 0
    for a in range(3):
        lo = np.take(m, np.arange(m.shape[a] - 1), axis=a)
        hi = np.take(m, np.arange(1, m.shape[a]), axis=a)
        n_face += int(np.count_nonzero(lo | hi))
        b, c = [ax for ax in range(3) if ax != a]
        e = m
        for ax in (b, c):
            e = np.take(e, np.arange(e.shape[ax] - 1), axis=ax) | np.take(e, np.arange(1, e.shape[ax]), axis=ax)
        n_edge += int(np.count_nonzero(e))
    v = m
    for ax in range(3):
        v = np.take(v, np.arange(v.shape[ax] - 1), axis=ax) | np.take(v, np.arange(1, v.shape[ax]), axis=ax)
    n_corner = int(np.count_nonzero(v))
    _, islands = ndi.label(m, structure=np.ones((3, 3, 3), dtype=bool))
    return n_corner - n_edge + n_face - n_vox, int(islands)


@dataclass
class Validation3DReport:
    ok: bool
    b1b2_zero: bool
    b2b3_zero: bool
    every_interface_two_cells: bool
    loops_match_boundary: bool
    every_line_has_interface: bool
    euler_residual: int
    euler_expected: int
    euler_occupied_set: int
    n_islands: int
    n_vertices: int
    n_lines: int
    n_interfaces: int
    n_cells: int
    n_gaps: int
    n_triple_lines: int
    n_quadruple_points: int
    n_artificial_vertices: int
    n_artificial_lines: int
    n_boundary_interfaces: int
    skeleton_components: int
    n_extra_loops: int
    degree_histogram: Dict[int, int] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["degree_histogram"] = {int(k): int(v) for k, v in self.degree_histogram.items()}
        return d


def euler_expected3d(cx: CellComplex3D) -> int:
    """Expected V - E + F - C (outer cell included).

    The bounded cells cover the closed voxel set S with exact Euler characteristic
    chi(S) (a single ball-like island gives 1, so with the outer cell the total is
    0, chi of S^3). Two corrections make the count exact for cellulations by balls:
    every 2-skeleton component beyond the islands of S is the surface of a cavity
    (+1, the enclosing cell is not a ball), and every extra boundary component of a
    planar interface (an annular contact, a slit pinch) is a hole (+1). Handles in
    cells or interfaces are not corrected and show up as a nonzero residual.
    """
    chi_s, islands = occupied_set_euler(cx)
    k = skeleton_components3d(cx)
    extra_loops = sum(max(len(loops), 1) - 1 for loops in cx.interface_loops)
    return int(chi_s - 1 + (k - islands) + extra_loops)


def euler_residual3d(cx: CellComplex3D) -> int:
    """V - E + F - C minus its expected value; 0 when every cell is a ball (with
    ball-like cavities), every interface planar and every line an arc or circle."""
    chi = cx.n_vertices - cx.n_lines + cx.n_interfaces - cx.n_cells
    return int(chi - euler_expected3d(cx))


def validate3d(cx: CellComplex3D) -> Validation3DReport:
    msgs: List[str] = []
    B1, B2, B3 = boundary_matrices3d(cx)
    p12 = B1 @ B2
    p12.eliminate_zeros()
    b1b2_zero = p12.nnz == 0
    if not b1b2_zero:
        msgs.append(f"B1@B2 has {p12.nnz} nonzeros")
    p23 = B2 @ B3
    p23.eliminate_zeros()
    b2b3_zero = p23.nnz == 0
    if not b2b3_zero:
        msgs.append(f"B2@B3 has {p23.nnz} nonzeros")

    F = cx.n_interfaces
    if F:
        counts = np.asarray(abs(B3).sum(axis=1)).ravel()
        distinct = cx.interface_cells[:, 0] != cx.interface_cells[:, 1]
        two_cells = bool(np.all(counts == 2) and np.all(distinct))
    else:
        two_cells = True
    if not two_cells:
        msgs.append("an interface does not separate exactly two distinct 3-cells")

    # Loop decomposition must reproduce the boundary coefficients.
    loops_match = True
    for f in range(F):
        acc: Dict[int, int] = {}
        for loop in cx.interface_loops[f]:
            for h in loop:
                acc[h >> 1] = acc.get(h >> 1, 0) + (1 if (h & 1) == 0 else -1)
        want = {int(e): int(c) for e, c in cx.interface_boundary[f]}
        acc = {e: c for e, c in acc.items() if c != 0}
        if acc != want:
            loops_match = False
            msgs.append(f"interface {f}: loops do not reproduce the boundary chain")
            break

    has_if = all(len(v) > 0 for v in cx.line_interfaces) if cx.n_lines else True
    if not has_if:
        msgs.append("a 1-cell has no incident interface")

    chi_s, islands = occupied_set_euler(cx)
    expected = euler_expected3d(cx)
    eul = int(cx.n_vertices - cx.n_lines + cx.n_interfaces - cx.n_cells - expected)
    if eul != 0:
        msgs.append(f"Euler residual {eul}")

    degs = cx.vertex_degrees()
    hist = {int(d): int(n) for d, n in zip(*np.unique(degs, return_counts=True))} if cx.n_vertices else {}
    outer = cx.outer_cell
    ok = b1b2_zero and b2b3_zero and two_cells and loops_match and has_if and eul == 0
    return Validation3DReport(
        ok=ok,
        b1b2_zero=b1b2_zero,
        b2b3_zero=b2b3_zero,
        every_interface_two_cells=two_cells,
        loops_match_boundary=loops_match,
        every_line_has_interface=has_if,
        euler_residual=eul,
        euler_expected=expected,
        euler_occupied_set=int(chi_s),
        n_islands=int(islands),
        n_vertices=cx.n_vertices,
        n_lines=cx.n_lines,
        n_interfaces=F,
        n_cells=int(cx.cells_of_kind(CellKind3D.CELL).size),
        n_gaps=int(cx.cells_of_kind(CellKind3D.GAP).size),
        n_triple_lines=int(np.sum(cx.line_kind == LineKind3D.REGULAR)),
        n_quadruple_points=int(np.sum(cx.vertex_kind == VertexKind3D.REGULAR)),
        n_artificial_vertices=int(np.sum(cx.vertex_kind == VertexKind3D.ARTIFICIAL)),
        n_artificial_lines=int(np.sum(cx.line_kind == LineKind3D.ARTIFICIAL)),
        n_boundary_interfaces=int(np.sum(cx.interface_cells[:, 1] == outer)) if F else 0,
        skeleton_components=skeleton_components3d(cx),
        n_extra_loops=int(sum(max(len(loops), 1) - 1 for loops in cx.interface_loops)),
        degree_histogram=hist,
        messages=msgs,
    )
