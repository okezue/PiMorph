"""Exact topological identities used as tracker and reconstruction QC checks.

These follow from the representation alone. They are not biological laws.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .halfedge import FaceKind, HalfEdgeComplex


def euler_characteristic(cx: HalfEdgeComplex, include_outer: bool = False) -> int:
    """V - E + F. Without the outer face this is chi of the cropped disk (1 when
    the skeleton is connected)."""
    F = cx.n_faces if include_outer else cx.n_faces - 1
    return int(cx.n_vertices - cx.n_edges + F)


def face_side_counts(cx: HalfEdgeComplex, faces: Optional[Iterable[int]] = None) -> np.ndarray:
    faces = range(cx.n_faces) if faces is None else faces
    return np.array([cx.face_sides(int(f)) for f in faces], dtype=np.int64)


def defect_law_residual(cx: HalfEdgeComplex) -> int:
    """Residual of  sum_f (6 - n_f) = 6 chi + 2 sum_v (d_v - 3) + E_boundary.

    Faces are cells and gaps (outer excluded), n_f counts edge occurrences around
    face f over all its loops, chi = V - E + F_interior, d_v is the vertex degree and
    E_boundary the number of edges incident to the outer face. Must be 0 for any valid
    complex because it is a rearrangement of two handshake identities.
    """
    interior = [f for f in range(cx.n_faces) if cx.face_kind[f] != FaceKind.OUTER]
    n_f = face_side_counts(cx, interior)
    lhs = int(np.sum(6 - n_f))
    chi = euler_characteristic(cx, include_outer=False)
    degs = cx.vertex_degrees()
    e_bd = int(cx.boundary_edges().size) if cx.n_edges else 0
    rhs = 6 * chi + 2 * int(np.sum(degs - 3)) + e_bd
    return lhs - rhs


def topological_charge(cx: HalfEdgeComplex) -> Dict[int, int]:
    """q_f = 6 - n_f for every cell face, counting cell-cell and cell-gap sides."""
    return {int(f): int(6 - cx.face_sides(int(f))) for f in cx.cell_faces}


def weaire_sum_rule_residual(cx: HalfEdgeComplex, cells_only: bool = True) -> float:
    """Residual of  sum_n p_n n m_n = <n^2>  over cell faces.

    Neighbors are counted by physical interface incidence (with multiplicity); with
    ``cells_only`` the side count n_f only counts sides shared with other cells so the
    directed neighbor sum closes. Exact up to floating point.
    """
    cells = cx.cell_faces
    if cells.size == 0:
        return 0.0
    n = {}
    nb = {}
    for f in cells:
        f = int(f)
        neigh = cx.face_neighbors(f, kinds=(FaceKind.CELL,))
        n[f] = len(neigh) if cells_only else cx.face_sides(f)
        nb[f] = neigh
    ns = np.array([n[int(f)] for f in cells], dtype=np.float64)
    # left: sum over cells of n_f * mean side count of neighbors = sum over directed pairs of n_neighbor
    lhs = 0.0
    for f in cells:
        f = int(f)
        for g in nb[f]:
            lhs += n[int(g)]
    lhs /= len(cells)
    rhs = float(np.mean(ns**2))
    return float(lhs - rhs)


def t1_charge_delta(n_before: Dict[int, int], n_after: Dict[int, int]) -> int:
    """Total change of topological charge over the faces present in both dicts.
    Zero for a generic T1 exchange."""
    common = set(n_before) & set(n_after)
    return int(sum((6 - n_after[f]) - (6 - n_before[f]) for f in common))
