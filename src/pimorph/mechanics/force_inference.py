"""Bayesian-style force inference from equilibrium geometry.

Unknowns are the line tensions gamma_e of every edge bordering a cell and the
pressures p_f of the cell faces (gap and outer faces sit at the reference pressure
0). Each interior vertex v of degree >= 3 gives two equations

    sum_h [ gamma_e t_h + (L_e / 2) (p_left(h) - p_right(h)) n_h ] = 0

over its outgoing half-edges h, with t_h the unit chord from v to the far vertex
and n_h its right normal. The pressure term is the shoelace derivative of the cell
areas, i.e. the exact vertex-model balance for straight edges (Ishihara and
Sugimura 2012; Chiou et al. 2012). For curved edges it equals, to first order in
kappa L, the CellFIT balance written with the true end tangents, so chord tangents
are used throughout. With ``curvature_pressure`` the Laplace law
``p_left - p_right = gamma_e kappa_e`` of the smoothed traces (``edge_geometry``) is
added per edge with weight L_e (Brodland et al. 2014), skipping arcs with
|kappa| L > 1.5. Complexes whose traces are straight carry no curvature
information: there the Laplace rows read p_left = p_right and must be switched off.

The system is homogeneous, so the global scale is a gauge: mean tension = 1.
Without curvature the vertex balance alone has fewer equations than unknowns once
pressures are included (Ishihara's counting plus boundary losses), so pressures
are then determined only by the ridge prior; with curvature the system is
overdetermined and pressures are identifiable up to the reference. Results are
RELATIVE tensions and pressures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pimorph.complex.halfedge import FaceKind, HalfEdgeComplex


@dataclass
class ForceInferenceResult:
    tensions: np.ndarray  # (E,) relative tension, NaN where not inferred
    pressures: np.ndarray  # (F,) relative pressure in tension / pixel, NaN off cells
    vertex_residuals: np.ndarray  # (V,) |force imbalance| in tension units, NaN off interior
    condition_number: float
    n_equations: int
    n_unknowns: int
    rank: int
    edge_ids: np.ndarray  # edges carrying a tension unknown
    interior_vertices: np.ndarray
    tension_identifiable: np.ndarray  # (E,) bool: edge appears in a vertex equation
    curvatures: Optional[np.ndarray] = None  # (E,) signed mean curvature used, if any
    residual_rms: float = float("nan")
    settings: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- geometry helpers
def edge_curvatures(cx: HalfEdgeComplex) -> np.ndarray:
    """Signed mean curvature per edge from the smoothed trace (1 / pixel).

    Least-squares fit of the perpendicular deviation from the chord to the parabola
    ``s (c - s) kappa / 2``, which is the small-sagitta form of a circular arc through
    both vertices. Sign follows ``polyline_curvature``: positive when the trace turns
    counter-clockwise on screen travelling tail -> head.
    """
    out = np.zeros(cx.n_edges, dtype=np.float64)
    for e in range(cx.n_edges):
        p = cx.edge_geometry(e)
        if p.shape[0] < 3:
            continue
        a, b = p[0], p[-1]
        chord = b - a
        c = float(np.hypot(*chord))
        if c < 1e-9:
            continue
        t = chord / c
        rel = p[1:-1] - a
        s = rel @ t
        # left of travel on screen is (-t_col, t_row) in (row, col) components
        left = np.array([-t[1], t[0]])
        d = rel @ left
        w = 0.5 * s * (c - s)
        ww = float(np.dot(w, w))
        if ww < 1e-12:
            continue
        k0 = float(-np.dot(w, d) / ww)
        if abs(k0) * c > 0.05:
            k0 = _refine_arc_curvature(s, d, c, k0)
        out[e] = k0
    return out


def _arc_deviation(s: np.ndarray, c: float, kappa: float) -> np.ndarray:
    """Left deviation from the chord of the circular arc with curvature kappa."""
    R = 1.0 / abs(kappa)
    dev = np.sqrt(np.maximum(R**2 - (s - 0.5 * c) ** 2, 0.0)) - np.sqrt(max(R**2 - 0.25 * c**2, 0.0))
    return -np.sign(kappa) * dev


def _refine_arc_curvature(s: np.ndarray, d: np.ndarray, c: float, k0: float) -> float:
    """Least-squares curvature of the exact circular arc through both chord ends."""
    from scipy.optimize import minimize_scalar

    kmax = 1.98 / c

    def cost(k: float) -> float:
        if abs(k) < 1e-12:
            return float(np.dot(d, d))
        r = d - _arc_deviation(s, c, k)
        return float(np.dot(r, r))

    lo, hi = (0.0, kmax) if k0 > 0 else (-kmax, 0.0)
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 1e-9 / c})
    return float(res.x) if res.success and cost(res.x) <= cost(k0) else k0


def _interior_vertices(cx: HalfEdgeComplex) -> np.ndarray:
    outer = cx.outer_face
    touch = np.zeros(cx.n_vertices, dtype=bool)
    for e in cx.boundary_edges():
        touch[cx.edge_tail[e]] = True
        touch[cx.edge_head[e]] = True
    for loop in cx.face_loops[outer]:
        for h in loop:
            touch[cx.he_origin(h)] = True
    deg = cx.vertex_degrees()
    return np.flatnonzero(~touch & (deg >= 3))


def _tension_edges(cx: HalfEdgeComplex) -> np.ndarray:
    k = cx.face_kind[cx.edge_faces]
    has_cell = np.any(k == int(FaceKind.CELL), axis=1)
    return np.flatnonzero(has_cell & (cx.edge_tail != cx.edge_head))


class _Design:
    """Force-balance design matrix in dimensionless unknowns (gamma, p * ell)."""

    def __init__(self, cx: HalfEdgeComplex, use_pressures: bool, curvature_pressure: bool, max_kappa_L: float = 1.5):
        self.cx = cx
        self.use_pressures = bool(use_pressures)
        self.curvature_pressure = bool(curvature_pressure and use_pressures)
        self.edge_ids = _tension_edges(cx)
        self.cells = cx.cell_faces if self.use_pressures else np.zeros(0, dtype=np.int64)
        self.interior = _interior_vertices(cx)
        self.n_t = int(self.edge_ids.size)
        self.n_p = int(self.cells.size)
        self.n_unknowns = self.n_t + self.n_p
        col_t = np.full(cx.n_edges, -1, dtype=np.int64)
        col_t[self.edge_ids] = np.arange(self.n_t)
        col_p = np.full(cx.n_faces, -1, dtype=np.int64)
        col_p[self.cells] = self.n_t + np.arange(self.n_p)
        self.col_t, self.col_p = col_t, col_p

        X = cx.vertex_xy
        d = X[cx.edge_head] - X[cx.edge_tail]
        self.L = np.sqrt((d**2).sum(axis=1))
        cell_area = [abs(_polygon_area(cx, f)) for f in cx.cell_faces]
        self.ell = float(np.sqrt(np.mean(cell_area))) if cell_area else 1.0

        rows_v: List[np.ndarray] = []
        appears = np.zeros(cx.n_edges, dtype=bool)
        for v in self.interior:
            block = np.zeros((2, self.n_unknowns))
            for h in cx.vertex_out_half_edges(v):
                e = h >> 1
                u = cx.he_target(h)
                L = self.L[e]
                if L <= 0:
                    continue
                t = (X[u] - X[v]) / L
                if col_t[e] >= 0:
                    block[:, col_t[e]] += t
                    appears[e] = True
                if self.use_pressures:
                    n_right = np.array([t[1], -t[0]])
                    coef = 0.5 * L / self.ell * n_right
                    fl, fr = cx.he_face(h), cx.he_face(h ^ 1)
                    if col_p[fl] >= 0:
                        block[:, col_p[fl]] += coef
                    if col_p[fr] >= 0:
                        block[:, col_p[fr]] -= coef
            rows_v.append(block)
        self.A_vertex = np.concatenate(rows_v, axis=0) if rows_v else np.zeros((0, self.n_unknowns))
        self.tension_identifiable = appears

        self.curvatures: Optional[np.ndarray] = None
        self.A_laplace = np.zeros((0, self.n_unknowns))
        if self.curvature_pressure:
            kappa = edge_curvatures(cx)
            self.curvatures = kappa
            rows_l = []
            for e in self.edge_ids:
                if not appears[e] or abs(kappa[e]) * self.L[e] > max_kappa_L:
                    continue
                fl, fr = cx.edge_faces[e]
                if col_p[fl] < 0 and col_p[fr] < 0:
                    continue
                row = np.zeros(self.n_unknowns)
                # (p_left - p_right - kappa gamma) * L, in dimensionless pressure units
                w = self.L[e] / self.ell
                if col_p[fl] >= 0:
                    row[col_p[fl]] += w
                if col_p[fr] >= 0:
                    row[col_p[fr]] -= w
                row[col_t[e]] -= kappa[e] * self.ell * w
                rows_l.append(row)
            if rows_l:
                self.A_laplace = np.stack(rows_l, axis=0)

    @property
    def A(self) -> np.ndarray:
        return np.concatenate([self.A_vertex, self.A_laplace], axis=0)

    def constant_pressure_is_null(self) -> bool:
        if self.n_p == 0:
            return False
        A = self.A
        v = np.zeros(self.n_unknowns)
        v[self.n_t :] = 1.0
        scale = np.abs(A).sum() / max(A.size, 1)
        return bool(np.linalg.norm(A @ v) < 1e-9 * max(scale, 1e-300) * np.sqrt(self.n_p))


def _polygon_area(cx: HalfEdgeComplex, f: int) -> float:
    total = 0.0
    for loop in cx.face_loops[f]:
        vs = np.array([cx.he_origin(h) for h in loop])
        if vs.size < 3:
            continue
        P = cx.vertex_xy[vs]
        r, c = P[:, 0], P[:, 1]
        total += 0.5 * float(np.dot(r, np.roll(c, -1)) - np.dot(np.roll(r, -1), c))
    return total


# ---------------------------------------------------------------------- inference
def infer_tensions(
    cx: HalfEdgeComplex,
    use_pressures: bool = True,
    ridge: float = 1e-3,
    curvature_pressure: bool = True,
    gauge_weight: float = 1e3,
) -> ForceInferenceResult:
    """Least-squares force inference with gauge rows and a ridge prior.

    Gauge: mean tension over identifiable edges = 1; mean cell pressure = 0 when the
    constant offset is a null direction (otherwise gap/outer faces set p = 0). The
    ridge pulls tensions towards 1 and dimensionless pressures towards 0.
    """
    des = _Design(cx, use_pressures, curvature_pressure)
    A = des.A
    n_eq = A.shape[0]
    n_unk = des.n_unknowns
    ident = des.tension_identifiable[des.edge_ids]
    n_ident = int(ident.sum())
    if n_unk == 0 or n_ident == 0:
        return ForceInferenceResult(
            tensions=np.full(cx.n_edges, np.nan),
            pressures=np.full(cx.n_faces, np.nan),
            vertex_residuals=np.full(cx.n_vertices, np.nan),
            condition_number=float("nan"),
            n_equations=int(n_eq),
            n_unknowns=int(n_unk),
            rank=0,
            edge_ids=des.edge_ids,
            interior_vertices=des.interior,
            tension_identifiable=des.tension_identifiable,
            curvatures=des.curvatures,
            settings={"use_pressures": use_pressures, "ridge": ridge, "curvature_pressure": curvature_pressure},
        )

    rows = [A]
    rhs = [np.zeros(n_eq)]
    # gauge: mean identifiable tension = 1
    g = np.zeros(n_unk)
    g[np.flatnonzero(ident)] = 1.0 / n_ident
    w_t = gauge_weight * np.sqrt(n_ident)
    rows.append((w_t * g)[None, :])
    rhs.append(np.array([w_t]))
    pressure_gauge = "none"
    if des.n_p and des.constant_pressure_is_null():
        gp = np.zeros(n_unk)
        gp[des.n_t :] = 1.0 / des.n_p
        w_p = gauge_weight * np.sqrt(des.n_p)
        rows.append((w_p * gp)[None, :])
        rhs.append(np.array([0.0]))
        pressure_gauge = "mean_zero"
    elif des.n_p:
        pressure_gauge = "outer_zero"
    # ridge prior: gamma -> 1, dimensionless pressure -> 0
    sr = np.sqrt(max(float(ridge), 0.0))
    if sr > 0:
        rows.append(sr * np.eye(n_unk))
        prior = np.zeros(n_unk)
        prior[: des.n_t] = 1.0
        rhs.append(sr * prior)
    M = np.concatenate(rows, axis=0)
    b = np.concatenate(rhs)
    x, _, rank_full, sv = np.linalg.lstsq(M, b, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
    rank_A = int(np.linalg.matrix_rank(A)) if A.size else 0

    tensions = np.full(cx.n_edges, np.nan)
    tensions[des.edge_ids[ident]] = x[: des.n_t][ident]
    pressures = np.full(cx.n_faces, np.nan)
    if des.n_p:
        pressures[des.cells] = x[des.n_t :] / des.ell
    res = np.full(cx.n_vertices, np.nan)
    if des.interior.size:
        r = (des.A_vertex @ x).reshape(-1, 2)
        res[des.interior] = np.sqrt((r**2).sum(axis=1))
    rms = float(np.sqrt(np.nanmean(res[des.interior] ** 2))) if des.interior.size else float("nan")
    return ForceInferenceResult(
        tensions=tensions,
        pressures=pressures,
        vertex_residuals=res,
        condition_number=cond,
        n_equations=int(n_eq),
        n_unknowns=int(n_unk),
        rank=rank_A,
        edge_ids=des.edge_ids,
        interior_vertices=des.interior,
        tension_identifiable=des.tension_identifiable,
        curvatures=des.curvatures,
        residual_rms=rms,
        settings={
            "use_pressures": bool(use_pressures),
            "ridge": float(ridge),
            "curvature_pressure": bool(des.curvature_pressure),
            "pressure_gauge": pressure_gauge,
            "tension_gauge": "mean_one",
            "n_vertex_equations": int(des.A_vertex.shape[0]),
            "n_laplace_equations": int(des.A_laplace.shape[0]),
            "n_identifiable_tensions": n_ident,
            "ell": des.ell,
        },
    )


def inferred_stress_tensor(cx: HalfEdgeComplex, result: ForceInferenceResult) -> np.ndarray:
    """Per-face Batchelor stress (F, 2, 2) in (row, col) components, NaN off cells.

    sigma_f = -p_f I + (1 / 2 A_f) sum_{e in boundary(f)} gamma_e l_e l_e^T / |l_e|
    with chord vectors l_e; the 1/2 shares each edge between its two faces.
    Missing pressures count as 0 and missing tensions are skipped.
    """
    out = np.full((cx.n_faces, 2, 2), np.nan)
    X = cx.vertex_xy
    for f in cx.cell_faces:
        area = abs(_polygon_area(cx, f))
        if area <= 0:
            continue
        s = np.zeros((2, 2))
        for e in cx.face_edges(f):
            g = result.tensions[e]
            if not np.isfinite(g):
                continue
            d = X[cx.edge_head[e]] - X[cx.edge_tail[e]]
            L = float(np.hypot(*d))
            if L <= 0:
                continue
            s += g * np.outer(d, d) / L
        p = result.pressures[f]
        p = 0.0 if not np.isfinite(p) else float(p)
        out[f] = -p * np.eye(2) + s / (2.0 * area)
    return out


# ---------------------------------------------------------------- identifiability
def identifiability_report(
    cx: HalfEdgeComplex,
    use_pressures: bool = True,
    curvature_pressure: bool = False,
    rank_tol: float = 1e-7,
) -> Dict[str, Any]:
    """Rank analysis of the raw force-balance design matrix (no gauge, no prior)."""
    des = _Design(cx, use_pressures, curvature_pressure)
    A = des.A
    n_unk = des.n_unknowns
    if A.size == 0 or n_unk == 0:
        sv = np.zeros(0)
        rank = 0
        null_dim = n_unk
        Vh = np.zeros((0, n_unk))
    else:
        _, sv, Vh = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(sv > rank_tol * sv[0])) if sv.size else 0
        null_dim = n_unk - rank
    null_basis = Vh[rank:].T if null_dim > 0 else np.zeros((n_unk, 0))

    ident = des.tension_identifiable[des.edge_ids]
    n_free_edges = int(np.sum(~ident))
    const_p_null = des.constant_pressure_is_null()
    pressure_in_null = False
    if des.n_p and null_dim:
        block = null_basis[des.n_t :]
        # remove the constant direction, then test for remaining pressure freedom
        block = block - block.mean(axis=0, keepdims=True)
        pressure_in_null = bool(np.linalg.norm(block) > 1e-8)

    statements = [
        "Force balance is homogeneous in (tension, pressure): only relative values are identifiable. "
        "Gauge: mean tension = 1.",
    ]
    if not use_pressures:
        statements.append("Pressures are not modelled; tensions absorb pressure forces.")
    elif not des.curvature_pressure:
        statements.append(
            "Pressures are constrained only by the chord pressure terms of the vertex balance and are the "
            "weakest-determined unknowns; a constant pressure offset is "
            + ("a null direction (gauge: mean pressure = 0)." if const_p_null else "fixed by gap/outer faces at 0.")
            + " Pressure differences are physically encoded in edge curvature (Laplace law); enable "
            "curvature_pressure on complexes with curved traces."
        )
    else:
        statements.append(
            "Pressures are constrained by the vertex balance and by the Laplace law of the smoothed traces; "
            + (
                "a constant offset remains a gauge (mean pressure = 0)."
                if const_p_null
                else "the reference is 0 outside."
            )
        )
    if n_free_edges:
        statements.append(
            f"{n_free_edges} tension unknowns appear in no vertex equation (edges between boundary or low-degree "
            "vertices) and are reported as NaN."
        )
    extra = null_dim - (1 if null_dim else 0) - (1 if (const_p_null and null_dim > 1) else 0) - n_free_edges
    if extra > 0:
        statements.append(f"{extra} further null directions: combinations not constrained by this geometry.")
    return {
        "n_unknowns": int(n_unk),
        "n_tensions": int(des.n_t),
        "n_pressures": int(des.n_p),
        "n_equations": int(A.shape[0]),
        "n_vertex_equations": int(des.A_vertex.shape[0]),
        "n_laplace_equations": int(des.A_laplace.shape[0]),
        "rank": int(rank),
        "null_dim": int(null_dim),
        "singular_values": sv,
        "smallest_relative_singular_value": float(sv[-1] / sv[0]) if sv.size and sv[0] > 0 else float("nan"),
        "n_unconstrained_tensions": n_free_edges,
        "constant_pressure_is_null": bool(const_p_null),
        "pressures_identifiable": bool(use_pressures and des.n_p > 0 and not pressure_in_null),
        "pressures_need_curvature": bool(use_pressures and not des.curvature_pressure),
        "use_pressures": bool(use_pressures),
        "curvature_pressure": bool(des.curvature_pressure),
        "statements": statements,
    }


def design_matrix(
    cx: HalfEdgeComplex, use_pressures: bool = True, curvature_pressure: bool = False
) -> Tuple[np.ndarray, Dict]:
    """Raw design matrix and column bookkeeping, for tests and diagnostics."""
    des = _Design(cx, use_pressures, curvature_pressure)
    return des.A, {"edge_ids": des.edge_ids, "cells": des.cells, "n_t": des.n_t, "n_p": des.n_p, "ell": des.ell}
