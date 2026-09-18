"""Forward vertex model on a fixed topology: energy, analytic forces, relaxation.

Within one topology the vertex positions follow overdamped mechanics driven by

    E = sum_cells [K_A/2 (A_f - A0_f)^2 + K_P/2 (P_f - P0_f)^2] + sum_edges gamma_e L_e

with straight edges between vertices. ``K_A`` and ``K_P`` are the dimensionless
moduli of the standard vertex-model normalization (lengths in units of
``ell = sqrt(mean A0)``, tensions in units of ``gamma_scale``); internally they are
converted to per-pixel moduli ``K_A / ell**3`` and ``K_P / ell`` so that the energy
is in tension * pixel units and forces are in tension units. The cell pressure is
``p_f = -K_A_px (A_f - A0_f)`` and the effective tension of an edge is
``gamma_e + K_P_px sum_{f ∋ e} (P_f - P0_f)``, which is what force inference sees.

By default the input tessellation is the elastic reference state (per-cell A0 and
P0). An optional short-range core (``core_length`` pixels) keeps edges that would
undergo a T1 from collapsing, since the topology is fixed; it enters the effective
tension of those edges only. Vertices touching the outer face are pinned. Gap faces
and the outer face carry no elastic energy; their edges still carry a line tension.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from pimorph.complex.extract import rebuild_topology
from pimorph.complex.halfedge import FaceKind, HalfEdgeComplex


@dataclass
class VertexModelParams:
    K_A: float = 1.0
    A0: Union[float, np.ndarray, None] = None  # None: initial area of each face; array: per face
    K_P: float = 0.1
    P0: Union[float, np.ndarray, None] = None  # None: initial perimeter of each face; array: per face
    gamma_scale: float = 1.0
    tension_dispersion: float = 0.3
    boundary_tension: float = 1.0  # multiple of gamma_scale for cell-outer edges
    gap_tension: float = 1.0  # multiple of gamma_scale for edges bordering a gap
    # Short-range core K_core/(2 ell) sum_e (core_length - L_e)_+^2 (pixels) that keeps
    # edges from collapsing at fixed topology; enters the effective tension.
    core_length: float = 0.0
    core_stiffness: float = 50.0
    seed: int = 0


class PolygonModel:
    """Straight-polygon view of a complex: loop tables and elastic reference state.

    Built once per topology; all quantities are functions of ``vertex_xy`` only.
    """

    def __init__(self, cx: HalfEdgeComplex, params: VertexModelParams):
        self.params = params
        self.n_vertices = cx.n_vertices
        self.n_edges = cx.n_edges
        self.n_faces = cx.n_faces
        self.edge_tail = cx.edge_tail.copy()
        self.edge_head = cx.edge_head.copy()
        self.is_cell = cx.face_kind == int(FaceKind.CELL)

        lp_face, lp_v, lp_next, lp_prev, lp_edge, lp_loop = [], [], [], [], [], []
        loop_id = 0
        for f in range(cx.n_faces):
            for loop in cx.face_loops[f]:
                n = len(loop)
                for i, h in enumerate(loop):
                    lp_face.append(f)
                    lp_v.append(cx.he_origin(h))
                    lp_next.append(cx.he_origin(loop[(i + 1) % n]))
                    lp_prev.append(cx.he_origin(loop[(i - 1) % n]))
                    lp_edge.append(h >> 1)
                    lp_loop.append(loop_id)
                loop_id += 1
        self.lp_face = np.asarray(lp_face, dtype=np.int64)
        self.lp_v = np.asarray(lp_v, dtype=np.int64)
        self.lp_next = np.asarray(lp_next, dtype=np.int64)
        self.lp_prev = np.asarray(lp_prev, dtype=np.int64)
        self.lp_edge = np.asarray(lp_edge, dtype=np.int64)
        self.lp_loop = np.asarray(lp_loop, dtype=np.int64)
        self.n_loops = loop_id
        self.loop_size = np.bincount(self.lp_loop, minlength=self.n_loops) if self.n_loops else np.zeros(0, np.int64)
        self.elastic_lp = self.is_cell[self.lp_face] if self.lp_face.size else np.zeros(0, bool)

        outer = cx.outer_face
        touch = np.zeros(cx.n_vertices, dtype=bool)
        for e in cx.boundary_edges():
            touch[cx.edge_tail[e]] = True
            touch[cx.edge_head[e]] = True
        # vertices whose link includes the outer face (also via loops through it)
        if self.lp_face.size:
            touch[self.lp_v[self.lp_face == outer]] = True
        self.free = ~touch

        # outgoing half-edges grouped by vertex in the original cyclic order
        rot_he, rot_next = [], []
        for v in range(cx.n_vertices):
            outs = cx.vertex_out_half_edges(v)
            k = len(outs)
            base = len(rot_he)
            rot_he.extend(outs)
            rot_next.extend(base + ((i + 1) % k) for i in range(k))
        self.rot_he = np.asarray(rot_he, dtype=np.int64)
        self.rot_next = np.asarray(rot_next, dtype=np.int64)
        self.rot_vertex = np.repeat(np.arange(cx.n_vertices), cx.vertex_degrees()) if rot_he else np.zeros(0, np.int64)
        self.rot_origin = np.where(self.rot_he & 1, self.edge_head[self.rot_he >> 1], self.edge_tail[self.rot_he >> 1])
        self.rot_target = np.where(self.rot_he & 1, self.edge_tail[self.rot_he >> 1], self.edge_head[self.rot_he >> 1])
        # vertices of degree >= 3 without loop edges take part in the order check
        has_loop = np.zeros(cx.n_vertices, dtype=bool)
        has_loop[self.rot_vertex[self.rot_origin == self.rot_target]] = True
        self.rot_checked = (cx.vertex_degrees() >= 3) & ~has_loop

        X = cx.vertex_xy
        A, P, _ = self.areas_perimeters(X)
        cells = np.flatnonzero(self.is_cell)
        # None: the input tessellation is the elastic reference state (per-cell A0, P0).
        # A single mean target drives fixed-topology sheets into blocked T1/T2 events.
        if params.A0 is None:
            A0 = A.copy()
        elif np.ndim(params.A0) == 0:
            A0 = np.full(cx.n_faces, float(params.A0))
        else:
            A0 = np.asarray(params.A0, dtype=np.float64).copy()
        if params.P0 is None:
            P0 = P.copy()
        elif np.ndim(params.P0) == 0:
            P0 = np.full(cx.n_faces, float(params.P0))
        else:
            P0 = np.asarray(params.P0, dtype=np.float64).copy()
        self.A0 = A0
        self.P0 = P0
        self.ell = float(np.sqrt(max(A0[cells].mean() if cells.size else 1.0, 1e-12)))
        self.K_A_px = params.K_A / self.ell**3
        self.K_P_px = params.K_P / self.ell
        self.K_core_px = params.core_stiffness / self.ell
        self.core_length = float(params.core_length)
        self.initial_loop_sign = np.sign(self.loop_areas(X))

    # ----------------------------------------------------------------- geometry
    def edge_vectors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = X[self.edge_head] - X[self.edge_tail]
        L = np.sqrt((d**2).sum(axis=1))
        return d, L

    def loop_areas(self, X: np.ndarray) -> np.ndarray:
        """Signed shoelace area per loop; positive for counter-clockwise on screen."""
        if self.lp_face.size == 0:
            return np.zeros(self.n_loops)
        r, c = X[self.lp_v, 0], X[self.lp_v, 1]
        rn, cn = X[self.lp_next, 0], X[self.lp_next, 1]
        return 0.5 * np.bincount(self.lp_loop, r * cn - rn * c, minlength=self.n_loops)

    def areas_perimeters(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Face areas (all loops, holes negative), perimeters and edge lengths."""
        _, L = self.edge_vectors(X)
        if self.lp_face.size == 0:
            return np.zeros(self.n_faces), np.zeros(self.n_faces), L
        r, c = X[self.lp_v, 0], X[self.lp_v, 1]
        rn, cn = X[self.lp_next, 0], X[self.lp_next, 1]
        A = 0.5 * np.bincount(self.lp_face, r * cn - rn * c, minlength=self.n_faces)
        P = np.bincount(self.lp_face, L[self.lp_edge], minlength=self.n_faces)
        return A, P, L

    def orientation_ok(self, X: np.ndarray) -> bool:
        """Every loop with at least three corners keeps its initial orientation and
        the cyclic order of chords around every vertex is unchanged."""
        a = self.loop_areas(X)
        mask = self.loop_size >= 3
        if not np.all(np.sign(a[mask]) == self.initial_loop_sign[mask]):
            return False
        return self.rotation_ok(X)

    def rotation_ok(self, X: np.ndarray, min_gap: float = 1e-9) -> bool:
        """Chord directions around each vertex keep their cyclic order (y-up angles)."""
        if self.rot_he.size == 0:
            return True
        d = X[self.rot_target] - X[self.rot_origin]
        ang = np.arctan2(-d[:, 0], d[:, 1])
        gap = np.mod(ang[self.rot_next] - ang, 2.0 * np.pi)
        sel = self.rot_checked[self.rot_vertex]
        if np.any(gap[sel] < min_gap):
            return False
        total = np.bincount(self.rot_vertex, gap, minlength=self.n_vertices)
        return bool(np.all(np.abs(total[self.rot_checked] - 2.0 * np.pi) < 1e-6))

    # ------------------------------------------------------------------ physics
    def pressures(self, X: np.ndarray) -> np.ndarray:
        A, _, _ = self.areas_perimeters(X)
        p = -self.K_A_px * (A - self.A0)
        p[~self.is_cell] = 0.0
        return p

    def effective_tensions(self, X: np.ndarray, gammas: np.ndarray) -> np.ndarray:
        """gamma_e plus the perimeter-elastic contributions of the incident cells."""
        _, P, _ = self.areas_perimeters(X)
        b = self.K_P_px * (P - self.P0)
        b[~self.is_cell] = 0.0
        g = np.asarray(gammas, dtype=np.float64).copy()
        if self.lp_face.size:
            g += np.bincount(self.lp_edge, b[self.lp_face], minlength=self.n_edges)
        if self.core_length > 0:
            _, L = self.edge_vectors(X)
            g -= self.K_core_px * self._core_gap(L)
        return g

    def _core_gap(self, L: np.ndarray) -> np.ndarray:
        gap = np.clip(self.core_length - L, 0.0, None)
        gap[self.edge_tail == self.edge_head] = 0.0
        return gap

    def energy(self, X: np.ndarray, gammas: np.ndarray) -> float:
        A, P, L = self.areas_perimeters(X)
        cells = self.is_cell
        e_area = 0.5 * self.K_A_px * np.sum((A[cells] - self.A0[cells]) ** 2)
        e_perim = 0.5 * self.K_P_px * np.sum((P[cells] - self.P0[cells]) ** 2)
        e_core = 0.5 * self.K_core_px * np.sum(self._core_gap(L) ** 2) if self.core_length > 0 else 0.0
        return float(e_area + e_perim + e_core + np.dot(gammas, L))

    def forces(self, X: np.ndarray, gammas: np.ndarray) -> np.ndarray:
        """-dE/dX (V, 2) in tension units per pixel of displacement."""
        d, L = self.edge_vectors(X)
        g_eff = self.effective_tensions(X, gammas)
        p = self.pressures(X)
        grad = np.zeros_like(X)
        safe = np.where(L > 0, L, 1.0)
        t = d / safe[:, None]  # unit tail -> head
        # dL_e/dx_tail = -t, dL_e/dx_head = +t
        np.add.at(grad, self.edge_tail, -g_eff[:, None] * t)
        np.add.at(grad, self.edge_head, g_eff[:, None] * t)
        if self.lp_face.size:
            # dA_f/dr_v = (c_next - c_prev)/2, dA_f/dc_v = (r_prev - r_next)/2
            a = -p[self.lp_face]  # K_A_px (A_f - A0_f), zero on non-cell faces
            dr = 0.5 * (X[self.lp_next, 1] - X[self.lp_prev, 1])
            dc = 0.5 * (X[self.lp_prev, 0] - X[self.lp_next, 0])
            np.add.at(grad[:, 0], self.lp_v, a * dr)
            np.add.at(grad[:, 1], self.lp_v, a * dc)
        return -grad


def assign_tensions(cx: HalfEdgeComplex, params: VertexModelParams) -> np.ndarray:
    """Per-edge line tensions: log-normal on cell-cell edges, fixed elsewhere."""
    rng = np.random.default_rng(params.seed)
    g = np.full(cx.n_edges, params.gamma_scale * params.boundary_tension, dtype=np.float64)
    kinds = cx.face_kind[cx.edge_faces]
    gap_edge = np.any(kinds == int(FaceKind.GAP), axis=1)
    g[gap_edge] = params.gamma_scale * params.gap_tension
    cc = cx.cell_cell_edges()
    g[cc] = params.gamma_scale * np.exp(rng.normal(0.0, params.tension_dispersion, size=cc.size))
    return g


def straighten_edges(cx: HalfEdgeComplex) -> HalfEdgeComplex:
    """Copy of ``cx`` whose edges are 2-point straight polylines between vertices.

    Loop edges (tail == head) keep their trace. The topology is rebuilt and must
    stay consistent with the straightened geometry.
    """
    out = copy.deepcopy(cx)
    out.invalidate_caches()
    poly = []
    for e in range(out.n_edges):
        t, h = int(out.edge_tail[e]), int(out.edge_head[e])
        if t == h:
            poly.append(out.edge_geometry(e).copy())
        else:
            poly.append(np.stack([out.vertex_xy[t], out.vertex_xy[h]], axis=0))
    out.edge_polyline = poly
    out.edge_smooth = None
    old_loops = [sorted(tuple(sorted(loop)) for loop in loops) for loops in out.face_loops]
    rebuild_topology(out)
    new_loops = [sorted(tuple(sorted(loop)) for loop in loops) for loops in out.face_loops]
    if old_loops != new_loops:
        raise RuntimeError("straightening edges changed the face loops; geometry is too degenerate")
    return out


def energy(cx: HalfEdgeComplex, gammas: np.ndarray, params: VertexModelParams) -> float:
    return PolygonModel(cx, params).energy(cx.vertex_xy, np.asarray(gammas, dtype=np.float64))


def vertex_forces(cx: HalfEdgeComplex, gammas: np.ndarray, params: VertexModelParams) -> np.ndarray:
    """Analytic force -dE/dx on every vertex (V, 2) in tension units; pinned vertices included."""
    return PolygonModel(cx, params).forces(cx.vertex_xy, np.asarray(gammas, dtype=np.float64))


def effective_tensions(cx: HalfEdgeComplex, gammas: np.ndarray, params: VertexModelParams) -> np.ndarray:
    """Tension actually carried by each edge: gamma_e plus perimeter-elastic terms."""
    return PolygonModel(cx, params).effective_tensions(cx.vertex_xy, np.asarray(gammas, dtype=np.float64))


def cell_pressures(cx: HalfEdgeComplex, params: VertexModelParams) -> np.ndarray:
    """p_f = -K_A_px (A_f - A0_f) on cell faces, zero elsewhere."""
    return PolygonModel(cx, params).pressures(cx.vertex_xy)


def _straight_polylines(cx: HalfEdgeComplex) -> None:
    for e in range(cx.n_edges):
        t, h = int(cx.edge_tail[e]), int(cx.edge_head[e])
        if t != h:
            cx.edge_polyline[e] = np.stack([cx.vertex_xy[t], cx.vertex_xy[h]], axis=0)


def arc_polyline(a: np.ndarray, b: np.ndarray, kappa: float, n_points: int = 9) -> np.ndarray:
    """Circular arc from a to b with signed curvature ``kappa`` (1 / pixel).

    Positive kappa turns counter-clockwise on screen along a -> b, so the arc bulges
    to the right of the chord (sign convention of ``polyline_curvature``).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    chord = b - a
    c = float(np.hypot(*chord))
    s = np.linspace(0.0, c, n_points)
    if c < 1e-12 or abs(kappa) < 1e-12:
        return a[None, :] + (s / max(c, 1e-12))[:, None] * chord[None, :]
    kmax = 1.98 / c  # stay below a semicircle
    k = float(np.clip(kappa, -kmax, kmax))
    R = 1.0 / abs(k)
    t = chord / c
    left = np.array([-t[1], t[0]])
    dev = np.sqrt(np.maximum(R**2 - (s - 0.5 * c) ** 2, 0.0)) - np.sqrt(max(R**2 - 0.25 * c**2, 0.0))
    return a[None, :] + s[:, None] * t[None, :] - np.sign(k) * dev[:, None] * left[None, :]


def render_pressure_arcs(
    cx: HalfEdgeComplex, gammas: np.ndarray, params: VertexModelParams, n_points: int = 9
) -> HalfEdgeComplex:
    """Fill ``cx.edge_smooth`` with the arcs implied by Laplace's law (in place).

    kappa_e = (p_left - p_right) / gamma_eff_e; these are the curved interfaces whose
    chords are the straight vertex-model edges. ``edge_polyline`` stays straight.
    """
    model = PolygonModel(cx, params)
    g = model.effective_tensions(cx.vertex_xy, np.asarray(gammas, dtype=np.float64))
    p = model.pressures(cx.vertex_xy)
    smooth = []
    for e in range(cx.n_edges):
        t, h = int(cx.edge_tail[e]), int(cx.edge_head[e])
        if t == h or g[e] <= 0:
            smooth.append(cx.edge_geometry(e).copy())
            continue
        fl, fr = cx.edge_faces[e]
        kappa = (p[fl] - p[fr]) / g[e]
        smooth.append(arc_polyline(cx.vertex_xy[t], cx.vertex_xy[h], kappa, n_points))
    cx.edge_smooth = smooth
    cx.provenance["smoothing"] = {"method": "pressure_arcs", "n_points": int(n_points)}
    return cx


def _contract_edge(cx: HalfEdgeComplex, e: int) -> Optional[HalfEdgeComplex]:
    """contact_death on edge e if its preconditions hold, else None."""
    from pimorph.complex.events import EventPreconditionError, EventRewriteError, contact_death

    try:
        return contact_death(cx, int(e)).cx
    except (EventPreconditionError, EventRewriteError):
        return None


def relax(
    cx: HalfEdgeComplex,
    gammas: np.ndarray,
    params: VertexModelParams,
    n_steps: int = 2000,
    dt: float = 0.05,
    tol: float = 1e-6,
    dt_max: Optional[float] = None,
    collapse_threshold: Optional[float] = None,
    arcs: bool = True,
) -> Tuple[HalfEdgeComplex, Dict[str, Any]]:
    """Gradient descent on the free vertex positions of the energy at fixed topology.

    Positions are advanced in units of ``ell`` with an Armijo-backtracked step, then
    the step is set by the Barzilai-Borwein rule. A step that would flip a loop
    orientation, reorder the edges around a vertex or raise the energy halves ``dt``
    and retries. Stops when the largest free-vertex force is below
    ``tol * gamma_scale``.

    ``collapse_threshold`` (pixels) optionally contracts cell-cell edges that shrink
    below it into a multicellular vertex (``contact_death``), which is the only
    topology change; the tension array is compacted accordingly and returned as
    ``info["gammas"]``. The returned complex has straight 2-point ``edge_polyline``
    and, with ``arcs``, the pressure-implied circular arcs in ``edge_smooth``.
    """
    gammas = np.asarray(gammas, dtype=np.float64).copy()
    out = straighten_edges(cx)
    model = PolygonModel(out, params)
    ell = model.ell
    X = out.vertex_xy / ell  # dimensionless positions
    if not model.orientation_ok(X * ell):
        raise ValueError("initial complex has a degenerate or flipped loop")
    ref_params = VertexModelParams(**{**params.__dict__, "A0": model.A0, "P0": model.P0})

    def energy_of(m: PolygonModel, Xn: np.ndarray) -> float:
        return m.energy(Xn * ell, gammas)

    def force_of(m: PolygonModel, Xn: np.ndarray) -> np.ndarray:
        F = m.forces(Xn * ell, gammas)
        F[~m.free] = 0.0
        return F

    def collapse(m: PolygonModel, Xn: np.ndarray, threshold: float) -> Optional[PolygonModel]:
        nonlocal out, gammas
        L = m.edge_vectors(Xn * ell)[1]
        short = np.flatnonzero((L < threshold) & (m.edge_tail != m.edge_head))
        for e in short[np.argsort(L[short])]:
            if not out.edge_is_cell_cell(int(e)):
                continue
            out.vertex_xy = Xn * ell
            _straight_polylines(out)
            new = _contract_edge(out, int(e))
            if new is None:
                continue
            out = new
            gammas = np.delete(gammas, int(e))
            collapsed.append(int(e))
            return PolygonModel(out, ref_params)
        return None

    force_scale = float(params.gamma_scale)
    tol_abs = tol * force_scale
    dt_max = float(dt_max) if dt_max is not None else max(20.0 * dt, 1.0)
    dt_cur = float(dt)
    E = energy_of(model, X)
    F = force_of(model, X)
    energies = [E]
    residuals = [float(np.abs(F).max()) if F.size else 0.0]
    converged = residuals[-1] < tol_abs
    n_backtracks = 0
    steps = 0
    collapsed: list = []
    while steps < n_steps and not converged:
        f2 = float((F**2).sum())
        accepted = False
        for _ in range(60):
            Xn = X + dt_cur * F
            if not model.orientation_ok(Xn * ell):
                dt_cur *= 0.5
                n_backtracks += 1
                continue
            En = energy_of(model, Xn)
            if En <= E - 1e-4 * dt_cur * f2:
                accepted = True
                break
            dt_cur *= 0.5
            n_backtracks += 1
        if not accepted:
            # blocked by the topology guard: contract the shortest edge if allowed
            new_model = collapse(model, X, np.inf) if collapse_threshold is not None else None
            if new_model is None:
                break
            model = new_model
            X = out.vertex_xy / ell
            E, F, dt_cur = energy_of(model, X), force_of(model, X), float(dt)
            continue
        steps += 1
        Fn = force_of(model, Xn)
        s = Xn - X
        y = F - Fn  # gradient difference
        sy = float((s * y).sum())
        if sy > 1e-30:
            dt_cur = float(np.clip((s * s).sum() / sy, 1e-6, dt_max))
        else:
            dt_cur = min(dt_cur * 2.0, dt_max)
        X, E, F = Xn, En, Fn
        if collapse_threshold is not None:
            new_model = collapse(model, X, float(collapse_threshold))
            if new_model is not None:
                model = new_model
                X = out.vertex_xy / ell
                E, F, dt_cur = energy_of(model, X), force_of(model, X), float(dt)
        energies.append(E)
        residuals.append(float(np.abs(F).max()))
        converged = residuals[-1] < tol_abs

    out.vertex_xy = X * ell
    _straight_polylines(out)
    rebuild_topology(out)
    if arcs:
        render_pressure_arcs(out, gammas, ref_params)
    out.provenance["mechanics"] = {
        "relaxed": True,
        "converged": bool(converged),
        "n_steps": int(steps),
        "residual": float(residuals[-1]),
        "n_collapsed_edges": len(collapsed),
    }
    info = {
        "converged": bool(converged),
        "n_steps": int(steps),
        "n_backtracks": int(n_backtracks),
        "residual": float(residuals[-1]),  # max free-vertex force, tension units
        "residual_history": np.asarray(residuals),
        "energy": float(E),
        "energy_history": np.asarray(energies),
        "ell": ell,
        "force_scale": force_scale,
        "dt_final": dt_cur,
        "n_free_vertices": int(model.free.sum()),
        "gammas": gammas,
        "collapsed_edges": collapsed,
        "params": ref_params,
        "pressures": model.pressures(out.vertex_xy),
        "effective_tensions": model.effective_tensions(out.vertex_xy, gammas),
    }
    return out, info


def jitter_vertices(
    cx: HalfEdgeComplex, sigma: float, rng: np.random.Generator, free_only: bool = True
) -> HalfEdgeComplex:
    """Copy of ``cx`` with Gaussian vertex noise (pixels) for robustness tests.

    Straight ``edge_polyline`` chords follow the vertices; an existing ``edge_smooth``
    trace is carried along by linearly blending the endpoint displacements, so its
    curvature survives while the chord directions change.
    """
    out = straighten_edges(cx)
    if sigma <= 0:
        out.edge_smooth = None if cx.edge_smooth is None else [p.copy() for p in cx.edge_smooth]
        return out
    move = np.ones(cx.n_vertices, dtype=bool)
    if free_only:
        move = PolygonModel(out, VertexModelParams()).free
    out.vertex_xy = cx.vertex_xy + sigma * rng.normal(size=cx.vertex_xy.shape) * move[:, None]
    _straight_polylines(out)
    if cx.edge_smooth is not None:
        smooth = []
        for e in range(out.n_edges):
            p = cx.edge_smooth[e].copy()
            t, h = int(out.edge_tail[e]), int(out.edge_head[e])
            if t != h and p.shape[0] >= 2:
                w = np.linspace(0.0, 1.0, p.shape[0])[:, None]
                p = p + (1 - w) * (out.vertex_xy[t] - p[0]) + w * (out.vertex_xy[h] - p[-1])
            smooth.append(p)
        out.edge_smooth = smooth
    out.invalidate_caches()
    return out
