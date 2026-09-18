import numpy as np
import pytest
from scipy.stats import spearmanr

from pimorph.complex import extract_complex, validate
from pimorph.complex.geometry import polyline_curvature
from pimorph.mechanics import (
    PolygonModel,
    VertexModelParams,
    assign_tensions,
    edge_curvatures,
    edge_tension_table,
    identifiability_report,
    infer_tensions,
    inferred_stress_tensor,
    jitter_vertices,
    per_field_summary,
    relax,
    straighten_edges,
    vertex_forces,
)
from pimorph.mechanics.force_inference import design_matrix
from pimorph.mechanics.vertex_model import arc_polyline

from .conftest import voronoi_labels

# Regime used for synthetic ground truth: per-cell reference state, moderately soft
# area and perimeter elasticity, short-range core so no edge collapses at fixed topology.
SYNTH = dict(K_A=1.0, K_P=0.1, core_length=3.0)


def _synthetic(seed: int = 0, dispersion: float = 0.3, n: int = 60, shape=(256, 256)):
    cx = extract_complex(voronoi_labels(n, shape, seed=seed))
    params = VertexModelParams(tension_dispersion=dispersion, seed=seed, **SYNTH)
    gammas = assign_tensions(cx, params)
    relaxed, info = relax(cx, gammas, params, n_steps=3000, tol=1e-6)
    return relaxed, info, params


def _cell_cell_mask(cx, tensions):
    kinds = cx.face_kind[cx.edge_faces]
    return np.isfinite(tensions) & np.all(kinds == 0, axis=1)


@pytest.fixture(scope="module")
def synthetic_03():
    return _synthetic(seed=0, dispersion=0.3)


# ------------------------------------------------------------------ forward model
def test_forces_match_finite_differences():
    cx = straighten_edges(extract_complex(voronoi_labels(40, (160, 160), seed=3)))
    params = VertexModelParams(K_A=3.0, K_P=0.5, tension_dispersion=0.4, seed=3, core_length=3.0)
    gammas = assign_tensions(cx, params)
    model = PolygonModel(cx, params)
    F = vertex_forces(cx, gammas, params)
    X = cx.vertex_xy
    rng = np.random.default_rng(0)
    free = np.flatnonzero(model.free)
    for v in rng.choice(free, 12, replace=False):
        for k in range(2):
            d = 1e-4
            Xp, Xm = X.copy(), X.copy()
            Xp[v, k] += d
            Xm[v, k] -= d
            fd = -(model.energy(Xp, gammas) - model.energy(Xm, gammas)) / (2 * d)
            assert abs(fd - F[v, k]) <= 1e-4 * max(abs(F[v, k]), 1e-3)


def test_assign_tensions_distribution():
    cx = extract_complex(voronoi_labels(60, (256, 256), seed=1))
    params = VertexModelParams(tension_dispersion=0.3, gamma_scale=2.0, seed=1)
    g = assign_tensions(cx, params)
    assert g.shape == (cx.n_edges,)
    assert np.all(g > 0)
    cc = cx.cell_cell_edges()
    assert abs(np.std(np.log(g[cc] / 2.0)) - 0.3) < 0.1
    assert np.allclose(g[cx.boundary_edges()], 2.0)


def test_relaxed_honeycomb_uniform_tension_is_balanced(honeycomb):
    cx = extract_complex(honeycomb)
    params = VertexModelParams(tension_dispersion=0.0, seed=0)
    gammas = assign_tensions(cx, params)
    relaxed, info = relax(cx, gammas, params, n_steps=3000, tol=1e-6)
    assert info["converged"], info["residual"]
    model = PolygonModel(relaxed, info["params"])
    F = vertex_forces(relaxed, gammas, info["params"])
    assert np.abs(F[model.free]).max() < 1e-6
    rep = validate(relaxed)
    assert rep.ok, rep.messages
    A, _, _ = model.areas_perimeters(relaxed.vertex_xy)
    assert np.all(A[relaxed.cell_faces] > 0)
    assert all(p.shape == (2, 2) for p in relaxed.edge_polyline)


def test_energy_decreases_monotonically(synthetic_03):
    _, info, _ = synthetic_03
    e = info["energy_history"]
    assert len(e) > 10
    assert np.all(np.diff(e) <= 1e-12 * max(1.0, abs(e[0])))
    assert info["converged"]
    assert info["residual"] < 1e-6


def test_relaxed_tissue_stays_valid_with_positive_areas(synthetic_03):
    relaxed, info, _ = synthetic_03
    rep = validate(relaxed)
    assert rep.ok, rep.messages
    model = PolygonModel(relaxed, info["params"])
    A, _, L = model.areas_perimeters(relaxed.vertex_xy)
    assert np.all(A[relaxed.cell_faces] > 0)
    assert L[relaxed.edge_tail != relaxed.edge_head].min() > 0.5


def test_pressure_arcs_match_laplace(synthetic_03):
    relaxed, info, _ = synthetic_03
    kappa = edge_curvatures(relaxed)
    g_eff = info["effective_tensions"]
    p = info["pressures"]
    L = np.hypot(*(relaxed.vertex_xy[relaxed.edge_head] - relaxed.vertex_xy[relaxed.edge_tail]).T)
    sel = (g_eff > 0.2) & (np.abs(kappa) * L < 1.0) & (relaxed.edge_tail != relaxed.edge_head)
    dp = p[relaxed.edge_faces[:, 0]] - p[relaxed.edge_faces[:, 1]]
    assert np.allclose(kappa[sel] * g_eff[sel], dp[sel], atol=2e-3 * np.abs(dp[sel]).max() + 1e-6)
    # arc sign convention agrees with polyline_curvature
    arc = arc_polyline(np.array([0.0, 0.0]), np.array([0.0, 20.0]), 0.02, 21)
    assert abs(polyline_curvature(arc)[1:-1].mean() - 0.02) < 1e-4
    assert abs(edge_curvatures_single(arc) - 0.02) < 1e-4


def edge_curvatures_single(poly):
    from pimorph.complex.halfedge import HalfEdgeComplex

    cx = HalfEdgeComplex(
        vertex_xy=np.stack([poly[0], poly[-1]]),
        vertex_kind=np.zeros(2, np.int8),
        edge_tail=np.array([0]),
        edge_head=np.array([1]),
        edge_faces=np.array([[0, 1]]),
        edge_polyline=[poly],
        face_kind=np.array([0, 2], np.int8),
        face_label=np.array([1, -1]),
        face_loops=[[[0]], [[1]]],
        he_next=np.array([0, 1]),
    )
    return float(edge_curvatures(cx)[0])


# --------------------------------------------------------------------- inference
def test_infer_recovers_tensions_and_pressures(synthetic_03):
    relaxed, info, _ = synthetic_03
    res = infer_tensions(relaxed, use_pressures=True, curvature_pressure=True)
    mask = _cell_cell_mask(relaxed, res.tensions)
    r_eff = spearmanr(info["effective_tensions"][mask], res.tensions[mask])[0]
    assert r_eff > 0.99
    r_bare = spearmanr(info["gammas"][mask], res.tensions[mask])[0]
    assert r_bare > 0.7
    cells = np.isfinite(res.pressures)
    r_p = np.corrcoef(info["pressures"][cells], res.pressures[cells])[0, 1]
    assert r_p > 0.99
    # gauge: mean identifiable tension is one
    assert abs(np.nanmean(res.tensions) - 1.0) < 1e-3
    assert res.residual_rms < 0.02


def test_infer_without_curvature_still_ranks_tensions(synthetic_03):
    relaxed, info, _ = synthetic_03
    res = infer_tensions(relaxed, use_pressures=True, curvature_pressure=False)
    mask = _cell_cell_mask(relaxed, res.tensions)
    assert spearmanr(info["effective_tensions"][mask], res.tensions[mask])[0] > 0.8


def test_uniform_tension_inference_has_low_cv(honeycomb):
    cx = extract_complex(honeycomb)
    params = VertexModelParams(tension_dispersion=0.0, seed=0)
    gammas = assign_tensions(cx, params)
    relaxed, _ = relax(cx, gammas, params, n_steps=3000, tol=1e-6)
    res = infer_tensions(relaxed, use_pressures=True, curvature_pressure=True)
    summary = per_field_summary(res)
    assert summary["tension_cv"] < 0.05
    assert summary["fraction_vertices_balanced"] > 0.95


def test_identifiability_report(synthetic_03):
    relaxed, _, _ = synthetic_03
    rep = identifiability_report(relaxed, use_pressures=True, curvature_pressure=False)
    assert rep["rank"] == rep["n_unknowns"] - rep["null_dim"]
    assert rep["null_dim"] >= 1
    assert rep["pressures_need_curvature"]
    assert not rep["pressures_identifiable"]
    assert any("curvature" in s for s in rep["statements"])
    assert any("relative" in s.lower() for s in rep["statements"])
    rep_c = identifiability_report(relaxed, use_pressures=True, curvature_pressure=True)
    assert rep_c["pressures_identifiable"]
    assert rep_c["null_dim"] == 1 + rep_c["n_unconstrained_tensions"]
    A, meta = design_matrix(relaxed, use_pressures=True, curvature_pressure=True)
    assert A.shape == (rep_c["n_equations"], meta["n_t"] + meta["n_p"])


def test_noise_robustness(synthetic_03):
    relaxed, info, _ = synthetic_03
    rng = np.random.default_rng(0)
    noisy = jitter_vertices(relaxed, 0.5, rng)
    assert not np.allclose(noisy.vertex_xy, relaxed.vertex_xy)
    # a stronger prior is what noisy data needs; ridge 1e-3 reproduces exact data
    res = infer_tensions(noisy, use_pressures=True, curvature_pressure=True, ridge=0.1)
    mask = _cell_cell_mask(noisy, res.tensions)
    assert spearmanr(info["effective_tensions"][mask], res.tensions[mask])[0] > 0.7
    cells = np.isfinite(res.pressures)
    assert np.corrcoef(info["pressures"][cells], res.pressures[cells])[0, 1] > 0.7


def test_tables_and_stress(synthetic_03):
    relaxed, _, _ = synthetic_03
    res = infer_tensions(relaxed)
    table = edge_tension_table(relaxed, res)
    assert len(table) == relaxed.n_edges
    for col in ("edge_id", "face_left", "face_right", "arclength", "tension", "relative_tension"):
        assert col in table.columns
    good = table["tension"].notna()
    assert abs(table.loc[good, "relative_tension"].median() - 1.0) < 1e-9
    summary = per_field_summary(res)
    for key in ("tension_cv", "residual_rms", "condition_number", "fraction_vertices_balanced"):
        assert key in summary
    sigma = inferred_stress_tensor(relaxed, res)
    assert sigma.shape == (relaxed.n_faces, 2, 2)
    cells = relaxed.cell_faces
    assert np.all(np.isfinite(sigma[cells]))
    assert np.allclose(sigma[cells], np.swapaxes(sigma[cells], 1, 2))
    assert np.all(np.isnan(sigma[relaxed.outer_face]))
