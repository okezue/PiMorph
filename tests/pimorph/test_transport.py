"""Transport proxies: parallel and in-plane resistor networks on a synthetic complex.

Checks the analytic parallel sum, Rayleigh monotonicity of the in-plane conductance,
adjoint sensitivities against finite differences, the gap short-circuit effect and the
barrier report's unvalidated flag.
"""

import numpy as np
import pandas as pd
import pytest

from pimorph.complex import extract_complex
from pimorph.complex.geometry import edge_arclengths, smooth_complex
from pimorph.complex.halfedge import FaceKind
from pimorph.fields.multichannel import multichannel_profiles
from pimorph.function import (
    VALIDATION_NOTE,
    barrier_network,
    barrier_report,
    boundary_cells,
    conductance_breakdown,
    edge_conductance_vector,
    effective_conductance,
    in_plane_effective_conductance,
    interface_conductance,
    permeability_index,
    sensitivity,
    solve_in_plane,
    tissue_area,
)

from .conftest import voronoi_labels
from .test_multijunction import render_marker


@pytest.fixture(scope="module")
def cx():
    lab = voronoi_labels(30, shape=(160, 160), seed=5, gaps=2)
    c = extract_complex(lab)
    smooth_complex(c)
    assert c.gap_faces.size >= 1
    return c


@pytest.fixture(scope="module")
def g_random(cx):
    rng = np.random.default_rng(0)
    g = np.zeros(cx.n_edges)
    g[cx.cell_cell_edges()] = rng.uniform(0.2, 2.0, size=cx.cell_cell_edges().size)
    return g


def _terminals(cx):
    """Two disjoint boundary cell sets on opposite sides of the field."""
    from pimorph.complex.geometry import face_centroid

    bc = boundary_cells(cx)
    cols = np.array([face_centroid(cx, int(f))[1] for f in bc])
    order = np.argsort(cols)
    k = max(1, len(bc) // 4)
    return bc[order[:k]], bc[order[-k:]]


def test_interface_conductance_formula():
    pe = pd.DataFrame(
        {
            "edge_id": [0, 1, 2],
            "arclength_px": [10.0, 20.0, 5.0],
            "TJ_coverage": [1.0, 0.5, np.nan],
            "AJ_coverage": [0.0, 1.0, 0.5],
        }
    )
    g = interface_conductance(pe, {"TJ": 1.0, "AJ": 0.5}, baseline=0.1)
    expected = np.array([0.1 + 0.5 * 1.0 * 10.0, 0.1 + 1.0 * 0.5 * 20.0, 0.1 + 1.0 * 5.0 + 0.5 * 0.5 * 5.0])
    assert np.allclose(g, expected)
    # fallback to <c>_occupancy and length scaling
    pe2 = pe.rename(columns={"TJ_coverage": "TJ_occupancy"}).drop(columns="AJ_coverage")
    g2 = interface_conductance(pe2, {"TJ": 1.0}, baseline=0.0, length_scale=0.5)
    assert np.allclose(g2, [0.0, 0.5 * 20.0 * 0.5, 1.0 * 5.0 * 0.5])
    with pytest.raises(KeyError):
        interface_conductance(pe, {"GJ": 1.0})


def test_parallel_conductance_is_analytic_sum(cx, g_random):
    cc = cx.cell_cell_edges()
    G = effective_conductance(cx, g_random, g_gap=3.0, g_trans=0.25)
    expected = g_random[cc].sum() + 3.0 * cx.gap_faces.size + 0.25 * cx.cell_faces.size
    assert np.isclose(G, expected)
    brk = conductance_breakdown(cx, g_random, g_gap=3.0, g_trans=0.25)
    assert np.isclose(brk["G_eff"], expected)
    assert np.isclose(brk["gap_fraction"], 3.0 * cx.gap_faces.size / expected)
    area, unit = tissue_area(cx)
    assert unit == "px^2" and area > 0
    assert np.isclose(permeability_index(cx, g_random, g_gap=3.0, g_trans=0.25), expected / area)
    s = sensitivity(cx, g_random, model="parallel")
    assert np.all(s[cc] == 1.0) and np.all(s[np.setdiff1d(np.arange(cx.n_edges), cc)] == 0.0)


def test_barrier_network_laplacian(cx, g_random):
    net = barrier_network(cx, g_random, gap_conductance=7.0)
    L = net.laplacian
    assert L.shape == (cx.cell_faces.size + cx.gap_faces.size, cx.cell_faces.size + cx.gap_faces.size)
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0)
    assert np.allclose((L - L.T).data, 0.0) if (L - L.T).nnz else True
    assert np.all(L.diagonal() >= 0)
    cc = set(int(e) for e in cx.cell_cell_edges())
    kinds = cx.face_kind[cx.edge_faces[net.edge_ids]]
    for e, w, k in zip(net.edge_ids, net.edge_g, kinds):
        if int(e) in cc:
            assert w == g_random[e]
        else:
            assert w == 7.0 and FaceKind.GAP in set(int(x) for x in k)
    net2 = barrier_network(cx, g_random, include_gaps=False)
    assert net2.n_nodes == cx.cell_faces.size


def test_in_plane_conductance_two_cells():
    lab = np.zeros((20, 40), dtype=np.int32)
    lab[:, :20] = 1
    lab[:, 20:] = 2
    c = extract_complex(lab)
    smooth_complex(c)
    cc = c.cell_cell_edges()
    assert cc.size == 1
    g = np.zeros(c.n_edges)
    g[cc] = 2.5
    cells = c.cell_faces
    G = in_plane_effective_conductance(c, g, [cells[0]], [cells[1]])
    assert np.isclose(G, 2.5)
    sol = solve_in_plane(c, g, [cells[0]], [cells[1]])
    assert np.isclose(sol.edge_sensitivity()[0], 1.0)


def test_rayleigh_monotonicity(cx, g_random):
    src, snk = _terminals(cx)
    base = in_plane_effective_conductance(cx, g_random, src, snk)
    assert base > 0
    rng = np.random.default_rng(1)
    cc = cx.cell_cell_edges()
    for _ in range(50):
        g = g_random.copy()
        e = int(rng.choice(cc))
        g[e] += rng.uniform(0.01, 5.0)
        G = in_plane_effective_conductance(cx, g, src, snk)
        assert G >= base - 1e-10
    # decreasing never increases either
    for _ in range(10):
        g = g_random.copy()
        e = int(rng.choice(cc))
        g[e] *= rng.uniform(0.0, 0.9)
        assert in_plane_effective_conductance(cx, g, src, snk) <= base + 1e-10


def test_in_plane_sensitivity_matches_finite_differences(cx, g_random):
    src, snk = _terminals(cx)
    s = sensitivity(cx, g_random, model="in_plane", source_cells=src, sink_cells=snk)
    cc = cx.cell_cell_edges()
    assert s.shape == (cx.n_edges,)
    assert np.all(s >= 0)
    rng = np.random.default_rng(2)
    G0 = in_plane_effective_conductance(cx, g_random, src, snk)
    for e in rng.choice(cc, size=12, replace=False):
        h = 1e-4 * g_random[e]
        gp, gm = g_random.copy(), g_random.copy()
        gp[e] += h
        gm[e] -= h
        fd = (in_plane_effective_conductance(cx, gp, src, snk) - in_plane_effective_conductance(cx, gm, src, snk)) / (
            2 * h
        )
        assert abs(fd - s[e]) <= 1e-4 * max(abs(s[e]), 1e-3 * G0)
    with pytest.raises(ValueError):
        sensitivity(cx, g_random, model="in_plane")


def test_gaps_increase_conductance(cx, g_random):
    src, snk = _terminals(cx)
    # parallel model: gaps are additive short circuits
    assert effective_conductance(cx, g_random, g_gap=10.0) > effective_conductance(cx, g_random, g_gap=0.0)
    # in-plane model: a gap face bridges its neighbours with a large conductance
    g_no = in_plane_effective_conductance(cx, g_random, src, snk, include_gaps=False)
    g_lo = in_plane_effective_conductance(cx, g_random, src, snk, gap_conductance=0.01)
    g_hi = in_plane_effective_conductance(cx, g_random, src, snk, gap_conductance=100.0)
    assert g_hi >= g_lo >= g_no - 1e-10
    assert g_hi > g_no


def test_in_plane_errors(cx, g_random):
    cells = cx.cell_faces
    with pytest.raises(ValueError):
        in_plane_effective_conductance(cx, g_random, [cells[0]], [cells[0]])
    with pytest.raises(ValueError):
        in_plane_effective_conductance(cx, g_random, [], [cells[0]])
    with pytest.raises(KeyError):
        in_plane_effective_conductance(cx, g_random, [cx.outer_face], [cells[0]])
    with pytest.raises(ValueError):
        effective_conductance(cx, g_random[:3])
    with pytest.raises(ValueError):
        effective_conductance(cx, -g_random)


def test_edge_vector_from_series(cx, g_random):
    cc = cx.cell_cell_edges()
    series = pd.Series(g_random[cc], index=cc)
    G_arr = effective_conductance(cx, g_random)
    G_ser = effective_conductance(cx, series)
    assert np.isclose(G_arr, G_ser)
    v = edge_conductance_vector(cx, cc, g_random[cc], fill=0.0)
    assert np.allclose(v, g_random)


def test_barrier_report_is_flagged_unvalidated(cx):
    cc = cx.cell_cell_edges()
    A = render_marker(cx, cc, (160, 160))
    B = render_marker(cx, cc[cc % 3 == 0], (160, 160), exclude=cc[cc % 3 != 0])
    mp = multichannel_profiles(cx, {"AJ": A, "TJ": B})
    src, snk = _terminals(cx)
    rep = barrier_report(cx, mp, weights={"TJ": 1.0, "AJ": 0.5}, source_cells=src, sink_cells=snk, top_k=5)
    assert rep["validated"] is False
    assert rep["note"] == VALIDATION_NOTE
    assert rep["n_gaps"] == cx.gap_faces.size and rep["n_cells"] == cx.cell_faces.size
    assert np.isclose(rep["G_eff"], rep["G_junction"] + rep["G_gap"] + rep["G_trans"])
    assert 0.0 < rep["gap_fraction"] < 1.0
    assert rep["permeability_index"] > 0 and rep["area_unit"] == "px^2" and rep["length_unit"] == "px"
    assert len(rep["top_leaking_edges"]) == 5
    gs = [t["g"] for t in rep["top_leaking_edges"]]
    assert gs == sorted(gs, reverse=True)
    assert rep["in_plane_conductance"] is not None and rep["in_plane_conductance"] > 0
    # TJ-poor interfaces leak more than TJ-rich ones
    pe = mp.per_edge
    g = interface_conductance(pe, {"TJ": 1.0, "AJ": 0.5})
    rich = pe["TJ_coverage"] > 0.8
    assert (g[rich.to_numpy()] / pe.loc[rich, "arclength_px"]).mean() < (
        g[~rich.to_numpy()] / pe.loc[~rich, "arclength_px"]
    ).mean()
    with pytest.raises(KeyError):
        barrier_report(cx, mp, weights={"GJ": 1.0})
    assert edge_arclengths(cx).shape == (cx.n_edges,)
