import json

import numpy as np
import pandas as pd
import pytest
from skimage.metrics import variation_of_information as sk_vi

from pimorph.metrics import (
    boundary_scores,
    brier_score,
    coverage,
    expected_calibration_error,
    legacy_adjacency_f1,
    log_score,
    plot_reliability,
    reliability_table,
    risk_coverage_curve,
    structural_metrics,
)
from pimorph.metrics.structural import variation_of_information

from .conftest import voronoi_labels
from .test_matching import _adjacent_label_pairs, merge_balanced_pair, split_largest_cell


@pytest.fixture
def voronoi40():
    return voronoi_labels(40, (128, 128), seed=2)


def merge_smallest_cell(lab):
    """Erase the border of the smallest cell into its neighbour (a merge that moves few pixels)."""
    pairs = _adjacent_label_pairs(lab)
    la, lb, aa, ab = min(pairs, key=lambda t: min(t[2], t[3]))
    small, big = (la, lb) if aa < ab else (lb, la)
    out = lab.copy()
    out[out == small] = big
    return out


# ---------------------------------------------------------------- structural
def test_identity_structural_metrics(voronoi40):
    m = structural_metrics(voronoi40, voronoi40, pixel_size_um=0.3)
    assert m["pq"] == 1.0 and m["sq"] == 1.0 and m["rq"] == 1.0
    assert m["ap50"] == 1.0 and m["ap75"] == 1.0
    assert m["adjacency_pair_f1"] == 1.0 and m["adjacency_component_f1"] == 1.0
    assert m["vertex_loc_error_mean_px"] == 0.0 and m["vertex_loc_error_p95_px"] == 0.0
    assert m["vertex_loc_error_mean_um"] == 0.0
    assert m["vertex_incident_set_accuracy"] == 1.0 and m["vertex_cyclic_order_accuracy"] == 1.0
    assert m["vi"] == pytest.approx(0.0, abs=1e-12)
    assert m["complex_edit_distance_approx"] == 0
    assert m["boundary_f1"] == 1.0 and m["hausdorff_px"] == 0.0
    assert m["valid_pred"] and m["valid_ref"] and m["validity_fraction"] == 1.0
    assert m["euler_residual_pred"] == 0 and m["euler_residual_ref"] == 0
    assert m["runtime_s"] > 0
    # flat and JSON-serializable with plain Python scalars
    json.dumps(m)
    assert all(isinstance(v, (int, float, bool)) for v in m.values())


def test_merge_perturbation(voronoi40):
    merged, la, lb = merge_balanced_pair(voronoi40)
    m = structural_metrics(merged, voronoi40)
    assert m["n_merges"] == 1 and m["n_splits"] == 0
    assert m["n_unmatched_ref_cells"] == 1
    assert m["adjacency_pair_recall"] < 1.0
    assert m["adjacency_pair_precision"] > 0.9
    assert m["adjacency_component_recall"] < 1.0
    assert m["pq"] < 1.0
    assert m["vi_merge"] > m["vi_split"]
    assert m["complex_edit_distance_approx"] > 0
    assert m["valid_pred"]


def test_split_perturbation(voronoi40):
    split, big = split_largest_cell(voronoi40)
    m = structural_metrics(split, voronoi40)
    assert m["n_splits"] == 1 and m["n_merges"] == 0
    assert m["n_pred_cells"] == m["n_ref_cells"] + 1
    assert m["vi_split"] > m["vi_merge"]
    assert m["adjacency_pair_precision"] < 1.0


def test_shift_perturbation(voronoi40):
    shifted = np.roll(voronoi40, 1, axis=0)
    shifted[0, :] = 0  # drop the wrapped row so this is a pure translation
    m = structural_metrics(shifted, voronoi40)
    assert m["pq"] > 0.8
    assert m["ap50"] == 1.0
    assert m["adjacency_pair_f1"] > 0.9 and m["adjacency_component_f1"] > 0.9
    assert 0.8 <= m["vertex_loc_error_mean_px"] <= 1.5
    assert m["vertex_loc_error_p95_px"] <= 1.5
    assert m["boundary_f1"] > 0.95 and m["hausdorff95_px"] <= 1.5
    assert m["vi"] == pytest.approx(float(sk_vi(voronoi40, shifted).sum()))


def test_missing_border_barely_moves_pixels_but_changes_adjacency(voronoi40):
    merged = merge_smallest_cell(voronoi40)
    frac_changed = float(np.mean(merged != voronoi40))
    assert frac_changed < 0.02
    m = structural_metrics(merged, voronoi40)
    # pixel-level scores barely move
    assert m["pq"] > 0.97
    assert m["boundary_f1"] > 0.99
    # structure changes: a face and its contacts are gone
    assert m["n_unmatched_ref_cells"] == 1
    assert m["adjacency_pair_f1"] < 0.99
    assert m["adjacency_pair_recall"] < 1.0
    assert m["n_unmatched_ref_edges"] >= 2
    assert m["complex_edit_distance_approx"] >= 3


def test_legacy_adjacency_f1(voronoi40):
    d = legacy_adjacency_f1(voronoi40, voronoi40)
    assert set(d) >= {"precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"}
    assert d["precision"] == 1.0 and d["recall"] == 1.0 and d["f1"] == 1.0
    assert d["n_predicted"] == d["n_ground_truth"] > 0
    merged = merge_smallest_cell(voronoi40)
    d2 = legacy_adjacency_f1(merged, voronoi40)
    assert d2["recall"] < 1.0 and d2["f1"] < 1.0


def test_variation_of_information_matches_skimage(voronoi40):
    split, _ = split_largest_cell(voronoi40)
    fid_p = split.astype(np.int64) - 1
    fid_r = voronoi40.astype(np.int64) - 1
    vi, h_ref_given_pred, h_pred_given_ref = variation_of_information(fid_p, fid_r)
    expected = sk_vi(voronoi40, split)
    assert h_pred_given_ref == pytest.approx(float(expected[0]))
    assert h_ref_given_pred == pytest.approx(float(expected[1]))
    assert vi == pytest.approx(float(expected.sum()))


def test_boundary_scores_identity_and_shift(voronoi40):
    b = boundary_scores(voronoi40, voronoi40, tol_px=1.0)
    assert b["boundary_f1"] == 1.0 and b["hausdorff_px"] == 0.0
    shifted = np.roll(voronoi40, 2, axis=1)
    shifted[:, :2] = 0
    b0 = boundary_scores(shifted, voronoi40, tol_px=1.0)
    b3 = boundary_scores(shifted, voronoi40, tol_px=3.0)
    assert b0["boundary_f1"] < b3["boundary_f1"]
    assert b3["hausdorff95_px"] >= 2.0


# --------------------------------------------------------------- calibration
def test_ece_of_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert expected_calibration_error(p, y, n_bins=10) < 0.05
    # systematically overconfident probabilities are far from calibrated
    assert expected_calibration_error(np.clip(p * 1.5, 0, 1), y) > 0.1


def test_brier_and_log_score():
    y = np.array([0, 1, 1, 0, 1, 0])
    assert brier_score(np.full(6, 0.5), y) == pytest.approx(0.25)
    assert brier_score(y.astype(float), y) == 0.0
    assert log_score(np.full(6, 0.5), y) == pytest.approx(np.log(2))
    assert log_score(y.astype(float), y) < 1e-5
    with pytest.raises(ValueError):
        brier_score(np.array([1.5, 0.2]), np.array([1, 0]))


def test_coverage_of_gaussian_intervals():
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 3, 5000)
    truth = mu + rng.normal(0, 1, 5000)
    z = 1.6449  # two-sided 90 percent
    assert coverage(mu - z, mu + z, truth) == pytest.approx(0.9, abs=0.03)
    assert coverage(np.zeros(3), np.ones(3), np.array([0.5, 2.0, -1.0])) == pytest.approx(1 / 3)


def test_risk_coverage_curve_monotone():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    unc = 1.0 - np.abs(p - 0.5) * 2  # least certain near 0.5
    df = risk_coverage_curve(p, y, unc)
    assert list(df.columns) == ["uncertainty", "retained", "retained_fraction", "n_errors", "error_rate"]
    assert len(df) == 500
    assert np.all(np.diff(df["retained_fraction"].to_numpy()) > 0)
    assert np.all(np.diff(df["uncertainty"].to_numpy()) >= 0)
    assert df["retained_fraction"].iloc[-1] == 1.0
    overall = float(np.mean((p >= 0.5).astype(int) != y))
    assert df["error_rate"].iloc[-1] == pytest.approx(overall)
    # confident half has a lower error rate than the full set
    assert df["error_rate"].iloc[249] < overall


def test_reliability_table_and_plot(tmp_path):
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    table = reliability_table(p, y, n_bins=10)
    assert isinstance(table, pd.DataFrame)
    assert {"bin", "mean_p", "frac_pos", "count"} <= set(table.columns)
    assert len(table) == 10 and int(table["count"].sum()) == 2000
    assert np.all(np.abs(table["frac_pos"] - table["mean_p"]) < 0.15)
    out = plot_reliability(table, tmp_path / "rel" / "reliability.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_3clique_enrichment_null_and_realization(voronoi40):
    import pandas as pd

    from pimorph.complex import extract_complex
    from pimorph.metrics.sensitivity import (
        RETICULAR,
        all_reticular_3clique_enrichment,
        all_reticular_3clique_fraction,
        tricellular_realized_3clique_fraction,
    )

    cx = extract_complex(voronoi40)
    # confluent Voronoi tiling: nearly every graph 3-clique is a tricellular vertex; the
    # rest are three pairwise-adjacent cells that meet around a fourth (not a junction)
    assert tricellular_realized_3clique_fraction(cx) > 0.95
    edges = cx.cell_cell_edges()
    rng = np.random.default_rng(0)
    random_labels = pd.Series([RETICULAR if rng.random() < 0.5 else "straight" for _ in edges], index=edges)
    enr = all_reticular_3clique_enrichment(cx, random_labels, n_perm=400)
    assert enr["observed"] == all_reticular_3clique_fraction(cx, random_labels)
    assert abs(enr["z"]) < 3.0 and enr["perm_p"] > 0.01
    # reticular labels concentrated on the edges of a few vertices: strong enrichment
    conc = pd.Series("straight", index=edges, dtype=object)
    for v in range(cx.n_vertices):
        if len(cx.vertex_cell_set(v)) >= 3 and rng.random() < 0.3:
            for h in cx.vertex_out_half_edges(v):
                e = h >> 1
                if e in conc.index:
                    conc[e] = RETICULAR
    enr2 = all_reticular_3clique_enrichment(cx, conc, n_perm=400)
    assert enr2["z"] > 3.0 and enr2["perm_p"] < 0.01
