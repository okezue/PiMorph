"""Unit tests for the conditional all-reticular 3-clique null in scripts/harden_network_stats.py."""

import importlib.util
import json
import sys
from math import comb, isnan
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "harden_network_stats.py"
spec = importlib.util.spec_from_file_location("harden_network_stats", SCRIPT)
hns = importlib.util.module_from_spec(spec)
sys.modules["harden_network_stats"] = hns
spec.loader.exec_module(hns)


def _k4_three_reticular():
    G = nx.complete_graph(4)
    labels = {
        (0, 1): "reticular", (1, 2): "reticular", (0, 2): "reticular",
        (0, 3): "straight", (1, 3): "punctate", (2, 3): "straight",
    }
    return G, labels


class TestAllReticularCliqueNull:
    def test_observed_counts(self):
        G, labels = _k4_three_reticular()
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=50, seed=0)
        assert r["n_edges"] == 6
        assert r["n_reticular_edges"] == 3
        assert r["n_3cliques"] == 4
        assert r["n_all_reticular"] == 1
        assert r["all_reticular_3clique_pct"] == pytest.approx(25.0)

    def test_null_mean_matches_hypergeometric(self):
        # 3 of 6 edges reticular: a given 3-edge clique is all-reticular with prob 1 / C(6,3)
        G, labels = _k4_three_reticular()
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=4000, seed=0)
        expected = 100.0 / comb(6, 3)
        assert r["null_mean_pct"] == pytest.approx(expected, abs=1.0)
        assert r["null_sd_pct"] > 0
        assert r["enrichment_z"] > 0
        assert 0 < r["perm_p"] <= 1

    def test_reticular_count_preserved_under_permutation(self):
        # every permutation keeps the reticular count, so the null is exactly 100% when all edges are reticular
        G = nx.complete_graph(5)
        labels = {e: "reticular" for e in G.edges()}
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=20, seed=0)
        assert r["all_reticular_3clique_pct"] == pytest.approx(100.0)
        assert r["null_mean_pct"] == pytest.approx(100.0)
        assert r["null_sd_pct"] == pytest.approx(0.0)
        assert isnan(r["enrichment_z"])

    def test_no_reticular_edges(self):
        G = nx.complete_graph(4)
        labels = {e: "straight" for e in G.edges()}
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=20, seed=0)
        assert r["n_all_reticular"] == 0
        assert r["all_reticular_3clique_pct"] == 0.0
        assert r["null_mean_pct"] == 0.0

    def test_no_3cliques(self):
        G = nx.cycle_graph(5)
        labels = {e: "reticular" for e in G.edges()}
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=20, seed=0)
        assert r["n_3cliques"] == 0
        assert isnan(r["all_reticular_3clique_pct"])
        assert isnan(r["enrichment_z"])

    def test_seed_reproducible(self):
        G, labels = _k4_three_reticular()
        a = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=200, seed=0)
        b = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=200, seed=0)
        c = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=200, seed=1)
        assert a == b
        assert a["seed"] == 0 and c["seed"] == 1

    def test_labels_in_either_orientation(self):
        G, labels = _k4_three_reticular()
        flipped = {(v, u): lab for (u, v), lab in labels.items()}
        a = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=10, seed=0)
        b = hns.all_reticular_clique_null(G, flipped, "reticular", n_perm=10, seed=0)
        assert a["n_all_reticular"] == b["n_all_reticular"] == 1

    def test_unlabelled_edges_count_as_non_reticular(self):
        G, labels = _k4_three_reticular()
        del labels[(0, 1)]
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=10, seed=0)
        assert r["n_reticular_edges"] == 2
        assert r["n_all_reticular"] == 0

    def test_enrichment_detects_clustered_labels(self):
        # 6-clique with all edges reticular inside {0..3} and none elsewhere: strong concentration
        G = nx.complete_graph(7)
        labels = {e: ("reticular" if max(e) <= 3 else "straight") for e in G.edges()}
        r = hns.all_reticular_clique_null(G, labels, "reticular", n_perm=1000, seed=0)
        assert r["enrichment_z"] > 3
        assert r["perm_p"] < 0.01


class TestPerImage3CliqueStats:
    def _frames(self):
        cells = pd.DataFrame({"cell_id": [1, 2, 3, 4], "image_id": "img0", "condition": "static"})
        edges = pd.DataFrame({
            "cell_i": [1, 2, 1, 1, 2, 3],
            "cell_j": [2, 3, 3, 4, 4, 4],
            "aj_morph": ["reticular", "reticular", "reticular", "straight", "straight", "straight"],
            "image_id": "img0",
            "condition": "static",
        })
        return cells, edges

    def test_new_keys_and_deprecated_aliases(self):
        cells, edges = self._frames()
        df = hns.compute_per_image_3clique_stats(cells, edges, n_perm=50, seed=0)
        assert len(df) == 1
        row = df.iloc[0]
        for k in ("all_reticular_3clique_pct", "n_3cliques", "null_mean_pct", "null_sd_pct", "enrichment_z", "perm_p"):
            assert k in df.columns, k
        assert row["n_3cliques"] == 4
        assert row["all_reticular_3clique_pct"] == pytest.approx(25.0)
        # deprecated aliases still present for one release
        assert row["all_reticular_triangle_pct"] == row["all_reticular_3clique_pct"]
        assert row["n_triangles"] == row["n_3cliques"]
        assert "frac_3cliques_with_common_vertex" not in df.columns

    def test_deprecated_function_alias(self):
        assert hns.compute_per_image_triangle_stats is hns.compute_per_image_3clique_stats

    def test_complex_dir_missing_is_skipped(self, tmp_path):
        cells, edges = self._frames()
        df = hns.compute_per_image_3clique_stats(cells, edges, n_perm=10, seed=0, complex_dir=tmp_path / "nope")
        assert "frac_3cliques_with_common_vertex" not in df.columns

    @pytest.mark.skipif(hns.HalfEdgeComplex is None, reason="pimorph not installed")
    def test_complex_dir_reports_vertex_fraction(self, tmp_path):
        pimorph_complex = pytest.importorskip("pimorph.complex")
        # three cells meeting at one T vertex: K3 dual, the single 3-clique is realized by that vertex
        lab = np.zeros((20, 20), dtype=np.int32)
        lab[:10, :10] = 1
        lab[:10, 10:] = 2
        lab[10:, :] = 3
        cx = pimorph_complex.extract_complex(lab)
        report = pimorph_complex.clique_vertex_report(cx)
        assert report["n_3cliques"] == 1
        assert report["n_3cliques_with_common_vertex"] == 1

        cells = pd.DataFrame({"cell_id": [1, 2, 3], "image_id": "img0", "condition": "static"})
        edges = pd.DataFrame({
            "cell_i": [1, 1, 2], "cell_j": [2, 3, 3], "aj_morph": "reticular",
            "image_id": "img0", "condition": "static",
        })
        (tmp_path / "img0").mkdir()
        (tmp_path / "img0" / "complex.json").write_text(json.dumps(cx.to_dict()))

        df = hns.compute_per_image_3clique_stats(cells, edges, n_perm=10, seed=0, complex_dir=tmp_path)
        row = df.iloc[0]
        assert row["n_3cliques"] == 1
        assert row["n_3cliques_complex"] == 1
        assert row["n_3cliques_with_common_vertex"] == 1
        assert row["n_tricellular_vertices"] == 1
        assert row["frac_3cliques_with_common_vertex"] == pytest.approx(1.0)
