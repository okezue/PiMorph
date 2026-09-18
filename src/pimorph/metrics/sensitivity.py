"""Propagate reconstruction uncertainty into downstream network statistics.

Given a posterior ensemble of legal complexes for one field, every statistic is
evaluated per hypothesis and summarized by its posterior mean, variance and credible
interval. This replaces a single overconfident label image with an interval that
includes segmentation ambiguity, as required for fragile statistics such as
clique counts (blueprint section 8.7).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..complex.dual import to_simple_graph
from ..complex.geometry import face_area
from ..complex.halfedge import HalfEdgeComplex
from ..fields.functionals import legacy_feature_frame, morph_labels
from ..fields.profile import profile_all_edges
from ..infer.posterior import Hypothesis, PosteriorEnsemble

RETICULAR = "reticular"


def edge_morph_labels(cx: HalfEdgeComplex, junction: np.ndarray, threshold: Optional[float] = None) -> pd.Series:
    """Heuristic morphology label per cell-cell edge (feature-derived state, not a
    validated biological class). Index = edge_id."""
    edges = cx.cell_cell_edges()
    if edges.size == 0:
        return pd.Series(dtype=object)
    prof = profile_all_edges(cx, junction, threshold=threshold, edges=edges)
    feats = legacy_feature_frame(prof)
    labels = morph_labels(feats)
    labels.index = prof["edge_id"].values
    return labels


def reticular_fraction(cx: HalfEdgeComplex, labels: pd.Series) -> float:
    if len(labels) == 0:
        return float("nan")
    return float((labels == RETICULAR).mean())


def all_reticular_3clique_fraction(cx: HalfEdgeComplex, labels: pd.Series) -> float:
    """Fraction of dual-graph 3-cliques whose three pair contacts are all reticular
    (a pair counts as reticular when any of its contact components is)."""
    G = to_simple_graph(cx)
    if G.number_of_nodes() == 0:
        return float("nan")
    ret = set()
    for e, lab in labels.items():
        if lab == RETICULAR:
            a, b = (int(x) for x in cx.edge_faces[int(e)])
            ret.add((min(a, b), max(a, b)))
    cliques = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    if not cliques:
        return float("nan")
    n_all = 0
    for a, b, c in cliques:
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        if all(p in ret for p in pairs):
            n_all += 1
    return n_all / len(cliques)


def _clique_pair_arrays(cx: HalfEdgeComplex, labels: pd.Series):
    """3-cliques of the dual graph as (n, 3) indices into a pair list, plus the
    per-pair reticular flag. Shared by the observed statistic and its null."""
    G = to_simple_graph(cx)
    cliques = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    if not cliques:
        return None, None
    pair_index: Dict[tuple, int] = {}
    ret_pairs = set()
    for e, lab in labels.items():
        a, b = (int(x) for x in cx.edge_faces[int(e)])
        key = (min(a, b), max(a, b))
        pair_index.setdefault(key, len(pair_index))
        if lab == RETICULAR:
            ret_pairs.add(key)
    idx = np.zeros((len(cliques), 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(cliques):
        for j, key in enumerate([(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]):
            idx[i, j] = pair_index.setdefault(key, len(pair_index))
    flags = np.zeros(len(pair_index), dtype=bool)
    for key in ret_pairs:
        flags[pair_index[key]] = True
    return idx, flags


def all_reticular_3clique_enrichment(
    cx: HalfEdgeComplex, labels: pd.Series, n_perm: int = 1000, seed: int = 0
) -> Dict[str, float]:
    """Observed all-reticular 3-clique fraction against a conditional null that keeps
    the number of reticular pairs fixed and permutes which pairs carry the label.
    Returns the observed fraction, null mean and sd, z score and two-sided
    permutation p. A z near 0 means the raw fraction is explained by the reticular
    fraction alone (no concentration of reticular contacts on 3-cliques)."""
    nan = {"observed": np.nan, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "perm_p": np.nan}
    idx, flags = _clique_pair_arrays(cx, labels)
    if idx is None or flags.sum() == 0:
        return nan
    obs = float(flags[idx].all(axis=1).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = flags[rng.permutation(flags.size)][idx].all(axis=1).mean()
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else 0.0
    p = float((np.sum(np.abs(null - mu) >= abs(obs - mu)) + 1) / (n_perm + 1))
    return {"observed": obs, "null_mean": mu, "null_sd": sd, "z": float(z), "perm_p": p}


def tricellular_realized_3clique_fraction(cx: HalfEdgeComplex) -> float:
    """Fraction of dual-graph 3-cliques whose three cells meet at one multicellular
    vertex of the complex. A graph 3-clique is a tricellular junction only then."""
    G = to_simple_graph(cx)
    cliques = [frozenset(c) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    if not cliques:
        return float("nan")
    vertex_sets = set()
    for v in range(cx.n_vertices):
        cells = cx.vertex_cell_set(v)
        if len(cells) >= 3:
            vertex_sets.add(frozenset(int(f) for f in cells))
    realized = 0
    for c in cliques:
        if c in vertex_sets or any(c <= s for s in vertex_sets if len(s) > 3):
            realized += 1
    return realized / len(cliques)


def area_degree_spearman(cx: HalfEdgeComplex) -> float:
    G = to_simple_graph(cx)
    cells = [int(f) for f in cx.cell_faces if not cx.face_touches_outer(int(f))]
    if len(cells) < 5:
        return float("nan")
    areas = [face_area(cx, f) for f in cells]
    degs = [G.degree(f) for f in cells]
    r, _ = spearmanr(areas, degs)
    return float(r)


def hypothesis_statistics(
    h: Hypothesis, junction: np.ndarray, threshold: Optional[float] = None, n_perm: int = 0
) -> Dict[str, float]:
    """Network statistics of one hypothesis. ``n_perm`` > 0 adds the conditional-null
    enrichment z of all-reticular 3-cliques and the tricellular realization fraction."""
    labels = edge_morph_labels(h.cx, junction, threshold)
    out = {
        "reticular_fraction": reticular_fraction(h.cx, labels),
        "all_reticular_3clique_fraction": all_reticular_3clique_fraction(h.cx, labels),
        "area_degree_spearman": area_degree_spearman(h.cx),
    }
    if n_perm > 0:
        enr = all_reticular_3clique_enrichment(h.cx, labels, n_perm=n_perm)
        out["all_reticular_3clique_enrichment_z"] = enr["z"]
        out["all_reticular_3clique_null_mean"] = enr["null_mean"]
        out["tricellular_realized_3clique_fraction"] = tricellular_realized_3clique_fraction(h.cx)
    out["n_cells"] = float(h.cx.cell_faces.size)
    out["n_cell_cell_edges"] = float(h.cx.cell_cell_edges().size)
    degs = [d for _, d in to_simple_graph(h.cx).degree()]
    out["mean_degree"] = float(np.mean(degs)) if degs else float("nan")
    return out


def posterior_statistics(
    post: PosteriorEnsemble,
    junction: np.ndarray,
    level: float = 0.9,
    threshold: Optional[float] = None,
    extra: Optional[Dict[str, Callable[[Hypothesis], float]]] = None,
    n_perm: int = 0,
) -> pd.DataFrame:
    """One row per statistic: posterior mean, sd, credible interval, MAP value, and the
    spread relative to the MAP value."""
    per_h = [hypothesis_statistics(h, junction, threshold, n_perm=n_perm) for h in post.hypotheses]
    names = list(per_h[0].keys())
    rows = []
    w = post.weights
    map_idx = int(np.argmax(w))
    for name in names:
        vals = np.array([d[name] for d in per_h], dtype=np.float64)
        ok = np.isfinite(vals)
        if not ok.any():
            rows.append(
                {"statistic": name, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "map": np.nan}
            )
            continue
        ww = w[ok] / w[ok].sum()
        m = float(np.sum(ww * vals[ok]))
        sd = float(np.sqrt(np.sum(ww * (vals[ok] - m) ** 2)))
        order = np.argsort(vals[ok])
        cw = np.cumsum(ww[order])
        lo = vals[ok][order][min(np.searchsorted(cw, (1 - level) / 2), ok.sum() - 1)]
        hi = vals[ok][order][min(np.searchsorted(cw, 1 - (1 - level) / 2), ok.sum() - 1)]
        rows.append(
            {
                "statistic": name,
                "mean": m,
                "sd": sd,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "map": float(vals[map_idx]) if np.isfinite(vals[map_idx]) else np.nan,
                "n_hypotheses": int(ok.sum()),
                "ess": float(1.0 / np.sum(ww**2)),
            }
        )
    if extra:
        for name, fn in extra.items():
            rows.append(
                {
                    "statistic": name,
                    "mean": post.expectation(fn),
                    "sd": float(np.sqrt(post.variance(fn))),
                    "ci_lo": post.credible_interval(fn, level)[0],
                    "ci_hi": post.credible_interval(fn, level)[1],
                    "map": float(fn(post.map_hypothesis)),
                    "n_hypotheses": len(post.hypotheses),
                    "ess": float(1.0 / np.sum(post.weights**2)),
                }
            )
    return pd.DataFrame(rows)
