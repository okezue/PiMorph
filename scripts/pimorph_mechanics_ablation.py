#!/usr/bin/env python3
"""Force inference against laser-ablation recoil (Lang et al. 2019, Zenodo 3257654).

For each of the 15 ablation movies (E-cadherin:GFP germband, one cut of a parasegment
boundary cable): average the frames before the cut, reconstruct the complex with the
PiMorph neural proposer, run ``mechanics.force_inference.infer_tensions`` with the
identifiability report, locate the cut edge from the reslice line and the tracked cut
ends, and compare its relative tension with the recoil velocity of the cut ends.

Two readouts:
- across ablations: Spearman between the inferred relative tension of the cut edge and
  the initial recoil velocity (n = 15, bootstrap CI);
- within each field: is the cut edge (a boundary cable, known from Scarpa et al. 2018 to
  recoil about twice as fast as non-boundary junctions) inferred above the field mean?
  Sign test over fields, plus the same contrast for every edge lying on the cable line.

Usage:
    python scripts/pimorph_mechanics_ablation.py [--data data/ablation_lang2019]
        [--out runs/mechanics_real] [--zoom 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import binomtest, pearsonr, spearmanr, wilcoxon

from pimorph.complex.geometry import smooth_complex
from pimorph.infer import ConstrainedDecoder, DecoderParams
from pimorph.io import ablation_lang2019 as A
from pimorph.mechanics import identifiability_report, infer_tensions, per_field_summary
from pimorph.mechanics.report import edge_tension_table

SETTINGS = {
    "tension_only_r0.1": dict(use_pressures=False, curvature_pressure=False, ridge=0.1),
    "pressure_curvature_r0.1": dict(use_pressures=True, curvature_pressure=True, ridge=0.1),
    "pressure_curvature_r0.01": dict(use_pressures=True, curvature_pressure=True, ridge=0.01),
}


def point_to_edge_distance(cx, e: int, pt: np.ndarray) -> float:
    p = cx.edge_geometry(e)
    if len(p) == 0:
        return float("inf")
    return float(np.sqrt(((p - pt) ** 2).sum(axis=1)).min())


def edge_alignment(cx, e: int, direction: np.ndarray) -> float:
    """|cos| between the edge chord and a unit direction."""
    d = cx.vertex_xy[cx.edge_head[e]] - cx.vertex_xy[cx.edge_tail[e]]
    n = float(np.hypot(*d))
    return abs(float(np.dot(d, direction)) / n) if n > 0 else 0.0


def cable_edges(cx, roi: Dict[str, float], zoom: float, tol_px: float, min_cos: float) -> np.ndarray:
    """Cell-cell edges lying on the reslice line (within ``tol_px``) and aligned with it."""
    p1 = np.array([roi["y1"], roi["x1"]]) * zoom
    p2 = np.array([roi["y2"], roi["x2"]]) * zoom
    d = p2 - p1
    L = float(np.hypot(*d))
    u = d / L
    keep = []
    for e in cx.cell_cell_edges():
        geom = cx.edge_geometry(e)
        mid = geom[len(geom) // 2]
        rel = mid - p1
        s = float(np.dot(rel, u))
        if s < 0 or s > L:
            continue
        perp = abs(float(rel[0] * u[1] - rel[1] * u[0]))
        if perp <= tol_px and edge_alignment(cx, int(e), u) >= min_cos:
            keep.append(int(e))
    return np.asarray(keep, dtype=np.int64)


def process(row: pd.Series, prop, zoom: int, pre_frames: int) -> Optional[Dict[str, object]]:
    roi = A.read_line_roi(Path(row.path_roi))
    rec = A.recoil(A.read_track(row.path_left), A.read_track(row.path_right))
    mv = A.read_movie(row.path_movie)
    cut_frame = int(np.floor(rec["cut_frame"]))
    pre = mv[max(0, cut_frame - pre_frames) : cut_frame].mean(axis=0).astype(np.float32)
    g = ndi.zoom(pre, zoom, order=1) if zoom != 1 else pre
    maps = prop(g)
    params = DecoderParams(cell_radius_px=float(maps.meta["cell_radius_px"]), vertex_weight=0.3)
    res = ConstrainedDecoder().decode(maps, params)
    cx = smooth_complex(res.cx)
    cut_pt = A.line_point(roi, rec["cut_pos_px"]) * zoom
    direction = A.line_direction(roi)
    cc = cx.cell_cell_edges()
    if cc.size == 0:
        return None
    dists = np.array([point_to_edge_distance(cx, int(e), cut_pt) for e in cc])
    order = np.argsort(dists)
    cut_edge = None
    for i in order[:5]:
        if edge_alignment(cx, int(cc[i]), direction) >= 0.5:
            cut_edge = int(cc[i])
            cut_dist = float(dists[i])
            break
    if cut_edge is None:
        cut_edge, cut_dist = int(cc[order[0]]), float(dists[order[0]])
    cable = cable_edges(cx, roi, zoom, tol_px=4.0 * zoom, min_cos=0.7)
    out: Dict[str, object] = {
        "ablation_id": row.ablation_id,
        "genotype": row.genotype,
        "zoom": zoom,
        "n_cells": int(cx.cell_faces.size),
        "n_gaps": int(cx.gap_faces.size),
        "n_cell_cell_edges": int(cc.size),
        "cut_frame": rec["cut_frame"],
        "recoil_px_per_frame": rec["recoil_px_per_frame"],
        "recoil_um_per_s": rec["recoil_um_per_s"],
        "initial_separation_px": rec["initial_separation_px"],
        "cut_edge": cut_edge,
        "cut_edge_distance_px": cut_dist / zoom,
        "cut_edge_alignment": edge_alignment(cx, cut_edge, direction),
        "n_cable_edges": int(cable.size),
    }
    tables = {}
    for name, kw in SETTINGS.items():
        fr = infer_tensions(cx, **kw)
        t = fr.tensions
        ident = np.isfinite(t)
        summ = per_field_summary(fr)
        out[f"{name}_cut_tension"] = float(t[cut_edge]) if ident[cut_edge] else float("nan")
        if ident[cut_edge]:
            out[f"{name}_cut_percentile"] = float((t[ident] < t[cut_edge]).mean())
        else:
            out[f"{name}_cut_percentile"] = float("nan")
        on = cable[ident[cable]] if cable.size else cable
        off = np.setdiff1d(cc[ident[cc]], cable)
        out[f"{name}_cable_mean"] = float(t[on].mean()) if on.size else float("nan")
        out[f"{name}_offcable_mean"] = float(t[off].mean()) if off.size else float("nan")
        out[f"{name}_n_cable_ident"] = int(on.size)
        out[f"{name}_residual_rms"] = summ["residual_rms"]
        out[f"{name}_balanced_5pct"] = summ["fraction_vertices_balanced"]
        out[f"{name}_tension_cv"] = summ["tension_cv"]
        out[f"{name}_cond"] = summ["condition_number"]
        out[f"{name}_rank"] = summ["rank"]
        out[f"{name}_n_unknowns"] = summ["n_unknowns"]
        out[f"{name}_n_equations"] = summ["n_equations"]
        tb = edge_tension_table(cx, fr)
        tb["setting"] = name
        tb["ablation_id"] = row.ablation_id
        tb["is_cut_edge"] = tb["edge_id"] == cut_edge
        tb["on_cable"] = tb["edge_id"].isin(cable)
        tables[name] = tb
    idr = identifiability_report(cx, use_pressures=True, curvature_pressure=True)
    out["null_dim_pc"] = idr["null_dim"]
    out["pressures_identifiable_pc"] = idr["pressures_identifiable"]
    out["_tables"] = tables
    out["_cx"] = cx
    out["_image"] = g
    out["_cut_pt"] = cut_pt
    return out


def spearman_boot(x, y, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = spearmanr(x, y)
    bs = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3:
            continue
        bs.append(spearmanr(x[idx], y[idx])[0])
    return float(rho), float(p), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def summarize(df: pd.DataFrame, out: Path, zoom: int) -> pd.DataFrame:
    rows = []
    for name in SETTINGS:
        ok = np.isfinite(df[f"{name}_cut_tension"]) & np.isfinite(df["recoil_px_per_frame"])
        d = df[ok]
        if len(d) < 4:
            continue
        rho, p, lo, hi = spearman_boot(d[f"{name}_cut_tension"], d["recoil_px_per_frame"])
        pr = pearsonr(d[f"{name}_cut_tension"], d["recoil_px_per_frame"])
        above = int((d[f"{name}_cut_tension"] > 1.0).sum())
        cab = d.dropna(subset=[f"{name}_cable_mean", f"{name}_offcable_mean"])
        diff = cab[f"{name}_cable_mean"] - cab[f"{name}_offcable_mean"]
        rows.append(
            {
                "setting": name,
                "n": int(len(d)),
                "spearman_cut_vs_recoil": rho,
                "p_spearman": p,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "pearson_cut_vs_recoil": float(pr[0]),
                "p_pearson": float(pr[1]),
                "cut_tension_mean": float(d[f"{name}_cut_tension"].mean()),
                "cut_percentile_median": float(d[f"{name}_cut_percentile"].median()),
                "n_cut_above_field_mean": above,
                "p_sign_cut_above_mean": float(binomtest(above, len(d), 0.5, alternative="greater").pvalue),
                "n_cable_fields": int(len(cab)),
                "cable_minus_offcable_mean": float(diff.mean()) if len(diff) else float("nan"),
                "p_wilcoxon_cable_gt_offcable": float(wilcoxon(diff, alternative="greater").pvalue)
                if len(diff) >= 5
                else float("nan"),
                "residual_rms_median": float(d[f"{name}_residual_rms"].median()),
                "balanced_5pct_median": float(d[f"{name}_balanced_5pct"].median()),
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(out / f"ablation_summary_zoom{zoom}.csv", index=False)
    return res


def figure(results: List[Dict[str, object]], df: pd.DataFrame, out: Path, setting: str, zoom: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax = axs[0]
    ax.scatter(df["recoil_px_per_frame"], df[f"{setting}_cut_tension"], c="k")
    for _, r in df.iterrows():
        ax.annotate(r.ablation_id, (r.recoil_px_per_frame, r[f"{setting}_cut_tension"]), fontsize=6)
    ax.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("recoil velocity of cut ends (px / frame)")
    ax.set_ylabel("inferred relative tension of cut edge")
    ax.set_title(f"{setting}, n = {len(df)}")
    ax = axs[1]
    ax.scatter(df[f"{setting}_offcable_mean"], df[f"{setting}_cable_mean"], c="k")
    lim = [0.5, max(1.6, float(np.nanmax(df[f"{setting}_cable_mean"])) + 0.1)]
    ax.plot(lim, lim, "--", c="grey", lw=0.8)
    ax.set_xlabel("mean tension, edges off the cable line")
    ax.set_ylabel("mean tension, edges on the cable line")
    ax.set_title("boundary cable vs other junctions (per field)")
    ax = axs[2]
    r0 = results[0]
    cx, g, cut_pt = r0["_cx"], r0["_image"], r0["_cut_pt"]
    ax.imshow(np.clip(g / np.percentile(g, 99.5), 0, 1), cmap="gray")
    tb = r0["_tables"][setting]
    segs, vals = [], []
    for _, e in tb[tb["identifiable"] & np.isfinite(tb["tension"])].iterrows():
        p = cx.edge_geometry(int(e.edge_id))
        segs.append(p[:, ::-1])
        vals.append(e.tension)
    lc = LineCollection(segs, cmap="jet", linewidths=2)
    lc.set_array(np.asarray(vals))
    lc.set_clim(0.5, 1.5)
    ax.add_collection(lc)
    ax.plot(cut_pt[1], cut_pt[0], "w+", ms=14, mew=2)
    ax.set_title(f"{r0['ablation_id']}: inferred tensions, cut at +")
    ax.axis("off")
    fig.colorbar(lc, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out / f"ablation_recoil_vs_tension_zoom{zoom}.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ablation_lang2019")
    ap.add_argument("--out", default="runs/mechanics_real")
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--pre-frames", type=int, default=3)
    ap.add_argument("--checkpoint", default="models/pimorph_proposals_v6_pool.pt")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from pimorph.infer.neural.proposer import NeuralProposer

    prop = NeuralProposer(a.checkpoint, tta=True)
    df_in = A.list_ablations(Path(a.data))
    results = []
    for _, row in df_in.iterrows():
        r = process(row, prop, a.zoom, a.pre_frames)
        if r is not None:
            results.append(r)
            print(
                r["ablation_id"],
                "cells",
                r["n_cells"],
                "cut edge dist",
                round(r["cut_edge_distance_px"], 1),
                "tension",
                {k: round(r[f"{k}_cut_tension"], 3) for k in SETTINGS},
                flush=True,
            )
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df.to_csv(out / f"ablation_per_cut_zoom{a.zoom}.csv", index=False)
    pd.concat([t for r in results for t in r["_tables"].values()]).to_csv(
        out / f"ablation_edge_tensions_zoom{a.zoom}.csv", index=False
    )
    res = summarize(df, out, a.zoom)
    print(res.to_string())
    figure(results, df, out, "tension_only_r0.1", a.zoom)
    (out / f"ablation_meta_zoom{a.zoom}.json").write_text(
        json.dumps(
            {
                "checkpoint": a.checkpoint,
                "zoom": a.zoom,
                "pre_frames": a.pre_frames,
                "settings": SETTINGS,
                "frame_interval_s": A.FRAME_INTERVAL_S,
                "field_um": A.FIELD_UM,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
