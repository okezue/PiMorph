#!/usr/bin/env python3
"""Render audited PiMorph result figures from compact, versioned source tables.

Usage: python scripts/figures_results.py [--refresh-data]

Default rendering needs only numpy, pandas, matplotlib, and docs/figure_data/results_*.
--refresh-data re-extracts those tables from the committed runs/ CSV and JSON files,
recording their SHA-256 hashes in results_provenance.json. No models or raw images
are required. Figures are descriptive: field/stack pairs are shown, never pooled
across datasets, and reconstruction intervals are not biological uncertainty.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/figure_data"
OUT = ROOT / "docs/figures"
TEAL = "#087F8C"
CORAL = "#C65B46"
INDIGO = "#485B87"
INK = "#24333B"
GRAY = "#86949B"
LIGHT = "#E8EDF0"
METRICS = ["vertex_f1", "adjacency_pair_f1", "pq"]
DS = ["hcec", "alizarine", "flywing", "rpe", "haec"]
TITLES = {"hcec": "hCEC", "alizarine": "Alizarine", "flywing": "FlyWing", "rpe": "RPE / ZO-1", "haec": "HAEC"}


def extract():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path):
        path = ROOT / path
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    bench = []
    for ds in DS:
        stem = "rpe_zo1" if ds == "rpe" else ds
        p = pd.read_csv(record(f"runs/frontier/{ds}_test_v6_tuned/{stem}_per_image.csv"))
        comparison = (
            f"runs/vertex/{ds}_test_cellpose_filled/{stem}_per_image.csv"
            if ds != "rpe"
            else "runs/vertex/rpe_zo1_test_cellpose/rpe_zo1_per_image.csv"
        )
        c = pd.read_csv(record(comparison))
        c = c[c.method == "cellpose_sam_filled"]
        cols = ["image_id"] + METRICS
        j = p[cols].merge(c[cols], on="image_id", suffixes=("_pimorph", "_cellpose"), validate="one_to_one")
        assert len(j) == len(p) == len(c), f"Unpaired benchmark rows for {ds}"
        j.insert(0, "dataset", ds)
        j.insert(2, "display_unit", j.image_id.str.rsplit("_", n=1).str[0] if ds == "rpe" else j.image_id)
        bench.append(j)
    pd.concat(bench).to_csv(DATA / "results_benchmark_pairs.csv", index=False, float_format="%.12g")

    s = pd.read_csv(record("runs/shear_retest/per_field_posterior_stats.csv"))
    s = s[s.statistic.isin(["reticular_fraction", "all_reticular_3clique_enrichment_z", "area_degree_spearman"])]
    s[
        ["image_id", "condition", "replicate", "statistic", "mean", "ci_lo", "ci_hi", "n_hypotheses_total", "ess"]
    ].to_csv(DATA / "results_shear_fields.csv", index=False, float_format="%.12g")

    m = pd.read_csv(record("runs/mechanics/synthetic_validation.csv"))
    keep = [
        "seed",
        "dispersion",
        "noise_px",
        "use_pressures",
        "curvature_pressure",
        "ridge",
        "pearson_effective",
        "pearson_pressure",
        "residual_rms",
        "n_cells",
        "converged",
        "valid",
    ]
    m[keep].to_csv(DATA / "results_mechanics.csv", index=False, float_format="%.12g")
    real = json.loads(record("runs/mechanics/ve_strat_Control_s1_summary.json").read_text())

    tm = json.loads(record("runs/dynamics/tissueminer_demo/summary.json").read_text())
    events = []
    for identity in ["tracker", "gt_ids"]:
        for event in ["t1", "division"]:
            for roi in [False, True]:
                if event == "t1":
                    key = f"t1 vs database ({identity}, {'ROI cells only' if roi else 'all'})"
                elif roi:
                    key = f"division detection ({identity}, ROI cells only)"
                else:
                    key = (
                        "division detection (tracker run, events)"
                        if identity == "tracker"
                        else "division detection (gt_ids run)"
                    )
                v = tm["extra"][key]
                events.append(
                    dict(
                        identity=identity,
                        event=event,
                        roi=roi,
                        **{k: v[k] for k in ["precision", "recall", "f1", "n_pred", "n_gt", "n_tp"]},
                    )
                )
    pd.DataFrame(events).to_csv(DATA / "results_dynamics_events.csv", index=False, float_format="%.12g")
    residuals = []
    for movie in [2, 3]:
        d = pd.read_csv(record(f"runs/dynamics/epicure_movie{movie}/admissibility_gt_ids.csv"))
        for _, row in d.iterrows():
            dr = ast.literal_eval(row.residual_dVEF)
            residuals.append(
                dict(
                    movie=movie,
                    frame=int(row.frame),
                    residual_V=dr[0],
                    residual_E=dr[1],
                    residual_F=dr[2],
                    residual_L1=sum(abs(v) for v in dr),
                    n_unexplained=int(row.n_unexplained),
                    fully_explained=bool(row.fully_explained),
                )
            )
    pd.DataFrame(residuals).to_csv(DATA / "results_dynamics_residuals.csv", index=False)
    metadata = {
        "benchmark": {
            "model": "v6_pool tuned",
            "comparison": "Cellpose-SAM with filled boundaries",
            "rpe_aggregation": (
                "Four tiles per held-out stack, mean scores before plotting paired stack difference; 5 stacks total."
            ),
            "sampling_caveat": (
                "Fields are not asserted to be independent donors or movies; "
                "hCEC cultures and FlyWing movie provenance are not separated by the documented split."
            ),
        },
        "shear": {
            "replicate_labels": ["rep 1", "rep 2", "rep 3"],
            "extra_batch_label": "rep1",
            "interval": (
                "5th–95th percentiles of weighted reconstruction candidates; not calibrated biological intervals."
            ),
            "hypotheses": "24–32 candidates; ensemble ESS is 4 per field.",
        },
        "mechanics_real": real,
        "tissueminer": {"frames": tm["meta"]["n_frames"], "n_gt_divisions": tm["meta"]["n_gt_divisions"]},
        "source_sha256": sources,
    }
    (DATA / "results_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#BBC6CC",
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titleweight": "medium",
        }
    )


def clean(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=LIGHT, linewidth=0.7)
    ax.tick_params(length=3, width=0.7)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=260, bbox_inches="tight", pad_inches=0.14)
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def swarm_offsets(values, span=0.19):
    """Deterministic vertical spreading; each dot remains at its exact x-value."""
    n = len(values)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(values, kind="stable")
    result = np.zeros(n)
    # Alternating bins separate points within dense runs, without random jitter.
    levels = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4], dtype=float) / 4 * span
    for i, idx in enumerate(order):
        result[idx] = levels[i % len(levels)]
    return result


def benchmark():
    d = pd.read_csv(DATA / "results_benchmark_pairs.csv")
    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(1, 3, left=0.115, right=0.99, bottom=0.17, top=0.86, wspace=0.25)
    labels = [
        "hCEC\n5 fields",
        "Alizarine\n10 fields",
        "FlyWing\n10 fields",
        "RPE / ZO-1\n5 stacks · 20 tiles",
        "HAEC\n86 fields",
    ]
    for k, (metric, title) in enumerate(zip(METRICS, ["Vertex F1", "Adjacency F1", "Panoptic quality"])):
        inner = gs[k].subgridspec(1, 2, width_ratios=[3.6, 1.2], wspace=0.07)
        ax, tab = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        ax.axvspan(-0.17, 0, color=CORAL, alpha=0.045)
        ax.axvspan(0, 0.57, color=TEAL, alpha=0.035)
        ax.axvline(0, color=GRAY, lw=0.9)
        ax.set_xlim(-0.17, 0.57)
        ax.set_ylim(4.5, -0.6)
        tab.set_xlim(0, 1)
        tab.set_ylim(4.5, -0.6)
        tab.axis("off")
        tab.text(0.20, -0.57, "CP", color=CORAL, ha="center", fontsize=9, fontweight="bold")
        tab.text(0.79, -0.57, "PiM", color=TEAL, ha="center", fontsize=9, fontweight="bold")
        for i, ds in enumerate(DS):
            part = d[d.dataset == ds]
            if ds == "rpe":
                part = part.groupby("display_unit", sort=True).mean(numeric_only=True)
            diff = (part[f"{metric}_pimorph"] - part[f"{metric}_cellpose"]).to_numpy()
            ax.hlines(i, -0.17, 0.57, color=LIGHT, lw=0.8, zorder=0)
            offs = swarm_offsets(diff)
            ax.scatter(
                diff,
                i + offs,
                c=np.where(diff >= 0, TEAL, CORAL),
                s=23 if ds != "haec" else 14,
                alpha=0.8,
                linewidths=0.35,
                edgecolors="white",
                zorder=3,
            )
            avg = float(np.mean(diff))
            ax.scatter(avg, i + 0.31, marker="D", s=32, facecolors="white", edgecolors=INK, linewidths=1.2, zorder=4)
            for xpos, suffix, color in [(0.20, "cellpose", CORAL), (0.79, "pimorph", TEAL)]:
                tab.text(
                    xpos,
                    i,
                    f"{part[f'{metric}_{suffix}'].mean():.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        ax.set_yticks(range(5), labels if k == 0 else [""] * 5)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_xticks([-0.1, 0, 0.2, 0.4], ["−0.1", "0", "+0.2", "+0.4"])
        ax.set_xlabel("Paired difference  (PiMorph − CP)", labelpad=9)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, pad=20, loc="left")
        ax.text(-0.11, 1.087, chr(65 + k), transform=ax.transAxes, fontweight="bold", fontsize=14)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=TEAL, label="PiMorph higher", markersize=5),
            Line2D([], [], marker="o", ls="", color=CORAL, label="Cellpose-SAM higher", markersize=5),
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                markerfacecolor="white",
                markeredgecolor=INK,
                label="Mean paired difference",
                markersize=5,
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.012),
        ncol=3,
        frameon=False,
        columnspacing=2.8,
    )
    save(fig, "results_benchmark")


def shear():
    d = pd.read_csv(DATA / "results_shear_fields.csv")
    conditions = ["static", "6dyne", "high_shear"]
    reps = [
        ("rep 1", TEAL, "o", -0.24),
        ("rep 2", CORAL, "s", -0.07),
        ("rep 3", INDIGO, "^", 0.10),
        ("rep1", GRAY, "D", 0.27),
    ]
    specs = [
        ("reticular_fraction", "Reticular contacts", "Reticular fraction", (-0.015, 0.36)),
        ("all_reticular_3clique_enrichment_z", "Conditional 3-clique enrichment", "Enrichment z", (-2.0, 5.0)),
        ("area_degree_spearman", "Area–degree association", "Within-field Spearman ρ", (0.26, 0.92)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    fig.subplots_adjust(left=0.064, right=0.985, bottom=0.25, top=0.83, wspace=0.32)
    for k, (ax, (stat, title, ylabel, ylim)) in enumerate(zip(axes, specs)):
        s = d[d.statistic == stat]
        if stat == "all_reticular_3clique_enrichment_z":
            ax.axhline(0, color=GRAY, ls="--", lw=1)
        for rep, color, marker, shift in reps:
            for ci, condition in enumerate(conditions):
                sub = s[(s.replicate == rep) & (s.condition == condition)].sort_values("mean")
                if sub.empty:
                    continue
                x = ci + shift + np.linspace(-0.054, 0.054, len(sub))
                ax.vlines(x, sub.ci_lo, sub.ci_hi, color=color, lw=0.8, alpha=0.5, zorder=2)
                ax.scatter(
                    x,
                    sub["mean"],
                    s=23,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.85,
                    zorder=3,
                )
                ax.plot(
                    [ci + shift - 0.064, ci + shift + 0.064],
                    [sub["mean"].mean()] * 2,
                    color=color,
                    lw=2.5,
                    zorder=4,
                    solid_capstyle="round",
                )
        ax.set_title(title, loc="left", pad=15)
        ax.text(-0.13, 1.075, chr(65 + k), transform=ax.transAxes, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.48, 2.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(3), ["Static\n34 fields", "6\n30 fields", "18–20\n38 fields"])
        ax.set_xlabel("Shear stress  (dyn cm⁻²)", labelpad=9)
        clean(ax)
        if k == 0:
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            ls="",
            color=c,
            label=(rep.replace("rep ", "Replicate ") if rep != "rep1" else "Additional batch"),
            markersize=5,
        )
        for rep, c, m, _ in reps
    ]
    handles.append(Line2D([], [], color=INK, lw=2.5, label="Mean within group"))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.005), ncol=5, frameon=False, columnspacing=1.5
    )
    save(fig, "results_shear")


def dynamics_mechanics():
    e = pd.read_csv(DATA / "results_dynamics_events.csv")
    r = pd.read_csv(DATA / "results_dynamics_residuals.csv")
    m = pd.read_csv(DATA / "results_mechanics.csv")
    meta = json.loads((DATA / "results_provenance.json").read_text())
    fig = plt.figure(figsize=(12.7, 8.8))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.92, bottom=0.09, hspace=0.58, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    cats = [("t1", False), ("t1", True), ("division", False), ("division", True)]
    for identity, col, shift in [("tracker", INDIGO, -0.14), ("gt_ids", TEAL, 0.14)]:
        for i, (event, roi) in enumerate(cats):
            row = e[(e.identity == identity) & (e.event == event) & (e.roi == roi)].iloc[0]
            ax.bar(i + shift, row.f1, width=0.25, color=col, alpha=0.92)
            ax.text(i + shift, row.f1 + 0.025, f"{row.f1:.2f}", ha="center", fontsize=9, color=col)
    ax.set_xticks(range(4), ["T1\nAll cells", "T1\nROI", "Division\nAll cells", "Division\nROI"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Event-detection F1")
    ax.set_title("TissueMiner · event matching", loc="left", pad=17)
    ax.text(-0.11, 1.09, "A", transform=ax.transAxes, fontweight="bold", fontsize=14)
    ax.legend(
        handles=[
            Line2D([], [], marker="s", ls="", color=INDIGO, label="PiMorph tracker", markersize=7),
            Line2D([], [], marker="s", ls="", color=TEAL, label="Reference identities", markersize=7),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.03),
        ncol=2,
        fontsize=9,
    )
    clean(ax)

    sub = gs[0, 1].subgridspec(2, 1, hspace=0.32)
    for q, movie in enumerate([2, 3]):
        ax = fig.add_subplot(sub[q])
        part = r[r.movie == movie]
        ax.bar(part.frame, part.residual_L1, color=CORAL, width=0.72, alpha=0.85)
        zero = part[part.residual_L1 == 0]
        ax.scatter(zero.frame, np.zeros(len(zero)), marker="o", color=TEAL, s=22, clip_on=False, zorder=3)
        ax.set_xlim(-0.7, part.frame.max() + 0.7)
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 15, 30])
        ax.text(
            0.985,
            0.86,
            f"Movie {movie} · {len(zero)}/{len(part)} zero residual",
            ha="right",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_ylabel("|ΔV, ΔE, ΔF|₁", fontsize=9)
        clean(ax)
        if q == 0:
            ax.set_title("EpiCure · unexplained topology change", loc="left", pad=17)
            ax.text(-0.11, 1.21, "B", transform=ax.transAxes, fontweight="bold", fontsize=14)
        else:
            ax.set_xlabel("Frame-pair index", labelpad=6)

    sub = gs[1, 0].subgridspec(1, 2, wspace=0.14)
    modes = [
        (False, False, GRAY, "Tensions only"),
        (True, False, INDIGO, "+ Pressures"),
        (True, True, TEAL, "+ Curvature"),
    ]
    for q, dispersion in enumerate([0.3, 0.6]):
        ax = fig.add_subplot(sub[q])
        for press, curve, color, label in modes:
            part = m[
                (m.dispersion == dispersion)
                & (m.ridge == 0.1)
                & (m.use_pressures == press)
                & (m.curvature_pressure == curve)
            ]
            for seed, seeds in part.groupby("seed"):
                seeds = seeds.sort_values("noise_px")
                ax.plot(seeds.noise_px, seeds.pearson_effective, color=color, lw=0.55, alpha=0.2)
            means = part.groupby("noise_px").pearson_effective.mean()
            ax.plot(means.index, means.values, color=color, marker="o", markersize=4, lw=2, label=label)
        ax.set_ylim(0, 1.06)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.text(0.96, 0.08, f"Dispersion {dispersion}", transform=ax.transAxes, ha="right", fontsize=9)
        ax.set_xlabel("Position noise  (px)")
        clean(ax)
        if q == 0:
            ax.set_ylabel("Tension recovery · Pearson r")
            ax.text(-0.23, 1.155, "C", transform=ax.transAxes, fontweight="bold", fontsize=14)
            ax.set_title("Synthetic force inference", loc="left", pad=30)
        else:
            ax.tick_params(labelleft=False)

    fig.legend(
        handles=[Line2D([], [], color=c, marker="o", markersize=4, lw=2, label=label) for _, _, c, label in modes],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.07, 0.433),
        ncol=3,
        columnspacing=1.1,
        handlelength=1.4,
        fontsize=9,
    )

    ax = fig.add_subplot(gs[1, 1])
    part = m[(m.ridge == 0.1) & m.use_pressures & m.curvature_pressure]
    for q, (dispersion, color, marker, offset) in enumerate([(0.3, TEAL, "o", -0.095), (0.6, INDIGO, "^", 0.095)]):
        for i, noise in enumerate([0, 0.5, 1]):
            s = part[(part.dispersion == dispersion) & (part.noise_px == noise)].sort_values("seed")
            x = i + offset + np.linspace(-0.04, 0.04, len(s))
            ax.scatter(x, s.residual_rms, color=color, marker=marker, s=25, edgecolors="white", linewidths=0.4)
            ax.plot([i + offset - 0.075, i + offset + 0.075], [s.residual_rms.mean()] * 2, color=color, lw=2)
    real = meta["mechanics_real"]["residual_rms"]
    ax.axhline(real, color=CORAL, lw=1.4, ls="--")
    ax.text(-0.38, real + 0.012, f"Real VE-cadherin field · {real:.3f}", fontsize=9, color=CORAL)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Force-balance residual · RMS")
    ax.set_xticks(range(3), ["0", "0.5", "1.0"])
    ax.set_xlabel("Synthetic position noise  (px)")
    ax.set_title("Model fit on synthetic and real geometry", loc="left", pad=30)
    ax.text(-0.11, 1.155, "D", transform=ax.transAxes, fontweight="bold", fontsize=14)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=TEAL, label="Dispersion 0.3", markersize=5),
            Line2D([], [], marker="^", ls="", color=INDIGO, label="Dispersion 0.6", markersize=5),
        ],
        loc="upper left",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0, 1.055),
        fontsize=9,
    )
    clean(ax)
    save(fig, "results_dynamics_mechanics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    if args.refresh_data:
        extract()
    style()
    benchmark()
    shear()
    dynamics_mechanics()
    print("Rendered results_benchmark, results_shear, and results_dynamics_mechanics (PNG, SVG, PDF).")


if __name__ == "__main__":
    main()
