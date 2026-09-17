#!/usr/bin/env python3
"""Reconstruction-uncertainty intervals for the three surviving EGM2 network findings.

For each selected S-BIAD1540 EGM2 field, build a posterior ensemble of legal complexes,
evaluate reticular fraction, all-reticular 3-clique fraction and area-degree Spearman r
per hypothesis, and report posterior mean with a 90% credible interval. Then compare
conditions on posterior means (Mann-Whitney U) and report how the credible half-widths
compare with the between-condition difference.

This does not change the published numbers; it adds the uncertainty they lacked.

Usage:
    python scripts/pimorph_sensitivity_egm2.py --manifest runs/egm2_full/manifest_egm2_local.csv \
        --per-condition 3 --conditions static 6dyne --out runs/pimorph_sensitivity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from pimorph.infer import ClassicalProposer, ConstrainedDecoder, DecoderParams, PosteriorEnsemble, generate_hypotheses
from pimorph.io.manifest import parse_manifest
from pimorph.metrics.sensitivity import posterior_statistics

STATS = ["reticular_fraction", "all_reticular_3clique_fraction", "area_degree_spearman"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/egm2_full/manifest_egm2_local.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--conditions", nargs="+", default=["static", "6dyne"])
    ap.add_argument("--per-condition", type=int, default=3)
    ap.add_argument("--out", default="runs/pimorph_sensitivity")
    ap.add_argument("--moves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = parse_manifest(args.manifest, root=args.root)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for cond in args.conditions:
        pool = [s for s in specs if (s.condition or "") == cond]
        idx = rng.choice(len(pool), size=min(args.per_condition, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]

    rows = []
    for sp in chosen:
        t0 = time.time()
        im = sp.load()
        geom = sp.role_channel(im, "geometry")
        nuc = sp.role_channel(im, "nuclei")
        junc = sp.role_channel(im, "junction")
        maps = ClassicalProposer()(geom, nuc, junc)
        dec = ConstrainedDecoder(pixel_size_um=im.pixel_size_um)
        base = DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)))
        hyps = generate_hypotheses(maps, dec, base, image=geom, n_merge_moves=args.moves, n_split_moves=args.moves)
        post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=4)
        stats = posterior_statistics(post, junc if junc is not None else geom)
        stats["image_id"] = sp.image_id
        stats["condition"] = sp.condition
        stats["geometry_source"] = sp.geometry_source
        rows.append(stats)
        summ = {
            r["statistic"]: (round(r["mean"], 3), round(r["ci_lo"], 3), round(r["ci_hi"], 3))
            for _, r in stats.iterrows()
        }
        head = f"{sp.image_id:32s} {sp.condition:8s} n_hyp={len(hyps):2d} ess={post.meta['ess']:.1f}"
        print(f"{head} {summ} ({time.time() - t0:.0f}s)", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "per_field_posterior_stats.csv", index=False)

    lines = ["# Reconstruction-uncertainty intervals for EGM2 network statistics", ""]
    lines.append(f"Fields: {len(chosen)} ({', '.join(args.conditions)}), hypotheses per field from the classical")
    lines.append("proposer + constrained decoder perturbation grid and merge/split moves. Geometry channel is")
    lines.append("the VE-cadherin channel (geometry_source = junction_channel), which is circular for junction")
    lines.append("scoring and is recorded as such.")
    lines.append("")
    header = "| Statistic | Condition | Posterior mean (fields) | Median 90% CI half-width "
    lines.append(header + "| Between-condition diff | MWU p (means) |")
    lines.append("|---|---|---|---|---|---|")
    report = {}
    for stat in STATS:
        sub = df[df["statistic"] == stat]
        per_cond = {c: sub[sub["condition"] == c] for c in args.conditions}
        means = {c: per_cond[c]["mean"].values for c in args.conditions}
        halfw = float(np.nanmedian((sub["ci_hi"] - sub["ci_lo"]) / 2))
        if len(args.conditions) >= 2 and all(len(means[c]) >= 2 for c in args.conditions[:2]):
            a, b = args.conditions[:2]
            diff = float(np.nanmean(means[b]) - np.nanmean(means[a]))
            p = float(mannwhitneyu(means[a], means[b], alternative="two-sided").pvalue)
        else:
            diff, p = float("nan"), float("nan")
        for c in args.conditions:
            lines.append(f"| {stat} | {c} | {np.nanmean(means[c]):.3f} | {halfw:.3f} | {diff:+.3f} | {p:.3g} |")
        report[stat] = {
            "means": {c: [float(x) for x in means[c]] for c in args.conditions},
            "median_ci_halfwidth": halfw,
            "diff": diff,
            "mwu_p": p,
        }
    lines.append("")
    lines.append("Reading: when the median credible half-width is comparable to or larger than the between-condition")
    lines.append("difference, the reconstruction ambiguity alone can account for the effect on these fields, and the")
    lines.append("finding needs the full 30+30 field set with the posterior propagated before it is claimed.")
    (out / "SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
