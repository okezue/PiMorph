#!/usr/bin/env python3
"""Reconstruction-uncertainty intervals for the three surviving EGM2 network findings.

For each selected S-BIAD1540 EGM2 field, build a posterior ensemble of legal complexes,
evaluate reticular fraction, all-reticular 3-clique fraction (raw, and as enrichment z
against the conditional null that fixes the reticular count), the fraction of graph
3-cliques realized by a common multicellular vertex, and area-degree Spearman r per
hypothesis, and report posterior mean with a 90% credible interval. Then compare
conditions on posterior means (Mann-Whitney U) and report how the credible half-widths
compare with the between-condition difference.

This does not change the published numbers; it adds the uncertainty they lacked.

Usage (demo, classical proposer, 3 fields per condition):
    python scripts/pimorph_sensitivity_egm2.py --per-condition 3 --out runs/pimorph_sensitivity

Full re-test (neural proposer, every field, 1000 permutations, 8 shards in parallel):
    for k in 0 1 2 3 4 5 6 7; do
      PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v2_endo.pt \
      python scripts/pimorph_sensitivity_egm2.py --proposer neural --per-condition 0 --n-perm 1000 \
          --shard $k/8 --out runs/shear_retest --conditions static 6dyne high_shear &
    done; wait
    python scripts/pimorph_sensitivity_egm2.py --report-only --out runs/shear_retest \
        --conditions static 6dyne high_shear
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from pimorph.infer import ClassicalProposer, ConstrainedDecoder, DecoderParams, PosteriorEnsemble, generate_hypotheses
from pimorph.io.manifest import parse_manifest
from pimorph.metrics.sensitivity import posterior_statistics

STATS = [
    "reticular_fraction",
    "all_reticular_3clique_fraction",
    "all_reticular_3clique_enrichment_z",
    "tricellular_realized_3clique_fraction",
    "area_degree_spearman",
    "n_cells",
    "mean_degree",
]


def _downsample(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1 or a is None:
        return a
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3)).astype(np.float32)


def build_proposer(kind: str, checkpoint: str):
    if kind == "classical":
        return ClassicalProposer(), "classical"
    from pimorph.infer.neural.proposer import NeuralProposer

    return NeuralProposer(checkpoint), f"neural:{Path(checkpoint).name}"


def run_fields(args, specs, out: Path) -> pd.DataFrame:
    proposer, source = build_proposer(args.proposer, args.checkpoint)
    rows = []
    for sp in specs:
        per_field = out / "fields" / f"{sp.image_id}.csv"
        if per_field.exists() and not args.overwrite:
            rows.append(pd.read_csv(per_field))
            continue
        t0 = time.time()
        im = sp.load()
        geom = _downsample(sp.role_channel(im, "geometry"), args.downsample)
        nuc = _downsample(sp.role_channel(im, "nuclei"), args.downsample)
        junc = _downsample(sp.role_channel(im, "junction"), args.downsample)
        maps = proposer(geom, nuc, junc) if args.proposer == "classical" else proposer(geom, nuc)
        px = im.pixel_size_um * args.downsample if im.pixel_size_um else None
        dec = ConstrainedDecoder(pixel_size_um=px)
        base = DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)))
        hyps = generate_hypotheses(maps, dec, base, image=geom, n_merge_moves=args.moves, n_split_moves=args.moves)
        post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=4)
        stats = posterior_statistics(post, junc if junc is not None else geom, n_perm=args.n_perm)
        stats["image_id"] = sp.image_id
        stats["condition"] = sp.condition
        stats["replicate"] = getattr(sp, "replicate", "")
        stats["geometry_source"] = sp.geometry_source
        stats["proposer"] = source
        stats["n_hypotheses_total"] = len(hyps)
        stats["downsample"] = args.downsample
        per_field.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(per_field, index=False)
        rows.append(stats)
        summ = {
            r["statistic"]: (round(r["mean"], 3), round(r["ci_lo"], 3), round(r["ci_hi"], 3))
            for _, r in stats.iterrows()
            if r["statistic"] in STATS[:5]
        }
        head = f"{sp.image_id:32s} {sp.condition:10s} n_hyp={len(hyps):2d} ess={post.meta['ess']:.1f}"
        print(f"{head} {summ} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(df: pd.DataFrame, conditions, out: Path, proposer_note: str) -> None:
    df.to_csv(out / "per_field_posterior_stats.csv", index=False)
    n_fields = df["image_id"].nunique()
    per_cond = df.drop_duplicates("image_id").groupby("condition")["image_id"].count().to_dict()
    lines = ["# Reconstruction-uncertainty intervals for EGM2 network statistics", ""]
    lines.append(f"Fields: {n_fields} ({per_cond}). Hypotheses per field from the {proposer_note} proposer +")
    lines.append("constrained decoder perturbation grid and merge/split moves; every statistic is evaluated per")
    lines.append("hypothesis and summarized by its posterior mean and 90% credible interval. Geometry channel is")
    lines.append("the VE-cadherin channel (geometry_source = junction_channel), which is circular for junction")
    lines.append("scoring and is recorded as such. Morphology labels are the heuristic feature-derived classes.")
    lines.append("")
    lines.append(
        "| Statistic | Condition | Posterior mean over fields (sd) | Median 90% CI half-width | Diff vs first | MWU p |"
    )
    lines.append("|---|---|---|---|---|---|")
    report = {}
    for stat in STATS:
        sub = df[df["statistic"] == stat]
        means = {c: sub[sub["condition"] == c]["mean"].dropna().values for c in conditions}
        halfw = float(np.nanmedian((sub["ci_hi"] - sub["ci_lo"]) / 2)) if len(sub) else float("nan")
        a = conditions[0]
        report[stat] = {"means": {c: [float(x) for x in means[c]] for c in conditions}, "median_ci_halfwidth": halfw}
        for c in conditions:
            if c != a and len(means[a]) >= 2 and len(means[c]) >= 2:
                diff = float(np.nanmean(means[c]) - np.nanmean(means[a]))
                p = float(mannwhitneyu(means[a], means[c], alternative="two-sided").pvalue)
            else:
                diff, p = float("nan"), float("nan")
            report[stat][f"diff_{c}_vs_{a}"] = diff
            report[stat][f"mwu_p_{c}_vs_{a}"] = p
            m, s = np.nanmean(means[c]) if len(means[c]) else np.nan, np.nanstd(means[c]) if len(means[c]) else np.nan
            lines.append(
                f"| {stat} | {c} (n={len(means[c])}) | {m:.3f} ({s:.3f}) | {halfw:.3f} | {diff:+.3f} | {p:.3g} |"
            )
    lines.append("")
    lines.append("Reading: when the median credible half-width is comparable to or larger than the between-condition")
    lines.append("difference, reconstruction ambiguity alone can account for the effect on these fields. The")
    lines.append("enrichment z tests whether reticular contacts concentrate on 3-cliques beyond what the reticular")
    lines.append("fraction predicts (z near 0: no concentration). The realized fraction states how many graph")
    lines.append("3-cliques are actual tricellular vertices of the complex.")
    (out / "SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/egm2_full/manifest_egm2_local.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--conditions", nargs="+", default=["static", "6dyne"])
    ap.add_argument("--per-condition", type=int, default=3, help="fields per condition; 0 = all")
    ap.add_argument("--out", default="runs/pimorph_sensitivity")
    ap.add_argument("--moves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proposer", choices=["classical", "neural"], default="classical")
    ap.add_argument(
        "--checkpoint", default=os.environ.get("PIMORPH_NEURAL_CKPT", "models/pimorph_proposals_v2_endo.pt")
    )
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=0, help="permutations for the conditional null (0 = skip)")
    ap.add_argument("--shard", default=None, help="k/n: process every n-th selected field starting at k")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="assemble the report from out/fields/*.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = parse_manifest(args.manifest, root=args.root)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for cond in args.conditions:
        pool = [s for s in specs if (s.condition or "") == cond]
        if args.per_condition and args.per_condition < len(pool):
            idx = rng.choice(len(pool), size=args.per_condition, replace=False)
            pool = [pool[i] for i in sorted(idx)]
        chosen += pool
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        chosen = [s for i, s in enumerate(chosen) if i % n == k]

    if args.report_only:
        files = sorted((out / "fields").glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[df["condition"].isin(args.conditions)]
        note = ", ".join(sorted(df["proposer"].dropna().unique())) if "proposer" in df else "classical"
        write_report(df, args.conditions, out, note)
        return 0

    df = run_fields(args, chosen, out)
    if args.shard is None and len(df):
        write_report(df, args.conditions, out, args.proposer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
