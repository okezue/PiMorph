#!/usr/bin/env python3
"""Multi-junction (AJ/TJ) fields and predicted barrier conductance on S-BIAD1169.

For every field of the BRAFi study (confluent dermal microvascular EC, DMSO vs BRAF
inhibitors): reconstruct the cell complex from VE-cadherin (geometry) and DAPI
(nuclei), profile VE-cadherin (AJ), claudin-5 (TJ) and F-actin along every cell-cell
interface on shared arclength bins, derive joint junction states from the multichannel
vector, and evaluate the resistor-network barrier proxy.

The permeability numbers are UNTESTED PREDICTIONS: no TEER or tracer data exists for
these monolayers, and the report says so.

Usage:
    python scripts/pimorph_multijunction_sbiad1169.py [--max-fields 12] [--download]
        [--data data/sbiad1169] [--out runs/multijunction]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu

from pimorph.fields.multichannel import (
    all_patterns,
    default_feature_columns,
    joint_morphology_states,
    multichannel_profiles,
)
from pimorph.function import barrier_report
from pimorph.infer import ClassicalProposer, ConstrainedDecoder, DecoderParams
from pimorph.io.sbiad1169 import (
    CHANNEL_SOURCE,
    CHANNEL_SUFFIX,
    PIXEL_SIZE_SOURCE,
    PIXEL_SIZE_UM,
    download_sbiad1169,
    list_fields,
    load_field,
)

CHANNEL_ROLES = {"AJ": "VE-Cadherin", "TJ": "Claudin-5", "actin": "F-actin"}
CHANNELS = list(CHANNEL_ROLES)
WEIGHTS = {"TJ": 1.0, "AJ": 0.5}
FIELD_STATS = [
    "AJ_coverage",
    "TJ_coverage",
    "actin_coverage",
    "co_occupancy_AJ_TJ",
    "jaccard_AJ_TJ",
    "pearson_AJ_TJ",
    "exclusive_AJ",
    "exclusive_TJ",
    "exclusive_actin",
    "n_gaps",
    "gap_fraction",
    "G_eff",
    "permeability_index",
    "mean_g_edge",
]


def process_field(row: pd.Series, bin_px: float, half_width_px: float, dapi_sigma_um: float) -> Dict[str, object]:
    ch = load_field(row)
    # nuclear texture in these 8-bit exports fragments the Otsu blob mask, which makes
    # the proposer's nucleus radius collapse (E10, P1: 10 px instead of 45 px) and
    # over-splits cells; smoothing at half a micron restores whole-nucleus blobs
    dapi = ndi.gaussian_filter(ch["DAPI"], dapi_sigma_um / float(ch["pixel_size_um"]))
    maps = ClassicalProposer()(ch["VE-Cadherin"], dapi)
    dec = ConstrainedDecoder(pixel_size_um=ch["pixel_size_um"])
    res = dec.decode(maps, DecoderParams(cell_radius_px=float(maps.meta["cell_radius_px"])))
    cx = res.cx
    images = {name: ch[role] for name, role in CHANNEL_ROLES.items()}
    mp = multichannel_profiles(cx, images, bin_px=bin_px, half_width_px=half_width_px)
    rep = barrier_report(cx, mp, weights=WEIGHTS)
    pe = mp.per_edge.copy()
    pe.insert(0, "field_id", row["field_id"])
    pe.insert(1, "condition", row["condition"])
    pe.insert(2, "treatment", row["treatment"])
    pe["arclength_um"] = pe["arclength_px"] * float(cx.pixel_size_um)
    field: Dict[str, object] = {
        "field_id": row["field_id"],
        "condition": row["condition"],
        "treatment": row["treatment"],
        "compound": row["compound"],
        "concentration_um": row["concentration_um"],
        "figure": row["figure"],
        "n_cells": int(cx.cell_faces.size),
        "n_gaps": int(cx.gap_faces.size),
        "n_edges": int(len(pe)),
        "cell_radius_px": float(maps.meta["cell_radius_px"]),
        "threshold_AJ": mp.thresholds["AJ"],
        "threshold_TJ": mp.thresholds["TJ"],
        "threshold_actin": mp.thresholds["actin"],
    }
    # arclength-weighted field means of the edge statistics
    w = pe["arclength_px"].to_numpy()
    for col in ("AJ_coverage", "TJ_coverage", "actin_coverage", "co_occupancy_AJ_TJ", "jaccard_AJ_TJ", "pearson_AJ_TJ"):
        v = pe[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        field[col] = float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")
    for c in CHANNELS:
        v = pe[f"exclusive_{c}"].to_numpy(dtype=float)
        field[f"exclusive_{c}"] = float(np.average(v, weights=w)) if len(v) else float("nan")
    for pname in all_patterns(CHANNELS):
        col = f"frac_{pname}"
        field[f"state_{pname}"] = float(np.average(pe[col].to_numpy(dtype=float), weights=w)) if len(pe) else np.nan
    for k in ("G_eff", "G_junction", "G_gap", "gap_fraction", "permeability_index", "mean_g_edge", "tissue_area"):
        field[k] = rep[k]
    field["area_unit"] = rep["area_unit"]
    field["validated"] = rep["validated"]
    return {"per_edge": pe, "field": field, "report": rep}


def condition_table(per_field: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FIELD_STATS if c in per_field.columns] + [c for c in per_field.columns if c.startswith("state_")]
    rows = []
    for cond, g in per_field.groupby("condition"):
        r = {
            "condition": cond,
            "n_fields": len(g),
            "n_cells": int(g["n_cells"].sum()),
            "n_edges": int(g["n_edges"].sum()),
        }
        for c in cols:
            r[f"{c}_mean"] = float(g[c].mean())
            r[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def mann_whitney(per_field: pd.DataFrame, a: str = "DMSO", b: str = "BRAFi") -> pd.DataFrame:
    rows = []
    cols = [c for c in FIELD_STATS if c in per_field.columns] + [c for c in per_field.columns if c.startswith("state_")]
    xa = per_field[per_field["condition"] == a]
    xb = per_field[per_field["condition"] == b]
    for c in cols:
        va = xa[c].dropna().to_numpy()
        vb = xb[c].dropna().to_numpy()
        if len(va) and len(vb):
            u, p = mannwhitneyu(va, vb, alternative="two-sided")
        else:
            u, p = float("nan"), float("nan")
        rows.append(
            {
                "statistic": c,
                f"{a}_mean": float(va.mean()) if len(va) else np.nan,
                f"{b}_mean": float(vb.mean()) if len(vb) else np.nan,
                f"n_{a}": len(va),
                f"n_{b}": len(vb),
                "U": float(u),
                "p": float(p),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table without the tabulate dependency."""

    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else format(float(v), floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(
    out: Path,
    per_field: pd.DataFrame,
    cond: pd.DataFrame,
    mw: pd.DataFrame,
    states_meta: Dict[str, object],
    state_by_cond: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    L: List[str] = []
    L.append("# Multi-junction fields and predicted barrier conductance on S-BIAD1169\n")
    L.append(
        "Dataset: BioImage Archive S-BIAD1169 (Bromberger and Schossleitner, CC BY 4.0). Confluent primary dermal "
        "microvascular endothelial cells, 1 h DMSO or BRAF inhibitor (dabrafenib, vemurafenib, encorafenib, PLX8394 at "
        "1, 10, 100 uM), confocal LSM-980 63x. One field per treatment folder.\n"
    )
    L.append("## Channel roles\n")
    L.append(
        f"Source: {CHANNEL_SOURCE}. Mapping of file suffix to marker: "
        + ", ".join(f"`_{k}` = {v}" for k, v in CHANNEL_SUFFIX.items())
        + "."
    )
    L.append(
        " The acquisition text lists DAPI (405), Claudin-5 (488), F-actin (514), VE-Cadherin (594), Prox1 (639). "
        "Exports are 8-bit RGB pseudocolor (VE-cadherin grey, claudin-5 cyan, F-actin yellow, DAPI blue), reduced to "
        f"max over RGB. Calibration: {PIXEL_SIZE_SOURCE}, {PIXEL_SIZE_UM:.4f} um/px; the TIFF resolution tag is a "
        "print dpi and was ignored. Roles used here: AJ = VE-Cadherin, TJ = Claudin-5, actin = F-actin; geometry from "
        "VE-Cadherin, nuclei from DAPI.\n"
    )
    L.append("## Pipeline\n")
    L.append(
        "ClassicalProposer on VE-cadherin with DAPI seeds, ConstrainedDecoder with cell_radius_px from the proposer, "
        f"DAPI smoothed with sigma {args.dapi_sigma_um} um before seeding, multichannel_profiles with "
        f"bin_px={args.bin_px}, half_width_px={args.half_width_px}, Otsu thresholds on "
        "pooled strip samples per channel, a bin counts as occupied at >= 30% of strip samples above threshold. Field "
        "values are arclength-weighted means over cell-cell edges. Barrier: g_e = 0.05 + 1.0 (1 - TJ coverage) L_e + "
        "0.5 (1 - AJ coverage) L_e with L_e in um, explicit gaps 10.0 each, G_eff = sum (parallel model), permeability "
        "index = G_eff per um^2 of cell area.\n"
    )
    L.append("## WARNING: permeability numbers are untested predictions\n")
    L.append(
        "`validated: False`. No TEER, tracer flux or any functional barrier measurement exists for these fields. G_eff "
        "and the permeability index are what the resistor model implies for the reconstructed junction coverage; they "
        "are prediction targets for a future functional experiment, not measurements.\n"
    )
    L.append(f"## Per condition (n fields: {dict(per_field['condition'].value_counts())})\n")
    show = [
        "AJ_coverage",
        "TJ_coverage",
        "co_occupancy_AJ_TJ",
        "jaccard_AJ_TJ",
        "exclusive_AJ",
        "exclusive_TJ",
        "n_gaps",
        "gap_fraction",
        "G_eff",
        "permeability_index",
    ]
    L.append("| statistic | " + " | ".join(f"{c} (mean +- sd)" for c in cond["condition"]) + " | Mann-Whitney p |")
    L.append("|---|" + "---|" * (len(cond) + 1))
    mw_i = mw.set_index("statistic")
    for c in show:
        cells = []
        for _, r in cond.iterrows():
            cells.append(f"{r[f'{c}_mean']:.3f} +- {r[f'{c}_sd']:.3f}")
        p = mw_i.loc[c, "p"] if c in mw_i.index else float("nan")
        tag = " (untested prediction)" if c in ("G_eff", "permeability_index", "gap_fraction") else ""
        L.append(f"| {c}{tag} | " + " | ".join(cells) + f" | {p:.3f} |")
    L.append("")
    L.append("### Vector-state fractions of interface arclength (mean over fields)\n")
    state_cols = [c for c in per_field.columns if c.startswith("state_")]
    L.append("| pattern | " + " | ".join(cond["condition"]) + " | p |")
    L.append("|---|" + "---|" * (len(cond) + 1))
    for c in state_cols:
        cells = [f"{r[f'{c}_mean']:.3f}" for _, r in cond.iterrows()]
        p = mw_i.loc[c, "p"] if c in mw_i.index else float("nan")
        L.append(f"| {c[len('state_') :]} | " + " | ".join(cells) + f" | {p:.3f} |")
    L.append("")
    L.append("## Joint junction states (GMM with BIC over the multichannel edge vector, all fields pooled)\n")
    ari_m = states_meta.get("stability_ari_mean", float("nan"))
    ari_s = states_meta.get("stability_ari_std", float("nan"))
    L.append(
        f"k = {states_meta.get('k')}, features = {states_meta.get('features')}, bootstrap stability ARI = "
        f"{ari_m:.3f} +- {ari_s:.3f} over {states_meta.get('n_boot')} resamples.\n"
    )
    if len(state_by_cond):
        L.append(md_table(state_by_cond))
        L.append("")
        means = states_meta.get("state_means", {})
        L.append("State means (original units):\n")
        for k, m in means.items():
            L.append(f"- state {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in m.items()))
        L.append("")
    L.append("## Per field\n")
    pf_show = ["field_id", "condition", "treatment", "n_cells", "n_gaps", "n_edges"] + show
    L.append(md_table(per_field[pf_show]))
    L.append("")
    L.append("## Limitations\n")
    L.append(
        "- One field per treatment (3 DMSO, 15 BRAFi); the Mann-Whitney test compares 3 against 15 fields and a "
        "significant p here is at most suggestive.\n"
        "- Images are 8-bit RGB pseudocolor exports with a burned-in scale bar, not raw acquisitions; intensities are "
        "display-scaled per image, so thresholds are per-field Otsu and intensities are not comparable across fields.\n"
        "- Geometry comes from VE-cadherin alone; where VE-cadherin is lost after BRAFi the decoder can miss or merge "
        "cells, which biases TJ coverage on the surviving edges.\n"
        "- Explicit gaps depend on the decoder's gap map; their count is a reconstruction output, not an annotation.\n"
        "- The barrier proxy is a passive resistor network with hand-set weights; G_eff and the permeability index are "
        "untested predictions (validated: False).\n"
    )
    (out / "REPORT.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sbiad1169")
    ap.add_argument("--out", default="runs/multijunction")
    ap.add_argument("--max-fields", type=int, default=None, help="balanced across conditions")
    ap.add_argument("--download", action="store_true", help="download missing files first")
    ap.add_argument("--bin-px", type=float, default=2.0)
    ap.add_argument("--half-width-px", type=float, default=3.0)
    ap.add_argument("--dapi-sigma-um", type=float, default=0.5, help="DAPI smoothing before seed/scale estimation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_sbiad1169(args.data, max_fields=args.max_fields)
    fields = list_fields(args.data)
    if fields.empty:
        print(f"no complete fields under {args.data}; run with --download", flush=True)
        return 1
    if args.max_fields is not None and args.max_fields < len(fields):
        picked = []
        groups = {c: list(g.index) for c, g in fields.groupby("condition")}
        while len(picked) < args.max_fields and any(groups.values()):
            for c in sorted(groups):
                if groups[c] and len(picked) < args.max_fields:
                    picked.append(groups[c].pop(0))
        fields = fields.loc[sorted(picked)].reset_index(drop=True)
    print(f"{len(fields)} fields: {fields['condition'].value_counts().to_dict()}", flush=True)

    edge_frames: List[pd.DataFrame] = []
    field_rows: List[Dict[str, object]] = []
    reports: Dict[str, object] = {}
    for _, row in fields.iterrows():
        t0 = time.time()
        r = process_field(row, bin_px=args.bin_px, half_width_px=args.half_width_px, dapi_sigma_um=args.dapi_sigma_um)
        edge_frames.append(r["per_edge"])
        field_rows.append(r["field"])
        reports[str(row["field_id"])] = {k: v for k, v in r["report"].items() if k != "top_leaking_edges"}
        f = r["field"]
        head = f"{row['treatment']:5s} {row['figure']:3s} cells={f['n_cells']:4d} gaps={f['n_gaps']:3d}"
        print(
            f"{head} edges={f['n_edges']:4d} AJ={f['AJ_coverage']:.2f} TJ={f['TJ_coverage']:.2f} "
            f"co={f['co_occupancy_AJ_TJ']:.2f} G_eff={f['G_eff']:.0f} PI={f['permeability_index']:.4f} "
            f"(prediction) {time.time() - t0:.0f}s",
            flush=True,
        )
        pd.DataFrame(field_rows).to_csv(out / "per_field.csv", index=False)

    per_edge = pd.concat(edge_frames, ignore_index=True)
    per_field = pd.DataFrame(field_rows)

    # joint states on all edges pooled, then their distribution per condition
    feat = default_feature_columns(CHANNELS)
    labels, meta = joint_morphology_states(per_edge, feat, k_range=(2, 6), seed=args.seed)
    per_edge["joint_state"] = labels
    ct = pd.crosstab(per_edge["condition"], per_edge["joint_state"], normalize="index")
    state_by_cond = ct.reset_index().rename(columns={c: f"state_{c}" for c in ct.columns})
    for cond in per_field["condition"].unique():
        sub = per_edge[per_edge["condition"] == cond]
        for k in sorted(set(labels)):
            per_field.loc[per_field["condition"] == cond, f"joint_state_{k}_fraction"] = float(
                (sub["joint_state"] == k).mean()
            )

    per_edge.to_csv(out / "per_edge.csv", index=False)
    per_field.to_csv(out / "per_field.csv", index=False)
    cond = condition_table(per_field)
    cond.to_csv(out / "per_condition.csv", index=False)
    mw = mann_whitney(per_field)
    mw.to_csv(out / "mann_whitney_dmso_vs_brafi.csv", index=False)
    (out / "joint_states_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "barrier_reports.json").write_text(json.dumps(reports, indent=1, default=str))
    write_report(out, per_field, cond, mw, meta, state_by_cond, args)

    print("\nPer condition (permeability = untested prediction):")
    show = [
        "AJ_coverage",
        "TJ_coverage",
        "co_occupancy_AJ_TJ",
        "exclusive_AJ",
        "exclusive_TJ",
        "n_gaps",
        "permeability_index",
    ]
    print(cond[["condition", "n_fields"] + [f"{c}_mean" for c in show]].to_string(index=False))
    print(mw[mw["statistic"].isin(show)].to_string(index=False))
    print(f"\njoint states: k={meta['k']} stability ARI={meta['stability_ari_mean']:.3f}")
    print(f"wrote {out}/per_edge.csv, per_field.csv, per_condition.csv, REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
