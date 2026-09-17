#!/usr/bin/env python3
"""Aggregate endothelial benchmark runs (runs/endo/<run>/<dataset>_per_image.csv) into one table.

Usage: python scripts/summarize_endo_bench.py runs/endo [--md out.md]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

COLS = {
    "adjacency_pair_f1": "adjF1",
    "adjacency_component_f1": "adjF1_comp",
    "vertex_f1": "vertexF1",
    "vertex_incident_set_accuracy": "incident",
    "pq": "PQ",
    "ap50": "AP50",
    "boundary_f1": "boundaryF1",
    "complex_edit_distance_approx": "edit",
    "validity_fraction": "valid",
    "n_ref_cells": "ref_cells",
    "n_pred_cells": "pred_cells",
    "method_runtime_s": "s_img",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", default="runs/endo", nargs="?")
    ap.add_argument("--md", default=None)
    ap.add_argument("--huvec-only", action="store_true", help="restrict mCellSeg rows to HUVEC images")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f"{args.root}/*/*_per_image.csv")):
        df = pd.read_csv(f)
        if "error" in df:
            df = df[df["error"].isna()]
        run = Path(f).parent.name
        ds = df["dataset"].iloc[0] if len(df) else "?"
        if args.huvec_only and ds == "mcellseg":
            df = df[df["image_id"].str.contains("HUVEC")]
        if not len(df):
            continue
        r = {"run": run, "dataset": ds, "method": df["method"].iloc[0], "n": len(df)}
        for k, name in COLS.items():
            if k in df:
                r[name] = float(df[k].median()) if k == "vertex_loc_error_median_px" else float(df[k].mean())
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["dataset", "run"])
    pd.set_option("display.width", 260)
    print(t.round(3).to_string(index=False))
    if args.md:
        r = t.round(3)
        lines = ["| " + " | ".join(r.columns) + " |", "|" + "---|" * len(r.columns)]
        for _, row in r.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
        Path(args.md).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
