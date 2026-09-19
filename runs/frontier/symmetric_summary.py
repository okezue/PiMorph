#!/usr/bin/env python3
"""Symmetric comparison table: Cellpose-SAM zero-shot vs Cellpose-SAM fine-tuned on the
PiMorph training fields vs PiMorph v6_pool (tuned), per held-out test split.

Reads the per-image CSVs written by ``pimorph benchmark`` and writes
``runs/frontier/symmetric_summary.csv`` and ``.md`` (means over test images; the vertex
localization column is the mean of per-image medians, as in the frontier table).

Usage: python runs/frontier/symmetric_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROWS = [
    # split label, dataset, run dir, method label
    ("hCEC (5)", "hcec", "runs/vertex/hcec_test_cellpose_filled", "Cellpose-SAM zero-shot"),
    ("hCEC (5)", "hcec", "runs/frontier/hcec_test_cpsam_ft", "Cellpose-SAM ft confluent pool"),
    ("hCEC (5)", "hcec", "runs/frontier/hcec_test_cpsam_ft_hcec_only", "Cellpose-SAM ft hCEC only"),
    ("hCEC (5)", "hcec", "runs/frontier/hcec_test_cpsam_ft_haec", "Cellpose-SAM ft HAEC (cross-domain)"),
    ("hCEC (5)", "hcec", "runs/frontier/hcec_test_v6_tuned", "PiMorph v6_pool tuned"),
    ("alizarine (10)", "alizarine", "runs/vertex/alizarine_test_cellpose_filled", "Cellpose-SAM zero-shot"),
    ("alizarine (10)", "alizarine", "runs/frontier/alizarine_test_cpsam_ft", "Cellpose-SAM ft confluent pool"),
    ("alizarine (10)", "alizarine", "runs/frontier/alizarine_test_v6_tuned", "PiMorph v6_pool tuned"),
    ("FlyWing (10)", "flywing", "runs/vertex/flywing_test_cellpose_filled", "Cellpose-SAM zero-shot"),
    ("FlyWing (10)", "flywing", "runs/frontier/flywing_test_cpsam_ft", "Cellpose-SAM ft confluent pool"),
    ("FlyWing (10)", "flywing", "runs/frontier/flywing_test_v6_tuned", "PiMorph v6_pool tuned"),
    ("RPE (20)", "rpe_zo1", "runs/vertex/rpe_zo1_test_cellpose", "Cellpose-SAM zero-shot"),
    ("RPE (20)", "rpe_zo1", "runs/frontier/rpe_zo1_test_cpsam_ft", "Cellpose-SAM ft confluent pool"),
    ("RPE (20)", "rpe_zo1", "runs/frontier/rpe_test_v6_tuned", "PiMorph v6_pool tuned"),
    ("HAEC (86)", "haec", "runs/vertex/haec_test_cellpose_filled", "Cellpose-SAM zero-shot"),
    ("HAEC (86)", "haec", "runs/frontier/haec_test_cpsam_ft", "Cellpose-SAM ft HAEC train"),
    (
        "HAEC (86)",
        "haec",
        "runs/frontier/haec_test_cpsam_ft_confluent",
        "Cellpose-SAM ft confluent pool (cross)",
    ),
    ("HAEC (86)", "haec", "runs/frontier/haec_test_v6_tuned", "PiMorph v6_pool tuned"),
]

COLS = [
    ("adjacency_pair_f1", "Adj F1 pair", "mean"),
    ("vertex_f1", "Vertex F1", "mean"),
    ("vertex_precision", "Vertex prec.", "mean"),
    ("vertex_recall", "Vertex rec.", "mean"),
    ("pq", "PQ", "mean"),
    ("boundary_f1", "Boundary F1", "mean"),
    ("n_pred_cells", "pred cells", "mean"),
    ("n_ref_cells", "true cells", "mean"),
    ("n_pred_vertices", "pred vertices", "mean"),
    ("n_ref_vertices", "true vertices", "mean"),
    ("vertex_loc_error_median_px", "Loc. median (px)", "mean"),
]


def main() -> int:
    recs = []
    for split, ds, run, label in ROWS:
        p = Path(run) / f"{ds}_per_image.csv"
        if not p.exists():
            print(f"missing {p}")
            continue
        d = pd.read_csv(p)
        if "error" in d.columns:
            d = d[d["error"].isna()]
        method = "cellpose_sam_filled" if "cellpose" in run else d["method"].iloc[0]
        d = d[d["method"] == method]
        r = {"split": split, "dataset": ds, "method": label, "run": run, "n": int(d["image_id"].nunique())}
        for key, _, agg in COLS:
            r[key] = float(d[key].agg(agg))
        recs.append(r)
    df = pd.DataFrame(recs)
    df.to_csv("runs/frontier/symmetric_summary.csv", index=False)

    lines = [
        "| Test split | Method | Adj F1 pair | Vertex F1 | Vertex prec. / rec. | pred / true cells | "
        "pred / true vertices | Loc. median (px) | PQ | Boundary F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['split']} | {r['method']} | {r['adjacency_pair_f1']:.3f} | {r['vertex_f1']:.3f} | "
            f"{r['vertex_precision']:.3f} / {r['vertex_recall']:.3f} | "
            f"{r['n_pred_cells']:.0f} / {r['n_ref_cells']:.0f} | "
            f"{r['n_pred_vertices']:.0f} / {r['n_ref_vertices']:.0f} | {r['vertex_loc_error_median_px']:.2f} | "
            f"{r['pq']:.3f} | {r['boundary_f1']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    Path("runs/frontier/symmetric_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
