#!/usr/bin/env python3
"""Predicted barrier conductance on S-BIAD1169 against the measured barrier readouts of
Bromberger et al. 2024 (Life Sci Alliance 7:e202402671, Fig 4).

The S-BIAD1169 images are the 1 h treatments of the same study whose Fig 4A reports the
ECIS resistance change (ohm, 250 Hz) at 1 h per inhibitor and dose, and Fig 4B the tracer
permeability fold change at 1 h (Na-fluorescein 376 Da; TRITC-dextran 70 kDa). The paper
has no source-data file; the values below were read off the published figure (PDF page 8)
and are approximate (about +-50 ohm and +-0.1 fold). Direction and significance follow the
paper's text: vemurafenib 10 to 100 uM and PLX8394 50 to 100 uM lower the resistance,
dabrafenib only at 100 uM, encorafenib not at all.

Usage:
    python scripts/pimorph_function_sbiad1169_vs_paper.py [--fields runs/multijunction/per_field.csv]
        [--out runs/function_real]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

# treatment code -> (resistance change at 1 h, ohm; permeability fold change 1 h 376 Da; 1 h 70 kDa;
#                    paper reports a significant barrier effect at this dose)
PAPER_FIG4 = {
    "DMSO": (0.0, 1.0, 1.0, False),
    "V1": (-50.0, 1.0, 1.0, False),
    "V10": (-500.0, 1.15, 1.05, True),
    "V100": (-1700.0, 2.2, 1.7, True),
    "D1": (-50.0, 1.0, 1.0, False),
    "D10": (-100.0, 1.05, 1.05, False),
    "D100": (-600.0, 1.3, 1.1, True),
    "E1": (-50.0, 1.15, 1.0, False),
    "E10": (-50.0, 1.1, 1.0, False),
    "E100": (-100.0, 1.4, 1.2, False),
    "P1": (-100.0, 1.1, 1.05, False),
    "P10": (-250.0, 1.2, 1.15, False),
    "P100": (-1500.0, 2.1, 1.85, True),
}
SOURCE = "Bromberger et al. 2024 LSA Fig 4A/4B, values read from the figure (no source data published)"
PREDICTIONS = ["G_eff", "permeability_index", "TJ_coverage", "AJ_coverage", "co_occupancy_AJ_TJ", "gap_fraction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="runs/multijunction/per_field.csv")
    ap.add_argument("--out", default="runs/function_real")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.fields)
    df["paper_resistance_change_ohm"] = df["treatment"].map({k: v[0] for k, v in PAPER_FIG4.items()})
    df["paper_perm_376Da_1h"] = df["treatment"].map({k: v[1] for k, v in PAPER_FIG4.items()})
    df["paper_perm_70kDa_1h"] = df["treatment"].map({k: v[2] for k, v in PAPER_FIG4.items()})
    df["paper_significant_effect"] = df["treatment"].map({k: v[3] for k, v in PAPER_FIG4.items()})
    keep = ["field_id", "condition", "treatment", "compound", "concentration_um", "n_cells", "n_gaps"] + PREDICTIONS
    keep += [c for c in df.columns if c.startswith("paper_")]
    df[keep].to_csv(out / "sbiad1169_fields_vs_paper.csv", index=False)

    rows = []
    for pred in PREDICTIONS:
        x = df[pred].to_numpy(float)
        for meas, expect in (
            ("paper_resistance_change_ohm", "negative"),
            ("paper_perm_376Da_1h", "positive"),
            ("paper_perm_70kDa_1h", "positive"),
        ):
            y = df[meas].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[ok], y[ok])
            tau, pt = kendalltau(x[ok], y[ok])
            sig = df["paper_significant_effect"].to_numpy(bool)
            mw = mannwhitneyu(x[ok & sig], x[ok & ~sig], alternative="two-sided")
            if pred in ("TJ_coverage", "AJ_coverage", "co_occupancy_AJ_TJ"):
                expect_here = "positive" if expect == "negative" else "negative"
            else:
                expect_here = expect
            rows.append(
                {
                    "prediction": pred,
                    "measurement": meas,
                    "expected_sign": expect_here,
                    "n_fields": int(ok.sum()),
                    "spearman": float(rho),
                    "p_spearman": float(p),
                    "kendall_tau": float(tau),
                    "p_kendall": float(pt),
                    "sign_matches_expectation": bool(np.sign(rho) == (1 if expect_here == "positive" else -1)),
                    "mean_significant_effect_fields": float(x[ok & sig].mean()),
                    "mean_other_fields": float(x[ok & ~sig].mean()),
                    "n_significant_effect_fields": int((ok & sig).sum()),
                    "p_mannwhitney_sig_vs_other": float(mw.pvalue),
                }
            )
    res = pd.DataFrame(rows)
    res.to_csv(out / "sbiad1169_vs_paper_correlations.csv", index=False)
    # per treatment summary
    per = (
        df.groupby("treatment")
        .agg(
            n_fields=("field_id", "size"),
            G_eff=("G_eff", "mean"),
            permeability_index=("permeability_index", "mean"),
            TJ_coverage=("TJ_coverage", "mean"),
            AJ_coverage=("AJ_coverage", "mean"),
            gap_fraction=("gap_fraction", "mean"),
            paper_resistance_change_ohm=("paper_resistance_change_ohm", "first"),
            paper_perm_376Da_1h=("paper_perm_376Da_1h", "first"),
            paper_perm_70kDa_1h=("paper_perm_70kDa_1h", "first"),
            paper_significant_effect=("paper_significant_effect", "first"),
        )
        .reset_index()
    )
    dmso = per.loc[per["treatment"] == "DMSO", "G_eff"].iloc[0]
    per["G_eff_ratio_to_DMSO"] = per["G_eff"] / dmso
    per.to_csv(out / "sbiad1169_per_treatment_vs_paper.csv", index=False)
    print(per.to_string())
    print(res.to_string())
    (out / "sbiad1169_vs_paper_source.txt").write_text(SOURCE + "\n")


if __name__ == "__main__":
    main()
