#!/usr/bin/env python3
"""Figure for runs/function_real: barrier proxy against measured TER (RPE) and against the
published ECIS / permeability effects (S-BIAD1169)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/function_real")
    a = ap.parse_args()
    out = Path(a.out)
    wells = pd.read_csv(out / "rpe_ter_per_well.csv")
    tiles = pd.read_csv(out / "rpe_ter_per_tile.csv")
    treat = pd.read_csv(out / "sbiad1169_per_treatment_vs_paper.csv")

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, src in zip(axs[:2], ("curated", "neural")):
        col = f"{src}_permeability_index"
        # tile spread per well as thin bars (5 to 95 percentile)
        for _, w in wells.iterrows():
            t = tiles.loc[tiles["well"] == w["well"], col].dropna()
            if len(t) > 2:
                lo, hi = np.percentile(t, [5, 95])
                ax.plot([w["ter_mean"], w["ter_mean"]], [lo, hi], c="lightgrey", lw=1, zorder=1)
        ax.errorbar(wells["ter_mean"], wells[col], xerr=wells["ter_sd"], fmt="o", c="k", ms=5, zorder=2)
        for _, w in wells.iterrows():
            ax.annotate(
                w["well"].replace("AMD", ""),
                (w["ter_mean"], w[col]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        rho, p = spearmanr(wells["ter_mean"], wells[col])
        ax.set_xlabel("measured TER (Ohm cm^2, mean of 3 readings, bar = SD)")
        ax.set_ylabel("predicted permeability index (G_eff per native px^2)")
        ax.set_title(f"AMD-iRPE, {src} complexes\nSpearman {rho:.2f}, p = {p:.2f}, n = {len(wells)} wells", fontsize=10)
    ax = axs[2]
    sig = treat["paper_significant_effect"].astype(bool)
    ax.scatter(
        treat.loc[~sig, "paper_resistance_change_ohm"],
        treat.loc[~sig, "G_eff_ratio_to_DMSO"],
        c="k",
        label="no significant effect in paper",
    )
    ax.scatter(
        treat.loc[sig, "paper_resistance_change_ohm"],
        treat.loc[sig, "G_eff_ratio_to_DMSO"],
        c="r",
        label="significant resistance drop (paper)",
    )
    for _, r in treat.iterrows():
        ax.annotate(
            r["treatment"],
            (r["paper_resistance_change_ohm"], r["G_eff_ratio_to_DMSO"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("ECIS resistance change at 1 h, ohm (Bromberger 2024 Fig 4A, read off figure)")
    ax.set_ylabel("predicted G_eff / DMSO (mean over fields)")
    ax.set_title("S-BIAD1169: 13 treatments, 17 fields", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "barrier_proxy_vs_measured.png", dpi=120)


if __name__ == "__main__":
    main()
