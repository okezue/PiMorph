#!/usr/bin/env python3
"""Validate force inference on vertex-model ground truth and run it on one real field.

1. Synthetic: Voronoi sheets (60 cells, 256 x 256, seeds 0..9) get log-normal edge
   tensions (dispersion 0.3 and 0.6), are relaxed to equilibrium at fixed topology,
   and the equilibrium geometry alone is handed to ``infer_tensions``. Reported are
   Pearson / Spearman correlations between inferred and true values (effective
   tension, bare tension and cell pressure), also after 0.5 px and 1.0 px Gaussian
   vertex noise. Writes runs/mechanics/synthetic_validation.csv and REPORT.md.
2. Real: runs/pimorph_ve_strat/Control_s1/complex.json if present. Output tensions
   are RELATIVE; no force measurement exists for this field, so nothing is validated.

Usage:
    .venv/bin/python scripts/pimorph_mechanics_eval.py [--seeds 10] [--out runs/mechanics]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "pimorph"))

from conftest import voronoi_labels  # noqa: E402
from pimorph.complex import HalfEdgeComplex, extract_complex, validate  # noqa: E402
from pimorph.mechanics import (  # noqa: E402
    VertexModelParams,
    assign_tensions,
    edge_tension_table,
    identifiability_report,
    infer_tensions,
    jitter_vertices,
    per_field_summary,
    relax,
)

# Forward-model regime: per-cell reference state, spec moduli, 3 px core so that no
# edge collapses at fixed topology (see vertex_model.VertexModelParams).
FORWARD = dict(K_A=1.0, K_P=0.1, gamma_scale=1.0, core_length=3.0)
NOISE_PX = (0.0, 0.5, 1.0)
RIDGES = (1e-3, 1e-1)
REAL_RIDGE = 1e-1
MODELS = ((True, True), (True, False), (False, False))  # (use_pressures, curvature_pressure)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a[ok], b[ok])[0]), float(spearmanr(a[ok], b[ok])[0])


def synthetic_validation(out_dir: Path, seeds: int) -> tuple[pd.DataFrame, list]:
    rows: list = []
    skipped: list = []
    for seed in range(seeds):
        cx = extract_complex(voronoi_labels(60, (256, 256), seed=seed))
        for disp in (0.3, 0.6):
            params = VertexModelParams(tension_dispersion=disp, seed=seed, **FORWARD)
            gammas = assign_tensions(cx, params)
            t0 = time.time()
            try:
                relaxed, info = relax(cx, gammas, params, n_steps=4000, tol=1e-6)
            except RuntimeError as exc:
                # straight chords inconsistent with the crack topology (short edge at a border vertex)
                skipped.append({"seed": seed, "dispersion": disp, "reason": str(exc)})
                print(f"seed {seed} disp {disp}: skipped ({exc})", flush=True)
                continue
            t_relax = time.time() - t0
            g_eff, g_bare, p_true = info["effective_tensions"], info["gammas"], info["pressures"]
            cc = np.all(relaxed.face_kind[relaxed.edge_faces] == 0, axis=1)
            rng = np.random.default_rng(1000 + seed)
            for noise in NOISE_PX:
                noisy = jitter_vertices(relaxed, noise, rng)
                for (use_p, curv), ridge in itertools.product(MODELS, RIDGES):
                    res = infer_tensions(noisy, use_pressures=use_p, curvature_pressure=curv, ridge=ridge)
                    mask = cc & np.isfinite(res.tensions)
                    pe_eff, sp_eff = _corr(g_eff[mask], res.tensions[mask])
                    pe_bare, sp_bare = _corr(g_bare[mask], res.tensions[mask])
                    pe_p, sp_p = _corr(p_true, res.pressures) if use_p else (float("nan"), float("nan"))
                    summ = per_field_summary(res)
                    rows.append(
                        {
                            "seed": seed,
                            "dispersion": disp,
                            "noise_px": noise,
                            "use_pressures": use_p,
                            "curvature_pressure": curv,
                            "ridge": ridge,
                            "n_cells": int(relaxed.cell_faces.size),
                            "n_edges_scored": int(mask.sum()),
                            "converged": bool(info["converged"]),
                            "relax_steps": int(info["n_steps"]),
                            "relax_residual": info["residual"],
                            "relax_seconds": t_relax,
                            "valid": bool(validate(relaxed).ok),
                            "pearson_effective": pe_eff,
                            "spearman_effective": sp_eff,
                            "pearson_bare": pe_bare,
                            "spearman_bare": sp_bare,
                            "pearson_pressure": pe_p,
                            "spearman_pressure": sp_p,
                            "residual_rms": summ["residual_rms"],
                            "condition_number": summ["condition_number"],
                            "n_equations": res.n_equations,
                            "n_unknowns": res.n_unknowns,
                            "rank": res.rank,
                        }
                    )
            print(
                f"seed {seed} disp {disp}: converged={info['converged']} steps={info['n_steps']} "
                f"res={info['residual']:.1e} ({t_relax:.1f}s)",
                flush=True,
            )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "synthetic_validation.csv", index=False)
    return df, skipped


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dispersion", "noise_px", "use_pressures", "curvature_pressure", "ridge"]
    cols = ["pearson_effective", "spearman_effective", "pearson_bare", "spearman_bare", "pearson_pressure"]
    agg = df.groupby(keys)[cols].agg(["mean", "min"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def real_demo(out_dir: Path, path: Path) -> dict | None:
    if not path.exists():
        return None
    cx = HalfEdgeComplex.from_dict(json.load(open(path)))
    # ridge 0.1 is the prior strength that held up under 0.5 px vertex noise above
    res = infer_tensions(cx, use_pressures=True, curvature_pressure=True, ridge=REAL_RIDGE)
    table = edge_tension_table(cx, res)
    table.to_csv(out_dir / "ve_strat_Control_s1_tensions.csv", index=False)
    summ = per_field_summary(res)
    summ["fraction_vertices_balanced_20pct"] = per_field_summary(res, residual_fraction=0.2)[
        "fraction_vertices_balanced"
    ]
    summ["ridge"] = REAL_RIDGE
    ident = identifiability_report(cx, use_pressures=True, curvature_pressure=True)
    summ["identifiability"] = {k: v for k, v in ident.items() if k != "singular_values"}
    summ["n_vertices"] = cx.n_vertices
    summ["n_edges"] = cx.n_edges
    summ["n_cells"] = int(cx.cell_faces.size)
    with open(out_dir / "ve_strat_Control_s1_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return summ


def write_report(out_dir: Path, df: pd.DataFrame, skipped: list, real: dict | None, real_path: Path) -> None:
    agg = summarize(df)
    runs = df.drop_duplicates(["seed", "dispersion"])
    lines = [
        "# Force inference validation",
        "",
        "Forward model: vertex model on Voronoi sheets (60 cells, 256 x 256), per-cell reference "
        f"area and perimeter, K_A={FORWARD['K_A']}, K_P={FORWARD['K_P']}, log-normal edge tensions, "
        f"{FORWARD['core_length']} px short-range core, relaxed at fixed topology to max residual force "
        "< 1e-6 (tension units). Edges are straight chords; ``edge_smooth`` carries the circular arcs "
        "implied by Laplace's law (kappa = delta p / effective tension).",
        "",
        "Truth: the effective tension of an edge (bare gamma plus perimeter-elastic and core terms) is what "
        "the geometry encodes; bare gamma correlations are reported for reference. All inferred values are "
        "RELATIVE (mean tension = 1, pressure gauge as reported); the global scale is not identifiable.",
        "",
        f"Seeds: {df['seed'].nunique()}, relaxations converged: {int(runs['converged'].sum())} / {runs.shape[0]}, "
        f"all complexes valid: {bool(df['valid'].all())}."
        + (
            " Skipped: " + "; ".join(f"seed {k['seed']} dispersion {k['dispersion']} ({k['reason']})" for k in skipped)
            if skipped
            else ""
        ),
        "",
        "## Synthetic recovery (mean over seeds, minimum in parentheses)",
        "",
        "| dispersion | noise px | pressures | curvature | ridge | Pearson eff | Spearman eff | Spearman bare "
        "| Pearson pressure |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['dispersion']} | {r['noise_px']} | {r['use_pressures']} | {r['curvature_pressure']} | "
            f"{r['ridge']:g} | "
            f"{r['pearson_effective_mean']:.3f} ({r['pearson_effective_min']:.3f}) | "
            f"{r['spearman_effective_mean']:.3f} ({r['spearman_effective_min']:.3f}) | "
            f"{r['spearman_bare_mean']:.3f} ({r['spearman_bare_min']:.3f}) | "
            + (
                f"{r['pearson_pressure_mean']:.3f} ({r['pearson_pressure_min']:.3f}) |"
                if np.isfinite(r["pearson_pressure_mean"])
                else "n/a |"
            )
        )
    lines += [
        "",
        "Noise is added to the free vertices only; arc traces follow their endpoints. With curvature the "
        "tension-plus-pressure system is overdetermined and recovery on noiseless data is exact up to scale. "
        "Without curvature the pressures are constrained only through the chord pressure terms and the "
        "system is underdetermined (ridge prior fills the null space); without pressures the tension-only "
        "balance absorbs pressure forces.",
        "",
        "## Real field",
        "",
    ]
    if real is None:
        lines.append(f"`{real_path}` not found; real-field demo skipped.")
    else:
        lines += [
            f"`{real_path}`: {real['n_cells']} cells, {real['n_edges']} edges, {real['n_vertices']} vertices. "
            f"Inferred {real['n_tensions']} relative tensions and {real['n_pressures']} relative pressures "
            f"(pressure gauge: {real['pressure_gauge']}). Tension CV {real['tension_cv']:.3f}, residual RMS "
            f"{real['residual_rms']:.3f} (tension units), fraction of interior vertices balanced within 5% "
            f"{real['fraction_vertices_balanced']:.2f} (within 20%: {real['fraction_vertices_balanced_20pct']:.2f}), "
            f"condition number {real['condition_number']:.2e}, {real['n_equations']} equations for "
            f"{real['n_unknowns']} unknowns (rank {real['rank']}), ridge {real['ridge']:g}.",
            "",
            "These are RELATIVE tensions with no validation: no force measurement (laser ablation, "
            "micropipette, FRET sensor) exists for this field, and absolute tension, absolute pressure and "
            "the elastic moduli are not identifiable from geometry. The residual force imbalance is large "
            "compared with the synthetic equilibria (RMS below 0.05 there), so the vertex-balance model "
            "describes this reconstruction poorly: reticular VE-cadherin junctions and crack-corner vertex "
            "positions do not deliver tension-balanced angles. Treat the table as a diagnostic, not a "
            "measurement.",
        ]
    lines += [
        "",
        "## Not validated",
        "",
        "- absolute tension and pressure scale (gauge);",
        "- pressures without edge curvature (underdetermined without the Laplace law);",
        "- any mechanics on real data (only self-consistency: residual force balance);",
        "- topology changes (T1/T2); the forward model runs at fixed topology.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="runs/mechanics")
    ap.add_argument("--real", default="runs/pimorph_ve_strat/Control_s1/complex.json")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    df, skipped = synthetic_validation(out_dir, args.seeds)
    print(summarize(df).to_string(index=False))
    real = real_demo(out_dir, ROOT / args.real)
    if real is not None:
        print(json.dumps({k: v for k, v in real.items() if k != "identifiability"}, indent=1, default=str))
    write_report(out_dir, df, skipped, real, Path(args.real))
    print(f"wrote {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
