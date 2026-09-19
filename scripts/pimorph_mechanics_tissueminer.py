#!/usr/bin/env python3
"""Tissue-scale consistency of force inference on the TissueMiner pupal-wing demo.

The demo (Etournay et al. 2016, `data/tissueminer/`, 71 frames, ~530 tracked cells per
frame, hinge to the left so the proximal-distal axis is the image x axis) covers a phase in
which wing-blade cells are elongated along PD and relax over time (database elongation
tensor, ``cells.elong_xx``). Under PD tissue stress the inferred tension anisotropy should
point along PD and follow the elongation over time. This is a CONSISTENCY check, not a
validation: no tension was measured on this tissue and the anisotropy is derived from the
same geometry as the elongation.

For every ``--step``-th frame: complex from the curated tracked labels, ``infer_tensions``
(tension only, ridge 0.1, and with pressures plus curvature), tissue-averaged tension
stress tensor S = sum_e gamma_e l_e l_e^T / |l_e| / A over identifiable edges, its
anisotropy (lambda_1 - lambda_2) / (lambda_1 + lambda_2) and axis angle to the x axis, and
the same tensor with gamma_e = 1 (geometric control: network anisotropy alone).

Usage:
    python scripts/pimorph_mechanics_tissueminer.py [--step 5] [--out runs/mechanics_real]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pimorph.complex import extract_complex
from pimorph.complex.geometry import face_area, smooth_complex
from pimorph.dynamics.tissueminer import load_tissueminer_demo
from pimorph.mechanics import infer_tensions, per_field_summary


def tension_stress(cx, tensions: np.ndarray) -> np.ndarray:
    """Tissue-averaged tension stress tensor in (x, y) = (col, row) components."""
    X = cx.vertex_xy
    s = np.zeros((2, 2))
    used = 0
    for e in cx.cell_cell_edges():
        g = tensions[e]
        if not np.isfinite(g):
            continue
        d = X[cx.edge_head[e]] - X[cx.edge_tail[e]]
        L = float(np.hypot(*d))
        if L <= 0:
            continue
        v = np.array([d[1], d[0]])  # (x, y)
        s += g * np.outer(v, v) / L
        used += 1
    area = float(sum(face_area(cx, int(f)) for f in cx.cell_faces))
    return s / max(area, 1e-9), used


def anisotropy(s: np.ndarray):
    w, v = np.linalg.eigh(s)
    lam1, lam2 = float(w[1]), float(w[0])
    axis = v[:, 1]
    ang = float(np.degrees(np.arctan2(axis[1], axis[0])))
    ang = ((ang + 90) % 180) - 90
    return (lam1 - lam2) / max(lam1 + lam2, 1e-12), ang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", default="runs/mechanics_real")
    ap.add_argument("--root", default="data/tissueminer")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_tissueminer_demo(a.root, download=False)
    with sqlite3.connect(data["db_path"]) as con:
        elong = pd.read_sql_query(
            "select frame, avg(elong_xx) exx, avg(elong_xy) exy, count(*) n_cells_db, avg(area) area_db "
            "from cells where cell_id > 10000 group by frame",
            con,
        )
    elong["elong_mag"] = np.hypot(elong["exx"], elong["exy"])
    elong["elong_angle_deg"] = 0.5 * np.degrees(np.arctan2(elong["exy"], elong["exx"]))
    rows = []
    for f in range(0, data["n_frames"], a.step):
        lab = data["label_stack"][f].astype(np.int32)
        cx = smooth_complex(extract_complex(lab))
        row = {
            "frame": f,
            "time_h": float(data["frames"].loc[f, "time_sec"]) / 3600.0,
            "n_cells": int(cx.cell_faces.size),
        }
        geo, _ = tension_stress(cx, np.ones(cx.n_edges))
        row["geom_anisotropy"], row["geom_axis_deg"] = anisotropy(geo)
        for name, kw in {
            "tension_only": dict(use_pressures=False, curvature_pressure=False, ridge=0.1),
            "pressure_curvature": dict(use_pressures=True, curvature_pressure=True, ridge=0.1),
        }.items():
            fr = infer_tensions(cx, **kw)
            s, used = tension_stress(cx, fr.tensions)
            row[f"{name}_anisotropy"], row[f"{name}_axis_deg"] = anisotropy(s)
            row[f"{name}_n_edges"] = used
            summ = per_field_summary(fr)
            row[f"{name}_residual_rms"] = summ["residual_rms"]
            row[f"{name}_tension_cv"] = summ["tension_cv"]
            # tension anisotropy beyond geometry: mean tension of PD-aligned vs AP-aligned edges
            X = cx.vertex_xy
            pd_t, ap_t = [], []
            for e in cx.cell_cell_edges():
                g = fr.tensions[e]
                if not np.isfinite(g):
                    continue
                d = X[cx.edge_head[e]] - X[cx.edge_tail[e]]
                ang = abs(np.degrees(np.arctan2(d[0], d[1])))  # angle to x axis
                ang = min(ang, 180 - ang)
                (pd_t if ang < 30 else ap_t if ang > 60 else []).append(g)
            row[f"{name}_mean_tension_PD_edges"] = float(np.mean(pd_t)) if pd_t else float("nan")
            row[f"{name}_mean_tension_AP_edges"] = float(np.mean(ap_t)) if ap_t else float("nan")
        rows.append(row)
        print(
            row["frame"], round(row["tension_only_anisotropy"], 3), round(row["tension_only_axis_deg"], 1), flush=True
        )
    df = pd.DataFrame(rows).merge(elong, on="frame", how="left")
    df.to_csv(out / "tissueminer_stress_anisotropy.csv", index=False)
    summ = {}
    for name in ("geom", "tension_only", "pressure_curvature"):
        ok = np.isfinite(df[f"{name}_anisotropy"])
        rho, p = spearmanr(df.loc[ok, f"{name}_anisotropy"], df.loc[ok, "elong_mag"])
        summ[name] = {
            "n_frames": int(ok.sum()),
            "axis_deg_median_abs": float(np.median(np.abs(df.loc[ok, f"{name}_axis_deg"]))),
            "axis_within_20deg_of_PD": float(np.mean(np.abs(df.loc[ok, f"{name}_axis_deg"]) < 20)),
            "anisotropy_first": float(df.loc[ok, f"{name}_anisotropy"].iloc[0]),
            "anisotropy_last": float(df.loc[ok, f"{name}_anisotropy"].iloc[-1]),
            "spearman_vs_db_elongation": float(rho),
            "p": float(p),
        }
        if name != "geom":
            r = df[f"{name}_mean_tension_PD_edges"] / df[f"{name}_mean_tension_AP_edges"]
            summ[name]["PD_over_AP_tension_median"] = float(np.nanmedian(r))
            summ[name]["PD_over_AP_tension_first_last"] = [float(r.iloc[0]), float(r.iloc[-1])]
    pd.DataFrame(summ).T.to_csv(out / "tissueminer_stress_summary.csv")
    print(pd.DataFrame(summ).T.to_string())


if __name__ == "__main__":
    main()
