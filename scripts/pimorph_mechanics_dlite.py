#!/usr/bin/env python3
"""PiMorph force inference against DLITE on the DLITE ZO-1 time series (Vasan et al. 2019).

Data: ``data/dlite/DLITE-master/Notebooks/Data/ZO-1_data`` (AllenCellModeling/DLITE,
Allen Institute licence): four time series of hiPSC colonies with endogenous GFP-ZO-1,
hand-traced in NeuronJ (curved interfaces), 10 to 31 frames each. DLITE tracks edge labels
across frames. This script runs the DLITE package on the same traces with its two solvers
(``CellFIT``: per-frame constrained least squares; ``DLITE``: dynamic optimisation seeded
by the previous frame), rasterizes the traced colony into a label image, extracts the
PiMorph complex, runs ``infer_tensions`` on it, matches edges geometrically and reports:

- agreement between PiMorph and DLITE relative tensions on the same edges (per frame and
  pooled; Spearman, Pearson, n);
- the temporal-consistency metric of the DLITE paper: correlation and mean absolute
  change of normalised tensions on edges present in consecutive frames, for each method.

No tension was measured on these colonies: DLITE tensions are another inference, so the
agreement is a cross-method check, not a validation against a physical readout.

Usage:
    python scripts/pimorph_mechanics_dlite.py [--dlite data/dlite/DLITE-master]
        [--out runs/mechanics_real] [--series 1 2 3 4] [--solvers CellFIT DLITE]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import pearsonr, spearmanr
from skimage.draw import line as draw_line
from skimage.segmentation import expand_labels

from pimorph.complex import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.mechanics import infer_tensions, per_field_summary

SERIES = {
    1: ("Time-series_1", None, list(range(0, 30))),
    2: ("Time-series_2", "small_v2", list(range(0, 31))),
    3: ("Time-series_3", "small_v3", list(range(0, 31))),
    4: ("Time-series_4", "small_v4", list(range(0, 10))),
}
CANVAS = (1024, 1024)
SETTINGS = {
    "tension_only_r0.1": dict(use_pressures=False, curvature_pressure=False, ridge=0.1),
    "pressure_curvature_r0.1": dict(use_pressures=True, curvature_pressure=True, ridge=0.1),
}


def _patch_numpy_for_dlite() -> None:
    """DLITE targets numpy 1.16: 2-vector cross products and ``np.math``."""
    cross = np.cross

    def cross2(a, b, *args, **kw):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape[-1] == 2 and b.shape[-1] == 2:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return cross(a, b, *args, **kw)

    np.cross = cross2
    np.math = math


def run_dlite(dlite_root: Path, series: int, solver: str) -> Dict[str, object]:
    """Run the DLITE package; returns per-frame edge and cell records."""
    _patch_numpy_for_dlite()
    sys.path.insert(0, str(dlite_root))
    import matplotlib

    matplotlib.use("Agg")
    from DLITE.ManualTracingMultiple import ManualTracingMultiple

    folder, kind, frames = SERIES[series]
    cwd = os.getcwd()
    os.chdir(dlite_root / "Notebooks" / "Data" / "ZO-1_data" / folder)
    try:
        mt = ManualTracingMultiple(frames, type=kind)
        t0 = time.time()
        colonies = mt.main_computation_based_on_prev(frames, solver=solver, maxiter=60 * 1000)
        dt = time.time() - t0
    finally:
        os.chdir(cwd)
    out_frames = {}
    for key, col in colonies.items():
        edges = []
        for e in col.tot_edges:
            xs = np.asarray(e.x_co_ords if e.x_co_ords is not None else [n.loc[0] for n in e.nodes], float)
            ys = np.asarray(e.y_co_ords if e.y_co_ords is not None else [n.loc[1] for n in e.nodes], float)
            edges.append(
                {
                    "label": e.label if e.label != [] else None,
                    "tension": float(e.tension) if e.tension is not None else float("nan"),
                    "radius": float(e.radius) if e.radius is not None else float("nan"),
                    "x": xs,
                    "y": ys,
                    "node_a": tuple(float(v) for v in e.node_a.loc),
                    "node_b": tuple(float(v) for v in e.node_b.loc),
                }
            )
        out_frames[int(key)] = {
            "edges": edges,
            "n_cells": len(col.cells),
            "n_nodes": len(col.tot_nodes),
        }
    return {"frames": out_frames, "seconds": dt}


def rasterize_colony(edges: List[Dict[str, object]], canvas=CANVAS) -> np.ndarray:
    """Label image of the traced colony: cells are the regions enclosed by the traces."""
    border = np.zeros(canvas, dtype=bool)
    for e in edges:
        xs, ys = e["x"], e["y"]
        pts = np.stack([ys, xs], axis=1)  # (row, col)
        if len(pts) < 2:
            pts = np.array([[e["node_a"][1], e["node_a"][0]], [e["node_b"][1], e["node_b"][0]]])
        pts = np.clip(np.round(pts).astype(int), 0, np.array(canvas) - 1)
        for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
            rr, cc = draw_line(r0, c0, r1, c1)
            border[rr, cc] = True
    border = ndi.binary_dilation(border, iterations=1)
    lab, n = ndi.label(~border)
    if n == 0:
        return lab.astype(np.int32)
    # the region touching the image border is the medium
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    lab[np.isin(lab, edge_labels)] = 0
    lab = expand_labels(lab, distance=2)
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < 50)
    lab[np.isin(lab, small[small > 0])] = 0
    return lab.astype(np.int32)


def match_edges(cx, edges: List[Dict[str, object]], max_dist: float = 4.0) -> Dict[int, int]:
    """DLITE edge index -> PiMorph edge id by mean point-to-polyline distance.

    NeuronJ traces split one interface at intermediate degree-2 nodes, and PiMorph merges
    those chains, so several DLITE edges may map to one PiMorph edge (many-to-one).
    """
    geoms = [cx.edge_geometry(e) for e in range(cx.n_edges)]
    cands = []
    for i, e in enumerate(edges):
        pts = np.stack([e["y"], e["x"]], axis=1)
        if len(pts) < 2:
            continue
        best, best_d = None, np.inf
        for eid, g in enumerate(geoms):
            if len(g) == 0:
                continue
            # cheap reject on bounding boxes
            if (g.min(0) - pts.max(0) > max_dist).any() or (pts.min(0) - g.max(0) > max_dist).any():
                continue
            d = np.sqrt(((pts[:, None, :] - g[None, :, :]) ** 2).sum(-1)).min(1).mean()
            if d < best_d:
                best, best_d = eid, d
        if best is not None and best_d <= max_dist:
            cands.append((best_d, i, best))
    return {i: eid for _, i, eid in cands}


def consistency(frame_table: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """Consecutive-frame agreement of per-frame mean-normalised tensions on tracked edges."""
    cors, abs_changes, ns = [], [], []
    frames = sorted(frame_table["frame"].unique())
    for f0, f1 in zip(frames[:-1], frames[1:]):
        a = frame_table[frame_table["frame"] == f0].dropna(subset=[value_col, "label"])
        b = frame_table[frame_table["frame"] == f1].dropna(subset=[value_col, "label"])
        m = a.merge(b, on="label", suffixes=("_0", "_1"))
        if len(m) < 4:
            continue
        x = m[f"{value_col}_0"].to_numpy(float)
        y = m[f"{value_col}_1"].to_numpy(float)
        if abs(x.mean()) < 1e-6 or abs(y.mean()) < 1e-6:
            continue
        x, y = x / x.mean(), y / y.mean()
        cors.append(pearsonr(x, y)[0])
        abs_changes.append(float(np.mean(np.abs(y - x))))
        ns.append(len(m))
    return {
        "n_pairs": len(cors),
        "pearson_consecutive_mean": float(np.mean(cors)) if cors else float("nan"),
        "pearson_consecutive_sd": float(np.std(cors)) if cors else float("nan"),
        "mean_abs_change": float(np.mean(abs_changes)) if abs_changes else float("nan"),
        "edges_per_pair": float(np.mean(ns)) if ns else float("nan"),
    }


def process_series(dlite_root: Path, series: int, solvers: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    runs = {}
    failed = {}
    for s in solvers:
        try:
            runs[s] = run_dlite(dlite_root, series, s)
        except Exception as exc:  # DLITE's own solver can fail on its data with current scipy
            failed[s] = f"{type(exc).__name__}: {exc}"
            print(f"series {series}: DLITE solver {s} failed ({failed[s]})", flush=True)
    if not runs:
        return pd.DataFrame(), {"frames": [], "seconds": {}, "failed": failed}
    solvers = [s for s in solvers if s in runs]
    ref = runs[solvers[0]]["frames"]
    rows = []
    frame_meta = []
    for frame in sorted(ref):
        edges = ref[frame]["edges"]
        lab = rasterize_colony(edges)
        cx = smooth_complex(extract_complex(lab))
        matched = match_edges(cx, edges)
        inferred = {name: infer_tensions(cx, **kw) for name, kw in SETTINGS.items()}
        summ = per_field_summary(inferred["tension_only_r0.1"])
        frame_meta.append(
            {
                "series": series,
                "frame": frame,
                "dlite_cells": ref[frame]["n_cells"],
                "pimorph_cells": int(cx.cell_faces.size),
                "dlite_edges": len(edges),
                "pimorph_edges": int(cx.n_edges),
                "matched": len(matched),
                "identifiable_matched": int(
                    sum(bool(inferred["tension_only_r0.1"].tension_identifiable[eid]) for eid in matched.values())
                ),
                "residual_rms": summ["residual_rms"],
                "n_interior_vertices": summ["n_interior_vertices"],
            }
        )
        # DLITE tensions by (solver, label) for this frame
        by_label = {}
        for s in solvers:
            for e in runs[s]["frames"].get(frame, {"edges": []})["edges"]:
                if e["label"] is not None:
                    by_label[(s, e["label"])] = e["tension"]
        for i, e in enumerate(edges):
            row = {
                "series": series,
                "frame": frame,
                "dlite_index": i,
                "label": e["label"],
                "radius": e["radius"],
                "pimorph_edge": matched.get(i),
            }
            for s in solvers:
                row[f"dlite_{s}"] = by_label.get((s, e["label"]), float("nan")) if e["label"] is not None else np.nan
            for name, fr in inferred.items():
                eid = matched.get(i)
                row[f"pimorph_{name}"] = float(fr.tensions[eid]) if eid is not None else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, {"frames": frame_meta, "seconds": {s: runs[s]["seconds"] for s in solvers}, "failed": failed}


def summarize(df: pd.DataFrame, solvers: List[str]) -> pd.DataFrame:
    rows = []
    for series, g in df.groupby("series"):
        for name in SETTINGS:
            pcol = f"pimorph_{name}"
            for s in solvers:
                dcol = f"dlite_{s}"
                ok = g.dropna(subset=[pcol, dcol])
                # per-frame normalisation to the frame mean (gauge); frames whose DLITE
                # solution collapsed to ~0 (CellFIT solver failures) are dropped
                fm = ok.groupby("frame")[dcol].transform("mean")
                ok = ok[fm.abs() > 1e-6]
                x = ok[pcol] / ok.groupby("frame")[pcol].transform("mean")
                y = ok[dcol] / ok.groupby("frame")[dcol].transform("mean")
                good = np.isfinite(x) & np.isfinite(y)
                x, y, ok = x[good], y[good], ok[good]
                if len(ok) < 5:
                    continue
                per_frame = []
                for _, gf in ok.groupby("frame"):
                    if len(gf) >= 5:
                        per_frame.append(spearmanr(gf[pcol], gf[dcol])[0])
                rows.append(
                    {
                        "series": series,
                        "pimorph_setting": name,
                        "dlite_solver": s,
                        "n_edges_pooled": int(len(ok)),
                        "n_frames": int(ok["frame"].nunique()),
                        "spearman_pooled": float(spearmanr(x, y)[0]) if len(ok) > 4 else float("nan"),
                        "pearson_pooled": float(pearsonr(x, y)[0]) if len(ok) > 4 else float("nan"),
                        "spearman_per_frame_mean": float(np.nanmean(per_frame)) if per_frame else float("nan"),
                        "spearman_per_frame_sd": float(np.nanstd(per_frame)) if per_frame else float("nan"),
                        "frac_frames_positive": float(np.mean(np.asarray(per_frame) > 0)) if per_frame else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def consistency_table(df: pd.DataFrame, solvers: List[str]) -> pd.DataFrame:
    rows = []
    for series, g in df.groupby("series"):
        cols = [f"dlite_{s}" for s in solvers] + [f"pimorph_{n}" for n in SETTINGS]
        for c in cols:
            r = consistency(g, c)
            r.update({"series": series, "method": c})
            rows.append(r)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, summ: pd.DataFrame, cons: pd.DataFrame, out: Path, solver: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax = axs[0]
    pcol, dcol = "pimorph_tension_only_r0.1", f"dlite_{solver}"
    for series, g in df.groupby("series"):
        ok = g.dropna(subset=[pcol, dcol])
        fm = ok.groupby("frame")[dcol].transform("mean")
        ok = ok[fm.abs() > 1e-6]
        x = ok[pcol] / ok.groupby("frame")[pcol].transform("mean")
        y = ok[dcol] / ok.groupby("frame")[dcol].transform("mean")
        ax.scatter(y, x, s=6, alpha=0.5, label=f"series {series} (n = {len(ok)})")
    ax.plot([0, 3], [0, 3], "--", c="grey", lw=0.8)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel(f"DLITE ({solver} solver) tension / frame mean")
    ax.set_ylabel("PiMorph tension / frame mean (tension only, ridge 0.1)")
    ax.legend(fontsize=8)
    ax.set_title("same traced edges, per frame")
    ax = axs[1]
    piv = cons.pivot(index="series", columns="method", values="pearson_consecutive_mean")
    piv.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Pearson r of normalised tensions, consecutive frames (mean)")
    ax.set_title("temporal consistency (DLITE paper metric)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "dlite_vs_pimorph.png", dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlite", default="data/dlite/DLITE-master")
    ap.add_argument("--out", default="runs/mechanics_real")
    ap.add_argument("--series", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--solvers", nargs="+", default=["CellFIT", "DLITE"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dlite_root = Path(a.dlite).resolve()
    frames_all: List[pd.DataFrame] = []
    meta: Dict[str, object] = {}
    for s in a.series:
        t0 = time.time()
        df, m = process_series(dlite_root, s, a.solvers)
        frames_all.append(df)
        meta[str(s)] = m
        df.to_csv(out / f"dlite_per_edge_series{s}.csv", index=False)
        print(f"series {s}: {len(df)} edge rows, {time.time() - t0:.0f} s", flush=True)
    df = pd.concat([d for d in frames_all if len(d)], ignore_index=True)
    for s in a.solvers:
        if f"dlite_{s}" not in df.columns:
            df[f"dlite_{s}"] = np.nan
    df.to_csv(out / "dlite_per_edge.csv", index=False)
    pd.DataFrame([f for s in meta.values() for f in s["frames"]]).to_csv(out / "dlite_per_frame.csv", index=False)
    summ = summarize(df, a.solvers)
    summ.to_csv(out / "dlite_agreement.csv", index=False)
    cons = consistency_table(df, a.solvers)
    cons.to_csv(out / "dlite_consistency.csv", index=False)
    (out / "dlite_meta.json").write_text(
        json.dumps(
            {
                "seconds": {k: v["seconds"] for k, v in meta.items()},
                "failed_solvers": {k: v.get("failed", {}) for k, v in meta.items()},
                "settings": SETTINGS,
            },
            indent=2,
        )
    )
    print(summ.to_string())
    print(cons.to_string())
    figure(df, summ, cons, out, a.solvers[-1])


if __name__ == "__main__":
    main()
