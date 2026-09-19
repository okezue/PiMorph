#!/usr/bin/env python3
"""Barrier proxy against measured TER on the NIST / NEI AMD-iRPE tiles.

For every registered 256 x 256 tile (10 wells, ZO-1 fluorescence plus hand-corrected
border mask, see ``pimorph.io.rpe_nist``): build the complex from the curated mask and,
independently, from the PiMorph neural reconstruction of the ZO-1 image; profile ZO-1
along every cell-cell interface (TJ channel); run ``function.barrier.barrier_report``.
Tile statistics are aggregated per well and correlated with the well's measured TER.

Images are upsampled 2x before profiling and reconstruction (cells are about 19 px
across at native resolution). The pixel size is unknown, so areas are in native px^2.

Usage:
    python scripts/pimorph_function_rpe_ter.py [--data data/rpe_nist] [--out runs/function_real]
        [--max-tiles N] [--skip-neural]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pimorph.complex import extract_complex
from pimorph.fields.multichannel import multichannel_profiles
from pimorph.function import barrier_report
from pimorph.infer import ConstrainedDecoder, DecoderParams
from pimorph.io import rpe_nist as R

UPSAMPLE = 2
WEIGHTS = {"TJ": 1.0}
TILE_COLUMNS = [
    "n_cells",
    "n_gaps",
    "n_edges",
    "TJ_coverage",
    "TJ_continuity",
    "TJ_width_mean_px",
    "TJ_intensity_rel",
    "G_eff",
    "permeability_index",
    "gap_fraction",
    "mean_g_edge",
    "cell_area_px",
    "cell_area_cv",
    "edge_length_px",
    "tissue_frac",
]


def tile_stats(cx, zo1_up: np.ndarray, tile_area_px: float) -> Dict[str, float]:
    """Junction-field statistics and barrier numbers of one complex (per native px)."""
    if cx.cell_faces.size < 3 or cx.cell_cell_edges().size < 3:
        return {c: float("nan") for c in TILE_COLUMNS}
    mp = multichannel_profiles(cx, {"TJ": zo1_up}, bin_px=2.0, half_width_px=3.0)
    rep = barrier_report(cx, mp, weights=WEIGHTS)
    pe = mp.per_edge
    w = pe["arclength_px"].to_numpy(dtype=float)
    ok = w > 0
    if not ok.any():
        return {c: float("nan") for c in TILE_COLUMNS}

    def wmean(col: str) -> float:
        v = pe[col].to_numpy(dtype=float)
        m = ok & np.isfinite(v)
        return float(np.average(v[m], weights=w[m])) if m.any() else float("nan")

    from pimorph.complex.geometry import face_area

    areas = np.array([face_area(cx, int(f)) for f in cx.cell_faces], dtype=float) / UPSAMPLE**2
    return {
        "n_cells": int(cx.cell_faces.size),
        "n_gaps": int(cx.gap_faces.size),
        "n_edges": int(len(pe)),
        "TJ_coverage": wmean("TJ_coverage"),
        "TJ_continuity": wmean("TJ_continuity"),
        "TJ_width_mean_px": wmean("TJ_width_mean_px") / UPSAMPLE,
        # strip intensity relative to the tile median, so staining level cancels
        "TJ_intensity_rel": wmean("TJ_mean_intensity") / max(float(np.median(zo1_up)), 1e-6),
        # conductances use arclength in upsampled px; rescale to native px
        "G_eff": float(rep["G_eff"]) / UPSAMPLE,
        "permeability_index": float(rep["permeability_index"]) * UPSAMPLE,
        "gap_fraction": float(rep["gap_fraction"]),
        "mean_g_edge": float(rep["mean_g_edge"]) / UPSAMPLE,
        "cell_area_px": float(areas.mean()),
        "cell_area_cv": float(areas.std() / areas.mean()) if areas.mean() > 0 else float("nan"),
        "edge_length_px": float(np.average(w[ok])) / UPSAMPLE,
        "tissue_frac": float(areas.sum() / tile_area_px),
    }


def run(data: Path, out: Path, max_tiles: Optional[int], skip_neural: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tiles = R.list_tiles(data)
    if max_tiles:
        per_well = max(1, max_tiles // tiles["well"].nunique())
        tiles = tiles.groupby("well", group_keys=False, as_index=False).head(per_well).reset_index(drop=True)
    ter = R.load_ter(data)
    csv_path = out / "rpe_ter_per_tile.csv"
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(prev["tile_id"])
    prop = None
    if not skip_neural:
        from pimorph.infer.neural.proposer import NeuralProposer

        prop = NeuralProposer("models/pimorph_proposals_v6_pool.pt", tta=True)
    tile_area = float(R.TILE_SHAPE[0] * R.TILE_SHAPE[1])
    t0 = time.time()
    rows = []
    for i, t in tiles.iterrows():
        if t.tile_id in done:
            continue
        z = R.read_tile(t.path_zo1).astype(np.float32)
        zu = R.upsample_image(z, UPSAMPLE)
        row = {"tile_id": t.tile_id, "well": t.well, "donor": t.donor, "day": t.day, "row": t.row, "col": t.col}
        m = R.read_tile(t.path_mask)
        gt = R.labels_from_border_mask(m)
        cx_gt = extract_complex(R.upsample_labels(gt, UPSAMPLE))
        for k, v in tile_stats(cx_gt, zu, tile_area).items():
            row[f"curated_{k}"] = v
        if prop is not None:
            maps = prop(zu)
            params = DecoderParams(cell_radius_px=float(maps.meta["cell_radius_px"]), vertex_weight=0.3)
            res = ConstrainedDecoder().decode(maps, params)
            for k, v in tile_stats(res.cx, zu, tile_area).items():
                row[f"neural_{k}"] = v
        rows.append(row)
        if len(rows) % 25 == 0:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            rows = []
            print(f"{len(done) + i + 1} tiles, {time.time() - t0:.0f} s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    per_tile = pd.read_csv(csv_path).drop_duplicates("tile_id")
    per_tile.to_csv(csv_path, index=False)
    summarize(per_tile, ter, out)


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    rho, p = spearmanr(x, y)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        bs.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(rho), float(p), float(lo), float(hi)


def summarize(per_tile: pd.DataFrame, ter: pd.DataFrame, out: Path) -> None:
    from scipy.stats import spearmanr

    prefixes = [p for p in ("curated", "neural") if f"{p}_permeability_index" in per_tile.columns]
    agg = per_tile.groupby("well").agg(n_tiles=("tile_id", "size"))
    for p in prefixes:
        for c in TILE_COLUMNS:
            agg[f"{p}_{c}"] = per_tile.groupby("well")[f"{p}_{c}"].mean()
    agg = agg.join(ter.set_index("well")[["ter_mean", "ter_sd", "donor", "day"]], how="inner").reset_index()
    agg.to_csv(out / "rpe_ter_per_well.csv", index=False)

    results = []
    y = agg["ter_mean"].to_numpy(dtype=float)
    for p in prefixes:
        for c in TILE_COLUMNS:
            x = agg[f"{p}_{c}"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4:
                continue
            rho, pv, lo, hi = spearman_ci(x[ok], y[ok])
            # tile-level association with a well-permutation null (TER shuffled across wells)
            pt = per_tile.dropna(subset=[f"{p}_{c}"])
            tx = pt[f"{p}_{c}"].to_numpy(dtype=float)
            ty = pt["well"].map(dict(zip(agg["well"], agg["ter_mean"]))).to_numpy(dtype=float)
            rho_t = spearmanr(tx, ty)[0]
            rng = np.random.default_rng(1)
            wells = agg["well"].to_numpy()
            null = []
            for _ in range(2000):
                perm = dict(zip(wells, rng.permutation(agg["ter_mean"].to_numpy())))
                null.append(spearmanr(tx, pt["well"].map(perm).to_numpy(dtype=float))[0])
            p_perm = float((np.sum(np.abs(null) >= abs(rho_t)) + 1) / (len(null) + 1))
            results.append(
                {
                    "source": p,
                    "statistic": c,
                    "n_wells": int(ok.sum()),
                    "spearman_well": rho,
                    "p_well": pv,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_tiles": int(len(pt)),
                    "spearman_tile": float(rho_t),
                    "p_tile_wellperm": p_perm,
                }
            )
    res = pd.DataFrame(results)
    res.to_csv(out / "rpe_ter_correlations.csv", index=False)
    # agreement between curated and neural per tile
    agree = {}
    if len(prefixes) == 2:
        for c in TILE_COLUMNS:
            a = per_tile[f"curated_{c}"].to_numpy(dtype=float)
            b = per_tile[f"neural_{c}"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            agree[c] = float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 3 else float("nan")
    (out / "rpe_ter_summary.json").write_text(
        json.dumps(
            {
                "n_tiles": int(len(per_tile)),
                "n_wells": int(len(agg)),
                "ter_range": [float(agg["ter_mean"].min()), float(agg["ter_mean"].max())],
                "curated_vs_neural_tile_spearman": agree,
            },
            indent=2,
        )
    )
    print(res.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rpe_nist")
    ap.add_argument("--out", default="runs/function_real")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        per_tile = pd.read_csv(Path(a.out) / "rpe_ter_per_tile.csv").drop_duplicates("tile_id")
        summarize(per_tile, R.load_ter(Path(a.data)), Path(a.out))
        return
    run(Path(a.data), Path(a.out), a.max_tiles, a.skip_neural)


if __name__ == "__main__":
    main()
