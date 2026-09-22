#!/usr/bin/env python3
"""Barrier proxy against measured TER on the NIST / NEI Healthy-2 iRPE series.

Healthy-2 (``pimorph.io.rpe_nist``, ``data/rpe_nist/healthy2/INVENTORY.md``) is live QBAM
bright-field absorbance of iPSC-RPE imaged weekly during maturation, 36 wells in three
conditions with TER measured on the day of imaging (216 well-timepoints, 96 to 1146 Ohm).
There is no junction stain, so only the STRUCTURAL part of the barrier proxy can be tested:
cells are segmented from the blue absorbance channel with a Cellpose-SAM fine-tuned on the
AMD QBAM tiles, decoded into a cell complex, and the resistor model of
``pimorph.function.transport`` is evaluated with a constant conductance per interface.

Subcommands (in the order they are run):
    export-train    AMD registered tiles -> Cellpose training pairs, held-out wells separate
    eval-segmenter  fine-tuned Cellpose + decoder on the held-out tiles, structural metrics
    inventory       TER long table and tile listing from the Drive listing pickle
    select          stratified subset of well-timepoints and the Drive ids to fetch
    download        fetch the selected Blue 488 tiles with gdown (logged)
    segment         Cellpose proposer + constrained decoder per tile; posterior on a subset
    stats           correlations against TER, controls, leave-one-well-out, figure, report
    qc              12 QBAM / outline crops stratified over TER for a human reviewer

Usage (devbox for the heavy steps):
    python scripts/pimorph_function_healthy2.py export-train
    python runs/cellpose_ft/train_cpsam.py --name cpsam_qbam --dirs runs/cellpose_ft/cpsam_qbam_data/train --uniform
    python scripts/pimorph_function_healthy2.py eval-segmenter --weights runs/cellpose_ft/cpsam_qbam/models/cpsam_qbam
    python scripts/pimorph_function_healthy2.py inventory && ... select && ... download
    python scripts/pimorph_function_healthy2.py segment --weights ... [--posterior-n 20]
    python scripts/pimorph_function_healthy2.py stats && ... qc
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

DATA = Path("data/rpe_nist")
H2 = DATA / "healthy2"
OUT = Path("runs/function_real/healthy2")
CP_DATA = Path("runs/cellpose_ft/cpsam_qbam_data")
CP_WEIGHTS = "runs/cellpose_ft/cpsam_qbam/models/cpsam_qbam"
HOLDOUT_WELLS = ("AMD1_A_D75", "AMD2_B_D73")  # 175 of 1032 tiles (17%), two donors, clones unseen in training
FILTER = "Blue 488"
CENTRE_TILES = ((1, 1), (2, 1))  # adjacent tiles at the centre of the 4 x 3 grid
CENTRE_FALLBACK = ((1, 0), (2, 0), (1, 2), (2, 2), (0, 1), (3, 1))
G_EDGE = 1.0  # constant conductance per cell-cell interface (no junction marker in QBAM)
G_GAP = 10.0
TILE_COLUMNS = [
    "n_cells",
    "cell_density_per_kpx",
    "cell_area_px",
    "cell_area_cv",
    "gap_area_frac",
    "n_gaps",
    "n_edges",
    "edges_per_cell",
    "edge_length_px",
    "G_eff",
    "permeability_index",
    "gap_fraction",
    "G_eff_len",
    "permeability_index_len",
    "in_plane_G",
    "in_plane_G_len",
    "tissue_frac",
]
PROXY_COLUMNS = [
    "permeability_index",
    "G_eff",
    "in_plane_G",
    "permeability_index_len",
    "in_plane_G_len",
    "gap_fraction",
]
CONTROL_COLUMNS = ["cell_density_per_kpx", "cell_area_cv", "gap_area_frac", "edge_length_px", "edges_per_cell"]


# ----------------------------------------------------------------------------- helpers
def _log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def _load_listing() -> pd.DataFrame:
    from pimorph.io import rpe_nist as R

    return R.healthy2_listing(H2 / "healthy2_listing.pkl")


def _ter_long() -> pd.DataFrame:
    from pimorph.io import rpe_nist as R

    return R.healthy2_ter_long(H2 / "meta" / "EVOM.csv")


def read_qbam(path: Path) -> np.ndarray:
    """Absorbance tile as float32; the registered AMD uint8 tiles are rescaled to [0, 1]."""
    import tifffile

    a = np.asarray(tifffile.imread(str(path)))
    if a.ndim == 3:
        a = a[..., -1] if a.shape[-1] in (3, 4) else a[0]
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return a


# ----------------------------------------------------------------------- export-train
def cmd_export_train(args) -> None:
    import tifffile

    from pimorph.io import rpe_nist as R

    tiles = R.list_tiles(DATA)
    split = {"holdout_wells": list(HOLDOUT_WELLS), "train": [], "test": []}
    n_cells = {"train": 0, "test": 0}
    for part in ("train", "test"):
        (CP_DATA / part).mkdir(parents=True, exist_ok=True)
    ignore_frac = []
    for _, t in tiles.iterrows():
        part = "test" if t.well in HOLDOUT_WELLS else "train"
        img = read_qbam(Path(t.path_abs))
        mask = tifffile.imread(t.path_mask)
        lab = R.qbam_instances(mask)
        # unannotated regions: no ignore mask in Cellpose, so flatten their image evidence
        # (as export_cellpose_training does outside ROIs) and keep the boolean for scoring
        ign = ndi.binary_dilation(R.qbam_unannotated(mask), iterations=2) & (lab == 0)
        img[ign] = float(np.median(img[~ign])) if (~ign).any() else float(img.mean())
        ignore_frac.append(float(ign.mean()))
        fid = f"{t.well}__{t.tile_id}"
        tifffile.imwrite(CP_DATA / part / f"{fid}.tif", img)
        tifffile.imwrite(CP_DATA / part / f"{fid}_masks.tif", lab.astype(np.int32))
        tifffile.imwrite(CP_DATA / part / f"{fid}_ignore.tif", ign.astype(np.uint8))
        split[part].append(fid)
        n_cells[part] += int(len(np.unique(lab)) - 1)
    info = {
        "source": "data/rpe_nist/tiles/absorbance-dataset (uint8 blue QBAM / 255) + segmentation_mask, qbam_instances",
        "unannotated_regions": "qbam_unannotated, image set to tile median there, labels 0, saved as <id>_ignore.tif",
        "ignore_frac_median": float(np.median(ignore_frac)),
        "n_train": len(split["train"]),
        "n_test": len(split["test"]),
        "cells_train": n_cells["train"],
        "cells_test": n_cells["test"],
        **split,
    }
    (CP_DATA / "split.json").write_text(json.dumps(info, indent=1))
    print(
        f"train {len(split['train'])} tiles / {n_cells['train']} cells; test {len(split['test'])} / {n_cells['test']}"
    )


# --------------------------------------------------------------------- segmentation
def _proposer(weights: str, device=None):
    from pimorph.infer.cellpose_proposer import CellposeProposer

    return CellposeProposer(pretrained_model=weights, device=device, nucleus_points=False)


def decode_tile(prop, img: np.ndarray, vertex_weight: float = 0.3):
    from pimorph.infer import ConstrainedDecoder, DecoderParams

    maps = prop(img)
    params = DecoderParams(cell_radius_px=float(maps.meta["cell_radius_px"]), vertex_weight=vertex_weight)
    res = ConstrainedDecoder().decode(maps, params)
    return res, maps


def cmd_eval_segmenter(args) -> None:
    import tifffile

    from pimorph.metrics.structural import structural_metrics

    split = json.loads((CP_DATA / "split.json").read_text())
    ids = split["test"]
    if args.max_items:
        ids = ids[: args.max_items]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "segmenter_holdout_per_tile.csv"
    done = set(pd.read_csv(csv_path)["tile"]) if csv_path.exists() else set()
    prop = _proposer(args.weights)
    rows = []
    t0 = time.time()
    for i, fid in enumerate(ids):
        if fid in done:
            continue
        img = read_qbam(CP_DATA / "test" / f"{fid}.tif")
        gt = tifffile.imread(str(CP_DATA / "test" / f"{fid}_masks.tif")).astype(np.int32)
        ign_p = CP_DATA / "test" / f"{fid}_ignore.tif"
        ign = tifffile.imread(str(ign_p)).astype(bool) if ign_p.exists() else np.zeros(gt.shape, bool)
        res, maps = decode_tile(prop, img)
        pred = np.where(ign, 0, res.labels.astype(np.int32))
        m = structural_metrics(pred, gt)
        raw_m = structural_metrics(np.where(ign, 0, prop.raw(img).masks.astype(np.int32)), gt) if args.raw_masks else {}
        row = {
            "tile": fid,
            "well": fid.split("__")[0],
            "n_pred": int(len(np.unique(pred[pred > 0]))),
            "n_true": int(len(np.unique(gt[gt > 0]))),
            "ignore_frac": float(ign.mean()),
        }
        row.update({k: v for k, v in m.items() if isinstance(v, (int, float, bool))})
        row.update({f"raw_{k}": v for k, v in raw_m.items() if k in ("ap50", "adjacency_pair_f1", "vertex_f1", "pq")})
        rows.append(row)
        if len(rows) % 20 == 0 or i == len(ids) - 1:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            rows = []
            print(f"{i + 1}/{len(ids)} tiles, {time.time() - t0:.0f} s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    df = pd.read_csv(csv_path).drop_duplicates("tile")
    df.to_csv(csv_path, index=False)
    prefixes = ("ap50", "ap75", "pq", "adjacency", "vertex", "boundary_f1", "vi", "raw_")
    keys = [c for c in df.columns if any(c.startswith(k) for k in prefixes) and df[c].dtype.kind in "fi"]
    summary = {
        "weights": args.weights,
        "n_tiles": int(len(df)),
        "wells": sorted(df["well"].unique().tolist()),
        "n_pred_total": int(df["n_pred"].sum()),
        "n_true_total": int(df["n_true"].sum()),
        "mean": {k: float(df[k].mean()) for k in keys},
        "per_well": {w: {k: float(g[k].mean()) for k in keys} for w, g in df.groupby("well")},
    }
    (out / "segmenter_holdout_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["mean"], indent=1))


# ---------------------------------------------------------------------- inventory
def cmd_inventory(args) -> None:
    lst = _load_listing()
    ter = _ter_long()
    lst.to_csv(H2 / "tile_listing.csv", index=False)
    folder = (
        lst[lst["filter"] == FILTER]
        .groupby(["plate", "well", "date"])
        .agg(n_tiles=("drive_id", "size"), drive_folder=("drive_path", lambda s: os.path.dirname(s.iloc[0])))
        .reset_index()
    )
    ter = ter.merge(folder, on=["plate", "well", "date"], how="left")
    ter.to_csv(H2 / "ter_long.csv", index=False)
    print(
        f"{len(lst)} absorbance tiles, {lst['plate'].nunique()} plates, {lst['well_id'].nunique()} wells, "
        f"{lst['date'].nunique()} dates, filters {sorted(lst['filter'].unique())}"
    )
    print(
        f"TER: {len(ter)} well-timepoints, {ter['well_id'].nunique()} wells, range {ter.ter_ohm.min():.0f} to "
        f"{ter.ter_ohm.max():.0f} Ohm; without images: {int(ter['n_tiles'].isna().sum())}"
    )
    print(ter.groupby(["condition", "week"])["ter_ohm"].agg(["count", "min", "median", "max"]).to_string())


# ------------------------------------------------------------------------ select
def cmd_select(args) -> None:
    ter = pd.read_csv(H2 / "ter_long.csv")
    ter = ter[ter["n_tiles"].fillna(0) > 0].copy()
    lst = _load_listing()
    blue = lst[lst["filter"] == FILTER]
    rng = np.random.default_rng(0)
    early, late = [3, 4, 5], [6, 7, 8]
    wells = sorted(ter["well_id"].unique())
    chosen = []
    # two dates per well, one early and one late, round-robin so each week gets 12 wells
    for i, w in enumerate(wells):
        sub = ter[ter.well_id == w]
        for group, k in ((early, i % 3), (late, (i // 3) % 3)):
            wk = group[k]
            row = sub[sub.week == wk]
            if row.empty:
                row = sub[sub.week.isin(group)].sample(1, random_state=int(rng.integers(1 << 30)))
            chosen.append(row.index[0])
    sel = ter.loc[sorted(set(chosen))].copy()
    # fill the least populated TER bins up to the target size
    bins = [0, 130, 200, 350, 550, 800, 1000, 2000]
    sel_bin = pd.cut(sel.ter_ohm, bins)
    rest = ter.drop(sel.index)
    rest_bin = pd.cut(rest.ter_ohm, bins)
    while len(sel) < args.n and len(rest):
        counts = sel_bin.value_counts()
        order = [b for b in counts.sort_values().index if (rest_bin == b).any()]
        if not order:
            break
        cand = rest[rest_bin == order[0]]
        # prefer wells with the fewest selected timepoints
        per_well = sel.well_id.value_counts()
        cand = cand.assign(k=cand.well_id.map(per_well).fillna(0)).sort_values(["k", "ter_ohm"])
        pick = cand.index[0]
        sel = pd.concat([sel, ter.loc[[pick]]])
        sel_bin = pd.cut(sel.ter_ohm, bins)
        rest = rest.drop(pick)
        rest_bin = pd.cut(rest.ter_ohm, bins)
    rows = []
    for _, r in sel.sort_values(["condition", "well_id", "week"]).iterrows():
        avail = blue[(blue.plate == r.plate) & (blue.well == r.well) & (blue.date == r.date)]
        have = {(int(a.grid_r), int(a.grid_c)): a for _, a in avail.iterrows()}
        picks = [p for p in CENTRE_TILES if p in have]
        for p in CENTRE_FALLBACK:
            if len(picks) >= 2:
                break
            if p in have and p not in picks:
                picks.append(p)
        for gr, gc in picks[:2]:
            a = have[(gr, gc)]
            rows.append({**r.to_dict(), "grid_r": gr, "grid_c": gc, "drive_id": a.drive_id, "drive_path": a.drive_path})
    out = pd.DataFrame(rows)
    from pimorph.io import rpe_nist as R

    out["file"] = [R.healthy2_tile_name(r.plate, r.well, r.date, FILTER, r.grid_r, r.grid_c) for _, r in out.iterrows()]
    out["wt_id"] = out["well_id"] + "_" + out["date"]
    out.to_csv(H2 / "selection.csv", index=False)
    wt = out.drop_duplicates("wt_id")
    print(f"{len(wt)} well-timepoints, {len(out)} tiles, TER {wt.ter_ohm.min():.0f} to {wt.ter_ohm.max():.0f}")
    print(wt.groupby("condition")["ter_ohm"].describe()[["count", "min", "50%", "max"]].to_string())
    print(wt.week.value_counts().sort_index().to_dict(), pd.cut(wt.ter_ohm, bins).value_counts().sort_index().to_dict())


# ---------------------------------------------------------------------- download
def cmd_download(args) -> None:
    import gdown

    sel = pd.read_csv(H2 / "selection.csv")
    tdir = H2 / "tiles"
    tdir.mkdir(parents=True, exist_ok=True)
    log = H2 / "download.log"
    n_ok = 0
    for _, r in sel.iterrows():
        dst = tdir / r.file
        if dst.exists() and dst.stat().st_size > 1_000_000:
            n_ok += 1
            continue
        for attempt in range(3):
            try:
                gdown.download(id=r.drive_id, output=str(dst), quiet=True)
                if dst.exists() and dst.stat().st_size > 1_000_000:
                    break
            except Exception as e:  # noqa: BLE001
                _log(log, f"retry {attempt} {r.file} id={r.drive_id}: {e}")
                time.sleep(5)
        if dst.exists() and dst.stat().st_size > 1_000_000:
            n_ok += 1
            _log(log, f"got {dst} id={r.drive_id} src={r.drive_path} size={dst.stat().st_size}")
        else:
            _log(log, f"FAILED {r.file} id={r.drive_id}")
        if n_ok % 20 == 0:
            print(f"{n_ok}/{len(sel)}", flush=True)
    print(f"{n_ok}/{len(sel)} tiles present")


# ----------------------------------------------------------------------- segment
def complex_stats(cx, tile_area_px: float) -> Dict[str, float]:
    """Structural statistics and resistor-model numbers of one complex with a CONSTANT
    conductance per cell-cell interface (``G_EDGE``) and, as a variant, conductance
    proportional to interface length. Gaps (background faces enclosed by cells) are short
    circuits of ``G_GAP``. In-plane conductance is solved between the cells touching the
    left and the right image border."""
    from pimorph.complex.geometry import edge_arclengths, face_area, face_centroid
    from pimorph.function.transport import (
        boundary_cells,
        conductance_breakdown,
        in_plane_effective_conductance,
        permeability_index,
        tissue_area,
    )

    nan = {c: float("nan") for c in TILE_COLUMNS}
    if cx.cell_faces.size < 3:
        return nan
    cc = cx.cell_cell_edges()
    if cc.size < 3:
        return nan
    areas = np.array([face_area(cx, int(f)) for f in cx.cell_faces], dtype=float)
    L = edge_arclengths(cx)
    g_const = np.full(cx.n_edges, G_EDGE, dtype=float)
    g_len = np.where(np.isfinite(L), L, 0.0) / max(float(np.median(L[cc])), 1e-6)
    brk = conductance_breakdown(cx, g_const, g_gap=G_GAP)
    brk_len = conductance_breakdown(cx, g_len, g_gap=G_GAP)
    area, _ = tissue_area(cx)
    # source and sink: boundary cells split by centroid x (left third vs right third)
    bc = boundary_cells(cx)
    in_plane = in_plane_len = float("nan")
    if bc.size >= 2:
        xs = np.array([face_centroid(cx, int(f))[1] for f in bc])
        src = [int(f) for f, x in zip(bc, xs) if x < xs.min() + 0.34 * (xs.max() - xs.min())]
        snk = [int(f) for f, x in zip(bc, xs) if x > xs.max() - 0.34 * (xs.max() - xs.min())]
        if src and snk and not set(src) & set(snk):
            in_plane = float(in_plane_effective_conductance(cx, g_const, src, snk, gap_conductance=G_GAP))
            in_plane_len = float(in_plane_effective_conductance(cx, g_len, src, snk, gap_conductance=G_GAP))
    n_cells = int(cx.cell_faces.size)
    return {
        "n_cells": n_cells,
        "cell_density_per_kpx": 1000.0 * n_cells / tile_area_px,
        "cell_area_px": float(areas.mean()),
        "cell_area_cv": float(areas.std() / areas.mean()) if areas.mean() > 0 else float("nan"),
        "gap_area_frac": float(1.0 - areas.sum() / tile_area_px),
        "n_gaps": int(cx.gap_faces.size),
        "n_edges": int(cc.size),
        "edges_per_cell": float(cc.size / n_cells),
        "edge_length_px": float(np.mean(L[cc])),
        "G_eff": float(brk["G_eff"]),
        "permeability_index": float(permeability_index(cx, g_const, g_gap=G_GAP, area=area)),
        "gap_fraction": float(brk["gap_fraction"]),
        "G_eff_len": float(brk_len["G_eff"]),
        "permeability_index_len": float(permeability_index(cx, g_len, g_gap=G_GAP, area=area)),
        "in_plane_G": in_plane,
        "in_plane_G_len": in_plane_len,
        "tissue_frac": float(areas.sum() / tile_area_px),
    }


def cmd_segment(args) -> None:
    import tifffile

    sel = pd.read_csv(H2 / "selection.csv")
    if args.max_items:
        sel = sel.head(args.max_items)
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    csv_path = out / "per_tile.csv"
    done = set(pd.read_csv(csv_path)["file"]) if csv_path.exists() else set()
    prop = _proposer(args.weights)
    rows = []
    t0 = time.time()
    for i, r in sel.iterrows():
        if r.file in done:
            continue
        p = H2 / "tiles" / r.file
        if not p.exists():
            continue
        img = read_qbam(p)
        if args.crop:
            h, w = img.shape
            img = img[(h - args.crop) // 2 : (h + args.crop) // 2, (w - args.crop) // 2 : (w + args.crop) // 2]
        res, maps = decode_tile(prop, img)
        row = {
            k: r[k]
            for k in (
                "wt_id",
                "well_id",
                "plate",
                "well",
                "condition",
                "treatment",
                "date",
                "week",
                "ter_ohm",
                "grid_r",
                "grid_c",
                "file",
            )
        }
        row["source"] = maps.source
        row["cell_radius_px"] = float(maps.meta["cell_radius_px"])
        row["n_raw_masks"] = int(maps.meta.get("n_masks", 0))
        row.update(complex_stats(res.cx, float(img.size)))
        rows.append(row)
        tifffile.imwrite(
            out / "labels" / r.file.replace(".tif", "_labels.tif"), res.labels.astype(np.uint16), compression="zlib"
        )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            rows = []
            print(f"{i + 1}/{len(sel)} tiles, {time.time() - t0:.0f} s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    df = pd.read_csv(csv_path).drop_duplicates("file")
    df.to_csv(csv_path, index=False)
    print(f"{len(df)} tiles segmented, {time.time() - t0:.0f} s")
    if args.posterior_n:
        run_posterior(prop, sel, out, args.posterior_n, args.posterior_crop)


def run_posterior(prop, sel: pd.DataFrame, out: Path, n: int, crop: int) -> None:
    """Credible intervals of the proxy on ``n`` tiles (one per well-timepoint, stratified
    over TER) from Cellpose's threshold family + decoder perturbations (``cellpose_hypotheses``)
    weighted by the complex energy (``PosteriorEnsemble``). A centre crop keeps the
    hypothesis family (tens of decodes per tile) affordable."""
    from pimorph.infer import ConstrainedDecoder, DecoderParams, PosteriorEnsemble
    from pimorph.infer.cellpose_proposer import cellpose_hypotheses

    csv_path = out / "posterior_tiles.csv"
    done = set(pd.read_csv(csv_path)["file"]) if csv_path.exists() else set()
    first = sel[sel.grid_r == sel.groupby("wt_id")["grid_r"].transform("min")].drop_duplicates("wt_id")
    first = first.sort_values("ter_ohm")
    idx = np.unique(np.round(np.linspace(0, len(first) - 1, n)).astype(int))
    picks = first.iloc[idx]
    t0 = time.time()
    for _, r in picks.iterrows():
        if r.file in done:
            continue
        p = H2 / "tiles" / r.file
        if not p.exists():
            continue
        img = read_qbam(p)
        h, w = img.shape
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        base = DecoderParams(cell_radius_px=15.0, vertex_weight=0.3)
        hyps, raw, ref = cellpose_hypotheses(
            prop, img, None, ConstrainedDecoder(), base, image=img, n_merge_moves=3, n_split_moves=3, max_per_setting=4
        )
        post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=4)
        area = float(img.size)
        stats_cache: Dict[int, Dict[str, float]] = {}

        def stat(name):
            def f(hyp):
                key = id(hyp)
                if key not in stats_cache:
                    stats_cache[key] = complex_stats(hyp.cx, area)
                return stats_cache[key][name]

            return f

        row = {k: r[k] for k in ("wt_id", "well_id", "condition", "week", "ter_ohm", "file")}
        row.update({"n_hypotheses": len(hyps), "ess": float(post.meta["ess"]), "temperature": float(post.temperature)})
        for c in (
            "n_cells",
            "cell_density_per_kpx",
            "cell_area_cv",
            "permeability_index",
            "G_eff",
            "in_plane_G",
            "gap_fraction",
        ):
            row[f"{c}_map"] = stat(c)(post.map_hypothesis)
            row[f"{c}_mean"] = post.expectation(stat(c))
            lo, hi = post.credible_interval(stat(c), level=0.9)
            row[f"{c}_ci90_lo"], row[f"{c}_ci90_hi"] = lo, hi
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        print(f"posterior {r.file}: {len(hyps)} hyps, ess {post.meta['ess']:.1f}, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------- stats
def _boot_ci(x, y, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3:
            continue
        vals.append(fn(x[idx], y[idx])[0])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _partial_corr(x, y, covariates: pd.DataFrame, rank: bool = True):
    """Partial correlation of x and y given the covariate dummies (rank-based when ``rank``);
    p from the t distribution with n - k - 2 degrees of freedom."""
    from scipy import stats as st

    Z = np.column_stack([np.ones(len(x)), pd.get_dummies(covariates, drop_first=True).to_numpy(dtype=float)])
    xx = st.rankdata(x) if rank else np.asarray(x, float)
    yy = st.rankdata(y) if rank else np.asarray(y, float)
    rx = xx - Z @ np.linalg.lstsq(Z, xx, rcond=None)[0]
    ry = yy - Z @ np.linalg.lstsq(Z, yy, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(x) - Z.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    p = float(2 * st.t.sf(abs(t), dof))
    return r, p, dof


def _lowo_r2(df: pd.DataFrame, cols: List[str], y: str = "ter_ohm", group: str = "well_id"):
    """Leave-one-well-out linear prediction of TER from ``cols``; returns pooled R^2 and the
    Spearman / Pearson of predictions against TER."""
    from scipy import stats as st

    preds = np.full(len(df), np.nan)
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    yv = df[y].to_numpy(float)
    for g in df[group].unique():
        te = (df[group] == g).to_numpy()
        beta = np.linalg.lstsq(X[~te], yv[~te], rcond=None)[0]
        preds[te] = X[te] @ beta
    ss_res = float(np.sum((yv - preds) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return 1 - ss_res / ss_tot, float(st.spearmanr(preds, yv)[0]), float(st.pearsonr(preds, yv)[0])


def aggregate(per_tile: pd.DataFrame) -> pd.DataFrame:
    """Mean over tiles per well-timepoint, with the TER and design columns."""
    keys = ["wt_id", "well_id", "plate", "well", "condition", "treatment", "date", "week", "ter_ohm"]
    agg = per_tile.groupby("wt_id").agg(n_tiles=("file", "size"), **{c: (c, "mean") for c in TILE_COLUMNS})
    agg = agg.join(
        per_tile.drop_duplicates("wt_id").set_index("wt_id")[[k for k in keys if k != "wt_id"]]
    ).reset_index()
    agg["log_ter"] = np.log10(agg["ter_ohm"])
    return agg


def fill_empty(per_tile: pd.DataFrame) -> pd.DataFrame:
    """Tiles where fewer than three cells were found (the segmenter sees almost nothing in
    unpigmented HPI4 wells) counted as empty tissue: count-like statistics 0, gap fraction 1."""
    t = per_tile.copy()
    empty = ~np.isfinite(t["permeability_index"])
    zero = [
        "n_cells",
        "cell_density_per_kpx",
        "n_gaps",
        "n_edges",
        "G_eff",
        "permeability_index",
        "G_eff_len",
        "permeability_index_len",
        "in_plane_G",
        "in_plane_G_len",
        "tissue_frac",
    ]
    t.loc[empty, zero] = 0.0
    t.loc[empty, "gap_area_frac"] = 1.0
    return t


def corr_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Association of every tile statistic with TER across well-timepoints."""
    from scipy import stats as st

    y = agg["ter_ohm"].to_numpy(float)
    results = []
    for c in TILE_COLUMNS:
        x = agg[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or np.nanstd(x[ok]) == 0:
            continue
        xs, ys = x[ok], y[ok]
        rho, p_rho = st.spearmanr(xs, ys)
        r, p_r = st.pearsonr(xs, np.log10(ys))
        lo, hi = _boot_ci(xs, ys, st.spearmanr)
        sub = agg[ok]
        pc_r, pc_p, pc_dof = _partial_corr(xs, ys, sub[["condition"]])
        pw_r, pw_p, _ = _partial_corr(xs, ys, sub[["week"]].astype(str))
        pcw_r, pcw_p, pcw_dof = _partial_corr(xs, ys, sub[["condition", "week"]].astype(str))
        # partial correlation given the trivial structural controls
        ctrl = (
            sub[["cell_density_per_kpx", "cell_area_cv"]] if c not in ("cell_density_per_kpx", "cell_area_cv") else None
        )
        if ctrl is not None and np.isfinite(ctrl.to_numpy(float)).all(axis=1).sum() > 6:
            okc = np.isfinite(ctrl.to_numpy(float)).all(axis=1)
            Z = np.column_stack([np.ones(int(okc.sum())), ctrl.to_numpy(float)[okc]])
            rx = st.rankdata(xs[okc]) - Z @ np.linalg.lstsq(Z, st.rankdata(xs[okc]), rcond=None)[0]
            ry = st.rankdata(ys[okc]) - Z @ np.linalg.lstsq(Z, st.rankdata(ys[okc]), rcond=None)[0]
            pd_r = float(np.corrcoef(rx, ry)[0, 1])
            dof = int(okc.sum()) - 3 - 1
            pd_p = float(2 * st.t.sf(abs(pd_r * np.sqrt(dof / max(1 - pd_r**2, 1e-12))), dof))
        else:
            pd_r, pd_p = float("nan"), float("nan")
        per_cond = {}
        for cond, g in sub.groupby("condition"):
            if len(g) >= 5 and g[c].std() > 0:
                rr, pp = st.spearmanr(g[c], g["ter_ohm"])
                per_cond[cond] = (float(rr), float(pp), int(len(g)))
        # leave-one-well-out: range of Spearman with each well removed, and pooled prediction
        lowo_rho = []
        for w in sub["well_id"].unique():
            m = (sub["well_id"] != w).to_numpy()
            lowo_rho.append(st.spearmanr(xs[m], ys[m])[0])
        r2, lowo_srho, lowo_prho = _lowo_r2(sub.assign(**{c: xs}), [c])
        results.append(
            {
                "statistic": c,
                "kind": "proxy" if c in PROXY_COLUMNS else ("control" if c in CONTROL_COLUMNS else "other"),
                "n": int(ok.sum()),
                "n_wells": int(sub["well_id"].nunique()),
                "spearman": float(rho),
                "p_spearman": float(p_rho),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "pearson_log_ter": float(r),
                "p_pearson": float(p_r),
                "partial_rho_condition": pc_r,
                "p_partial_condition": pc_p,
                "partial_rho_week": pw_r,
                "p_partial_week": pw_p,
                "partial_rho_condition_week": pcw_r,
                "p_partial_condition_week": pcw_p,
                "partial_rho_given_density_areacv": pd_r,
                "p_partial_given_density_areacv": pd_p,
                "rho_Control": per_cond.get("Control", (np.nan,) * 3)[0],
                "p_Control": per_cond.get("Control", (np.nan,) * 3)[1],
                "rho_Aphidicolin": per_cond.get("Aphidicolin", (np.nan,) * 3)[0],
                "p_Aphidicolin": per_cond.get("Aphidicolin", (np.nan,) * 3)[1],
                "rho_HPI4": per_cond.get("HPI4", (np.nan,) * 3)[0],
                "p_HPI4": per_cond.get("HPI4", (np.nan,) * 3)[1],
                "lowo_spearman_min": float(np.min(lowo_rho)),
                "lowo_spearman_max": float(np.max(lowo_rho)),
                "lowo_pred_r2": float(r2),
                "lowo_pred_spearman": lowo_srho,
            }
        )
    return pd.DataFrame(results)


def cmd_stats(args) -> None:
    out = Path(args.out)
    raw = pd.read_csv(out / "per_tile.csv")
    per_tile = raw[np.isfinite(raw["permeability_index"])]
    agg = aggregate(per_tile)
    agg.to_csv(out / "per_well_timepoint.csv", index=False)
    res = corr_table(agg)
    res.to_csv(out / "ter_correlations.csv", index=False)
    # sensitivity: every selected well-timepoint, empty tiles counted as no visible tissue
    agg_all = aggregate(fill_empty(raw))
    agg_all.to_csv(out / "per_well_timepoint_all.csv", index=False)
    corr_table(agg_all).to_csv(out / "ter_correlations_all_filled.csv", index=False)
    # model comparison: density alone vs density + area CV vs + proxy (leave-one-well-out R^2)
    models = {
        "density": ["cell_density_per_kpx"],
        "density+areaCV": ["cell_density_per_kpx", "cell_area_cv"],
        "density+areaCV+gapfrac": ["cell_density_per_kpx", "cell_area_cv", "gap_area_frac"],
        "permeability_index": ["permeability_index"],
        "in_plane_G": ["in_plane_G"],
        "density+areaCV+permeability": ["cell_density_per_kpx", "cell_area_cv", "permeability_index"],
        "density+areaCV+in_plane_G": ["cell_density_per_kpx", "cell_area_cv", "in_plane_G"],
        "density+areaCV+permeability+in_plane_G": [
            "cell_density_per_kpx",
            "cell_area_cv",
            "permeability_index",
            "in_plane_G",
        ],
    }
    ok = np.all([np.isfinite(agg[c]) for cols in models.values() for c in cols], axis=0)
    sub = agg[ok].reset_index(drop=True)
    mrows = []
    for name, cols in models.items():
        r2, srho, prho = _lowo_r2(sub, cols)
        r2c, _, _ = _lowo_r2(
            sub.assign(
                **{f"c_{k}": v for k, v in pd.get_dummies(sub["condition"], drop_first=True, dtype=float).items()}
            ),
            cols + [f"c_{k}" for k in pd.get_dummies(sub["condition"], drop_first=True).columns],
        )
        mrows.append(
            {
                "model": name,
                "n": len(sub),
                "lowo_r2": r2,
                "lowo_r2_with_condition": r2c,
                "lowo_pred_spearman": srho,
                "lowo_pred_pearson": prho,
            }
        )
    mod = pd.DataFrame(mrows)
    mod.to_csv(out / "model_comparison_lowo.csv", index=False)
    print(
        res[
            [
                "statistic",
                "n",
                "spearman",
                "p_spearman",
                "ci95_lo",
                "ci95_hi",
                "partial_rho_condition",
                "p_partial_condition",
                "partial_rho_condition_week",
                "partial_rho_given_density_areacv",
                "lowo_spearman_min",
                "lowo_spearman_max",
                "lowo_pred_r2",
            ]
        ].to_string(float_format=lambda v: f"{v:.3f}")
    )
    print(mod.to_string(float_format=lambda v: f"{v:.3f}"))
    make_figure(agg, out / "proxy_vs_ter.png")
    write_report(agg, res, mod, out)


def make_figure(agg: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as st

    panels = [
        ("permeability_index", "permeability index (G_eff per px^2, constant g per interface)"),
        ("in_plane_G", "in-plane conductance (left to right boundary cells)"),
        ("gap_fraction", "gap share of G_eff"),
        ("cell_density_per_kpx", "cell density (cells per 1000 px^2)"),
        ("cell_area_cv", "cell area CV"),
        ("gap_area_frac", "gap area fraction"),
    ]
    colors = {"Control": "tab:blue", "Aphidicolin": "tab:green", "HPI4": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (c, label) in zip(axes.ravel(), panels):
        for cond, g in agg.groupby("condition"):
            ax.scatter(
                g[c],
                g["ter_ohm"],
                s=18 + 6 * (g["week"] - 3),
                c=colors.get(cond, "k"),
                alpha=0.75,
                label=cond,
                edgecolor="none",
            )
        ok = np.isfinite(agg[c])
        rho, p = st.spearmanr(agg.loc[ok, c], agg.loc[ok, "ter_ohm"])
        ax.set_title(f"Spearman {rho:.2f} (p = {p:.2g}, n = {int(ok.sum())})", fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("TER (Ohm)")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8, title="marker size = week 3 to 8", title_fontsize=8)
    fig.suptitle(
        "Healthy-2 iRPE: structural barrier proxy from QBAM segmentation vs measured TER (per well-timepoint)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(agg: pd.DataFrame, res: pd.DataFrame, mod: pd.DataFrame, out: Path) -> None:
    """Tables and numbers for REPORT.md (written to ``tables.md``; the report prose is
    written by hand from these)."""
    seg = {}
    sp = out / "segmenter_holdout_summary.json"
    if sp.exists():
        seg = json.loads(sp.read_text())
    post = pd.read_csv(out / "posterior_tiles.csv") if (out / "posterior_tiles.csv").exists() else None
    r = res.set_index("statistic")
    lines = [
        "## Data",
        "",
        f"{len(agg)} well-timepoints, {agg['well_id'].nunique()} wells, weeks {int(agg['week'].min())} to "
        f"{int(agg['week'].max())}, TER {agg['ter_ohm'].min():.0f} to {agg['ter_ohm'].max():.0f} Ohm, "
        f"{int(agg['n_tiles'].sum())} tiles.",
        "",
        "## Segmenter on held-out AMD wells",
        "",
    ]
    if seg:
        m = seg["mean"]
        keys = (
            "ap50",
            "ap75",
            "pq",
            "adjacency_pair_f1",
            "vertex_f1",
            "vertex_precision",
            "vertex_recall",
            "boundary_f1",
        )
        lines.append(
            f"wells {', '.join(seg['wells'])}: {seg['n_tiles']} tiles, {seg['n_true_total']} true cells, "
            f"{seg['n_pred_total']} predicted"
        )
        lines.append("")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * len(keys))
        lines.append("| " + " | ".join(f"{m.get(k, float('nan')):.3f}" for k in keys) + " |")
        for w, mw in seg.get("per_well", {}).items():
            lines.append(f"| {w}: " + " | ".join(f"{mw.get(k, float('nan')):.3f}" for k in keys) + " |")
    lines += [
        "",
        "## Correlation with TER across well-timepoints",
        "",
        "| statistic | kind | n | Spearman | p | 95% CI | Pearson (log TER) | partial rho, condition | p "
        "| partial rho, condition + week | p | partial rho given density + area CV | p | LOWO Spearman range "
        "| LOWO pred. R^2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, row in r.iterrows():
        lines.append(
            f"| {c} | {row['kind']} | {int(row['n'])} | {row['spearman']:.3f} | {row['p_spearman']:.2g} | "
            f"[{row['ci95_lo']:.2f}, {row['ci95_hi']:.2f}] | {row['pearson_log_ter']:.3f} | "
            f"{row['partial_rho_condition']:.3f} | {row['p_partial_condition']:.2g} | "
            f"{row['partial_rho_condition_week']:.3f} | {row['p_partial_condition_week']:.2g} | "
            f"{row['partial_rho_given_density_areacv']:.3f} | {row['p_partial_given_density_areacv']:.2g} | "
            f"[{row['lowo_spearman_min']:.2f}, {row['lowo_spearman_max']:.2f}] | {row['lowo_pred_r2']:.3f} |"
        )
    lines += ["", "## Per-condition Spearman", "", "| statistic | Control | Aphidicolin | HPI4 |", "|---|---|---|---|"]
    for c, row in r.iterrows():
        lines.append(
            f"| {c} | {row['rho_Control']:.2f} (p {row['p_Control']:.2g}) | {row['rho_Aphidicolin']:.2f} "
            f"(p {row['p_Aphidicolin']:.2g}) | {row['rho_HPI4']:.2f} (p {row['p_HPI4']:.2g}) |"
        )
    lines += [
        "",
        "## Leave-one-well-out linear prediction of TER",
        "",
        "| model | n | LOWO R^2 | LOWO R^2 with condition dummies | Spearman(pred, TER) |",
        "|---|---|---|---|---|",
    ]
    for _, row in mod.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['n'])} | {row['lowo_r2']:.3f} | {row['lowo_r2_with_condition']:.3f} | "
            f"{row['lowo_pred_spearman']:.3f} |"
        )
    if post is not None and len(post):
        from scipy.stats import spearmanr

        def relw(c):
            return float(np.nanmedian((post[f"{c}_ci90_hi"] - post[f"{c}_ci90_lo"]) / post[f"{c}_mean"]))

        lines += [
            "",
            "## Posterior credible intervals",
            "",
            f"{len(post)} tiles, {post['n_hypotheses'].mean():.0f} hypotheses per tile, ESS {post['ess'].mean():.1f}.",
            f"Median relative 90% credible width: cell count {relw('n_cells'):.3f}, permeability index "
            f"{relw('permeability_index'):.3f}, in-plane G {relw('in_plane_G'):.3f}, "
            f"area CV {relw('cell_area_cv'):.3f}.",
            f"Spearman of the posterior-mean permeability index with TER over these tiles: "
            f"{spearmanr(post['permeability_index_mean'], post['ter_ohm'], nan_policy='omit')[0]:.2f}; "
            f"of the MAP value: {spearmanr(post['permeability_index_map'], post['ter_ohm'], nan_policy='omit')[0]:.2f} "
            f"(tiles with a complex: {int(np.isfinite(post['permeability_index_mean']).sum())}).",
        ]
    (out / "tables.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- qc
def cmd_qc(args) -> None:
    import tifffile
    from PIL import Image, ImageDraw, ImageFont
    from skimage.segmentation import find_boundaries

    from pimorph.infer.cellpose_proposer import tricellular_points

    out = Path(args.out)
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    per_tile = pd.read_csv(out / "per_tile.csv")
    first = per_tile.sort_values("ter_ohm").drop_duplicates("wt_id")
    idx = np.unique(np.round(np.linspace(0, len(first) - 1, 12)).astype(int))
    picks = first.iloc[idx]
    rows = []
    for _, r in picks.iterrows():
        img = read_qbam(H2 / "tiles" / r.file)
        lab = tifffile.imread(str(out / "labels" / r.file.replace(".tif", "_labels.tif"))).astype(np.int32)
        # crop of about 25 cells: side = sqrt(25 * mean cell area)
        area = r.cell_area_px if np.isfinite(r.cell_area_px) else 500.0
        side = int(np.clip(np.sqrt(25 * max(area, 100.0)), 96, 400))
        h, w = img.shape
        r0, c0 = (h - side) // 2, (w - side) // 2
        crop = img[r0 : r0 + side, c0 : c0 + side]
        lcrop = lab[r0 : r0 + side, c0 : c0 + side]
        lo, hi = np.percentile(crop, [0.5, 99.5])
        g = (255 * np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
        scale = 450 / side
        left = Image.fromarray(g).resize((450, 450), Image.BICUBIC).convert("RGB")
        right = left.copy()
        # outlines drawn at display resolution so they stay one pixel wide after upscaling
        lbig = np.array(Image.fromarray(lcrop.astype(np.int32)).resize((450, 450), Image.NEAREST))
        big = find_boundaries(lbig, mode="inner")
        arr = np.array(right)
        arr[big] = (255, 220, 0)
        right = Image.fromarray(arr)
        d = ImageDraw.Draw(right)
        for pr, pc in tricellular_points(lcrop):
            x, yy = pc * scale, pr * scale
            d.ellipse([x - 3, yy - 3, x + 3, yy + 3], outline=(0, 255, 255), width=2)
        n_in = len(np.unique(lcrop[lcrop > 0]))
        canvas = Image.new("RGB", (900, 480), "black")
        canvas.paste(left, (0, 30))
        canvas.paste(right, (450, 30))
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
        d.text(
            (6, 6),
            f"{r.wt_id}  {r.condition}  week {int(r.week)}  TER {r.ter_ohm:.0f} Ohm   |   "
            f"{side} px crop, {n_in} predicted cells (labels touching the crop included)",
            fill="white",
            font=font,
        )
        name = f"qc_{r.well_id}_w{int(r.week)}.png"
        canvas.save(qc / name)
        rows.append(
            {
                "tile": name,
                "file": r.file,
                "wt_id": r.wt_id,
                "condition": r.condition,
                "week": int(r.week),
                "ter_ohm": r.ter_ohm,
                "crop_px": side,
                "cells_predicted_in_crop": n_in,
            }
        )
    pd.DataFrame(rows).to_csv(qc / "qc_index.csv", index=False)
    readme = """# QC pack: QBAM crops with decoded outlines

12 well-timepoints stratified over TER (see `qc_index.csv`). Each PNG (900 x 480) shows a centre crop
of the blue absorbance tile sized to hold about 25 cells on the left and, on the right, the same crop
with the decoded cell outlines (yellow) and multicellular vertices (cyan circles). Bright = high
absorbance (pigment); cell borders are the thin darker lines. The title gives plate, well, condition,
week and the measured TER.

How to record judgments: fill `qc_judgments.csv` in this folder with one row per PNG and the columns

    tile, cells_visible, cells_predicted, merges, splits, verdict

- `tile`: the PNG file name.
- `cells_visible`: your count of cells in the LEFT panel (count a cell if more than half of it is
  inside the crop).
- `cells_predicted`: your count of outlined cells in the RIGHT panel with the same rule (the title
  reports the number of label ids touching the crop, which is larger).
- `merges`: outlined regions that clearly contain two or more real cells.
- `splits`: real cells cut into two or more outlined regions.
- `verdict`: `good` (merges + splits <= 2), `usable` (3 to 5) or `bad` (more than 5, or outlines
  unrelated to the visible borders).

The segmenter was fine-tuned on mature, well pigmented AMD tiles; the weeks 3 to 5 crops (low TER)
are less pigmented and are where failures are expected. Please note in `verdict` if borders are not
visible to you either.
"""
    (qc / "README.md").write_text(readme)
    print(f"{len(rows)} QC crops in {qc}")


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("export-train")
    p = sp.add_parser("eval-segmenter")
    p.add_argument("--weights", default=CP_WEIGHTS)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--raw-masks", action="store_true", help="also score Cellpose's raw masks (no decoder)")
    sp.add_parser("inventory")
    p = sp.add_parser("select")
    p.add_argument("--n", type=int, default=84)
    sp.add_parser("download")
    p = sp.add_parser("segment")
    p.add_argument("--weights", default=CP_WEIGHTS)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--crop", type=int, default=0, help="centre crop side in px (0 = full tile)")
    p.add_argument("--posterior-n", type=int, default=0)
    p.add_argument("--posterior-crop", type=int, default=512)
    p.add_argument("--max-items", type=int, default=None)
    p = sp.add_parser("stats")
    p.add_argument("--out", default=str(OUT))
    p = sp.add_parser("qc")
    p.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {
        "export-train": cmd_export_train,
        "eval-segmenter": cmd_eval_segmenter,
        "inventory": cmd_inventory,
        "select": cmd_select,
        "download": cmd_download,
        "segment": cmd_segment,
        "stats": cmd_stats,
        "qc": cmd_qc,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
