#!/usr/bin/env python3
"""Consensus pseudo-labels on Jacquemet PECAM-1 HUVEC fields.

Same policy as ``pimorph.infer.neural.pseudolabel`` ``cellpose_primary`` (Cellpose-SAM
masks are the labels), but the second opinion is the PiMorph neural proposer
(``runs/neural/v4_confluent/best.pt``) through the constrained decoder instead of the
classical ridge proposer. Boundaries without support in the neural boundary map and
cells the two segmenters disagree on (IoU < 0.7) are ``ignore``. Tiles use the
standard 512 px format with ``has_nuclei=False``. One shard per GPU.

Usage (on the devbox, shard k of 4 on GPU 4 + k):
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python runs/pimorph_dev/pecam_pseudolabel.py --shard 0 --nshards 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from pimorph.bench.datasets import fill_gt_slivers
from pimorph.complex.extract import extract_complex
from pimorph.complex.matching import match_faces
from pimorph.infer.cellpose_sam import CellposeSAM
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural.proposer import NeuralProposer
from pimorph.infer.neural.pseudolabel import _downsample2, _tile_starts, classical_only_ignore, consensus_labels
from pimorph.io.manifest import parse_manifest
from pimorph.synth.targets import TARGET_KEYS, make_targets


def agreement_stats(labels_cp: np.ndarray, labels_nn: np.ndarray, iou_thresh: float = 0.7) -> Dict[str, float]:
    """Face-level agreement between the Cellpose labels and the PiMorph labels."""
    cx_a, cx_b = extract_complex(labels_cp), extract_complex(labels_nn)
    fm = match_faces(labels_cp, labels_nn, cx_a, cx_b, iou_thresh=iou_thresh)
    matched = int(sum(1 for iou in fm.iou if iou >= iou_thresh))
    n_a, n_b = int(cx_a.cell_faces.size), int(cx_b.cell_faces.size)
    return {
        "n_cells_cellpose": n_a,
        "n_cells_pimorph": n_b,
        "n_matched": matched,
        "frac_cellpose_matched": matched / max(n_a, 1),
        "frac_pimorph_matched": matched / max(n_b, 1),
        "mean_iou_matched": float(np.mean([iou for iou in fm.iou if iou >= iou_thresh])) if matched else float("nan"),
    }


def save_figure(path: Path, geometry, labels_cp, labels_nn, ignore, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(geometry, dtype=np.float32)
    lo, hi = np.percentile(g, [1, 99.5])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    axes[0].imshow(g, cmap="gray")
    axes[0].set_title("PECAM-1")
    axes[1].imshow(g, cmap="gray")
    axes[1].contour(find_boundaries(labels_cp, mode="inner"), levels=[0.5], colors="cyan", linewidths=0.5)
    axes[1].set_title(f"Cellpose-SAM ({int(np.unique(labels_cp[labels_cp > 0]).size)} cells)")
    axes[2].imshow(g, cmap="gray")
    axes[2].contour(find_boundaries(labels_nn, mode="inner"), levels=[0.5], colors="orange", linewidths=0.5)
    axes[2].set_title(f"PiMorph v4_confluent ({int(np.unique(labels_nn[labels_nn > 0]).size)} cells)")
    agree = np.where(ignore, 0.0, np.where(labels_cp > 0, 1.0, 0.5))
    axes[3].imshow(agree, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(f"agreement (dark = ignore, {ignore.mean():.0%})")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/jacquemet_huvec/manifest_pecam.csv")
    ap.add_argument("--checkpoint", default="runs/neural/v4_confluent/best.pt")
    ap.add_argument("--out", default="data/tiles/pseudo_pecam")
    ap.add_argument("--max-fields", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--iou-thresh", type=float, default=0.7)
    ap.add_argument("--boundary-tol-px", type=float, default=3.0)
    ap.add_argument("--downsample-radius-px", type=float, default=35.0)
    ap.add_argument("--figures", type=int, default=0, help="example figures for the first N fields of this shard")
    ap.add_argument("--figure-dir", default="runs/pimorph_dev")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = parse_manifest(args.manifest, root=".")
    rng = np.random.default_rng(args.seed)
    if len(specs) > args.max_fields:
        idx = np.sort(rng.choice(len(specs), size=args.max_fields, replace=False))
        specs = [specs[i] for i in idx]
    mine = [(k, s) for k, s in enumerate(specs) if k % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] {len(mine)} of {len(specs)} fields", flush=True)

    prop = NeuralProposer(args.checkpoint)
    cp = CellposeSAM()
    rows: List[Dict] = []
    field_rows: List[Dict] = []
    for i, (k, spec) in enumerate(mine):
        t0 = time.time()
        im = spec.load()
        geometry = np.asarray(spec.role_channel(im, "geometry"), dtype=np.float32)
        maps = prop(geometry, None)
        scale = 1
        if float(maps.meta["cell_radius_px"]) > args.downsample_radius_px:
            scale = 2
            geometry = _downsample2(geometry)
            maps = prop(geometry, None)
        radius = float(maps.meta["cell_radius_px"])
        res = ConstrainedDecoder(pixel_size_um=im.pixel_size_um).decode(maps, DecoderParams(cell_radius_px=radius))
        labels_nn = res.labels.astype(np.int32)
        labels_cp = cp(geometry, None)

        labels = fill_gt_slivers(labels_cp, max_area_px=12)
        ignore = classical_only_ignore(labels, maps.boundary, support=0.3, tol_px=args.boundary_tol_px)
        _, ign_disagree = consensus_labels(
            labels, labels_nn, iou_thresh=args.iou_thresh, boundary_tol_px=args.boundary_tol_px
        )
        ignore = ignore | (ign_disagree & find_boundaries(labels, connectivity=1, mode="thick"))
        stats = agreement_stats(labels, labels_nn, args.iou_thresh)

        H, W = labels.shape
        n_tiles = 0
        for r0 in _tile_starts(H, args.tile, args.tile):
            for c0 in _tile_starts(W, args.tile, args.tile):
                rs, cs = slice(r0, min(r0 + args.tile, H)), slice(c0, min(c0 + args.tile, W))
                lab_t, ign_t = labels[rs, cs], ignore[rs, cs]
                targets = make_targets(lab_t)
                arrays = {
                    "junction": geometry[rs, cs].astype(np.float32),
                    "nuclei": np.zeros(lab_t.shape, dtype=np.float32),
                    "has_nuclei": np.array(False),
                    "labels": lab_t.astype(np.int32),
                    "ignore": ign_t.astype(bool),
                    **{key: targets[key] for key in TARGET_KEYS},
                }
                fname = f"field_{k:03d}_r{r0:04d}_c{c0:04d}.npz"
                np.savez_compressed(out / fname, **arrays)
                n_tiles += 1
                rows.append(
                    {
                        "file": fname,
                        "source_field": spec.image_id,
                        "row0": r0,
                        "col0": c0,
                        "scale": scale,
                        "consensus": "cellpose_primary+neural_v4",
                        "has_nuclei": False,
                        "has_membrane": False,
                        "n_cells": int(np.unique(lab_t[lab_t > 0]).size),
                        "ignore_frac": float(ign_t.mean()),
                        "cell_radius_px": radius,
                        "pixel_size_um": im.pixel_size_um,
                        "condition": spec.condition,
                        "dataset": spec.dataset,
                    }
                )
        field_rows.append(
            {
                "source_field": spec.image_id,
                "field_index": k,
                "scale": scale,
                "cell_radius_px": radius,
                "ignore_frac": float(ignore.mean()),
                "n_tiles": n_tiles,
                "seconds": time.time() - t0,
                **stats,
            }
        )
        print(
            f"[pseudolabel] {spec.image_id}: scale={scale} r={radius:.1f} cellpose={stats['n_cells_cellpose']} "
            f"pimorph={stats['n_cells_pimorph']} matched={stats['frac_cellpose_matched']:.2f} "
            f"ignore={ignore.mean():.2f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if i < args.figures:
            fig_dir = Path(args.figure_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)
            save_figure(
                fig_dir / f"pecam_pseudo_example_{i + 1}.png",
                geometry,
                labels,
                labels_nn,
                ignore,
                f"{spec.image_id} (scale {scale}, {stats['frac_cellpose_matched']:.0%} of Cellpose cells matched)",
            )
    pd.DataFrame(rows).to_csv(out / f"manifest_shard{args.shard}.csv", index=False)
    pd.DataFrame(field_rows).to_csv(out / f"fields_shard{args.shard}.csv", index=False)
    (out / f"done_shard{args.shard}.json").write_text(json.dumps({"n_fields": len(mine), "n_tiles": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
