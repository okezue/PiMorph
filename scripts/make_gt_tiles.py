#!/usr/bin/env python3
"""Turn a benchmark dataset with real instance masks into training tiles.

Targets (boundary, signed distance, seed, vertex, gap, outer) are derived exactly from
the ground-truth labels with the same code the synthetic generator uses, so real and
synthetic tiles share one format. Only the image channel differs: the dataset's
geometry channel goes into ``junction`` (there is no separate membrane or nuclei
channel for these datasets; ``has_nuclei`` is False).

Usage:
    python scripts/make_gt_tiles.py --dataset livecell --split train --out data/tiles/gt_livecell_train
    python scripts/make_gt_tiles.py --dataset neurips_cellseg --out data/tiles/gt_neurips
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pimorph.bench.datasets import load_dataset
from pimorph.bench.run import geometry_for_bright_boundaries, resolve_polarity
from pimorph.synth.targets import make_targets


def tile_starts(n: int, tile: int, stride: int):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] + tile < n:
        s.append(n - tile)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default=None, help="LIVECell split (train|val|test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--min-cells", type=int, default=3, help="skip tiles with fewer GT cells")
    args = ap.parse_args()
    if args.split:
        os.environ["PIMORPH_LIVECELL_SPLIT"] = args.split
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for item in load_dataset(args.dataset, root=args.root, max_items=args.max_items):
        g = geometry_for_bright_boundaries(item)  # dark boundaries inverted so junction-like
        pol = resolve_polarity(item)
        lab = item.labels_gt
        H, W = lab.shape
        for r0 in tile_starts(H, args.tile, args.stride):
            for c0 in tile_starts(W, args.tile, args.stride):
                rs, cs = slice(r0, min(r0 + args.tile, H)), slice(c0, min(c0 + args.tile, W))
                lab_t = lab[rs, cs].astype(np.int32)
                if len(np.unique(lab_t[lab_t > 0])) < args.min_cells:
                    continue
                t = make_targets(lab_t)
                fname = f"field_{k:05d}_r{r0:04d}_c{c0:04d}.npz"
                np.savez_compressed(
                    out / fname,
                    junction=g[rs, cs].astype(np.float32),
                    nuclei=np.zeros(lab_t.shape, np.float32),
                    has_nuclei=np.array(False),
                    labels=lab_t,
                    ignore=np.zeros(lab_t.shape, bool),
                    **t,
                )
                rows.append(
                    {
                        "file": fname,
                        "source_field": item.image_id,
                        "dataset": args.dataset,
                        "polarity": pol,
                        "n_cells": int(len(np.unique(lab_t[lab_t > 0]))),
                        **{f"meta_{kk}": v for kk, v in item.meta.items() if isinstance(v, (str, int, float, bool))},
                    }
                )
        k += 1
        if k % 50 == 0:
            print(f"{k} fields, {len(rows)} tiles", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"wrote {len(df)} tiles from {k} fields to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
