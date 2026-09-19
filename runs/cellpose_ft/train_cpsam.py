#!/usr/bin/env python3
"""Fine-tune Cellpose-SAM (cpsam) on exported PiMorph training pairs.

Official cellpose 4 recipe (``cellpose.train.train_seg`` on ``model.net``): AdamW,
learning rate 1e-5 with 10 warm-up epochs, weight decay 0.1, 256 px crops with random
rotation, flips and 0.75 to 1.25 rescaling, 100 epochs. One crop per image per epoch is
the cellpose default; because the fields differ 30x in area (FlyWing 512x512 tiles vs
2048x2048 hCEC fields) each epoch here samples ``area / 256^2`` crops per image
(``train_probs`` proportional to area, ``nimg_per_epoch`` = total crops), so an epoch
covers every training pixel about once regardless of field size.

Usage:
    python runs/cellpose_ft/train_cpsam.py --name cpsam_confluent \
        --dirs data/cellpose_train/hcec data/cellpose_train/alizarine ... [--epochs 100 --batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tifffile
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="export directories with <id>.tif and <id>_masks.tif")
    ap.add_argument("--out", default="runs/cellpose_ft")
    ap.add_argument("--pretrained", default="cpsam")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--crops-per-epoch", type=int, default=None, help="override nimg_per_epoch")
    ap.add_argument("--uniform", action="store_true", help="cellpose default: one crop per image per epoch")
    args = ap.parse_args()

    from cellpose import models, train

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    files, images, labels, areas = [], [], [], []
    for d in args.dirs:
        for p in sorted(Path(d).glob("*_masks.tif")):
            img_p = p.with_name(p.name.replace("_masks.tif", ".tif"))
            img = tifffile.imread(str(img_p)).astype(np.float32)
            lab = tifffile.imread(str(p)).astype(np.int32)
            if img.ndim == 2:
                img = img[None]
            files.append(str(img_p))
            images.append(img)
            labels.append(lab)
            areas.append(lab.shape[0] * lab.shape[1])
    areas = np.asarray(areas, dtype=np.float64)
    crops = np.maximum(1, np.round(areas / 256**2)).astype(int)
    n_cells = sum(int(len(np.unique(lab)) - 1) for lab in labels)
    if args.uniform:
        probs, per_epoch = None, len(images)
    else:
        probs = crops / crops.sum()
        per_epoch = int(args.crops_per_epoch or crops.sum())
    print(
        f"{args.name}: {len(images)} images, {n_cells} cells, {int(crops.sum())} crops/epoch grid, "
        f"nimg_per_epoch={per_epoch}, dirs={args.dirs}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained, device=device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        train_files=None,
        train_probs=probs,
        channel_axis=0,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        nimg_per_epoch=per_epoch,
        save_path=str(out),
        save_every=25,
        model_name=args.name,
        min_train_masks=1,
    )
    wall = time.time() - t0
    info = {
        "name": args.name,
        "pretrained": args.pretrained,
        "dirs": args.dirs,
        "n_images": len(images),
        "n_cells": n_cells,
        "nimg_per_epoch": per_epoch,
        "area_weighted": not args.uniform,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "bsize": 256,
        "gpu": gpu_name,
        "wall_time_s": round(wall, 1),
        "time_per_epoch_s": round(wall / max(args.epochs, 1), 2),
        "model_path": str(path),
        "train_losses": [round(float(x), 5) for x in train_losses],
        "cellpose_version": __import__("cellpose").version,
        "torch_version": torch.__version__,
    }
    (out / "train_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(
        f"done {args.name}: {wall / 60:.1f} min, {wall / args.epochs:.1f} s/epoch on {gpu_name}, model {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
