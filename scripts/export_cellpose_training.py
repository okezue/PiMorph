#!/usr/bin/env python3
"""Export benchmark TRAIN splits as Cellpose training pairs.

For every item of a dataset restricted to an id list (``path.json:key``, the same
``PIMORPH_ID_LIST`` spec the benchmark uses), write ``<id>.tif`` and ``<id>_masks.tif``
(int32 labels) into ``data/cellpose_train/<dataset>/``. The image is the raw geometry
channel, which is what ``method_cellpose_sam`` feeds Cellpose-SAM at test time (the
benchmark inverts dark borders with ``geometry_for_bright_boundaries`` only for the
classical and neural methods); ``--bright-boundaries`` applies that inversion instead.
When the loader provides nuclei the image is a 2-channel stack ``[geometry, nuclei]`` in
CYX order, matching the stack the benchmark passes at test time (Cellpose zero-pads the
third channel).

Items with a ``roi`` are cropped to the ROI bounding box and image pixels outside the
ROI are set to the image median inside the crop. Cellpose has no ignore mask, so untraced
cells outside the ROI would otherwise be learnt as background; a flat median fill
removes their evidence and leaves an unambiguous background region instead.

Usage:
    python scripts/export_cellpose_training.py --dataset hcec --ids runs/vertex/splits_confluent.json:hcec_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from pimorph.bench.datasets import load_dataset
from pimorph.bench.run import geometry_for_bright_boundaries, resolve_polarity


def crop_to_roi(img: np.ndarray, labels: np.ndarray, roi: np.ndarray, pad: int = 8):
    """Crop channels-first image and labels to the padded ROI bounding box and flatten
    pixels outside the ROI to the median of the ROI pixels of each channel."""
    ys, xs = np.nonzero(roi)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, roi.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, roi.shape[1])
    roi_c = roi[y0:y1, x0:x1]
    img_c = img[:, y0:y1, x0:x1].copy()
    for c in range(img_c.shape[0]):
        img_c[c][~roi_c] = float(np.median(img_c[c][roi_c]))
    lab_c = np.where(roi_c, labels[y0:y1, x0:x1], 0).astype(np.int32)
    return img_c, lab_c, (y0, y1, x0, x1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", required=True, help="path.json:key id list (TRAIN split)")
    ap.add_argument("--root", default=None, help="dataset root (default from pimorph.bench.datasets)")
    ap.add_argument("--out", default="data/cellpose_train")
    ap.add_argument("--no-nuclei", action="store_true", help="write the geometry channel only")
    ap.add_argument(
        "--bright-boundaries",
        action="store_true",
        help="invert dark borders (geometry_for_bright_boundaries) instead of the raw channel fed to Cellpose",
    )
    args = ap.parse_args()

    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    n_cells_total = 0
    for item in load_dataset(args.dataset, root=args.root, id_list=args.ids):
        g = geometry_for_bright_boundaries(item) if args.bright_boundaries else np.asarray(item.geometry, np.float32)
        chans = [g]
        if item.nuclei is not None and not args.no_nuclei:
            chans.append(np.asarray(item.nuclei, dtype=np.float32))
        img = np.stack(chans, axis=0)
        lab = np.asarray(item.labels_gt, dtype=np.int32)
        bbox = None
        if item.roi is not None:
            img, lab, bbox = crop_to_roi(img, lab, np.asarray(item.roi, dtype=bool))
        fid = item.image_id.replace("/", "_")
        tifffile.imwrite(out / f"{fid}.tif", img[0] if img.shape[0] == 1 else img)
        tifffile.imwrite(out / f"{fid}_masks.tif", lab)
        n_cells = int(len(np.unique(lab)) - (1 if (lab == 0).any() else 0))
        n_cells_total += n_cells
        manifest.append(
            {
                "image_id": item.image_id,
                "file": f"{fid}.tif",
                "shape": list(lab.shape),
                "n_channels": int(img.shape[0]),
                "n_cells": n_cells,
                "roi_bbox_y0y1x0x1": list(bbox) if bbox else None,
                "boundary_polarity": resolve_polarity(item),
                "inverted": bool(args.bright_boundaries and resolve_polarity(item) == "dark"),
            }
        )
        print(f"{args.dataset} {item.image_id:32s} {lab.shape} chans={img.shape[0]} cells={n_cells}")
    nch = sorted({m["n_channels"] for m in manifest})
    usage = "single channel (geometry)" if nch == [1] else "2 channels: chan 1 geometry, chan 2 nuclei (CYX tif)"
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "ids": args.ids,
                "n_images": len(manifest),
                "n_cells": n_cells_total,
                "channels": usage,
                "geometry": "bright-boundary inverted" if args.bright_boundaries else "raw (as method_cellpose_sam)",
                "roi_handling": "crop to ROI bbox (+8 px), image outside ROI set to in-ROI median, labels 0",
                "items": manifest,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{args.dataset}: {len(manifest)} images, {n_cells_total} cells, {usage} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
