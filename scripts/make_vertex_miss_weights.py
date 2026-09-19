#!/usr/bin/env python3
"""Hard-example weights for topologically missed vertices.

Every real-truth training tile is decoded with the current model exactly as at test time
(proposals, nucleus seeds, constrained decoder), its multicellular vertices are matched to
the truth, and a per-pixel ``loss_weight`` array is written back into the tile:

    1 + w_miss  inside a disk of ``radius`` px around every TRUE vertex with no predicted
                counterpart (the vertex does not exist in the decoded complex: a missing
                boundary branch, a merged or missed cell)
    1 + w_fp    around every PREDICTED vertex with no true counterpart (a split cell, a
                false contact)

``TileDataset`` multiplies its loss weight by this array, so the next fine-tune spends its
capacity on the junction geometry the decoder currently gets wrong. Ignored pixels stay
ignored. A CSV with per-tile miss counts is written next to the tiles.

Usage:
    PIMORPH_NEURAL_CKPT=runs/neural/v4_confluent/best.pt \
    python scripts/make_vertex_miss_weights.py --tiles data/tiles/gt_hcec_train --shard 0/8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from pimorph.complex.extract import extract_complex
from pimorph.complex.matching import match_faces, match_vertices
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural.proposer import NeuralProposer


def disk_weight(shape, points, radius: float, amount: float) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    if len(points):
        pts = np.round(np.asarray(points)).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, shape[0] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, shape[1] - 1)
        m[pts[:, 0], pts[:, 1]] = True
        m = ndi.distance_transform_edt(~m) <= radius
    return amount * m.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True, help="tile directories (npz with labels)")
    ap.add_argument("--checkpoint", default=os.environ.get("PIMORPH_NEURAL_CKPT", "runs/neural/v4_confluent/best.pt"))
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--w-miss", type=float, default=3.0)
    ap.add_argument("--w-fp", type=float, default=2.0)
    ap.add_argument("--tol-px", type=float, default=3.0)
    ap.add_argument("--shard", default=None, help="k/n")
    ap.add_argument("--dry-run", action="store_true", help="compute stats only, do not rewrite tiles")
    args = ap.parse_args()
    k, n = (0, 1) if not args.shard else (int(args.shard.split("/")[0]), int(args.shard.split("/")[1]))
    prop = NeuralProposer(args.checkpoint)
    dec = ConstrainedDecoder()
    rows = []
    t0 = time.time()
    for d in args.tiles:
        paths = sorted(Path(d).glob("*.npz"))
        for i, p in enumerate(paths):
            if i % n != k:
                continue
            with np.load(p) as z:
                data = {key: z[key] for key in z.files}
            lab = data["labels"].astype(np.int32)
            if lab.max() == 0:
                continue
            nuc = data["nuclei"] if bool(data.get("has_nuclei", np.array(False))) else None
            maps = prop(data["junction"].astype(np.float32), nuc)
            res = dec.decode(maps, DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0))))
            cx_r = extract_complex(lab)
            fm = match_faces(res.labels, lab, res.cx, cx_r)
            vm = match_vertices(res.cx, cx_r, fm, dist_px=args.tol_px)
            missed = cx_r.vertex_xy[vm.unmatched_ref] if vm.unmatched_ref else np.zeros((0, 2))
            spurious = res.cx.vertex_xy[vm.unmatched_pred] if vm.unmatched_pred else np.zeros((0, 2))
            if "ignore" in data and len(spurious):
                # predicted vertices in unannotated regions are not errors
                ign = data["ignore"].astype(bool)
                pr = np.clip(np.round(spurious[:, 0]).astype(int), 0, lab.shape[0] - 1)
                pc = np.clip(np.round(spurious[:, 1]).astype(int), 0, lab.shape[1] - 1)
                spurious = spurious[~ign[pr, pc]]
            w = 1.0 + disk_weight(lab.shape, missed, args.radius, args.w_miss)
            w = w + disk_weight(lab.shape, spurious, args.radius, args.w_fp)
            if not args.dry_run:
                data["loss_weight"] = w.astype(np.float32)
                np.savez_compressed(p, **data)
            rows.append(
                {
                    "tile": str(p),
                    "n_ref_vertices": vm.n_ref_vertices,
                    "n_pred_vertices": vm.n_pred_vertices,
                    "n_matched": len(vm.pairs),
                    "n_missed": len(vm.unmatched_ref),
                    "n_spurious": len(vm.unmatched_pred),
                    "weighted_frac": float((w > 1).mean()),
                }
            )
            if len(rows) % 50 == 0:
                print(f"{len(rows)} tiles, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.tiles[0]) / (f"vertex_miss_stats_shard{k:02d}.csv" if n > 1 else "vertex_miss_stats.csv")
    df.to_csv(out, index=False)
    if len(df):
        tot = df[["n_ref_vertices", "n_matched", "n_missed", "n_spurious"]].sum()
        print(
            f"{len(df)} tiles: true vertices {tot.n_ref_vertices}, matched {tot.n_matched} "
            f"({tot.n_matched / max(tot.n_ref_vertices, 1):.3f}), missed {tot.n_missed}, spurious {tot.n_spurious}; "
            f"mean weighted fraction {df.weighted_frac.mean():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
