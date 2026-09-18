#!/usr/bin/env python3
"""Evaluate constrained-decoder variants on one dataset split with fixed proposals.

Proposal maps are computed once per field (neural checkpoint from PIMORPH_NEURAL_CKPT)
and every variant of DecoderParams is decoded and scored against the reference
complex, so the table isolates the decoder's decisions (flood mask, gap threshold,
vertex-consistent merges) from the proposal model.

Usage:
    PIMORPH_NEURAL_CKPT=models/pimorph_proposals_v2_endo.pt \
    PIMORPH_ID_LIST=runs/endo/splits.json:haec_train \
    python scripts/tune_decoder.py --dataset haec --max-items 20 --out runs/decoder_tuning/haec_train.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pimorph.bench.datasets import load_dataset
from pimorph.bench.run import geometry_for_bright_boundaries, restrict_to_roi, tissue_for_item
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural.proposer import NeuralProposer
from pimorph.metrics.structural import structural_metrics

KEYS = [
    "n_pred_cells",
    "n_ref_cells",
    "n_matched_cells",
    "n_splits",
    "n_merges",
    "adjacency_pair_f1",
    "adjacency_pair_precision",
    "adjacency_pair_recall",
    "adjacency_component_f1",
    "n_pred_pairs",
    "n_ref_pairs",
    "n_pred_vertices",
    "n_ref_vertices",
    "vertex_f1",
    "vertex_precision",
    "vertex_recall",
    "vertex_incident_set_accuracy",
    "pq",
    "ap50",
    "boundary_f1",
]

# name -> (use permissive intensity tissue mask, DecoderParams overrides)
DEFAULT_VARIANTS = {
    "legacy_perm_tissue": (True, dict(outside_px=None)),
    "legacy_neural_tissue": (False, dict(outside_px=None)),
    "outside1": (False, dict(outside_px=1.0)),
    "outside1_gap0.5": (False, dict(outside_px=1.0, gap_threshold=0.5)),
    "outside2": (False, dict(outside_px=2.0)),
    "outside0": (False, dict(outside_px=0.0)),
    "outside1_merge0.3": (False, dict(outside_px=1.0, merge_boundary_max=0.3, max_merges=400)),
    "outside1_merge0.5": (False, dict(outside_px=1.0, merge_boundary_max=0.5, max_merges=400)),
    "outside1_merge0.5_nov": (
        False,
        dict(outside_px=1.0, merge_boundary_max=0.5, max_merges=400, merge_vertex_min=2.0),
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=None, help="JSON file: name -> [use_perm_tissue, params dict]")
    ap.add_argument(
        "--checkpoint", default=os.environ.get("PIMORPH_NEURAL_CKPT", "models/pimorph_proposals_v2_endo.pt")
    )
    args = ap.parse_args()
    variants = DEFAULT_VARIANTS
    if args.variants:
        variants = {k: (bool(v[0]), dict(v[1])) for k, v in json.load(open(args.variants)).items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prop = NeuralProposer(args.checkpoint)
    rows = []
    for item in load_dataset(args.dataset, root=args.root, max_items=args.max_items):
        g = geometry_for_bright_boundaries(item)
        t0 = time.time()
        maps_perm = prop(g, item.nuclei, tissue=tissue_for_item(item))
        maps_neur = prop(g, item.nuclei, tissue=None)
        t_prop = time.time() - t0
        cr = float(maps_perm.meta.get("cell_radius_px", 15.0))
        dec = ConstrainedDecoder(pixel_size_um=item.pixel_size_um)
        for name, (perm, over) in variants.items():
            maps = maps_perm if perm else maps_neur
            t1 = time.time()
            res = dec.decode(maps, DecoderParams(cell_radius_px=cr, **over))
            labels = restrict_to_roi(res.labels, item.roi)
            met = structural_metrics(labels, item.labels_gt, pixel_size_um=item.pixel_size_um, tol_px=3.0)
            r = {k: met.get(k, np.nan) for k in KEYS}
            r.update(
                variant=name,
                image_id=item.image_id,
                dataset=args.dataset,
                n_vertex_merges=res.info.get("n_vertex_merges", 0),
                decode_s=time.time() - t1,
                proposal_s=t_prop,
            )
            rows.append(r)
            print(
                f"[{item.image_id}] {name:26s} cells={met['n_pred_cells']}/{met['n_ref_cells']} "
                f"pairF1={met['adjacency_pair_f1']:.3f} vtx={met['n_pred_vertices']}/{met['n_ref_vertices']} "
                f"vF1={met['vertex_f1']:.3f} PQ={met['pq']:.3f}",
                flush=True,
            )
        pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summ = df.groupby("variant")[KEYS + ["n_vertex_merges"]].mean().round(3)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"))
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
