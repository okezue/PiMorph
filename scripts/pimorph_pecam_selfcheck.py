#!/usr/bin/env python3
"""Self-consistency comparison of proposal checkpoints on Jacquemet PECAM-1 HUVEC fields.

The PECAM-1 fields (Zenodo 10611092, ``pimorph.io.jacquemet``) have no masks and no
nuclear channel, so checkpoints are compared on label-free statistics in the style of
``scripts/select_checkpoint_egm2.py``: cell count, tricellular vertices, gap faces,
PECAM-1 boundary-to-interior contrast along the reconstructed edges and the renderer
log-likelihood (mean per pixel, as ``render_log_likelihood`` returns it). Fields whose
proposer cell radius exceeds ``--downsample-radius-px`` are downsampled by 2 first, the
same rule as the pseudo-label pipeline.

Usage:
    python scripts/pimorph_pecam_selfcheck.py --fields 20 \
        --checkpoints runs/neural/v3_endo/best.pt runs/neural/v4_confluent/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pimorph.infer import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural.proposer import NeuralProposer
from pimorph.infer.neural.pseudolabel import _downsample2
from pimorph.infer.renderer import RenderModel, boundary_interior_ratio, render_log_likelihood
from pimorph.io.manifest import parse_manifest

SUMMARY_COLS = [
    "n_cells",
    "n_tricellular",
    "tricellular_per_cell",
    "n_gaps",
    "cell_area_um2_median",
    "tissue_fraction",
    "boundary_interior_ratio",
    "render_loglik_per_px",
    "seconds",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/jacquemet_huvec/manifest_pecam.csv")
    ap.add_argument("--fields", type=int, default=20)
    ap.add_argument("--split", default="training", help="manifest split to sample from ('' for all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoints", nargs="+", default=["runs/neural/v3_endo/best.pt", "runs/neural/v4_confluent/best.pt"]
    )
    ap.add_argument("--downsample-radius-px", type=float, default=35.0)
    ap.add_argument("--out", default="runs/pecam_selfcheck/selfcheck.csv")
    args = ap.parse_args()

    specs = parse_manifest(args.manifest, root=".")
    if args.split:
        specs = [s for s in specs if s.extra.get("split", "") == args.split]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(specs), size=min(args.fields, len(specs)), replace=False))
    chosen = [specs[i] for i in idx]
    print(f"{len(chosen)} fields: {[s.image_id for s in chosen]}", flush=True)

    rows = []
    for ck in args.checkpoints:
        prop = NeuralProposer(ck)
        for sp in chosen:
            t0 = time.time()
            im = sp.load()
            geom = np.asarray(sp.role_channel(im, "geometry"), dtype=np.float32)
            px = im.pixel_size_um
            maps = prop(geom, None)
            scale = 1
            if float(maps.meta["cell_radius_px"]) > args.downsample_radius_px:
                scale = 2
                geom = _downsample2(geom)
                maps = prop(geom, None)
                px = None if px is None else 2.0 * px
            radius = float(maps.meta.get("cell_radius_px", 15.0))
            res = ConstrainedDecoder(pixel_size_um=px).decode(maps, DecoderParams(cell_radius_px=radius))
            labels = res.labels
            width = float(maps.meta.get("ridge_width_px", 2.0))
            ll, _ = render_log_likelihood(
                geom, labels, res.cx, RenderModel(line_width_px=width, psf_sigma_px=max(0.5, 0.3 * width))
            )
            areas = np.bincount(labels.ravel())[1:]
            areas = areas[areas > 0]
            n_tri = int(sum(1 for v in range(res.cx.n_vertices) if len(res.cx.vertex_cell_set(v)) >= 3))
            n_cells = int(res.cx.cell_faces.size)
            rows.append(
                {
                    "checkpoint": Path(ck).parent.name if Path(ck).stem == "best" else Path(ck).stem,
                    "image_id": sp.image_id,
                    "scale": scale,
                    "cell_radius_px": radius,
                    "n_cells": n_cells,
                    "n_tricellular": n_tri,
                    "tricellular_per_cell": n_tri / max(n_cells, 1),
                    "n_gaps": int(res.cx.gap_faces.size),
                    "cell_area_um2_median": float(np.median(areas)) * (px or 1.0) ** 2 if areas.size else np.nan,
                    "tissue_fraction": float((labels > 0).mean()),
                    "boundary_interior_ratio": float(boundary_interior_ratio(geom, labels, res.cx)),
                    "render_loglik_per_px": float(ll),
                    "seconds": time.time() - t0,
                }
            )
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    mean = df.groupby("checkpoint")[SUMMARY_COLS].mean()
    med = df.groupby("checkpoint")[SUMMARY_COLS].median()
    table = mean.round(3).to_string()
    (out.parent / "summary.md").write_text(
        f"# PECAM-1 self-consistency ({len(chosen)} fields, seed {args.seed}, split {args.split or 'all'})\n\n"
        "Means per checkpoint:\n\n```\n"
        + table
        + "\n```\n\nMedians per checkpoint:\n\n```\n"
        + med.round(3).to_string()
        + "\n```\n"
    )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
