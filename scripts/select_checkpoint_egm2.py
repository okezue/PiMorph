#!/usr/bin/env python3
"""Self-consistency comparison of proposal checkpoints on S-BIAD1540 EGM2 fields.

No instance truth exists for these fields, so checkpoints are compared on statistics
that do not use labels: cell count against the nucleus count (DAPI), the fraction of
nuclei that fall in exactly one cell, boundary-to-interior VE-cadherin contrast along
the reconstructed edges and the renderer log-likelihood. The checkpoint used for the
shear re-test is chosen from this table and the choice is recorded with it.

Usage:
    python scripts/select_checkpoint_egm2.py --fields 3 --checkpoints models/*.pt runs/neural/v3_endo/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.filters import gaussian

from pimorph.infer import ConstrainedDecoder, DecoderParams
from pimorph.infer.neural.proposer import NeuralProposer
from pimorph.infer.renderer import RenderModel, boundary_interior_ratio, render_log_likelihood
from pimorph.io.manifest import parse_manifest


def nucleus_points(nuclei: np.ndarray, sigma_px: float = 4.0) -> np.ndarray:
    n = gaussian(nuclei.astype(np.float32), sigma=sigma_px, preserve_range=True)
    thr = np.percentile(n, 60)
    return peak_local_max(n, min_distance=8, threshold_abs=float(thr), exclude_border=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/egm2_full/manifest_egm2_local.csv")
    ap.add_argument("--fields", type=int, default=3, help="fields per condition (static, 6dyne)")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="runs/shear_retest/checkpoint_selection.csv")
    args = ap.parse_args()
    specs = parse_manifest(args.manifest, root=".")
    rng = np.random.default_rng(0)
    chosen = []
    for cond in ("static", "6dyne"):
        pool = [s for s in specs if s.condition == cond]
        idx = rng.choice(len(pool), size=min(args.fields, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    rows = []
    for ck in args.checkpoints:
        prop = NeuralProposer(ck)
        for sp in chosen:
            t0 = time.time()
            im = sp.load()
            geom = sp.role_channel(im, "geometry")
            nuc = sp.role_channel(im, "nuclei")
            maps = prop(geom, nuc)
            res = ConstrainedDecoder(pixel_size_um=im.pixel_size_um).decode(
                maps, DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)))
            )
            pts = nucleus_points(nuc)
            owner = res.labels[pts[:, 0], pts[:, 1]]
            per_cell = np.bincount(owner[owner > 0], minlength=res.labels.max() + 1)
            width = float(maps.meta.get("ridge_width_px", 2.0))
            ll, _ = render_log_likelihood(
                geom, res.labels, res.cx, RenderModel(line_width_px=width, psf_sigma_px=max(0.5, 0.3 * width))
            )
            rows.append(
                {
                    "checkpoint": Path(ck).stem,
                    "image_id": sp.image_id,
                    "condition": sp.condition,
                    "n_cells": int(res.cx.cell_faces.size),
                    "n_nuclei_peaks": int(len(pts)),
                    "frac_nuclei_in_cells": float(np.mean(owner > 0)) if len(pts) else np.nan,
                    "frac_cells_one_nucleus": float(np.mean(per_cell[1:] == 1)) if res.labels.max() else np.nan,
                    "frac_cells_no_nucleus": float(np.mean(per_cell[1:] == 0)) if res.labels.max() else np.nan,
                    "n_tricellular": int(
                        sum(1 for v in range(res.cx.n_vertices) if len(res.cx.vertex_cell_set(v)) >= 3)
                    ),
                    "n_gaps": int(res.cx.gap_faces.size),
                    "boundary_interior_ratio": float(boundary_interior_ratio(geom, res.labels, res.cx)),
                    "render_loglik_per_px": float(ll) / geom.size,
                    "seconds": time.time() - t0,
                }
            )
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = [c for c in df.columns if c not in ("checkpoint", "image_id", "condition")]
    print(df.groupby("checkpoint")[cols].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
