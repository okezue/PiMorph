#!/usr/bin/env python3
"""Development smoke run of the PiMorph M3 chain on one field (or crop).

Writes a six-panel figure and prints the structured-vs-Voronoi self-consistency
numbers from the blueprint audit.

Usage:
    python scripts/pimorph_dev_smoke.py --manifest data/ve_strat/manifest_paired.csv --index 0 \
        --crop 512 1024 512 1024 --out runs/pimorph_dev/smoke_crop.png
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.segmentation import find_boundaries, watershed  # noqa: E402

from pimorph.complex import extract_complex, validate  # noqa: E402
from pimorph.complex.geometry import smooth_complex  # noqa: E402
from pimorph.infer import ClassicalProposer, ConstrainedDecoder, DecoderParams, PosteriorEnsemble, generate_hypotheses  # noqa: E402
from pimorph.infer.renderer import boundary_interior_ratio, render_log_likelihood  # noqa: E402
from pimorph.io.manifest import parse_manifest  # noqa: E402


def norm(x):
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ve_strat/manifest_paired.csv")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--out", default="runs/pimorph_dev/smoke.png")
    ap.add_argument("--posterior", action="store_true")
    args = ap.parse_args()

    sp = parse_manifest(args.manifest)[args.index]
    im = sp.load()
    print(f"{sp.image_id}: {im.data.shape} roles={sp.roles} geometry_source={sp.geometry_source} px={im.pixel_size_um}")
    geom = sp.role_channel(im, "geometry")
    nuc = sp.role_channel(im, "nuclei")
    if args.crop:
        r0, r1, c0, c1 = args.crop
        geom = geom[r0:r1, c0:c1]
        nuc = nuc[r0:r1, c0:c1] if nuc is not None else None

    t = time.time()
    maps = ClassicalProposer()(geom, nuc)
    print(f"proposals {time.time() - t:.1f}s: {maps.meta}")
    print(f"   tissue={maps.tissue.mean():.2f} gap>0.7={np.mean(maps.gap > 0.7):.3f}")
    dec = ConstrainedDecoder(pixel_size_um=im.pixel_size_um)
    t = time.time()
    res = dec.decode(maps, DecoderParams())
    rep = validate(res.cx)
    print(f"decode {time.time() - t:.1f}s: {res.info} valid={rep.ok}")
    print(f"   vertices={res.cx.n_vertices} deg={rep.degree_histogram}")
    ratio_s = boundary_interior_ratio(geom, res.labels, res.cx)

    pts = np.round(maps.seed_points[maps.seed_scores >= DecoderParams().seed_threshold]).astype(int)
    m = np.zeros(geom.shape, np.int32)
    m[pts[:, 0], pts[:, 1]] = np.arange(1, len(pts) + 1)
    vor = watershed(ndi.distance_transform_edt(m == 0), m, mask=maps.tissue)
    cxv = extract_complex(vor)
    smooth_complex(cxv)
    ratio_v = boundary_interior_ratio(geom, vor, cxv)
    from pimorph.infer.renderer import RenderModel

    width = float(maps.meta.get("ridge_width_px", 1.0))
    rm = RenderModel(line_width_px=width, psf_sigma_px=max(0.5, 0.3 * width))
    ll_s, _ = render_log_likelihood(geom, res.labels, res.cx, rm)
    ll_v, _ = render_log_likelihood(geom, vor, cxv, rm)
    print(f"boundary/interior ratio: structured={ratio_s:.2f} voronoi={ratio_v:.2f}")
    print(f"render ll/px: structured={ll_s:.3f} voronoi={ll_v:.3f}")

    if args.posterior:
        t = time.time()
        hyps = generate_hypotheses(maps, dec, DecoderParams(), image=geom, n_merge_moves=6, n_split_moves=4)
        post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=4)
        print(f"posterior {time.time() - t:.1f}s:", post.summary())
        for h, w in zip(post.hypotheses, post.weights):
            parts = {k: round(v, 3) for k, v in h.breakdown["parts"].items()}
            print(f"   {h.tag:24s} E={h.energy:.3f} w={w:.3f} cells={h.cx.cell_faces.size} {parts}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(norm(geom), cmap="gray")
    ax[0, 0].set_title("geometry channel")
    ax[0, 1].imshow(norm(nuc) if nuc is not None else np.zeros_like(geom), cmap="gray")
    ax[0, 1].set_title("nuclei")
    rgb = np.stack([norm(geom)] * 3, -1)
    rgb[find_boundaries(res.labels, mode="inner")] = [0, 1, 0]
    bg = res.labels == 0
    rgb[bg] = rgb[bg] * 0.5 + np.array([0.3, 0, 0])
    ax[0, 2].imshow(rgb)
    ax[0, 2].set_title(f"decoded: {res.cx.cell_faces.size} cells, {res.cx.gap_faces.size} gaps, ratio {ratio_s:.2f}")
    ax[1, 0].imshow(maps.boundary, cmap="magma")
    ax[1, 0].set_title("boundary map")
    ax[1, 1].imshow(maps.gap, cmap="viridis")
    ax[1, 1].set_title("gap map")
    rgb2 = np.stack([norm(geom)] * 3, -1)
    rgb2[find_boundaries(vor, mode="inner")] = [1, 0.5, 0]
    ax[1, 2].imshow(rgb2)
    ax[1, 2].set_title(f"nucleus-only Voronoi, ratio {ratio_v:.2f}")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=70)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
