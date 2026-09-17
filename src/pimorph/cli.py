"""Command-line entry point for PiMorph.

Subcommands are registered lazily so that importing the CLI does not pull in
torch or cellpose.

    pimorph reconstruct --manifest data/ve_strat/manifest_paired.csv --out runs/pimorph_ve_strat
    pimorph benchmark --dataset cornea --methods classical --max-items 20 --out runs/pimorph_bench
    pimorph validate --labels path/to/labels.tif
    pimorph synth --n 200 --out data/tiles/synth_val --shape 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

Handler = Callable[[argparse.Namespace], int]


def _cmd_version(_: argparse.Namespace) -> int:
    from pimorph import __version__

    print(__version__)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import numpy as np
    import tifffile

    from pimorph.complex import defect_law_residual, extract_complex, validate, weaire_sum_rule_residual
    from pimorph.complex.geometry import smooth_complex

    labels = np.asarray(tifffile.imread(args.labels)).astype(np.int32)
    cx = extract_complex(labels, pixel_size_um=args.pixel_size_um)
    smooth_complex(cx)
    rep = validate(cx)
    out = rep.as_dict()
    out["defect_law_residual"] = defect_law_residual(cx)
    out["weaire_residual"] = weaire_sum_rule_residual(cx)
    print(json.dumps(out, indent=2, default=str))
    return 0 if rep.ok else 1


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    import numpy as np

    from pimorph.fields import profile_all_edges
    from pimorph.infer import (
        ClassicalProposer,
        ConstrainedDecoder,
        DecoderParams,
        PosteriorEnsemble,
        generate_hypotheses,
    )
    from pimorph.io import complex_tables, write_legacy_outputs, write_tables
    from pimorph.io.manifest import parse_manifest

    specs = parse_manifest(args.manifest)
    if args.ids:
        want = set(args.ids)
        specs = [s for s in specs if s.image_id in want]
    if args.max_items:
        specs = specs[: args.max_items]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report = []
    for sp in specs:
        im = sp.load()
        geom = sp.role_channel(im, "geometry")
        nuc = sp.role_channel(im, "nuclei")
        junc = sp.role_channel(im, "junction")
        if args.crop:
            r0, r1, c0, c1 = args.crop
            geom = geom[r0:r1, c0:c1]
            nuc = None if nuc is None else nuc[r0:r1, c0:c1]
            junc = None if junc is None else junc[r0:r1, c0:c1]
        maps = ClassicalProposer()(geom, nuc, junc)
        dec = ConstrainedDecoder(pixel_size_um=im.pixel_size_um)
        base = DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)))
        img_dir = out_root / sp.image_id
        img_dir.mkdir(parents=True, exist_ok=True)
        if args.posterior:
            hyps = generate_hypotheses(maps, dec, base, image=geom, n_merge_moves=args.moves, n_split_moves=args.moves)
            post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=args.ess)
            best = post.map_hypothesis
            labels, cx = best.labels, best.cx
            summ = post.summary()
            cp = post.contact_probabilities()
            (img_dir / "posterior_summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
            (img_dir / "contact_probabilities.json").write_text(
                json.dumps({f"{a}_{b}": p for (a, b), p in cp.items()}, indent=0), encoding="utf-8"
            )
            vr = post.vertex_credible_radius()
            (img_dir / "vertex_credible.json").write_text(
                json.dumps([{"cells": sorted(k), **v} for k, v in vr.items()], indent=0), encoding="utf-8"
            )
        else:
            res = dec.decode(maps, base)
            labels, cx = res.labels, res.cx
            summ = {"n_hypotheses": 1}
        cx.provenance.update(
            {
                "image_id": sp.image_id,
                "geometry_source": sp.geometry_source,
                "roles": sp.roles,
                "pixel_size_source": im.pixel_size_source,
                "proposals": maps.meta,
            }
        )
        import tifffile

        tifffile.imwrite(img_dir / "labels.tif", labels.astype(np.int32))
        profiles = profile_all_edges(cx, junc if junc is not None else geom) if not args.no_profiles else None
        write_legacy_outputs(cx, out_root, sp.image_id, profiles_df=profiles, labels=labels)
        tables = complex_tables(cx, labels=labels)
        write_tables(tables, img_dir, fmt="csv")
        row = {
            "image_id": sp.image_id,
            "condition": sp.condition,
            "n_cells": int(cx.cell_faces.size),
            "n_edges": int(cx.n_edges),
            "n_cell_cell_edges": int(cx.cell_cell_edges().size),
            "n_vertices": int(cx.n_vertices),
            "n_gaps": int(cx.gap_faces.size),
            "geometry_source": sp.geometry_source,
            **{f"posterior_{k}": v for k, v in summ.items()},
        }
        report.append(row)
        print(json.dumps(row, default=str))
    import pandas as pd

    pd.DataFrame(report).to_csv(out_root / "run_report.csv", index=False)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from pimorph.bench import run_benchmark

    df = run_benchmark(
        args.dataset,
        methods=args.methods,
        root=args.root,
        max_items=args.max_items,
        out_dir=args.out,
        tol_px=args.tol_px,
    )
    print(f"{len(df)} rows written to {args.out}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from pimorph.synth.targets import default_params_sampler, default_render_sampler, make_dataset

    def ps(rng):
        p = default_params_sampler(rng)
        p.shape = (args.shape, args.shape)
        return p

    df = make_dataset(args.n, Path(args.out), ps, default_render_sampler, seed=args.seed)
    print(f"wrote {len(df)} tiles to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pimorph", description="PiMorph endothelial cell-complex inference")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version").set_defaults(_handler=_cmd_version)

    p = sub.add_parser("validate", help="extract a complex from a label TIFF and print the validity report")
    p.add_argument("--labels", required=True)
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.set_defaults(_handler=_cmd_validate)

    p = sub.add_parser("reconstruct", help="proposals -> constrained decoder -> complex (+ posterior) for a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "R1", "C0", "C1"))
    p.add_argument("--posterior", action="store_true")
    p.add_argument("--moves", type=int, default=6)
    p.add_argument("--ess", type=float, default=4.0)
    p.add_argument("--no-profiles", action="store_true")
    p.set_defaults(_handler=_cmd_reconstruct)

    p = sub.add_parser("benchmark", help="score reconstruction methods against ground-truth label images")
    p.add_argument("--dataset", required=True, choices=["cornea", "nuinsseg", "mcellseg", "livecell", "synth"])
    p.add_argument("--methods", nargs="+", default=["classical"])
    p.add_argument("--root", default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--out", default="runs/pimorph_bench")
    p.add_argument("--tol-px", type=float, default=3.0)
    p.set_defaults(_handler=_cmd_benchmark)

    p = sub.add_parser("synth", help="write synthetic tiles with exact targets")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--shape", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(_handler=_cmd_synth)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args._handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
