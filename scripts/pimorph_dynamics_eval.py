#!/usr/bin/env python3
"""Evaluate the PiMorph event layer on tracked ground-truth label stacks.

The label stack carries ground-truth track ids (CTC: silver-truth masks relabelled
with the gold TRA marker ids; TissueMiner: Tissue Analyzer tracked cells mapped to
database cell ids), so the numbers measure the event layer independently of
segmentation. Two runs are reported: ``tracker`` (our IoU linker re-derives the
tracks, checked against the ground-truth ids) and ``gt_ids`` (ground-truth ids and
lineage used directly).

Usage:
    python scripts/pimorph_dynamics_eval.py --dataset ctc --name DIC-C2DH-HeLa --seq 01
    python scripts/pimorph_dynamics_eval.py --dataset tissueminer
    python scripts/pimorph_dynamics_eval.py --dataset epicure --no-tracker \
        --movie data/epicure/data_generalisations/movie2

EpiCure movies carry curated track-consistent ids and no lineage, so only the
``gt_ids`` run is meaningful there; ``--no-tracker`` skips the tracker run. Every run
also writes ``residuals_<run>.csv`` with the largest unexplained (dV, dE, dF)
residuals per frame pair and a one-line diagnosis.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pimorph.complex import extract_complex
from pimorph.dynamics import (
    Event,
    admissibility_check,
    complexes_over_time,
    detect_events,
    division_detection_metrics,
    event_summary,
    fill_small_gaps,
    snapshot,
    t1_metrics,
    track,
    tracking_metrics,
    tracks_from_labels,
)
from pimorph.dynamics.ctc import ctc_divisions, load_ctc, tracked_dense_labels
from pimorph.dynamics.epicure import load_epicure_movie
from pimorph.dynamics.tissueminer import UNMAPPED_OFFSET, db_snapshots, load_tissueminer_demo


def run_events(cxs, lineage: Dict[int, int], label: str) -> Dict[str, Any]:
    """Detect events and admissibility per frame pair over a list of complexes/snapshots."""
    snaps = [snapshot(c) if not hasattr(c, "contacts") else c for c in cxs]
    events: List[Event] = []
    checks: List[Dict[str, Any]] = []
    for t in range(len(snaps) - 1):
        ev = detect_events(snaps[t], snaps[t + 1], frame=t, lineage=lineage)
        chk = admissibility_check(snaps[t], snaps[t + 1], ev)
        chk["frame"] = t
        events.extend(ev)
        checks.append(chk)
    n_pairs = len(checks)
    boundary_kinds = {"exit", "entry", "appearance", "disappearance", "gap_appearance", "gap_disappearance"}
    pairs_without_boundary = [
        c
        for c, t in zip(checks, range(n_pairs))
        if not any(e.frame == t and (e.kind in boundary_kinds or e.evidence.get("connectivity_change")) for e in events)
    ]
    t1_events = [e for e in events if e.kind == "t1"]
    counts = Counter(e.kind for e in events)
    out = {
        "run": label,
        "n_frames": len(snaps),
        "n_frame_pairs": n_pairs,
        "event_counts": dict(sorted(counts.items())),
        "n_events": len(events),
        "frac_pairs_fully_explained": float(np.mean([c["fully_explained"] for c in checks])) if checks else None,
        "frac_pairs_zero_residual": float(np.mean([c["residual_dVEF"] == (0, 0, 0) for c in checks]))
        if checks
        else None,
        "n_pairs_without_boundary_events": len(pairs_without_boundary),
        "frac_pairs_fully_explained_no_boundary": (
            float(np.mean([c["fully_explained"] for c in pairs_without_boundary])) if pairs_without_boundary else None
        ),
        "residual_abs_sum_histogram": dict(
            sorted(Counter(int(sum(abs(x) for x in c["residual_dVEF"])) for c in checks).items())
        ),
        "n_unexplained_events": int(sum(c["n_unexplained"] for c in checks)),
        "n_connectivity_changes": int(sum(c["n_connectivity_changes"] for c in checks)),
        "n_t1": len(t1_events),
        "n_t1_charge_conserved": int(sum(1 for e in t1_events if e.evidence.get("charge_delta") == 0)),
        "n_t1_generic_side_pattern": int(sum(1 for e in t1_events if e.evidence.get("generic"))),
        "n_t1_isolated": int(sum(c["n_t1_isolated"] for c in checks)),
        "n_t1_isolated_charge_conserved": int(sum(c["n_t1_isolated_charge_conserved"] for c in checks)),
        "n_t1_isolated_generic_side_pattern": int(sum(c["n_t1_isolated_generic"] for c in checks)),
        "t1_charge_delta_histogram": dict(
            sorted(Counter(int(e.evidence.get("charge_delta", 0)) for e in t1_events).items())
        ),
        "charge_total_first": checks[0]["charge_a"] if checks else None,
        "charge_total_last": checks[-1]["charge_b"] if checks else None,
        "mean_cells_per_frame": float(np.mean([len(s.cells) for s in snaps])),
        "mean_contacts_per_frame": float(np.mean([len(s.contacts) for s in snaps])),
        # cells touching the outer face sit at the free edge of the annotated region; events
        # there mostly record cells entering or leaving the curated ROI
        "n_extrusion_at_free_edge": int(
            sum(1 for e in events if e.kind == "extrusion" and 0 in e.participants.get("neighbours", []))
        ),
        "n_extrusion_neighbours_meet_at_vertex": int(
            sum(1 for e in events if e.kind == "extrusion" and e.evidence.get("neighbours_meet_at_vertex"))
        ),
        "n_t1_with_outer_face": int(sum(1 for e in t1_events if 0 in e.participants.get("cells", []))),
        "n_appearance_at_free_edge": int(
            sum(
                1 for e in events if e.kind in ("appearance", "disappearance") and 0 in e.evidence.get("neighbours", [])
            )
        ),
    }
    return {"summary": out, "events": events, "checks": checks, "snaps": snaps}


def diagnose_pair(t: int, events: List[Event], snaps, chk: Dict[str, Any]) -> str:
    """One-line reading of a frame pair whose (dV, dE, dF) identity does not close."""
    sa, sb = snaps[t], snaps[t + 1]
    ev = [e for e in events if e.frame == t]
    kinds = Counter(e.kind for e in ev)
    parts: List[str] = []
    n_border = kinds.get("exit", 0) + kinds.get("entry", 0)
    if n_border:
        parts.append(f"{n_border} cell(s) cross the image border (exit/entry, no fixed delta)")
    for kind, verb, snap in (("disappearance", "leave", sa), ("appearance", "enter", sb)):
        cells = [e.participants["cell"] for e in ev if e.kind == kind]
        if not cells:
            continue
        edge = [c for c in cells if 0 in snap.neighbours(c)]
        inner = [c for c in cells if c not in edge]
        if edge:
            parts.append(
                f"{len(edge)} cell(s) {edge[:4]} at the free edge of the annotated region {verb} the curated ROI "
                "(unlabelled tissue beyond, no fixed delta)"
            )
        if inner:
            parts.append(f"{len(inner)} cell(s) {inner[:4]} {verb} inside the tissue: id change or dropped label")
    # extrusions whose neighbours do not collapse onto one vertex leave k extra contacts: (+k, +k, 0)
    loose = [e for e in ev if e.kind == "extrusion" and not e.evidence.get("neighbours_meet_at_vertex")]
    if loose:
        k = sum(int(e.evidence.get("new_contacts_among_neighbours", 0)) for e in loose)
        at_edge = sum(1 for e in loose if 0 in e.participants.get("neighbours", []))
        parts.append(
            f"{len(loose)} extrusion(s) ({at_edge} at the free edge) whose neighbours close the footprint with "
            f"{k} new contact(s) instead of one vertex: adds about (+{k}, +{k}, 0) beyond the table"
        )
    n_gap = kinds.get("gap_appearance", 0) + kinds.get("gap_disappearance", 0)
    if n_gap:
        parts.append(f"{n_gap} background pocket(s) open/close without a matching nucleation/closure/rupture")
    n_conn = sum(1 for e in ev if e.evidence.get("connectivity_change"))
    if n_conn:
        parts.append(f"{n_conn} contact(s) join or split skeleton components (annotation islands)")
    n_ng = sum(1 for e in ev if e.kind == "t1" and not e.evidence.get("generic"))
    n_div = kinds.get("division", 0)
    if not parts:
        if n_ng:
            parts.append(
                f"{n_ng} T1(s) with a non-generic side pattern (fourfold vertex or overlapping rewrites), "
                f"{n_div} division(s)"
            )
        else:
            vs = len(sb.vertices - sa.vertices), len(sa.vertices - sb.vertices)
            parts.append(
                f"all events fixed-delta yet residual {chk['residual_dVEF']}: vertex set changed by "
                f"+{vs[0]}/-{vs[1]} without a matched contact change (fourfold vertex or division geometry)"
            )
    return "; ".join(parts)


def residual_table(r: Dict[str, Any], top: int = 10) -> pd.DataFrame:
    """Frame pairs with the largest |residual| + unexplained events, with a diagnosis each."""
    rows = []
    for chk in r["checks"]:
        t = int(chk["frame"])
        mag = int(sum(abs(x) for x in chk["residual_dVEF"])) + int(chk["n_unexplained"])
        if chk["fully_explained"]:
            continue
        ev = [e for e in r["events"] if e.frame == t]
        unexplained = [e for e in ev if e.expected_delta is None]
        rows.append(
            {
                "frame": t,
                "magnitude": mag,
                "observed_dVEF": chk["observed_dVEF"],
                "expected_dVEF": chk["expected_dVEF"],
                "residual_dVEF": chk["residual_dVEF"],
                "n_events": len(ev),
                "n_unexplained": len(unexplained),
                "unexplained_kinds": dict(sorted(Counter(e.kind for e in unexplained).items())),
                "diagnosis": diagnose_pair(t, r["events"], r["snaps"], chk),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["magnitude", "frame"], ascending=[False, True]).head(top).reset_index(drop=True)
    return df


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_report(
    out_dir: Path,
    meta: Dict[str, Any],
    runs: List[Dict[str, Any]],
    extra: Dict[str, Any],
    residuals: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    lines = [f"# Dynamics evaluation: {meta['dataset_label']}", ""]
    lines.append("Label stack: " + meta["label_stack_note"])
    lines.append("")
    for k, v in meta.items():
        if k in ("dataset_label", "label_stack_note"):
            continue
        lines.append(f"- {k}: {fmt(v)}")
    lines.append("")
    kinds = sorted({k for r in runs for k in r["summary"]["event_counts"]})
    lines.append("## Event counts (whole sequence)")
    lines.append("")
    lines.append("| kind | " + " | ".join(r["summary"]["run"] for r in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for k in kinds:
        lines.append(f"| {k} | " + " | ".join(str(r["summary"]["event_counts"].get(k, 0)) for r in runs) + " |")
    lines.append("")
    lines.append("## Admissibility (exact (dV, dE, dF) identity per frame pair)")
    lines.append("")
    keys = [
        "n_frame_pairs",
        "frac_pairs_fully_explained",
        "frac_pairs_zero_residual",
        "n_pairs_without_boundary_events",
        "frac_pairs_fully_explained_no_boundary",
        "n_unexplained_events",
        "n_connectivity_changes",
        "residual_abs_sum_histogram",
        "n_extrusion_at_free_edge",
        "n_extrusion_neighbours_meet_at_vertex",
        "n_t1_with_outer_face",
        "n_appearance_at_free_edge",
    ]
    lines.append("| metric | " + " | ".join(r["summary"]["run"] for r in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(r["summary"].get(k, "")) for r in runs) + " |")
    lines.append("")
    lines.append("## T1 charge conservation")
    lines.append("")
    lines.append("| metric | " + " | ".join(r["summary"]["run"] for r in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for k in (
        "n_t1",
        "n_t1_charge_conserved",
        "n_t1_generic_side_pattern",
        "n_t1_isolated",
        "n_t1_isolated_charge_conserved",
        "n_t1_isolated_generic_side_pattern",
        "t1_charge_delta_histogram",
    ):
        lines.append(f"| {k} | " + " | ".join(fmt(r["summary"][k]) for r in runs) + " |")
    lines.append("")
    lines.append("## Divisions and tracking")
    lines.append("")
    for name, d in extra.items():
        lines.append(f"### {name}")
        lines.append("")
        for k, v in d.items():
            if isinstance(v, (list, dict)) and len(str(v)) > 200:
                continue
            lines.append(f"- {k}: {fmt(v)}")
        lines.append("")
    for run, df in (residuals or {}).items():
        lines.append(f"## Largest unexplained residuals ({run})")
        lines.append("")
        if not len(df):
            lines.append("every frame pair is fully explained")
            lines.append("")
            continue
        lines.append("| frame pair | observed dVEF | expected dVEF | residual | unexplained | diagnosis |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            lines.append(
                f"| {row['frame']} to {row['frame'] + 1} | {row['observed_dVEF']} | {row['expected_dVEF']} | "
                f"{row['residual_dVEF']} | {row['unexplained_kinds']} | {row['diagnosis']} |"
            )
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ctc", "tissueminer", "epicure"], required=True)
    ap.add_argument("--name", default="DIC-C2DH-HeLa")
    ap.add_argument("--seq", default="01")
    ap.add_argument("--root", default=None)
    ap.add_argument("--movie", default=None, help="epicure: movie directory holding <name>.tif and epics_corrected/")
    ap.add_argument("--labels-subdir", default="epics_corrected", help="epicure: label folder inside the movie dir")
    ap.add_argument("--no-tracker", action="store_true", help="skip the tracker run (curated ids only)")
    ap.add_argument("--top-residuals", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--iou-min", type=float, default=0.3)
    ap.add_argument("--min-gap-px", type=int, default=0, help="fill enclosed background components below this size")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--out", default="runs/dynamics")
    args = ap.parse_args()

    t0 = time.time()
    extra: Dict[str, Any] = {}
    db_snaps: Optional[Dict[int, Any]] = None
    if args.dataset == "ctc":
        data = load_ctc(args.name, seq=args.seq, root=args.root or "data/ctc", max_frames=args.max_frames)
        if data["st_labels"] is None:
            raise SystemExit("this sequence has no silver-truth dense masks; cannot build a dense label stack")
        stack, stats = tracked_dense_labels(data["tra_labels"], data["st_labels"])
        gt_divisions = ctc_divisions(data["lineage"])
        gt_divisions = [d for d in gt_divisions if d["frame"] < len(stack)]
        gt_lineage = {c: d["parent"] for d in gt_divisions for c in d["children"]}
        name = f"{args.name}_{args.seq}"
        meta = {
            "dataset_label": name,
            "label_stack_note": (
                "CTC silver-truth masks (xx_ST/SEG) relabelled with the gold TRA marker ids "
                "(gold TRA images are markers, not masks); "
                f"mask/marker stats {stats}"
            ),
            "n_frames": len(stack),
            "n_gt_divisions": len(gt_divisions),
        }
    elif args.dataset == "epicure":
        if not args.movie:
            raise SystemExit("--dataset epicure needs --movie <dir>")
        lab, _, ep_meta = load_epicure_movie(args.movie, labels_subdir=args.labels_subdir, load_images=False)
        if args.max_frames is not None:
            lab = lab[: args.max_frames]
        stack = [fr.astype(np.int64) for fr in lab]
        gt_divisions = []
        gt_lineage = {}
        name = f"epicure_{Path(args.movie).name}"
        meta = {
            "dataset_label": name,
            "label_stack_note": (
                f"EpiCure curated labels ({ep_meta['labels_folder']}/{ep_meta['name']}_labels.tif, ids "
                "track-consistent, no lineage in the label file); 1 px background seams between cells filled "
                f"with fill_gt_slivers plus a seam pass ({ep_meta['n_seam_pixels_filled']} px over the movie)"
            ),
            "n_frames": len(stack),
            "n_gt_divisions": 0,
            "pixel_size_um": ep_meta["pixel_size_um"],
            "frame_interval_s": ep_meta["frame_interval_s"],
            "n_ids": ep_meta["n_ids"],
            "cells_per_frame_min": min(ep_meta["cells_per_frame"]),
            "cells_per_frame_max": max(ep_meta["cells_per_frame"]),
        }
    else:
        data = load_tissueminer_demo(root=args.root or "data/tissueminer", max_frames=args.max_frames)
        stack = data["label_stack"]
        gt_divisions = [d for d in data["divisions"] if d["frame"] < len(stack)]
        gt_lineage = {c: d["parent"] for d in gt_divisions for c in d["children"]}
        name = "tissueminer_demo"
        meta = {
            "dataset_label": name,
            "label_stack_note": (
                "Tissue Analyzer tracked_cells_resized.tif decoded (id = R*65536 + G*256 + B), 1 px white "
                "boundary lattice handed to the nearest cell, ids mapped to demo.sqlite cell_id through "
                f"cell_histories; stats {data['stats']}"
            ),
            "n_frames": len(stack),
            "n_gt_divisions": len(gt_divisions),
        }
        db_snaps = db_snapshots(Path(data["db_path"]), frames=list(range(len(stack))))
    if args.min_gap_px > 0:
        stack = [fill_small_gaps(lab, args.min_gap_px) for lab in stack]
        meta["min_gap_px"] = args.min_gap_px
        name = f"{name}{args.tag or '_mingap' + str(args.min_gap_px)}"
    elif args.tag:
        name = name + args.tag
    meta["dataset_label"] = name
    print(f"loaded {name}: {len(stack)} frames, {len(gt_divisions)} GT divisions ({time.time() - t0:.1f}s)")

    # run 1: our tracker on the ground-truth label stack
    runs = []
    r_tracker = None
    id_map = None
    tracks = None
    if not args.no_tracker:
        t1 = time.time()
        tracks = track(stack, iou_min=args.iou_min)
        tm = tracking_metrics(tracks, stack)
        id_map = tm.pop("pred_to_gt")
        print(f"tracked: {tracks.n_tracks} tracks, {len(tracks.lineage)} lineage links ({time.time() - t1:.1f}s)")
        cxs = complexes_over_time(tracks)
        r_tracker = run_events(cxs, tracks.lineage, "tracker")
        div_tracker = division_detection_metrics(r_tracker["events"], gt_divisions, tolerance_frames=1, id_map=id_map)
        div_tracker_linker = division_detection_metrics(
            tracks.divisions(), gt_divisions, tolerance_frames=1, id_map=id_map
        )
        extra["tracking_metrics (tracker vs GT ids)"] = tm
        extra["division detection (tracker run, events)"] = {
            k: v for k, v in div_tracker.items() if not k.startswith("unmatched")
        }
        extra["division detection (tracker run, linker lineage only)"] = {
            k: v for k, v in div_tracker_linker.items() if not k.startswith("unmatched")
        }
        runs.append(r_tracker)

    # run 2: ground-truth ids and lineage
    gt_tracks = tracks_from_labels(stack, lineage=gt_lineage)
    cxs_gt = [extract_complex(lab, provenance={"frame": t}) for t, lab in enumerate(gt_tracks.frame_labels)]
    r_gt = run_events(cxs_gt, gt_lineage, "gt_ids")
    div_gt = division_detection_metrics(r_gt["events"], gt_divisions, tolerance_frames=1)
    extra["division detection (gt_ids run)"] = {k: v for k, v in div_gt.items() if not k.startswith("unmatched")}
    runs.append(r_gt)

    # TissueMiner: the same detector on the database topology gives the reference T1 list.
    # The database only covers the analysis ROI, so predictions are also scored after
    # restricting them to events whose cells (mapped to GT ids) are database cells.
    if db_snaps is not None:
        frames = sorted(db_snaps)
        r_db = run_events([db_snaps[f] for f in frames], gt_lineage, "database")
        runs.append(r_db)
        gt_t1 = [e for e in r_db["events"] if e.kind == "t1"]
        if tracks is not None:
            tm_codes = tracking_metrics(tracks, data["code_stack"])
            tm_codes.pop("pred_to_gt")
            extra["tracking_metrics (tracker vs Tissue Analyzer codes)"] = tm_codes

        def in_roi(e: Event, im: Optional[Dict[int, int]]) -> bool:
            cells = e.participants.get("cells") or [e.participants.get("parent"), *e.participants.get("children", [])]
            ids = [int((im or {}).get(c, c)) for c in cells if c is not None]
            return all(0 < c < UNMAPPED_OFFSET for c in ids)

        scored = [(r_gt, None)] if r_tracker is None else [(r_tracker, id_map), (r_gt, None)]
        for r, im in scored:
            pred_t1 = [e for e in r["events"] if e.kind == "t1"]
            pred_t1_roi = [e for e in pred_t1 if in_roi(e, im)]
            extra[f"t1 vs database ({r['summary']['run']}, all)"] = t1_metrics(pred_t1, gt_t1, 0, id_map=im)
            extra[f"t1 vs database ({r['summary']['run']}, ROI cells only)"] = t1_metrics(
                pred_t1_roi, gt_t1, 0, id_map=im
            )
            pred_div_roi = [e for e in r["events"] if e.kind == "division" and in_roi(e, im)]
            extra[f"division detection ({r['summary']['run']}, ROI cells only)"] = {
                k: v
                for k, v in division_detection_metrics(pred_div_roi, gt_divisions, 1, id_map=im).items()
                if not k.startswith("unmatched")
            }
        extra["division detection (database run)"] = {
            k: v
            for k, v in division_detection_metrics(r_db["events"], gt_divisions, tolerance_frames=1).items()
            if not k.startswith("unmatched")
        }

    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in runs:
        for e in r["events"]:
            rec = e.as_record()
            rec["run"] = r["summary"]["run"]
            rec["participants"] = json.dumps(rec["participants"])
            rec["evidence"] = json.dumps(rec["evidence"])
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "events.csv", index=False)
    residuals: Dict[str, pd.DataFrame] = {}
    for r in runs:
        event_summary(r["events"]).to_csv(out_dir / f"event_summary_{r['summary']['run']}.csv", index=False)
        pd.DataFrame(r["checks"]).to_csv(out_dir / f"admissibility_{r['summary']['run']}.csv", index=False)
        res = residual_table(r, top=args.top_residuals)
        res.to_csv(out_dir / f"residuals_{r['summary']['run']}.csv", index=False)
        residuals[r["summary"]["run"]] = res
    summary = {"meta": meta, "runs": [r["summary"] for r in runs], "extra": extra, "seconds": time.time() - t0}
    summary["largest_residuals"] = {k: v.to_dict(orient="records") for k, v in residuals.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out_dir, meta, runs, extra, residuals)
    print(json.dumps({"runs": [r["summary"] for r in runs], "extra": extra}, indent=1, default=str))
    print(f"wrote {out_dir} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
