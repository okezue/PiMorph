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
    }
    return {"summary": out, "events": events, "checks": checks}


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_report(out_dir: Path, meta: Dict[str, Any], runs: List[Dict[str, Any]], extra: Dict[str, Any]) -> None:
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
    ]
    lines.append("| metric | " + " | ".join(r["summary"]["run"] for r in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(r["summary"][k]) for r in runs) + " |")
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
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ctc", "tissueminer"], required=True)
    ap.add_argument("--name", default="DIC-C2DH-HeLa")
    ap.add_argument("--seq", default="01")
    ap.add_argument("--root", default=None)
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
    t1 = time.time()
    tracks = track(stack, iou_min=args.iou_min)
    tm = tracking_metrics(tracks, stack)
    id_map = tm.pop("pred_to_gt")
    print(f"tracked: {tracks.n_tracks} tracks, {len(tracks.lineage)} lineage links ({time.time() - t1:.1f}s)")
    cxs = complexes_over_time(tracks)
    r_tracker = run_events(cxs, tracks.lineage, "tracker")
    div_tracker = division_detection_metrics(r_tracker["events"], gt_divisions, tolerance_frames=1, id_map=id_map)
    div_tracker_linker = division_detection_metrics(tracks.divisions(), gt_divisions, tolerance_frames=1, id_map=id_map)
    extra["tracking_metrics (tracker vs GT ids)"] = tm
    extra["division detection (tracker run, events)"] = {
        k: v for k, v in div_tracker.items() if not k.startswith("unmatched")
    }
    extra["division detection (tracker run, linker lineage only)"] = {
        k: v for k, v in div_tracker_linker.items() if not k.startswith("unmatched")
    }

    # run 2: ground-truth ids and lineage
    gt_tracks = tracks_from_labels(stack, lineage=gt_lineage)
    cxs_gt = [extract_complex(lab, provenance={"frame": t}) for t, lab in enumerate(gt_tracks.frame_labels)]
    r_gt = run_events(cxs_gt, gt_lineage, "gt_ids")
    div_gt = division_detection_metrics(r_gt["events"], gt_divisions, tolerance_frames=1)
    extra["division detection (gt_ids run)"] = {k: v for k, v in div_gt.items() if not k.startswith("unmatched")}
    runs = [r_tracker, r_gt]

    # TissueMiner: the same detector on the database topology gives the reference T1 list.
    # The database only covers the analysis ROI, so predictions are also scored after
    # restricting them to events whose cells (mapped to GT ids) are database cells.
    if db_snaps is not None:
        frames = sorted(db_snaps)
        r_db = run_events([db_snaps[f] for f in frames], gt_lineage, "database")
        runs.append(r_db)
        gt_t1 = [e for e in r_db["events"] if e.kind == "t1"]
        tm_codes = tracking_metrics(tracks, data["code_stack"])
        tm_codes.pop("pred_to_gt")
        extra["tracking_metrics (tracker vs Tissue Analyzer codes)"] = tm_codes

        def in_roi(e: Event, im: Optional[Dict[int, int]]) -> bool:
            cells = e.participants.get("cells") or [e.participants.get("parent"), *e.participants.get("children", [])]
            ids = [int((im or {}).get(c, c)) for c in cells if c is not None]
            return all(0 < c < UNMAPPED_OFFSET for c in ids)

        for r, im in ((r_tracker, id_map), (r_gt, None)):
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
    for r in runs:
        event_summary(r["events"]).to_csv(out_dir / f"event_summary_{r['summary']['run']}.csv", index=False)
        pd.DataFrame(r["checks"]).to_csv(out_dir / f"admissibility_{r['summary']['run']}.csv", index=False)
    summary = {"meta": meta, "runs": [r["summary"] for r in runs], "extra": extra, "seconds": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out_dir, meta, runs, extra)
    print(json.dumps({"runs": [r["summary"] for r in runs], "extra": extra}, indent=1, default=str))
    print(f"wrote {out_dir} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
