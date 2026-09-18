"""Evaluation of detected divisions, T1s and track continuity against ground truth."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .events_detect import Event
from .tracking import Tracks


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)


def _ratio(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def divisions_from_lineage(lineage: Mapping[int, Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    """CTC ``man_track.txt`` rows {id: (start, end, parent)} -> divisions with two children.

    The division frame is the first frame of the children.
    """
    kids: Dict[int, List[Tuple[int, int]]] = {}
    for cid, (start, _end, parent) in lineage.items():
        if parent and parent > 0:
            kids.setdefault(int(parent), []).append((int(start), int(cid)))
    out = []
    for p, cs in sorted(kids.items()):
        if len(cs) < 2:
            continue
        frame = min(s for s, _ in cs)
        out.append({"parent": int(p), "children": sorted(c for _, c in cs), "frame": int(frame)})
    return out


def _as_division_records(items: Iterable[Union[Event, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        if isinstance(it, Event):
            if it.kind != "division":
                continue
            out.append(
                {
                    "parent": int(it.participants["parent"]),
                    "children": [int(c) for c in it.participants["children"]],
                    "frame": int(it.frame) + 1,
                }
            )
        else:
            out.append(
                {"parent": int(it["parent"]), "children": [int(c) for c in it["children"]], "frame": int(it["frame"])}
            )
    return out


def division_detection_metrics(
    pred_events: Iterable[Union[Event, Mapping[str, Any]]],
    gt_lineage: Union[Mapping[int, Tuple[int, int, int]], Sequence[Mapping[str, Any]]],
    tolerance_frames: int = 1,
    id_map: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """Precision/recall of detected divisions against ground-truth divisions.

    Predicted divisions are ``Event`` objects (frame = first frame of the children
    is ``event.frame + 1``) or dicts with parent/children/frame. Ground truth is a
    CTC lineage dict or a list of such dicts. ``id_map`` maps predicted track ids to
    ground-truth ids. A prediction matches a ground-truth division within
    ``tolerance_frames`` when the mapped parent agrees or the mapped children agree.
    """
    pred = _as_division_records(pred_events)
    if isinstance(gt_lineage, Mapping):
        gt = divisions_from_lineage(gt_lineage)
    else:
        gt = _as_division_records(gt_lineage)
    id_map = dict(id_map or {})

    def m(x: int) -> int:
        return int(id_map.get(x, x))

    used = set()
    tp = 0
    matches = []
    for k, p in enumerate(pred):
        best = None
        for j, g in enumerate(gt):
            if j in used or abs(p["frame"] - g["frame"]) > tolerance_frames:
                continue
            parent_ok = m(p["parent"]) == g["parent"]
            kids_ok = {m(c) for c in p["children"]} == set(g["children"])
            if parent_ok or kids_ok:
                cost = abs(p["frame"] - g["frame"]) - (0.5 if parent_ok and kids_ok else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, j)
        if best is not None:
            used.add(best[1])
            tp += 1
            matches.append((k, best[1]))
    precision = _ratio(tp, len(pred))
    recall = _ratio(tp, len(gt))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "n_pred": len(pred),
        "n_gt": len(gt),
        "n_tp": tp,
        "tolerance_frames": int(tolerance_frames),
        "unmatched_gt": [gt[j] for j in range(len(gt)) if j not in used],
        "unmatched_pred": [pred[k] for k in range(len(pred)) if k not in {a for a, _ in matches}],
    }


def _majority_map(pred: np.ndarray, gt: np.ndarray, min_frac: float = 0.5) -> Dict[int, int]:
    """gt label -> predicted label holding the largest share (>= min_frac) of its pixels."""
    m = gt > 0
    g = gt[m].astype(np.int64)
    p = pred[m].astype(np.int64)
    if g.size == 0:
        return {}
    base = np.int64(p.max()) + 1
    code = g * base + p
    uniq, cnt = np.unique(code, return_counts=True)
    gl = uniq // base
    pl = uniq % base
    area = np.bincount(g)
    out: Dict[int, int] = {}
    best: Dict[int, int] = {}
    for gi, pi, c in zip(gl.tolist(), pl.tolist(), cnt.tolist()):
        if pi == 0:
            continue
        if c > best.get(gi, 0) and c >= min_frac * area[gi]:
            best[gi] = c
            out[gi] = pi
    return out


def tracking_metrics(tracks: Tracks, gt_tra_labels: Sequence[np.ndarray], min_frac: float = 0.5) -> Dict[str, Any]:
    """Consistency of predicted track ids along ground-truth tracks.

    Per frame every GT cell is assigned the predicted track covering >= ``min_frac``
    of its pixels. A GT track is consistent when it is matched in every frame of its
    life by one and the same predicted track. ``n_id_switches`` counts frames where
    the matched predicted id differs from the previously matched one,
    ``n_fragmentations`` counts GT tracks matched to more than one predicted id.
    ``pred_to_gt`` is the majority mapping predicted track -> GT id over all frames.
    """
    if len(gt_tra_labels) != tracks.n_frames:
        raise ValueError("ground-truth stack and tracks differ in frame count")
    seq: Dict[int, List[Tuple[int, Optional[int]]]] = {}
    overlap: Dict[int, Dict[int, int]] = {}
    n_cells = 0
    n_matched = 0
    for t, (pred, gt) in enumerate(zip(tracks.frame_labels, gt_tra_labels)):
        gt = np.asarray(gt)
        mm = _majority_map(pred, gt, min_frac=min_frac)
        for gid in np.unique(gt[gt > 0]).tolist():
            n_cells += 1
            pid = mm.get(int(gid))
            if pid is not None:
                n_matched += 1
            seq.setdefault(int(gid), []).append((t, pid))
        m = (gt > 0) & (pred > 0)
        if m.any():
            base = np.int64(gt.max()) + 1
            code = pred[m].astype(np.int64) * base + gt[m].astype(np.int64)
            uniq, cnt = np.unique(code, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                overlap.setdefault(u // base, {})
                overlap[u // base][u % base] = overlap[u // base].get(u % base, 0) + c
    n_switch = 0
    n_frag = 0
    n_consistent = 0
    n_gaps = 0
    for gid, items in seq.items():
        ids = [pid for _, pid in items if pid is not None]
        distinct = set(ids)
        if len(ids) == len(items) and len(distinct) == 1:
            n_consistent += 1
        if len(distinct) > 1:
            n_frag += 1
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                n_switch += 1
        n_gaps += sum(1 for _, pid in items if pid is None)
    pred_to_gt = {int(p): int(max(d, key=d.get)) for p, d in overlap.items()}
    gt_per_pred = {p: len(d) for p, d in overlap.items()}
    return {
        "n_gt_tracks": len(seq),
        "n_pred_tracks": int(tracks.n_tracks),
        "cell_matching_accuracy": _ratio(n_consistent, len(seq)),
        "frame_match_rate": _ratio(n_matched, n_cells),
        "n_id_switches": int(n_switch),
        "n_fragmentations": int(n_frag),
        "n_unmatched_cell_frames": int(n_gaps),
        "n_pred_tracks_spanning_multiple_gt": int(sum(1 for n in gt_per_pred.values() if n > 1)),
        "pred_to_gt": pred_to_gt,
    }


def _t1_key(e: Union[Event, Mapping[str, Any]], id_map: Mapping[int, int]) -> Tuple[frozenset, frozenset, int]:
    if isinstance(e, Event):
        lost, gained, frame = e.participants["lost"], e.participants["gained"], e.frame
    else:
        lost, gained, frame = e["lost"], e["gained"], e["frame"]
    m = lambda x: int(id_map.get(int(x), int(x)))  # noqa: E731
    return frozenset(m(x) for x in lost), frozenset(m(x) for x in gained), int(frame)


def t1_metrics(
    pred_t1_events: Sequence[Union[Event, Mapping[str, Any]]],
    gt_t1_events: Optional[Sequence[Union[Event, Mapping[str, Any]]]] = None,
    tolerance_frames: int = 0,
    id_map: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """Match predicted T1s to ground-truth T1s by (lost pair, gained pair) and frame.

    Without ground truth only the counts are reported.
    """
    id_map = dict(id_map or {})
    pred = [_t1_key(e, id_map) for e in pred_t1_events if not isinstance(e, Event) or e.kind == "t1"]
    out: Dict[str, Any] = {"n_pred": len(pred)}
    if gt_t1_events is None:
        out.update({"n_gt": None, "n_tp": None, "precision": None, "recall": None, "f1": None})
        return out
    gt = [_t1_key(e, {}) for e in gt_t1_events if not isinstance(e, Event) or e.kind == "t1"]
    used = set()
    tp = 0
    for L, G, f in pred:
        for j, (L2, G2, f2) in enumerate(gt):
            if j in used or abs(f - f2) > tolerance_frames:
                continue
            if L == L2 and G == G2:
                used.add(j)
                tp += 1
                break
    precision = _ratio(tp, len(pred))
    recall = _ratio(tp, len(gt))
    out.update({"n_gt": len(gt), "n_tp": tp, "precision": precision, "recall": recall, "f1": _f1(precision, recall)})
    return out


__all__ = ["division_detection_metrics", "divisions_from_lineage", "t1_metrics", "tracking_metrics"]
