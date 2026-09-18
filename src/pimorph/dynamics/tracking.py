"""Frame-to-frame cell linking by pixel overlap and track-id relabelling.

Cells are label ids of the given images. ``link_frames`` matches consecutive
frames by IoU (Hungarian assignment), ``track`` chains links into stable track ids
and rewrites every frame so that each pixel carries its track id. Complexes
extracted from the relabelled frames then share ``face_label`` identities across
time, which is what ``pimorph.dynamics.events_detect`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon as draw_polygon
from skimage.measure import label as cc_label
from skimage.segmentation import expand_labels

from ..complex.extract import extract_complex
from ..complex.geometry import loop_signed_area
from ..complex.halfedge import FaceKind, HalfEdgeComplex


# ----------------------------------------------------------------- overlaps
def _overlap_table(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel intersections between nonzero labels of a and b: (labels_a, labels_b, counts)."""
    m = (a > 0) & (b > 0)
    la = a[m].astype(np.int64)
    lb = b[m].astype(np.int64)
    if la.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    base = np.int64(lb.max()) + 1
    code = la * base + lb
    uniq, counts = np.unique(code, return_counts=True)
    return uniq // base, uniq % base, counts.astype(np.int64)


def _areas(a: np.ndarray) -> np.ndarray:
    return np.bincount(a.ravel().astype(np.int64))


@dataclass
class FrameLink:
    """Matching between the labels of frame a and frame b."""

    pairs: List[Tuple[int, int, float]]  # (label_a, label_b, iou)
    unmatched_a: List[int]
    unmatched_b: List[int]
    candidates_split: Dict[int, List[int]]  # label_a -> b labels each holding >= split_frac of a's area
    candidates_merge: Dict[int, List[int]]  # label_b -> a labels each holding >= split_frac of b's area
    divisions: List[Tuple[int, Tuple[int, int]]]  # (parent label_a, (child label_b, child label_b))
    iou_min: float
    a_to_b: Dict[int, int] = field(default_factory=dict)
    b_to_a: Dict[int, int] = field(default_factory=dict)


def link_frames(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    iou_min: float = 0.3,
    split_frac: float = 0.2,
    child_frac: float = 0.35,
    max_child_iou: float = 0.6,
    detect_divisions: bool = True,
) -> FrameLink:
    """Match cells of two consecutive label images by pixel IoU.

    Hungarian assignment maximises the summed IoU over label pairs with any
    overlap; pairs below ``iou_min`` are dropped. ``candidates_split`` lists a-labels
    overlapping at least two b-labels that each hold >= ``split_frac`` of a's area;
    ``candidates_merge`` is the symmetric list. With ``detect_divisions`` a split
    candidate whose two children each lie >= ``child_frac`` inside the parent, none
    of which continues the parent (IoU <= ``max_child_iou``), and whose assignment
    partner (if any) is one of them is reported as a division and removed from the
    one-to-one pairs.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label images differ in shape: {a.shape} vs {b.shape}")
    ids_a = np.unique(a[a > 0]).astype(np.int64)
    ids_b = np.unique(b[b > 0]).astype(np.int64)
    ra, cb, cnt = _overlap_table(a, b)
    area_a = _areas(a)
    area_b = _areas(b)

    pairs: List[Tuple[int, int, float]] = []
    if ra.size:
        rows = np.unique(ra)
        cols = np.unique(cb)
        ri = np.searchsorted(rows, ra)
        ci = np.searchsorted(cols, cb)
        inter = np.zeros((rows.size, cols.size), dtype=np.float64)
        inter[ri, ci] = cnt
        union = area_a[rows][:, None] + area_b[cols][None, :] - inter
        iou = np.where(inter > 0, inter / np.maximum(union, 1), 0.0)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        for i, j in zip(ii, jj):
            if iou[i, j] >= iou_min:
                pairs.append((int(rows[i]), int(cols[j]), float(iou[i, j])))

    frac_a = cnt / np.maximum(area_a[ra], 1) if ra.size else cnt
    frac_b = cnt / np.maximum(area_b[cb], 1) if ra.size else cnt
    split: Dict[int, List[int]] = {}
    merge: Dict[int, List[int]] = {}
    for la, lb, fa, fb in zip(ra.tolist(), cb.tolist(), frac_a.tolist(), frac_b.tolist()):
        if fa >= split_frac:
            split.setdefault(la, []).append(lb)
        if fb >= split_frac:
            merge.setdefault(lb, []).append(la)
    split = {k: v for k, v in split.items() if len(v) >= 2}
    merge = {k: v for k, v in merge.items() if len(v) >= 2}

    divisions: List[Tuple[int, Tuple[int, int]]] = []
    if detect_divisions and split:
        inside = {(la, lb): fb for la, lb, fb in zip(ra.tolist(), cb.tolist(), frac_b.tolist())}
        pair_iou = {
            (la, lb): c / max(area_a[la] + area_b[lb] - c, 1)
            for la, lb, c in zip(ra.tolist(), cb.tolist(), cnt.tolist())
        }
        a2b = {p[0]: p[1] for p in pairs}
        b2a = {p[1]: p[0] for p in pairs}
        # predecessor holding the largest share of each b label; a daughter's is the mother
        best_pred: Dict[int, Tuple[int, int]] = {}
        for la, lb, c in zip(ra.tolist(), cb.tolist(), cnt.tolist()):
            if c > best_pred.get(lb, (0, -1))[0]:
                best_pred[lb] = (c, la)
        for la, kids in split.items():
            kids = [lb for lb in kids if inside[(la, lb)] >= child_frac and best_pred[lb][1] == la]
            if len(kids) != 2:
                continue
            if any(pair_iou[(la, k)] > max_child_iou for k in kids):
                continue
            partner = a2b.get(la)
            if partner is not None and partner not in kids:
                continue
            # a child already claimed by another parent label is a neighbour, not a daughter
            if any(b2a.get(k, la) != la for k in kids):
                continue
            divisions.append((int(la), (int(kids[0]), int(kids[1]))))
        parents = {d[0] for d in divisions}
        children = {k for d in divisions for k in d[1]}
        pairs = [p for p in pairs if p[0] not in parents and p[1] not in children]

    matched_a = {p[0] for p in pairs}
    matched_b = {p[1] for p in pairs}
    return FrameLink(
        pairs=pairs,
        unmatched_a=[int(x) for x in ids_a if int(x) not in matched_a],
        unmatched_b=[int(x) for x in ids_b if int(x) not in matched_b],
        candidates_split=split,
        candidates_merge=merge,
        divisions=divisions,
        iou_min=float(iou_min),
        a_to_b={p[0]: p[1] for p in pairs},
        b_to_a={p[1]: p[0] for p in pairs},
    )


# ------------------------------------------------------------------ cleaning
def clean_labels(labels: np.ndarray) -> np.ndarray:
    """Keep one 4-connected component per label so that face_label is unique.

    Smaller fragments are handed to the nearest surviving label; fragments that
    would still form a second component afterwards become background.
    """
    lab = np.asarray(labels).astype(np.int64)
    cc = cc_label(lab, connectivity=1, background=0)
    n_cc = int(cc.max())
    n_lab = int(np.count_nonzero(np.bincount(lab.ravel())[1:])) if lab.size else 0
    if n_cc == n_lab:
        return lab
    out = lab.copy()
    for _ in range(2):
        cc = cc_label(out, connectivity=1, background=0)
        flat_c = cc.ravel()
        flat_l = out.ravel()
        idx = np.flatnonzero(flat_c)
        sizes = np.bincount(flat_c[idx])
        comp_label = np.zeros(sizes.size, dtype=np.int64)
        comp_label[flat_c[idx]] = flat_l[idx]
        keep: Dict[int, int] = {}
        for c in np.argsort(-sizes):
            if c == 0 or sizes[c] == 0:
                continue
            keep.setdefault(int(comp_label[c]), int(c))
        keep_mask = np.zeros(sizes.size, dtype=bool)
        keep_mask[list(keep.values())] = True
        frag = (cc > 0) & ~keep_mask[cc]
        if not frag.any():
            break
        out[frag] = 0
        filled = expand_labels(out, distance=float(max(out.shape)))
        out[frag] = filled[frag]
    cc = cc_label(out, connectivity=1, background=0)
    if int(cc.max()) != int(np.count_nonzero(np.bincount(out.ravel())[1:])):
        flat_c = cc.ravel()
        idx = np.flatnonzero(flat_c)
        sizes = np.bincount(flat_c[idx])
        comp_label = np.zeros(sizes.size, dtype=np.int64)
        comp_label[flat_c[idx]] = out.ravel()[idx]
        keep = {}
        for c in np.argsort(-sizes):
            if c == 0 or sizes[c] == 0:
                continue
            keep.setdefault(int(comp_label[c]), int(c))
        keep_mask = np.zeros(sizes.size, dtype=bool)
        keep_mask[list(keep.values())] = True
        out[(cc > 0) & ~keep_mask[cc]] = 0
    return out


# ------------------------------------------------------------------- tracks
@dataclass
class Tracks:
    """Track-id relabelled frames plus lineage and life spans."""

    frame_labels: List[np.ndarray]  # pixel value = track id, 0 background
    lineage: Dict[int, int]  # child track -> parent track
    births: Dict[int, int]  # track -> first frame
    deaths: Dict[int, int]  # track -> last frame
    label_to_track: List[Dict[int, int]]  # per frame: input label -> track id
    links: List[FrameLink]
    n_tracks: int
    iou_min: float

    @property
    def n_frames(self) -> int:
        return len(self.frame_labels)

    def track_ids(self, t: int) -> List[int]:
        return sorted(set(self.label_to_track[t].values()))

    def divisions(self) -> List[Dict[str, Any]]:
        """Divisions as (parent, children, frame) with frame = first frame of the children."""
        kids: Dict[int, List[int]] = {}
        for c, p in self.lineage.items():
            kids.setdefault(p, []).append(c)
        out = []
        for p, cs in sorted(kids.items()):
            out.append({"parent": int(p), "children": sorted(int(c) for c in cs), "frame": int(self.births[cs[0]])})
        return out


def fill_small_gaps(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Hand enclosed background components smaller than ``min_area`` px to the
    nearest cell (a resolution floor for gap faces; border-touching background stays)."""
    lab = np.asarray(labels).astype(np.int64)
    if min_area <= 0:
        return lab
    bg = cc_label(lab == 0, connectivity=1)
    n = int(bg.max())
    if n == 0:
        return lab
    sizes = np.bincount(bg.ravel(), minlength=n + 1)
    touch = np.zeros(n + 1, dtype=bool)
    for edge in (bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1]):
        touch[np.unique(edge)] = True
    small = (sizes < min_area) & ~touch
    small[0] = False
    holes = small[bg]
    if not holes.any():
        return lab
    filled = expand_labels(lab, distance=float(max(lab.shape)))
    out = lab.copy()
    out[holes] = filled[holes]
    return out


def tracks_from_labels(
    label_stack: Sequence[np.ndarray], lineage: Optional[Dict[int, int]] = None, clean: bool = True
) -> Tracks:
    """Wrap an already tracked label stack (pixel value = track id) as ``Tracks``."""
    frames = [np.asarray(lab).astype(np.int64) for lab in label_stack]
    if clean:
        frames = [clean_labels(lab) for lab in frames]
    births: Dict[int, int] = {}
    deaths: Dict[int, int] = {}
    l2t: List[Dict[int, int]] = []
    for t, lab in enumerate(frames):
        ids = np.unique(lab[lab > 0]).tolist()
        l2t.append({int(i): int(i) for i in ids})
        for i in ids:
            births.setdefault(int(i), t)
            deaths[int(i)] = t
    return Tracks(
        frame_labels=frames,
        lineage=dict(lineage or {}),
        births=births,
        deaths=deaths,
        label_to_track=l2t,
        links=[],
        n_tracks=len(births),
        iou_min=float("nan"),
    )


def track(label_stack: Sequence[np.ndarray], iou_min: float = 0.3, clean: bool = True) -> Tracks:
    """Chain ``link_frames`` over a label stack into stable track ids.

    Matched cells inherit the track id, unmatched cells of the new frame start a
    new track, and a parent that splits into two children (``FrameLink.divisions``)
    ends while both children start new tracks recorded in ``lineage``.
    """
    if len(label_stack) == 0:
        raise ValueError("empty label stack")
    frames: List[np.ndarray] = []
    l2t: List[Dict[int, int]] = []
    births: Dict[int, int] = {}
    deaths: Dict[int, int] = {}
    lineage: Dict[int, int] = {}
    links: List[FrameLink] = []
    next_id = 1

    first = np.asarray(label_stack[0]).astype(np.int64)
    first = clean_labels(first) if clean else first
    mapping: Dict[int, int] = {}
    for lab in np.unique(first[first > 0]).tolist():
        mapping[int(lab)] = next_id
        births[next_id] = 0
        deaths[next_id] = 0
        next_id += 1
    frames.append(_relabel(first, mapping))
    l2t.append(mapping)

    for t in range(1, len(label_stack)):
        cur = np.asarray(label_stack[t]).astype(np.int64)
        cur = clean_labels(cur) if clean else cur
        link = link_frames(frames[-1], cur, iou_min=iou_min)
        links.append(link)
        mapping = {}
        for ta, lb, _ in link.pairs:
            mapping[int(lb)] = int(ta)
            deaths[int(ta)] = t
        for parent, kids in link.divisions:
            for k in kids:
                mapping[int(k)] = next_id
                lineage[next_id] = int(parent)
                births[next_id] = t
                deaths[next_id] = t
                next_id += 1
        for lb in link.unmatched_b:
            if int(lb) in mapping:
                continue
            mapping[int(lb)] = next_id
            births[next_id] = t
            deaths[next_id] = t
            next_id += 1
        frames.append(_relabel(cur, mapping))
        l2t.append(mapping)

    return Tracks(
        frame_labels=frames,
        lineage=lineage,
        births=births,
        deaths=deaths,
        label_to_track=l2t,
        links=links,
        n_tracks=next_id - 1,
        iou_min=float(iou_min),
    )


def _relabel(lab: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    if lab.size == 0:
        return lab.astype(np.int64)
    lut = np.zeros(int(lab.max()) + 1, dtype=np.int64)
    for k, v in mapping.items():
        lut[k] = v
    return lut[lab]


def complexes_over_time(tracks: Tracks, pixel_size_um: Optional[float] = None) -> List[HalfEdgeComplex]:
    """Extract one complex per relabelled frame; ``face_label`` equals the track id."""
    return [
        extract_complex(lab, pixel_size_um=pixel_size_um, provenance={"frame": t})
        for t, lab in enumerate(tracks.frame_labels)
    ]


# --------------------------------------------------------------- rasterize
def raster_face_labels(cx: HalfEdgeComplex) -> np.ndarray:
    """Per-face label used by ``complex_to_labels``: ``face_label`` where unique
    among cell faces, a fresh label above the maximum otherwise (a divided cell's
    child keeps the parent's label in the complex)."""
    lab = cx.face_label.astype(np.int64).copy()
    cells = [int(f) for f in cx.cell_faces]
    nxt = int(lab[cells].max()) + 1 if cells else 1
    used = set()
    for f in cells:
        if int(lab[f]) <= 0 or int(lab[f]) in used:
            lab[f] = nxt
            nxt += 1
        used.add(int(lab[f]))
    return lab


def _loop_points(cx: HalfEdgeComplex, loop: List[int]) -> np.ndarray:
    pts = []
    for h in loop:
        p = cx.edge_polyline[h >> 1]
        pts.append(p[:-1] if (h & 1) == 0 else p[::-1][:-1])
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))


def complex_to_labels(cx: HalfEdgeComplex, shape: Optional[Tuple[int, int]] = None, clean: bool = True) -> np.ndarray:
    """Rasterize the cell faces of a complex into a label image.

    Every non-outer face is filled from its outer loop polygon, largest area first,
    so that faces nested inside holes overwrite their container; gap faces write 0.
    Crack-aligned complexes round-trip exactly. After a geometric rewrite a polygon
    may pinch off a stray pixel; ``clean`` hands such fragments to a neighbour.
    """
    if shape is None:
        shape = cx.shape
    if shape is None:
        raise ValueError("shape is required when the complex has no shape")
    H, W = int(shape[0]), int(shape[1])
    lab = raster_face_labels(cx)
    items = []
    for f in range(cx.n_faces):
        if cx.face_kind[f] == FaceKind.OUTER or not cx.face_loops[f]:
            continue
        pts = _loop_points(cx, cx.face_loops[f][0])
        if pts.shape[0] < 3:
            continue
        items.append((abs(loop_signed_area(pts)), f, pts))
    items.sort(key=lambda t: -t[0])
    out = np.zeros((H, W), dtype=np.int64)
    covered = np.zeros((H, W), dtype=bool)
    gap_mask = np.zeros((H, W), dtype=bool)
    for _, f, pts in items:
        rr, cc = draw_polygon(pts[:, 0], pts[:, 1], shape=(H, W))
        if cx.face_kind[f] == FaceKind.CELL:
            out[rr, cc] = int(lab[f])
            covered[rr, cc] = True
        else:
            out[rr, cc] = 0
            gap_mask[rr, cc] = True
    if not clean:
        return out
    # Pixels no polygon claims and that neither a gap nor the outer region owns are
    # rasterization holes; hand them to the nearest cell.
    outside = ndi.binary_propagation(_border_mask(H, W) & ~covered, mask=~covered)
    holes = (out == 0) & ~gap_mask & ~outside
    if holes.any():
        filled = expand_labels(out, distance=float(max(H, W)))
        out[holes] = filled[holes]
    return clean_labels(out)


def _border_mask(H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


__all__ = [
    "FrameLink",
    "Tracks",
    "clean_labels",
    "complex_to_labels",
    "complexes_over_time",
    "fill_small_gaps",
    "link_frames",
    "raster_face_labels",
    "track",
    "tracks_from_labels",
]
