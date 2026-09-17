"""Per-edge arclength profiles of a marker channel.

The strip P_e(s, r) of an edge is reduced along the lateral axis into functions of
arclength (one value per arclength bin) and then into scalar summaries. Left and
right side means are reported separately so that half-edge asymmetry is available
downstream: ``left`` is r > 0, into ``edge_faces[e, 0]``; ``right`` is r < 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from ..complex.halfedge import HalfEdgeComplex
from .strip import Strip, edge_strip

SCALAR_FIELDS: Tuple[str, ...] = (
    "arclength_px",
    "n_bins",
    "mean_intensity",
    "max_intensity",
    "occupancy",
    "continuity",
    "n_segments",
    "gap_fraction",
    "width_mean_px",
    "width_cv",
    "left_intensity",
    "right_intensity",
    "threshold",
)


@dataclass
class EdgeProfile:
    """Arclength-binned marker profile of one edge.

    Per-bin arrays have length ``n_bins``; ``bin_edges_px`` has ``n_bins + 1`` entries
    from 0 to ``arclength_px``. ``width_px_s`` is the number of lateral samples above
    threshold (averaged over the sample rows in the bin) times the lateral step.
    """

    edge_id: int
    face_left: int
    face_right: int
    threshold: float
    bin_edges_px: np.ndarray
    mean_intensity_s: np.ndarray
    max_intensity_s: np.ndarray
    occupancy_s: np.ndarray
    width_px_s: np.ndarray
    left_intensity_s: np.ndarray
    right_intensity_s: np.ndarray
    arclength_px: float
    mean_intensity: float
    max_intensity: float
    occupancy: float
    continuity: float
    n_segments: int
    gap_fraction: float
    width_mean_px: float
    width_cv: float
    left_intensity: float
    right_intensity: float
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return int(self.occupancy_s.shape[0])

    @property
    def bin_widths_px(self) -> np.ndarray:
        return np.diff(self.bin_edges_px)

    def scalars(self) -> Dict[str, float]:
        d = {k: getattr(self, k) for k in SCALAR_FIELDS}
        d.update(self.extra)
        return d

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Start/stop (exclusive) index pairs of True runs."""
    if mask.size == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def profile_from_strip(
    strip: Strip,
    threshold: float,
    face_left: int,
    face_right: int,
    bin_px: float = 1.0,
) -> EdgeProfile:
    """Reduce a (single channel) strip to an EdgeProfile."""
    vals = np.asarray(strip.values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError("profile_from_strip expects a single-channel strip of shape (S, R)")
    S, R = vals.shape
    L = float(strip.arclength_px)
    if bin_px <= 0:
        raise ValueError("bin_px must be positive")
    B = max(1, int(round(L / bin_px))) if L > 0 else 1
    bin_edges = np.linspace(0.0, max(L, 0.0), B + 1)
    if L > 0 and S > 0:
        which = np.clip(np.searchsorted(bin_edges, strip.s, side="right") - 1, 0, B - 1)
    else:
        which = np.zeros(S, dtype=np.int64)

    above = vals > threshold
    lat_step = strip.lateral_step_px
    r = np.asarray(strip.r)
    left_cols = r > 0
    right_cols = r < 0

    mean_s = np.full(B, np.nan)
    max_s = np.full(B, np.nan)
    occ_s = np.zeros(B)
    width_s = np.zeros(B)
    left_s = np.full(B, np.nan)
    right_s = np.full(B, np.nan)
    for b in range(B):
        rows = which == b
        if not rows.any():
            continue
        v = vals[rows]
        a = above[rows]
        mean_s[b] = v.mean()
        max_s[b] = v.max()
        occ_s[b] = a.mean()
        width_s[b] = a.sum(axis=1).mean() * lat_step
        if left_cols.any():
            left_s[b] = v[:, left_cols].mean()
        if right_cols.any():
            right_s[b] = v[:, right_cols].mean()

    occupied = occ_s > 0
    runs = _runs(occupied)
    longest = max((b - a for a, b in runs), default=0)
    w_occ = width_s[occupied]
    if w_occ.size:
        w_mean = float(w_occ.mean())
        w_cv = float(w_occ.std() / w_mean) if (w_occ.size > 1 and w_mean > 0) else float("nan")
    else:
        w_mean = float("nan")
        w_cv = float("nan")

    def _nanmean(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    return EdgeProfile(
        edge_id=int(strip.edge_id),
        face_left=int(face_left),
        face_right=int(face_right),
        threshold=float(threshold),
        bin_edges_px=bin_edges,
        mean_intensity_s=mean_s,
        max_intensity_s=max_s,
        occupancy_s=occ_s,
        width_px_s=width_s,
        left_intensity_s=left_s,
        right_intensity_s=right_s,
        arclength_px=L,
        mean_intensity=float(vals.mean()) if vals.size else float("nan"),
        max_intensity=float(vals.max()) if vals.size else float("nan"),
        occupancy=float(above.mean()) if vals.size else float("nan"),
        continuity=float(longest / B),
        n_segments=int(len(runs)),
        gap_fraction=float(1.0 - occupied.mean()),
        width_mean_px=w_mean,
        width_cv=w_cv,
        left_intensity=_nanmean(left_s),
        right_intensity=_nanmean(right_s),
    )


def _channel(image: np.ndarray, channel: Optional[int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[0 if channel is None else int(channel)]
    raise ValueError(f"image must be (H, W) or (C, H, W), got shape {image.shape}")


def edge_profile(
    cx: HalfEdgeComplex,
    e: int,
    image: np.ndarray,
    threshold: float,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    bin_px: float = 1.0,
    channel: Optional[int] = None,
    order: int = 1,
) -> EdgeProfile:
    """Profile of edge ``e`` in one channel of ``image`` at a fixed threshold."""
    img = _channel(image, channel)
    strip = edge_strip(cx, e, img, half_width_px=half_width_px, step_px=step_px, n_lateral=n_lateral, order=order)
    a, b = (int(x) for x in cx.edge_faces[int(e)])
    return profile_from_strip(strip, threshold, a, b, bin_px=bin_px)


def strip_threshold(
    cx: HalfEdgeComplex,
    image: np.ndarray,
    edges: Optional[Iterable[int]] = None,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    channel: Optional[int] = None,
    order: int = 1,
) -> float:
    """Otsu threshold over the pooled strip samples of ``edges`` (default: all cell-cell edges)."""
    img = _channel(image, channel)
    ids = cx.cell_cell_edges() if edges is None else np.asarray(list(edges), dtype=np.int64)
    pooled = []
    for e in ids:
        st = edge_strip(cx, int(e), img, half_width_px=half_width_px, step_px=step_px, n_lateral=n_lateral, order=order)
        pooled.append(st.values.ravel())
    if not pooled:
        return float("nan")
    v = np.concatenate(pooled)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(threshold_otsu(v))


def profile_edges(
    cx: HalfEdgeComplex,
    image: np.ndarray,
    threshold: Optional[float] = None,
    edges: Optional[Sequence[int]] = None,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    bin_px: float = 1.0,
    channel: Optional[int] = None,
    order: int = 1,
) -> Dict[int, EdgeProfile]:
    """EdgeProfile for every requested edge (default: all cell-cell edges), keyed by edge id.

    A ``threshold`` of None is replaced by Otsu on the pooled strip samples of all
    cell-cell edges, independent of which edges are requested.
    """
    img = _channel(image, channel)
    ids = cx.cell_cell_edges() if edges is None else np.asarray(list(edges), dtype=np.int64)
    if threshold is None:
        threshold = strip_threshold(
            cx, img, edges=None, half_width_px=half_width_px, step_px=step_px, n_lateral=n_lateral, order=order
        )
    out: Dict[int, EdgeProfile] = {}
    for e in ids:
        e = int(e)
        out[e] = edge_profile(
            cx,
            e,
            img,
            float(threshold),
            half_width_px=half_width_px,
            step_px=step_px,
            n_lateral=n_lateral,
            bin_px=bin_px,
            order=order,
        )
    return out


def profiles_to_frame(cx: HalfEdgeComplex, profiles: Dict[int, EdgeProfile]) -> pd.DataFrame:
    """One row per edge with identifiers and the scalar summaries.

    The EdgeProfile objects are kept in ``df.attrs["profiles"]`` so functionals that
    need the per-bin vectors can be evaluated from the frame.
    """
    rows = []
    for e, prof in profiles.items():
        a, b = prof.face_left, prof.face_right
        row = {
            "edge_id": int(e),
            "face_left": a,
            "face_right": b,
            "label_left": int(cx.face_label[a]),
            "label_right": int(cx.face_label[b]),
        }
        row.update(prof.scalars())
        rows.append(row)
    cols = ["edge_id", "face_left", "face_right", "label_left", "label_right", *SCALAR_FIELDS]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    df.attrs["profiles"] = dict(profiles)
    return df


def profile_all_edges(
    cx: HalfEdgeComplex,
    image: np.ndarray,
    threshold: Optional[float] = None,
    edges: Optional[Sequence[int]] = None,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    bin_px: float = 1.0,
    channel: Optional[int] = None,
    order: int = 1,
) -> pd.DataFrame:
    """Profile table for ``edges`` (default: all cell-cell edges); see ``profile_edges``."""
    profiles = profile_edges(
        cx,
        image,
        threshold=threshold,
        edges=edges,
        half_width_px=half_width_px,
        step_px=step_px,
        n_lateral=n_lateral,
        bin_px=bin_px,
        channel=channel,
        order=order,
    )
    return profiles_to_frame(cx, profiles)
