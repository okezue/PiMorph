"""Multi-marker junction fields along interfaces.

Junction type is not an exclusive label. Every edge carries a vector field
z_e(s) in R^m over arclength, one component per marker channel (AJ = VE-cadherin,
TJ = claudin-5, cytoskeleton = F-actin, ...). All channels of an edge are sampled on
the same strip geometry and reduced with the same arclength bins, so the per-bin
frames align across channels and cross-channel statistics (co-occupancy, Jaccard,
Pearson, exclusive fractions, combinatorial occupancy pattern) are exact.
Morphology classes are derived from the joint vector afterwards
(``joint_morphology_states``), never assigned per channel first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from ..complex.halfedge import HalfEdgeComplex
from .profile import EdgeProfile, profile_from_strip
from .strip import Strip, lateral_offsets, resample_polyline, sample_strip

PER_CHANNEL_SCALARS: Tuple[str, ...] = (
    "occupancy",
    "coverage",
    "mean_intensity",
    "max_intensity",
    "continuity",
    "n_segments",
    "gap_fraction",
    "width_mean_px",
    "width_cv",
    "left_intensity",
    "right_intensity",
    "threshold",
)


def pattern_name(channels: Sequence[str], mask: Sequence[bool]) -> str:
    """Name of a combinatorial occupancy pattern, e.g. ``"AJ+TJ"``, ``"AJ only"``, ``"none"``."""
    on = [c for c, m in zip(channels, mask) if m]
    if not on:
        return "none"
    if len(on) == 1 and len(channels) > 1:
        return f"{on[0]} only"
    return "+".join(on)


def all_patterns(channels: Sequence[str]) -> List[str]:
    """Every pattern name for ``channels`` in a fixed order (``none`` first)."""
    names = []
    for mask in product((False, True), repeat=len(channels)):
        names.append(pattern_name(channels, mask))
    return names


@dataclass
class MultiProfile:
    """Aligned multi-channel profiles of the cell-cell edges of one complex.

    ``per_bin`` has one row per (edge_id, s_bin) with ``s_lo_px``/``s_hi_px`` and, per
    channel ``c``: ``<c>_intensity`` (bin mean), ``<c>_occupancy`` (fraction of strip
    samples above threshold), ``<c>_width_px`` and ``<c>_occupied`` (occupancy >=
    ``bin_occupancy_min``). ``per_edge`` has one row per edge with per-channel scalars
    (``<c>_<name>`` for ``PER_CHANNEL_SCALARS``), cross-channel statistics and the
    dominant occupancy pattern ``vector_state`` with its arclength fraction
    ``vector_state_fraction`` (all pattern fractions are in ``frac_<pattern>``).
    """

    channels: List[str]
    per_bin: pd.DataFrame
    per_edge: pd.DataFrame
    thresholds: Dict[str, float]
    bin_px: float
    bin_occupancy_min: float
    profiles: Dict[str, Dict[int, EdgeProfile]] = field(default_factory=dict, repr=False)

    @property
    def edge_ids(self) -> np.ndarray:
        return self.per_edge["edge_id"].to_numpy(dtype=np.int64)

    @property
    def patterns(self) -> List[str]:
        return all_patterns(self.channels)

    def occupancy_matrix(self, e: int) -> np.ndarray:
        """(n_bins, m) boolean occupancy pattern z_e(s) thresholded per bin."""
        rows = self.per_bin[self.per_bin["edge_id"] == int(e)].sort_values("s_bin")
        return rows[[f"{c}_occupied" for c in self.channels]].to_numpy(dtype=bool)

    def vector_field(self, e: int) -> np.ndarray:
        """(n_bins, m) continuous per-bin occupancy fractions z_e(s) in [0, 1]^m."""
        rows = self.per_bin[self.per_bin["edge_id"] == int(e)].sort_values("s_bin")
        return rows[[f"{c}_occupancy" for c in self.channels]].to_numpy(dtype=np.float64)


def _as_plane(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] == 1:
        return img[0]
    raise ValueError(f"each channel must be a single (H, W) plane, got shape {img.shape}")


def _resolve_bin_px(cx: HalfEdgeComplex, bin_px: float, n_bins_per_um: Optional[float]) -> float:
    if n_bins_per_um is None:
        return float(bin_px)
    if cx.pixel_size_um is None:
        raise ValueError("n_bins_per_um requires a calibrated complex (cx.pixel_size_um)")
    if n_bins_per_um <= 0:
        raise ValueError("n_bins_per_um must be positive")
    return float(1.0 / (n_bins_per_um * cx.pixel_size_um))


def _otsu_or_max(v: np.ndarray) -> float:
    """Otsu threshold of pooled samples; a (near) constant channel gets its maximum so
    that nothing counts as occupied."""
    if v.size == 0:
        return float("nan")
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-9 * max(abs(hi), 1.0):
        return hi
    try:
        return float(threshold_otsu(v))
    except ValueError:
        return hi


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def multichannel_profiles(
    cx: HalfEdgeComplex,
    images: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
    edges: Optional[Sequence[int]] = None,
    n_bins_per_um: Optional[float] = None,
    bin_px: float = 1.0,
    half_width_px: float = 3.0,
    step_px: float = 1.0,
    n_lateral: int = 7,
    order: int = 1,
    bin_occupancy_min: float = 0.3,
) -> MultiProfile:
    """Profile every channel of ``images`` (name -> (H, W) array) on shared strips.

    Strip geometry is resampled once per edge and every channel is sampled on it, so the
    per-bin frames share (edge_id, s_bin) exactly. Missing thresholds are Otsu on the
    pooled strip samples of all cell-cell edges of that channel. ``n_bins_per_um``
    overrides ``bin_px`` when the complex is calibrated. A bin counts as occupied by a
    channel when at least ``bin_occupancy_min`` of its strip samples exceed the
    threshold.
    """
    channels = list(images.keys())
    if not channels:
        raise ValueError("at least one channel is required")
    planes = {c: _as_plane(images[c]) for c in channels}
    shapes = {p.shape for p in planes.values()}
    if len(shapes) != 1:
        raise ValueError(f"channels differ in shape: {shapes}")
    bpx = _resolve_bin_px(cx, bin_px, n_bins_per_um)
    ids = cx.cell_cell_edges() if edges is None else np.asarray(list(edges), dtype=np.int64)
    r = lateral_offsets(half_width_px, n_lateral)

    # sample strips once; pool values for Otsu thresholds over all cell-cell edges
    strips: Dict[int, Dict[str, Strip]] = {}
    pooled: Dict[str, List[np.ndarray]] = {c: [] for c in channels}
    thr = dict(thresholds or {})
    need_pool = [c for c in channels if c not in thr]
    pool_ids = cx.cell_cell_edges() if need_pool else np.zeros(0, dtype=np.int64)
    for e in np.union1d(ids, pool_ids):
        e = int(e)
        p = np.asarray(cx.edge_geometry(e), dtype=np.float64)
        pts, t, n, s = resample_polyline(p, step_px=step_px)
        seg = np.diff(p, axis=0)
        L = float(np.sqrt((seg**2).sum(axis=1)).sum()) if seg.size else 0.0
        per_c: Dict[str, Strip] = {}
        for c in channels:
            vals = sample_strip(planes[c], pts, n, half_width_px=half_width_px, n_lateral=n_lateral, order=order)
            per_c[c] = Strip(edge_id=e, values=vals, points=pts, tangents=t, normals=n, s=s, r=r, arclength_px=L)
            if c in need_pool and e in pool_ids:
                pooled[c].append(vals.ravel())
        strips[e] = per_c
    for c in need_pool:
        v = np.concatenate(pooled[c]) if pooled[c] else np.zeros(0)
        v = v[np.isfinite(v)]
        thr[c] = _otsu_or_max(v)

    profiles: Dict[str, Dict[int, EdgeProfile]] = {c: {} for c in channels}
    bin_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    patterns = all_patterns(channels)
    for e in ids:
        e = int(e)
        a, b = (int(x) for x in cx.edge_faces[e])
        profs = {c: profile_from_strip(strips[e][c], thr[c], a, b, bin_px=bpx) for c in channels}
        for c in channels:
            profiles[c][e] = profs[c]
        ref = profs[channels[0]]
        B = ref.n_bins
        occ = np.stack([profs[c].occupancy_s for c in channels], axis=1)  # (B, m)
        occupied = occ >= bin_occupancy_min
        inten = np.stack([profs[c].mean_intensity_s for c in channels], axis=1)
        for k in range(B):
            row: Dict[str, object] = {
                "edge_id": e,
                "s_bin": k,
                "s_lo_px": float(ref.bin_edges_px[k]),
                "s_hi_px": float(ref.bin_edges_px[k + 1]),
            }
            for j, c in enumerate(channels):
                row[f"{c}_intensity"] = float(inten[k, j])
                row[f"{c}_occupancy"] = float(occ[k, j])
                row[f"{c}_width_px"] = float(profs[c].width_px_s[k])
                row[f"{c}_occupied"] = bool(occupied[k, j])
            bin_rows.append(row)

        erow: Dict[str, object] = {
            "edge_id": e,
            "face_left": a,
            "face_right": b,
            "label_left": int(cx.face_label[a]),
            "label_right": int(cx.face_label[b]),
            "arclength_px": float(ref.arclength_px),
            "n_bins": int(B),
        }
        for j, c in enumerate(channels):
            sc = profs[c].scalars()
            sc["coverage"] = float(occupied[:, j].mean()) if B else float("nan")
            for name in PER_CHANNEL_SCALARS:
                erow[f"{c}_{name}"] = sc[name]
        for i, j in combinations(range(len(channels)), 2):
            ca, cb = channels[i], channels[j]
            both = occupied[:, i] & occupied[:, j]
            union = occupied[:, i] | occupied[:, j]
            erow[f"co_occupancy_{ca}_{cb}"] = float(both.mean()) if B else float("nan")
            erow[f"jaccard_{ca}_{cb}"] = float(both.sum() / union.sum()) if union.any() else float("nan")
            erow[f"pearson_{ca}_{cb}"] = _pearson(inten[:, i], inten[:, j])
        for j, c in enumerate(channels):
            others = np.delete(occupied, j, axis=1).any(axis=1) if len(channels) > 1 else np.zeros(B, dtype=bool)
            erow[f"exclusive_{c}"] = float((occupied[:, j] & ~others).mean()) if B else float("nan")
        names = np.array([pattern_name(channels, occupied[k]) for k in range(B)]) if B else np.zeros(0, dtype=str)
        widths = ref.bin_widths_px if B else np.zeros(0)
        total = float(widths.sum())
        best_name, best_frac = "none", float("nan")
        for pname in patterns:
            frac = float(widths[names == pname].sum() / total) if total > 0 else float("nan")
            erow[f"frac_{pname}"] = frac
            if total > 0 and (np.isnan(best_frac) or frac > best_frac):
                best_name, best_frac = pname, frac
        erow["vector_state"] = best_name
        erow["vector_state_fraction"] = best_frac
        edge_rows.append(erow)

    per_bin = pd.DataFrame(bin_rows)
    per_edge = pd.DataFrame(edge_rows)
    if per_edge.empty:
        per_edge = pd.DataFrame(columns=["edge_id", "face_left", "face_right", "vector_state"])
    if per_bin.empty:
        per_bin = pd.DataFrame(columns=["edge_id", "s_bin"])
    return MultiProfile(
        channels=channels,
        per_bin=per_bin,
        per_edge=per_edge,
        thresholds=thr,
        bin_px=bpx,
        bin_occupancy_min=float(bin_occupancy_min),
        profiles=profiles,
    )


def default_feature_columns(channels: Sequence[str]) -> List[str]:
    """Joint morphology features: per-channel coverage, continuity, width and intensity,
    plus the pairwise co-occupancy fractions."""
    cols = []
    for c in channels:
        cols += [f"{c}_coverage", f"{c}_continuity", f"{c}_width_mean_px", f"{c}_mean_intensity"]
    for a, b in combinations(channels, 2):
        cols.append(f"co_occupancy_{a}_{b}")
    return cols


def joint_morphology_states(
    per_edge: pd.DataFrame,
    feature_cols: Sequence[str],
    k_range: Tuple[int, int] = (2, 6),
    seed: int = 0,
    n_boot: int = 20,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Unsupervised joint junction states from the multichannel per-edge vector.

    Features are standardized (non-finite values imputed by the column median) and a
    Gaussian mixture is fitted for every k in ``k_range`` (inclusive); the BIC minimum
    is kept. Stability is the adjusted Rand index between the reference labels and the
    labels of a mixture refitted on a bootstrap resample, over ``n_boot`` resamples.
    Rows with no finite feature at all get label -1. Returns ``(labels, meta)`` with
    ``meta`` holding ``k``, ``bic`` per k, ``stability_ari_mean``/``_std``,
    ``state_means`` (original units) and ``state_counts``.
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    cols = [c for c in feature_cols if c in per_edge.columns]
    if not cols:
        raise ValueError("none of the requested feature columns are present")
    X = per_edge[cols].to_numpy(dtype=np.float64)
    n = X.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    meta: Dict[str, object] = {"features": cols, "n_edges": int(n)}
    valid = np.isfinite(X).any(axis=1)
    if valid.sum() < 2:
        meta.update({"k": 0, "bic": {}, "stability_ari_mean": float("nan"), "stability_ari_std": float("nan")})
        return labels, meta
    Xv = X[valid].copy()
    med = np.nanmedian(Xv, axis=0)
    med[~np.isfinite(med)] = 0.0
    bad = ~np.isfinite(Xv)
    Xv[bad] = np.take(med, np.nonzero(bad)[1])
    mu, sd = Xv.mean(axis=0), Xv.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xv - mu) / sd
    m = Z.shape[0]

    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    k_hi = max(k_lo, min(k_hi, max(1, m // 5)))
    ks = list(range(max(1, k_lo), k_hi + 1))
    bics: Dict[int, float] = {}
    fits: Dict[int, GaussianMixture] = {}
    for k in ks:
        gm = GaussianMixture(
            n_components=k, covariance_type=covariance_type, random_state=seed, n_init=3, reg_covar=1e-4
        )
        gm.fit(Z)
        bics[k] = float(gm.bic(Z))
        fits[k] = gm
    best_k = min(bics, key=bics.get)
    ref = fits[best_k]
    ref_labels = ref.predict(Z)
    labels[np.flatnonzero(valid)] = ref_labels

    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, m, size=m)
        gm = GaussianMixture(
            n_components=best_k,
            covariance_type=covariance_type,
            random_state=int(rng.integers(1 << 30)),
            reg_covar=1e-4,
        )
        try:
            gm.fit(Z[idx])
        except ValueError:
            continue
        aris.append(float(adjusted_rand_score(ref_labels, gm.predict(Z))))
    means = ref.means_ * sd + mu
    meta.update(
        {
            "k": int(best_k),
            "k_range_used": (ks[0], ks[-1]),
            "bic": bics,
            "stability_ari_mean": float(np.mean(aris)) if aris else float("nan"),
            "stability_ari_std": float(np.std(aris)) if aris else float("nan"),
            "n_boot": len(aris),
            "state_means": {int(i): dict(zip(cols, map(float, means[i]))) for i in range(best_k)},
            "state_counts": {int(i): int((ref_labels == i).sum()) for i in range(best_k)},
        }
    )
    return labels, meta
