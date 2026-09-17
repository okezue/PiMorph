"""Edge functionals: integrals over arclength profiles and legacy feature adapters.

``weighted_layer`` turns a per-bin functional phi into an edge weight
w_e = integral phi(z_e(s)) ds, the scalar layer the dual graph is weighted with.

``legacy_feature_frame`` maps profile scalars onto the per-edge feature names of the
EndoPiGraph pipeline (``AJ_occupancy``, ``AJ_cluster_count``, ...) and
``heuristic_morph_label`` reuses ``endopigraph.ajmorph.infer_ajmorph_label_heuristic``
so the strings stay comparable with earlier runs.

The morphology labels produced here are feature-derived states: deterministic
functions of occupancy, segment count, skeleton length and width thresholds. They
are not validated biological classes and have not been benchmarked against expert
annotation. Treat them as a coarse, comparable bookkeeping of junction appearance.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Union

import numpy as np
import pandas as pd

from endopigraph.ajmorph import AJMORPH_CLASSES, infer_ajmorph_label_heuristic

from .profile import EdgeProfile

ProfileSource = Union[pd.DataFrame, Mapping[int, EdgeProfile], Iterable[EdgeProfile]]

LEGACY_FEATURE_COLUMNS = (
    "AJ_occupancy",
    "AJ_mean_intensity",
    "AJ_max_intensity",
    "AJ_skeleton_len",
    "AJ_cluster_count",
    "AJ_cluster_density",
    "AJ_thickness_proxy",
)

# Bare legacy key -> possible column names, in lookup order.
_LEGACY_KEYS = {
    "occupancy": ("AJ_occupancy", "occupancy"),
    "cluster_count": ("AJ_cluster_count", "cluster_count", "n_segments"),
    "thickness_proxy": ("AJ_thickness_proxy", "thickness_proxy", "width_mean_px"),
    "skeleton_len": ("AJ_skeleton_len", "skeleton_len"),
    "mean": ("AJ_mean_intensity", "mean_intensity"),
    "max": ("AJ_max_intensity", "max_intensity"),
}


def _iter_profiles(profiles: ProfileSource) -> Iterable[EdgeProfile]:
    if isinstance(profiles, pd.DataFrame):
        stored = profiles.attrs.get("profiles")
        if stored is None:
            raise ValueError("DataFrame has no attrs['profiles']; pass the output of profile_all_edges")
        if "edge_id" in profiles.columns:
            return [stored[int(e)] for e in profiles["edge_id"]]
        return list(stored.values())
    if isinstance(profiles, Mapping):
        return list(profiles.values())
    return list(profiles)


def weighted_layer(profiles: ProfileSource, phi: Callable[[EdgeProfile], Union[float, np.ndarray]]) -> Dict[int, float]:
    """w_e = integral phi(z_e(s)) ds, evaluated as sum_k phi_k * bin_width_k.

    ``phi(profile)`` may return a per-bin vector of length ``n_bins`` or a scalar, in
    which case it is treated as constant along the edge (w_e = phi * arclength).
    """
    out: Dict[int, float] = {}
    for prof in _iter_profiles(profiles):
        val = np.asarray(phi(prof), dtype=np.float64)
        widths = prof.bin_widths_px
        if val.ndim == 0:
            w = float(val) * float(prof.arclength_px)
        else:
            val = val.reshape(-1)
            if val.shape[0] != widths.shape[0]:
                raise ValueError(
                    f"phi returned {val.shape[0]} values for edge {prof.edge_id} with {widths.shape[0]} bins"
                )
            w = float(np.nansum(val * widths))
        out[int(prof.edge_id)] = w
    return out


def legacy_feature_frame(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy ``AJ_*`` feature columns derived from profile scalars, index-aligned.

    Mapping: occupancy -> AJ_occupancy; mean/max intensity -> AJ_mean_intensity /
    AJ_max_intensity; arclength * occupancy -> AJ_skeleton_len (px of occupied
    boundary); n_segments -> AJ_cluster_count; n_segments / arclength ->
    AJ_cluster_density (per px, not the legacy per-1000-marker-px definition);
    width_mean_px -> AJ_thickness_proxy.
    """
    needed = ("arclength_px", "occupancy", "mean_intensity", "max_intensity", "n_segments", "width_mean_px")
    missing = [c for c in needed if c not in profiles_df.columns]
    if missing:
        raise KeyError(f"profiles_df missing columns: {missing}")
    L = profiles_df["arclength_px"].astype(float)
    occ = profiles_df["occupancy"].astype(float)
    nseg = profiles_df["n_segments"].astype(float)
    out = pd.DataFrame(index=profiles_df.index)
    if "edge_id" in profiles_df.columns:
        out["edge_id"] = profiles_df["edge_id"].values
    out["AJ_occupancy"] = occ.values
    out["AJ_mean_intensity"] = profiles_df["mean_intensity"].astype(float).values
    out["AJ_max_intensity"] = profiles_df["max_intensity"].astype(float).values
    out["AJ_skeleton_len"] = (L * occ).values
    out["AJ_cluster_count"] = nseg.astype(int).values
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(L.values > 0, nseg.values / L.values, 0.0)
    out["AJ_cluster_density"] = dens
    out["AJ_thickness_proxy"] = profiles_df["width_mean_px"].astype(float).values
    return out


def _lookup(row: Mapping, key: str, default):
    for name in _LEGACY_KEYS[key]:
        if name in row:
            v = row[name]
            if v is None:
                continue
            return v
    return default


def heuristic_morph_label(row: Mapping) -> str:
    """Legacy heuristic morphology class for one edge row.

    ``row`` (dict or pandas Series) may use the ``AJ_*`` names or the bare profile
    names. The result is one of ``endopigraph.ajmorph.AJMORPH_CLASSES``.
    """
    occ = float(_lookup(row, "occupancy", np.nan))
    th = float(_lookup(row, "thickness_proxy", np.nan))
    sk = _lookup(row, "skeleton_len", 0)
    ncl = _lookup(row, "cluster_count", 0)
    feats = {
        "occupancy": occ,
        "thickness_proxy": th,
        "skeleton_len": int(sk) if np.isfinite(float(sk)) else 0,
        "cluster_count": int(ncl) if np.isfinite(float(ncl)) else 0,
    }
    label = str(infer_ajmorph_label_heuristic(feats))
    return label if label in AJMORPH_CLASSES else "unknown"


def morph_labels(df: pd.DataFrame) -> pd.Series:
    """Heuristic label per row of a profile or legacy feature frame, named ``AJ_morph_label``."""
    if "AJ_occupancy" not in df.columns and "occupancy" in df.columns and "arclength_px" in df.columns:
        feats = legacy_feature_frame(df)
    else:
        feats = df
    labels = [heuristic_morph_label(r) for _, r in feats.iterrows()]
    return pd.Series(labels, index=df.index, name="AJ_morph_label", dtype=object)
