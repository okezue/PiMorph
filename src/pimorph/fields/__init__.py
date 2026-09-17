"""Molecular fields on the complex: interface-centered strips, profiles, functionals.

Every edge e carries a curve gamma_e(s) and a left normal n_e(s) (into
``edge_faces[e, 0]``). A marker channel is decoded along the edge as the oriented
strip P_e(s, r) = I(gamma_e(s) + r * n_e(s)), reduced to arclength profiles and then
to scalar edge functionals. Legacy adapters reproduce the EndoPiGraph ``AJ_*``
feature names and heuristic morphology strings.
"""

from .functionals import (
    LEGACY_FEATURE_COLUMNS,
    heuristic_morph_label,
    legacy_feature_frame,
    morph_labels,
    weighted_layer,
)
from .profile import (
    SCALAR_FIELDS,
    EdgeProfile,
    edge_profile,
    profile_all_edges,
    profile_edges,
    profile_from_strip,
    profiles_to_frame,
    strip_threshold,
)
from .strip import Strip, edge_strip, lateral_offsets, resample_polyline, sample_strip

__all__ = [
    "LEGACY_FEATURE_COLUMNS",
    "SCALAR_FIELDS",
    "EdgeProfile",
    "Strip",
    "edge_profile",
    "edge_strip",
    "heuristic_morph_label",
    "lateral_offsets",
    "legacy_feature_frame",
    "morph_labels",
    "profile_all_edges",
    "profile_edges",
    "profile_from_strip",
    "profiles_to_frame",
    "resample_polyline",
    "sample_strip",
    "strip_threshold",
    "weighted_layer",
]
