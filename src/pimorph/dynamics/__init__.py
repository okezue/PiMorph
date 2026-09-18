"""Temporal layer: cell linking, admissible-event detection and exact QC identities.

Dynamics on the cell complex are continuous motion within one combinatorial type
plus discrete admissible rewrites (blueprint section 7). This package links cells
across frames, extracts one complex per frame with track ids as face labels,
classifies topology changes into the event vocabulary of
``pimorph.complex.events`` and checks the exact (dV, dE, dF) bookkeeping.
"""

from .events_detect import (
    EXPECTED_DELTA_KINDS,
    UNEXPLAINED_KINDS,
    Event,
    TopologySnapshot,
    admissibility_check,
    detect_events,
    detect_events_from_snapshots,
    event_summary,
    expected_delta_vef,
    snapshot,
)
from .metrics import division_detection_metrics, divisions_from_lineage, t1_metrics, tracking_metrics
from .tracking import (
    FrameLink,
    Tracks,
    clean_labels,
    complex_to_labels,
    complexes_over_time,
    fill_small_gaps,
    link_frames,
    raster_face_labels,
    track,
    tracks_from_labels,
)

__all__ = [
    "EXPECTED_DELTA_KINDS",
    "Event",
    "FrameLink",
    "TopologySnapshot",
    "Tracks",
    "UNEXPLAINED_KINDS",
    "admissibility_check",
    "clean_labels",
    "complex_to_labels",
    "complexes_over_time",
    "detect_events",
    "detect_events_from_snapshots",
    "division_detection_metrics",
    "divisions_from_lineage",
    "event_summary",
    "expected_delta_vef",
    "fill_small_gaps",
    "link_frames",
    "raster_face_labels",
    "snapshot",
    "t1_metrics",
    "track",
    "tracking_metrics",
    "tracks_from_labels",
]
