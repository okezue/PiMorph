"""Barrier report: junction coverage -> predicted paracellular conductance.

Everything reported here is a prediction target. No TEER, tracer or permeability
measurement is available for the imaged monolayers, so the numbers are what the
resistor model of ``pimorph.function.transport`` implies for the reconstructed
junction coverage, and nothing more. The report says so in its ``validated`` and
``note`` fields.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..complex.halfedge import HalfEdgeComplex
from ..fields.multichannel import MultiProfile
from .transport import (
    boundary_cells,
    conductance_breakdown,
    edge_conductance_vector,
    in_plane_effective_conductance,
    interface_conductance,
    permeability_index,
    tissue_area,
)

VALIDATION_NOTE = "prediction target; no TEER or tracer data available"


def barrier_report(
    cx: HalfEdgeComplex,
    multi_profile: MultiProfile,
    weights: Optional[Dict[str, float]] = None,
    baseline: float = 0.05,
    gap_conductance: float = 10.0,
    g_trans: float = 0.0,
    top_k: int = 10,
    source_cells: Optional[Sequence[int]] = None,
    sink_cells: Optional[Sequence[int]] = None,
    occupancy_template: str = "{c}_coverage",
) -> Dict[str, object]:
    """Predicted barrier numbers for one complex (untested prediction).

    ``weights`` maps channel names of ``multi_profile`` to leak weights (default TJ 1.0,
    AJ 0.5; channels absent from the profile are ignored). Arclengths are converted to
    micrometres when the complex is calibrated. Returns G_eff and its breakdown, the
    permeability index (G_eff per tissue area), the gap contribution fraction, the
    ``top_k`` leakiest interfaces, an optional in-plane conductance between
    ``source_cells`` and ``sink_cells`` (default: none), and ``validated=False``.
    """
    weights = {"TJ": 1.0, "AJ": 0.5} if weights is None else dict(weights)
    use = {c: w for c, w in weights.items() if c in multi_profile.channels}
    if not use:
        raise KeyError(f"none of the weighted channels {sorted(weights)} are in the profile {multi_profile.channels}")
    per_edge = multi_profile.per_edge
    px = cx.pixel_size_um
    length_scale = float(px) if px is not None else 1.0
    g_rows = interface_conductance(
        per_edge, use, baseline=baseline, occupancy_template=occupancy_template, length_scale=length_scale
    )
    g_edge = edge_conductance_vector(cx, per_edge["edge_id"].to_numpy(dtype=np.int64), g_rows, fill=baseline)
    brk = conductance_breakdown(cx, g_edge, g_gap=gap_conductance, g_trans=g_trans)
    area, area_unit = tissue_area(cx)
    pidx = permeability_index(cx, g_edge, g_gap=gap_conductance, g_trans=g_trans, area=area)

    order = np.argsort(-g_rows)[: max(0, int(top_k))]
    top = []
    for i in order:
        row = per_edge.iloc[int(i)]
        entry = {
            "edge_id": int(row["edge_id"]),
            "label_left": int(row["label_left"]),
            "label_right": int(row["label_right"]),
            "arclength_px": float(row["arclength_px"]),
            "g": float(g_rows[i]),
        }
        for c in use:
            col = occupancy_template.format(c=c)
            if col in per_edge.columns:
                entry[col] = float(row[col])
        top.append(entry)

    in_plane = None
    if source_cells is not None and sink_cells is not None:
        in_plane = float(
            in_plane_effective_conductance(cx, g_edge, source_cells, sink_cells, gap_conductance=gap_conductance)
        )

    return {
        "validated": False,
        "note": VALIDATION_NOTE,
        "model": "parallel resistor network (through-plane leaks) with explicit gaps as short circuits",
        "weights": use,
        "baseline": float(baseline),
        "gap_conductance": float(gap_conductance),
        "g_trans": float(g_trans),
        "length_unit": "um" if px is not None else "px",
        "n_cells": int(cx.cell_faces.size),
        "n_gaps": int(cx.gap_faces.size),
        "n_cell_cell_edges": int(cx.cell_cell_edges().size),
        "n_profiled_edges": int(len(per_edge)),
        "G_eff": brk["G_eff"],
        "G_junction": brk["G_junction"],
        "G_gap": brk["G_gap"],
        "G_trans": brk["G_trans"],
        "gap_fraction": brk["gap_fraction"],
        "junction_fraction": brk["junction_fraction"],
        "tissue_area": float(area),
        "area_unit": area_unit,
        "permeability_index": float(pidx),
        "mean_g_edge": float(g_rows.mean()) if g_rows.size else float("nan"),
        "top_leaking_edges": top,
        "in_plane_conductance": in_plane,
        "n_boundary_cells": int(boundary_cells(cx).size),
    }
