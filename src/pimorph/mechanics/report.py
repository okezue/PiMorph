"""Tables and summaries of force-inference results. All tensions are relative."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from pimorph.complex.geometry import edge_arclengths, face_kind_name
from pimorph.complex.halfedge import HalfEdgeComplex

from .force_inference import ForceInferenceResult


def edge_tension_table(cx: HalfEdgeComplex, result: ForceInferenceResult) -> pd.DataFrame:
    """One row per edge: faces, arclength, inferred tension and tension / median."""
    arc = edge_arclengths(cx, smoothed=True) if cx.n_edges else np.zeros(0)
    t = result.tensions
    med = float(np.nanmedian(t)) if np.any(np.isfinite(t)) else float("nan")
    kinds = cx.face_kind
    rows = []
    for e in range(cx.n_edges):
        fl, fr = int(cx.edge_faces[e, 0]), int(cx.edge_faces[e, 1])
        rows.append(
            {
                "edge_id": e,
                "face_left": fl,
                "face_right": fr,
                "kind_left": face_kind_name(kinds[fl]),
                "kind_right": face_kind_name(kinds[fr]),
                "label_left": int(cx.face_label[fl]),
                "label_right": int(cx.face_label[fr]),
                "arclength": float(arc[e]),
                "tension": float(t[e]),
                "relative_tension": float(t[e] / med) if np.isfinite(med) and med != 0 else float("nan"),
                "identifiable": bool(result.tension_identifiable[e]),
            }
        )
    cols = [
        "edge_id",
        "face_left",
        "face_right",
        "kind_left",
        "kind_right",
        "label_left",
        "label_right",
        "arclength",
        "tension",
        "relative_tension",
        "identifiable",
    ]
    return pd.DataFrame(rows, columns=cols)


def per_field_summary(result: ForceInferenceResult, residual_fraction: float = 0.05) -> Dict[str, Any]:
    """Scale-free summary: tension CV, residual RMS, condition number, balanced-vertex fraction."""
    t = result.tensions[np.isfinite(result.tensions)]
    p = result.pressures[np.isfinite(result.pressures)]
    res = result.vertex_residuals[np.isfinite(result.vertex_residuals)]
    typical = float(np.median(t)) if t.size else float("nan")
    frac = float(np.mean(res < residual_fraction * typical)) if res.size and np.isfinite(typical) else float("nan")
    return {
        "n_tensions": int(t.size),
        "n_pressures": int(p.size),
        "n_interior_vertices": int(res.size),
        "tension_mean": float(t.mean()) if t.size else float("nan"),
        "tension_cv": float(t.std() / t.mean()) if t.size and t.mean() != 0 else float("nan"),
        "tension_min": float(t.min()) if t.size else float("nan"),
        "tension_max": float(t.max()) if t.size else float("nan"),
        "pressure_std": float(p.std()) if p.size else float("nan"),
        "residual_rms": float(np.sqrt(np.mean(res**2))) if res.size else float("nan"),
        "residual_max": float(res.max()) if res.size else float("nan"),
        "fraction_vertices_balanced": frac,
        "condition_number": float(result.condition_number),
        "n_equations": int(result.n_equations),
        "n_unknowns": int(result.n_unknowns),
        "rank": int(result.rank),
        "pressure_gauge": result.settings.get("pressure_gauge", "none"),
        "curvature_pressure": bool(result.settings.get("curvature_pressure", False)),
    }
