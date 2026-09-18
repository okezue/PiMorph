"""Mechanochemical phase: vertex-model forward dynamics and force inference.

Within one topology the vertex positions follow overdamped mechanics of a
vertex-model energy (``vertex_model``); the inverse problem (``force_inference``)
recovers edge tensions and cell pressures from the equilibrium geometry alone.

Identifiability, stated once here and enforced in the solver:

- tensions are identifiable only up to a global scale (the force balance is
  homogeneous), so every reported tension is RELATIVE (gauge: mean tension = 1);
- pressures enter only through edge curvature (Laplace law). Without curvature
  information they are unidentifiable; with it they are identifiable up to a
  constant (gauge: mean pressure = 0), and the scale is shared with the tensions.
"""

from .force_inference import (
    ForceInferenceResult,
    edge_curvatures,
    identifiability_report,
    infer_tensions,
    inferred_stress_tensor,
)
from .report import edge_tension_table, per_field_summary
from .vertex_model import (
    PolygonModel,
    VertexModelParams,
    assign_tensions,
    cell_pressures,
    effective_tensions,
    energy,
    jitter_vertices,
    relax,
    render_pressure_arcs,
    straighten_edges,
    vertex_forces,
)

__all__ = [
    "ForceInferenceResult",
    "PolygonModel",
    "VertexModelParams",
    "assign_tensions",
    "cell_pressures",
    "edge_curvatures",
    "edge_tension_table",
    "effective_tensions",
    "energy",
    "identifiability_report",
    "infer_tensions",
    "inferred_stress_tensor",
    "jitter_vertices",
    "per_field_summary",
    "relax",
    "render_pressure_arcs",
    "straighten_edges",
    "vertex_forces",
]
