"""Embedded half-edge cell complex for planar endothelial sheets.

Conventions (docs/COMPLEX.md):

- Coordinates are image coordinates (row, col) with pixel (r, c) centered at (r, c)
  and occupying [r-0.5, r+0.5) x [c-0.5, c+0.5). Crack corners therefore sit on
  half-integers.
- Half-edge ``2*e`` runs tail -> head of edge ``e``; ``2*e + 1`` is its twin.
  ``face(h)`` is the face on the LEFT of ``h`` when travelling along it on screen
  (row down, col right). Cell faces are traced counter-clockwise on screen.
- Faces are cells (label > 0), gaps (enclosed background) or the single outer face.
- Only representation validity is hard. Everything biological is a soft prior.
"""

from .halfedge import FaceKind, HalfEdgeComplex, VertexKind
from .extract import extract_complex, rebuild_topology
from .incidence import ValidationReport, boundary_matrices, validate
from .invariants import (
    defect_law_residual,
    euler_characteristic,
    topological_charge,
    weaire_sum_rule_residual,
)
from .dual import clique_vertex_report, to_multigraph, to_simple_graph
from .events import (
    EventPreconditionError,
    EventResult,
    EventRewriteError,
    contact_birth,
    contact_death,
    divide,
    extrude,
    nucleate_gap,
    reseal,
    rupture,
    t1_exchange,
)

__all__ = [
    "EventPreconditionError",
    "EventResult",
    "EventRewriteError",
    "contact_birth",
    "contact_death",
    "divide",
    "extrude",
    "nucleate_gap",
    "reseal",
    "rupture",
    "t1_exchange",
    "FaceKind",
    "HalfEdgeComplex",
    "VertexKind",
    "ValidationReport",
    "boundary_matrices",
    "clique_vertex_report",
    "defect_law_residual",
    "euler_characteristic",
    "extract_complex",
    "rebuild_topology",
    "to_multigraph",
    "to_simple_graph",
    "topological_charge",
    "validate",
    "weaire_sum_rule_residual",
]
