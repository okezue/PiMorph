"""3-D extension of the PiMorph cell complex (blueprint open problem 8).

Conventions (see ``extract3d`` for the algorithm):

- Coordinates are voxel coordinates (z, row, col); voxel (z, y, x) is centred at
  (z, y, x). Doubled-grid coordinate d maps to d / 2 - 1.5.
- 3-cells are cells (label > 0, 6-connected), gaps (enclosed background) or the
  single outer cell. 2-cells are interfaces, 1-cells triple lines, 0-cells
  quadruple points (any corner where the skeleton branches or ends).
- ``interface_cells[f] = (a, b)`` with a < b; the interface normal points from a
  into b, so B3[f, a] = +1 and B3[f, b] = -1. 1-cells are oriented tail -> head
  and B2[e, f] is the signed number of times the oriented boundary of f runs along
  e (right-hand rule with the normal). B1 @ B2 == 0 and B2 @ B3 == 0 identically.
- A closed interface without triple lines carries one artificial 1-cell (a voxel
  edge of the surface) with two artificial 0-cells; a closed triple line without a
  0-cell carries one artificial degree-2 vertex.
- ``validate3d`` checks the two chain identities and an Euler residual that is 0
  for cellulations by balls (cavities and planar interfaces with holes allowed);
  handles in cells are flagged, and ``per_cell_euler(..., exact=True)`` locates them.
"""

from .extract3d import CellComplex3D, CellKind3D, LineKind3D, VertexKind3D, extract_complex3d
from .incidence3d import (
    Validation3DReport,
    boundary_matrices3d,
    euler_expected3d,
    euler_residual3d,
    occupied_set_euler,
    skeleton_components3d,
    validate3d,
)
from .invariants3d import (
    boundary_surface_euler,
    contact_multiplicity,
    euler_characteristic3d,
    neighbor_counts,
    per_cell_euler,
)
from .surface import (
    SurfaceComplex,
    SurfaceValidationReport,
    surface_boundary_matrices,
    surface_complex_from_labels,
    validate_surface,
)

__all__ = [
    "CellComplex3D",
    "CellKind3D",
    "LineKind3D",
    "SurfaceComplex",
    "SurfaceValidationReport",
    "Validation3DReport",
    "VertexKind3D",
    "boundary_matrices3d",
    "boundary_surface_euler",
    "contact_multiplicity",
    "euler_characteristic3d",
    "euler_expected3d",
    "euler_residual3d",
    "extract_complex3d",
    "neighbor_counts",
    "occupied_set_euler",
    "per_cell_euler",
    "skeleton_components3d",
    "surface_boundary_matrices",
    "surface_complex_from_labels",
    "validate3d",
    "validate_surface",
]
