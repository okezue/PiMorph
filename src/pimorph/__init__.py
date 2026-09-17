"""PiMorph: uncertainty-aware endothelial cell-complex inference.

Layers (see docs/ARCHITECTURE.md):

1. pixels        multichannel microscopy with physical calibration
2. geometry      an embedded half-edge cell complex K with faces (cells, gaps, outer),
                 connected interface components (edges) and multicellular vertices
3. state         vector-valued molecular fields on edges, half-edges, vertices, cells
4. dynamics      admissible topology rewrites (T1, division, extrusion, rupture, reseal)
5. function      prediction targets that other measurements must validate

Only representation validity is hard (B1 @ B2 == 0, disjoint interiors, valid
links). Biological regularities are soft priors in `pimorph.infer.energy`.
"""

from importlib.metadata import version as _version

try:
    __version__ = _version("endopigraph-ajmorph")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
