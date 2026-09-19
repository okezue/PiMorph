"""EndoPiGraph-AJmorph v1.

Core idea:
- segment endothelial cells (instance labels)
- infer cell-cell contacts (interfaces)
- quantify junction-marker signal on each interface (e.g. VE-cad for AJ)
- build a typed cell-contact graph ("pi-graph" in the manuscript sense)

This package aims to be:
- explicit about assumptions
- easy to run on public microscopy data (e.g. BioImage Archive accessions)
- easy to extend with better segmentation / classification models

Blur Robustness
---------------
Some metrics are sensitive to image blur. The `blur_robust` module provides:
- `estimate_blur_score()`: Detect blur level using Laplacian variance
- `detect_blur()`: Check if image is blurry (threshold-based)
- `correct_blur()`: Apply unsharp masking correction
- `blur_robust_classifier()`: Classification using only stable metrics
- `compute_blur_robust_features()`: Compute only blur-stable features
- `compute_adaptive_features()`: Auto-detect and correct blur

For detailed guidance, see the EXPLAINER.md documentation.

Metric Stability Reference
--------------------------
Stable (|Cohen's d| < 0.3 under 1-2px blur):
  - mean_intensity, median_intensity, occupancy

Marginal (|Cohen's d| 0.3-0.5):
  - skeleton_len, total_area

Unstable (|Cohen's d| > 0.5):
  - cluster_count, cluster_density, thickness_proxy
"""

from importlib.metadata import version as _version

__all__ = [
    "__version__",
    "interface_marker_features",
    "heuristic_ajmorph_class",
    "compute_threshold",
    "estimate_blur_score",
    "detect_blur",
    "correct_blur",
    "blur_robust_classifier",
    "compute_blur_robust_features",
    "compute_adaptive_features",
    "cluster_junctions_gmm",
    "evaluate_cluster_stability",
]

try:
    __version__ = _version("endopigraph-ajmorph")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Core AJ morphology functions
from .ajmorph import (
    interface_marker_features,
    heuristic_ajmorph_class,
    compute_threshold,
)

# Blur-robust functions
from .blur_robust import (
    estimate_blur_score,
    detect_blur,
    correct_blur,
    blur_robust_classifier,
    compute_blur_robust_features,
    compute_adaptive_features,
)

# Data-driven unsupervised clustering
from .ajmorph_unsupervised import (
    cluster_junctions_gmm,
    evaluate_cluster_stability,
)
