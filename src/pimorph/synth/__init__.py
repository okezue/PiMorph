"""Synthetic tissues with exact targets for pretraining the neural proposal model.

- ``tissue``: Lloyd-relaxed Voronoi sheets with flow elongation, boundary jitter,
  tricellular/bicellular gaps and approximately one nucleus per cell.
- ``render``: junction (VE-cadherin-like, with broken segments), membrane and nuclei
  channels with PSF, Poisson + read noise and flat-field/bleaching gradients.
- ``targets``: boundary, signed distance, seed/vertex heatmaps, gap and outer masks,
  and ``make_dataset`` for writing ``.npz`` tiles plus a manifest.
"""

from .render import RenderParams, junction_intensity_for_snr, render_channels, resolve_render_params
from .targets import (
    CHANNEL_KEYS,
    TARGET_KEYS,
    default_params_sampler,
    default_render_sampler,
    load_tile,
    make_dataset,
    make_targets,
)
from .tissue import GAP_KINDS, SynthTissue, SynthTissueParams, generate_tissue

__all__ = [
    "CHANNEL_KEYS",
    "GAP_KINDS",
    "RenderParams",
    "SynthTissue",
    "SynthTissueParams",
    "TARGET_KEYS",
    "default_params_sampler",
    "default_render_sampler",
    "generate_tissue",
    "junction_intensity_for_snr",
    "load_tile",
    "make_dataset",
    "make_targets",
    "render_channels",
    "resolve_render_params",
]
