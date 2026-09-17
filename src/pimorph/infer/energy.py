"""Energy of a reconstruction hypothesis.

E = E_image + l_curve E_curve + l_seed E_seed + l_vertex E_vertex + l_prior E_prior

Only E_image and E_curve refer to the data. E_seed, E_vertex and E_prior encode
biological typicality (about one nucleus per cell, mostly trivalent vertices,
plausible areas and aspect ratios) and are SOFT: they can be down-weighted or zeroed
for mitosis, binucleation, wounds, sprouts or pathology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.measure import regionprops_table

from ..complex.geometry import polyline_curvature
from ..complex.halfedge import HalfEdgeComplex
from .proposals import ProposalMaps
from .renderer import RenderModel, render_log_likelihood


@dataclass
class EnergyWeights:
    image: float = 1.0
    curve: float = 1.0
    curvature_beta: float = 0.02
    seed: float = 0.5
    vertex: float = 0.2
    prior: float = 0.2
    log_area_z_cut: float = 2.5
    aspect_cut: float = 6.0


def curve_energy(cx: HalfEdgeComplex, boundary: np.ndarray, beta: float, eps: float = 1e-3) -> Tuple[float, Dict]:
    """Arclength-weighted mean of -log b along cell-cell edges plus beta * mean curvature^2."""
    H, W = boundary.shape
    tot_len = 0.0
    tot_nll = 0.0
    tot_curv = 0.0
    nb = -np.log(np.clip(boundary, eps, 1.0))
    for e in cx.cell_cell_edges():
        p = cx.edge_geometry(int(e))
        if p.shape[0] < 2:
            continue
        seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
        mid = 0.5 * (p[1:] + p[:-1])
        pr = np.clip(np.round(mid[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(mid[:, 1]).astype(int), 0, W - 1)
        tot_nll += float((nb[pr, pc] * seg).sum())
        k = polyline_curvature(p)
        tot_curv += float((k[1:-1] ** 2 * 0.5 * (seg[1:] + seg[:-1])).sum()) if k.size > 2 else 0.0
        tot_len += float(seg.sum())
    if tot_len == 0:
        return 0.0, {"mean_neg_log_b": 0.0, "mean_curv2": 0.0, "total_arclength": 0.0}
    mean_nll = tot_nll / tot_len
    mean_curv = tot_curv / tot_len
    return mean_nll + beta * mean_curv, {
        "mean_neg_log_b": mean_nll,
        "mean_curv2": mean_curv,
        "total_arclength": tot_len,
    }


def seed_energy(
    labels: np.ndarray, seed_points: np.ndarray, seed_scores: Optional[np.ndarray] = None
) -> Tuple[float, Dict]:
    """Per cell: penalty 1 for no seed inside, (k - 1) for k >= 2 seeds. Mean over cells."""
    n_cells = int(labels.max())
    if n_cells == 0:
        return 0.0, {"cells_without_seed": 0, "cells_multi_seed": 0}
    H, W = labels.shape
    pts = np.round(seed_points).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
    owner = labels[pts[:, 0], pts[:, 1]]
    present = np.unique(labels[labels > 0])
    counts = np.bincount(owner[owner > 0], minlength=n_cells + 1)
    k = counts[present]
    zero = int(np.sum(k == 0))
    multi = int(np.sum(np.clip(k - 1, 0, None)))
    return (zero + multi) / len(present), {"cells_without_seed": zero, "cells_multi_seed": int(np.sum(k >= 2))}


def vertex_energy(cx: HalfEdgeComplex) -> Tuple[float, Dict]:
    """Fraction of regular vertices whose degree differs from 3."""
    if cx.n_vertices == 0:
        return 0.0, {"frac_non_trivalent": 0.0}
    degs = cx.vertex_degrees()
    reg = degs >= 3
    if reg.sum() == 0:
        return 0.0, {"frac_non_trivalent": 0.0}
    frac = float(np.mean(degs[reg] != 3))
    return frac, {"frac_non_trivalent": frac}


def shape_prior_energy(labels: np.ndarray, log_area_z_cut: float, aspect_cut: float) -> Tuple[float, Dict]:
    """Fraction of cells that are log-area outliers or extremely elongated."""
    if labels.max() == 0:
        return 0.0, {"area_outliers": 0, "aspect_outliers": 0}
    props = regionprops_table(labels, properties=("label", "area", "axis_major_length", "axis_minor_length"))
    area = np.asarray(props["area"], dtype=np.float64)
    la = np.log(np.maximum(area, 1.0))
    z = (la - np.median(la)) / max(1.4826 * np.median(np.abs(la - np.median(la))), 1e-6)
    area_out = int(np.sum(np.abs(z) > log_area_z_cut))
    minor = np.maximum(np.asarray(props["axis_minor_length"]), 1e-6)
    aspect = np.asarray(props["axis_major_length"]) / minor
    asp_out = int(np.sum(aspect > aspect_cut))
    n = len(area)
    return (area_out + asp_out) / n, {"area_outliers": area_out, "aspect_outliers": asp_out}


def complex_energy(
    labels: np.ndarray,
    cx: HalfEdgeComplex,
    maps: ProposalMaps,
    image: Optional[np.ndarray] = None,
    weights: Optional[EnergyWeights] = None,
    render_model: Optional[RenderModel] = None,
) -> Tuple[float, Dict]:
    """Total energy and a breakdown. ``image`` is the geometry channel for the
    renderer; when None the image term is skipped."""
    w = weights or EnergyWeights()
    parts: Dict[str, float] = {}
    info: Dict = {}
    total = 0.0
    if image is not None and w.image > 0:
        if render_model is None:
            width = float(maps.meta.get("ridge_width_px", 1.0))
            render_model = RenderModel(line_width_px=width, psf_sigma_px=max(0.5, 0.3 * width))
        ll, rinfo = render_log_likelihood(image, labels, cx, render_model)
        parts["image"] = -ll
        info["render"] = {k: v for k, v in rinfo.items() if k in ("gain", "read_noise", "residual_rms")}
        total += w.image * parts["image"]
    ec, cinfo = curve_energy(cx, maps.boundary, w.curvature_beta)
    parts["curve"] = ec
    info["curve"] = cinfo
    total += w.curve * ec
    es, sinfo = seed_energy(labels, maps.seed_points)
    parts["seed"] = es
    info["seed"] = sinfo
    total += w.seed * es
    ev, vinfo = vertex_energy(cx)
    parts["vertex"] = ev
    info["vertex"] = vinfo
    total += w.vertex * ev
    ep, pinfo = shape_prior_energy(labels, w.log_area_z_cut, w.aspect_cut)
    parts["prior"] = ep
    info["prior"] = pinfo
    total += w.prior * ep
    info["parts"] = parts
    info["weights"] = w.__dict__
    return float(total), info
