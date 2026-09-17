"""Forward image model: does the proposed complex explain the observed channel?

mu_c(x) = b_c + alpha_c [PSF * rho_c](x), with rho_c a stratified density made of
per-face interior levels, per-edge line densities and per-vertex spots. Densities are
fitted per structure from the data given the complex (medians), so the renderer asks
only whether the GEOMETRY of the complex is consistent with where the signal is. A
missing true boundary leaves a bright line unexplained; a spurious boundary through
dark interior fits a near-zero line density and is nearly neutral here (the curve
energy penalizes it instead).

Noise: heteroscedastic Gaussian approximation to Poisson + read noise,
var = gain * mu + sigma_read^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

from ..complex.halfedge import FaceKind, HalfEdgeComplex


@dataclass
class RenderModel:
    psf_sigma_px: float = 1.2
    line_width_px: float = 1.0
    gain: Optional[float] = None  # estimated when None
    read_noise: Optional[float] = None
    background_percentile: float = 5.0
    fitted: Dict = field(default_factory=dict)


def rasterize_edges(
    cx: HalfEdgeComplex, shape: Tuple[int, int], values: np.ndarray, width_px: float = 1.0
) -> np.ndarray:
    """Draw every edge's smoothed polyline into a float image with the given per-edge value."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)
    for e in range(cx.n_edges):
        p = cx.edge_geometry(e)
        v = float(values[e])
        if v == 0.0:
            continue
        pr = np.clip(np.round(p[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(p[:, 1]).astype(int), 0, W - 1)
        for k in range(len(pr) - 1):
            rr, cc = draw_line(pr[k], pc[k], pr[k + 1], pc[k + 1])
            img[rr, cc] = np.maximum(img[rr, cc], v)
    if width_px > 1.0:
        img = ndi.grey_dilation(img, size=(int(round(width_px)), int(round(width_px))))
    return img


def face_id_image(labels: np.ndarray, cx: HalfEdgeComplex) -> np.ndarray:
    """Face id per pixel (outer/gap pixels get their face ids as well)."""
    from skimage.measure import label as cc_label

    H, W = labels.shape
    fid = np.full((H, W), cx.outer_face, dtype=np.int64)
    cells = cc_label(labels, connectivity=1, background=0)
    m = cells > 0
    fid[m] = cells[m] - 1  # extract() numbers cell faces in cc_label order
    if cx.gap_faces.size:
        bg = cc_label(labels == 0, connectivity=1)
        border = np.zeros(bg.max() + 1, dtype=bool)
        for edge in (bg[0], bg[-1], bg[:, 0], bg[:, -1]):
            border[np.unique(edge)] = True
        border[0] = False
        enclosed = np.flatnonzero(~border[1:]) + 1
        gap_index = np.full(bg.max() + 1, -1, dtype=np.int64)
        gap_index[enclosed] = np.arange(enclosed.size) + int(cx.cell_faces.size)
        mm = (labels == 0) & (gap_index[bg] >= 0)
        fid[mm] = gap_index[bg[mm]]
    return fid


def fit_densities(
    image: np.ndarray, labels: np.ndarray, cx: HalfEdgeComplex, model: RenderModel
) -> Dict[str, np.ndarray]:
    """Per-face interior level (median), per-edge line excess (median along the line
    minus the mean of the two flanking face levels, clipped at 0), per-vertex spot."""
    H, W = image.shape
    fid = face_id_image(labels, cx)
    n_faces = cx.n_faces
    face_level = np.zeros(n_faces, dtype=np.float32)
    flat_f = fid.ravel()
    flat_i = image.ravel()
    order = np.argsort(flat_f, kind="stable")
    sf = flat_f[order]
    si = flat_i[order]
    bounds = np.searchsorted(sf, np.arange(n_faces + 1))
    for f in range(n_faces):
        seg = si[bounds[f] : bounds[f + 1]]
        face_level[f] = np.median(seg) if seg.size else 0.0

    edge_excess = np.zeros(cx.n_edges, dtype=np.float32)
    for e in range(cx.n_edges):
        p = cx.edge_geometry(e)
        pr = np.clip(np.round(p[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(p[:, 1]).astype(int), 0, W - 1)
        vals = image[pr, pc]
        a, b = cx.edge_faces[e]
        base = 0.5 * (face_level[a] + face_level[b])
        edge_excess[e] = max(float(np.median(vals)) - base, 0.0) if vals.size else 0.0

    vertex_spot = np.zeros(cx.n_vertices, dtype=np.float32)
    if cx.n_vertices:
        vr = np.clip(np.round(cx.vertex_xy[:, 0]).astype(int), 0, H - 1)
        vc = np.clip(np.round(cx.vertex_xy[:, 1]).astype(int), 0, W - 1)
        # excess at the vertex beyond the mean of incident edge lines
        for v in range(cx.n_vertices):
            inc = [h >> 1 for h in cx.vertex_out_half_edges(v)]
            base = float(np.mean(edge_excess[inc])) if inc else 0.0
            faces = cx.vertex_faces(v)
            base += float(np.mean(face_level[faces])) if faces else 0.0
            vertex_spot[v] = max(float(image[vr[v], vc[v]]) - base, 0.0)
    return {"face_level": face_level, "edge_excess": edge_excess, "vertex_spot": vertex_spot, "fid": fid}


def render(
    image_shape: Tuple[int, int], labels: np.ndarray, cx: HalfEdgeComplex, dens: Dict, model: RenderModel
) -> np.ndarray:
    H, W = image_shape
    fid = dens["fid"]
    mu = dens["face_level"][fid].astype(np.float32)
    lines = rasterize_edges(cx, (H, W), dens["edge_excess"], width_px=model.line_width_px)
    mu = mu + lines
    if cx.n_vertices and np.any(dens["vertex_spot"] > 0):
        vr = np.clip(np.round(cx.vertex_xy[:, 0]).astype(int), 0, H - 1)
        vc = np.clip(np.round(cx.vertex_xy[:, 1]).astype(int), 0, W - 1)
        spots = np.zeros((H, W), dtype=np.float32)
        np.maximum.at(spots, (vr, vc), dens["vertex_spot"])
        mu = mu + spots
    if model.psf_sigma_px > 0:
        mu = ndi.gaussian_filter(mu, sigma=model.psf_sigma_px)
    return mu


def estimate_noise_model(image: np.ndarray, mu: np.ndarray, n_bins: int = 20) -> Tuple[float, float]:
    """Fit var = gain * mu + read^2 by binned residual variance against mu."""
    r = (image - mu).ravel()
    m = mu.ravel()
    lo, hi = np.percentile(m, [1, 99])
    if hi <= lo:
        v = float(np.var(r))
        return 0.0, float(np.sqrt(max(v, 1e-6)))
    bins = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(m, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() > 50:
            xs.append(float(m[sel].mean()))
            ys.append(float(np.var(r[sel])))
    if len(xs) < 3:
        v = float(np.var(r))
        return 0.0, float(np.sqrt(max(v, 1e-6)))
    A = np.stack([np.asarray(xs), np.ones(len(xs))], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    gain = max(float(coef[0]), 0.0)
    read2 = max(float(coef[1]), 1e-6)
    return gain, float(np.sqrt(read2))


def render_log_likelihood(
    image: np.ndarray,
    labels: np.ndarray,
    cx: HalfEdgeComplex,
    model: Optional[RenderModel] = None,
    per_pixel: bool = False,
) -> Tuple[float, Dict]:
    """Heteroscedastic Gaussian log-likelihood of ``image`` under the rendered complex.

    Returns (mean log-likelihood per pixel, info dict with rendered mu, fitted
    densities and the noise model). Comparing hypotheses on the same image only needs
    the mean per pixel, which keeps values comparable across fields of different size.
    """
    model = model or RenderModel()
    image = np.asarray(image, dtype=np.float32)
    dens = fit_densities(image, labels, cx, model)
    mu = render(image.shape, labels, cx, dens, model)
    gain, read = (model.gain, model.read_noise)
    if gain is None or read is None:
        gain, read = estimate_noise_model(image, mu)
    var = np.maximum(gain * np.maximum(mu, 0.0) + read**2, 1e-6)
    ll_map = -0.5 * ((image - mu) ** 2 / var + np.log(var))
    ll = float(ll_map.mean())
    info = {
        "mu": mu,
        "densities": dens,
        "gain": float(gain),
        "read_noise": float(read),
        "residual_rms": float(np.sqrt(np.mean((image - mu) ** 2))),
    }
    if per_pixel:
        info["ll_map"] = ll_map
    return ll, info


def boundary_interior_ratio(image: np.ndarray, labels: np.ndarray, cx: HalfEdgeComplex) -> float:
    """Mean signal on cell-cell interface pixels over mean signal in cell interiors.
    Blueprint self-consistency check (2.96 structured vs 1.48 Voronoi in the audit)."""
    H, W = image.shape
    on = np.zeros((H, W), dtype=bool)
    for e in cx.cell_cell_edges():
        p = cx.edge_geometry(int(e))
        pr = np.clip(np.round(p[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(p[:, 1]).astype(int), 0, W - 1)
        on[pr, pc] = True
    on = ndi.binary_dilation(on, iterations=1)
    interior = (labels > 0) & ~ndi.binary_dilation(on, iterations=2)
    if on.sum() == 0 or interior.sum() == 0:
        return float("nan")
    return float(image[on].mean() / max(image[interior].mean(), 1e-9))


def kind_name(k: int) -> str:
    return FaceKind(int(k)).name.lower()
