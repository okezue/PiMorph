"""Posterior over reconstructions: a weighted ensemble of legal complexes.

Hypotheses share a master seed list, so a cell is identified by its seed index
across hypotheses and a contact by the unordered pair of seed indices. This is what
makes "the contact between cells 12 and 40 exists with probability 0.92" well
defined without pixel-level matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..complex.halfedge import FaceKind, HalfEdgeComplex
from .decoder import ConstrainedDecoder, DecodeResult, DecoderParams
from .energy import EnergyWeights, complex_energy
from .proposals import ProposalMaps
from .renderer import RenderModel


@dataclass
class Hypothesis:
    labels: np.ndarray
    cx: HalfEdgeComplex
    energy: float
    breakdown: Dict
    params: DecoderParams
    tag: str = ""

    def contacts(self) -> Dict[Tuple[int, int], int]:
        """Unordered seed-label pairs -> number of contact components."""
        out: Dict[Tuple[int, int], int] = {}
        for e in self.cx.cell_cell_edges():
            a, b = (int(self.cx.face_label[f]) for f in self.cx.edge_faces[int(e)])
            key = (min(a, b), max(a, b))
            out[key] = out.get(key, 0) + 1
        return out

    def cells(self) -> FrozenSet[int]:
        return frozenset(int(x) for x in self.cx.face_label[self.cx.cell_faces])

    def vertices_by_cellset(self) -> Dict[FrozenSet[int], np.ndarray]:
        out: Dict[FrozenSet[int], List[np.ndarray]] = {}
        for v in range(self.cx.n_vertices):
            cells = frozenset(
                int(self.cx.face_label[f]) for f in self.cx.vertex_faces(v) if self.cx.face_kind[f] == FaceKind.CELL
            )
            if len(cells) >= 3:
                out.setdefault(cells, []).append(self.cx.vertex_xy[v])
        return {k: np.mean(v, axis=0) for k, v in out.items()}


def softmax_weights(energies: np.ndarray, temperature: float) -> np.ndarray:
    e = np.asarray(energies, dtype=np.float64)
    z = -(e - e.min()) / max(temperature, 1e-12)
    w = np.exp(z - z.max())
    return w / w.sum()


def effective_sample_size(w: np.ndarray) -> float:
    return float(1.0 / np.sum(np.asarray(w) ** 2))


def temperature_for_ess(energies: np.ndarray, ess_min: float) -> float:
    """Smallest temperature (sharpest posterior) whose weights still reach ess_min.
    Returns a tiny temperature if a single hypothesis is all that exists."""
    e = np.asarray(energies, dtype=np.float64)
    if e.size <= 1:
        return 1e-6
    spread = float(e.max() - e.min())
    if spread <= 0:
        return 1e-6
    ess_min = min(ess_min, float(e.size))
    lo, hi = 1e-4 * spread, 1e3 * spread
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if effective_sample_size(softmax_weights(e, mid)) >= ess_min:
            hi = mid
        else:
            lo = mid
    return float(hi)


@dataclass
class PosteriorEnsemble:
    hypotheses: List[Hypothesis]
    weights: np.ndarray
    temperature: float
    meta: Dict = field(default_factory=dict)

    @classmethod
    def from_hypotheses(
        cls,
        hyps: Sequence[Hypothesis],
        ess_min: float = 4.0,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        hyps = sorted(hyps, key=lambda h: h.energy)
        if top_k is not None:
            hyps = hyps[:top_k]
        E = np.array([h.energy for h in hyps])
        T = temperature if temperature is not None else temperature_for_ess(E, ess_min)
        w = softmax_weights(E, T)
        return cls(list(hyps), w, T, {"ess": effective_sample_size(w), "n": len(hyps)})

    @property
    def map_hypothesis(self) -> Hypothesis:
        return self.hypotheses[int(np.argmax(self.weights))]

    def expectation(self, stat: Callable[[Hypothesis], float]) -> float:
        vals = np.array([stat(h) for h in self.hypotheses], dtype=np.float64)
        return float(np.sum(self.weights * vals))

    def variance(self, stat: Callable[[Hypothesis], float]) -> float:
        vals = np.array([stat(h) for h in self.hypotheses], dtype=np.float64)
        m = float(np.sum(self.weights * vals))
        return float(np.sum(self.weights * (vals - m) ** 2))

    def credible_interval(self, stat: Callable[[Hypothesis], float], level: float = 0.9) -> Tuple[float, float]:
        vals = np.array([stat(h) for h in self.hypotheses], dtype=np.float64)
        order = np.argsort(vals)
        cw = np.cumsum(self.weights[order])
        lo = vals[order][np.searchsorted(cw, (1 - level) / 2)]
        hi = vals[order][min(np.searchsorted(cw, 1 - (1 - level) / 2), len(vals) - 1)]
        return float(lo), float(hi)

    def contact_probabilities(self) -> Dict[Tuple[int, int], float]:
        probs: Dict[Tuple[int, int], float] = {}
        for h, w in zip(self.hypotheses, self.weights):
            for key in h.contacts():
                probs[key] = probs.get(key, 0.0) + float(w)
        return probs

    def cell_probabilities(self) -> Dict[int, float]:
        probs: Dict[int, float] = {}
        for h, w in zip(self.hypotheses, self.weights):
            for c in h.cells():
                probs[c] = probs.get(c, 0.0) + float(w)
        return probs

    def vertex_credible_radius(self) -> Dict[FrozenSet[int], Dict[str, float]]:
        """For each tricellular cell set: posterior probability of existence, weighted
        mean position and RMS radius around it."""
        acc: Dict[FrozenSet[int], List[Tuple[float, np.ndarray]]] = {}
        for h, w in zip(self.hypotheses, self.weights):
            for cells, pos in h.vertices_by_cellset().items():
                acc.setdefault(cells, []).append((float(w), pos))
        out = {}
        for cells, lst in acc.items():
            ws = np.array([w for w, _ in lst])
            P = np.stack([p for _, p in lst])
            p_exist = float(ws.sum())
            wn = ws / ws.sum()
            mean = (wn[:, None] * P).sum(axis=0)
            rms = float(np.sqrt((wn * ((P - mean) ** 2).sum(axis=1)).sum()))
            out[cells] = {"p_exist": p_exist, "row": float(mean[0]), "col": float(mean[1]), "rms_radius_px": rms}
        return out

    def summary(self) -> Dict:
        cp = self.contact_probabilities()
        return {
            "n_hypotheses": len(self.hypotheses),
            "ess": effective_sample_size(self.weights),
            "temperature": self.temperature,
            "map_energy": float(self.map_hypothesis.energy),
            "energy_range": [
                float(min(h.energy for h in self.hypotheses)),
                float(max(h.energy for h in self.hypotheses)),
            ],
            "n_contacts_any": len(cp),
            "n_contacts_p_ge_0.5": int(sum(1 for p in cp.values() if p >= 0.5)),
            "n_contacts_uncertain_0.2_0.8": int(sum(1 for p in cp.values() if 0.2 <= p <= 0.8)),
            "expected_n_cells": self.expectation(lambda h: float(h.cx.cell_faces.size)),
            "expected_n_gaps": self.expectation(lambda h: float(h.cx.gap_faces.size)),
        }


def _score(result: DecodeResult, maps: ProposalMaps, image, weights, render_model, tag) -> Hypothesis:
    E, info = complex_energy(result.labels, result.cx, maps, image=image, weights=weights, render_model=render_model)
    return Hypothesis(result.labels, result.cx, E, info, result.params, tag)


def _labels_key(labels: np.ndarray) -> bytes:
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(labels).tobytes(), digest_size=16).digest()


def generate_hypotheses(
    maps: ProposalMaps,
    decoder: ConstrainedDecoder,
    base: DecoderParams,
    image: Optional[np.ndarray] = None,
    seed_thresholds: Iterable[float] = (0.2, 0.35, 0.5),
    seed_drop_fracs: Iterable[float] = (0.0, 0.05),
    boundary_smooth_sigmas: Iterable[float] = (0.0, 1.5),
    distance_mixes: Iterable[float] = (0.0, 0.3),
    gap_thresholds: Iterable[float] = (0.7,),
    n_merge_moves: int = 6,
    n_split_moves: int = 4,
    weights: Optional[EnergyWeights] = None,
    render_model: Optional[RenderModel] = None,
    max_hypotheses: int = 48,
    energy_maps: Optional[ProposalMaps] = None,
) -> List[Hypothesis]:
    """Perturbation grid over the decoder plus local merge/split moves from the best
    grid point. Pixel-identical label images are kept once.

    Perturbations change the flooding order (boundary smoothing, distance mixing) and
    the seed set (threshold, dropping the weakest seeds), which is what produces
    genuinely different legal complexes. A monotone rescaling of the elevation alone
    would not. ``energy_maps`` scores every hypothesis against a fixed reference map
    (needed when hypotheses from several proposal settings are pooled).
    """
    hyps: List[Hypothesis] = []
    seen = set()
    emaps = energy_maps if energy_maps is not None else maps
    cell_radius = float(maps.meta.get("cell_radius_px", base.cell_radius_px))
    grid = product(seed_thresholds, seed_drop_fracs, boundary_smooth_sigmas, distance_mixes, gap_thresholds)
    for st, sd, bs, dm, gt in grid:
        p = base.with_(
            seed_threshold=st,
            seed_drop_frac=sd,
            boundary_smooth_sigma=bs,
            distance_mix=dm,
            gap_threshold=gt,
            cell_radius_px=cell_radius,
            label=f"st{st}_drop{sd}_bs{bs}_dm{dm}_gap{gt}",
        )
        res = decoder.decode(maps, p)
        if res.cx.cell_faces.size == 0:
            continue
        key = _labels_key(res.labels)
        if key in seen:
            continue
        seen.add(key)
        hyps.append(_score(res, emaps, image, weights, render_model, p.label))
        if len(hyps) >= max_hypotheses:
            break
    if not hyps:
        return hyps
    best = min(hyps, key=lambda h: h.energy)
    res_best = DecodeResult(best.labels, best.cx, np.unique(best.labels[best.labels > 0]) - 1, best.params)
    H, W = maps.shape

    # merge moves across the weakest-supported cell-cell edges
    support = []
    for e in best.cx.cell_cell_edges():
        pth = best.cx.edge_geometry(int(e))
        pr = np.clip(np.round(pth[:, 0]).astype(int), 0, H - 1)
        pc = np.clip(np.round(pth[:, 1]).astype(int), 0, W - 1)
        support.append((float(maps.boundary[pr, pc].mean()), int(e)))
    support.sort()
    for _, e in support[:n_merge_moves]:
        a, b = (int(best.cx.face_label[f]) for f in best.cx.edge_faces[e])
        if a == b:
            continue
        merged = decoder.merge_faces(res_best, a, b)
        key = _labels_key(merged.labels)
        if key in seen:
            continue
        seen.add(key)
        hyps.append(_score(merged, emaps, image, weights, render_model, f"merge_{a}_{b}"))

    # split moves: cells containing a second, weaker seed candidate, or unusually large cells
    if n_split_moves > 0:
        pts = np.round(maps.seed_points).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
        owner = best.labels[pts[:, 0], pts[:, 1]]
        used = set(int(x) for x in np.unique(best.labels[best.labels > 0]))
        next_label = int(max(len(maps.seed_points), best.labels.max())) + 1
        candidates = []
        for k in range(len(pts)):
            lab = int(owner[k])
            if lab > 0 and lab != k + 1 and (k + 1) not in used:
                candidates.append((float(-maps.seed_scores[k]), lab, tuple(pts[k])))
        if len(candidates) < n_split_moves:
            areas = np.bincount(best.labels.ravel())
            big = np.argsort(-areas)
            for lab in big:
                lab = int(lab)
                if lab == 0 or areas[lab] == 0 or any(c[1] == lab for c in candidates):
                    continue
                region = best.labels == lab
                d = ndi.distance_transform_edt(region)
                # second seed at the interior point farthest from the existing seed
                own = pts[lab - 1] if lab - 1 < len(pts) else np.array(np.unravel_index(int(np.argmax(d)), d.shape))
                rr, cc = np.nonzero(region)
                far = np.argmax((rr - own[0]) ** 2 + (cc - own[1]) ** 2)
                candidates.append((0.0, lab, (int(rr[far]), int(cc[far]))))
                if len(candidates) >= n_split_moves:
                    break
        candidates.sort()
        for _, lab, rc in candidates[:n_split_moves]:
            split = decoder.split_face(maps, res_best, lab, rc, next_label)
            next_label += 1
            key = _labels_key(split.labels)
            if key in seen:
                continue
            seen.add(key)
            hyps.append(_score(split, emaps, image, weights, render_model, f"split_{lab}"))
    return hyps
