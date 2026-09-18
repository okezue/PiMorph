"""Paracellular transport proxies on the cell complex.

Two passive conductance models turn per-interface junction coverage into barrier
numbers. Both are PREDICTION TARGETS: no TEER, tracer flux or permeability
measurement exists for any dataset processed here, so nothing below is validated
against function. The models are exact for what they state, which is a resistor
network, not a claim about biology.

Parallel (through-plane) model. Every cell-cell interface e is a leak path of
conductance g_e from the luminal to the abluminal side, every explicit gap is a short
circuit of conductance g_gap, and every cell offers a transcellular path g_trans.
Parallel paths add, so G_eff = sum_e g_e + n_gaps g_gap + n_cells g_trans and
dG_eff/dg_e = 1. Divided by tissue area this is a permeability index.

In-plane model. Cells (and gap faces) are nodes, interfaces are resistors of
conductance g_e, and the effective conductance between a source and a sink cell set is
the current through the network at unit potential difference. This expresses how a
tracer entering at the source cells would spread along the paracellular network. It
is a graph Laplacian with Dirichlet data; by Rayleigh's monotonicity law G_eff never
decreases when any g_e increases, and the adjoint sensitivity is
dG/dg_e = (phi_i - phi_j)^2 (Dirichlet energy is stationary in phi).

A faithful barrier model would be a thin 3-D domain with lateral leak channels; the
in-plane network with normal leaks is the accepted 2-D proxy (blueprint section 8.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from ..complex.geometry import face_area
from ..complex.halfedge import FaceKind, HalfEdgeComplex


def interface_conductance(
    per_edge: pd.DataFrame,
    channel_weights: Dict[str, float],
    baseline: float = 0.05,
    occupancy_template: str = "{c}_coverage",
    arclength_col: str = "arclength_px",
    length_scale: float = 1.0,
) -> np.ndarray:
    """Per-interface leak conductance g_e = baseline + sum_c w_c (1 - occ_c) L_e.

    ``occ_c`` is read from ``occupancy_template.format(c=channel)``; when that column
    is missing ``<c>_occupancy`` is used. Less junction coverage means more leak.
    Missing (NaN) coverage counts as zero coverage: an interface with no junction
    evidence is treated as open. ``length_scale`` converts the arclength column (for
    example ``pixel_size_um`` to get micrometres). Returned array is aligned with the
    rows of ``per_edge``.
    """
    if baseline < 0:
        raise ValueError("baseline must be non-negative")
    L = per_edge[arclength_col].to_numpy(dtype=np.float64) * float(length_scale)
    g = np.full(len(per_edge), float(baseline))
    for c, w in channel_weights.items():
        col = occupancy_template.format(c=c)
        if col not in per_edge.columns:
            col = f"{c}_occupancy"
        if col not in per_edge.columns:
            raise KeyError(f"no coverage column for channel {c!r} in per_edge")
        occ = per_edge[col].to_numpy(dtype=np.float64)
        occ = np.where(np.isfinite(occ), occ, 0.0)
        if w < 0:
            raise ValueError("channel weights must be non-negative")
        g = g + float(w) * (1.0 - np.clip(occ, 0.0, 1.0)) * L
    return g


def edge_conductance_vector(
    cx: HalfEdgeComplex, edge_ids: Sequence[int], values: Sequence[float], fill: float = 0.0
) -> np.ndarray:
    """Scatter per-row values onto an (n_edges,) array indexed by edge id."""
    out = np.full(cx.n_edges, float(fill), dtype=np.float64)
    out[np.asarray(edge_ids, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    return out


def _edge_vector(cx: HalfEdgeComplex, g_edge) -> np.ndarray:
    if isinstance(g_edge, pd.Series):
        return edge_conductance_vector(cx, g_edge.index.to_numpy(), g_edge.to_numpy())
    g = np.asarray(g_edge, dtype=np.float64).reshape(-1)
    if g.shape[0] != cx.n_edges:
        raise ValueError(f"g_edge must have one entry per edge ({cx.n_edges}), got {g.shape[0]}")
    if (g < 0).any():
        raise ValueError("conductances must be non-negative")
    return g


def tissue_area(cx: HalfEdgeComplex, units: str = "auto") -> Tuple[float, str]:
    """Total area of the cell faces (smoothed geometry) and its unit.

    ``units="auto"`` returns um^2 when the complex is calibrated, else px^2.
    """
    a_px = float(sum(face_area(cx, int(f)) for f in cx.cell_faces))
    if units == "px" or (units == "auto" and cx.pixel_size_um is None):
        return a_px, "px^2"
    if cx.pixel_size_um is None:
        raise ValueError("complex is not calibrated")
    return a_px * float(cx.pixel_size_um) ** 2, "um^2"


# ------------------------------------------------------------ parallel model
def effective_conductance(
    cx: HalfEdgeComplex,
    g_edge,
    g_gap: float = 10.0,
    g_trans: float = 0.0,
) -> float:
    """Through-plane conductance of the parallel model (untested prediction).

    Sums g_e over cell-cell edges, g_gap per explicit gap face and g_trans per cell.
    """
    g = _edge_vector(cx, g_edge)
    cc = cx.cell_cell_edges()
    return float(g[cc].sum() + cx.gap_faces.size * float(g_gap) + cx.cell_faces.size * float(g_trans))


def conductance_breakdown(cx: HalfEdgeComplex, g_edge, g_gap: float = 10.0, g_trans: float = 0.0) -> Dict[str, float]:
    """Junction, gap and transcellular contributions of the parallel model."""
    g = _edge_vector(cx, g_edge)
    gj = float(g[cx.cell_cell_edges()].sum())
    gg = float(cx.gap_faces.size * g_gap)
    gt = float(cx.cell_faces.size * g_trans)
    tot = gj + gg + gt
    return {
        "G_junction": gj,
        "G_gap": gg,
        "G_trans": gt,
        "G_eff": tot,
        "gap_fraction": gg / tot if tot > 0 else float("nan"),
        "junction_fraction": gj / tot if tot > 0 else float("nan"),
    }


def permeability_index(
    cx: HalfEdgeComplex,
    g_edge,
    g_gap: float = 10.0,
    g_trans: float = 0.0,
    area: Optional[float] = None,
) -> float:
    """G_eff of the parallel model per unit tissue area (untested prediction).

    ``area`` defaults to ``tissue_area(cx)`` (um^2 when calibrated, else px^2).
    """
    if area is None:
        area, _ = tissue_area(cx)
    if area <= 0:
        return float("nan")
    return effective_conductance(cx, g_edge, g_gap=g_gap, g_trans=g_trans) / float(area)


# ------------------------------------------------------------- in-plane model
@dataclass
class BarrierNetwork:
    """Resistor network of the in-plane model.

    ``laplacian`` is (n_nodes, n_nodes) CSR; ``node_faces[i]`` is the face id of node
    i (cells first, then gap faces when included); ``edge_ids``/``edge_nodes``/``edge_g``
    list the resistors (one per contributing edge) with their node pair and weight.
    """

    laplacian: sp.csr_matrix
    node_faces: np.ndarray
    edge_ids: np.ndarray
    edge_nodes: np.ndarray  # (n_res, 2)
    edge_g: np.ndarray
    face_to_node: Dict[int, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.node_faces.shape[0])


def barrier_network(
    cx: HalfEdgeComplex,
    g_edge,
    gap_conductance: float = 10.0,
    include_gaps: bool = True,
) -> BarrierNetwork:
    """Weighted graph Laplacian of the paracellular network.

    Nodes are cell faces (plus gap faces when ``include_gaps``); every cell-cell edge e
    is a resistor of conductance g_e between its two faces. Cell-gap edges are short
    circuits of conductance ``gap_conductance``. Edges to the outer face are dropped.
    """
    g = _edge_vector(cx, g_edge)
    faces = list(int(f) for f in cx.cell_faces)
    if include_gaps:
        faces += [int(f) for f in cx.gap_faces]
    face_to_node = {f: i for i, f in enumerate(faces)}
    ids: List[int] = []
    pairs: List[Tuple[int, int]] = []
    ws: List[float] = []
    kinds = cx.face_kind
    for e in range(cx.n_edges):
        a, b = (int(x) for x in cx.edge_faces[e])
        ka, kb = int(kinds[a]), int(kinds[b])
        if ka == FaceKind.CELL and kb == FaceKind.CELL:
            w = float(g[e])
        elif include_gaps and {ka, kb} == {int(FaceKind.CELL), int(FaceKind.GAP)}:
            w = float(gap_conductance)
        else:
            continue
        if a == b:
            continue
        ids.append(e)
        pairs.append((face_to_node[a], face_to_node[b]))
        ws.append(w)
    n = len(faces)
    if pairs:
        p = np.asarray(pairs, dtype=np.int64)
        w = np.asarray(ws, dtype=np.float64)
        rows = np.concatenate([p[:, 0], p[:, 1], p[:, 0], p[:, 1]])
        cols = np.concatenate([p[:, 1], p[:, 0], p[:, 0], p[:, 1]])
        vals = np.concatenate([-w, -w, w, w])
        lap = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    else:
        p = np.zeros((0, 2), dtype=np.int64)
        w = np.zeros(0)
        lap = sp.csr_matrix((n, n))
    return BarrierNetwork(
        laplacian=lap,
        node_faces=np.asarray(faces, dtype=np.int64),
        edge_ids=np.asarray(ids, dtype=np.int64),
        edge_nodes=p,
        edge_g=w,
        face_to_node=face_to_node,
    )


@dataclass
class InPlaneSolution:
    conductance: float
    potential: np.ndarray  # per network node
    network: BarrierNetwork
    source_nodes: np.ndarray
    sink_nodes: np.ndarray

    def edge_sensitivity(self) -> np.ndarray:
        """dG/dg for every resistor of the network, (phi_i - phi_j)^2."""
        d = self.potential[self.network.edge_nodes[:, 0]] - self.potential[self.network.edge_nodes[:, 1]]
        return d**2


def _nodes_for_faces(net: BarrierNetwork, faces: Iterable[int]) -> np.ndarray:
    out = []
    for f in faces:
        f = int(f)
        if f not in net.face_to_node:
            raise KeyError(f"face {f} is not a node of the barrier network")
        out.append(net.face_to_node[f])
    return np.unique(np.asarray(out, dtype=np.int64))


def solve_in_plane(
    cx: HalfEdgeComplex,
    g_edge,
    source_cells: Sequence[int],
    sink_cells: Sequence[int],
    gap_conductance: float = 10.0,
    include_gaps: bool = True,
) -> InPlaneSolution:
    """Dirichlet problem on the paracellular network: phi = 1 on the source cells,
    phi = 0 on the sink cells, Kirchhoff elsewhere. The conductance is the current
    leaving the sources. Free nodes not connected to any Dirichlet node stay at 0
    potential and carry no current."""
    net = barrier_network(cx, g_edge, gap_conductance=gap_conductance, include_gaps=include_gaps)
    n = net.n_nodes
    src = _nodes_for_faces(net, source_cells)
    snk = _nodes_for_faces(net, sink_cells)
    if src.size == 0 or snk.size == 0:
        raise ValueError("source and sink sets must be non-empty")
    if np.intersect1d(src, snk).size:
        raise ValueError("source and sink sets overlap")
    phi = np.zeros(n)
    phi[src] = 1.0
    fixed = np.zeros(n, dtype=bool)
    fixed[src] = True
    fixed[snk] = True
    L = net.laplacian
    if n and L.nnz:
        adj = sp.csr_matrix((np.ones(L.nnz), L.indices, L.indptr), shape=L.shape)
        _, comp = connected_components(adj, directed=False)
        touched = np.unique(comp[fixed])
        free = ~fixed & np.isin(comp, touched)
    else:
        free = np.zeros(n, dtype=bool)
    fi = np.flatnonzero(free)
    di = np.flatnonzero(fixed)
    if fi.size:
        Luu = L[fi][:, fi].tocsc()
        Lud = L[fi][:, di]
        rhs = -Lud @ phi[di]
        # weakly connected free components still have a full-rank block; add a tiny
        # ridge for round-off safety
        Luu = Luu + sp.identity(fi.size, format="csc") * 1e-14
        phi[fi] = spsolve(Luu, rhs)
    current = float((L @ phi)[src].sum()) if L.nnz else 0.0
    return InPlaneSolution(conductance=max(current, 0.0), potential=phi, network=net, source_nodes=src, sink_nodes=snk)


def in_plane_effective_conductance(
    cx: HalfEdgeComplex,
    g_edge,
    source_cells: Sequence[int],
    sink_cells: Sequence[int],
    gap_conductance: float = 10.0,
    include_gaps: bool = True,
) -> float:
    """Effective conductance between source and sink cell sets (untested prediction)."""
    return solve_in_plane(
        cx, g_edge, source_cells, sink_cells, gap_conductance=gap_conductance, include_gaps=include_gaps
    ).conductance


def sensitivity(
    cx: HalfEdgeComplex,
    g_edge,
    model: str = "parallel",
    source_cells: Optional[Sequence[int]] = None,
    sink_cells: Optional[Sequence[int]] = None,
    gap_conductance: float = 10.0,
    include_gaps: bool = True,
) -> np.ndarray:
    """dG_eff/dg_e for every edge id (zero for edges that are not cell-cell resistors).

    ``model="parallel"``: 1 on every cell-cell edge. ``model="in_plane"``: adjoint
    sensitivity (phi_i - phi_j)^2 from the Dirichlet solution between ``source_cells``
    and ``sink_cells``.
    """
    g = _edge_vector(cx, g_edge)
    out = np.zeros(cx.n_edges, dtype=np.float64)
    if model == "parallel":
        out[cx.cell_cell_edges()] = 1.0
        return out
    if model != "in_plane":
        raise ValueError("model must be 'parallel' or 'in_plane'")
    if source_cells is None or sink_cells is None:
        raise ValueError("in_plane sensitivity needs source_cells and sink_cells")
    sol = solve_in_plane(cx, g, source_cells, sink_cells, gap_conductance=gap_conductance, include_gaps=include_gaps)
    s = sol.edge_sensitivity()
    net = sol.network
    cc = np.isin(net.edge_ids, cx.cell_cell_edges())
    out[net.edge_ids[cc]] = s[cc]
    return out


def boundary_cells(cx: HalfEdgeComplex) -> np.ndarray:
    """Cell faces that share an edge with the outer face."""
    outer = cx.outer_face
    touch = (cx.edge_faces == outer).any(axis=1)
    faces = np.unique(cx.edge_faces[touch])
    return faces[cx.face_kind[faces] == FaceKind.CELL]
