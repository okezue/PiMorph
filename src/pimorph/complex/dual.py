"""Projection of the complex to the dual cell-adjacency graph and clique checks.

The dual graph is an analysis view. It loses contact multiplicity (unless a
MultiGraph is used), cyclic order, gap faces and geometry.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from .geometry import edge_arclength
from .halfedge import FaceKind, HalfEdgeComplex


def to_multigraph(cx: HalfEdgeComplex) -> nx.MultiGraph:
    """One node per cell face, one edge per connected cell-cell contact component."""
    G = nx.MultiGraph()
    for f in cx.cell_faces:
        f = int(f)
        G.add_node(f, label=int(cx.face_label[f]))
    for e in cx.cell_cell_edges():
        e = int(e)
        a, b = (int(x) for x in cx.edge_faces[e])
        G.add_edge(a, b, key=e, edge_id=e, arclength_px=edge_arclength(cx, e))
    return G


def to_simple_graph(cx: HalfEdgeComplex) -> nx.Graph:
    """Collapse multiplicity: edge attributes n_components and total arclength."""
    G = nx.Graph()
    for f in cx.cell_faces:
        f = int(f)
        G.add_node(f, label=int(cx.face_label[f]))
    for e in cx.cell_cell_edges():
        e = int(e)
        a, b = (int(x) for x in cx.edge_faces[e])
        L = edge_arclength(cx, e)
        if G.has_edge(a, b):
            G[a][b]["n_components"] += 1
            G[a][b]["arclength_px"] += L
            G[a][b]["edge_ids"].append(e)
        else:
            G.add_edge(a, b, n_components=1, arclength_px=L, edge_ids=[e])
    return G


def tricellular_vertices(cx: HalfEdgeComplex) -> Dict[int, frozenset]:
    """Vertices incident to at least three distinct cell faces -> their cell set."""
    out: Dict[int, frozenset] = {}
    for v in range(cx.n_vertices):
        cells = cx.vertex_cell_set(v)
        if len(cells) >= 3:
            out[v] = cells
    return out


def clique_vertex_report(cx: HalfEdgeComplex) -> Dict[str, object]:
    """Separate graph 3-cliques from physically realized tricellular vertices.

    A 3-clique in the dual graph is three pairwise contacts. It corresponds to a
    tricellular junction only when some vertex is incident to all three cells.
    """
    G = to_simple_graph(cx)
    cliques: List[Tuple[int, int, int]] = [tuple(sorted(c)) for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    vert_sets: Set[frozenset] = set()
    for v, cells in tricellular_vertices(cx).items():
        # a degree-4 vertex with 4 cells realizes 4 triples
        cells_l = sorted(cells)
        if len(cells_l) == 3:
            vert_sets.add(frozenset(cells_l))
        else:
            from itertools import combinations

            for trip in combinations(cells_l, 3):
                vert_sets.add(frozenset(trip))
    with_vertex = [c for c in cliques if frozenset(c) in vert_sets]
    without_vertex = [c for c in cliques if frozenset(c) not in vert_sets]
    degs = cx.vertex_degrees()
    n_cells_at_vertex = np.array([len(cx.vertex_cell_set(v)) for v in range(cx.n_vertices)], dtype=np.int64)
    return {
        "n_3cliques": len(cliques),
        "n_3cliques_with_common_vertex": len(with_vertex),
        "n_3cliques_without_common_vertex": len(without_vertex),
        "cliques_without_common_vertex": without_vertex,
        "n_vertices": int(cx.n_vertices),
        "n_tricellular_vertices": int(np.sum(n_cells_at_vertex >= 3)),
        "n_vertices_touching_gap": int(
            sum(1 for v in range(cx.n_vertices) if any(cx.face_kind[f] == FaceKind.GAP for f in cx.vertex_faces(v)))
        ),
        "vertex_degree_histogram": {int(d): int(n) for d, n in zip(*np.unique(degs, return_counts=True))}
        if cx.n_vertices
        else {},
    }
