"""Legacy EndoPiGraph outputs (cells.csv, edges.csv, graph.graphml, graph.json) from a complex.

Consumers: ``scripts/harden_network_stats.py`` (cells: cell_id, area; edges: cell_i,
cell_j, aj_morph, aj_occupancy) and ``labeler.py`` (cells: cell_id, cx, cy; edges:
cell_i, cell_j, contact_px, aj_morph or AJ_morph_label, optional iface_cy/iface_cx).
Both label spellings are written because the two consumers disagree.

Cells are keyed by the original segmentation label. A label that was split into
several 4-connected faces yields one row per face with a repeated cell_id; the
per-pair graph merges them into one node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from endopigraph.graph_build import build_graph, write_graph_outputs

from ..complex.geometry import edge_arclength, face_area, face_centroid, face_perimeter
from ..complex.halfedge import FaceKind, HalfEdgeComplex
from ..fields.functionals import LEGACY_FEATURE_COLUMNS, legacy_feature_frame, morph_labels
from ..fields.profile import SCALAR_FIELDS

# Legacy pipeline default for the has_AJ flag (endopigraph.pipeline min_occupancy).
_MIN_OCCUPANCY = 0.05

# "area" duplicates area_px for scripts/harden_network_stats.py.
CELL_COLUMNS = (
    "cell_id",
    "face_id",
    "area_px",
    "area",
    "cx",
    "cy",
    "perimeter_px",
    "n_neighbors",
    "degree",
    "touches_border",
)


def legacy_cells_frame(cx: HalfEdgeComplex, labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    smoothed = cx.edge_smooth is not None
    counts = np.bincount(np.asarray(labels).ravel()) if labels is not None else None
    rows = []
    for f in cx.cell_faces:
        f = int(f)
        lab = int(cx.face_label[f])
        cy, cxx = face_centroid(cx, f, smoothed)
        if counts is not None and lab < counts.size:
            area = float(counts[lab])
        else:
            area = float(face_area(cx, f, smoothed))
        n_nb = len(set(cx.face_neighbors(f, kinds=(FaceKind.CELL,))))
        rows.append(
            {
                "cell_id": lab,
                "face_id": f,
                "area_px": area,
                "area": area,
                "cx": float(cxx),
                "cy": float(cy),
                "perimeter_px": float(face_perimeter(cx, f, smoothed)),
                "n_neighbors": n_nb,
                "degree": n_nb,
                "touches_border": bool(cx.face_touches_outer(f)),
            }
        )
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


def legacy_edges_frame(
    cx: HalfEdgeComplex,
    profiles_df: Optional[pd.DataFrame] = None,
    labels_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """One row per cell-cell edge (contact component), keyed by original labels."""
    px = cx.pixel_size_um
    pair_counter: Dict[tuple, int] = {}
    rows = []
    for e in cx.cell_cell_edges():
        e = int(e)
        a, b = (int(x) for x in cx.edge_faces[e])
        la, lb = int(cx.face_label[a]), int(cx.face_label[b])
        ci, cj = (la, lb) if la <= lb else (lb, la)
        fi, fj = (a, b) if la <= lb else (b, a)
        k = pair_counter.get((ci, cj), 0)
        pair_counter[(ci, cj)] = k + 1
        L = float(edge_arclength(cx, e, smoothed=True))
        p = cx.edge_geometry(e)
        mid = p[p.shape[0] // 2]
        rows.append(
            {
                "cell_i": ci,
                "cell_j": cj,
                "edge_id": e,
                "component_index": k,
                "face_i": fi,
                "face_j": fj,
                "contact_px": L,
                "arclength_um": L * float(px) if px is not None else np.nan,
                "iface_cy": float(mid[0]),
                "iface_cx": float(mid[1]),
            }
        )
    base_cols = [
        "cell_i",
        "cell_j",
        "edge_id",
        "component_index",
        "face_i",
        "face_j",
        "contact_px",
        "arclength_um",
        "iface_cy",
        "iface_cx",
    ]
    edges = pd.DataFrame(rows, columns=base_cols)
    if len(edges):
        keys = list(zip(edges["cell_i"], edges["cell_j"]))
        edges["n_components"] = [pair_counter[k] for k in keys]
    else:
        edges["n_components"] = pd.Series(dtype=int)

    if profiles_df is None:
        return edges

    prof = profiles_df.copy()
    if "edge_id" not in prof.columns:
        raise KeyError("profiles_df must contain an edge_id column")
    scalar_cols = [c for c in SCALAR_FIELDS if c in prof.columns]
    feats = legacy_feature_frame(prof)
    if labels_series is not None:
        morph = pd.Series(np.asarray(labels_series, dtype=object), index=prof.index)
    elif "AJ_morph_label" in prof.columns:
        morph = prof["AJ_morph_label"].astype(object)
    elif "aj_morph" in prof.columns:
        morph = prof["aj_morph"].astype(object)
    else:
        morph = morph_labels(feats)
    extra = prof[["edge_id", *scalar_cols]].copy()
    for c in LEGACY_FEATURE_COLUMNS:
        extra[c] = feats[c].values
    extra["AJ_morph_label"] = morph.values
    extra["aj_morph"] = morph.values
    extra["aj_occupancy"] = feats["AJ_occupancy"].values
    extra["has_AJ"] = feats["AJ_occupancy"].values >= _MIN_OCCUPANCY
    if "threshold" in prof.columns:
        extra["AJ_threshold"] = prof["threshold"].values
    extra = extra.drop_duplicates("edge_id")
    merged = edges.merge(extra, on="edge_id", how="left")
    # Edges outside profiles_df (edges= subset) stay unflagged and unclassified.
    has = merged["has_AJ"]
    merged["has_AJ"] = np.where(has.isna(), False, has.values).astype(bool)
    for c in ("AJ_morph_label", "aj_morph"):
        merged[c] = merged[c].astype(object).where(merged[c].notna(), "unknown")
    return merged


def legacy_pair_frame(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse contact components to one row per (cell_i, cell_j) pair."""
    if len(edges) == 0:
        return pd.DataFrame(columns=["cell_i", "cell_j", "n_components", "contact_px", "edge_ids"])
    agg = {"contact_px": ("contact_px", "sum"), "edge_id": ("edge_id", lambda s: json.dumps(sorted(int(x) for x in s)))}
    if "arclength_um" in edges.columns:
        agg["arclength_um"] = ("arclength_um", "sum")
    if "AJ_occupancy" in edges.columns:
        agg["AJ_occupancy"] = ("AJ_occupancy", "max")
        agg["has_AJ"] = ("has_AJ", "any")
    if "AJ_morph_label" in edges.columns:
        # label of the longest component represents the pair
        agg["AJ_morph_label"] = ("AJ_morph_label", "first")
    src = edges.sort_values(["cell_i", "cell_j", "contact_px"], ascending=[True, True, False])
    g = src.groupby(["cell_i", "cell_j"], sort=True)
    out = g.agg(**agg).reset_index()
    out["n_components"] = g.size().values
    out = out.rename(columns={"edge_id": "edge_ids"})
    if "AJ_morph_label" in out.columns:
        out["aj_morph"] = out["AJ_morph_label"]
    return out


def write_legacy_outputs(
    cx: HalfEdgeComplex,
    out_dir: Path | str,
    image_id: str,
    profiles_df: Optional[pd.DataFrame] = None,
    labels: Optional[np.ndarray] = None,
    morph: Optional[pd.Series] = None,
) -> Dict[str, Path]:
    """Write ``out_dir/<image_id>/{cells.csv, edges.csv, graph.graphml, graph.json, complex.json}``.

    ``labels`` is the label image (exact pixel areas); ``profiles_df`` the output of
    ``pimorph.fields.profile_all_edges``; ``morph`` optional precomputed morphology
    strings aligned with ``profiles_df`` (default: heuristic labels).
    """
    out = Path(out_dir) / str(image_id)
    out.mkdir(parents=True, exist_ok=True)

    cells = legacy_cells_frame(cx, labels)
    edges = legacy_edges_frame(cx, profiles_df, morph)
    pairs = legacy_pair_frame(edges)

    cells_path = out / "cells.csv"
    edges_path = out / "edges.csv"
    cells.to_csv(cells_path, index=False)
    edges.to_csv(edges_path, index=False)

    # build_graph keys nodes by cell_id; drop repeated split-label rows first.
    node_frame = cells.drop_duplicates("cell_id")
    G = build_graph(node_frame, pairs, ["AJ"] if "has_AJ" in pairs.columns else [])
    graph_paths = write_graph_outputs(G, out / "graph")

    complex_path = out / "complex.json"
    complex_path.write_text(json.dumps(cx.to_dict(), default=_json_default), encoding="utf-8")

    return {
        "cells": cells_path,
        "edges": edges_path,
        "graphml": graph_paths["graphml"],
        "json": graph_paths["json"],
        "complex": complex_path,
    }


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
