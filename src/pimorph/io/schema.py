"""Tabular views of a complex: cells, interfaces, vertices, gaps, provenance.

Geometry is reported in pixels and, when ``pixel_size_um`` is known, in micrometers.
Missing calibration is recorded as ``units = "px"`` in provenance rather than guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..complex.geometry import (
    edge_arclengths,
    face_area,
    face_centroid,
    face_perimeter,
    polyline_curvature,
)
from ..complex.halfedge import FaceKind, HalfEdgeComplex, VertexKind


@dataclass
class ComplexTables:
    cells: pd.DataFrame
    interfaces: pd.DataFrame
    vertices: pd.DataFrame
    gaps: pd.DataFrame
    provenance: Dict


def _um(x: np.ndarray, px: Optional[float], power: int = 1) -> np.ndarray:
    if px is None:
        return np.full_like(np.asarray(x, dtype=np.float64), np.nan)
    return np.asarray(x, dtype=np.float64) * (float(px) ** power)


def complex_tables(cx: HalfEdgeComplex, labels: Optional[np.ndarray] = None) -> ComplexTables:
    px = cx.pixel_size_um
    smoothed = cx.edge_smooth is not None
    outer = cx.outer_face

    # ---------------------------------------------------------------- cells
    rows = []
    counts = np.bincount(labels.ravel()) if labels is not None else None
    for f in cx.cell_faces:
        f = int(f)
        cy, cxx = face_centroid(cx, f, smoothed)
        n_cell_nb = len(cx.face_neighbors(f, kinds=(FaceKind.CELL,)))
        n_gap_nb = len(cx.face_neighbors(f, kinds=(FaceKind.GAP,)))
        lab = int(cx.face_label[f])
        rows.append(
            {
                "face_id": f,
                "label": lab,
                "area_poly_px": face_area(cx, f, smoothed),
                "area_pixels": int(counts[lab]) if (counts is not None and lab < counts.size) else np.nan,
                "perimeter_px": face_perimeter(cx, f, smoothed),
                "n_sides": cx.face_sides(f),
                "n_cell_neighbors_multiplicity": n_cell_nb,
                "n_cell_neighbors_distinct": len(set(cx.face_neighbors(f, kinds=(FaceKind.CELL,)))),
                "n_gap_sides": n_gap_nb,
                "n_holes": max(len(cx.face_loops[f]) - 1, 0),
                "touches_border": cx.face_touches_outer(f),
                "centroid_row": cy,
                "centroid_col": cxx,
            }
        )
    cells = pd.DataFrame(rows)
    if len(cells):
        cells["area_um2"] = _um(cells["area_poly_px"].values, px, 2)
        cells["perimeter_um"] = _um(cells["perimeter_px"].values, px)

    # ----------------------------------------------------------- interfaces
    L = edge_arclengths(cx, smoothed) if cx.n_edges else np.zeros(0)
    pair_counter: Dict[tuple, int] = {}
    irows = []
    for e in range(cx.n_edges):
        a, b = (int(x) for x in cx.edge_faces[e])
        key = (min(a, b), max(a, b))
        k = pair_counter.get(key, 0)
        pair_counter[key] = k + 1
        p = cx.edge_geometry(e)
        curv = polyline_curvature(p)
        irows.append(
            {
                "edge_id": e,
                "face_left": a,
                "face_right": b,
                "label_left": int(cx.face_label[a]),
                "label_right": int(cx.face_label[b]),
                "kind": _edge_kind(cx, e, outer),
                "component_index": k,
                "tail_vertex": int(cx.edge_tail[e]),
                "head_vertex": int(cx.edge_head[e]),
                "arclength_px": float(L[e]),
                "n_points": int(p.shape[0]),
                "mean_abs_curvature_px": float(np.mean(np.abs(curv))) if curv.size else 0.0,
                "is_loop": bool(cx.edge_tail[e] == cx.edge_head[e]),
            }
        )
    interfaces = pd.DataFrame(irows)
    if len(interfaces):
        interfaces["arclength_um"] = _um(interfaces["arclength_px"].values, px)
        pair_keys = [(min(a, b), max(a, b)) for a, b in zip(interfaces["face_left"], interfaces["face_right"])]
        interfaces["n_components_between_pair"] = [pair_counter[k] for k in pair_keys]

    # ------------------------------------------------------------- vertices
    degs = cx.vertex_degrees()
    vrows = []
    for v in range(cx.n_vertices):
        faces = cx.vertex_faces(v)
        kinds = [int(cx.face_kind[f]) for f in faces]
        vrows.append(
            {
                "vertex_id": v,
                "row": float(cx.vertex_xy[v, 0]),
                "col": float(cx.vertex_xy[v, 1]),
                "degree": int(degs[v]),
                "kind": "artificial" if cx.vertex_kind[v] == VertexKind.ARTIFICIAL else "regular",
                "n_cells": sum(1 for k in kinds if k == FaceKind.CELL),
                "n_gaps": sum(1 for k in kinds if k == FaceKind.GAP),
                "touches_outer": any(k == FaceKind.OUTER for k in kinds),
                "incident_faces_cyclic": json.dumps([int(f) for f in faces]),
                "incident_edges_cyclic": json.dumps([int(h >> 1) for h in cx.vertex_out_half_edges(v)]),
            }
        )
    vertices = pd.DataFrame(vrows)
    if len(vertices):
        vertices["row_um"] = _um(vertices["row"].values, px)
        vertices["col_um"] = _um(vertices["col"].values, px)

    # ----------------------------------------------------------------- gaps
    grows = []
    for f in cx.gap_faces:
        f = int(f)
        cy, cxx = face_centroid(cx, f, smoothed)
        cells_around = sorted(set(cx.face_neighbors(f, kinds=(FaceKind.CELL,))))
        grows.append(
            {
                "face_id": f,
                "area_poly_px": abs(face_area(cx, f, smoothed)),
                "perimeter_px": face_perimeter(cx, f, smoothed),
                "n_sides": cx.face_sides(f),
                "n_bounding_cells": len(cells_around),
                "bounding_cells": json.dumps(cells_around),
                "centroid_row": cy,
                "centroid_col": cxx,
            }
        )
    gaps = pd.DataFrame(grows)
    if len(gaps):
        gaps["area_um2"] = _um(gaps["area_poly_px"].values, px, 2)

    prov = {
        **cx.provenance,
        "units": "um" if px is not None else "px",
        "pixel_size_um": px,
        "shape": None if cx.shape is None else list(cx.shape),
        "geometry": "smoothed" if smoothed else "crack",
        "n_vertices": cx.n_vertices,
        "n_edges": cx.n_edges,
        "n_faces": cx.n_faces,
    }
    return ComplexTables(cells=cells, interfaces=interfaces, vertices=vertices, gaps=gaps, provenance=prov)


def _edge_kind(cx: HalfEdgeComplex, e: int, outer: int) -> str:
    a, b = (int(x) for x in cx.edge_faces[e])
    ka, kb = int(cx.face_kind[a]), int(cx.face_kind[b])
    if a == outer or b == outer:
        return "cell_outer" if FaceKind.CELL in (ka, kb) else "gap_outer"
    if ka == FaceKind.CELL and kb == FaceKind.CELL:
        return "cell_cell"
    if FaceKind.GAP in (ka, kb) and FaceKind.CELL in (ka, kb):
        return "cell_gap"
    return "gap_gap"


def write_tables(tables: ComplexTables, out_dir: Path | str, prefix: str = "", fmt: str = "parquet") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name in ("cells", "interfaces", "vertices", "gaps"):
        df: pd.DataFrame = getattr(tables, name)
        p = out_dir / f"{prefix}{name}.{'parquet' if fmt == 'parquet' else 'csv'}"
        if fmt == "parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        paths[name] = p
    pp = out_dir / f"{prefix}provenance.json"
    pp.write_text(json.dumps(tables.provenance, indent=2, default=str), encoding="utf-8")
    paths["provenance"] = pp
    return paths
