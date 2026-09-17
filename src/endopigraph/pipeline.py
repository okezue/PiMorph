from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from .ajmorph import compute_threshold, compute_interface_features, infer_ajmorph_label_heuristic
from .config import load_config, resolve_channel_roles
from .graph_build import build_graph, write_graph_outputs
from .interfaces import extract_interfaces, interface_mask_from_coords
from .io import read_image
from .report import write_html_report
from .segmentation import segment_cells
from .utils import ensure_dir
from .viz import save_feature_distributions, save_graph_plot, save_segmentation_qc


def _cell_table(labels: np.ndarray) -> pd.DataFrame:
    props = regionprops_table(labels, properties=("label", "area", "centroid"))
    df = pd.DataFrame(props)
    df = df.rename(columns={"label": "cell_id", "centroid-0": "cy", "centroid-1": "cx", "area": "area_px"})
    return df


def _resolve_channel_index(
    channel_names: List[str],
    *,
    channel_name: Optional[str] = None,
    channel_index: Optional[int] = None,
) -> int:
    if channel_index is not None:
        if channel_index < 0 or channel_index >= len(channel_names):
            raise ValueError(f"channel_index {channel_index} out of range for {len(channel_names)} channels")
        return int(channel_index)
    if channel_name is not None:
        for i, n in enumerate(channel_names):
            if n == channel_name:
                return i
        # fallback: case-insensitive contains match
        lname = channel_name.lower()
        for i, n in enumerate(channel_names):
            if lname in n.lower():
                return i
        raise ValueError(f"Could not find channel_name '{channel_name}' in channel_names={channel_names}")
    # default
    return 0


class PipelineError(RuntimeError):
    pass


def process_one_image(
    image_id: str,
    path: Path,
    cfg: dict,
    out_root: Path,
) -> Dict[str, str]:
    """Process one image. Returns a dict for the HTML report."""
    arr, channel_names = read_image(path)

    # Choose a display channel (AJ if configured, else channel 0)
    junction_markers: Dict[str, dict] = cfg.get("junction_markers", {})
    display_idx = 0
    if "AJ" in junction_markers:
        jm = junction_markers["AJ"]
        display_idx = _resolve_channel_index(channel_names, channel_name=jm.get("channel_name"), channel_index=jm.get("channel_index"))

    seg_cfg = cfg.get("segmentation", {})
    labels = segment_cells(arr, channel_names, seg_cfg)

    # Cell table
    cells_df = _cell_table(labels)

    # Interfaces
    iface = extract_interfaces(labels)
    edges_df = iface.edges.copy()

    # Filter edges by minimum contact
    min_contact = int(cfg.get("graph", {}).get("min_contact_px", 10))
    edges_df = edges_df.loc[edges_df["contact_px"] >= min_contact].reset_index(drop=True)

    # Compute marker thresholds and per-edge features
    for jtype, jcfg in junction_markers.items():
        ch_idx = _resolve_channel_index(channel_names, channel_name=jcfg.get("channel_name"), channel_index=jcfg.get("channel_index"))
        marker = arr[ch_idx].astype(np.float32)

        # use all boundary pixels to compute a global threshold per image
        boundary_values = marker[iface.all_boundary_mask]
        method = str(jcfg.get("threshold", "otsu"))
        thr = compute_threshold(boundary_values, method)

        dilate_px = int(jcfg.get("dilate_px", 2))
        min_occ = float(jcfg.get("min_occupancy", 0.05))

        feats_rows: List[Dict[str, float]] = []
        has_list: List[bool] = []
        ajmorph_list: List[str] = []

        for _, erow in edges_df.iterrows():
            i = int(erow["cell_i"])
            j = int(erow["cell_j"])
            coords = iface.boundary_coords.get((min(i, j), max(i, j)))
            if coords is None:
                # should not happen, but keep robust
                coords = np.zeros((0, 2), dtype=int)
            mask = interface_mask_from_coords(coords, labels.shape, dilate_px=dilate_px)
            feats = compute_interface_features(marker, mask, thr)
            feats_rows.append(feats)

            has = bool(feats.get("occupancy", 0.0) >= min_occ)
            has_list.append(has)

            if jtype == "AJ":
                ajmorph_list.append(infer_ajmorph_label_heuristic(feats))

        feats_df = pd.DataFrame(feats_rows)
        # prefix columns with junction type
        feats_df = feats_df.add_prefix(f"{jtype}_")
        edges_df = pd.concat([edges_df, feats_df], axis=1)

        edges_df[f"has_{jtype}"] = has_list
        edges_df[f"{jtype}_threshold"] = thr

        if jtype == "AJ":
            edges_df["AJ_morph_label"] = ajmorph_list

    # Build typed graph
    junction_types = list(junction_markers.keys())
    G = build_graph(cells_df, edges_df, junction_types)

    # Write outputs
    masks_dir = ensure_dir(out_root / "masks")
    tables_dir = ensure_dir(out_root / "tables")
    graphs_dir = ensure_dir(out_root / "graphs")
    qc_dir = ensure_dir(out_root / "qc")

    label_path = masks_dir / f"{image_id}__labels.tif"
    tifffile.imwrite(label_path, labels.astype(np.int32))

    cells_path = tables_dir / f"{image_id}__cells.csv"
    edges_path = tables_dir / f"{image_id}__edges.csv"
    cells_df.to_csv(cells_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    graph_paths = write_graph_outputs(G, graphs_dir / image_id)
    graph_json_path = graph_paths["json"]

    # QC figures
    qc_seg_path = qc_dir / f"{image_id}__qc_seg.png"
    qc_graph_path = qc_dir / f"{image_id}__qc_graph.png"
    qc_feat_path = qc_dir / f"{image_id}__qc_features.png"

    save_segmentation_qc(arr[display_idx], labels, qc_seg_path, title=image_id)

    save_graph_plot(G, cells_df, qc_graph_path, title=f"Graph: {image_id}", node_x="cx", node_y="cy")

    # pick some feature columns for the histogram figure
    feature_cols = []
    if "AJ" in junction_markers:
        for col in ("AJ_occupancy", "AJ_cluster_density", "AJ_mean_intensity"):
            if col in edges_df.columns:
                feature_cols.append(col)
    if not feature_cols:
        # fallback: histogram contact lengths
        feature_cols = ["contact_px"]
    save_feature_distributions(edges_df, qc_feat_path, title=f"Features: {image_id}")

    # For HTML report, use paths relative to out_root
    def rel(p: Path) -> str:
        return str(p.relative_to(out_root))

    return {
        "image_id": image_id,
        "path": str(path),
        "qc_seg": rel(qc_seg_path),
        "qc_graph": rel(qc_graph_path),
        "qc_feat": rel(qc_feat_path),
        "cells_csv": rel(cells_path),
        "edges_csv": rel(edges_path),
        "graph_json": rel(Path(graph_json_path)),
    }


def run_pipeline(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    out_root = Path(cfg["output_dir"]).resolve()
    ensure_dir(out_root)

    manifest_path = Path(cfg["manifest_csv"]).resolve()
    if not manifest_path.exists():
        raise PipelineError(f"manifest_csv not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "image_id" not in df.columns or "path" not in df.columns:
        raise PipelineError("Manifest must have at least columns: image_id, path")

    max_images = cfg.get("qc", {}).get("max_images")
    if max_images is not None:
        df = df.head(int(max_images))

    # Channel roles are resolved once per run; warns if geometry comes from a junction channel.
    channel_roles = resolve_channel_roles(cfg)
    pixel_size_um = cfg.get("pixel_size_um")
    provenance = {
        "channel_roles": channel_roles,
        "geometry_source": channel_roles["geometry_source"],
        "pixel_size_um": pixel_size_um,
    }

    image_items: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        path = Path(row["path"]).expanduser().resolve()
        item = process_one_image(image_id, path, cfg, out_root)
        item["pixel_size_um"] = pixel_size_um
        image_items.append(item)

    report_path = write_html_report(out_root, cfg.get("study_accession"), image_items)

    # also write a machine-readable summary
    summary_path = out_root / "run_summary.json"
    summary_path.write_text(
        json.dumps({"config": cfg, "provenance": provenance, "items": image_items}, indent=2), encoding="utf-8"
    )

    return report_path
