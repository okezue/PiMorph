"""NIH-NEI retinal pigment epithelium (RPE) tight-junction monolayer set.

Source: Bharti lab RPE Pipeline (Ortolan et al. 2022 PNAS), Mask R-CNN training
data. Full set ``RpeMapTrainingData.zip`` on figshare+ 28832501 (4.5 GB, CC0):
``DataRoot/RPE_Training/Mask_RCNN/Actin`` holds 18 confocal stacks x 27 z x 4 tiles
as PNG (130,201 polygons, 12 stacks named ``ZO1``). Demo subset
``RPE_Training_Actin_RGB.zip`` from the NIH-NEI/RPE_Segmentation GitHub releases:
3 of those stacks as TIF with identical pixels. Tiles are 768x768 RGB uint8
composites of iPSC-derived RPE monolayers: red nuclei, green cell borders, blue
third stain named by the stack token (TOM, LAMP1, SEC, TUB, LMNB, ZO1). The green
border channel is the upstream ``Actin`` target (RPE Pipeline: phalloidin or ZO-1
borders; several stacks show basal stress fibres in green). File names are
``<stack>-<z>-<tile>.png``; one VIA JSON per stack holds manual polygon outlines with
``region_attributes.cell`` (cell id) and ``frame`` (z).

Annotators outlined each cell only on the z-frames where it is in focus, so a single
frame covers a fraction of the tile. The 2-D truth used here is the per-cell union
of its polygons over z (later frames overwrite earlier ones where they overlap),
which tiles 82 to 87 percent of a tile with touching neighbours; background slivers
under 12 px left between independently drawn polygons are filled as for FlyWing. The
geometry image is the green channel of the frame with the most polygons for that tile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_polygon
from skimage.measure import label as cc_label

from .datasets import BenchItem, _active_id_list, fill_gt_slivers

MIN_CELL_AREA_PX = 60
ROI_DILATION_PX = 3
SLIVER_MAX_PX = 12


def _split_name(filename: str) -> Tuple[str, int, str]:
    """``<stack>-<z>-<tile>.tif`` -> (stack, z, tile)."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("-")
    return "-".join(parts[:-2]), int(parts[-2]), parts[-1]


def group_via_polygons(via: Dict) -> Dict[str, Dict[str, List[Tuple[int, int, np.ndarray, np.ndarray]]]]:
    """VIA JSON -> {stack: {tile: [(z, cell_id, xs, ys), ...]}}.

    ``frame`` is taken from the region attributes when present, else from the file
    name; ``cell`` ids are per stack.
    """
    out: Dict[str, Dict[str, List[Tuple[int, int, np.ndarray, np.ndarray]]]] = {}
    for entry in via.values():
        stack, z_name, tile = _split_name(entry["filename"])
        regions = entry["regions"]
        if isinstance(regions, dict):
            regions = list(regions.values())
        for r in regions:
            sa = r["shape_attributes"]
            if sa.get("name", "polygon") not in ("polygon", "polyline"):
                continue
            ra = r.get("region_attributes", {})
            z = int(ra.get("frame", z_name))
            cell = int(ra["cell"])
            xs = np.asarray(sa["all_points_x"], dtype=np.float64)
            ys = np.asarray(sa["all_points_y"], dtype=np.float64)
            if xs.size < 3:
                continue
            out.setdefault(stack, {}).setdefault(tile, []).append((z, cell, xs, ys))
    return out


def rasterize_union_over_z(
    polys: List[Tuple[int, int, np.ndarray, np.ndarray]],
    shape: Tuple[int, int],
    min_area_px: int = MIN_CELL_AREA_PX,
    roi_dilation_px: int = ROI_DILATION_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Union each cell's polygons over z into one int32 instance map plus the ROI.

    Polygons are painted in increasing z, so a later frame overwrites an earlier one
    where two cells' outlines overlap. The painted cell ids are then relabelled with
    4-connectivity and fragments under ``min_area_px`` are dropped. The ROI is the
    union of every painted pixel dilated by ``roi_dilation_px``.
    """
    canvas = np.zeros(shape, dtype=np.int32)
    for z, cell, xs, ys in sorted(polys, key=lambda p: p[0]):
        rr, cc = draw_polygon(ys, xs, shape=shape)
        canvas[rr, cc] = cell
    painted = canvas > 0
    lab = cc_label(canvas, connectivity=1, background=0).astype(np.int32)
    if lab.max() > 0:
        areas = np.bincount(lab.ravel())
        small = areas < min_area_px
        small[0] = False
        lab[small[lab]] = 0
        lab = cc_label(lab, connectivity=1, background=0).astype(np.int32)
    roi = painted
    if roi_dilation_px > 0:
        roi = ndi.binary_dilation(painted, iterations=roi_dilation_px)
    return lab, roi


def _find_base(root: Path) -> Path:
    """Prefer the full figshare set (18 stacks, PNG) over the GitHub demo (3 stacks, TIF)."""
    for cand in (
        root / "full" / "DataRoot" / "RPE_Training" / "Mask_RCNN" / "Actin",
        root / "RPE_Training" / "Mask_RCNN" / "Actin",
        root / "Mask_RCNN" / "Actin",
        root,
    ):
        if any(cand.glob("*_annotations_via.json")):
            return cand
    raise FileNotFoundError(f"no *_annotations_via.json under {root}")


def _read_tile(base: Path, stack: str, z: int, tile: str) -> np.ndarray:
    for ext in (".png", ".tif"):
        p = base / f"{stack}-{z:03d}-{tile}{ext}"
        if p.exists():
            if ext == ".tif":
                return np.asarray(tifffile.imread(str(p)))
            from imageio.v3 import imread

            return np.asarray(imread(str(p)))
    raise FileNotFoundError(f"tile {stack}-{z:03d}-{tile} not found under {base}")


def load_rpe_zo1(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """One item per (stack, tile): green border channel of the most-annotated z-frame,
    red nuclei, instance truth from the per-cell union of VIA polygons over z, ROI
    limited to the annotated area. Ids are ``<stack>_<tile>``."""
    base = _find_base(Path(root))
    keep = _active_id_list()
    items: List[Tuple[str, str, List[Tuple[int, int, np.ndarray, np.ndarray]]]] = []
    for jp in sorted(base.glob("*_annotations_via.json")):
        grouped = group_via_polygons(json.load(open(jp)))
        for stack in sorted(grouped):
            for tile in sorted(grouped[stack]):
                fid = f"{stack}_{tile}"
                if keep is not None and fid not in keep:
                    continue
                items.append((stack, tile, grouped[stack][tile]))
    if max_items:
        items = items[:max_items]
    for stack, tile, polys in items:
        counts: Dict[int, int] = {}
        for z, *_ in polys:
            counts[z] = counts.get(z, 0) + 1
        z_best = max(sorted(counts), key=counts.get)
        img = _read_tile(base, stack, z_best, tile)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        shape = img.shape[:2]
        lab, roi = rasterize_union_over_z(polys, shape)
        # independent per-cell polygons leave 1 to 3 px background slivers at shared
        # borders; fill them as for FlyWing so vertices are tricellular
        lab = fill_gt_slivers(lab, SLIVER_MAX_PX)
        green = img[..., 1].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
        red = img[..., 0].astype(np.float32) if img.ndim == 3 else None
        yield BenchItem(
            image_id=f"{stack}_{tile}",
            geometry=green,
            labels_gt=lab,
            nuclei=red,
            junction=green,
            boundary_polarity="bright",
            roi=roi,
            meta={
                "dataset": "rpe_zo1",
                "modality": "confocal_fluorescence",
                "geometry_channel": "green border channel (upstream target Actin: phalloidin or ZO-1)",
                "stack": stack,
                "tile": tile,
                "z_frame": int(z_best),
                "n_z_frames_annotated": len(counts),
                "n_polygons": len(polys),
                "gt_kind": "manual_via_polygons_union_over_z",
                "confluent": True,
                "roi_fraction": float(roi.mean()),
            },
        )
