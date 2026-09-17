"""Dataset loaders yielding (image channels, ground-truth labels) pairs.

Ground truth is a label image. Vertices, incident sets and cyclic order are derived
exactly from it with ``extract_complex`` at evaluation time, so every mask dataset
becomes a complex benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import watershed


@dataclass
class BenchItem:
    image_id: str
    geometry: np.ndarray  # (H,W) float32 channel used for boundary evidence
    labels_gt: np.ndarray  # (H,W) int32
    nuclei: Optional[np.ndarray] = None
    junction: Optional[np.ndarray] = None
    pixel_size_um: Optional[float] = None
    boundary_polarity: str = "bright"  # "bright" | "dark" ridges in the geometry channel
    meta: Dict = field(default_factory=dict)


def fill_gt_slivers(labels: np.ndarray, max_area_px: int = 12) -> np.ndarray:
    """Assign enclosed background components smaller than ``max_area_px`` to the
    neighbouring cell with the most contact.

    Polygon annotations rasterized independently per cell leave 1 to 3 px background
    slivers where cells meet. They are annotation artifacts, not biological gaps, and
    they turn every tricellular vertex into a cell-cell-gap vertex, which makes vertex
    metrics meaningless. Filling them restores the shared crack geometry.
    """
    from ..infer.decoder import _fill_small_background

    return _fill_small_background(np.asarray(labels).astype(np.int32), max_area_px)


def _read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        a = tifffile.imread(str(path))
    else:
        from imageio.v3 import imread

        a = imread(str(path))
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1) if a.shape[-1] in (3, 4) else a[0]
    return a.astype(np.float32)


# ---------------------------------------------------------------- cornea cells
def semantic_to_instance_watershed(sem: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """Cornea labels are semantic (1 interior, 2 border, 3 background) and the interior
    class is not closed by the border class, so instances are DERIVED: distance-transform
    peaks (min_distance px apart) seed a watershed through the non-background region.
    This is the construction used by the legacy benchmark (scripts/benchmark_cornea_final.py),
    so numbers stay comparable; it is not expert instance truth."""
    from skimage.feature import peak_local_max

    border = sem == 2
    background = sem == 3
    dist = ndi.distance_transform_edt(~border & ~background)
    peaks = peak_local_max(dist, min_distance=min_distance, exclude_border=False)
    markers = np.zeros(sem.shape, dtype=np.int32)
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    inst = watershed(-dist, markers, mask=~background)
    return inst.astype(np.int32)


def load_cornea(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    labels_dir = root / "labels"
    images_dir = root / "dataset"
    paths = sorted(labels_dir.glob("*.tif"))
    if max_items:
        paths = paths[:max_items]
    for lp in paths:
        ip = images_dir / lp.name
        if not ip.exists():
            continue
        sem = tifffile.imread(str(lp))
        img = _read_gray(ip)
        yield BenchItem(
            image_id=lp.stem,
            geometry=img,
            labels_gt=semantic_to_instance_watershed(sem),
            boundary_polarity="dark",
            # The semantic border class does not close cells (the interior class is one
            # connected blob), so the derived instances over-segment along medial axes.
            # Face/adjacency/vertex metrics against this GT are NOT meaningful; boundary
            # F1 against the border class is. Kept for continuity with the legacy report.
            meta={
                "dataset": "cornea",
                "modality": "specular",
                "gt_kind": "derived_from_semantic_oversegmented",
                "instance_metrics_reliable": False,
            },
        )


# -------------------------------------------------------------------- NuInsSeg
def load_nuinsseg(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """NuInsSeg layout: <tissue>/tissue images/*.png and <tissue>/label masks/*.tif."""
    items = []
    for tissue_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        img_dir = tissue_dir / "tissue images"
        lab_dir = tissue_dir / "label masks"
        if not img_dir.exists() or not lab_dir.exists():
            continue
        for lp in sorted(lab_dir.glob("*.tif")):
            ip = img_dir / (lp.stem + ".png")
            if ip.exists():
                items.append((tissue_dir.name, ip, lp))
    if max_items:
        items = items[:max_items]
    for tissue, ip, lp in items:
        yield BenchItem(
            image_id=f"{tissue}/{lp.stem}",
            geometry=_read_gray(ip),
            labels_gt=np.asarray(tifffile.imread(str(lp))).astype(np.int32),
            boundary_polarity="dark",
            meta={"dataset": "nuinsseg", "tissue": tissue, "modality": "H&E"},
        )


# --------------------------------------------------------------------- mCellSeg
def load_mcellseg(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """mCellSeg (Kaggle). Pairs every image with a mask of the same stem found under a
    sibling folder whose name contains 'mask' or 'label'."""
    imgs: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".png", ".tif", ".tiff", ".jpg"):
            continue
        parent = p.parent.name.lower()
        if "mask" in parent or "label" in parent or "gt" in parent:
            masks[p.stem] = p
        else:
            imgs[p.stem] = p
    stems = sorted(set(imgs) & set(masks))
    if max_items:
        stems = stems[:max_items]
    for s in stems:
        lab = np.asarray(_read_gray(masks[s])).astype(np.int64)
        if lab.max() <= 1:  # binary mask: instances by connected components
            lab = cc_label(lab > 0, connectivity=1)
        yield BenchItem(
            image_id=s,
            geometry=_read_gray(imgs[s]),
            labels_gt=fill_gt_slivers(lab.astype(np.int32), 12),
            boundary_polarity="auto",
            meta={
                "dataset": "mcellseg",
                "image_path": str(imgs[s]),
                "mask_path": str(masks[s]),
                "gt_sliver_fill_px": 12,
            },
        )


# ---------------------------------------------------------------------- LIVECell
def load_livecell(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """LIVECell COCO validation annotations plus images under root/images/."""
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils
    except Exception as e:  # pragma: no cover
        raise ImportError("pycocotools is required for LIVECell") from e
    ann = root / "livecell_coco_val.json"
    coco = COCO(str(ann))
    img_ids = sorted(coco.imgs.keys())
    n = 0
    for iid in img_ids:
        info = coco.imgs[iid]
        ip = None
        for cand in (root / "images" / info["file_name"], root / "images" / "livecell_test_images" / info["file_name"]):
            if cand.exists():
                ip = cand
                break
        if ip is None:
            matches = list((root / "images").rglob(info["file_name"])) if (root / "images").exists() else []
            if matches:
                ip = matches[0]
        if ip is None:
            continue
        H, W = info["height"], info["width"]
        lab = np.zeros((H, W), dtype=np.int32)
        for k, a in enumerate(coco.loadAnns(coco.getAnnIds(imgIds=iid)), start=1):
            seg = a["segmentation"]
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, H, W)
                m = mask_utils.decode(mask_utils.merge(rles))
            else:
                m = mask_utils.decode(seg)
            lab[m.astype(bool) & (lab == 0)] = k
        yield BenchItem(
            image_id=Path(info["file_name"]).stem,
            geometry=_read_gray(ip),
            labels_gt=fill_gt_slivers(lab, 12),
            boundary_polarity="auto",
            meta={"dataset": "livecell", "modality": "phase-contrast", "gt_sliver_fill_px": 12},
        )
        n += 1
        if max_items and n >= max_items:
            break


# ----------------------------------------------------------------------- synth
def load_synth(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """Tiles written by pimorph.synth.targets.make_dataset (npz with channels and labels)."""
    paths = sorted(root.glob("tile_*.npz"))
    if max_items:
        paths = paths[:max_items]
    for p in paths:
        d = np.load(p)
        yield BenchItem(
            image_id=p.stem,
            geometry=d["junction"].astype(np.float32),
            labels_gt=d["labels"].astype(np.int32),
            nuclei=d["nuclei"].astype(np.float32) if "nuclei" in d else None,
            junction=d["junction"].astype(np.float32),
            boundary_polarity="bright",
            meta={"dataset": "synth", "membrane_available": "membrane" in d},
        )


LOADERS: Dict[str, Callable[[Path, Optional[int]], Iterator[BenchItem]]] = {
    "cornea": load_cornea,
    "nuinsseg": load_nuinsseg,
    "mcellseg": load_mcellseg,
    "livecell": load_livecell,
    "synth": load_synth,
}

DEFAULT_ROOTS = {
    "cornea": Path("data/cornea_cells"),
    "nuinsseg": Path("data/NuInsSeg"),
    "mcellseg": Path("data/mcellseg"),
    "livecell": Path("data/LIVECell"),
    "synth": Path("data/tiles/synth_val"),
}


def list_datasets() -> List[str]:
    return sorted(LOADERS)


def load_dataset(name: str, root: Optional[Path | str] = None, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {list_datasets()}")
    r = Path(root) if root is not None else DEFAULT_ROOTS[name]
    if not r.exists():
        raise FileNotFoundError(f"dataset root {r} does not exist")
    return LOADERS[name](r, max_items)
