"""Dataset loaders yielding (image channels, ground-truth labels) pairs.

Ground truth is a label image. Vertices, incident sets and cyclic order are derived
exactly from it with ``extract_complex`` at evaluation time, so every mask dataset
becomes a complex benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
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
def load_mcellseg(root: Path, max_items: Optional[int] = None, cell_line: Optional[str] = None) -> Iterator[BenchItem]:
    """mCellSeg (Alam et al. 2026; Zenodo 10.5281/zenodo.20174259, Kaggle tukunzil/...).

    200 expert-annotated DIC / transmitted-light images (100 HUVEC, 100 HEK-293T) with
    uint16 instance masks. Layout: root/labeled/images/<stem>.tif (RGB uint8 with
    identical channels) and root/labeled/masks/<stem>_mask.tif. The cell line is taken
    from the file name (HUVEC when present, else HEK293T); ``cell_line`` filters
    (env PIMORPH_MCELLSEG_LINE does the same for the CLI).
    """
    cell_line = cell_line or os.environ.get("PIMORPH_MCELLSEG_LINE") or None
    img_dir = root / "labeled" / "images"
    mask_dir = root / "labeled" / "masks"
    if not img_dir.exists():
        img_dir, mask_dir = root, root
    imgs = sorted(p for p in img_dir.rglob("*.tif") if "_mask" not in p.name and "masks" not in p.parts)
    masks: Dict[str, Path] = {p.name[: -len("_mask.tif")]: p for p in mask_dir.rglob("*_mask.tif")}
    items = [(ip, masks[ip.stem]) for ip in imgs if ip.stem in masks]
    if cell_line:
        want_huvec = cell_line.upper() == "HUVEC"
        items = [(ip, mp) for ip, mp in items if ("HUVEC" in ip.name) == want_huvec]
    ids = _active_id_list()
    if ids is not None:
        items = [(ip, mp) for ip, mp in items if ip.stem in ids]
    if max_items and len(items) > max_items:
        step = len(items) / max_items
        items = [items[int(i * step)] for i in range(max_items)]
    for ip, mp in items:
        lab = np.asarray(tifffile.imread(str(mp))).astype(np.int64)
        if lab.ndim == 3:
            lab = lab[..., 0]
        if lab.max() <= 1:
            lab = cc_label(lab > 0, connectivity=1)
        yield BenchItem(
            image_id=ip.stem,
            geometry=_read_gray(ip),
            labels_gt=fill_gt_slivers(lab.astype(np.int32), 12),
            boundary_polarity="auto",
            meta={
                "dataset": "mcellseg",
                "cell_line": "HUVEC" if "HUVEC" in ip.name else "HEK293T",
                "modality": "DIC",
                "gt_kind": "expert_instance",
                "gt_sliver_fill_px": 12,
            },
        )


# ---------------------------------------------- Human aortic endothelial cells (HAEC)
def haec_semantic_to_instance(gt: np.ndarray) -> np.ndarray:
    """HAEC ground truth is categorical: 0 background, 1 cell border, 2 cell body,
    3 nucleus. Instances are the body+nucleus components flooded through the 1 px
    border class, so touching cells share a crack edge."""
    body = (gt == 2) | (gt == 3)
    border = gt == 1
    markers = cc_label(body, connectivity=1)
    dist = ndi.distance_transform_edt(~border)
    inst = watershed(dist, markers, mask=gt > 0)
    return inst.astype(np.int32)


def load_haec(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """Human aortic endothelial cell ground truth (Harrison, Wu, Fang, Huang; Zenodo
    4898011, CC-BY 4.0). 434 fields, 1200x1200 16-bit: GFP_original/NNNN.tif is a
    cytoplasmic Laconic-GFP reporter (an independent whole-cell geometry channel),
    Hoechst_1/Pos<NNNN-1>/img_*.tif the nuclei, gtruth_uint8/NNNN.png the 4-class
    ground truth. Instances are derived by ``haec_semantic_to_instance``.
    """
    from imageio.v3 import imread

    gfp_dir = root / "GFP_original"
    gt_dir = root / "gtruth_uint8"
    hoechst_dir = root / "Hoechst_1"
    ids = sorted(int(p.stem) for p in gfp_dir.glob("*.tif") if p.stem.isdigit())
    keep = _active_id_list()
    if keep is not None:
        ids = [n for n in ids if f"{n:04d}" in keep or str(n) in keep]
    if max_items and len(ids) > max_items:
        step = len(ids) / max_items
        ids = [ids[int(i * step)] for i in range(max_items)]
    for n in ids:
        gtp = gt_dir / f"{n:04d}.png"
        if not gtp.exists():
            continue
        gt = np.asarray(imread(str(gtp)))
        if gt.ndim == 3:
            gt = gt[..., 0]
        geometry = np.asarray(tifffile.imread(str(gfp_dir / f"{n:04d}.tif"))).astype(np.float32)
        nuclei = None
        pos = hoechst_dir / f"Pos{n - 1}"
        if pos.exists():
            tifs = sorted(pos.glob("*.tif"))
            if tifs:
                nuclei = np.asarray(tifffile.imread(str(tifs[0]))).astype(np.float32)
                if nuclei.ndim == 3:
                    nuclei = nuclei[0]
        yield BenchItem(
            image_id=f"{n:04d}",
            geometry=geometry,
            labels_gt=haec_semantic_to_instance(gt),
            nuclei=nuclei,
            # cytoplasmic reporter: cells are bright, the borders between them are dark seams
            boundary_polarity="dark",
            meta={
                "dataset": "haec",
                "modality": "fluorescence",
                "geometry_channel": "cytoplasmic GFP (Laconic)",
                "gt_kind": "derived_from_4class_semantic",
                "confluent": False,
            },
        )


# ---------------------------------------------------------------------- LIVECell
LIVECELL_SPLIT = os.environ.get("PIMORPH_LIVECELL_SPLIT", "val")


def load_livecell(root: Path, max_items: Optional[int] = None, split: Optional[str] = None) -> Iterator[BenchItem]:
    """LIVECell COCO annotations (split from PIMORPH_LIVECELL_SPLIT: train | val | test,
    default val) plus images under root/images/. The 8 cell lines are recorded in meta."""
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils
    except Exception as e:  # pragma: no cover
        raise ImportError("pycocotools is required for LIVECell") from e
    split = split or LIVECELL_SPLIT
    ann = root / f"livecell_coco_{split}.json"
    coco = COCO(str(ann))
    img_ids = sorted(coco.imgs.keys())
    if max_items and len(img_ids) > max_items:
        # deterministic stratified-ish subsample: every k-th image keeps cell lines mixed
        step = len(img_ids) / max_items
        img_ids = [img_ids[int(i * step)] for i in range(max_items)]
    # index the image tree once (the archive nests images two levels deep)
    index: Dict[str, Path] = {p.name: p for p in (root / "images").rglob("*.tif")} if (root / "images").exists() else {}
    n = 0
    for iid in img_ids:
        info = coco.imgs[iid]
        ip = index.get(Path(info["file_name"]).name)
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
        stem = Path(info["file_name"]).stem
        yield BenchItem(
            image_id=stem,
            geometry=_read_gray(ip),
            labels_gt=fill_gt_slivers(lab, 12),
            boundary_polarity="auto",
            meta={
                "dataset": "livecell",
                "split": split,
                "cell_line": stem.split("_")[0],
                "modality": "phase-contrast",
                "gt_sliver_fill_px": 12,
            },
        )
        n += 1
        if max_items and n >= max_items:
            break


# ------------------------------------------------------------ NeurIPS CellSeg 2022
def load_neurips_cellseg(root: Path, max_items: Optional[int] = None) -> Iterator[BenchItem]:
    """NeurIPS 2022 Cell Segmentation Challenge (Zenodo 10719375).

    Layout after unzip: root/Training-labeled/images/*.{png,tif,tiff,bmp} and
    root/Training-labeled/labels/<stem>_label.tiff (instance ids); Tuning/ has the same
    layout. Modalities (brightfield, fluorescence, phase contrast, DIC) are mixed; the
    image is converted to gray by channel mean when RGB.
    """
    items = []
    for sub in ("Training-labeled", "Tuning"):
        img_dir = root / sub / "images"
        lab_dir = root / sub / "labels"
        if not img_dir.exists():
            continue
        for ip in sorted(img_dir.iterdir()):
            if ip.suffix.lower() not in (".png", ".tif", ".tiff", ".bmp", ".jpg"):
                continue
            cands = [
                lab_dir / f"{ip.stem}_label.tiff",
                lab_dir / f"{ip.stem}_label.tif",
                lab_dir / f"{ip.stem}_label.png",
            ]
            lp = next((c for c in cands if c.exists()), None)
            if lp is not None:
                items.append((sub, ip, lp))
    if max_items and len(items) > max_items:
        step = len(items) / max_items
        items = [items[int(i * step)] for i in range(max_items)]
    for sub, ip, lp in items:
        lab = np.asarray(tifffile.imread(str(lp)) if lp.suffix.lower() in (".tif", ".tiff") else _read_gray(lp))
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab = lab.astype(np.int64)
        if lab.max() <= 1:
            lab = cc_label(lab > 0, connectivity=1)
        yield BenchItem(
            image_id=f"{sub}/{ip.stem}",
            geometry=_read_gray(ip),
            labels_gt=fill_gt_slivers(lab.astype(np.int32), 12),
            boundary_polarity="auto",
            meta={"dataset": "neurips_cellseg", "subset": sub, "modality": "mixed", "gt_sliver_fill_px": 12},
        )


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
    "neurips_cellseg": load_neurips_cellseg,
    "haec": load_haec,
    "synth": load_synth,
}

DEFAULT_ROOTS = {
    "cornea": Path("data/cornea_cells"),
    "nuinsseg": Path("data/NuInsSeg"),
    "mcellseg": Path("data/mcellseg"),
    "livecell": Path("data/LIVECell"),
    "neurips_cellseg": Path("data/neurips_cellseg"),
    "haec": Path("data/haec_gt"),
    "synth": Path("data/tiles/synth_val"),
}


def list_datasets() -> List[str]:
    return sorted(LOADERS)


_ID_LIST_OVERRIDE: Optional[str] = None


def _active_id_list() -> Optional[set]:
    """Id set from the load_dataset call (or env PIMORPH_ID_LIST); loaders that support
    it filter before any expensive decoding."""
    return load_id_list(_ID_LIST_OVERRIDE or os.environ.get("PIMORPH_ID_LIST"))


def load_id_list(spec: Optional[str]) -> Optional[set]:
    """``path.json:key`` -> set of image ids (as strings) from that JSON list, or None.
    Used to restrict a dataset to a train or test split."""
    if not spec:
        return None
    import json

    path, _, key = spec.partition(":")
    data = json.load(open(path))
    vals = data[key] if key else data
    out = set()
    for v in vals:
        out.add(str(v))
        if isinstance(v, int):
            out.add(f"{v:04d}")
    return out


def load_dataset(
    name: str,
    root: Optional[Path | str] = None,
    max_items: Optional[int] = None,
    id_list: Optional[str] = None,
) -> Iterator[BenchItem]:
    """Iterate a dataset. ``id_list`` (or env PIMORPH_ID_LIST) restricts to the given
    image ids; ``max_items`` then applies to the filtered stream."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {list_datasets()}")
    r = Path(root) if root is not None else DEFAULT_ROOTS[name]
    if not r.exists():
        raise FileNotFoundError(f"dataset root {r} does not exist")
    global _ID_LIST_OVERRIDE
    ids = load_id_list(id_list or os.environ.get("PIMORPH_ID_LIST"))
    if ids is None:
        yield from LOADERS[name](r, max_items)
        return
    _ID_LIST_OVERRIDE = id_list or os.environ.get("PIMORPH_ID_LIST")
    try:
        n = 0
        # loaders with native id filtering (mcellseg, haec) already restricted the stream;
        # the check below is a no-op for them and the fallback for the others
        for item in LOADERS[name](r, None):
            if item.image_id not in ids and item.image_id.split("/")[-1] not in ids:
                continue
            yield item
            n += 1
            if max_items and n >= max_items:
                break
    finally:
        _ID_LIST_OVERRIDE = None
