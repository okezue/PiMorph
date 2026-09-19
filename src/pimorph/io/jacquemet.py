"""Zenodo 10611092 (Jacquemet lab): PECAM-1 stained HUVEC monolayers.

``pix2pix_HUVEC_juctions_dataset.zip`` (6.15 GB) is the training material of a
brightfield-to-junction pix2pix model. Only two folders hold real fluorescence:
``Training_data/230419_all_pecam/`` (484 fields) and
``Quality_control_data/230427_QC_all_pecam/`` (11 held-out fields). Their ``_BF``
twins are the brightfield inputs and ``230502_pix2pix_...(Checkpoint_60)/`` holds
model weights plus ``real_A`` / ``real_B`` / ``fake_B`` PNG exports of the QC
fields, so it adds no real image. The real PECAM-1 TIFFs are unpacked flat into
``<root>/training/NNN.tif`` and ``<root>/qc/NNN.tif`` (ids overlap between the two
folders, so the split is kept as a directory).

Each file is a single-channel 1022 x 1024 uint16 ImageJ TIFF (12-bit range,
``Labels = c:3/4`` or ``c:2/3``: the PECAM-1 channel of the source .nd2) with
``unit = micron`` and ``XResolution = 1.539376`` px/um, i.e. ``PIXEL_SIZE_UM`` =
0.6496 um/px. There are no masks; the set is a real endothelial junction domain for
self-consistency and pseudo-label work.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tifffile

from .images import pixel_size_from_tiff

PathLike = Union[str, Path]

ZENODO_RECORD = "10611092"
ARCHIVE_NAME = "pix2pix_HUVEC_juctions_dataset.zip"
REAL_PECAM_FOLDERS = {
    "training": "pix2pix_HUVEC_juctions_dataset/Training_data/230419_all_pecam",
    "qc": "pix2pix_HUVEC_juctions_dataset/Quality_control_data/230427_QC_all_pecam",
}
SPLITS = tuple(REAL_PECAM_FOLDERS)
DATASET = "Zenodo-10611092"
CHANNEL_NAME = "PECAM-1"
PIXEL_SIZE_UM = 1.0 / 1.539376
PIXEL_SIZE_SOURCE = "ImageJ TIFF resolution tag (unit micron)"


def list_pecam_fields(root: PathLike, splits: Optional[List[str]] = None) -> List[Path]:
    """Real PECAM-1 TIFFs under ``root`` (``training/`` then ``qc/``), sorted by split and id."""
    root = Path(root)
    out: List[Path] = []
    for split in splits or SPLITS:
        d = root / split
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in (".tif", ".tiff")))
    return out


def _junction_plane(arr: np.ndarray) -> np.ndarray:
    """2-D plane with the junction signal: RGB or multi-plane files keep the plane
    with the largest 99th percentile (the pseudocolour channel that carries PECAM-1)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
            arr = np.moveaxis(arr, -1, 0)
        k = int(np.argmax([np.percentile(a, 99) for a in arr]))
        return arr[k]
    raise ValueError(f"unsupported image shape {arr.shape}")


def load_pecam_field(path: PathLike) -> np.ndarray:
    """One PECAM-1 field as a 2-D float32 image (raw counts, no normalisation)."""
    return _junction_plane(tifffile.imread(str(path))).astype(np.float32)


def field_pixel_size_um(path: PathLike) -> Optional[float]:
    with tifffile.TiffFile(str(path)) as tf:
        px, _ = pixel_size_from_tiff(tf)
    return px


def field_id(path: PathLike) -> str:
    p = Path(path)
    return f"pecam_{p.parent.name}_{p.stem}"


def write_pecam_manifest(root: PathLike, out_csv: PathLike, manifest_root: Optional[PathLike] = None) -> pd.DataFrame:
    """Project manifest (``image_id,path,dataset,channel_1`` plus ``split`` and
    ``pixel_size_um``) for every real PECAM-1 field. Paths are relative to
    ``manifest_root`` (default: the current directory) so ``parse_manifest(..., root=".")``
    resolves them."""
    root = Path(root)
    base = Path(manifest_root) if manifest_root is not None else Path.cwd()
    rows = []
    for p in list_pecam_fields(root):
        try:
            rel = p.resolve().relative_to(base.resolve())
        except ValueError:
            rel = p
        px = field_pixel_size_um(p)
        rows.append(
            {
                "image_id": field_id(p),
                "path": str(rel),
                "dataset": DATASET,
                "channel_1": CHANNEL_NAME,
                "split": p.parent.name,
                "pixel_size_um": PIXEL_SIZE_UM if px is None else px,
            }
        )
    df = pd.DataFrame(rows, columns=["image_id", "path", "dataset", "channel_1", "split", "pixel_size_um"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


__all__ = [
    "CHANNEL_NAME",
    "DATASET",
    "PIXEL_SIZE_UM",
    "REAL_PECAM_FOLDERS",
    "SPLITS",
    "field_id",
    "field_pixel_size_um",
    "list_pecam_fields",
    "load_pecam_field",
    "write_pecam_manifest",
]
