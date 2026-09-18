"""Loader for Cell Tracking Challenge 2-D training sets.

Layout of ``<root>/<NAME>``: ``01/t000.tif`` images, ``01_GT/SEG/man_seg000.tif``
(sparse frames, full masks), ``01_GT/TRA/man_track000.tif`` (every frame, label =
track id; for most datasets these are markers, not full masks) with
``man_track.txt`` rows ``id start end parent``, and ``01_ST/SEG/man_seg000.tif``
(silver truth: dense computer-generated masks for every frame).
``tracked_dense_labels`` combines the silver-truth masks with the gold tracking
markers into a dense stack whose labels are the ground-truth track ids.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed

CTC_URL = "https://data.celltrackingchallenge.net/training-datasets/{name}.zip"


def download_ctc(name: str, root: str = "data/ctc") -> Path:
    """Fetch and unzip ``<name>.zip`` into ``root`` unless the folder already exists."""
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    target = root_p / name
    if target.is_dir():
        return target
    zip_path = root_p / f"{name}.zip"
    if not zip_path.exists():
        subprocess.run(["curl", "-sSL", "-o", str(zip_path), CTC_URL.format(name=name)], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root_p)
    if not target.is_dir():
        raise FileNotFoundError(f"{zip_path} did not unpack to {target}")
    return target


def read_man_track(path: Path) -> Dict[int, Tuple[int, int, int]]:
    """``man_track.txt`` -> {track id: (start frame, end frame, parent id or 0)}."""
    out: Dict[int, Tuple[int, int, int]] = {}
    for line in Path(path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cid, start, end, parent = (int(x) for x in parts[:4])
        out[cid] = (start, end, parent)
    return out


def _frame_index(path: Path, prefix: str) -> int:
    return int(path.stem[len(prefix) :])


def load_ctc(
    name: str,
    seq: str = "01",
    root: str = "data/ctc",
    download: bool = True,
    load_images: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Load one CTC training sequence.

    Returns images (paths, arrays only with ``load_images``), ``tra_labels`` (list
    per frame), ``lineage`` {id: (start, end, parent)}, ``seg_frames`` and
    ``seg_labels`` (sparse gold masks), and ``st_labels`` (silver truth, None when
    the sequence has none).
    """
    base = Path(root) / name
    if not base.is_dir():
        if not download:
            raise FileNotFoundError(base)
        base = download_ctc(name, root)
    img_dir = base / seq
    tra_dir = base / f"{seq}_GT" / "TRA"
    seg_dir = base / f"{seq}_GT" / "SEG"
    st_dir = base / f"{seq}_ST" / "SEG"
    image_paths = sorted(img_dir.glob("t*.tif"))
    tra_paths = sorted(tra_dir.glob("man_track*.tif"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
        tra_paths = tra_paths[:max_frames]
    n = len(tra_paths)
    tra_labels = [tifffile.imread(p).astype(np.int64) for p in tra_paths]
    lineage = read_man_track(tra_dir / "man_track.txt")
    seg_labels: Dict[int, np.ndarray] = {}
    for p in sorted(seg_dir.glob("man_seg*.tif")):
        k = _frame_index(p, "man_seg")
        if k < n:
            seg_labels[k] = tifffile.imread(p).astype(np.int64)
    st_labels: Optional[List[np.ndarray]] = None
    if st_dir.is_dir():
        st_paths = sorted(st_dir.glob("man_seg*.tif"))[:n]
        if len(st_paths) == n:
            st_labels = [tifffile.imread(p).astype(np.int64) for p in st_paths]
    return {
        "name": name,
        "seq": seq,
        "root": str(base),
        "n_frames": n,
        "image_paths": [str(p) for p in image_paths],
        "images": [tifffile.imread(p) for p in image_paths] if load_images else None,
        "tra_paths": [str(p) for p in tra_paths],
        "tra_labels": tra_labels,
        "lineage": lineage,
        "seg_frames": sorted(seg_labels),
        "seg_labels": seg_labels,
        "st_labels": st_labels,
    }


def tracked_dense_labels(
    tra_labels: Sequence[np.ndarray],
    dense_labels: Sequence[np.ndarray],
    fresh_offset: int = 100_000,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Relabel dense masks with the track ids of the markers they contain.

    A mask holding one marker takes that id; a mask holding several markers is
    split between them by a distance-transform watershed; a mask without marker
    keeps a fresh id ``fresh_offset + frame * 1000 + k`` (an untracked cell).
    Markers without any mask are reported in the stats.
    """
    out: List[np.ndarray] = []
    stats = {"n_masks": 0, "n_single": 0, "n_split": 0, "n_no_marker": 0, "n_marker_without_mask": 0}
    for t, (tra, dense) in enumerate(zip(tra_labels, dense_labels)):
        tra = np.asarray(tra).astype(np.int64)
        dense = np.asarray(dense).astype(np.int64)
        res = np.zeros_like(dense)
        ids = np.unique(dense[dense > 0])
        stats["n_masks"] += int(ids.size)
        k = 0
        covered = set()
        for lab in ids.tolist():
            m = dense == lab
            marks = np.unique(tra[m])
            marks = marks[marks > 0]
            if marks.size == 1:
                res[m] = int(marks[0])
                covered.add(int(marks[0]))
                stats["n_single"] += 1
            elif marks.size == 0:
                k += 1
                res[m] = fresh_offset + t * 1000 + k
                stats["n_no_marker"] += 1
            else:
                markers = np.where(m, tra, 0)
                markers[~np.isin(markers, marks)] = 0
                dist = ndi.distance_transform_edt(markers == 0)
                ws = watershed(dist, markers=markers, mask=m)
                res[m] = ws[m]
                covered.update(int(x) for x in marks.tolist())
                stats["n_split"] += 1
        present = set(np.unique(tra[tra > 0]).tolist())
        stats["n_marker_without_mask"] += len(present - covered)
        out.append(res)
    return out, stats


def ctc_divisions(lineage: Dict[int, Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    """Divisions (parent, children, frame of the children's first appearance)."""
    from .metrics import divisions_from_lineage

    return divisions_from_lineage(lineage)


__all__ = ["CTC_URL", "ctc_divisions", "download_ctc", "load_ctc", "read_man_track", "tracked_dense_labels"]
