"""Loader for EpiCure curated epithelium movies (Zenodo 20607705, Letort et al. 2026).

``MovieEpitheliumFigure5.zip`` unpacks to ``data_generalisations/movieN/`` with the
raw movie ``<name>.tif`` (T[C]YX, ImageJ), ``epics_init/`` (automatic segmentation),
``epics_corrected/`` (curated in EpiCure) and, for movie4, ``epics_correctedWithTA/``.
Each folder holds ``<name>_labels.tif`` (float32 TYX, integer values; ids are
track-consistent across frames), ``<name>_skeleton.tif`` and ``<name>_epidata.pkl``.
Neighbouring cells are separated by a 1 px background skeleton, so the labels are
first passed through ``pimorph.bench.datasets.fill_gt_slivers`` (fills the seam
fragments enclosed by cells) and then a seam pass hands every remaining background
pixel that touches two different cells to its nearest cell; open background and
real gaps wider than the seam are left alone. Movies: movie1 Drosophila notum crop
(30 frames, partial field), movie2 abdomen histoblasts (30 frames, 2 channels,
0.275 um/px, 300 s), movie3 zebrafish telencephalon (11 frames, 3 channels), movie4
quail epiblast (1 frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tifffile
from scipy import ndimage as ndi

from ..bench.datasets import fill_gt_slivers
from ..io.images import pixel_size_from_tiff
from .tracking import Tracks, tracks_from_labels

PathLike = Union[str, Path]
ZENODO_RECORD = "20607705"
LABEL_FOLDERS = ("epics_corrected", "epics_correctedWithTA", "epics_init", "epics")


def seam_pixels(labels: np.ndarray) -> np.ndarray:
    """Background pixels of a 1 px seam: two different cell ids on opposite sides
    (left/right, up/down or either diagonal pair). Concave corners of open background
    between two cells are not seams and stay background."""
    lab = np.asarray(labels).astype(np.int64)
    H, W = lab.shape
    p = np.pad(lab, 1)

    def shifted(dr: int, dc: int) -> np.ndarray:
        return p[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]

    seam = np.zeros(lab.shape, dtype=bool)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a, b = shifted(dr, dc), shifted(-dr, -dc)
        seam |= (a > 0) & (b > 0) & (a != b)
    return seam & (lab == 0)


def fill_seams(labels: np.ndarray, max_area_px: int = 12, max_passes: int = 3) -> Tuple[np.ndarray, int]:
    """``fill_gt_slivers`` followed by seam passes; returns (labels, n_filled_pixels)."""
    lab = np.asarray(labels).astype(np.int32)
    n0 = int((lab == 0).sum())
    lab = fill_gt_slivers(lab, max_area_px=max_area_px)
    for _ in range(max_passes):
        seam = seam_pixels(lab)
        if not seam.any():
            break
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab[seam] = lab[idx[0][seam], idx[1][seam]]
    # seam junction pixels and 1 px holes become small enclosed components once the seams are gone
    lab = fill_gt_slivers(lab, max_area_px=max_area_px)
    return lab, n0 - int((lab == 0).sum())


def _find_labels_file(movie_dir: Path, labels_subdir: Optional[str]) -> Path:
    folders = [labels_subdir] if labels_subdir else list(LABEL_FOLDERS)
    for f in folders:
        hits = sorted((movie_dir / f).glob("*_labels.tif")) if (movie_dir / f).is_dir() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no *_labels.tif under {movie_dir} in {folders}")


def _stack_axes(tf: tifffile.TiffFile) -> Tuple[np.ndarray, str]:
    s = tf.series[0]
    return np.asarray(s.asarray()), str(s.axes)


def _pixel_size(tf: tifffile.TiffFile) -> Tuple[Optional[float], str]:
    """Pixel size from the TIFF tags; ImageJ writes XResolution (1, 1) for uncalibrated images."""
    xres = tf.pages[0].tags.get("XResolution")
    if xres is not None and tuple(xres.value) == (1, 1):
        return None, "none"
    return pixel_size_from_tiff(tf)


def load_epicure_movie(
    movie_dir: PathLike,
    labels_subdir: Optional[str] = "epics_corrected",
    fill: bool = True,
    max_seam_area_px: int = 12,
    load_images: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Return ``(label_stack int32 (T, H, W), image_stack or None, meta)``.

    ``image_stack`` is ``(T, C, H, W)`` float32 when ``<name>.tif`` sits next to the
    label folder. ``meta`` records paths, pixel size and frame interval when the
    ImageJ tags carry them, the number of ids and the number of seam pixels filled.
    """
    movie_dir = Path(movie_dir)
    lab_path = _find_labels_file(movie_dir, labels_subdir)
    with tifffile.TiffFile(str(lab_path)) as tf:
        lab, axes = _stack_axes(tf)
        px, px_src = _pixel_size(tf)
        ij = dict(tf.imagej_metadata or {})
    if lab.ndim == 2:
        lab = lab[None]
    if lab.ndim != 3:
        raise ValueError(f"{lab_path}: expected TYX labels, got {lab.shape} ({axes})")
    if not np.all(lab == np.round(lab)):
        raise ValueError(f"{lab_path}: non-integer label values")
    lab = lab.astype(np.int32)
    n_filled = 0
    if fill:
        frames = []
        for fr in lab:
            f, n = fill_seams(fr, max_area_px=max_seam_area_px)
            frames.append(f)
            n_filled += n
        lab = np.stack(frames)

    name = lab_path.name[: -len("_labels.tif")]
    raw_path = movie_dir / f"{name}.tif"
    images = None
    frame_interval_s = None
    if raw_path.exists():
        with tifffile.TiffFile(str(raw_path)) as tf:
            ij_raw = dict(tf.imagej_metadata or {})
            if px is None:
                px, px_src = _pixel_size(tf)
            frame_interval_s = ij_raw.get("finterval")
            if load_images:
                arr, raxes = _stack_axes(tf)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[None, None]
                elif arr.ndim == 3:
                    arr = arr[:, None] if raxes.startswith("T") or arr.shape[0] == lab.shape[0] else arr[None]
                if arr.shape[0] != lab.shape[0] or arr.shape[-2:] != lab.shape[-2:]:
                    raise ValueError(f"{raw_path}: shape {arr.shape} does not match labels {lab.shape}")
                images = arr
    ids = np.unique(lab[lab > 0])
    meta = {
        "movie": movie_dir.name,
        "name": name,
        "labels_path": str(lab_path),
        "labels_folder": lab_path.parent.name,
        "raw_path": str(raw_path) if raw_path.exists() else None,
        "n_frames": int(lab.shape[0]),
        "shape": (int(lab.shape[1]), int(lab.shape[2])),
        "n_channels": None if images is None else int(images.shape[1]),
        "pixel_size_um": px,
        "pixel_size_source": px_src,
        "frame_interval_s": None if frame_interval_s is None else float(frame_interval_s),
        "n_ids": int(ids.size),
        "cells_per_frame": [int(np.unique(fr[fr > 0]).size) for fr in lab],
        "n_seam_pixels_filled": int(n_filled),
        "imagej_unit": ij.get("unit"),
    }
    return lab, images, meta


def epicure_tracks(label_stack: np.ndarray, clean: bool = True) -> Tracks:
    """Wrap the curated track-consistent ids as ``Tracks`` (no linking, no lineage)."""
    return tracks_from_labels([np.asarray(fr) for fr in label_stack], lineage=None, clean=clean)


__all__ = ["LABEL_FOLDERS", "epicure_tracks", "fill_seams", "load_epicure_movie", "seam_pixels"]
