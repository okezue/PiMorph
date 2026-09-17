"""Calibrated image reading.

Pixel size is taken from OME-XML, then ImageJ/TIFF resolution tags, then the
manifest. When none is available the image is returned uncalibrated and the caller
records ``units = "px"``; nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


@dataclass
class CalibratedImage:
    data: np.ndarray  # (C, H, W) float32
    channel_names: List[str]
    pixel_size_um: Optional[float]
    pixel_size_source: str  # "ome" | "tiff_resolution" | "manifest" | "none"
    path: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[2])

    def channel(self, name_or_index) -> np.ndarray:
        if isinstance(name_or_index, (int, np.integer)):
            return self.data[int(name_or_index)]
        if name_or_index in self.channel_names:
            return self.data[self.channel_names.index(name_or_index)]
        low = str(name_or_index).lower()
        for i, n in enumerate(self.channel_names):
            if low in n.lower():
                return self.data[i]
        raise KeyError(f"channel {name_or_index!r} not in {self.channel_names}")


def pixel_size_from_tiff(tf: tifffile.TiffFile) -> Tuple[Optional[float], str]:
    """Return (pixel_size_um, source) from OME-XML or TIFF/ImageJ resolution tags."""
    if tf.ome_metadata:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
        unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
        if m:
            v = float(m.group(1))
            u = unit.group(1) if unit else "µm"
            if u in ("µm", "um", "micron", "microns"):
                return v, "ome"
            if u in ("nm",):
                return v / 1000.0, "ome"
            if u in ("mm",):
                return v * 1000.0, "ome"
    page = tf.pages[0]
    xres = page.tags.get("XResolution")
    if xres is not None:
        num, den = xres.value
        if num and den:
            px_per_unit = float(num) / float(den)
            unit_tag = page.tags.get("ResolutionUnit")
            unit = unit_tag.value if unit_tag is not None else 1
            ij_unit = (tf.imagej_metadata or {}).get("unit") if tf.is_imagej else None
            if ij_unit in ("micron", "um", "µm", "microns") or unit == 1:
                if px_per_unit > 0:
                    return 1.0 / px_per_unit, "tiff_resolution"
            if unit == 3:  # centimeter
                return 1e4 / px_per_unit, "tiff_resolution"
            if unit == 2:  # inch
                return 25400.0 / px_per_unit, "tiff_resolution"
    return None, "none"


def read_tiff(path: Path | str, channel_names: Optional[Sequence[str]] = None) -> CalibratedImage:
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.asarray()
        px, src = pixel_size_from_tiff(tf)
        meta = {"axes": tf.series[0].axes, "dtype": str(arr.dtype)}
        axes = tf.series[0].axes
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        # (C,H,W) unless axes says otherwise; interleaved (H,W,C) with small C is moved
        if axes.endswith("S") or (arr.shape[-1] <= 4 and arr.shape[0] > 4):
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"unsupported TIFF dimensionality {arr.shape} for {path}")
    arr = arr.astype(np.float32, copy=False)
    C = arr.shape[0]
    names = list(channel_names) if channel_names else [f"ch{i}" for i in range(C)]
    if len(names) != C:
        names = (names + [f"ch{i}" for i in range(len(names), C)])[:C]
    return CalibratedImage(
        data=arr, channel_names=names, pixel_size_um=px, pixel_size_source=src, path=str(path), meta=meta
    )


def stack_channel_files(paths: Dict[str, Path | str], pixel_size_um: Optional[float] = None) -> CalibratedImage:
    """Stack single-channel files (name -> path) into one calibrated image.

    Calibration is taken from the first file that carries one, else from the
    argument (source "manifest"), else None.
    """
    arrs = []
    names = []
    px, src = None, "none"
    for name, p in paths.items():
        im = read_tiff(p)
        if im.data.shape[0] != 1:
            raise ValueError(f"{p} has {im.data.shape[0]} channels; expected single-channel file")
        arrs.append(im.data[0])
        names.append(name)
        if px is None and im.pixel_size_um is not None:
            px, src = im.pixel_size_um, im.pixel_size_source
    if px is None and pixel_size_um is not None:
        px, src = float(pixel_size_um), "manifest"
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"channel files differ in shape: {shapes}")
    return CalibratedImage(
        data=np.stack(arrs, axis=0).astype(np.float32),
        channel_names=names,
        pixel_size_um=px,
        pixel_size_source=src,
        path=str(list(paths.values())[0]),
        meta={"files": {k: str(v) for k, v in paths.items()}},
    )
