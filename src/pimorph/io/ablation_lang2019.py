"""Laser nanoablation movies with manually tracked recoil (Lang et al. 2019, Zenodo 3257654).

15 confocal time lapses (E-cadherin:GFP, Drosophila embryo germband, about 255 x 255 px
for 42.2 um, 727.67 ms per frame) of a single plasma-induced ablation of a supracellular
actomyosin cable at a parasegment boundary (Scarpa et al. 2018 Dev Cell). Each folder has
the movie, a Fiji line ROI (``reslice.roi``) along the cut cable, kymographs resliced
along that line, and manually tracked kymograph features: ``cutend_L.txt`` and
``cutend_R.txt`` give (position along the line in px, frame) of the two cut ends after
the ablation. Their separation rate right after the cut is the recoil velocity, which is
proportional to the tension the cable carried before the cut (for equal friction).

Licence CC BY-NC-SA 4.0. Files live under ``data/ablation_lang2019``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tifffile

FRAME_INTERVAL_S = 0.72767
FIELD_UM = 42.2


def read_line_roi(path: Path) -> Dict[str, float]:
    """Endpoints of an ImageJ line ROI (``x1, y1, x2, y2`` in pixel units)."""
    b = Path(path).read_bytes()
    if b[:4] != b"Iout":
        raise ValueError(f"{path} is not an ImageJ ROI")
    roi_type = b[6]
    if roi_type != 3:  # ImageJ RoiDecoder: 3 = line
        raise ValueError(f"{path}: ROI type {roi_type} is not a line")
    x1, y1, x2, y2 = struct.unpack(">ffff", b[18:34])
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _track_file(folder: Path, side: str) -> Optional[Path]:
    for name in (f"cutend_{side}.txt", f"{side}_cutend.txt"):
        p = folder / name
        if p.exists():
            return p
    return None


def list_ablations(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for folder in sorted(Path(root).glob("*/*/")):
        if "__MACOSX" in str(folder):
            continue
        movies = [p for p in folder.glob("*.tif") if "Reslice" not in p.name]
        roi = folder / "reslice.roi"
        left, right = _track_file(folder, "L"), _track_file(folder, "R")
        if not movies or not roi.exists() or left is None or right is None:
            continue
        rows.append(
            {
                "ablation_id": folder.name,
                "genotype": folder.parent.name,
                "path_movie": str(movies[0]),
                "path_roi": str(roi),
                "path_left": str(left),
                "path_right": str(right),
            }
        )
    return pd.DataFrame(rows)


def read_movie(path: str) -> np.ndarray:
    with tifffile.TiffFile(path) as t:
        arr = t.pages[0].asarray() if len(t.pages) == 1 else np.stack([p.asarray() for p in t.pages])
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    return arr


def read_track(path: str) -> np.ndarray:
    """(n, 2) array of (position px along the reslice line, frame index)."""
    return np.atleast_2d(np.loadtxt(path))


def recoil(left: np.ndarray, right: np.ndarray, n_fit: int = 4) -> Dict[str, float]:
    """Initial recoil velocity of the cut ends (px per frame) from the two tracks.

    The separation |x_R - x_L| is interpolated on the common frames of both tracks; the
    initial velocity is the slope of a least-squares line over the first ``n_fit`` common
    frames (the standard first-seconds recoil measure), and ``cut_frame`` the first frame
    of either track.
    """
    tl = left[np.argsort(left[:, 1])]
    tr = right[np.argsort(right[:, 1])]
    t0 = max(tl[0, 1], tr[0, 1])
    t1 = min(tl[-1, 1], tr[-1, 1])
    if t1 <= t0:
        return {"cut_frame": float(min(tl[0, 1], tr[0, 1])), "recoil_px_per_frame": float("nan")}
    ts = np.linspace(t0, t1, max(int(round(t1 - t0)) + 1, 2))
    sep = np.abs(np.interp(ts, tr[:, 1], tr[:, 0]) - np.interp(ts, tl[:, 1], tl[:, 0]))
    k = min(max(n_fit, 2), len(ts))
    slope = float(np.polyfit(ts[:k], sep[:k], 1)[0])
    return {
        "cut_frame": float(min(tl[0, 1], tr[0, 1])),
        "cut_pos_px": float(0.5 * (tl[0, 0] + tr[0, 0])),
        "initial_separation_px": float(sep[0]),
        "recoil_px_per_frame": slope,
        "recoil_um_per_s": slope * (FIELD_UM / 255.0) / FRAME_INTERVAL_S,
        "n_fit_frames": int(k),
        "final_separation_px": float(sep[-1]),
    }


def line_point(roi: Dict[str, float], s: float) -> np.ndarray:
    """Image (row, col) of the point at distance ``s`` px from (x1, y1) along the line."""
    dx, dy = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
    L = float(np.hypot(dx, dy))
    if L == 0:
        return np.array([roi["y1"], roi["x1"]])
    return np.array([roi["y1"] + dy * s / L, roi["x1"] + dx * s / L])


def line_direction(roi: Dict[str, float]) -> np.ndarray:
    """Unit (row, col) direction of the reslice line."""
    d = np.array([roi["y2"] - roi["y1"], roi["x2"] - roi["x1"]], dtype=float)
    n = float(np.hypot(*d))
    return d / n if n > 0 else d
