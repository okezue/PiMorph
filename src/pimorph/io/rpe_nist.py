"""NIST / NEI iPSC-RPE data with measured transepithelial resistance (Schaub et al. 2020).

Source: https://isg.nist.gov/deepzoomweb/data/RPEimplants (DOI 10.18434/T4/1503229),
"Deep learning predicts function of live retinal pigment epithelium from quantitative
microscopy", J Clin Invest 130:1010 (2020). Two parts are used here:

- 1032 registered 256 x 256 tiles from 10 AMD-iRPE wells: ZO-1 fluorescence
  (``fluorescentZ01.zip``), hand-corrected border masks (``segmentation_mask.zip``,
  255 = cell interior, 0 = border or background) and QBAM blue absorbance
  (``absorbance-dataset.zip``). Tile names carry the well as ``<date>-D<k><clone>-D<day>``
  with ``D2`` = AMD1, ``D3`` = AMD2, ``D4`` = AMD3 (donor codes taken from
  ``AMD_Data/DonorMatching/AMD_Donor-Match_DNNI.csv``) and the tile position
  ``_n<index>_r<row>_c<col>`` in the whole-well ZO-1 image.
- whole-well stitched ZO-1 images (``NN_Data/Unregistered_Images/ZO1``, about
  3000 x 2700 px, uint16) for the same 10 wells.

TER per well (Ohm cm^2, mean and SD of three readings) is in
``AMD_Data/AnalysisOfSegmentedData/AMD_TER-Data_Mean-SD.csv`` with the clone letters of
the paper (AMD1: A, B; AMD2 and AMD3: A, B, C). The image files use internal clone
letters. The mapping is forced by which culture days exist per clone and agrees with the
paper-style names used by the ``NN_Data/QBAM_DNNS`` tiles: image ``AMD1B`` = clone 1A,
``AMD1C`` = clone 1B, ``AMD2C`` = clone 2B, ``AMD3B`` = clone 3B, ``AMD3C`` = clone 3C.

The pixel size is not recorded in any file of the deposit; all wells were imaged with the
same optics, so per-pixel quantities are comparable across wells but not in micrometres.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels

DONOR_CODE = {"D2": "AMD1", "D3": "AMD2", "D4": "AMD3"}
# image clone letter -> TER-table clone letter, per donor (see module docstring)
CLONE_MAP = {
    ("AMD1", "B"): "A",
    ("AMD1", "C"): "B",
    ("AMD2", "C"): "B",
    ("AMD3", "B"): "B",
    ("AMD3", "C"): "C",
}
TILE_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<code>D\d|AMD\d)(?P<clone>[A-Z])-D(?P<day>\d+)_.*_n(?P<n>\d+)_r(?P<r>\d+)_c(?P<c>\d+)"
)
WELL_RE = re.compile(r"^(?P<date>\d{8})-(?P<code>D\d|AMD\d)(?P<clone>[A-Z])-D(?P<day>\d+)")
TILE_SHAPE = (256, 256)


def well_key(donor: str, ter_clone: str, day: int) -> str:
    return f"{donor}_{ter_clone}_D{day}"


def parse_well(name: str) -> Dict[str, object]:
    """Well fields of a tile or whole-well file name, mapped to the TER-table naming."""
    m = WELL_RE.match(name)
    if m is None:
        raise ValueError(f"not an RPE file name: {name!r}")
    code = m.group("code")
    donor = DONOR_CODE.get(code, code)
    ter_clone = CLONE_MAP.get((donor, m.group("clone")))
    day = int(m.group("day"))
    return {
        "date": m.group("date"),
        "donor": donor,
        "image_clone": m.group("clone"),
        "ter_clone": ter_clone,
        "day": day,
        "well": well_key(donor, ter_clone, day) if ter_clone else None,
    }


def load_ter(root: Path) -> pd.DataFrame:
    """TER table with a ``well`` key matching :func:`parse_well`."""
    p = Path(root) / "AMD_Data" / "AnalysisOfSegmentedData" / "AMD_TER-Data_Mean-SD.csv"
    df = pd.read_csv(p)
    df["ter_clone"] = df["clone"].astype(str).str[-1]
    df["day"] = df["day"].astype(int)
    df["well"] = [well_key(d, c, int(y)) for d, c, y in zip(df["donor"], df["ter_clone"], df["day"])]
    return df.rename(columns={"ter.Mean": "ter_mean", "ter.StDev": "ter_sd"})


def load_vegf(root: Path) -> pd.DataFrame:
    p = Path(root) / "AMD_Data" / "AnalysisOfSegmentedData" / "AMD_VEGF-Data_Mean-SD.csv"
    df = pd.read_csv(p)
    df["ter_clone"] = df["clone"].astype(str).str[-1]
    df["day"] = df["day"].astype(str).str.lstrip("D").astype(int)
    df["well"] = [well_key(d, c, int(y)) for d, c, y in zip(df["donor"], df["ter_clone"], df["day"])]
    return df


def list_tiles(root: Path) -> pd.DataFrame:
    """One row per registered tile with the ZO-1, mask and absorbance paths and its well."""
    root = Path(root)
    zo1_dir = root / "tiles" / "fluorescentZ01" / "images"
    seg_dir = root / "tiles" / "segmentation_mask" / "images"
    abs_dir = root / "tiles" / "absorbance-dataset" / "images"
    # mask and absorbance names differ from the ZO-1 names in suffixes ("-1", "-C", ".tif-C"),
    # so files are matched on well prefix and tile index n
    abs_by_n = _index_by_well_and_n(abs_dir)
    seg_by_n = _index_by_well_and_n(seg_dir)
    rows: List[Dict[str, object]] = []
    for p in sorted(zo1_dir.glob("*.tif")):
        m = TILE_RE.match(p.name)
        if m is None:
            continue
        w = parse_well(p.name)
        key = _well_n_key(m)
        rows.append(
            {
                "tile_id": p.stem.replace(".ome", ""),
                **w,
                "n": int(m.group("n")),
                "row": int(m.group("r")),
                "col": int(m.group("c")),
                "path_zo1": str(p),
                "path_mask": str(seg_by_n[key]) if key in seg_by_n else None,
                "path_abs": str(abs_by_n[key]) if key in abs_by_n else None,
            }
        )
    return pd.DataFrame(rows)


def _well_n_key(m: "re.Match[str]") -> str:
    return f"{m.group('date')}-{m.group('code')}{m.group('clone')}-D{m.group('day')}_n{m.group('n')}"


def _index_by_well_and_n(directory: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in directory.glob("*.tif"):
        m = TILE_RE.match(p.name)
        if m:
            out[_well_n_key(m)] = p
    return out


def list_wells(root: Path) -> pd.DataFrame:
    """Whole-well ZO-1 images with their TER-table well key."""
    d = Path(root) / "NN_Data" / "Unregistered_Images" / "ZO1"
    rows = []
    for p in sorted(d.glob("*.tif")):
        rows.append({**parse_well(p.name), "path_zo1": str(p)})
    return pd.DataFrame(rows)


def read_tile(path: str) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def labels_from_border_mask(mask: np.ndarray, border_px: int = 3, min_area_px: int = 9) -> np.ndarray:
    """Cell labels from a hand-corrected border mask (non-zero = cell interior).

    Interiors are labelled by 4-connectivity, fragments below ``min_area_px`` dropped, and
    the labels grown by ``border_px`` so neighbouring cells meet on a shared crack; the
    border lines are 2 to 3 px wide, so unfilled background is left only where no cell is
    within ``border_px``.
    """
    interior = np.asarray(mask) > 0
    lab, n = ndi.label(interior, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return lab.astype(np.int32)
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < min_area_px)
    small = small[small > 0]
    if small.size:
        lab[np.isin(lab, small)] = 0
    lab = expand_labels(lab, distance=border_px)
    return lab.astype(np.int32)


def upsample_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(labels), np.ones((factor, factor), dtype=labels.dtype)).astype(np.int32)


def upsample_image(image: np.ndarray, factor: int, order: int = 1) -> np.ndarray:
    return ndi.zoom(np.asarray(image, dtype=np.float32), factor, order=order)


def qbam_unannotated(mask: np.ndarray, area_factor: float = 5.0, min_area_px: int = 1500) -> np.ndarray:
    """Regions the annotators left without borders: 255 components larger than
    ``area_factor`` times the median component area and ``min_area_px`` (about five cells).
    Almost every registered tile has one (median 21% of the tile); they are not cells and
    are excluded from training and scoring."""
    interior = np.asarray(mask) > 0
    lab, n = ndi.label(interior, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return np.zeros_like(interior)
    counts = np.bincount(lab.ravel())[1:]
    big = np.flatnonzero((counts > area_factor * np.median(counts)) & (counts > min_area_px)) + 1
    return np.isin(lab, big)


def qbam_instances(mask: np.ndarray, border_px: int = 3, min_area_px: int = 20) -> np.ndarray:
    """Instances from a registered border mask by closing the drawn borders, as
    ``pimorph.bench.datasets.skeleton_roi_to_instance`` does for skeleton tracings: the
    border pixels (mask == 0) within ``border_px`` of an interior become part of the nearest
    interior by watershed on the distance to the border, so neighbours share a crack; the
    remaining background (no cell within ``border_px``) stays 0. Unannotated regions
    (:func:`qbam_unannotated`) are 0 as well; callers that need to ignore them keep the
    boolean separately."""
    from ..bench.datasets import skeleton_roi_to_instance

    interior = (np.asarray(mask) > 0) & ~qbam_unannotated(mask)
    roi = ndi.binary_dilation(interior, iterations=int(border_px))
    return skeleton_roi_to_instance(~interior, roi, min_area_px=min_area_px)


# ------------------------------------------------------------------ Healthy-2 series
# Live QBAM of healthy-donor iRPE (Drive folder 1sHfNWudfXvZXTd1hTo2FYalbzU39zTLN): plates
# LORD-1..7, wells B1 B2 C1 C2 C3 C4, weekly imaging on 8 dates, filters Blue 488 / Green 561 /
# Red 633, a 4 x 3 tile grid per well (1040 x 1388 float32 absorbance). TER (Ohm) for plates
# LORD-2..7 on weeks 3 to 8 in EVOM.csv. See data/rpe_nist/healthy2/INVENTORY.md.
HEALTHY2_FOLDER_ID = "1sHfNWudfXvZXTd1hTo2FYalbzU39zTLN"
HEALTHY2_WEEK1 = pd.Timestamp("2017-02-02")
HEALTHY2_TILE_RE = re.compile(
    r"^(?P<plate>LORD-\d)/Date (?P<date>\d{4}-\d{2}-\d{2})T[^/]+/(?P<filter>[^/]+)/Absorption Images/"
    r"(?P<well>[A-D]\d)_r(?P<r>\d{3})_c(?P<c>\d{3})\.tif$"
)


def healthy2_listing(pkl_path: Path) -> pd.DataFrame:
    """Absorbance tiles of the Healthy-2 Drive listing (``(file_id, path)`` pairs as saved by
    ``gdown.download_folder(skip_download=True)``), one row per tile with plate, date, week,
    filter, well, grid row/col and Drive file id."""
    import os
    import pickle

    pairs = pickle.load(open(pkl_path, "rb"))
    root = os.path.commonpath([p for _, p in pairs]) if len(pairs) > 1 else ""
    rows = []
    for fid, p in pairs:
        rel = os.path.relpath(p, root) if root else p
        m = HEALTHY2_TILE_RE.match(rel)
        if m is None:
            continue
        date = pd.Timestamp(m.group("date"))
        rows.append(
            {
                "plate": m.group("plate"),
                "well": m.group("well"),
                "well_id": f"{m.group('plate')}_{m.group('well')}",
                "date": date.strftime("%Y-%m-%d"),
                "week": int((date - HEALTHY2_WEEK1).days // 7) + 1,
                "filter": m.group("filter"),
                "grid_r": int(m.group("r")),
                "grid_c": int(m.group("c")),
                "drive_id": fid,
                "drive_path": rel,
            }
        )
    return pd.DataFrame(rows)


def healthy2_ter_long(evom_csv: Path) -> pd.DataFrame:
    """EVOM.csv (wide, one column per date) as one row per well-timepoint with condition
    (A Aphidicolin, C Control, H HPI4), ISO date, week index and TER in Ohm."""
    wide = pd.read_csv(evom_csv)
    cond = {"A": "Aphidicolin", "C": "Control", "H": "HPI4"}
    rows = []
    for _, r in wide.iterrows():
        for col in wide.columns[3:]:
            v = r[col]
            if not np.isfinite(float(v)):
                continue
            date = pd.to_datetime(col, format="%d-%b-%y")
            rows.append(
                {
                    "plate": r["plate_name"],
                    "well": r["well"],
                    "well_id": f"{r['plate_name']}_{r['well']}",
                    "treatment": r["treatment"],
                    "condition": cond.get(str(r["treatment"]), str(r["treatment"])),
                    "date": date.strftime("%Y-%m-%d"),
                    "week": int((date - HEALTHY2_WEEK1).days // 7) + 1,
                    "ter_ohm": float(v),
                }
            )
    return pd.DataFrame(rows)


def healthy2_tile_name(plate: str, well: str, date: str, filt: str, grid_r: int, grid_c: int) -> str:
    """Local file name of a downloaded Healthy-2 tile."""
    return f"{plate}_{well}_{date}_{filt.split()[0]}_r{grid_r:03d}_c{grid_c:03d}.tif"


def tile_crop_in_well(well_image: np.ndarray, row: int, col: int, shape=TILE_SHAPE) -> Optional[np.ndarray]:
    """Crop of the whole-well image at the tile position, or None if it falls outside."""
    r0, c0 = int(row), int(col)
    r1, c1 = r0 + shape[0], c0 + shape[1]
    if r0 < 0 or c0 < 0 or r1 > well_image.shape[0] or c1 > well_image.shape[1]:
        return None
    return well_image[r0:r1, c0:c1]
