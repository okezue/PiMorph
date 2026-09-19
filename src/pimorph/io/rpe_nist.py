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


def tile_crop_in_well(well_image: np.ndarray, row: int, col: int, shape=TILE_SHAPE) -> Optional[np.ndarray]:
    """Crop of the whole-well image at the tile position, or None if it falls outside."""
    r0, c0 = int(row), int(col)
    r1, c1 = r0 + shape[0], c0 + shape[1]
    if r0 < 0 or c0 < 0 or r1 > well_image.shape[0] or c1 > well_image.shape[1]:
        return None
    return well_image[r0:r1, c0:c1]
