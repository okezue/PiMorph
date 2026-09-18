"""BioImage Archive S-BIAD1169: BRAFi-treated dermal microvascular endothelial cells.

Bromberger and Schossleitner (Medical University of Vienna), released 2024-06-01,
CC BY 4.0. Confluent primary DMEC on ibidi chamber slides, fixed and stained for
junction integrity markers after 1 h of BRAF inhibitor, imaged on a Zeiss LSM-980
with a 63x/1.40 oil objective. Every field is stored as single-channel TIFFs plus a
merge; folders name the condition (``5A_D10`` = Dabrafenib 10 uM, ``S6_DMSO`` =
vehicle, ``S7_E100`` = Encorafenib 100 uM, ``S6_V1`` = Vemurafenib 1 uM, ``S7_P10`` =
PLX8394 10 uM).

Channel roles (``CHANNEL_SUFFIX``) come from the ``Channel`` attribute of the study
file list (``/api/v1/files/S-BIAD1169``), which is authoritative, and agree with the
acquisition text (DAPI 405, Claudin-5 488, F-actin 514, VE-Cadherin 594, Prox1 639):
``_c1`` Prox1 (supplementary fields only), ``_c2`` VE-Cadherin, ``_c3`` F-actin,
``_c4`` Claudin-5, ``_c5`` DAPI; ``_c2-4`` and ``_c1-5`` are merges and are ignored.

The files are 8-bit RGB pseudocolor exports (1024 x 1024; VE-cadherin grey, claudin-5
cyan, F-actin yellow, DAPI blue), so the intensity is the max over RGB. Their TIFF
resolution tag is a 300 dpi print setting, not a physical calibration. Every field
carries the same burned-in white 50 um scale bar (rows 970-975, cols 614-992, 379 px),
which gives ``PIXEL_SIZE_UM`` = 50 / 379 = 0.1319 um/px, consistent with a 63x
objective on a 1024 px LSM frame (135 um field). The bar pixels are masked to zero.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .images import read_tiff

ACCESSION = "S-BIAD1169"
FILES_API = f"https://www.ebi.ac.uk/biostudies/api/v1/files/{ACCESSION}?start=0&length=400"
STUDY_API = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{ACCESSION}"
FTP_BASE = f"https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/169/{ACCESSION}/Files/"

# suffix -> channel role, from the file-list "Channel" attribute (verified 2026-09-18)
CHANNEL_SUFFIX: Dict[str, str] = {
    "c1": "Prox1",
    "c2": "VE-Cadherin",
    "c3": "F-actin",
    "c4": "Claudin-5",
    "c5": "DAPI",
}
CHANNEL_SOURCE = "biostudies file-list Channel attribute"
MERGE_SUFFIXES = ("c2-4", "c1-5")
CHANNEL_COLUMNS = [f"path_{c}" for c in CHANNEL_SUFFIX.values()]
# burned-in 50 um scale bar, identical in all 18 fields (measured on the DAPI exports)
SCALE_BAR_UM = 50.0
SCALE_BAR_PX = 379
SCALE_BAR_BOX = (968, 977, 612, 994)  # row_lo, row_hi, col_lo, col_hi (exclusive), with margin
PIXEL_SIZE_UM: Optional[float] = SCALE_BAR_UM / SCALE_BAR_PX
PIXEL_SIZE_SOURCE = "burned-in 50 um scale bar (379 px)"

# folder code -> compound (folder names like "S6_D10", "Figure 5A_V100", "S7_DMSO")
COMPOUND_CODES = {"D": "Dabrafenib", "V": "Vemurafenib", "E": "Encorafenib", "P": "PLX8394"}
_COND_RE = re.compile(r"_(DMSO|[DVEP])(\d+)?$")


def parse_condition(folder: str) -> Dict[str, object]:
    """Compound, concentration (uM) and coarse condition from a field folder name.

    ``condition`` is ``"DMSO"`` for vehicle and ``"BRAFi"`` for every inhibitor;
    ``treatment`` keeps the code (``"D10"``, ``"V100"``, ``"DMSO"``).
    """
    leaf = folder.rstrip("/").rsplit("/", 1)[-1]
    m = _COND_RE.search(leaf)
    if m is None:
        return {"condition": "unknown", "treatment": leaf, "compound": "unknown", "concentration_um": np.nan}
    code, conc = m.group(1), m.group(2)
    if code == "DMSO":
        return {"condition": "DMSO", "treatment": "DMSO", "compound": "DMSO", "concentration_um": 0.0}
    return {
        "condition": "BRAFi",
        "treatment": f"{code}{conc}",
        "compound": COMPOUND_CODES[code],
        "concentration_um": float(conc) if conc else np.nan,
    }


def _suffix(path: str) -> Optional[str]:
    m = re.search(r"_(c[0-9](?:-[0-9])?)\.tif$", path)
    return m.group(1) if m else None


def _field_stem(path: str) -> str:
    return re.sub(r"_c[0-9](?:-[0-9])?\.tif$", "", path)


def fetch_file_list(root: Path | str, refresh: bool = False) -> List[Dict[str, object]]:
    """File records (path, Channel, Compound, ...) from the BioStudies API, cached as JSON."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "files.json"
    if refresh or not cache.exists():
        import requests

        r = requests.get(FILES_API, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
    return list(json.loads(cache.read_text())["data"])


def download_sbiad1169(
    root: Path | str,
    max_fields: Optional[int] = None,
    include_merges: bool = False,
    conditions: Optional[Iterable[str]] = None,
    verbose: bool = True,
    retries: int = 5,
) -> pd.DataFrame:
    """Download the single-channel TIFFs into ``root`` (mirroring the archive folders).

    ``max_fields`` limits the number of fields, taken alternately from each condition so
    that DMSO and BRAFi stay balanced. Existing files with the archived size are skipped.
    Returns ``list_fields(root)``.
    """
    import requests

    root = Path(root)
    records = fetch_file_list(root)
    wanted = [r for r in records if _suffix(str(r["path"])) in CHANNEL_SUFFIX or include_merges]
    by_field: Dict[str, List[Dict[str, object]]] = {}
    for r in wanted:
        by_field.setdefault(_field_stem(str(r["path"])), []).append(r)
    stems = sorted(by_field)
    if conditions is not None:
        keep = set(conditions)
        stems = [s for s in stems if parse_condition(s.rsplit("/", 1)[0])["condition"] in keep]
    if max_fields is not None and max_fields < len(stems):
        groups: Dict[str, List[str]] = {}
        for s in stems:
            groups.setdefault(str(parse_condition(s.rsplit("/", 1)[0])["condition"]), []).append(s)
        picked: List[str] = []
        queues = [list(v) for _, v in sorted(groups.items())]
        while len(picked) < max_fields and any(queues):
            for q in queues:
                if q and len(picked) < max_fields:
                    picked.append(q.pop(0))
        stems = sorted(picked)
    session = requests.Session()
    for stem in stems:
        for r in by_field[stem]:
            rel = str(r["path"])
            dst = root / rel
            size = int(r.get("Size") or r.get("size") or 0)
            if dst.exists() and (size == 0 or dst.stat().st_size == size):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            url = FTP_BASE + urllib.parse.quote(rel)
            if verbose:
                print(f"GET {rel}", flush=True)
            _download(session, url, dst, size, retries=retries)
    return list_fields(root)


def _download(session, url: str, dst: Path, size: int, retries: int = 5) -> None:
    """Stream ``url`` to ``dst`` with retries; the FIRE endpoint drops connections."""
    import time

    import requests

    tmp = dst.with_suffix(dst.suffix + ".part")
    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            if size and tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {dst.name}: {tmp.stat().st_size} != {size}")
            tmp.replace(dst)
            return
        except (requests.RequestException, IOError) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last}")


def list_fields(root: Path | str, unique_snaps: bool = True) -> pd.DataFrame:
    """One row per field with condition metadata and the path of every channel present.

    Columns: field_id, folder, condition, treatment, compound, concentration_um,
    figure, ``path_<channel>`` for each role in ``CHANNEL_SUFFIX`` (None when missing).
    Fields lacking any of VE-Cadherin, Claudin-5, F-actin or DAPI are dropped. ``snap``
    is the microscope export name; with ``unique_snaps`` a snap filed under two figure
    folders (Figure 5A and S6 share D10, DMSO, V10, V100) is kept once.
    """
    root = Path(root)
    tifs = sorted(p for p in root.rglob("*.tif") if _suffix(p.name) in CHANNEL_SUFFIX)
    fields: Dict[str, Dict[str, object]] = {}
    for p in tifs:
        rel = p.relative_to(root).as_posix()
        stem = _field_stem(rel)
        row = fields.setdefault(
            stem, {"field_id": stem.replace("/", "__").replace(" ", "_"), "folder": stem.rsplit("/", 1)[0]}
        )
        row[f"path_{CHANNEL_SUFFIX[_suffix(p.name)]}"] = str(p)
    rows = []
    for stem, row in sorted(fields.items()):
        cond = parse_condition(str(row["folder"]))
        fig = re.search(r"(5A|S\d+)_", str(row["folder"]).rsplit("/", 1)[-1])
        row.update(cond)
        row["figure"] = fig.group(1) if fig else ""
        # the same snap can be filed under a main and a supplementary figure folder
        row["snap"] = stem.rsplit("/", 1)[-1]
        for c in CHANNEL_COLUMNS:
            row.setdefault(c, None)
        rows.append(row)
    cols = [
        "field_id",
        "folder",
        "condition",
        "treatment",
        "compound",
        "concentration_um",
        "figure",
        "snap",
        *CHANNEL_COLUMNS,
    ]
    df = pd.DataFrame(rows, columns=cols)
    required = [f"path_{c}" for c in ("VE-Cadherin", "Claudin-5", "F-actin", "DAPI")]
    if len(df):
        df = df[df[required].notna().all(axis=1)].reset_index(drop=True)
        if unique_snaps:
            df = df.drop_duplicates("snap", keep="first").reset_index(drop=True)
    return df


def mask_scale_bar(plane: np.ndarray, box=SCALE_BAR_BOX) -> np.ndarray:
    """Zero the burned-in scale bar (only when the box holds saturated pixels)."""
    r0, r1, c0, c1 = box
    if plane.shape[0] < r1 or plane.shape[1] < c1:
        return plane
    if float(plane[r0:r1, c0:c1].max()) >= 250.0:
        plane = plane.copy()
        plane[r0:r1, c0:c1] = 0.0
    return plane


def load_field(
    row: pd.Series | Dict[str, object], pixel_size_um: Optional[float] = PIXEL_SIZE_UM, mask_bar: bool = True
) -> Dict[str, object]:
    """Channel arrays (H, W) float32 keyed by role, plus ``pixel_size_um``.

    RGB pseudocolor exports are reduced to one plane (max over color). The TIFF
    resolution tag is a print dpi and is ignored; the calibration is the argument
    (default: the 50 um scale bar, ``PIXEL_SIZE_SOURCE``). The bar is zeroed when
    ``mask_bar``.
    """
    out: Dict[str, object] = {}
    for role in CHANNEL_SUFFIX.values():
        p = row.get(f"path_{role}") if isinstance(row, dict) else row.get(f"path_{role}", None)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        arr = read_tiff(p).data
        plane = np.asarray(arr[0] if arr.shape[0] == 1 else arr.max(axis=0), dtype=np.float32)
        out[role] = mask_scale_bar(plane) if mask_bar else plane
    out["pixel_size_um"] = pixel_size_um
    out["pixel_size_source"] = PIXEL_SIZE_SOURCE if pixel_size_um == PIXEL_SIZE_UM else "argument"
    out["channel_source"] = CHANNEL_SOURCE
    return out
