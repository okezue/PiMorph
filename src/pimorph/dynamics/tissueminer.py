"""Loader for the TissueMiner demo (Drosophila pupal wing, 71 frames).

Archive ``http://bds.mpi-cbg.de/tissue_miner/demo.tar.gz`` unpacks to
``example_data/demo/`` with

- ``demo.sqlite``: tables ``frames`` (frame, time_sec), ``cell_histories``
  (cell_id, tissue_analyzer_group_id, first_occ, last_occ, left/right daughter,
  appears_by, disappears_by, lineage_group, generation), ``cells`` (per frame
  centre, area, elongation, polarity), ``bonds``, ``vertices`` (per frame
  positions) and ``directed_bonds`` (frame, cell_id, dbond_id, conj_dbond_id,
  bond_id, vertex_id, left_dbond_id). Cell 10000 is the margin pseudo-cell.
  There is no dedicated T1 table: TissueMiner derives topology changes in R from
  the neighbour relation encoded by ``directed_bonds``; ``db_snapshots`` does the
  same here so the event detector can run on the database topology.
- ``Segmentation/demo_NNN/tracked_cells_resized.tif``: Tissue Analyzer tracked
  cells as RGB, id = R * 65536 + G * 256 + B, white = 1 px boundary lattice;
  ``original.png`` raw image, ``dividing_cells.tif`` division mask.

Tissue Analyzer ids map to database ``cell_id`` through ``cell_histories`` (an id
can map to a mother and, after division, one daughter, so the frame decides).
"""

from __future__ import annotations

import sqlite3
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import expand_labels

from .events_detect import TopologySnapshot

TM_URL = "http://bds.mpi-cbg.de/tissue_miner/demo.tar.gz"
WHITE = 255 * 65536 + 255 * 256 + 255
MARGIN_CELL = 10000
UNMAPPED_OFFSET = 20_000_000


def download_tissueminer_demo(root: str = "data/tissueminer") -> Path:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    demo = root_p / "example_data" / "demo"
    if demo.is_dir():
        return demo
    tgz = root_p / "demo.tar.gz"
    if not tgz.exists():
        subprocess.run(["curl", "-sSL", "-o", str(tgz), TM_URL], check=True)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(root_p)
    if not demo.is_dir():
        raise FileNotFoundError(f"{tgz} did not unpack to {demo}")
    return demo


def decode_tracked_cells(rgb: np.ndarray) -> np.ndarray:
    """RGB Tissue Analyzer image -> integer id per pixel, 0 on the white boundary lattice."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected an RGB image, got shape {rgb.shape}")
    code = rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64)
    code[code == WHITE] = 0
    return code


def _history_table(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as con:
        return pd.read_sql_query("select * from cell_histories", con)


def _ta_lookup(hist: pd.DataFrame) -> Dict[int, List[Tuple[int, int, int]]]:
    """tissue_analyzer_group_id -> [(first_occ, last_occ, cell_id)]."""
    out: Dict[int, List[Tuple[int, int, int]]] = {}
    for ta, f0, f1, cid in zip(hist["tissue_analyzer_group_id"], hist["first_occ"], hist["last_occ"], hist["cell_id"]):
        out.setdefault(int(ta), []).append((int(f0), int(f1), int(cid)))
    return out


def codes_to_cell_ids(code: np.ndarray, frame: int, lookup: Dict[int, List[Tuple[int, int, int]]]) -> np.ndarray:
    """Map Tissue Analyzer ids of one frame to database cell ids; unknown ids keep
    ``UNMAPPED_OFFSET + id`` so they stay consistent across frames."""
    ids = np.unique(code[code > 0])
    lut_keys = []
    lut_vals = []
    for ta in ids.tolist():
        cid = None
        for f0, f1, c in lookup.get(ta, []):
            if f0 <= frame <= f1:
                cid = c
                break
        lut_keys.append(ta)
        lut_vals.append(cid if cid is not None else UNMAPPED_OFFSET + ta)
    out = np.zeros_like(code)
    if ids.size:
        idx = np.searchsorted(ids, code, side="left")
        idx = np.clip(idx, 0, ids.size - 1)
        hit = (code > 0) & (ids[idx] == code)
        vals = np.asarray(lut_vals, dtype=np.int64)
        out[hit] = vals[idx[hit]]
    return out


def load_tissueminer_demo(
    root: str = "data/tissueminer",
    download: bool = True,
    fill_boundaries: bool = True,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-frame label images (database cell ids) plus ground truth from the database.

    With ``fill_boundaries`` the 1 px white lattice is handed to the nearest cell so
    that neighbouring cells touch (the complex needs cell-cell cracks, not a
    background skeleton). Returns label_stack, code_stack (raw Tissue Analyzer
    ids), db_path, frames (time in seconds), cell_histories, divisions and
    mapping statistics.
    """
    demo = Path(root) / "example_data" / "demo"
    if not demo.is_dir():
        if not download:
            raise FileNotFoundError(demo)
        demo = download_tissueminer_demo(root)
    db_path = demo / "demo.sqlite"
    hist = _history_table(db_path)
    lookup = _ta_lookup(hist)
    with sqlite3.connect(str(db_path)) as con:
        frames = pd.read_sql_query("select frame, time_sec from frames order by frame", con)
    seg_dirs = sorted(p for p in (demo / "Segmentation").iterdir() if p.is_dir() and p.name.startswith("demo_"))
    if max_frames is not None:
        seg_dirs = seg_dirs[:max_frames]
    label_stack: List[np.ndarray] = []
    code_stack: List[np.ndarray] = []
    n_unmapped = 0
    n_ids = 0
    for d in seg_dirs:
        t = int(d.name.split("_")[-1])
        code = decode_tracked_cells(tifffile.imread(d / "tracked_cells_resized.tif"))
        lab = codes_to_cell_ids(code, t, lookup)
        ids = np.unique(lab[lab > 0])
        n_ids += int(ids.size)
        n_unmapped += int(np.count_nonzero(ids >= UNMAPPED_OFFSET))
        if fill_boundaries:
            filled = expand_labels(lab, distance=2)
            lab = np.where(lab == 0, filled, lab)
        code_stack.append(code)
        label_stack.append(lab.astype(np.int64))
    return {
        "root": str(demo),
        "db_path": str(db_path),
        "n_frames": len(label_stack),
        "frames": frames,
        "label_stack": label_stack,
        "code_stack": code_stack,
        "cell_histories": hist,
        "divisions": db_divisions(db_path),
        "stats": {"n_cell_frames": n_ids, "n_unmapped_cell_frames": n_unmapped},
    }


def db_divisions(db_path: Path) -> List[Dict[str, Any]]:
    """Divisions from ``cell_histories`` (mother disappears by Division, two daughters)."""
    hist = _history_table(db_path)
    out = []
    for row in hist.itertuples(index=False):
        if row.disappears_by != "Division":
            continue
        kids = [int(x) for x in (row.left_daughter_cell_id, row.right_daughter_cell_id) if pd.notna(x)]
        if len(kids) != 2:
            continue
        first = hist.loc[hist["cell_id"].isin(kids), "first_occ"]
        frame = int(first.min()) if len(first) else int(row.last_occ) + 1
        out.append({"parent": int(row.cell_id), "children": sorted(kids), "frame": frame})
    return out


def db_snapshots(db_path: Path, frames: Optional[List[int]] = None) -> Dict[int, TopologySnapshot]:
    """Topology snapshots from ``directed_bonds`` per frame.

    Contacts are (cell, conjugate cell) pairs, vertices the cells around each
    ``vertex_id``, sides the number of directed bonds per cell. The margin cell
    10000 is mapped to 0 like the outer face of an extracted complex. Counts are
    (distinct vertices, bonds, cells + 1).
    """
    with sqlite3.connect(str(db_path)) as con:
        q = (
            "select d.frame, d.cell_id, c.cell_id as other, d.vertex_id, d.bond_id "
            "from directed_bonds d join directed_bonds c on d.conj_dbond_id = c.dbond_id and d.frame = c.frame"
        )
        if frames is not None:
            q += " where d.frame in (%s)" % ",".join(str(int(f)) for f in frames)
        df = pd.read_sql_query(q, con)
        areas = pd.read_sql_query("select frame, cell_id, area from cells where area > 0", con)
    area_by_frame = {
        int(t): dict(zip(g["cell_id"].astype(int), g["area"].astype(float))) for t, g in areas.groupby("frame")
    }
    out: Dict[int, TopologySnapshot] = {}
    for t, g in df.groupby("frame"):
        a = np.where(g["cell_id"].to_numpy() == MARGIN_CELL, 0, g["cell_id"].to_numpy()).astype(np.int64)
        b = np.where(g["other"].to_numpy() == MARGIN_CELL, 0, g["other"].to_numpy()).astype(np.int64)
        contacts: Set[Tuple[int, int]] = set()
        for x, y in zip(a.tolist(), b.tolist()):
            if x != y:
                contacts.add((x, y) if x <= y else (y, x))
        sides: Dict[int, int] = {}
        for x in a.tolist():
            sides[x] = sides.get(x, 0) + 1
        cells = {x for x in sides if x > 0}
        vert: Dict[int, Set[int]] = {}
        for vid, x in zip(g["vertex_id"].tolist(), a.tolist()):
            vert.setdefault(int(vid), set()).add(x)
        vertices: Set[FrozenSet[int]] = {frozenset(s) for s in vert.values() if len(s) >= 3}
        n_bonds = int(g["bond_id"].nunique())
        out[int(t)] = TopologySnapshot(
            cells=cells,
            contacts=contacts,
            vertices=vertices,
            gaps={},
            sides=sides,
            counts=(len(vert), n_bonds, len(cells) + 1),
            area={c: a for c, a in area_by_frame.get(int(t), {}).items() if c in cells},
        )
    return out


__all__ = [
    "MARGIN_CELL",
    "TM_URL",
    "UNMAPPED_OFFSET",
    "codes_to_cell_ids",
    "db_divisions",
    "db_snapshots",
    "decode_tracked_cells",
    "download_tissueminer_demo",
    "load_tissueminer_demo",
]
