"""Loaders written for the real-data validation of mechanics and function (2026-09-19):
well naming of the NIST RPE deposit, border-mask labelling, ImageJ line ROIs and recoil
velocities of the Lang et al. ablation tracks. Pure unit tests, no data download."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from pimorph.complex import extract_complex
from pimorph.io import ablation_lang2019 as A
from pimorph.io import rpe_nist as R


def test_rpe_well_mapping_matches_ter_table_naming():
    w = R.parse_well("20161109-D2C-D75_Med_c1_ORG_n681_r401_c2001.ome.tif")
    assert (w["donor"], w["image_clone"], w["ter_clone"], w["day"]) == ("AMD1", "C", "B", 75)
    assert w["well"] == "AMD1_B_D75"
    assert R.parse_well("20161117-AMD1B-D75_Med_c1_ORG.tif")["well"] == "AMD1_A_D75"
    assert R.parse_well("20161012-D3C-D73_Blue_stitched-0_n100_r2196_c1007.ome.tif")["well"] == "AMD2_B_D73"
    assert R.parse_well("20161027-D4C-D75_x.tif")["well"] == "AMD3_C_D75"
    # unknown internal clone letters map to no well instead of a wrong one
    assert R.parse_well("20161027-D4A-D75_x.tif")["well"] is None
    with pytest.raises(ValueError):
        R.parse_well("not_a_tile.tif")


def test_rpe_tile_regex_reads_position():
    m = R.TILE_RE.match("20161102-D4B-D74_Med_c1_ORG_n494_r1201_c1401-C.ome.tif")
    assert m is not None
    assert (int(m.group("n")), int(m.group("r")), int(m.group("c"))) == (494, 1201, 1401)


def test_labels_from_border_mask_gives_touching_cells():
    mask = np.full((40, 40), 255, dtype=np.uint8)
    mask[:, 19:22] = 0  # 3 px border between two cells
    mask[19:22, :] = 0
    lab = R.labels_from_border_mask(mask, border_px=3)
    assert lab.max() == 4
    assert (lab == 0).sum() == 0  # borders filled, cells share cracks
    cx = extract_complex(lab)
    assert cx.cell_faces.size == 4
    assert cx.cell_cell_edges().size == 4
    up = R.upsample_labels(lab, 2)
    assert up.shape == (80, 80) and up.max() == 4


def test_imagej_line_roi_roundtrip(tmp_path):
    hdr = bytearray(64)
    hdr[0:4] = b"Iout"
    hdr[6] = 3  # line
    hdr[18:34] = struct.pack(">ffff", 9.0, 149.0, 235.0, 143.0)
    p = tmp_path / "reslice.roi"
    p.write_bytes(bytes(hdr))
    roi = A.read_line_roi(p)
    assert roi == {"x1": 9.0, "y1": 149.0, "x2": 235.0, "y2": 143.0}
    pt = A.line_point(roi, 0.0)
    assert np.allclose(pt, [149.0, 9.0])  # (row, col)
    d = A.line_direction(roi)
    assert np.isclose(np.hypot(*d), 1.0) and d[1] > 0
    hdr[6] = 1
    p.write_bytes(bytes(hdr))
    with pytest.raises(ValueError):
        A.read_line_roi(p)


def test_recoil_velocity_from_tracks():
    frames = np.arange(5, 15, dtype=float)
    left = np.stack([70.0 - 1.0 * (frames - 5), frames], axis=1)
    right = np.stack([90.0 + 1.5 * (frames - 5), frames], axis=1)
    rec = A.recoil(left, right, n_fit=4)
    assert rec["cut_frame"] == 5.0
    assert np.isclose(rec["initial_separation_px"], 20.0)
    assert np.isclose(rec["recoil_px_per_frame"], 2.5)
    assert np.isclose(rec["recoil_um_per_s"], 2.5 * (A.FIELD_UM / 255.0) / A.FRAME_INTERVAL_S)
    # tracks with no common frames give NaN rather than a number
    bad = A.recoil(left, np.array([[90.0, 40.0], [95.0, 41.0]]))
    assert np.isnan(bad["recoil_px_per_frame"])
