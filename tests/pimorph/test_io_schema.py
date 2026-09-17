import json

import numpy as np

from pimorph.complex import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.io import complex_tables, write_tables


def test_tables_shapes_and_units(voronoi_small, tmp_path):
    cx = extract_complex(voronoi_small, pixel_size_um=0.4)
    smooth_complex(cx)
    t = complex_tables(cx, labels=voronoi_small)
    assert len(t.cells) == cx.cell_faces.size
    assert len(t.interfaces) == cx.n_edges
    assert len(t.vertices) == cx.n_vertices
    assert len(t.gaps) == cx.gap_faces.size
    assert t.provenance["units"] == "um"
    assert np.all(np.isfinite(t.cells["area_um2"]))
    assert set(t.interfaces["kind"]) <= {"cell_cell", "cell_gap", "cell_outer", "gap_outer", "gap_gap"}
    # pixel areas present and close to polygon areas
    rel = (t.cells["area_poly_px"] - t.cells["area_pixels"]).abs() / t.cells["area_pixels"]
    assert rel.max() < 0.1
    paths = write_tables(t, tmp_path, fmt="csv")
    assert all(p.exists() for p in paths.values())
    prov = json.loads(paths["provenance"].read_text())
    assert prov["pixel_size_um"] == 0.4


def test_units_px_when_uncalibrated(double_contact):
    cx = extract_complex(double_contact)
    t = complex_tables(cx)
    assert t.provenance["units"] == "px"
    assert t.interfaces["arclength_um"].isna().all()
    assert t.interfaces["n_components_between_pair"].max() == 2
    v = t.vertices
    assert (v["degree"] == 3).all()
    assert json.loads(v["incident_faces_cyclic"].iloc[0]).__len__() == 3
