import json

import numpy as np
import pandas as pd
import pytest

from endopigraph.ajmorph import AJMORPH_CLASSES
from pimorph.complex import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.fields import LEGACY_FEATURE_COLUMNS, profile_all_edges
from pimorph.io import legacy_pair_frame, write_legacy_outputs

from .test_fields import render_marker

CELL_COLS = {"cell_id", "face_id", "area_px", "cx", "cy", "perimeter_px", "n_neighbors", "degree", "touches_border"}
EDGE_BASE_COLS = {"cell_i", "cell_j", "edge_id", "component_index", "contact_px", "arclength_um"}


@pytest.fixture
def voronoi_outputs(voronoi_small, tmp_path):
    cx = extract_complex(voronoi_small, pixel_size_um=0.5)
    smooth_complex(cx)
    img = render_marker(cx, cx.cell_cell_edges(), voronoi_small.shape)
    prof = profile_all_edges(cx, img, threshold=0.3)
    paths = write_legacy_outputs(cx, tmp_path, "img_static_01", profiles_df=prof, labels=voronoi_small)
    return cx, prof, paths, tmp_path


def test_output_files_and_columns(voronoi_outputs):
    cx, prof, paths, root = voronoi_outputs
    assert set(paths) == {"cells", "edges", "graphml", "json", "complex"}
    assert all(p.exists() for p in paths.values())
    assert paths["cells"].parent == root / "img_static_01"
    cells = pd.read_csv(paths["cells"])
    edges = pd.read_csv(paths["edges"])
    assert CELL_COLS <= set(cells.columns)
    assert EDGE_BASE_COLS <= set(edges.columns)
    assert set(LEGACY_FEATURE_COLUMNS) <= set(edges.columns)
    assert {"AJ_morph_label", "aj_morph", "aj_occupancy", "has_AJ"} <= set(edges.columns)
    assert len(cells) == cx.cell_faces.size
    assert len(edges) == cx.cell_cell_edges().size
    assert (edges["cell_i"] < edges["cell_j"]).all()
    assert (edges["AJ_morph_label"] == edges["aj_morph"]).all()
    assert set(edges["aj_morph"]) <= set(AJMORPH_CLASSES)


def test_cells_use_original_labels_and_pixel_areas(voronoi_outputs, voronoi_small):
    cx, _, paths, _ = voronoi_outputs
    cells = pd.read_csv(paths["cells"])
    counts = np.bincount(voronoi_small.ravel())
    assert (cells["cell_id"].values == cx.face_label[cells["face_id"].values]).all()
    assert np.allclose(cells["area_px"].values, counts[cells["cell_id"].values])
    assert (cells["degree"] == cells["n_neighbors"]).all()
    assert cells["cx"].between(0, voronoi_small.shape[1]).all()
    assert cells["cy"].between(0, voronoi_small.shape[0]).all()
    assert cells["touches_border"].dtype == bool


def test_edges_contact_px_matches_profiles_and_units(voronoi_outputs):
    cx, prof, paths, _ = voronoi_outputs
    edges = pd.read_csv(paths["edges"]).set_index("edge_id")
    p = prof.set_index("edge_id")
    assert np.allclose(edges.loc[p.index, "contact_px"], p["arclength_px"])
    assert np.allclose(edges["arclength_um"], edges["contact_px"] * 0.5)
    assert np.allclose(edges.loc[p.index, "AJ_occupancy"], p["occupancy"])
    assert np.allclose(edges.loc[p.index, "aj_occupancy"], p["occupancy"])
    assert np.allclose(edges.loc[p.index, "AJ_cluster_count"], p["n_segments"])


def test_graph_json_nodes_equal_cell_faces(voronoi_outputs):
    cx, _, paths, _ = voronoi_outputs
    g = json.loads(paths["json"].read_text())
    assert len(g["nodes"]) == cx.cell_faces.size
    assert set(n["id"] for n in g["nodes"]) == set(int(x) for x in cx.face_label[cx.cell_faces])
    edges = pd.read_csv(paths["edges"])
    n_pairs = edges.groupby(["cell_i", "cell_j"]).ngroups
    assert len(g["edges"]) == n_pairs
    assert all("AJ" in e["junction_types"] for e in g["edges"] if e["has_AJ"])
    assert paths["graphml"].stat().st_size > 0
    cd = json.loads(paths["complex"].read_text())
    assert len(cd["edge_tail"]) == cx.n_edges and cd["pixel_size_um"] == 0.5


def test_double_contact_two_components(double_contact, tmp_path):
    cx = extract_complex(double_contact)
    smooth_complex(cx)
    paths = write_legacy_outputs(cx, tmp_path, "dc", labels=double_contact)
    edges = pd.read_csv(paths["edges"])
    assert len(edges) == 2
    assert (edges["cell_i"] == 1).all() and (edges["cell_j"] == 2).all()
    assert sorted(edges["component_index"]) == [0, 1]
    assert (edges["n_components"] == 2).all()
    assert edges["arclength_um"].isna().all()
    assert "aj_morph" not in edges.columns
    g = json.loads(paths["json"].read_text())
    assert len(g["nodes"]) == 2 and len(g["edges"]) == 1
    assert g["edges"][0]["n_components"] == 2
    assert np.isclose(g["edges"][0]["contact_px"], edges["contact_px"].sum())


def test_pair_frame_collapses_components(double_contact):
    cx = extract_complex(double_contact)
    from pimorph.io import legacy_edges_frame

    edges = legacy_edges_frame(cx)
    pairs = legacy_pair_frame(edges)
    assert len(pairs) == 1
    assert pairs.loc[0, "n_components"] == 2
    assert json.loads(pairs.loc[0, "edge_ids"]) == sorted(int(e) for e in cx.cell_cell_edges())


def test_partial_profiles_leave_other_edges_unknown(voronoi_small, tmp_path):
    cx = extract_complex(voronoi_small)
    smooth_complex(cx)
    cc = cx.cell_cell_edges()
    img = render_marker(cx, cc, voronoi_small.shape)
    sub = [int(e) for e in cc[:3]]
    prof = profile_all_edges(cx, img, threshold=0.3, edges=sub)
    paths = write_legacy_outputs(cx, tmp_path, "partial", profiles_df=prof)
    edges = pd.read_csv(paths["edges"]).set_index("edge_id")
    assert edges.loc[sub, "has_AJ"].all()
    others = edges.index.difference(sub)
    assert not edges.loc[others, "has_AJ"].any()
    assert (edges.loc[others, "aj_morph"] == "unknown").all()
    assert edges.loc[others, "AJ_occupancy"].isna().all()
    g = json.loads(paths["json"].read_text())
    flagged = [e for e in g["edges"] if e["has_AJ"]]
    assert 0 < len(flagged) <= len(sub)


def test_quadrants_without_profiles(quadrants, tmp_path):
    cx = extract_complex(quadrants)
    paths = write_legacy_outputs(cx, tmp_path, "quad")
    cells = pd.read_csv(paths["cells"])
    edges = pd.read_csv(paths["edges"])
    assert sorted(cells["cell_id"]) == [1, 2, 3, 4]
    # polygon area when no label image is given
    assert np.allclose(cells["area_px"], 100.0)
    assert (cells["n_neighbors"] == 2).all()
    assert cells["touches_border"].all()
    assert len(edges) == 4
    assert set(zip(edges["cell_i"], edges["cell_j"])) == {(1, 2), (1, 3), (2, 4), (3, 4)}
    assert np.allclose(edges["contact_px"], 10.0)
