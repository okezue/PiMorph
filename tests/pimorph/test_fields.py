import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.filters import gaussian

from endopigraph.ajmorph import AJMORPH_CLASSES
from pimorph.complex import extract_complex
from pimorph.complex.geometry import polyline_length, smooth_complex
from pimorph.fields import (
    LEGACY_FEATURE_COLUMNS,
    edge_profile,
    edge_strip,
    heuristic_morph_label,
    legacy_feature_frame,
    morph_labels,
    profile_all_edges,
    resample_polyline,
    sample_strip,
    weighted_layer,
)

THR = 0.3


def _disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2


def render_marker(cx, edge_ids, shape, band=2, sigma=1.0, floor=0.05):
    """Bright blurred band along the crack traces of ``edge_ids`` over a dim interior."""
    m = np.zeros(shape, dtype=bool)
    hi = np.array(shape) - 1
    for e in edge_ids:
        p = np.clip(np.rint(cx.edge_polyline[int(e)]).astype(int), 0, hi)
        for a, b in zip(p[:-1], p[1:]):
            rr, cc = line(a[0], a[1], b[0], b[1])
            m[rr, cc] = True
    m = ndi.binary_dilation(m, structure=_disk(band))
    g = gaussian(m.astype(float), sigma=sigma)
    g = g / g.max() if g.max() > 0 else g
    return floor + (1.0 - floor) * g


def _longest_edge(cx):
    cc = cx.cell_cell_edges()
    lens = np.array([polyline_length(cx.edge_geometry(int(e))) for e in cc])
    return int(cc[np.argmax(lens)])


def _edge_vertex_sets(cx, e):
    return {int(cx.edge_tail[e]), int(cx.edge_head[e])}


@pytest.fixture
def cx_voronoi(voronoi_small):
    cx = extract_complex(voronoi_small)
    smooth_complex(cx)
    return cx


@pytest.fixture
def bright_all(cx_voronoi, voronoi_small):
    return render_marker(cx_voronoi, cx_voronoi.cell_cell_edges(), voronoi_small.shape)


# ------------------------------------------------------------------- strips
def test_resample_polyline_uniform_spacing():
    p = np.array([[0.0, 0.0], [0.0, 4.0], [3.0, 4.0]])  # length 7
    pts, t, n, s = resample_polyline(p, step_px=1.0)
    assert pts.shape == (7, 2) and t.shape == (7, 2) and n.shape == (7, 2) and s.shape == (7,)
    assert np.allclose(np.diff(s), 1.0)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    # travelling +col: left on screen is -row
    assert np.allclose(n[1], [-1.0, 0.0])
    pts_e, _, _, s_e = resample_polyline(p, step_px=1.0, endpoints=True)
    assert np.allclose(pts_e[0], p[0]) and np.allclose(pts_e[-1], p[-1])
    assert s_e[0] == 0.0 and np.isclose(s_e[-1], 7.0)


def test_strip_shapes_single_and_multichannel(cx_voronoi, bright_all):
    e = _longest_edge(cx_voronoi)
    st = edge_strip(cx_voronoi, e, bright_all, half_width_px=3.0, step_px=1.0, n_lateral=7)
    S = st.n_samples
    assert st.values.shape == (S, 7)
    assert S == max(1, int(round(st.arclength_px / 1.0)))
    multi = np.stack([bright_all, 0.5 * bright_all])
    st3 = edge_strip(cx_voronoi, e, multi, half_width_px=3.0, n_lateral=7)
    assert st3.values.shape == (2, S, 7)
    assert np.allclose(st3.values[1], 0.5 * st3.values[0])
    direct = sample_strip(bright_all, st.points, st.normals, half_width_px=3.0, n_lateral=7)
    assert np.allclose(direct, st.values)


def test_strip_left_side_is_edge_faces_0(cx_voronoi, voronoi_small):
    e = _longest_edge(cx_voronoi)
    a = int(cx_voronoi.edge_faces[e, 0])
    img = (voronoi_small == cx_voronoi.face_label[a]).astype(float)
    st = edge_strip(cx_voronoi, e, img, half_width_px=2.0, n_lateral=5, order=0)
    left = st.values[:, st.r > 0]
    right = st.values[:, st.r < 0]
    assert left.mean() > 0.9
    assert right.mean() < 0.1


# ----------------------------------------------------------------- profiles
def test_left_right_intensity_asymmetry(cx_voronoi, voronoi_small):
    e = _longest_edge(cx_voronoi)
    a = int(cx_voronoi.edge_faces[e, 0])
    img = (voronoi_small == cx_voronoi.face_label[a]).astype(float)
    prof = edge_profile(cx_voronoi, e, img, threshold=0.5, half_width_px=2.0, n_lateral=5)
    assert prof.left_intensity > prof.right_intensity
    assert np.nanmean(prof.left_intensity_s) > np.nanmean(prof.right_intensity_s)
    assert prof.left_intensity_s.shape == (prof.n_bins,)
    assert prof.bin_edges_px.shape == (prof.n_bins + 1,)
    assert np.isclose(prof.bin_edges_px[-1], prof.arclength_px)


def test_occupancy_bright_vs_dark(cx_voronoi, voronoi_small):
    e = _longest_edge(cx_voronoi)
    img = render_marker(cx_voronoi, [e], voronoi_small.shape)
    bright = edge_profile(cx_voronoi, e, img, threshold=THR, half_width_px=2.0)
    assert bright.occupancy > 0.9
    assert bright.continuity == 1.0 and bright.n_segments == 1 and bright.gap_fraction == 0.0
    # a dark edge sharing no vertex with the bright one is entirely below threshold
    ve = _edge_vertex_sets(cx_voronoi, e)
    dark_ids = [int(d) for d in cx_voronoi.cell_cell_edges() if d != e and not (_edge_vertex_sets(cx_voronoi, d) & ve)]
    far = max(
        dark_ids,
        key=lambda d: np.min(
            np.linalg.norm(cx_voronoi.edge_geometry(d).mean(axis=0) - cx_voronoi.edge_geometry(e), axis=1)
        ),
    )
    dark = edge_profile(cx_voronoi, far, img, threshold=THR, half_width_px=2.0)
    assert dark.occupancy < 0.05
    assert dark.n_segments == 0 and dark.continuity == 0.0 and dark.gap_fraction == 1.0
    assert np.isnan(dark.width_mean_px)


def test_continuity_and_segments_with_blanked_middle_third(cx_voronoi, bright_all):
    e = _longest_edge(cx_voronoi)
    full = edge_profile(cx_voronoi, e, bright_all, threshold=THR, half_width_px=2.0)
    assert full.continuity == 1.0 and full.n_segments == 1
    pts, _, _, s = resample_polyline(cx_voronoi.edge_geometry(e), step_px=0.5)
    L = s[-1] + 0.25
    mid = pts[(s > L / 3) & (s < 2 * L / 3)]
    img = bright_all.copy()
    rr, cc = np.mgrid[: img.shape[0], : img.shape[1]]
    for r, c in mid:
        img[(rr - r) ** 2 + (cc - c) ** 2 <= 4.5**2] = 0.0
    blanked = edge_profile(cx_voronoi, e, img, threshold=THR, half_width_px=2.0)
    assert blanked.continuity < 0.6
    assert blanked.n_segments == 2
    assert 0.2 < blanked.gap_fraction < 0.6
    assert blanked.occupancy < full.occupancy


def test_profile_all_edges_rows_and_otsu(cx_voronoi, bright_all):
    cc = cx_voronoi.cell_cell_edges()
    df = profile_all_edges(cx_voronoi, bright_all, threshold=THR)
    assert len(df) == cc.size
    assert set(df["edge_id"]) == set(int(e) for e in cc)
    sub = [int(e) for e in cc[:5]]
    df5 = profile_all_edges(cx_voronoi, bright_all, threshold=THR, edges=sub)
    assert df5["edge_id"].tolist() == sub
    assert {"face_left", "face_right", "label_left", "label_right", "occupancy", "continuity"} <= set(df5.columns)
    auto = profile_all_edges(cx_voronoi, bright_all, edges=sub)
    thr = auto["threshold"].iloc[0]
    assert 0.05 < thr < 1.0 and (auto["threshold"] == thr).all()
    assert (auto["occupancy"] > 0.5).all()


# -------------------------------------------------------------- functionals
def test_weighted_layer_integrates_per_bin(cx_voronoi, bright_all):
    df = profile_all_edges(cx_voronoi, bright_all, threshold=THR, edges=[_longest_edge(cx_voronoi)])
    w_len = weighted_layer(df, lambda p: np.ones(p.n_bins))
    w_occ = weighted_layer(df, lambda p: p.occupancy_s)
    w_const = weighted_layer(df, lambda p: 1.0)
    e = int(df["edge_id"].iloc[0])
    L = float(df["arclength_px"].iloc[0])
    assert np.isclose(w_len[e], L) and np.isclose(w_const[e], L)
    assert 0 < w_occ[e] <= L + 1e-9
    with pytest.raises(ValueError):
        weighted_layer(df, lambda p: np.ones(p.n_bins + 1))


def test_legacy_feature_frame_columns(cx_voronoi, bright_all):
    df = profile_all_edges(cx_voronoi, bright_all, threshold=THR)
    feats = legacy_feature_frame(df)
    assert set(LEGACY_FEATURE_COLUMNS) <= set(feats.columns)
    assert len(LEGACY_FEATURE_COLUMNS) == 7
    assert len(feats) == len(df)
    assert np.allclose(feats["AJ_skeleton_len"], df["arclength_px"] * df["occupancy"])
    assert (feats["AJ_cluster_count"] == df["n_segments"]).all()
    assert np.allclose(feats["AJ_cluster_density"], df["n_segments"] / df["arclength_px"])
    assert np.allclose(feats["AJ_thickness_proxy"], df["width_mean_px"], equal_nan=True)


def test_heuristic_labels_are_legacy_classes(cx_voronoi, bright_all):
    df = profile_all_edges(cx_voronoi, bright_all, threshold=THR)
    labels = morph_labels(df)
    assert len(labels) == len(df) and labels.name == "AJ_morph_label"
    assert set(labels) <= set(AJMORPH_CLASSES)
    assert all(isinstance(x, str) for x in labels)
    # bright continuous edges must not be called minimal
    assert not (labels == "minimal").any()
    assert heuristic_morph_label({"AJ_occupancy": 0.0, "AJ_cluster_count": 0, "AJ_skeleton_len": 0}) == "minimal"
    assert heuristic_morph_label({"occupancy": float("nan")}) == "unknown"
    row = {"AJ_occupancy": 0.9, "AJ_cluster_count": 1, "AJ_skeleton_len": 40, "AJ_thickness_proxy": 1.0}
    assert heuristic_morph_label(row) == "straight"
