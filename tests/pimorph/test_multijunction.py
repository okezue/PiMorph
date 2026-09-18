"""Multi-channel junction fields: aligned per-bin frames, cross-channel statistics,
vector states and joint morphology states on a synthetic two-marker complex."""

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.filters import gaussian

from pimorph.complex import extract_complex
from pimorph.complex.geometry import smooth_complex
from pimorph.fields.multichannel import (
    MultiProfile,
    all_patterns,
    default_feature_columns,
    joint_morphology_states,
    multichannel_profiles,
    pattern_name,
)

from .conftest import voronoi_labels


def _disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2


def _trace_mask(cx, edge_ids, shape):
    m = np.zeros(shape, dtype=bool)
    hi = np.array(shape) - 1
    for e in edge_ids:
        p = np.clip(np.rint(cx.edge_polyline[int(e)]).astype(int), 0, hi)
        for a, b in zip(p[:-1], p[1:]):
            rr, cc = line(a[0], a[1], b[0], b[1])
            m[rr, cc] = True
    return m


def render_marker(cx, edge_ids, shape, band=2, sigma=1.0, floor=0.05, exclude=None, radius=2):
    """Bright blurred band along ``edge_ids``; pixels within ``radius`` of ``exclude``
    edges are kept dark so a marker painted on some edges does not bleed onto the
    others at shared vertices."""
    m = ndi.binary_dilation(_trace_mask(cx, edge_ids, shape), structure=_disk(band))
    if exclude is not None and len(exclude):
        m &= ndi.distance_transform_edt(~_trace_mask(cx, exclude, shape)) > radius
    g = gaussian(m.astype(float), sigma=sigma)
    g = g / g.max() if g.max() > 0 else g
    return floor + (1.0 - floor) * g


@pytest.fixture(scope="module")
def synthetic():
    lab = voronoi_labels(16, shape=(192, 192), seed=2, gaps=2)
    cx = extract_complex(lab)
    smooth_complex(cx)
    cc = cx.cell_cell_edges()
    b_edges = cc[cc % 2 == 0]
    nb_edges = cc[cc % 2 == 1]
    A = render_marker(cx, cc, lab.shape)
    B = render_marker(cx, b_edges, lab.shape, exclude=nb_edges)
    return {"labels": lab, "cx": cx, "A": A, "B": B, "b_edges": b_edges, "nb_edges": nb_edges}


@pytest.fixture(scope="module")
def mp(synthetic) -> MultiProfile:
    return multichannel_profiles(synthetic["cx"], {"AJ": synthetic["A"], "TJ": synthetic["B"]})


def test_per_bin_frames_align_across_channels(mp, synthetic):
    cx = synthetic["cx"]
    pb = mp.per_bin
    assert set(pb["edge_id"]) == set(int(e) for e in cx.cell_cell_edges())
    for c in ("AJ", "TJ"):
        for suffix in ("intensity", "occupancy", "width_px", "occupied"):
            assert f"{c}_{suffix}" in pb.columns
    # one row per (edge, bin) with no channel dropping or duplicating bins
    assert not pb.duplicated(["edge_id", "s_bin"]).any()
    for e in mp.edge_ids[:10]:
        rows = pb[pb["edge_id"] == e].sort_values("s_bin")
        assert np.array_equal(rows["s_bin"].to_numpy(), np.arange(len(rows)))
        nb = int(mp.per_edge.loc[mp.per_edge["edge_id"] == e, "n_bins"].iloc[0])
        assert len(rows) == nb == mp.profiles["AJ"][int(e)].n_bins == mp.profiles["TJ"][int(e)].n_bins
        assert np.allclose(rows["s_hi_px"].iloc[-1], mp.profiles["AJ"][int(e)].arclength_px)
    assert mp.vector_field(mp.edge_ids[0]).shape[1] == 2
    assert mp.occupancy_matrix(mp.edge_ids[0]).dtype == bool


def test_channel_a_covers_all_edges(mp):
    pe = mp.per_edge
    assert len(pe) == len(mp.edge_ids)
    assert pe["AJ_coverage"].min() >= 0.9
    assert pe["AJ_continuity"].min() >= 0.9


def test_jaccard_separates_painted_and_unpainted_edges(mp, synthetic):
    pe = mp.per_edge.set_index("edge_id")
    jb = pe.loc[synthetic["b_edges"], "jaccard_AJ_TJ"]
    jn = pe.loc[synthetic["nb_edges"], "jaccard_AJ_TJ"]
    assert jb.min() > 0.8 and jb.mean() > 0.9
    assert jn.max() < 0.05
    # exclusive fractions mirror the painting
    assert pe.loc[synthetic["nb_edges"], "exclusive_AJ"].min() > 0.95
    assert pe.loc[synthetic["b_edges"], "exclusive_AJ"].max() < 0.2
    assert pe["exclusive_TJ"].max() < 0.05


def test_co_occupancy_about_half(mp, synthetic):
    pe = mp.per_edge
    frac_b = len(synthetic["b_edges"]) / len(mp.edge_ids)
    co = pe["co_occupancy_AJ_TJ"].mean()
    # half the edges carry TJ; their coverage is < 1 near the suppressed vertices
    assert 0.8 * frac_b <= co <= frac_b + 0.02
    # with AJ everywhere, co-occupancy equals TJ coverage on every edge
    assert np.allclose(pe["co_occupancy_AJ_TJ"], pe["TJ_coverage"], atol=0.05)


def test_vector_state_counts_match_painting(mp, synthetic):
    pe = mp.per_edge.set_index("edge_id")
    assert (pe.loc[synthetic["b_edges"], "vector_state"] == "AJ+TJ").all()
    assert (pe.loc[synthetic["nb_edges"], "vector_state"] == "AJ only").all()
    assert pe["vector_state_fraction"].between(0.0, 1.0).all()
    frac_cols = [f"frac_{p}" for p in mp.patterns]
    assert set(frac_cols) <= set(pe.columns)
    assert np.allclose(pe[frac_cols].sum(axis=1), 1.0)


def test_pattern_names():
    assert all_patterns(["AJ", "TJ"]) == ["none", "TJ only", "AJ only", "AJ+TJ"]
    assert pattern_name(["AJ", "TJ", "actin"], (True, False, True)) == "AJ+actin"
    assert pattern_name(["AJ"], (True,)) == "AJ"


def test_thresholds_shared_binning_and_explicit_thresholds(synthetic):
    cx = synthetic["cx"]
    mp1 = multichannel_profiles(cx, {"AJ": synthetic["A"], "TJ": synthetic["B"]}, thresholds={"AJ": 0.5, "TJ": 0.5})
    assert mp1.thresholds == {"AJ": 0.5, "TJ": 0.5}
    mp2 = multichannel_profiles(cx, {"AJ": synthetic["A"], "TJ": synthetic["B"]}, bin_px=2.0)
    assert mp2.bin_px == 2.0
    assert len(mp2.per_bin) < len(mp1.per_bin)
    with pytest.raises(ValueError):
        multichannel_profiles(cx, {"AJ": synthetic["A"]}, n_bins_per_um=2.0)
    cx.pixel_size_um = 0.5
    try:
        mp3 = multichannel_profiles(cx, {"AJ": synthetic["A"]}, n_bins_per_um=2.0)
        assert np.isclose(mp3.bin_px, 1.0)
    finally:
        cx.pixel_size_um = None


def test_constant_third_channel_is_never_occupied(synthetic):
    cx = synthetic["cx"]
    flat = np.full(synthetic["A"].shape, 0.05)
    mp3 = multichannel_profiles(cx, {"AJ": synthetic["A"], "TJ": synthetic["B"], "actin": flat})
    assert mp3.per_edge["actin_coverage"].max() == 0.0
    assert mp3.per_edge["exclusive_actin"].max() == 0.0
    assert set(mp3.per_edge["vector_state"]) <= {"AJ+TJ", "AJ only"}
    assert "co_occupancy_TJ_actin" in mp3.per_edge.columns and "pearson_AJ_actin" in mp3.per_edge.columns


def test_joint_states_recover_painting(mp, synthetic):
    from sklearn.metrics import adjusted_rand_score

    pe = mp.per_edge
    labels, meta = joint_morphology_states(pe, default_feature_columns(["AJ", "TJ"]), k_range=(2, 4), seed=0)
    assert labels.shape == (len(pe),)
    assert 2 <= meta["k"] <= 4
    assert set(meta["bic"]) == {2, 3, 4} or min(meta["bic"]) == 2
    is_b = pe["edge_id"].isin(synthetic["b_edges"]).to_numpy()
    assert adjusted_rand_score(is_b, labels) > 0.9
    assert meta["stability_ari_mean"] > 0.7
    assert sum(meta["state_counts"].values()) == len(pe)


def test_joint_states_handle_nan_rows(mp):
    pe = mp.per_edge.copy()
    pe.loc[pe.index[:3], "TJ_width_mean_px"] = np.nan
    labels, meta = joint_morphology_states(pe, ["AJ_coverage", "TJ_coverage", "TJ_width_mean_px"], k_range=(2, 3))
    assert labels.min() >= 0 and len(labels) == len(pe)
    with pytest.raises(ValueError):
        joint_morphology_states(pe, ["does_not_exist"])
