import numpy as np
import pytest
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from pimorph.complex import FaceKind, extract_complex, validate
from pimorph.synth import (
    CHANNEL_KEYS,
    RenderParams,
    SynthTissueParams,
    TARGET_KEYS,
    generate_tissue,
    load_tile,
    make_dataset,
    make_targets,
    render_channels,
)

SHAPE = (160, 160)


def _params(**kw) -> SynthTissueParams:
    base = dict(shape=SHAPE, n_cells=50, seed=11)
    base.update(kw)
    return SynthTissueParams(**base)


@pytest.fixture(scope="module")
def plain():
    return generate_tissue(_params())


@pytest.fixture(scope="module")
def tri_gaps():
    return generate_tissue(_params(n_gaps=6, gap_kind="tricellular", boundary_jitter_px=1.0, seed=5))


@pytest.fixture(scope="module")
def bi_gaps():
    return generate_tissue(_params(n_gaps=5, gap_kind="bicellular", seed=6))


# ------------------------------------------------------------------ tissue
def test_labels_connected_and_gap_free(plain):
    lab = plain.labels
    assert lab.dtype == np.int32 and lab.shape == SHAPE
    assert (lab > 0).all()  # no gaps requested: no background at all
    n_labels = np.unique(lab).size
    assert n_labels == lab.max()  # consecutive 1..K
    assert int(cc_label(lab, connectivity=1).max()) == n_labels  # every label is one 4-connected region
    assert 40 <= n_labels <= 50
    assert validate(extract_complex(lab)).ok


def test_tricellular_gaps(tri_gaps):
    lab = tri_gaps.labels
    cx = extract_complex(lab)
    assert validate(cx).ok
    n_labels = np.unique(lab[lab > 0]).size
    assert int(cc_label(lab, connectivity=1).max()) == n_labels
    assert 1 <= cx.gap_faces.size <= 6
    assert cx.gap_faces.size == tri_gaps.n_gaps
    # each tricellular gap replaced a vertex, so it is bounded by >= 3 distinct cells
    for g in cx.gap_faces:
        assert len(set(cx.face_neighbors(int(g), kinds=(FaceKind.CELL,)))) >= 3
    # background only inside enclosed gaps
    tg = make_targets(lab, cx=cx)
    assert not tg["outer"].any()
    assert np.array_equal(tg["gap"], lab == 0)


def test_bicellular_gaps(bi_gaps):
    cx = extract_complex(bi_gaps.labels)
    assert validate(cx).ok
    assert 1 <= cx.gap_faces.size <= 5
    for g in cx.gap_faces:
        assert len(set(cx.face_neighbors(int(g), kinds=(FaceKind.CELL,)))) == 2
    assert all(k == "bicellular" for k in bi_gaps.gap_kinds)


def test_mixed_gaps_and_area_fraction():
    t = generate_tissue(_params(gap_area_fraction=0.03, gap_kind="mixed", seed=8))
    cx = extract_complex(t.labels)
    assert validate(cx).ok
    assert cx.gap_faces.size >= 1
    frac = (t.labels == 0).mean()
    assert 0.01 < frac < 0.06
    assert set(t.gap_kinds) <= {"tricellular", "bicellular"}


def test_elongation_increases_aspect_ratio():
    def mean_ar(e):
        t = generate_tissue(_params(elongation=e, flow_angle_deg=35.0, seed=2))
        props = regionprops(t.labels)
        return float(np.mean([p.axis_major_length / max(p.axis_minor_length, 1e-6) for p in props]))

    assert mean_ar(2.5) >= 1.4 * mean_ar(1.0)


def test_jitter_keeps_connectivity_and_changes_labels():
    a = generate_tissue(_params(seed=4))
    b = generate_tissue(_params(seed=4, boundary_jitter_px=2.0))
    assert (b.labels > 0).all()
    assert int(cc_label(b.labels, connectivity=1).max()) == np.unique(b.labels).size
    assert (a.labels != b.labels).mean() > 0.02
    assert validate(extract_complex(b.labels)).ok


def test_nuclei_inside_cells_and_fractions(plain):
    for (r, c), lab in zip(plain.nuclei_xy, plain.nuclei_cell):
        assert plain.labels[int(round(r)), int(round(c))] == lab
    assert plain.nuclei_xy.shape[0] == plain.n_cells
    t = generate_tissue(_params(binucleate_fraction=0.3, anucleate_fraction=0.2, seed=9))
    counts = np.bincount(t.nuclei_cell, minlength=t.n_cells + 1)[1:]
    assert (counts == 2).any() and (counts == 0).any() and (counts == 1).any()


def test_generation_is_deterministic():
    a = generate_tissue(_params(n_gaps=3, gap_kind="mixed", boundary_jitter_px=1.0, seed=21))
    b = generate_tissue(_params(n_gaps=3, gap_kind="mixed", boundary_jitter_px=1.0, seed=21))
    assert np.array_equal(a.labels, b.labels) and np.allclose(a.nuclei_xy, b.nuclei_xy)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        generate_tissue(_params(gap_kind="hexacellular"))
    with pytest.raises(ValueError):
        generate_tissue(_params(elongation=0.5))


# ------------------------------------------------------------------ render
@pytest.fixture(scope="module")
def rendered(plain):
    rp = RenderParams(psf_sigma_px=1.2, junction_width_px=2.0, broken_fraction=0.0, nucleus_intensity=400.0, seed=3)
    return render_channels(plain, rp)


@pytest.fixture(scope="module")
def rendered_broken(plain):
    rp = RenderParams(psf_sigma_px=1.2, junction_width_px=2.0, broken_fraction=0.4, seed=3)
    return render_channels(plain, rp)


def test_render_shapes_dtypes(rendered, plain):
    for key in ("junction", "membrane", "nuclei"):
        arr = rendered[key]
        assert arr.shape == plain.labels.shape and arr.dtype == np.float32
        assert (arr >= 0).all()
    assert rendered["broken_mask"].dtype == bool and rendered["broken_mask"].shape == plain.labels.shape
    assert not rendered["broken_mask"].any()


def test_render_junction_contrast(rendered, plain):
    bnd = find_boundaries(plain.labels, connectivity=1, mode="inner")
    interior = (plain.labels > 0) & ~bnd
    J = rendered["junction"]
    assert J[bnd].mean() > 2.0 * J[interior].mean()


def test_render_broken_segments(rendered_broken, plain):
    bnd = find_boundaries(plain.labels, connectivity=1, mode="inner")
    B = rendered_broken["broken_mask"]
    frac = (B & bnd).sum() / bnd.sum()
    assert 0.25 <= frac <= 0.55
    J = rendered_broken["junction"]
    assert J[B & bnd].mean() < 0.5 * J[bnd & ~B].mean()
    # the membrane channel does not know about breaks
    M = rendered_broken["membrane"]
    assert M[B & bnd].mean() > 0.8 * M[bnd & ~B].mean()
    assert M[bnd].mean() > 2.0 * M[(plain.labels > 0) & ~bnd].mean()


def test_render_nuclei_peaks(rendered, plain):
    N = rendered["nuclei"]
    H, W = N.shape
    for r, c in plain.nuclei_xy:
        r0, c0 = int(round(r)), int(round(c))
        rs = slice(max(r0 - 4, 0), min(r0 + 5, H))
        cs = slice(max(c0 - 4, 0), min(c0 + 5, W))
        win = N[rs, cs]
        i, j = np.unravel_index(int(np.argmax(win)), win.shape)
        assert np.hypot(rs.start + i - r, cs.start + j - c) <= 2.0


def test_render_ranges_and_determinism(plain):
    rp = RenderParams(psf_sigma_px=(0.8, 2.5), junction_width_px=(1.0, 3.0), seed=12)
    a = render_channels(plain, rp)
    b = render_channels(plain, rp)
    assert 0.8 <= a["params"].psf_sigma_px <= 2.5 and 1.0 <= a["params"].junction_width_px <= 3.0
    assert np.array_equal(a["junction"], b["junction"])


# ----------------------------------------------------------------- targets
def test_targets_boundary_and_distance(tri_gaps):
    lab = tri_gaps.labels
    tg = make_targets(lab)
    assert set(TARGET_KEYS) <= set(tg)
    for v in tg.values():
        assert v.shape == lab.shape
    assert np.array_equal(tg["boundary"], find_boundaries(lab, connectivity=1, mode="inner"))
    sd = tg["signed_distance"]
    assert sd.dtype == np.float32
    assert (sd[lab > 0] > 0).all() and (sd[lab == 0] < 0).all()
    assert sd.max() <= 16.0 and sd.min() >= -16.0
    assert np.allclose(sd[tg["boundary"]], 0.5)


def test_targets_vertex_heatmap(plain):
    cx = extract_complex(plain.labels)
    tg = make_targets(plain.labels, cx=cx)
    V = tg["vertex"]
    assert V.dtype == np.float32 and 0.9 < V.max() <= 1.0
    true = np.array([cx.vertex_xy[v] for v in range(cx.n_vertices) if len(cx.vertex_cell_set(v)) >= 3])
    peaks = peak_local_max(V, min_distance=1, threshold_abs=0.5, exclude_border=False)
    assert peaks.shape[0] >= 0.8 * true.shape[0]
    d = np.sqrt(((peaks[:, None, :] - true[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert (d <= 2.0).all()


def test_targets_seed_heatmap_uses_nuclei(plain):
    tg = make_targets(plain.labels, nuclei_xy=plain.nuclei_xy)
    S = tg["seed"]
    for r, c in plain.nuclei_xy:
        assert S[int(round(r)), int(round(c))] > 0.9
    assert not tg["gap"].any() and not tg["outer"].any()


def test_make_dataset_writes_tiles(tmp_path):
    def tissue_sampler(rng):
        return SynthTissueParams(shape=(96, 96), n_cells=int(rng.integers(12, 20)), n_gaps=2, gap_kind="mixed")

    def render_sampler(rng):
        return RenderParams(psf_sigma_px=float(rng.uniform(0.8, 2.5)), broken_fraction=float(rng.uniform(0, 0.4)))

    manifest = make_dataset(3, tmp_path, tissue_sampler, render_sampler, seed=1)
    assert len(manifest) == 3
    assert (tmp_path / "manifest.csv").exists()
    for col in ("file", "n_cells", "elongation", "gap_kind", "psf_sigma_px", "broken_fraction", "junction_intensity"):
        assert col in manifest.columns
    files = sorted(tmp_path.glob("tile_*.npz"))
    assert len(files) == 3
    tile = load_tile(files[0])
    for key in CHANNEL_KEYS + TARGET_KEYS + ("labels", "nuclei_xy"):
        assert key in tile
    assert tile["junction"].shape == (96, 96) and tile["labels"].dtype == np.int32
    assert manifest["psf_sigma_px"].between(0.8, 2.5).all()
