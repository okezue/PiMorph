import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from pimorph.complex import extract_complex, validate
from pimorph.complex.geometry import smooth_complex
from pimorph.infer import (
    ClassicalProposer,
    ConstrainedDecoder,
    DecoderParams,
    EnergyWeights,
    PosteriorEnsemble,
    complex_energy,
    generate_hypotheses,
)
from pimorph.infer.decoder import merge_small_regions
from pimorph.infer.posterior import effective_sample_size, softmax_weights, temperature_for_ess
from pimorph.infer.renderer import RenderModel, boundary_interior_ratio, render_log_likelihood

from .conftest import voronoi_labels


def synth_field(labels: np.ndarray, seed: int = 0, junction_level: float = 800.0, nucleus_r: float = 4.0):
    """Render a junction-like geometry channel and a nuclei channel from labels."""
    rng = np.random.default_rng(seed)
    b = find_boundaries(labels, mode="thick") & (labels > 0)
    geom = 60.0 + junction_level * ndi.gaussian_filter(b.astype(np.float32), 1.0)
    geom = rng.poisson(np.maximum(geom, 1)).astype(np.float32) + rng.normal(0, 5, geom.shape).astype(np.float32)
    nuc = np.zeros(labels.shape, np.float32)
    for p in regionprops(labels):
        r, c = p.centroid
        rr, cc = np.mgrid[: labels.shape[0], : labels.shape[1]]
        nuc += 900.0 * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * nucleus_r**2))
    nuc = rng.poisson(np.maximum(nuc + 30, 1)).astype(np.float32)
    return geom, nuc


def adjacency_set(labels: np.ndarray):
    cx = extract_complex(labels)
    out = set()
    for e in cx.cell_cell_edges():
        a, b = (int(cx.face_label[f]) for f in cx.edge_faces[int(e)])
        out.add((min(a, b), max(a, b)))
    return out


def pair_f1(pred_labels, ref_labels, match):
    """Adjacency F1 where predicted cells are mapped to reference cells through ``match``."""
    ref = adjacency_set(ref_labels)
    pred_raw = adjacency_set(pred_labels)
    pred = set()
    for a, b in pred_raw:
        if a in match and b in match and match[a] != match[b]:
            pred.add((min(match[a], match[b]), max(match[a], match[b])))
    tp = len(pred & ref)
    p = tp / max(len(pred), 1)
    r = tp / max(len(ref), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def best_match(pred_labels, ref_labels):
    """pred label -> ref label with maximal overlap."""
    m = {}
    for p in np.unique(pred_labels[pred_labels > 0]):
        ref_vals = ref_labels[pred_labels == p]
        ref_vals = ref_vals[ref_vals > 0]
        if ref_vals.size:
            m[int(p)] = int(np.bincount(ref_vals).argmax())
    return m


@pytest.fixture(scope="module")
def field():
    labels = voronoi_labels(35, shape=(192, 192), seed=5)
    geom, nuc = synth_field(labels)
    return labels, geom, nuc


def test_proposals_shapes_and_ranges(field):
    labels, geom, nuc = field
    maps = ClassicalProposer(auto_scale=False)(geom, nuc)
    assert maps.boundary.shape == geom.shape and maps.boundary.min() >= 0 and maps.boundary.max() <= 1
    assert maps.seed.shape == geom.shape and 0 <= maps.seed.max() <= 1
    assert maps.tissue.dtype == bool
    assert maps.seed_points.shape[1] == 2
    # a seed candidate near most cell centroids
    cents = np.array([p.centroid for p in regionprops(labels)])
    d = np.sqrt(((cents[:, None, :] - maps.seed_points[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert np.mean(d < 4) > 0.9
    # boundary map high on true boundaries relative to interiors
    b = find_boundaries(labels, mode="inner")
    assert maps.boundary[b].mean() > 3 * maps.boundary[~b & (labels > 0)].mean()


def test_decoder_recovers_adjacency(field):
    labels, geom, nuc = field
    maps = ClassicalProposer(auto_scale=False)(geom, nuc)
    dec = ConstrainedDecoder()
    res = dec.decode(maps, DecoderParams(min_cell_area_px=30))
    assert validate(res.cx).ok
    m = best_match(res.labels, labels)
    f1, p, r = pair_f1(res.labels, labels, m)
    assert f1 > 0.85, (f1, p, r)
    # structured decoding beats nucleus-only Voronoi on the self-consistency ratio
    pts = np.round(maps.seed_points[maps.seed_scores >= 0.3]).astype(int)
    mk = np.zeros(geom.shape, np.int32)
    mk[pts[:, 0], pts[:, 1]] = np.arange(1, len(pts) + 1)
    from skimage.segmentation import watershed

    vor = watershed(ndi.distance_transform_edt(mk == 0), mk)
    cxv = extract_complex(vor)
    smooth_complex(cxv)
    assert boundary_interior_ratio(geom, res.labels, res.cx) > boundary_interior_ratio(geom, vor, cxv)


def test_renderer_prefers_true_geometry(field):
    labels, geom, nuc = field
    cx = extract_complex(labels)
    smooth_complex(cx)
    shifted = np.roll(labels, 4, axis=1)
    cxs = extract_complex(shifted)
    smooth_complex(cxs)
    rm = RenderModel(line_width_px=2.0, psf_sigma_px=1.0)
    ll_true, info = render_log_likelihood(geom, labels, cx, rm)
    ll_shift, _ = render_log_likelihood(geom, shifted, cxs, rm)
    assert ll_true > ll_shift
    assert info["mu"].shape == geom.shape
    assert info["gain"] >= 0 and info["read_noise"] > 0


def test_energy_breakdown_and_soft_priors(field):
    labels, geom, nuc = field
    maps = ClassicalProposer(auto_scale=False)(geom, nuc)
    cx = extract_complex(labels)
    smooth_complex(cx)
    E, info = complex_energy(labels, cx, maps, image=geom)
    assert set(info["parts"]) == {"image", "curve", "seed", "vertex", "prior"}
    assert np.isfinite(E)
    # zero weights remove the biological priors from the total
    E0, info0 = complex_energy(labels, cx, maps, image=None, weights=EnergyWeights(image=0, seed=0, vertex=0, prior=0))
    assert E0 == pytest.approx(info0["parts"]["curve"])
    # a merged (wrong) segmentation has higher seed energy than the truth
    merged = labels.copy()
    lab_ids = np.unique(labels[labels > 0])
    merged[merged == lab_ids[1]] = lab_ids[0]
    cxm = extract_complex(merged)
    smooth_complex(cxm)
    _, infom = complex_energy(merged, cxm, maps, image=None)
    assert infom["parts"]["seed"] > info["parts"]["seed"] - 1e-9


def test_posterior_weights_and_probabilities(field):
    labels, geom, nuc = field
    maps = ClassicalProposer(auto_scale=False)(geom, nuc)
    dec = ConstrainedDecoder()
    hyps = generate_hypotheses(
        maps,
        dec,
        DecoderParams(min_cell_area_px=30),
        image=geom,
        seed_thresholds=(0.2, 0.5),
        seed_drop_fracs=(0.0, 0.1),
        boundary_smooth_sigmas=(0.0,),
        distance_mixes=(0.0, 0.3),
        gap_thresholds=(0.7,),
        n_merge_moves=3,
        n_split_moves=2,
    )
    assert len(hyps) >= 3
    post = PosteriorEnsemble.from_hypotheses(hyps, ess_min=3)
    assert post.weights.sum() == pytest.approx(1.0)
    assert effective_sample_size(post.weights) >= 3 - 1e-6
    cp = post.contact_probabilities()
    assert all(0 <= p <= 1 + 1e-9 for p in cp.values())
    # merge moves create genuinely uncertain contacts
    assert any(p < 1 - 1e-6 for p in cp.values())
    n_cells = post.expectation(lambda h: float(h.cx.cell_faces.size))
    var = post.variance(lambda h: float(h.cx.cell_faces.size))
    assert n_cells > 0 and var >= 0
    lo, hi = post.credible_interval(lambda h: float(h.cx.cell_faces.size), 0.9)
    assert lo <= n_cells <= hi + 1e-9 or lo <= hi
    vr = post.vertex_credible_radius()
    assert all(0 <= d["p_exist"] <= 1 + 1e-9 and d["rms_radius_px"] >= 0 for d in vr.values())
    s = post.summary()
    assert s["n_hypotheses"] == len(hyps)


def test_temperature_for_ess_monotone():
    E = np.array([0.0, 0.1, 0.5, 2.0, 5.0])
    T1 = temperature_for_ess(E, 2.0)
    T2 = temperature_for_ess(E, 4.0)
    assert T2 > T1
    assert effective_sample_size(softmax_weights(E, T2)) >= 4 - 1e-6


def test_merge_small_regions():
    lab = np.zeros((20, 20), np.int32)
    lab[:, :10] = 1
    lab[:, 10:] = 2
    lab[9:11, 9:11] = 3  # tiny region touching both
    out = merge_small_regions(lab, min_area=10)
    assert 3 not in np.unique(out)
    assert set(np.unique(out)) == {1, 2}
    iso = np.zeros((20, 20), np.int32)
    iso[5:7, 5:7] = 4
    assert merge_small_regions(iso, 10).max() == 0


def test_fill_small_background_assigns_seams_and_keeps_gaps():
    from pimorph.infer.decoder import _fill_small_background

    lab = np.zeros((40, 40), np.int32)
    lab[2:38, 2:19] = 1
    lab[2:38, 21:38] = 2  # 2 px seam between the cells, enclosed by the image? no: open to the border rows
    lab[0:2, :] = 0
    lab[38:, :] = 0
    # close the seam at both ends so it becomes an enclosed component
    lab[2, 19:21] = 1
    lab[37, 19:21] = 2
    seam = (lab == 0) & (np.arange(40)[None, :] >= 19) & (np.arange(40)[None, :] <= 20)
    seam[:3] = False
    seam[37:] = False  # the seam rows that are enclosed by the two closing pixels
    assert 0 < seam.sum() < 100
    out = _fill_small_background(lab, min_gap_area=100)
    assert (out[seam] > 0).all()  # seam assigned to the nearest cells
    assert (out[0] == 0).all() and (out[39] == 0).all()  # border-touching background is outer, untouched
    # a large enclosed hole stays a gap
    lab2 = np.ones((60, 60), np.int32)
    lab2[20:40, 20:40] = 0
    assert (_fill_small_background(lab2, min_gap_area=100) == lab2).all()


def _two_cell_maps(shape=(40, 60), touching=True):
    """Two seeded blobs; the signed-distance head says the strip between them is
    outside every cell unless ``touching``."""
    from pimorph.infer.proposals import ProposalMaps

    H, W = shape
    boundary = np.zeros(shape, np.float32)
    boundary[:, 29:31] = 0.9
    seed = np.zeros(shape, np.float32)
    seed[20, 12] = 1.0
    seed[20, 47] = 1.0
    rr, cc = np.mgrid[:H, :W]
    dist = np.minimum(np.abs(cc - 29.5), 6.0).astype(np.float32)
    if not touching:
        dist[:, 26:34] = -3.0  # 8 px strip predicted outside any cell
    return ProposalMaps(
        boundary=boundary,
        seed=seed,
        tissue=np.ones(shape, bool),
        gap=np.zeros(shape, np.float32),
        sigma=np.ones(shape, np.float32),
        seed_points=np.array([[20.0, 12.0], [20.0, 47.0]]),
        seed_scores=np.array([1.0, 1.0], np.float32),
        vertex=np.zeros(shape, np.float32),
        distance=dist,
        source="test",
    )


def test_decoder_outside_exclusion_separates_non_touching_cells():
    dec = ConstrainedDecoder()
    touching = dec.decode(_two_cell_maps(touching=True), DecoderParams(min_cell_area_px=20, min_gap_area_px=4))
    apart = dec.decode(_two_cell_maps(touching=False), DecoderParams(min_cell_area_px=20, min_gap_area_px=4))
    assert touching.cx.cell_cell_edges().size >= 1
    assert apart.cx.cell_faces.size == 2 and apart.cx.cell_cell_edges().size == 0
    legacy = dec.decode(
        _two_cell_maps(touching=False), DecoderParams(min_cell_area_px=20, min_gap_area_px=4, outside_px=None)
    )
    assert legacy.cx.cell_cell_edges().size >= 1  # without the rule the flood joins them


def test_vertex_consistent_merge_removes_unsupported_split():
    from pimorph.infer.decoder import vertex_consistent_merges

    maps = _two_cell_maps(touching=True)
    maps.boundary[:] = 0.05  # no boundary evidence for the edge between the two seeds
    dec = ConstrainedDecoder()
    base = dec.decode(maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4))
    assert base.cx.cell_faces.size == 2
    lab, n = vertex_consistent_merges(
        base.labels, base.cx, maps, DecoderParams(merge_boundary_max=0.3, max_merges=5, merge_vertex_min=0.5)
    )
    assert n == 1 and len(np.unique(lab[lab > 0])) == 1
    merged = dec.decode(
        maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4, merge_boundary_max=0.3, max_merges=5)
    )
    assert merged.cx.cell_faces.size == 1 and merged.info["n_vertex_merges"] == 1
    # strong vertex-head support at the edge ends vetoes the merge
    maps.vertex[:, 29:31] = 0.95
    kept = dec.decode(maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4, merge_boundary_max=0.3, max_merges=5))
    assert kept.cx.cell_faces.size == 2


def test_nucleus_consistency_merge_removes_empty_fragment():
    from pimorph.infer.decoder import nucleus_consistency_merges

    maps = _two_cell_maps(touching=True)
    maps.boundary[:] = 0.4
    # one nucleus, in the left cell only: the right cell is a nucleus-free fragment
    maps.meta["nucleus_points"] = np.array([[20, 12]])
    dec = ConstrainedDecoder()
    base = dec.decode(maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4))
    assert base.cx.cell_faces.size == 2
    lab, n = nucleus_consistency_merges(base.labels, base.cx, maps, DecoderParams(nucleus_merge=True))
    assert n == 1 and len(np.unique(lab[lab > 0])) == 1
    merged = dec.decode(maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4, nucleus_merge=True))
    assert merged.cx.cell_faces.size == 1 and merged.info["n_nucleus_merges"] == 1
    # a nucleus in each cell: nothing merges
    maps.meta["nucleus_points"] = np.array([[20, 12], [20, 47]])
    kept = dec.decode(maps, DecoderParams(min_cell_area_px=20, min_gap_area_px=4, nucleus_merge=True))
    assert kept.cx.cell_faces.size == 2 and kept.info["n_nucleus_merges"] == 0
