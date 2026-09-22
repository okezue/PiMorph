"""Cellpose-as-proposer: maps from masks, decode round trip, threshold-grid hypotheses.
None of these tests needs cellpose; the network pass is faked."""

from __future__ import annotations

import numpy as np
import pytest

from pimorph.complex.incidence import validate
from pimorph.infer.cellpose_proposer import (
    CellposeRaw,
    maps_from_masks,
    signed_distance,
    tricellular_points,
)
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams


def _voronoi_masks(shape=(96, 128), n=14, seed=3, seam=True):
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(4, shape[0] - 4, n), rng.uniform(4, shape[1] - 4, n)], 1)
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    d = (rr[..., None] - pts[:, 0]) ** 2 + (cc[..., None] - pts[:, 1]) ** 2
    lab = (np.argmin(d, axis=-1) + 1).astype(np.int32)
    lab[:3] = 0
    lab[-3:] = 0  # open background top and bottom, as Cellpose masks have
    if seam:
        # 1 px background seams between cells, the usual Cellpose mask format
        from skimage.segmentation import find_boundaries

        lab[find_boundaries(lab, mode="inner") & (lab > 0) & np.roll(lab > 0, 1, 0)] = 0
    return lab


def test_signed_distance_sign_and_boundary_value():
    lab = np.zeros((20, 20), np.int32)
    lab[4:16, 4:16] = 1
    d = signed_distance(lab)
    assert d[10, 10] > 3
    assert d[4, 10] == pytest.approx(0.5)  # inner boundary ring
    assert d[0, 0] < -4  # far outside
    assert d[3, 10] == pytest.approx(-1.0)


def test_tricellular_points_count_on_three_cells():
    lab = np.zeros((10, 10), np.int32)
    lab[:5, :5] = 1
    lab[:5, 5:] = 2
    lab[5:, :] = 3
    pts = tricellular_points(lab)
    assert pts.shape == (1, 2)
    assert pts[0] == pytest.approx((4.5, 4.5))


def test_maps_from_masks_reproduce_masks_through_the_decoder():
    masks = _voronoi_masks()
    maps = maps_from_masks(masks, cellprob=None, source="test")
    assert maps.seed_points.shape[0] == len(np.unique(masks[masks > 0]))
    assert maps.distance is not None and maps.vertex is not None
    assert 0.0 <= maps.boundary.min() and maps.boundary.max() == pytest.approx(1.0)
    res = ConstrainedDecoder().decode(maps, DecoderParams(cell_radius_px=maps.meta["cell_radius_px"]))
    assert validate(res.cx).ok
    # every decoded cell overlaps one mask almost entirely (seams filled, no cell lost)
    filled_ids = np.unique(masks[masks > 0])
    assert res.cx.cell_faces.size == len(filled_ids)
    agree = 0
    for lid in np.unique(res.labels[res.labels > 0]):
        region = (res.labels == lid) & (masks > 0)  # seam pixels were background in the masks
        owner = np.bincount(masks[region]).argmax()
        agree += owner > 0 and (masks[region] == owner).mean() > 0.95
    assert agree == len(filled_ids)
    # most tricellular corners of the seam-free tiling are recovered by the complex
    n_tri = sum(1 for v in range(res.cx.n_vertices) if len(res.cx.vertex_cell_set(v)) >= 3)
    assert n_tri >= 0.8 * tricellular_points(_voronoi_masks(seam=False)).shape[0]


def test_maps_from_masks_uses_cellprob_for_gap_and_scores():
    masks = _voronoi_masks()
    cellprob = np.where(masks > 0, 3.0, -3.0).astype(np.float32)
    maps = maps_from_masks(masks, cellprob=cellprob)
    assert maps.gap[masks > 0].max() < 0.1
    assert maps.gap[masks == 0].min() > 0.9 or (masks == 0).sum() == 0
    assert np.all(maps.seed_scores > 0.9)


def test_cellpose_hypotheses_pool_settings_with_a_fake_network():
    from pimorph.infer import cellpose_proposer as cp

    masks = _voronoi_masks()
    coarse = masks.copy()
    ids = np.unique(masks[masks > 0])
    coarse[coarse == ids[1]] = ids[0]  # a "higher flow threshold" merges two cells

    class FakeProposer:
        flow_threshold = 0.4
        cellprob_threshold = 0.0
        seam_px = 12
        pretrained_model = "fake"
        nucleus_points = False
        device = None

        def raw(self, geometry, nuclei=None):
            return CellposeRaw(masks, np.zeros((2,) + masks.shape, np.float32), np.where(masks > 0, 2.0, -2.0), None)

        def masks_at(self, raw, ft, ct):
            return coarse if ft > 0.4 else masks

        maps = cp.CellposeProposer.maps
        maps_grid = cp.CellposeProposer.maps_grid
        _nucleus_points = cp.CellposeProposer._nucleus_points

    prop = FakeProposer()
    img = (masks > 0).astype(np.float32)
    hyps, raw, ref = cp.cellpose_hypotheses(
        prop,
        img,
        None,
        ConstrainedDecoder(),
        DecoderParams(),
        image=img,
        flow_thresholds=(0.4, 0.6),
        cellprob_thresholds=(0.0,),
        n_merge_moves=1,
        n_split_moves=1,
        max_per_setting=2,
    )
    assert len(hyps) >= 2
    counts = {h.cx.cell_faces.size for h in hyps}
    assert len(counts) >= 2  # the coarse setting produced a different complex
    assert all(validate(h.cx).ok for h in hyps)
    assert all(np.isfinite(h.energy) for h in hyps)
