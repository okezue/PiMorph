"""Union-over-z rasterization of VIA polygon annotations (NIH-NEI RPE loader)."""

import numpy as np

from pimorph.bench.rpe import group_via_polygons, rasterize_union_over_z


def _region(cell, frame, xs, ys):
    return {
        "shape_attributes": {"name": "polygon", "all_points_x": xs, "all_points_y": ys},
        "region_attributes": {"cell": cell, "frame": frame},
    }


def test_union_over_z_touching_cells_and_roi():
    # three cells sharing borders at x=30 and y=30 on a 64x64 tile; frame 0 has all
    # of cell 1 and part of cell 2, frame 1 redraws cell 2 fully and adds cell 3.
    via = {
        "S-000-000.tif100": {
            "filename": "S-000-000.tif",
            "size": 100,
            "file_attributes": {},
            "regions": [
                _region(1, 0, [5, 30, 30, 5], [5, 5, 30, 30]),
                _region(2, 0, [30, 55, 55, 30], [5, 5, 18, 18]),
            ],
        },
        "S-001-000.tif100": {
            "filename": "S-001-000.tif",
            "size": 100,
            "file_attributes": {},
            "regions": [
                _region(2, 1, [30, 55, 55, 30], [5, 5, 30, 30]),
                _region(3, 1, [5, 55, 55, 5], [30, 30, 55, 55]),
                # a 4 px speck must be dropped
                _region(9, 1, [60, 62, 62, 60], [60, 60, 62, 62]),
            ],
        },
    }
    grouped = group_via_polygons(via)
    assert list(grouped) == ["S"] and list(grouped["S"]) == ["000"]
    polys = grouped["S"]["000"]
    assert len(polys) == 5

    lab, roi = rasterize_union_over_z(polys, (64, 64))
    ids = [i for i in np.unique(lab) if i > 0]
    assert len(ids) == 3
    areas = np.bincount(lab.ravel())[1:]
    assert areas.min() >= 60

    # cell 2 is the union of its two frames: it reaches y=30, not just y=18
    assert lab[25, 40] > 0 and lab[25, 40] == lab[10, 40]
    # neighbours touch along the shared borders (4-neighbour label-to-label transitions)
    h = (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0)
    v = (lab[1:] != lab[:-1]) & (lab[1:] > 0) & (lab[:-1] > 0)
    assert h.sum() > 10 and v.sum() > 10
    assert lab[10, 29] != lab[10, 31] and lab[10, 29] > 0 and lab[10, 31] > 0

    # ROI is the painted union dilated by 3 px; the far corner is outside it
    assert roi.dtype == bool
    assert roi[5:56, 5:56].all()
    assert roi[2, 20] and not roi[1, 20]
    assert roi[20, 58] and not roi[20, 59]
    # the dropped speck still counts as annotated area
    assert roi[61, 61]
    assert not roi[45, 63]
