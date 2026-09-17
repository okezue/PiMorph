"""Regression checks on real fields (skipped when the data is not present).

These reproduce the blueprint's self-consistency audit: junction-guided structured
decoding should put boundaries on VE-cadherin signal far more often than a
nucleus-only Voronoi partition of the same seeds.
"""

from pathlib import Path

import numpy as np
import pytest

MANIFEST = Path("data/ve_strat/manifest_paired.csv")

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def ve_strat_crop():
    if not MANIFEST.exists():
        pytest.skip("VE-strat paired manifest not present")
    from pimorph.io.manifest import parse_manifest

    sp = parse_manifest(MANIFEST)[0]
    if not all(p.exists() for p in sp.files.values()):
        pytest.skip("VE-strat files not present")
    im = sp.load()
    geom = sp.role_channel(im, "geometry")[512:1024, 512:1024]
    nuc = sp.role_channel(im, "nuclei")[512:1024, 512:1024]
    return geom, nuc, sp


def test_channel_roles_and_calibration_flags(ve_strat_crop):
    _, _, sp = ve_strat_crop
    # w2 is VE-cadherin and doubles as geometry; w4 is nuclei; no membrane channel
    assert sp.roles["junction"] == "geometry"
    assert sp.geometry_source == "junction_channel"
    assert "_w2" in str(sp.files["geometry"]) and "_w4" in str(sp.files["nuclei"])


def test_structured_beats_voronoi_on_ve_strat(ve_strat_crop):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    from pimorph.complex import extract_complex, validate
    from pimorph.complex.geometry import smooth_complex
    from pimorph.infer import ClassicalProposer, ConstrainedDecoder, DecoderParams
    from pimorph.infer.renderer import boundary_interior_ratio

    geom, nuc, _ = ve_strat_crop
    maps = ClassicalProposer()(geom, nuc)
    assert maps.meta["nucleus_radius_px"] > 10  # 2048 px fields, large nuclei
    assert maps.tissue.mean() > 0.9  # confluent monolayer
    res = ConstrainedDecoder().decode(maps, DecoderParams(cell_radius_px=maps.meta["cell_radius_px"]))
    rep = validate(res.cx)
    assert rep.ok and rep.n_cells >= 25
    assert set(rep.degree_histogram) <= {2, 3, 4}

    pts = np.round(maps.seed_points[maps.seed_scores >= 0.3]).astype(int)
    m = np.zeros(geom.shape, np.int32)
    m[pts[:, 0], pts[:, 1]] = np.arange(1, len(pts) + 1)
    vor = watershed(ndi.distance_transform_edt(m == 0), m, mask=maps.tissue)
    cxv = extract_complex(vor)
    smooth_complex(cxv)
    r_struct = boundary_interior_ratio(geom, res.labels, res.cx)
    r_vor = boundary_interior_ratio(geom, vor, cxv)
    assert r_struct > 1.4 * r_vor, (r_struct, r_vor)
