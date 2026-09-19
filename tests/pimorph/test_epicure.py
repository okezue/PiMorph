"""EpiCure loader on a synthetic 3-frame label movie with 1 px background seams."""

import numpy as np
import tifffile

from pimorph.complex import extract_complex
from pimorph.dynamics import admissibility_check, detect_events, snapshot
from pimorph.dynamics.epicure import epicure_tracks, load_epicure_movie, seam_pixels


def _seamed_frame(shift: int) -> np.ndarray:
    """Four cells in a 2x2 block with 1 px background seams; the vertical seam of the
    top row is shifted by ``shift`` px so ids stay constant while geometry moves."""
    lab = np.zeros((40, 40), dtype=np.float32)
    lab[4:19, 4 : 19 + shift] = 1
    lab[4:19, 20 + shift : 36] = 2
    lab[20:36, 4:19] = 3
    lab[20:36, 20:36] = 4
    return lab


def test_load_epicure_movie_fills_seams_and_keeps_ids(tmp_path):
    movie = tmp_path / "movieX"
    (movie / "epics_corrected").mkdir(parents=True)
    stack = np.stack([_seamed_frame(s) for s in (0, 1, 2)])
    tifffile.imwrite(movie / "epics_corrected" / "toy_labels.tif", stack, imagej=True, metadata={"axes": "TYX"})
    raw = (np.random.default_rng(0).random((3, 2, 40, 40)) * 1000).astype(np.uint16)
    tifffile.imwrite(movie / "toy.tif", raw, imagej=True, metadata={"axes": "TCYX"})

    assert seam_pixels(stack[0].astype(np.int32)).sum() > 0
    lab, im, meta = load_epicure_movie(movie)
    assert lab.shape == (3, 40, 40) and lab.dtype == np.int32
    assert im is not None and im.shape == (3, 2, 40, 40)
    assert meta["n_frames"] == 3 and meta["n_ids"] == 4 and meta["cells_per_frame"] == [4, 4, 4]
    assert meta["n_seam_pixels_filled"] > 0
    for fr in lab:
        assert not seam_pixels(fr).any()
        assert set(np.unique(fr[fr > 0]).tolist()) == {1, 2, 3, 4}
        # neighbours now share a crack: direct label-to-label transitions exist
        assert (((fr[:, 1:] != fr[:, :-1]) & (fr[:, 1:] > 0) & (fr[:, :-1] > 0)).sum()) > 0
    # open background around the block is untouched
    assert lab[0, 0, 0] == 0 and lab[0, 2, 20] == 0

    tracks = epicure_tracks(lab)
    assert tracks.n_tracks == 4 and tracks.n_frames == 3
    assert tracks.births == {1: 0, 2: 0, 3: 0, 4: 0} and tracks.deaths == {1: 2, 2: 2, 3: 2, 4: 2}

    # the filled frames form valid complexes; shifting the seam resolves the fourfold
    # vertex into a 1-4 contact (contact_birth, frame 0) and then only lengthens it
    snaps = [snapshot(extract_complex(fr)) for fr in tracks.frame_labels]
    kinds = []
    for t in range(2):
        ev = detect_events(snaps[t], snaps[t + 1], frame=t)
        chk = admissibility_check(snaps[t], snaps[t + 1], ev)
        assert chk["fully_explained"]
        kinds.append([(e.kind, tuple(e.participants["cells"])) for e in ev])
    assert kinds == [[("contact_birth", (1, 4))], []]
