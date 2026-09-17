import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def voronoi_labels(n_seeds: int, shape=(128, 128), seed: int = 0, gaps: int = 0, gap_size: int = 4) -> np.ndarray:
    """Voronoi-like tessellation via watershed on the seed distance transform."""
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.integers(0, shape[0], n_seeds), rng.integers(0, shape[1], n_seeds)], axis=1)
    pts = np.unique(pts, axis=0)
    markers = np.zeros(shape, dtype=np.int32)
    markers[pts[:, 0], pts[:, 1]] = np.arange(1, pts.shape[0] + 1)
    dist = ndi.distance_transform_edt(markers == 0)
    lab = watershed(dist, markers).astype(np.int32)
    for _ in range(gaps):
        r = rng.integers(gap_size, shape[0] - gap_size)
        c = rng.integers(gap_size, shape[1] - gap_size)
        lab[r - gap_size : r + gap_size, c - gap_size : c + gap_size] = 0
    return lab


@pytest.fixture
def quadrants():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[:10, :10] = 1
    lab[:10, 10:] = 2
    lab[10:, :10] = 3
    lab[10:, 10:] = 4
    return lab


@pytest.fixture
def island():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[2:18, 2:18] = 1
    return lab


@pytest.fixture
def hole_and_island():
    lab = np.zeros((30, 30), dtype=np.int32)
    lab[2:28, 2:28] = 1
    lab[10:20, 10:20] = 0
    lab[13:17, 13:17] = 2
    return lab


@pytest.fixture
def pinch():
    lab = np.zeros((10, 10), dtype=np.int32)
    lab[2:5, 2:5] = 1
    lab[5:8, 5:8] = 1
    lab[2:5, 5:8] = 2
    lab[5:8, 2:5] = 3
    return lab


@pytest.fixture
def double_contact():
    lab = np.zeros((20, 30), dtype=np.int32)
    lab[:, :10] = 1
    lab[:, 10:] = 2
    lab[8:12, 10:20] = 0
    return lab


@pytest.fixture
def honeycomb():
    """Hexagonal tiling as label image: every interior vertex is trivalent."""
    H, W = 120, 140
    a = 10.0  # hex circumradius
    dy, dx = 1.5 * a, np.sqrt(3) * a
    centers = []
    for i in range(-1, int(H / dy) + 2):
        for j in range(-1, int(W / dx) + 2):
            centers.append((i * dy, j * dx + (dx / 2 if i % 2 else 0)))
    centers = np.array(centers)
    rr, cc = np.mgrid[:H, :W]
    d = (rr[..., None] - centers[:, 0]) ** 2 + (cc[..., None] - centers[:, 1]) ** 2
    return (np.argmin(d, axis=-1) + 1).astype(np.int32)


@pytest.fixture
def voronoi_small():
    return voronoi_labels(40, shape=(128, 128), seed=1, gaps=3)
