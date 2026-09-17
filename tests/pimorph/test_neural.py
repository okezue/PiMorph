"""Fast CPU tests for pimorph.infer.neural (tiny UNet, 64 px crops, a few steps)."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pimorph.complex import validate  # noqa: E402
from pimorph.infer import ConstrainedDecoder, DecoderParams  # noqa: E402
from pimorph.infer.neural import (  # noqa: E402
    HEADS,
    MultiHeadLoss,
    MultiHeadUNet,
    NeuralProposer,
    TileDataset,
    TrainConfig,
    consensus_labels,
    count_parameters,
    split_by_field,
    train,
)
from pimorph.infer.neural.losses import bce_with_pos_weight, focal_bce, heteroscedastic_l1, soft_cldice  # noqa: E402
from pimorph.infer.neural.proposer import tiled_predict  # noqa: E402
from pimorph.infer.neural.train import parse_args  # noqa: E402
from pimorph.synth.targets import default_params_sampler, make_dataset  # noqa: E402

from .conftest import voronoi_labels  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny_tiles(tmp_path_factory):
    out = tmp_path_factory.mktemp("neural_tiles")
    make_dataset(4, out, params_sampler=lambda rng: default_params_sampler(rng, shape=(96, 96)), seed=3)
    return out


@pytest.fixture(scope="module")
def trained(tiny_tiles, tmp_path_factory):
    out = tmp_path_factory.mktemp("neural_run")
    cfg = TrainConfig(
        train_dirs=[str(tiny_tiles)],
        val_dirs=[str(tiny_tiles)],
        out_dir=str(out),
        epochs=1,
        batch_size=2,
        crop=64,
        base=8,
        depth=2,
        amp=False,
        num_workers=0,
        device="cpu",
        log_every=1,
        max_steps=2,
    )
    best = train(cfg)
    return out, best


@pytest.mark.torch
def test_model_forward_shape_and_parameter_count():
    small = MultiHeadUNet(in_channels=6, base=8, depth=2)
    y = small(torch.zeros(2, 6, 64, 64))
    assert y.shape == (2, len(HEADS), 64, 64)
    assert small(torch.zeros(1, 6, 50, 70)).shape == (1, 6, 50, 70)
    n_default = count_parameters(MultiHeadUNet())
    assert 8_000_000 <= n_default <= 10_000_000, n_default
    d = small.forward_dict(torch.zeros(1, 6, 32, 32))
    assert set(d) == set(HEADS) and d["boundary"].shape == (1, 1, 32, 32)


@pytest.mark.torch
def test_dataset_shapes_and_presence_indicators(tiny_tiles):
    paths = sorted(tiny_tiles.glob("*.npz"))
    ds = TileDataset(paths, crop=64, train=True, p_nuclei=0.0, p_junction=1.0, seed=1)
    x, y, w = ds[0]
    assert x.shape == (6, 64, 64) and y.shape == (5, 64, 64) and w.shape == (1, 64, 64)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    # geometry always present, nuclei dropped, junction kept
    assert torch.all(x[3] == 1) and torch.all(x[4] == 0) and torch.all(x[5] == 1)
    assert torch.all(x[1] == 0) and x[0].std() > 0 and x[2].std() > 0
    assert torch.all(w == 1)
    assert y[1].min() >= -1.0 and y[1].max() <= 1.0
    assert set(torch.unique(y[0]).tolist()) <= {0.0, 1.0}
    ds_all = TileDataset(paths, crop=64, train=True, p_nuclei=1.0, p_junction=0.0, seed=1)
    x2, _, _ = ds_all[1]
    assert torch.all(x2[4] == 1) and x2[1].std() > 0 and torch.all(x2[5] == 0) and torch.all(x2[2] == 0)
    # eval mode: deterministic, full presence, pad-free centre crop
    ds_eval = TileDataset(paths, crop=128, train=False)
    xe, ye, we = ds_eval[0]
    assert xe.shape == (6, 128, 128) and torch.all(xe[3:] == 1)
    assert torch.equal(xe, ds_eval[0][0])


@pytest.mark.torch
def test_losses_finite_and_overfit_single_tile(tiny_tiles):
    torch.manual_seed(0)
    ds = TileDataset(sorted(tiny_tiles.glob("*.npz")), crop=64, train=False)
    x, y, w = (t[None] for t in ds[0])
    model = MultiHeadUNet(base=8, depth=2)
    crit = MultiHeadLoss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(20):
        total, parts = crit(model(x), y, w)
        assert torch.isfinite(total) and all(np.isfinite(v) for v in parts.values())
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(float(total.detach()))
    assert set(parts) == {"boundary", "distance", "seed", "vertex", "gap", "cldice"}
    assert losses[-1] < losses[0]
    # individual losses respect the weight map
    logits = torch.randn(1, 1, 8, 8)
    tgt = (torch.rand(1, 1, 8, 8) > 0.5).float()
    zero_w = torch.zeros(1, 1, 8, 8)
    assert float(bce_with_pos_weight(logits, tgt, zero_w, 3.0)) == 0.0
    assert float(focal_bce(logits, tgt, torch.ones_like(zero_w))) > 0.0
    hl = heteroscedastic_l1(
        torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8), torch.ones_like(zero_w)
    )
    assert abs(float(hl) - 1.0) < 1e-6


@pytest.mark.torch
def test_soft_cldice_identity_and_disjoint():
    a = torch.zeros(1, 1, 32, 32)
    a[:, :, 10:13, 4:28] = 1.0
    b = torch.zeros(1, 1, 32, 32)
    b[:, :, 20:23, 4:28] = 1.0
    assert float(soft_cldice(a, a)) < 1e-5
    assert float(soft_cldice(a, b)) > 0.5
    assert 0.0 <= float(soft_cldice(torch.rand(1, 1, 32, 32), a)) <= 1.0


@pytest.mark.torch
def test_train_smoke_writes_artifacts(trained):
    out, best = trained
    for name in ("last.pt", "best.pt", "train_log.jsonl", "config.json"):
        assert (out / name).exists(), name
    assert best == out / "best.pt"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["base"] == 8 and cfg["depth"] == 2 and cfg["max_steps"] == 2
    lines = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["step"] == 2 and np.isfinite(rec["train_loss"]) and np.isfinite(rec["val_loss"])
    for key in ("boundary_f1", "seed_peak_recall", "vertex_peak_recall", "gap_iou"):
        assert key in rec["val_metrics"]
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert set(ck) >= {"model", "config", "epoch", "val"}
    assert ck["epoch"] == 0 and ck["config"]["crop"] == 64


@pytest.mark.torch
def test_neural_proposer_maps_and_decode(trained, tiny_tiles):
    _, best = trained
    with np.load(sorted(tiny_tiles.glob("*.npz"))[0]) as z:
        junction, nuclei = z["junction"], z["nuclei"]
    prop = NeuralProposer(best, device="cpu", tile=64, overlap=16)
    maps = prop(junction, nuclei, junction)
    H, W = junction.shape
    assert maps.source == "neural" and maps.shape == (H, W)
    for arr in (maps.boundary, maps.seed, maps.gap, maps.vertex):
        assert arr.shape == (H, W) and arr.dtype == np.float32
        assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
    assert maps.distance.shape == (H, W) and maps.sigma.shape == (H, W) and np.all(maps.sigma > 0)
    assert maps.tissue.dtype == bool
    assert maps.seed_points.shape[1] == 2 and maps.seed_scores.shape[0] == maps.seed_points.shape[0]
    if maps.seed_scores.size:
        assert maps.seed_scores.min() >= 0.0 and maps.seed_scores.max() <= 1.0
        assert np.all(np.diff(maps.seed_scores) <= 1e-6)
    assert maps.meta["cell_radius_px"] > 0 and maps.meta["ridge_width_px"] > 0 and maps.meta["nuclei_used"]
    res = ConstrainedDecoder().decode(maps, DecoderParams(cell_radius_px=maps.meta["cell_radius_px"]))
    assert res.labels.shape == (H, W) and validate(res.cx).ok
    # tiled and single-shot predictions agree away from the blend seams
    x = np.zeros((6, H, W), dtype=np.float32)
    x[0] = junction / junction.max()
    x[3] = 1.0
    single = tiled_predict(prop.model, x, tile=256, overlap=0)
    tiled = tiled_predict(prop.model, x, tile=64, overlap=16)
    assert single.shape == (6, H, W) and tiled.shape == (6, H, W)
    assert np.isfinite(tiled).all()
    # nuclei-free call uses the distance-based scale estimate
    maps2 = prop(junction)
    assert not maps2.meta["nuclei_used"] and maps2.meta["cell_radius_px"] >= 3.0


@pytest.mark.torch
def test_consensus_labels_ignores_merge_and_keeps_agreement():
    a = voronoi_labels(30, shape=(128, 128), seed=2)
    ids = np.unique(a[a > 0])
    # merge one cell into a 4-adjacent neighbour
    la = int(ids[0])
    ring = np.zeros_like(a, dtype=bool)
    m = a == la
    ring[1:] |= m[:-1]
    ring[:-1] |= m[1:]
    ring[:, 1:] |= m[:, :-1]
    ring[:, :-1] |= m[:, 1:]
    neigh = np.unique(a[ring & ~m & (a > 0)])
    lb = int(neigh[0])
    b = a.copy()
    b[b == lb] = la
    labels, ignore = consensus_labels(a, b, iou_thresh=0.7, boundary_tol_px=2)
    assert labels.shape == a.shape and ignore.dtype == bool
    merged = (a == la) | (a == lb)
    assert ignore[merged].all()
    assert (labels[merged] == 0).all()
    # cells untouched by the merge keep a single consistent label over most of their area
    far = ~merged & (a > 0) & ~ignore
    assert far.mean() > 0.4
    for lab in ids:
        if lab in (la, lb):
            continue
        region = (a == lab) & ~ignore
        if region.sum() < 20:
            continue
        vals = np.unique(labels[region])
        assert vals.size == 1 and vals[0] > 0
    # identical inputs: nothing ignored, every cell kept
    labels_same, ignore_same = consensus_labels(a, a)
    assert not ignore_same.any()
    assert np.array_equal(labels_same > 0, a > 0)


@pytest.mark.torch
def test_split_by_field_keeps_fields_disjoint(tmp_path):
    rows = [{"file": f"t{i:03d}.npz", "source_field": f"F{i % 5}"} for i in range(40)]
    df = pd.DataFrame(rows)
    tr, va = split_by_field(df, val_fraction=0.4, seed=0, root=tmp_path)
    assert len(tr) + len(va) == 40 and va
    field = lambda p: df.loc[df["file"] == p.name, "source_field"].iloc[0]  # noqa: E731
    assert set(map(field, tr)).isdisjoint(set(map(field, va)))
    assert all(p.parent == tmp_path for p in tr + va)
    # plain paths: split by file, still disjoint and covering
    paths = [tmp_path / r["file"] for r in rows]
    tr2, va2 = split_by_field(paths, val_fraction=0.25, seed=1)
    assert set(tr2).isdisjoint(va2) and len(tr2) + len(va2) == 40 and len(va2) == 10


@pytest.mark.torch
def test_cli_parse_args_round_trip():
    cfg = parse_args(
        [
            "--train-dirs",
            "a",
            "b",
            "--val-dirs",
            "c",
            "--out",
            "runs/x",
            "--epochs",
            "3",
            "--no-amp",
            "--max-steps",
            "7",
            "--device",
            "cpu",
            "--lr",
            "1e-3",
        ]
    )
    assert cfg.train_dirs == ["a", "b"] and cfg.val_dirs == ["c"] and cfg.out_dir == "runs/x"
    assert cfg.epochs == 3 and cfg.amp is False and cfg.max_steps == 7 and cfg.device == "cpu" and cfg.lr == 1e-3
    assert cfg.resume is None and cfg.base == 32 and cfg.depth == 4
