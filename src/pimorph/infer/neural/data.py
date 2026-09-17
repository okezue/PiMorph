"""Tile dataset for the multi-head model.

Tiles are ``.npz`` files written by ``pimorph.synth.targets.make_dataset`` or by
``pimorph.infer.neural.pseudolabel.make_pseudolabel_tiles``. Both carry ``junction``,
``labels`` and the targets ``boundary``, ``signed_distance``, ``seed``, ``vertex``,
``gap``, ``outer``; synthetic tiles also carry ``membrane`` and ``nuclei``, pseudo-label
tiles carry ``ignore`` and may carry ``nuclei`` as zeros with ``has_nuclei=False``.

Sample layout: ``x`` (6, crop, crop) = [geometry, nuclei, junction, present_geometry,
present_nuclei, present_junction]; ``y`` (5, crop, crop) in ``TARGET_CHANNELS`` order
with distance scaled by 1/16; ``w`` (1, crop, crop) loss weight (0 on ignored pixels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..proposals import robust_normalize
from .model import DISTANCE_SCALE

INPUT_CHANNELS: Tuple[str, ...] = ("geometry", "nuclei", "junction")
TARGET_CHANNELS: Tuple[str, ...] = ("boundary", "distance", "seed", "vertex", "gap")
PathLike = Union[str, Path]


def list_tiles(dirs: Iterable[PathLike]) -> List[Path]:
    out: List[Path] = []
    for d in dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".npz":
            out.append(d)
        else:
            # recursive so sharded sets (synth_train/part0..7) can be passed as one dir;
            # skip macOS AppleDouble sidecars (._name) that tar can carry along
            out.extend(sorted(p for p in d.rglob("*.npz") if not p.name.startswith("._")))
    return out


def _has_channel(z, key: str) -> bool:
    if key not in z.files:
        return False
    if key == "nuclei" and "has_nuclei" in z.files and not bool(z["has_nuclei"]):
        return False
    return True


def build_input(
    geometry: Optional[np.ndarray],
    nuclei: Optional[np.ndarray],
    junction: Optional[np.ndarray],
    shape: Tuple[int, int],
    normalize: bool = True,
) -> np.ndarray:
    """(6, H, W) float32 model input from optional channels; shared by training and inference."""
    x = np.zeros((6,) + tuple(shape), dtype=np.float32)
    for i, ch in enumerate((geometry, nuclei, junction)):
        if ch is None:
            continue
        x[i] = robust_normalize(ch) if normalize else np.asarray(ch, dtype=np.float32)
        x[3 + i] = 1.0
    return x


class TileDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[PathLike],
        crop: int = 256,
        train: bool = True,
        p_membrane_as_geometry: float = 0.5,
        p_nuclei: float = 0.85,
        p_junction: float = 0.9,
        seed: int = 0,
    ):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("TileDataset needs at least one tile")
        self.crop = int(crop)
        self.train = bool(train)
        self.p_membrane_as_geometry = float(p_membrane_as_geometry)
        self.p_nuclei = float(p_nuclei)
        self.p_junction = float(p_junction)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        # per-sample stream so worker processes never share augmentation draws
        return np.random.default_rng([self.seed, self._epoch, index])

    def _load(self, index: int):
        with np.load(self.paths[index]) as z:
            junction = z["junction"].astype(np.float32)
            membrane = z["membrane"].astype(np.float32) if _has_channel(z, "membrane") else None
            nuclei = z["nuclei"].astype(np.float32) if _has_channel(z, "nuclei") else None
            targets = np.stack(
                [
                    z["boundary"].astype(np.float32),
                    z["signed_distance"].astype(np.float32) / DISTANCE_SCALE,
                    z["seed"].astype(np.float32),
                    z["vertex"].astype(np.float32),
                    z["gap"].astype(np.float32),
                ]
            )
            ignore = z["ignore"].astype(bool) if "ignore" in z.files else np.zeros(junction.shape, dtype=bool)
        return junction, membrane, nuclei, targets, ignore

    def _crop_window(self, shape: Tuple[int, int], rng: np.random.Generator) -> Tuple[slice, slice, Tuple]:
        H, W = shape
        c = self.crop
        pad_h, pad_w = max(c - H, 0), max(c - W, 0)
        pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        Hp, Wp = H + pad_h, W + pad_w
        if self.train:
            r0 = int(rng.integers(0, Hp - c + 1))
            c0 = int(rng.integers(0, Wp - c + 1))
        else:
            r0, c0 = (Hp - c) // 2, (Wp - c) // 2
        return slice(r0, r0 + c), slice(c0, c0 + c), pads

    @staticmethod
    def _pad(a: np.ndarray, pads) -> np.ndarray:
        if pads == ((0, 0), (0, 0)):
            return a
        full = ((0, 0),) * (a.ndim - 2) + tuple(pads)
        mode = "reflect" if a.dtype != bool else "constant"
        return np.pad(a, full, mode=mode)

    def _augment_intensity(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gain = rng.uniform(0.7, 1.4)
        gamma = rng.uniform(0.7, 1.4)
        noise = rng.uniform(0.0, 0.05)
        out = gain * np.power(np.clip(img, 0.0, None), gamma)
        if noise > 0:
            out = out + rng.normal(0.0, noise, size=out.shape)
        return out.astype(np.float32)

    def __getitem__(self, index: int):
        rng = self._rng(index)
        junction, membrane, nuclei, targets, ignore = self._load(index)
        H, W = junction.shape

        # eval mode is deterministic and mirrors the common real case: geometry is the
        # junction channel, nuclei and junction present whenever available
        use_membrane = self.train and membrane is not None and rng.random() < self.p_membrane_as_geometry
        geometry = membrane if use_membrane else junction
        keep_nuclei = nuclei is not None and (not self.train or rng.random() < self.p_nuclei)
        keep_junction = not self.train or rng.random() < self.p_junction

        chans = [
            robust_normalize(geometry),
            robust_normalize(nuclei) if keep_nuclei else None,
            robust_normalize(junction) if keep_junction else None,
        ]
        x = build_input(chans[0], chans[1], chans[2], (H, W), normalize=False)
        w = (~ignore).astype(np.float32)[None]

        rs, cs, pads = self._crop_window((H, W), rng)
        x = self._pad(x, pads)[:, rs, cs]
        y = self._pad(targets, pads)[:, rs, cs]
        w = self._pad(w, pads)[:, rs, cs]

        if self.train:
            for i in range(3):
                if x[3 + i, 0, 0] > 0:
                    x[i] = self._augment_intensity(x[i], rng)
            k = int(rng.integers(0, 4))
            if k:
                x, y, w = (np.rot90(a, k, axes=(1, 2)) for a in (x, y, w))
            if rng.random() < 0.5:
                x, y, w = (a[:, ::-1, :] for a in (x, y, w))
            if rng.random() < 0.5:
                x, y, w = (a[:, :, ::-1] for a in (x, y, w))

        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
        )


def split_by_field(
    items: Union[pd.DataFrame, Sequence[PathLike]],
    val_fraction: float = 0.2,
    seed: int = 0,
    root: Optional[PathLike] = None,
) -> Tuple[List[Path], List[Path]]:
    """Group-wise split: tiles from one ``source_field`` never straddle the two splits.

    ``items`` is a manifest DataFrame (``file`` column, optional ``source_field`` and
    ``path`` columns) or a sequence of tile paths (each path is its own group).
    """
    if isinstance(items, pd.DataFrame):
        df = items
        if "path" in df.columns:
            paths = [Path(p) for p in df["path"]]
        else:
            base = Path(root) if root is not None else Path(".")
            paths = [base / str(f) for f in df["file"]]
        groups = [str(g) for g in df["source_field"]] if "source_field" in df.columns else [str(p) for p in paths]
    else:
        paths = [Path(p) for p in items]
        groups = [str(p) for p in paths]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(round(val_fraction * len(uniq)))
    if len(uniq) > 1 and 0 < val_fraction:
        n_val = min(max(n_val, 1), len(uniq) - 1)
    val_groups = set(uniq[:n_val])
    train = [p for p, g in zip(paths, groups) if g not in val_groups]
    val = [p for p, g in zip(paths, groups) if g in val_groups]
    return train, val


def make_loader(
    paths: Sequence[PathLike],
    batch_size: int = 8,
    crop: int = 256,
    train: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: Optional[bool] = None,
    distributed: bool = False,
    **dataset_kw,
) -> DataLoader:
    """``distributed=True`` shards the dataset across torch.distributed ranks with a
    DistributedSampler (call ``loader.sampler.set_epoch(e)`` each epoch)."""
    ds = TileDataset(paths, crop=crop, train=train, seed=seed, **dataset_kw)
    g = torch.Generator()
    g.manual_seed(int(seed))
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=bool(train), seed=int(seed), drop_last=bool(train))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(train) and sampler is None,
        sampler=sampler,
        num_workers=int(num_workers),
        drop_last=bool(train and len(ds) > batch_size) if drop_last is None else bool(drop_last),
        generator=g,
        pin_memory=distributed,
        persistent_workers=bool(num_workers > 0),
    )


__all__ = [
    "INPUT_CHANNELS",
    "TARGET_CHANNELS",
    "TileDataset",
    "build_input",
    "list_tiles",
    "make_loader",
    "split_by_field",
]
