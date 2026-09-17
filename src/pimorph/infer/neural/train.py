"""Training loop for the multi-head UNet.

    python -m pimorph.infer.neural.train --train-dirs data/tiles/synth_train \\
        --val-dirs data/tiles/synth_val --out runs/neural/v1 --epochs 40

Writes ``config.json``, ``train_log.jsonl`` (one line per epoch), ``last.pt`` every
epoch and ``best.pt`` on the best validation loss. Checkpoints hold
``{"model", "config", "epoch", "val"}`` plus optimizer state for resuming.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from .data import list_tiles, make_loader, split_by_field
from .losses import MultiHeadLoss
from .model import HEAD_INDEX, MultiHeadUNet, count_parameters


@dataclass
class TrainConfig:
    train_dirs: List[str] = field(default_factory=list)
    val_dirs: List[str] = field(default_factory=list)
    out_dir: str = "runs/neural"
    epochs: int = 20
    batch_size: int = 8
    crop: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    base: int = 32
    depth: int = 4
    amp: bool = True
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    log_every: int = 20
    max_steps: Optional[int] = None
    resume: Optional[str] = None
    val_fraction: float = 0.1  # used only when val_dirs is empty
    p_membrane_as_geometry: float = 0.5
    p_nuclei: float = 0.85
    p_junction: float = 0.9
    warmup_fraction: float = 0.03
    grad_clip: float = 1.0


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_with_warmup(optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(int(round(warmup_fraction * total_steps)), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ metrics
def _peaks(heat: np.ndarray, thresh: float, min_distance: int = 3) -> np.ndarray:
    if heat.max() < thresh:
        return np.zeros((0, 2))
    return peak_local_max(heat, min_distance=min_distance, threshold_abs=thresh, exclude_border=False)


def peak_recall(pred_heat: np.ndarray, gt_heat: np.ndarray, tol_px: float, pred_thresh: float = 0.3) -> Tuple[int, int]:
    """(hits, n_gt): GT maxima with a predicted peak within ``tol_px``."""
    gt = _peaks(gt_heat, 0.5)
    if gt.shape[0] == 0:
        return 0, 0
    pr = _peaks(pred_heat, pred_thresh)
    if pr.shape[0] == 0:
        return 0, int(gt.shape[0])
    d, _ = cKDTree(pr).query(gt, k=1)
    return int(np.count_nonzero(d <= tol_px)), int(gt.shape[0])


class _MetricAccumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.seed_hit = self.seed_n = 0
        self.vert_hit = self.vert_n = 0
        self.gap_inter = self.gap_union = 0
        self.dist_abs = 0.0
        self.dist_n = 0.0

    def update(self, pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
        p = torch.sigmoid(pred[:, [HEAD_INDEX[k] for k in ("boundary", "seed", "vertex", "gap")]]).float().cpu().numpy()
        d = pred[:, HEAD_INDEX["distance"]].float().cpu().numpy()
        yy = y.float().cpu().numpy()
        valid = w.float().cpu().numpy()[:, 0] > 0
        for b in range(p.shape[0]):
            v = valid[b]
            pb, tb = p[b, 0] > 0.5, yy[b, 0] > 0.5
            self.tp += int(np.count_nonzero(pb & tb & v))
            self.fp += int(np.count_nonzero(pb & ~tb & v))
            self.fn += int(np.count_nonzero(~pb & tb & v))
            h, n = peak_recall(p[b, 1] * v, yy[b, 2] * v, tol_px=4.0)
            self.seed_hit += h
            self.seed_n += n
            h, n = peak_recall(p[b, 2] * v, yy[b, 3] * v, tol_px=3.0)
            self.vert_hit += h
            self.vert_n += n
            pg, tg = (p[b, 3] > 0.5) & v, (yy[b, 4] > 0.5) & v
            self.gap_inter += int(np.count_nonzero(pg & tg))
            self.gap_union += int(np.count_nonzero(pg | tg))
            self.dist_abs += float(np.abs(d[b] - yy[b, 1])[v].sum())
            self.dist_n += float(v.sum())

    def summary(self) -> Dict[str, float]:
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        return {
            "boundary_f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "boundary_precision": prec,
            "boundary_recall": rec,
            "seed_peak_recall": self.seed_hit / self.seed_n if self.seed_n else float("nan"),
            "vertex_peak_recall": self.vert_hit / self.vert_n if self.vert_n else float("nan"),
            "gap_iou": self.gap_inter / self.gap_union if self.gap_union else float("nan"),
            "distance_mae_px": 16.0 * self.dist_abs / max(self.dist_n, 1.0),
        }


# ----------------------------------------------------------------- helpers
def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _use_scaler(device: torch.device, enabled: bool) -> bool:
    return bool(enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported())


def _mean_parts(parts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not parts:
        return {}
    keys = parts[0].keys()
    return {k: float(np.mean([p[k] for p in parts])) for k in keys}


def _resolve_splits(cfg: TrainConfig) -> Tuple[List[Path], List[Path]]:
    train_paths = list_tiles(cfg.train_dirs)
    if cfg.val_dirs:
        val_paths = list_tiles(cfg.val_dirs)
    else:
        train_paths, val_paths = split_by_field(train_paths, cfg.val_fraction, cfg.seed)
    if not train_paths:
        raise FileNotFoundError(f"no .npz tiles under {cfg.train_dirs}")
    if not val_paths:
        val_paths = train_paths[: max(1, len(train_paths) // 10)]
    return train_paths, val_paths


def evaluate(model, loader, criterion, device: torch.device, amp: bool) -> Tuple[float, Dict[str, float], Dict]:
    model.eval()
    parts: List[Dict[str, float]] = []
    totals: List[float] = []
    acc = _MetricAccumulator()
    with torch.no_grad():
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            with _autocast(device, amp):
                pred = model(x)
            total, part = criterion(pred, y, w)
            totals.append(float(total))
            parts.append(part)
            acc.update(pred, y, w)
    return float(np.mean(totals)) if totals else float("nan"), _mean_parts(parts), acc.summary()


def save_checkpoint(
    path: Path,
    model,
    cfg: TrainConfig,
    epoch: int,
    val: Dict,
    optimizer=None,
    scheduler=None,
    best_val: Optional[float] = None,
) -> None:
    ck = {"model": model.state_dict(), "config": asdict(cfg), "epoch": int(epoch), "val": val}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler"] = scheduler.state_dict()
    if best_val is not None:
        ck["best_val"] = float(best_val)
    torch.save(ck, path)


def load_model(checkpoint_path, device: torch.device = torch.device("cpu")) -> Tuple[MultiHeadUNet, Dict]:
    ck = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = MultiHeadUNet(in_channels=6, base=int(cfg.get("base", 32)), depth=int(cfg.get("depth", 4)))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


# ------------------------------------------------------------------- train
def train(cfg: TrainConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp = bool(cfg.amp and device.type == "cuda")

    train_paths, val_paths = _resolve_splits(cfg)
    ds_kw = dict(p_membrane_as_geometry=cfg.p_membrane_as_geometry, p_nuclei=cfg.p_nuclei, p_junction=cfg.p_junction)
    train_loader = make_loader(
        train_paths, cfg.batch_size, cfg.crop, train=True, num_workers=cfg.num_workers, seed=cfg.seed, **ds_kw
    )
    val_loader = make_loader(
        val_paths, cfg.batch_size, cfg.crop, train=False, num_workers=cfg.num_workers, seed=cfg.seed, **ds_kw
    )

    model = MultiHeadUNet(in_channels=6, base=cfg.base, depth=cfg.depth).to(device)
    criterion = MultiHeadLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = (
        cfg.epochs * steps_per_epoch if cfg.max_steps is None else min(cfg.max_steps, cfg.epochs * steps_per_epoch)
    )
    total_steps = max(int(total_steps), 1)
    scheduler = cosine_with_warmup(optimizer, total_steps, cfg.warmup_fraction)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_scaler(device, amp))

    start_epoch, best_val, step = 0, float("inf"), 0
    if cfg.resume:
        ck = torch.load(Path(cfg.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_val = float(ck.get("best_val", best_val))
        step = start_epoch * steps_per_epoch

    print(
        f"[train] device={device} params={count_parameters(model):,} train_tiles={len(train_paths)} "
        f"val_tiles={len(val_paths)} steps/epoch={steps_per_epoch} total_steps={total_steps} amp={amp}",
        flush=True,
    )
    log_path = out_dir / "train_log.jsonl"
    done = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loader.dataset.set_epoch(epoch)
        t0 = time.time()
        parts: List[Dict[str, float]] = []
        totals: List[float] = []
        for x, y, w in train_loader:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            with _autocast(device, amp):
                pred = model(x)
            total, part = criterion(pred, y, w)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            totals.append(float(total.detach()))
            parts.append(part)
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"[train] epoch {epoch} step {step} loss {np.mean(totals[-cfg.log_every :]):.4f}", flush=True)
            if cfg.max_steps is not None and step >= cfg.max_steps:
                done = True
                break

        val_total, val_parts, val_metrics = evaluate(model, val_loader, criterion, device, amp)
        record = {
            "epoch": epoch,
            "step": step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_s": time.time() - t0,
            "train_loss": float(np.mean(totals)) if totals else float("nan"),
            "train_parts": _mean_parts(parts),
            "val_loss": val_total,
            "val_parts": val_parts,
            "val_metrics": val_metrics,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[val] epoch {epoch} loss {val_total:.4f} "
            + " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items() if not np.isnan(v)),
            flush=True,
        )
        val_record = {"loss": val_total, "parts": val_parts, "metrics": val_metrics}
        is_best = val_total < best_val or not math.isfinite(best_val)
        if is_best:
            best_val = val_total
        save_checkpoint(out_dir / "last.pt", model, cfg, epoch, val_record, optimizer, scheduler, best_val)
        if is_best:
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch, val_record, best_val=best_val)
        if done:
            break
    if not (out_dir / "best.pt").exists():
        save_checkpoint(out_dir / "best.pt", model, cfg, start_epoch, {}, best_val=best_val)
    return out_dir / "best.pt"


# --------------------------------------------------------------------- CLI
def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("train_dirs", "val_dirs"):
            parser.add_argument(flag, nargs="*", default=list(), help=f"{f.name} (directories or .npz files)")
        elif f.name == "out_dir":
            parser.add_argument("--out", "--out-dir", dest="out_dir", default=f.default)
        elif f.type == "bool" or isinstance(f.default, bool):
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.name in ("max_steps",):
            parser.add_argument(flag, type=int, default=None)
        elif f.name in ("resume",):
            parser.add_argument(flag, type=str, default=None)
        elif isinstance(f.default, int):
            parser.add_argument(flag, type=int, default=f.default)
        elif isinstance(f.default, float):
            parser.add_argument(flag, type=float, default=f.default)
        else:
            parser.add_argument(flag, type=str, default=f.default)


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the PiMorph multi-head UNet.")
    _add_config_flags(parser)
    ns = parser.parse_args(argv)
    return TrainConfig(**{f.name: getattr(ns, f.name) for f in fields(TrainConfig)})


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    if not cfg.train_dirs:
        raise SystemExit("--train-dirs is required")
    best = train(cfg)
    print(f"[train] best checkpoint: {best}")


if __name__ == "__main__":
    main()


__all__ = ["TrainConfig", "cosine_with_warmup", "evaluate", "load_model", "peak_recall", "resolve_device", "train"]
