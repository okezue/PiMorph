"""Losses for the multi-head model. All pixel losses take a weight map ``w`` (B,1,H,W)
and normalise by its sum, so ignored pixels contribute nothing."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import HEAD_INDEX

_EPS = 1e-6


def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp_min(_EPS)


def bce_with_pos_weight(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, pos_weight: float = 1.0
) -> torch.Tensor:
    pw = torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return _wmean(loss, weight)


def focal_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy against a (possibly soft) target."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    return _wmean(bce * (1.0 - p_t).clamp_min(0.0) ** gamma, weight)


def heteroscedastic_l1(
    pred: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Laplace negative log-likelihood up to a constant: |e| exp(-s) + s."""
    log_sigma = log_sigma.clamp(-6.0, 6.0)
    return _wmean((pred - target).abs() * torch.exp(-log_sigma) + log_sigma, weight)


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, 3, stride=1, padding=1)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_skeletonize(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Iterated soft opening residuals (Shit et al., clDice)."""
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(int(iters)):
        x = soft_erode(x)
        skel = skel + F.relu(x - soft_dilate(soft_erode(x))) * (1.0 - skel)
    return skel


def soft_cldice(prob: torch.Tensor, target: torch.Tensor, iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    """1 - clDice between a probability map and a binary target (both (B,1,H,W))."""
    skel_p = soft_skeletonize(prob, iters)
    skel_t = soft_skeletonize(target, iters)
    tprec = ((skel_p * target).sum() + smooth) / (skel_p.sum() + smooth)
    tsens = ((skel_t * prob).sum() + smooth) / (skel_t.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens).clamp_min(_EPS)


DEFAULT_WEIGHTS: Dict[str, float] = dict(boundary=1.0, distance=1.0, seed=1.0, vertex=1.0, gap=0.5, cldice=0.3)
DEFAULT_POS_WEIGHTS: Dict[str, float] = dict(boundary=3.0, seed=5.0, vertex=5.0)


class MultiHeadLoss(nn.Module):
    """Combine per-head losses. ``pred`` (B,6,H,W) in HEADS order, ``y`` (B,5,H,W) in
    TARGET_CHANNELS order (boundary, distance/16, seed, vertex, gap), ``w`` (B,1,H,W)."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        pos_weights: Optional[Dict[str, float]] = None,
        gap_focal_gamma: float = 2.0,
        cldice_iters: int = 3,
    ):
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.pos_weights = dict(DEFAULT_POS_WEIGHTS, **(pos_weights or {}))
        self.gap_focal_gamma = float(gap_focal_gamma)
        self.cldice_iters = int(cldice_iters)

    def forward(self, pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = pred.float()
        y = y.float()
        w = w.float()
        b_logit = pred[:, HEAD_INDEX["boundary"] : HEAD_INDEX["boundary"] + 1]
        d_pred = pred[:, HEAD_INDEX["distance"] : HEAD_INDEX["distance"] + 1]
        s_logit = pred[:, HEAD_INDEX["seed"] : HEAD_INDEX["seed"] + 1]
        v_logit = pred[:, HEAD_INDEX["vertex"] : HEAD_INDEX["vertex"] + 1]
        g_logit = pred[:, HEAD_INDEX["gap"] : HEAD_INDEX["gap"] + 1]
        log_sigma = pred[:, HEAD_INDEX["log_sigma"] : HEAD_INDEX["log_sigma"] + 1]
        yb, yd, ys, yv, yg = (y[:, i : i + 1] for i in range(5))

        parts = {
            "boundary": bce_with_pos_weight(b_logit, yb, w, self.pos_weights["boundary"]),
            "distance": heteroscedastic_l1(d_pred, log_sigma, yd, w),
            "seed": bce_with_pos_weight(s_logit, ys, w, self.pos_weights["seed"]),
            "vertex": bce_with_pos_weight(v_logit, yv, w, self.pos_weights["vertex"]),
            "gap": focal_bce(g_logit, yg, w, self.gap_focal_gamma),
            "cldice": soft_cldice(torch.sigmoid(b_logit) * w, yb * w, self.cldice_iters),
        }
        total = sum(self.weights.get(k, 0.0) * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


__all__ = [
    "DEFAULT_POS_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "MultiHeadLoss",
    "bce_with_pos_weight",
    "focal_bce",
    "heteroscedastic_l1",
    "soft_cldice",
    "soft_skeletonize",
]
