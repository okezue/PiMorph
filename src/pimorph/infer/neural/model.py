"""Multi-head UNet producing the dense evidence maps consumed by the decoder.

Input (B, 6, H, W): image channels [geometry, nuclei, junction], each robust
normalized to roughly [0, 1] (zeros when absent), followed by three constant
presence-indicator channels (1 where the corresponding image channel is present).

Output (B, 6, H, W) in ``HEADS`` order: boundary, seed, vertex and gap are logits,
distance regresses signed_distance / 16 and log_sigma is the per-pixel log scale of
the Laplace likelihood used for the distance head.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS: Tuple[str, ...] = ("boundary", "distance", "seed", "vertex", "gap", "log_sigma")
HEAD_INDEX: Dict[str, int] = {name: i for i, name in enumerate(HEADS)}
DISTANCE_SCALE: float = 16.0


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, 3x3 conv halving the channels, concat with the skip, double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class MultiHeadUNet(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 32, depth: int = 4, out_channels: int = len(HEADS)):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_channels = int(in_channels)
        self.base = int(base)
        self.depth = int(depth)
        self.out_channels = int(out_channels)
        chans = [base * (2**i) for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList([DoubleConv(chans[i], chans[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Up(chans[i + 1], chans[i], chans[i]) for i in reversed(range(depth))])
        self.head = nn.Sequential(
            nn.Conv2d(chans[0], chans[0], 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(chans[0]), chans[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(chans[0], out_channels, 1),
        )
        # start log_sigma near 0 and keep the other heads unbiased
        nn.init.zeros_(self.head[-1].bias)

    @property
    def divisor(self) -> int:
        """Spatial size multiple that avoids interpolation mismatches in the decoder."""
        return 2**self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.inc(x)
        for down in self.downs:
            skips.append(h)
            h = down(F.max_pool2d(h, 2))
        for up in self.ups:
            h = up(h, skips.pop())
        return self.head(h)

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        return {name: out[:, i : i + 1] for i, name in enumerate(HEADS)}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


__all__ = ["DISTANCE_SCALE", "HEADS", "HEAD_INDEX", "MultiHeadUNet", "count_parameters"]
