"""Cellpose-SAM baseline (cellpose >= 4). Optional dependency; import lazily."""

from __future__ import annotations

from typing import Optional

import numpy as np


def cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CellposeSAM:
    """Thin wrapper returning an int32 label image.

    Cellpose-SAM takes up to three channels; nuclei are stacked as the second channel
    when provided. ``diameter=None`` lets the model estimate cell size.
    """

    def __init__(self, pretrained_model: str = "cpsam", device=None):
        from cellpose import models

        self.device = device or pick_device()
        gpu = self.device.type in ("cuda", "mps")
        try:
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model, device=self.device)
        except TypeError:  # older signature
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    def __call__(
        self,
        geometry: np.ndarray,
        nuclei: Optional[np.ndarray] = None,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
    ) -> np.ndarray:
        g = np.asarray(geometry, dtype=np.float32)
        if nuclei is not None:
            img = np.stack([g, np.asarray(nuclei, dtype=np.float32)], axis=-1)
        else:
            img = g
        out = self.model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        masks = out[0]
        return np.asarray(masks).astype(np.int32)
