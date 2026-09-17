"""Neural proposal module (requires torch; nothing outside this package imports it).

- ``model``: ``MultiHeadUNet`` with heads boundary, distance, seed, vertex, gap, log_sigma.
- ``data``: ``TileDataset`` over synthetic or pseudo-label ``.npz`` tiles with channel dropout.
- ``losses``: weighted BCE / focal / heteroscedastic L1 / soft clDice and ``MultiHeadLoss``.
- ``train``: ``TrainConfig`` + ``train()`` (also ``python -m pimorph.infer.neural.train``).
- ``pseudolabel``: classical + Cellpose consensus tiles from real manifests.
- ``proposer``: ``NeuralProposer`` producing ``ProposalMaps`` for the constrained decoder.
"""

from .data import TileDataset, build_input, list_tiles, make_loader, split_by_field
from .losses import MultiHeadLoss, bce_with_pos_weight, focal_bce, heteroscedastic_l1, soft_cldice
from .model import DISTANCE_SCALE, HEADS, MultiHeadUNet, count_parameters
from .proposer import NeuralProposer, tiled_predict
from .pseudolabel import consensus_labels, make_pseudolabel_tiles
from .train import TrainConfig, load_model, train  # the function shadows the submodule name on purpose

__all__ = [
    "DISTANCE_SCALE",
    "HEADS",
    "MultiHeadLoss",
    "MultiHeadUNet",
    "NeuralProposer",
    "TileDataset",
    "TrainConfig",
    "bce_with_pos_weight",
    "build_input",
    "consensus_labels",
    "count_parameters",
    "focal_bce",
    "heteroscedastic_l1",
    "list_tiles",
    "load_model",
    "make_loader",
    "make_pseudolabel_tiles",
    "soft_cldice",
    "split_by_field",
    "tiled_predict",
    "train",
]
