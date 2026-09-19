"""Run reconstruction methods over a benchmark dataset and score them.

Each method maps a BenchItem to a label image; the complex is then extracted and
compared with the complex of the ground-truth labels via
``pimorph.metrics.structural_metrics``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from skimage.filters import sato

from ..infer.decoder import ConstrainedDecoder, DecoderParams
from ..infer.proposals import ClassicalProposer, robust_normalize
from ..metrics.structural import legacy_adjacency_f1, structural_metrics
from .datasets import BenchItem, load_dataset

SUMMARY_KEYS = [
    "adjacency_pair_f1",
    "adjacency_pair_precision",
    "adjacency_pair_recall",
    "adjacency_component_f1",
    "vertex_f1",
    "vertex_loc_error_median_px",
    "vertex_incident_set_accuracy",
    "vertex_cyclic_order_accuracy",
    "pq",
    "ap50",
    "vi",
    "boundary_f1",
    "n_splits",
    "n_merges",
    "complex_edit_distance_approx",
    "validity_fraction",
    "euler_residual_pred",
    "legacy_f1",
    "method_runtime_s",
]


def resolve_polarity(item: BenchItem) -> str:
    """Bright or dark boundaries in the geometry channel, without touching the GT.

    Compares the strength of bright-ridge and dark-ridge responses (99th percentile
    over median) and picks the larger."""
    if item.boundary_polarity in ("bright", "dark"):
        return item.boundary_polarity
    g = robust_normalize(item.geometry)
    if max(g.shape) > 512:
        g = g[:: int(np.ceil(max(g.shape) / 512)), :: int(np.ceil(max(g.shape) / 512))]
    rb = sato(g, sigmas=(1.0, 2.0), black_ridges=False)
    rd = sato(g, sigmas=(1.0, 2.0), black_ridges=True)
    sb = np.percentile(rb, 99) / (np.median(rb) + 1e-6)
    sd = np.percentile(rd, 99) / (np.median(rd) + 1e-6)
    return "bright" if sb >= sd else "dark"


def geometry_for_bright_boundaries(item: BenchItem) -> np.ndarray:
    pol = resolve_polarity(item)
    g = np.asarray(item.geometry, dtype=np.float32)
    if pol == "dark":
        g = float(g.max()) - g
    return g


def tissue_for_item(item: BenchItem, scale_px: float = 15.0) -> np.ndarray:
    """Tissue mask on the ORIGINAL image so dark image corners stay background even
    when the geometry channel is inverted for dark boundaries."""
    from ..infer.proposals import tissue_mask

    return tissue_mask(np.asarray(item.geometry, dtype=np.float32), scale_px=scale_px)


def method_gt(item: BenchItem) -> np.ndarray:
    return item.labels_gt.copy()


def method_classical(item: BenchItem) -> np.ndarray:
    g = geometry_for_bright_boundaries(item)
    maps = ClassicalProposer()(g, item.nuclei, tissue=tissue_for_item(item))
    dec = ConstrainedDecoder(pixel_size_um=item.pixel_size_um)
    params = DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)))
    if item.nuclei is None:
        # no nuclear evidence: gaps cannot be told from cells without seeds
        params = params.with_(fill_gaps=True)
    return dec.decode(maps, params).labels


_cellpose_model = None


def method_cellpose_sam(item: BenchItem) -> np.ndarray:
    global _cellpose_model
    from ..infer.cellpose_sam import CellposeSAM

    if _cellpose_model is None:
        _cellpose_model = CellposeSAM()
    return _cellpose_model(item.geometry, item.nuclei)


_neural_proposer = None
NEURAL_CHECKPOINT = os.environ.get("PIMORPH_NEURAL_CKPT", "models/pimorph_proposals_v0.pt")
# optional decoder overrides for the neural methods, e.g. '{"vertex_weight": 0.5, "boundary_smooth_sigma": 1.0}'
NEURAL_DECODER_PARAMS = json.loads(os.environ.get("PIMORPH_DECODER_PARAMS", "{}"))
NEURAL_TTA = os.environ.get("PIMORPH_NEURAL_TTA", "0") not in ("", "0", "false", "False")


def _neural_maps(item: BenchItem):
    global _neural_proposer
    from ..infer.neural.proposer import NeuralProposer

    if _neural_proposer is None:
        _neural_proposer = NeuralProposer(NEURAL_CHECKPOINT, tta=NEURAL_TTA)
    g = geometry_for_bright_boundaries(item)
    maps = _neural_proposer(g, item.nuclei, tissue=tissue_for_item(item))
    params = DecoderParams(cell_radius_px=float(maps.meta.get("cell_radius_px", 15.0)), **NEURAL_DECODER_PARAMS)
    return g, maps, params


def method_neural(item: BenchItem) -> np.ndarray:
    """Neural proposal maps -> constrained decoder. Checkpoint from PIMORPH_NEURAL_CKPT,
    decoder overrides from PIMORPH_DECODER_PARAMS (JSON), TTA from PIMORPH_NEURAL_TTA."""
    _, maps, params = _neural_maps(item)
    return ConstrainedDecoder(pixel_size_um=item.pixel_size_um).decode(maps, params).labels


def method_neural_map(item: BenchItem) -> np.ndarray:
    """Posterior MAP: the hypothesis grid plus merge/split moves scored by the full
    energy (renderer likelihood, curve, soft priors); the lowest-energy complex wins.
    Tests whether the energy chooses better than the single default decode."""
    from ..infer.posterior import PosteriorEnsemble, generate_hypotheses

    g, maps, params = _neural_maps(item)
    dec = ConstrainedDecoder(pixel_size_um=item.pixel_size_um)
    hyps = generate_hypotheses(maps, dec, params, image=g, n_merge_moves=4, n_split_moves=4)
    if not hyps:
        return dec.decode(maps, params).labels
    return PosteriorEnsemble.from_hypotheses(hyps, ess_min=4).map_hypothesis.labels


def method_cellpose_sam_filled(item: BenchItem) -> np.ndarray:
    """Cellpose-SAM masks with sub-12 px background seams between touching cells filled,
    so cells share crack edges and vertex metrics measure the segmentation rather than
    the mask format. The same rule is applied to polygon ground truth."""
    from .datasets import fill_gt_slivers

    return fill_gt_slivers(method_cellpose_sam(item), 12)


METHODS: Dict[str, Callable[[BenchItem], np.ndarray]] = {
    "gt": method_gt,
    "classical": method_classical,
    "cellpose_sam": method_cellpose_sam,
    "cellpose_sam_filled": method_cellpose_sam_filled,
    "neural": method_neural,
    "neural_map": method_neural_map,
}


def restrict_to_roi(labels: np.ndarray, roi: Optional[np.ndarray], min_area_px: int = 30) -> np.ndarray:
    """Zero predictions outside the annotated region and drop the fragments this leaves
    behind, so partially annotated fields score only what the annotators traced."""
    if roi is None:
        return labels
    from skimage.measure import label as cc_label

    from ..infer.decoder import merge_small_regions

    out = np.where(roi, labels, 0).astype(np.int32)
    out = cc_label(out, connectivity=1).astype(np.int32)
    return merge_small_regions(out, min_area_px)


def evaluate_item(item: BenchItem, method: str, tol_px: float = 3.0) -> Dict:
    t0 = time.time()
    labels_pred = METHODS[method](item)
    rt = time.time() - t0
    labels_pred = restrict_to_roi(labels_pred, item.roi)
    m = structural_metrics(labels_pred, item.labels_gt, pixel_size_um=item.pixel_size_um, tol_px=tol_px)
    leg = legacy_adjacency_f1(labels_pred, item.labels_gt)
    m["legacy_f1"] = float(leg.get("f1", np.nan))
    m["legacy_precision"] = float(leg.get("precision", np.nan))
    m["legacy_recall"] = float(leg.get("recall", np.nan))
    m["method_runtime_s"] = rt
    m["method"] = method
    m["image_id"] = item.image_id
    m["dataset"] = item.meta.get("dataset", "")
    m["shape"] = list(item.labels_gt.shape)
    return m


def run_benchmark(
    dataset: str,
    methods: Iterable[str] = ("classical",),
    root: Optional[Path | str] = None,
    max_items: Optional[int] = None,
    out_dir: Optional[Path | str] = None,
    tol_px: float = 3.0,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: List[Dict] = []
    methods = list(methods)
    for item in load_dataset(dataset, root=root, max_items=max_items):
        for method in methods:
            if method in ("cellpose_sam", "cellpose_sam_filled", "neural", "neural_map"):
                from ..infer.cellpose_sam import cellpose_available

                if not cellpose_available():
                    if verbose:
                        print(f"torch not installed; skipping {method}")
                    continue
                if method.startswith("neural") and not Path(NEURAL_CHECKPOINT).exists():
                    if verbose:
                        print(f"no checkpoint at {NEURAL_CHECKPOINT}; skipping neural")
                    continue
            try:
                r = evaluate_item(item, method, tol_px=tol_px)
            except Exception as e:  # keep the run going, record the failure
                r = {"method": method, "image_id": item.image_id, "dataset": dataset, "error": repr(e)}
            rows.append(r)
            if verbose:
                f1 = r.get("adjacency_pair_f1", float("nan"))
                vf = r.get("vertex_f1", float("nan"))
                rt = r.get("method_runtime_s", 0.0)
                print(f"[{dataset}] {item.image_id:40s} {method:13s} adjF1={f1:.3f} vertexF1={vf:.3f} rt={rt:.1f}s")
    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{dataset}_per_image.csv", index=False)
        summ = summarize(df)
        summ.to_csv(out_dir / f"{dataset}_summary.csv")
        (out_dir / f"{dataset}_summary.md").write_text(summary_markdown(dataset, df, summ), encoding="utf-8")
        (out_dir / f"{dataset}_summary.json").write_text(
            json.dumps(
                {"dataset": dataset, "n_images": int(df["image_id"].nunique()), "summary": json.loads(summ.to_json())},
                indent=2,
            ),
            encoding="utf-8",
        )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in SUMMARY_KEYS if k in df.columns]
    good = df[df.get("error").isna()] if "error" in df.columns else df
    agg = good.groupby("method")[keys].agg(["mean", "median"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_images"] = good.groupby("method")["image_id"].nunique()
    if "error" in df.columns:
        agg["n_errors"] = df.groupby("method")["error"].apply(lambda s: int(s.notna().sum()))
    return agg


def summary_markdown(dataset: str, df: pd.DataFrame, summ: pd.DataFrame) -> str:
    cols = [
        ("adjacency_pair_f1_mean", "Adjacency F1 (pair)"),
        ("adjacency_component_f1_mean", "Adjacency F1 (component)"),
        ("vertex_f1_mean", "Vertex F1"),
        ("vertex_loc_error_median_px_median", "Vertex loc. err. median (px)"),
        ("vertex_incident_set_accuracy_mean", "Incident-set acc."),
        ("vertex_cyclic_order_accuracy_mean", "Cyclic-order acc."),
        ("pq_mean", "PQ"),
        ("boundary_f1_mean", "Boundary F1"),
        ("complex_edit_distance_approx_mean", "Edit dist. (approx)"),
        ("validity_fraction_mean", "Valid"),
        ("legacy_f1_mean", "Legacy 4-nbr adj. F1"),
        ("method_runtime_s_mean", "Runtime (s)"),
    ]
    cols = [(k, n) for k, n in cols if k in summ.columns]
    lines = [f"# Benchmark: {dataset}", "", f"Images: {df['image_id'].nunique()}", ""]
    lines.append("| Method | " + " | ".join(n for _, n in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for method, row in summ.iterrows():
        vals = []
        for k, _ in cols:
            v = row[k]
            vals.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        lines.append(f"| {method} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("Vertex metrics compare multicellular vertices (>= 3 incident cells) derived exactly from the GT")
    lines.append("label image with those of the reconstruction; incident-set and cyclic-order accuracy are over")
    lines.append("matched vertices.")
    return "\n".join(lines)
