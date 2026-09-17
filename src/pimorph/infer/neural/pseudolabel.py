"""Consensus pseudo-labels from real fields for domain adaptation.

Two independent segmentations of the same field (classical proposer + constrained
decoder, and Cellpose-SAM when available) are matched face by face. Only the
intersection of confidently matched faces is kept as a label; everything the two
segmentations disagree on is marked ``ignore`` and carries no loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from ...complex.extract import extract_complex
from ...complex.matching import match_faces
from ...io.manifest import FieldSpec, parse_manifest
from ...synth.targets import TARGET_KEYS, make_targets
from ..decoder import ConstrainedDecoder, DecoderParams
from ..proposals import ClassicalProposer, ProposalMaps

PathLike = Union[str, Path]


def _near(mask: np.ndarray, tol_px: float) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    return ndi.distance_transform_edt(~mask) <= tol_px


def consensus_labels(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    iou_thresh: float = 0.7,
    boundary_tol_px: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, ignore)``.

    Matched face pairs with IoU >= ``iou_thresh`` keep their intersection under the
    a-side face id + 1. The symmetric difference of matched faces, all unmatched faces
    and pixels within ``boundary_tol_px`` of a boundary present in only one of the two
    segmentations are ignored.
    """
    labels_a = np.asarray(labels_a).astype(np.int32)
    labels_b = np.asarray(labels_b).astype(np.int32)
    if labels_a.shape != labels_b.shape:
        raise ValueError("label images must have the same shape")
    cx_a = extract_complex(labels_a)
    cx_b = extract_complex(labels_b)
    fm = match_faces(labels_a, labels_b, cx_a, cx_b, iou_thresh=iou_thresh)
    fid_a, fid_b = fm.fid_pred, fm.fid_ref

    out = np.zeros(labels_a.shape, dtype=np.int32)
    ignore = np.zeros(labels_a.shape, dtype=bool)
    matched_a = np.zeros(cx_a.n_faces + 1, dtype=bool)
    matched_b = np.zeros(cx_b.n_faces + 1, dtype=bool)
    b_of_a = np.full(cx_a.n_faces + 1, -1, dtype=np.int64)
    for (fa, fb), iou in zip(fm.pairs, fm.iou):
        if iou < iou_thresh:
            continue
        matched_a[fa] = True
        matched_b[fb] = True
        b_of_a[fa] = fb
    # cell pixels: -1 -> index n_faces (never matched)
    ia = np.where(fid_a >= 0, fid_a, cx_a.n_faces)
    ib = np.where(fid_b >= 0, fid_b, cx_b.n_faces)
    in_matched_a = matched_a[ia]
    in_matched_b = matched_b[ib]
    agree = in_matched_a & (b_of_a[ia] == ib)
    out[agree] = fid_a[agree] + 1
    ignore |= in_matched_a & ~agree  # a-side part of the symmetric difference
    ignore |= in_matched_b & ~agree  # b-side part
    ignore |= (fid_a >= 0) & ~in_matched_a  # unmatched a faces
    ignore |= (fid_b >= 0) & ~in_matched_b  # unmatched b faces

    ba = find_boundaries(labels_a, connectivity=1, mode="thick")
    bb = find_boundaries(labels_b, connectivity=1, mode="thick")
    if ba.any() and bb.any():
        da = ndi.distance_transform_edt(~ba)
        db = ndi.distance_transform_edt(~bb)
        disagree = (ba & (db > boundary_tol_px)) | (bb & (da > boundary_tol_px))
    else:
        disagree = ba | bb
    ignore |= _near(disagree, boundary_tol_px)
    return out, ignore


def classical_only_ignore(labels: np.ndarray, boundary_map: np.ndarray, support: float = 0.3, tol_px: float = 3.0):
    """Ignore pixels near label boundaries that the boundary map does not support."""
    b = find_boundaries(labels, connectivity=1, mode="thick")
    weak = b & (np.asarray(boundary_map) < support)
    return _near(weak, tol_px)


def _downsample2(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    x = np.asarray(img[:H, :W], dtype=np.float32)
    return x.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def _tile_starts(n: int, tile: int, stride: int) -> List[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return starts


def _load_field(spec: FieldSpec):
    im = spec.load()
    geometry = spec.role_channel(im, "geometry")
    nuclei = spec.role_channel(im, "nuclei")
    junction = spec.role_channel(im, "junction")
    if geometry is None:
        geometry = junction
    if geometry is None:
        raise ValueError(f"field {spec.image_id} has neither geometry nor junction channel")
    return (
        np.asarray(geometry, dtype=np.float32),
        None if nuclei is None else np.asarray(nuclei, dtype=np.float32),
        None if junction is None else np.asarray(junction, dtype=np.float32),
        im.pixel_size_um,
    )


def segment_field(
    geometry: np.ndarray,
    nuclei: Optional[np.ndarray],
    junction: Optional[np.ndarray],
    use_cellpose: bool = True,
    cellpose_model=None,
    iou_thresh: float = 0.7,
    boundary_tol_px: float = 3.0,
    downsample_radius_px: float = 35.0,
    policy: str = "consensus",
) -> Dict:
    """Classical + optional Cellpose pseudo-labels on one field. Fields whose classical
    cell radius exceeds ``downsample_radius_px`` are downsampled by 2 first so cell scale
    roughly matches the synthetic prior.

    ``policy``: "consensus" keeps only cells both segmenters agree on (strict, can ignore
    almost everything when they differ); "cellpose_primary" uses Cellpose masks and
    ignores only unsupported or disputed boundaries (denser, inherits Cellpose bias)."""
    proposer = ClassicalProposer()
    maps: ProposalMaps = proposer(geometry, nuclei, junction)
    scale = 1
    if float(maps.meta.get("cell_radius_px", 0.0)) > downsample_radius_px:
        scale = 2
        geometry = _downsample2(geometry)
        nuclei = None if nuclei is None else _downsample2(nuclei)
        junction = None if junction is None else _downsample2(junction)
        maps = proposer(geometry, nuclei, junction)
    res = ConstrainedDecoder().decode(maps, DecoderParams(cell_radius_px=float(maps.meta["cell_radius_px"])))
    labels_cl = res.labels

    labels_cp = None
    if use_cellpose:
        from ..cellpose_sam import CellposeSAM, cellpose_available

        if cellpose_available():
            model = cellpose_model if cellpose_model is not None else CellposeSAM()
            labels_cp = model(geometry, nuclei)
    if labels_cp is not None and policy == "consensus":
        labels, ignore = consensus_labels(labels_cl, labels_cp, iou_thresh=iou_thresh, boundary_tol_px=boundary_tol_px)
        consensus = "classical+cellpose"
    elif labels_cp is not None and policy == "cellpose_primary":
        # Cellpose masks as labels; 1 to 3 px background seams between touching cells are
        # filled so cells share crack edges; boundaries without support in the boundary
        # map, and cells the two segmenters disagree on (IoU < iou_thresh with any
        # classical cell), are ignored.
        from ...bench.datasets import fill_gt_slivers

        labels = fill_gt_slivers(labels_cp, max_area_px=12)
        ignore = classical_only_ignore(labels, maps.boundary, support=0.3, tol_px=boundary_tol_px)
        _, ign_disagree = consensus_labels(labels, labels_cl, iou_thresh=iou_thresh, boundary_tol_px=boundary_tol_px)
        ignore = ignore | (ign_disagree & find_boundaries(labels, connectivity=1, mode="thick"))
        consensus = "cellpose_primary"
    else:
        labels = labels_cl
        ignore = classical_only_ignore(labels_cl, maps.boundary, support=0.3, tol_px=boundary_tol_px)
        consensus = "classical_only"
    return {
        "geometry": geometry,
        "nuclei": nuclei,
        "junction": junction,
        "labels": labels.astype(np.int32),
        "ignore": ignore,
        "labels_classical": labels_cl,
        "labels_cellpose": labels_cp,
        "maps": maps,
        "scale": scale,
        "consensus": consensus,
    }


def make_pseudolabel_tiles(
    manifest_csv: PathLike,
    out_dir: PathLike,
    tile: int = 512,
    stride: int = 512,
    max_fields: Optional[int] = None,
    use_cellpose: bool = True,
    seed: int = 0,
    root: Optional[PathLike] = None,
    iou_thresh: float = 0.7,
    boundary_tol_px: float = 3.0,
    policy: str = "consensus",
) -> pd.DataFrame:
    """Write ``field_XXX_rYYYY_cZZZZ.npz`` tiles plus ``manifest.csv`` into ``out_dir``.

    Tile keys: ``junction`` (the geometry channel; the separate junction channel when
    geometry comes from a membrane marker, in which case ``membrane`` holds the
    geometry), ``nuclei`` (zeros with ``has_nuclei=False`` when absent), ``labels``,
    ``ignore`` and the targets of ``pimorph.synth.targets.make_targets``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_manifest(manifest_csv, root=root)
    rng = np.random.default_rng(seed)
    if max_fields is not None and len(specs) > max_fields:
        idx = np.sort(rng.choice(len(specs), size=int(max_fields), replace=False))
        specs = [specs[i] for i in idx]

    cellpose_model = None
    if use_cellpose:
        from ..cellpose_sam import CellposeSAM, cellpose_available

        if cellpose_available():
            cellpose_model = CellposeSAM()

    rows: List[Dict] = []
    for k, spec in enumerate(specs):
        geometry, nuclei, junction, px_um = _load_field(spec)
        seg = segment_field(
            geometry,
            nuclei,
            junction,
            use_cellpose=use_cellpose and cellpose_model is not None,
            cellpose_model=cellpose_model,
            iou_thresh=iou_thresh,
            boundary_tol_px=boundary_tol_px,
            policy=policy,
        )
        g, n, j = seg["geometry"], seg["nuclei"], seg["junction"]
        labels, ignore = seg["labels"], seg["ignore"]
        separate_junction = j is not None and spec.geometry_source == "membrane_channel" and j.shape == g.shape
        H, W = labels.shape
        for r0 in _tile_starts(H, tile, stride):
            for c0 in _tile_starts(W, tile, stride):
                rs, cs = slice(r0, min(r0 + tile, H)), slice(c0, min(c0 + tile, W))
                lab_t = labels[rs, cs]
                ign_t = ignore[rs, cs]
                targets = make_targets(lab_t)
                arrays: Dict[str, np.ndarray] = {
                    "junction": (j if separate_junction else g)[rs, cs].astype(np.float32),
                    "nuclei": (n[rs, cs] if n is not None else np.zeros(lab_t.shape)).astype(np.float32),
                    "has_nuclei": np.array(n is not None),
                    "labels": lab_t.astype(np.int32),
                    "ignore": ign_t.astype(bool),
                    **{key: targets[key] for key in TARGET_KEYS},
                }
                if separate_junction:
                    arrays["membrane"] = g[rs, cs].astype(np.float32)
                fname = f"field_{k:03d}_r{r0:04d}_c{c0:04d}.npz"
                np.savez_compressed(out_dir / fname, **arrays)
                n_cells = int(np.unique(lab_t[lab_t > 0]).size)
                rows.append(
                    {
                        "file": fname,
                        "source_field": spec.image_id,
                        "row0": r0,
                        "col0": c0,
                        "scale": seg["scale"],
                        "consensus": seg["consensus"],
                        "has_nuclei": n is not None,
                        "has_membrane": separate_junction,
                        "n_cells": n_cells,
                        "ignore_frac": float(ign_t.mean()),
                        "cell_radius_px": float(seg["maps"].meta["cell_radius_px"]),
                        "pixel_size_um": px_um,
                        "condition": spec.condition,
                        "dataset": spec.dataset,
                    }
                )
        print(
            f"[pseudolabel] {spec.image_id}: {seg['consensus']} scale={seg['scale']} "
            f"cells={int(np.unique(labels[labels > 0]).size)} ignore={ignore.mean():.2f}",
            flush=True,
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


__all__ = ["classical_only_ignore", "consensus_labels", "make_pseudolabel_tiles", "segment_field"]
