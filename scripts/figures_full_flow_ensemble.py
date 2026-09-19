#!/usr/bin/env python3
"""Rebuild the finite ensemble used by the complete PiMorph flow figure.

Run from the repository root after installing PiMorph::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/figures_full_flow_ensemble.py

The input, ``docs/figure_data/methods_in_action_arrays.npz``, is reproduced by
``scripts/figures_methods.py``. This script writes
``docs/figure_data/methods_full_flow_ensemble.npz`` and the matching JSON provenance
record. It uses the same synthetic membrane/nuclear images, classical proposer,
and base decoder as the in-action figure; no checkpoint or GPU is required.

This is a finite-candidate sensitivity demonstration, not empirical calibration.
The best-energy candidate need not recover the known synthetic ground truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pimorph.complex import validate
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams
from pimorph.infer.posterior import PosteriorEnsemble, generate_hypotheses
from pimorph.infer.proposals import ClassicalProposer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figure_data"


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    source = DATA / "methods_in_action_arrays.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}; run scripts/figures_methods.py first")
    with np.load(source) as arrays:
        membrane = arrays["membrane"]
        nuclei = arrays["nuclei"]

    proposer_params = dict(
        auto_scale=False,
        nucleus_radius_px=3.5,
        seed_min_distance_px=13,
        cell_radius_px=30.0,
        ridge_sigmas=(0.8, 1.3, 2.0),
        use_tissue_mask=False,
    )
    base_params = dict(cell_radius_px=30.0, min_cell_area_px=150, fill_gaps=True)
    grid_params = dict(
        seed_thresholds=(0.2, 0.35, 0.5),
        seed_drop_fracs=(0.0, 0.05),
        boundary_smooth_sigmas=(0.0, 1.5),
        distance_mixes=(0.0, 0.3),
        gap_thresholds=(0.7,),
        n_merge_moves=2,
        n_split_moves=2,
        max_hypotheses=24,
    )
    maps = ClassicalProposer(**proposer_params)(membrane, nuclei)
    hypotheses = generate_hypotheses(
        maps,
        ConstrainedDecoder(),
        DecoderParams(**base_params),
        image=membrane,
        **grid_params,
    )
    if not hypotheses:
        raise RuntimeError("The decoder generated no hypotheses")
    posterior = PosteriorEnsemble.from_hypotheses(hypotheses, ess_min=4.0)

    reports = []
    for index, hypothesis in enumerate(posterior.hypotheses):
        report = validate(hypothesis.cx)
        if not report.ok:
            raise RuntimeError(f"Invalid hypothesis {index} ({hypothesis.tag}): {report.as_dict()}")
        reports.append(
            dict(
                index=index,
                tag=hypothesis.tag,
                energy=float(hypothesis.energy),
                weight=float(posterior.weights[index]),
                n_cells=int(hypothesis.cx.cell_faces.size),
                n_gaps=int(hypothesis.cx.gap_faces.size),
                validation=report.as_dict(),
                decoder_params=hypothesis.params.__dict__,
                energy_breakdown=hypothesis.breakdown,
            )
        )
    contacts = [
        dict(cells=list(key), probability=float(value))
        for key, value in sorted(posterior.contact_probabilities().items())
    ]
    vertices = [
        dict(cells=sorted(key), **value)
        for key, value in sorted(
            posterior.vertex_credible_radius().items(),
            key=lambda item: (-item[1]["rms_radius_px"], tuple(sorted(item[0]))),
        )
    ]
    energies = np.array([hypothesis.energy for hypothesis in posterior.hypotheses])
    labels = np.stack([hypothesis.labels for hypothesis in posterior.hypotheses]).astype(np.int32)
    npz_path = DATA / "methods_full_flow_ensemble.npz"
    np.savez_compressed(
        npz_path,
        energies=energies,
        weights=posterior.weights,
        labels=labels,
        tags=np.array([hypothesis.tag for hypothesis in posterior.hypotheses]),
        contact_cells=np.array([contact["cells"] for contact in contacts], dtype=np.int32).reshape(-1, 2),
        contact_probabilities=np.array([contact["probability"] for contact in contacts]),
        vertex_mean_rc=np.array([[vertex["row"], vertex["col"]] for vertex in vertices]),
        vertex_rms_radius_px=np.array([vertex["rms_radius_px"] for vertex in vertices]),
        vertex_p_exist=np.array([vertex["p_exist"] for vertex in vertices]),
        seed_points=maps.seed_points,
    )
    with np.load(npz_path) as arrays:
        schema = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
    metadata = dict(
        status="complete; every generated hypothesis passed structural validation",
        demonstration=(
            "Synthetic membrane and nuclear images from the existing in-action example; "
            "finite candidate sensitivity demonstration, not empirical calibration"
        ),
        source_path=source.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        proposer="ClassicalProposer",
        proposer_params=proposer_params,
        base_decoder_params=base_params,
        hypothesis_generation_params=grid_params,
        ess_min=4.0,
        summary=posterior.summary(),
        hypotheses=reports,
        contacts=contacts,
        vertices=vertices,
        npz_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        npz_schema=schema,
        interpretation_notes=[
            "Hypotheses are sorted by increasing energy; weights and labels share that order.",
            "All labels derive from the same observed synthetic channels. "
            "No ground-truth labels enter reconstruction or energy scoring.",
            "Parameter perturbations are followed by up to two merge and two split moves; "
            "exact pixel duplicate label fields are removed.",
            "Temperature is chosen to reach ESS >= 4, not fitted against correctness.",
            "Contact identities are original seed-label pairs; vertex summaries are "
            "conditional weighted locations for incident cell sets.",
            "The original deterministic in-action reconstruction may differ from "
            "the finite-ensemble maximum-weight candidate.",
        ],
    )
    npz_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=json_default) + "\n")
    print(json.dumps(dict(all_valid=True, **posterior.summary()), indent=2))


if __name__ == "__main__":
    main()
