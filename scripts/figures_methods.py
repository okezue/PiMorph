#!/usr/bin/env python3
"""Rebuild exact PiMorph method figures and their auditable source arrays.

Run from the repository root: ``python scripts/figures_methods.py``.
No training checkpoint or downloaded microscopy is needed: the real examples
recompose archived image panels without changing their image pixels; the synthetic
example executes the public PiMorph modules with all parameters recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage as ndi

from pimorph.complex import extract_complex, validate
from pimorph.complex.geometry import smooth_complex
from pimorph.fields.profile import edge_profile
from pimorph.fields.strip import edge_strip
from pimorph.infer.decoder import ConstrainedDecoder, DecoderParams
from pimorph.infer.proposals import ClassicalProposer, robust_normalize
from pimorph.synth.render import RenderParams, render_channels
from pimorph.synth.tissue import SynthTissueParams, generate_tissue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DATA = ROOT / "docs" / "figure_data"
TEAL, INDIGO, AMBER = "#138B91", "#525A97", "#D99126"
INK, GRAY = "#263745", "#82919A"
PASTELS = ["#D3ECE8", "#DADDF0", "#E5EFE6", "#D0E1E8", "#EFE9D5", "#E1DCF0"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": "#BBC6CC",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def panel(ax, letter, title):
    ax.text(0, 1.055, letter, transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom", ha="left")
    ax.text(0.075, 1.06, title, transform=ax.transAxes, fontsize=10.2, va="bottom", ha="left")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=240, bbox_inches="tight", pad_inches=0.1)
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def json_out(name, value):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def quiet(ax, shape=None):
    ax.set_axis_off()
    if shape:
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")


def label_rgb(labels, background="#F1F3F4"):
    colors = np.array(
        [to_rgb(background)] + [to_rgb(PASTELS[(k * 5) % len(PASTELS)]) for k in range(int(labels.max()))]
    )
    return colors[labels]


def draw_complex(ax, cx, color=INK, lw=0.8, vertices=True, highlight=None):
    for e in range(cx.n_edges):
        p = cx.edge_geometry(e)
        ax.plot(
            p[:, 1],
            p[:, 0],
            color=AMBER if e == highlight else color,
            lw=2.7 if e == highlight else lw,
            solid_capstyle="round",
            zorder=3,
        )
    if vertices:
        ax.scatter(
            cx.vertex_xy[:, 1], cx.vertex_xy[:, 0], s=9, facecolor="white", edgecolor=color, linewidth=0.65, zorder=5
        )


def real_microscopy():
    """Lossless crops of the archived qualitative figures, with exact coordinates."""
    configs = [
        (
            "runs/pimorph_dev/sbiad1540_6dyn_neural_vs_classical.png",
            [(20, 22, 334, 336), (706, 358, 1020, 672), (706, 22, 1020, 336)],
            "S-BIAD1540 · EGM2_regular_6dyn-24",
            "v0 mixed",
            ("A", "B", "C"),
        ),
        (
            "runs/pimorph_dev/ve_strat_neural_vs_classical.png",
            [(17, 23, 340, 346), (710, 366, 1033, 689), (710, 23, 1033, 346)],
            "VE-strat · fluorescence crop",
            "v0 synthetic",
            ("D", "E", "F"),
        ),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(11.6, 8.1))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.025, hspace=0.24, wspace=0.055)
    provenance = []
    for row, (path, boxes, title, model, letters) in enumerate(configs):
        source = ROOT / path
        im = Image.open(source).convert("RGB")
        for j, box in enumerate(boxes):
            ax = axs[row, j]
            crop = np.array(im.crop(box))
            ax.imshow(crop, interpolation="none")
            quiet(ax, crop.shape[:2])
            panel(ax, letters[j], ["VE-cadherin", "Boundary evidence", "Decoded interfaces"][j])
        y = axs[row, 0].get_position().y1 + 0.065
        fig.text(0.047, y, title, fontsize=11.8, fontweight="bold")
        fig.text(0.995, y, f"{model} · qualitative example", fontsize=9.2, ha="right", color=GRAY)
        provenance.append(
            {
                "source": path,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_dimensions_xy": list(im.size),
                "crop_boxes_xyxy_exclusive": boxes,
                "display_order": ["raw_geometry", "neural_boundary", "neural_overlay"],
                "model": model,
                "interpretation": "Archived illustrative output; no instance ground truth.",
            }
        )
    save(fig, "methods_real_microscopy")
    json_out("methods_real_microscopy_provenance.json", provenance)


def in_action():
    tissue_params = SynthTissueParams(
        shape=(384, 384),
        n_cells=42,
        elongation=1.45,
        flow_angle_deg=-18,
        boundary_jitter_px=1.1,
        nucleus_jitter=0.1,
        seed=47,
    )
    rp = RenderParams(
        psf_sigma_px=1.05,
        junction_width_px=2.3,
        membrane_width_px=1.8,
        membrane_intensity=180.0,
        junction_intensity=240.0,
        broken_fraction=0.27,
        broken_segment_len_px=11.0,
        photobleach_gradient=0.2,
        nucleus_radius_px=7.0,
        seed=82,
    )
    tissue = generate_tissue(tissue_params)
    channels = render_channels(tissue, rp)
    prop = ClassicalProposer(
        auto_scale=False,
        nucleus_radius_px=3.5,
        seed_min_distance_px=13,
        cell_radius_px=30.0,
        ridge_sigmas=(0.8, 1.3, 2.0),
        use_tissue_mask=False,
    )
    maps = prop(channels["membrane"], channels["nuclei"])
    dp = DecoderParams(cell_radius_px=30.0, min_cell_area_px=150, fill_gaps=True)
    res = ConstrainedDecoder().decode(maps, dp)
    cx = res.cx
    report = validate(cx)
    assert report.ok and report.b1b2_zero and report.euler_residual == 0
    threshold = 38.0
    candidates = []
    for e in cx.cell_cell_edges():
        e = int(e)
        st = edge_strip(cx, e, channels["junction"], half_width_px=5.0, n_lateral=21, step_px=0.75)
        p = edge_profile(
            cx, e, channels["junction"], threshold=threshold, half_width_px=5.0, n_lateral=21, step_px=0.75, bin_px=1.5
        )
        if st.arclength_px > 45 and 0.1 < p.gap_fraction < 0.6:
            candidates.append((st.arclength_px * p.gap_fraction * (1 - p.gap_fraction), e, st, p))
    if not candidates:
        raise RuntimeError("No informative interrupted interface was generated")
    _, e, strip, profile = max(candidates, key=lambda t: t[0])
    rgb = np.zeros((384, 384, 3))
    m = robust_normalize(channels["membrane"], 1, 99.8)
    n = robust_normalize(channels["nuclei"], 1, 99.8)
    rgb += m[..., None] * np.array([0.35, 0.91, 0.88])
    rgb += n[..., None] * np.array([0.62, 0.50, 0.98])
    rgb = np.clip(rgb, 0, 1)
    fig = plt.figure(figsize=(12.7, 8.5))
    gs = fig.add_gridspec(
        2, 3, height_ratios=(1, 0.86), left=0.055, right=0.987, bottom=0.075, top=0.92, wspace=0.22, hspace=0.30
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.text(0.987, 0.975, "Reproducible synthetic example", ha="right", color=GRAY, fontsize=9.4)

    ax = axes[0]
    ax.imshow(rgb, interpolation="none")
    quiet(ax, (384, 384))
    panel(ax, "A", "Membrane + nuclei")
    ax.plot([23, 87], [354, 354], color="white", lw=2.5)
    ax.text(55, 344, "64 px", color="white", ha="center", fontsize=8)

    ax = axes[1]
    ax.imshow(maps.boundary, cmap="magma", vmin=0, vmax=1, interpolation="none")
    pts = maps.seed_points
    ax.scatter(pts[:, 1], pts[:, 0], s=9, facecolor="#69D7CE", edgecolor="white", linewidth=0.3)
    quiet(ax, (384, 384))
    panel(ax, "B", "Boundary evidence + seeds")

    ax = axes[2]
    ax.imshow(label_rgb(res.labels), interpolation="none")
    draw_complex(ax, cx, lw=0.75, highlight=e)
    p = strip.points
    center = 0.5 * (p.min(axis=0) + p.max(axis=0))
    side = int(np.ceil(np.ptp(p, axis=0).max() + 42))
    lo = np.floor(center - side / 2).astype(int)
    lo = np.maximum(0, np.minimum(lo, np.array(tissue_params.shape) - side))
    hi = lo + side
    ax.add_patch(
        Rectangle(
            (lo[1], lo[0]),
            hi[1] - lo[1],
            hi[0] - lo[0],
            fill=False,
            edgecolor=AMBER,
            linewidth=1,
            linestyle=(0, (3, 2)),
        )
    )
    quiet(ax, (384, 384))
    panel(ax, "C", "Faces, edges and vertices")
    ax.text(
        0.03,
        0.025,
        f"{len(cx.cell_faces)} cells",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
    )

    ax = axes[3]
    ax.imshow(
        channels["junction"], cmap="gray", vmin=0, vmax=np.percentile(channels["junction"], 99.6), interpolation="none"
    )
    for r in (-5, 5):
        pp = p + r * strip.normals
        ax.plot(pp[:, 1], pp[:, 0], color=AMBER, lw=0.9, alpha=0.9)
    ax.plot(p[:, 1], p[:, 0], color=AMBER, lw=0.5, alpha=0.7, linestyle=(0, (2, 3)))
    idx = np.linspace(3, len(p) - 4, 9).astype(int)
    for k in idx:
        pp = np.array([p[k] - 5 * strip.normals[k], p[k] + 5 * strip.normals[k]])
        ax.plot(pp[:, 1], pp[:, 0], color=AMBER, lw=0.65, alpha=0.75)
    k = len(p) // 4
    step = min(12, len(p) - k - 1)
    xy = p[k + step] + 8 * strip.normals[k + step]
    xy0 = p[k] + 8 * strip.normals[k]
    ax.annotate(
        "", xy=(xy[1], xy[0]), xytext=(xy0[1], xy0[0]), arrowprops={"arrowstyle": "->", "color": "white", "lw": 1}
    )
    ax.text(xy0[1] + 4, xy0[0] - 3, "s", color="white", fontstyle="italic")
    ax.set_xlim(lo[1], hi[1])
    ax.set_ylim(hi[0], lo[0])
    quiet(ax)
    panel(ax, "D", "Junction-marker sampling strip")

    ax = axes[4]
    im = ax.imshow(
        strip.values.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(0, strip.arclength_px, -5, 5),
        vmin=0,
        vmax=140,
        interpolation="nearest",
    )
    ax.axhline(0, color="white", lw=0.6, ls=(0, (3, 3)), alpha=0.8)
    ax.set_xlabel("Arclength s (px)")
    ax.set_ylabel("Lateral offset r (px)")
    ax.set_yticks([-5, 0, 5])
    panel(ax, "E", "Unwrapped marker intensity")
    # A compact horizontal bar above the image leaves all lateral values visible.
    cax = ax.inset_axes([0.62, 1.23, 0.36, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 140])
    cb.ax.tick_params(labelsize=7, pad=1, length=2)
    cb.outline.set_visible(False)

    ax = axes[5]
    s = 0.5 * (profile.bin_edges_px[:-1] + profile.bin_edges_px[1:])
    ax.plot(s, profile.max_intensity_s, color=TEAL, lw=1.8, label="Strip maximum")
    ax.plot(s, profile.mean_intensity_s, color=INDIGO, lw=1.1, label="Strip mean")
    ax.axhline(threshold, color=GRAY, ls=(0, (3, 3)), lw=0.8)
    off = profile.occupancy_s == 0
    for k in np.flatnonzero(off):
        ax.axvspan(profile.bin_edges_px[k], profile.bin_edges_px[k + 1], color=AMBER, alpha=0.2, lw=0, zorder=-1)
    ax.set_xlabel("Arclength s (px)")
    ax.set_ylabel("Simulated signal (photon units)")
    ax.set_xlim(0, strip.arclength_px)
    ax.set_ylim(0, max(140, np.nanmax(profile.max_intensity_s) * 1.14))
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.5)
    ax.text(
        0.99,
        0.02,
        "Amber: no sample above threshold",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#966213",
    )
    panel(ax, "F", "Signal along one physical contact")
    save(fig, "methods_in_action")
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA / "methods_in_action_arrays.npz",
        truth_labels=tissue.labels,
        membrane=channels["membrane"],
        nuclei=channels["nuclei"],
        junction=channels["junction"],
        broken_mask=channels["broken_mask"],
        boundary=maps.boundary,
        seed_points=maps.seed_points,
        predicted_labels=res.labels,
        strip_values=strip.values,
        strip_s=strip.s,
        strip_r=strip.r,
        strip_points=strip.points,
        strip_normals=strip.normals,
        profile_arclength=s,
        profile_max=profile.max_intensity_s,
        profile_mean=profile.mean_intensity_s,
        profile_occupancy=profile.occupancy_s,
    )
    json_out(
        "methods_in_action_provenance.json",
        {
            "kind": "New synthetic demonstration; not a benchmark or biological observation",
            "tissue_parameters": asdict(tissue_params),
            "render_parameters": asdict(rp),
            "classical_proposer_parameters": asdict(prop),
            "decoder_parameters": asdict(dp),
            "geometry_channel": "membrane",
            "measurement_channel": "junction",
            "ground_truth_cells": int(tissue.n_cells),
            "decoded_cells": len(cx.cell_faces),
            "selected_edge_id": e,
            "strip_half_width_px": 5,
            "strip_n_lateral": 21,
            "strip_step_px": 0.75,
            "profile_bin_px": 1.5,
            "threshold_photon_units": threshold,
            "selected_edge_gap_fraction": profile.gap_fraction,
            "selected_edge_continuity": profile.continuity,
            "valid_complex": bool(report.ok),
            "boundary_of_boundary_zero": bool(report.b1b2_zero),
            "euler_residual": int(report.euler_residual),
            "crop_rows_cols_in_panel_d": [lo.tolist(), hi.tolist()],
            "units": "Pixels only; no physical scale assigned to this synthetic tissue",
        },
    )


def topology():
    rr, cc = np.mgrid[:190, :210]
    centers = np.array(
        [
            [40, 42],
            [35, 92],
            [40, 149],
            [45, 189],
            [89, 38],
            [90, 99],
            [88, 151],
            [95, 191],
            [147, 40],
            [145, 95],
            [148, 147],
            [148, 187],
        ]
    )
    labels = np.argmin(((np.stack([rr, cc], -1)[:, :, None, :] - centers[None, None, :, :]) ** 2).sum(-1), axis=2) + 1
    tissue = ((rr - 94) / 83) ** 4 + ((cc - 104) / 96) ** 4 < 1
    labels[~tissue] = 0
    labels[((rr - 63) / 10) ** 2 + ((cc - 69) / 14) ** 2 < 1] = 0
    labels = labels.astype(np.int32)
    cx = extract_complex(labels)
    smooth_complex(cx)
    assert len(cx.gap_faces) == 1 and validate(cx).ok
    # Two connected cells, one enclosed gap, two physical contacts of the same pair.
    rr2, cc2 = np.mgrid[:170, :180]
    two = np.zeros((170, 180), np.int32)
    inside = ((rr2 - 84.5) / 73) ** 8 + ((cc2 - 89.5) / 76) ** 8 < 1
    two[inside & (cc2 < 90)] = 1
    two[inside & (cc2 >= 90)] = 2
    two[((rr2 - 85) / 17) ** 2 + ((cc2 - 91) / 23) ** 2 < 1] = 0
    ct = extract_complex(two)
    smooth_complex(ct)
    f1 = int(ct.cell_faces[0])
    f2 = int(ct.cell_faces[1])
    edges = ct.edges_between(f1, f2)
    assert len(edges) == 2 and validate(ct).ok
    # Four connected cell faces share one degree-four vertex; their dual is a cycle.
    rr3, cc3 = np.mgrid[:120, :120]
    quad = np.zeros((120, 120), np.int32)
    inside = (rr3 >= 10) & (rr3 < 110) & (cc3 >= 10) & (cc3 < 110)
    quad[inside] = 1 + (cc3[inside] >= 60) + 2 * (rr3[inside] >= 60)
    cq = extract_complex(quad)
    degree4 = [v for v in range(cq.n_vertices) if cq.vertex_degree(v) == 4]
    assert len(degree4) == 1 and validate(cq).ok

    fig = plt.figure(figsize=(13.2, 4.65))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.08, 1, 1.12], left=0.025, right=0.992, top=0.78, bottom=0.075, wspace=0.12
    )
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.text(0.992, 0.945, "Exact synthetic label geometry", ha="right", color=GRAY, fontsize=9.4)

    ax = axs[0]
    rgb = label_rgb(labels)
    gap = ndi.label(labels == 0)[0]
    glabel = gap[63, 69]
    rgb[gap == glabel] = to_rgb("#F4D9AC")
    ax.imshow(rgb, interpolation="none")
    draw_complex(ax, cx, lw=0.85)
    ax.text(70, 64, "gap", ha="center", va="center", fontsize=8.5, color="#805314")
    ax.text(176, 184, "outer", ha="center", fontsize=8.5, color=GRAY)
    quiet(ax, labels.shape)
    panel(ax, "A", "Cells, an enclosed gap and outer face")

    ax = axs[1]
    rgb = label_rgb(two)
    ccg = ndi.label(two == 0)[0]
    rgb[ccg == ccg[85, 91]] = to_rgb("#F4D9AC")
    ax.imshow(rgb, interpolation="none")
    draw_complex(ax, ct, lw=0.9)
    ax.text(39, 83, "A", ha="center", fontsize=15, color=TEAL, fontweight="bold")
    ax.text(137, 83, "B", ha="center", fontsize=15, color=INDIGO, fontweight="bold")
    for k, e in enumerate(sorted(edges, key=lambda e: ct.edge_geometry(int(e))[:, 0].mean())):
        p = ct.edge_geometry(int(e))
        ax.plot(p[:, 1], p[:, 0], color=TEAL, lw=2.1, zorder=4)
        mid = p.mean(0)
        ax.text(mid[1] + 10, mid[0] + 2, f"e{k + 1}", ha="left", fontsize=9, color=INK)
    # The arrows run on opposite sides of the same real upper contact.
    ax.annotate("", xy=(85, 30), xytext=(85, 55), arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.4})
    ax.annotate("", xy=(95, 55), xytext=(95, 30), arrowprops={"arrowstyle": "-|>", "color": INDIGO, "lw": 1.4})
    ax.text(91, 87, "gap", ha="center", va="center", fontsize=8.5, color="#805314")
    quiet(ax, two.shape)
    panel(ax, "B", "Two contacts; two directed half-edges each")

    ax = axs[2]
    ax.set_xlim(0, 265)
    ax.set_ylim(160, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.imshow(label_rgb(quad), extent=(0, 120, 120, 0), interpolation="none")
    draw_complex(ax, cq, lw=1, vertices=False)
    center = cq.vertex_xy[degree4[0]]
    ax.scatter([center[1]], [center[0]], s=65, facecolor=AMBER, edgecolor="white", linewidth=1, zorder=7)
    # Small cell names allow the spatial and graph objects to be compared directly.
    positions = {"A": (35, 35), "B": (85, 35), "C": (35, 85), "D": (85, 85)}
    for name, (x, y) in positions.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=11)
    graphpos = {"A": (172, 35), "B": (232, 35), "C": (172, 95), "D": (232, 95)}
    for a, b in [("A", "B"), ("B", "D"), ("D", "C"), ("C", "A")]:
        x0, y0 = graphpos[a]
        x1, y1 = graphpos[b]
        ax.plot([x0, x1], [y0, y1], color=GRAY, lw=1.7, zorder=1)
    for name, (x, y) in graphpos.items():
        ax.scatter([x], [y], s=310, facecolor="white", edgecolor=INDIGO, lw=1.3, zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, zorder=3)
    ax.text(60, 141, "Physical vertex", ha="center", fontsize=9)
    ax.text(202, 141, "Dual adjacency", ha="center", fontsize=9)
    panel(ax, "C", "Four-cell junction without a graph triangle")
    save(fig, "methods_topology")
    np.savez_compressed(
        DATA / "methods_topology_labels.npz", cells_gap_outer=labels, disconnected_contacts=two, four_cell_junction=quad
    )
    json_out(
        "methods_topology_provenance.json",
        {
            "kind": "Synthetic explanatory label arrays passed to actual extract_complex",
            "cells_gap_outer": {"cells": len(cx.cell_faces), "gaps": len(cx.gap_faces), "outer_faces": 1},
            "disconnected_contacts": {
                "cells": 2,
                "gaps": 1,
                "physical_contact_components": len(edges),
                "simple_graph_cell_pair_count": 1,
                "edge_ids": edges.tolist(),
            },
            "four_cell_junction": {"physical_vertex_degree": 4, "adjacency_cycle_length": 4, "adjacency_triangles": 0},
            "note": "Geometry and topology are generated with PiMorph; no microscopy data in this figure.",
        },
    )


if __name__ == "__main__":
    real_microscopy()
    in_action()
    topology()
    print("Wrote 3 method figures (PNG/SVG/PDF) and exact source arrays/provenance.")
