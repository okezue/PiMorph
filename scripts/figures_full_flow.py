#!/usr/bin/env python3
"""Compose the full PiMorph flow from executed arrays and explicit schematics.

Run from the repository root after figures_methods.py and
figures_full_flow_ensemble.py. No neural inference or biological measurements
are invented: the executed path is classical and synthetic; module glyphs are
conceptual. BioRender artwork is embedded with its separate attribution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

from pimorph.complex import extract_complex, validate
from pimorph.complex.geometry import smooth_complex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/figure_data"
OUT = ROOT / "docs/figures"
INK, GRAY, LIGHT = "#223744", "#71828c", "#dce6e9"
TEAL, PURPLE, AMBER, RED = "#168e91", "#7661a5", "#d89631", "#c67163"
PASTELS = ["#cbe8e3", "#dbdcf0", "#e5eddf", "#d5e5ee", "#f0e5cd", "#e9d9e8"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": LIGHT,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def label_rgb(labels):
    palette = np.array(
        [to_rgb("#f7f9fa")] + [to_rgb(PASTELS[(k * 5) % len(PASTELS)]) for k in range(int(labels.max()))]
    )
    return palette[labels]


def normalize(a):
    lo, hi = np.percentile(a, [1, 99.8])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def main():
    a = np.load(DATA / "methods_in_action_arrays.npz")
    ensemble = np.load(DATA / "methods_full_flow_ensemble.npz")
    topo = np.load(DATA / "methods_topology_labels.npz")
    labels = a["predicted_labels"]
    cx = extract_complex(labels)
    smooth_complex(cx)
    assert validate(cx).ok
    fig = plt.figure(figsize=(22, 17), facecolor="white")

    def text(x, y, s, size=11, color=INK, weight="normal", ha="left", **kw):
        return fig.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, **kw)

    def axis(rect, image=None, cmap=None, extent=None):
        ax = fig.add_axes(rect)
        if image is not None:
            ax.imshow(image, cmap=cmap, interpolation="nearest", extent=extent)
        ax.set_axis_off()
        return ax

    def heading(x, y, letter, title):
        text(x, y, letter, 16, TEAL, "bold")
        text(x + 0.017, y + 0.001, title, 12.5, weight="bold")

    def arrow(start, end, color=GRAY, dashed=False, bend=0, lw=1.4):
        p = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={bend}",
            zorder=30,
        )
        fig.add_artist(p)

    def complex_view(ax, c, labs=None, color=INK, width=0.65, vertices=True):
        if labs is not None:
            ax.imshow(label_rgb(labs), interpolation="nearest")
        for e in range(c.n_edges):
            p = c.edge_geometry(e)
            ax.plot(p[:, 1], p[:, 0], color=color, lw=width, solid_capstyle="round")
        if vertices:
            ax.scatter(
                c.vertex_xy[:, 1], c.vertex_xy[:, 0], s=7, color=AMBER, edgecolor="white", linewidth=0.25, zorder=5
            )
        ax.set_xlim(-1, c.shape[1])
        ax.set_ylim(c.shape[0], -1)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def geometric_patch(ax, vertical=True):
        # A valid planar T1 schematic: four distinct cell regions surround a
        # shrinking vertical contact or its newly created horizontal contact.
        if vertical:
            segments = [
                ((0.5, 0.35), (0.5, 0.65)),
                ((0.5, 0.35), (0.05, 0.02)),
                ((0.5, 0.35), (0.95, 0.02)),
                ((0.5, 0.65), (0.05, 0.98)),
                ((0.5, 0.65), (0.95, 0.98)),
            ]
            polygons = [
                [(0, 0), (0.5, 0.35), (0, 1)],
                [(1, 0), (1, 1), (0.5, 0.65), (0.5, 0.35)],
                [(0, 0), (1, 0), (0.5, 0.35)],
                [(0, 1), (0.5, 0.65), (1, 1)],
            ]
            # Explicit face fills trace the same endpoints as the interfaces.
            polygons = [
                [(0, 0), (0.05, 0.02), (0.5, 0.35), (0.5, 0.65), (0.05, 0.98), (0, 1)],
                [(1, 0), (1, 1), (0.95, 0.98), (0.5, 0.65), (0.5, 0.35), (0.95, 0.02)],
                [(0.05, 0.02), (0.95, 0.02), (0.5, 0.35)],
                [(0.05, 0.98), (0.5, 0.65), (0.95, 0.98)],
            ]
        else:
            segments = [
                ((0.35, 0.5), (0.65, 0.5)),
                ((0.35, 0.5), (0.02, 0.05)),
                ((0.35, 0.5), (0.02, 0.95)),
                ((0.65, 0.5), (0.98, 0.05)),
                ((0.65, 0.5), (0.98, 0.95)),
            ]
            polygons = [
                [(0.02, 0.05), (0.35, 0.5), (0.02, 0.95)],
                [(0.98, 0.05), (0.98, 0.95), (0.65, 0.5)],
                [(0, 0), (1, 0), (0.98, 0.05), (0.65, 0.5), (0.35, 0.5), (0.02, 0.05)],
                [(0, 1), (0.02, 0.95), (0.35, 0.5), (0.65, 0.5), (0.98, 0.95), (1, 1)],
            ]
        for p, col in zip(polygons, PASTELS):
            ax.add_patch(Polygon(p, facecolor=col, edgecolor="none"))
        for k, (u, v) in enumerate(segments):
            ax.plot(*zip(u, v), color=AMBER if k == 0 else INK, lw=2.2 if k == 0 else 1.4)
        xy = np.array([segments[0][0], segments[0][1]])
        ax.scatter(xy[:, 0], xy[:, 1], s=22, color=AMBER, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    text(0.025, 0.969, "PiMorph", 27, weight="bold")
    text(0.148, 0.972, "From microscopy to physical interfaces and biological measurements", 17)
    text(0.025, 0.934, "01  IMAGE EVIDENCE AND RECONSTRUCTION", 10.5, TEAL, "bold")
    text(0.975, 0.934, "Executed synthetic example + conceptual module views", 9.5, GRAY, ha="right")

    heading(0.025, 0.903, "A", "Acquire and assign channels")
    bio = Image.open(OUT / "biology_biorender.png").convert("RGB")
    bio_crop = (90, 330, 910, 1220)
    axis([0.062, 0.756, 0.134, 0.135], np.asarray(bio.crop(bio_crop)))
    arrow((0.128, 0.753), (0.128, 0.729), TEAL)
    for i, (key, name, col) in enumerate(
        [("membrane", "geometry", TEAL), ("nuclei", "nuclei", PURPLE), ("junction", "junction", AMBER)]
    ):
        x = 0.025 + i * 0.072
        rgb = normalize(a[key])[..., None] * np.array(to_rgb(col))
        axis([x, 0.635, 0.066, 0.086], rgb)
        text(x + 0.033, 0.619, name, 10, ha="center")
    text(0.128, 0.596, "registered roles · pixel-scale metadata", 9.5, GRAY, ha="center")

    heading(0.275, 0.903, "B", "Propose image evidence")
    # Classical ridge glyph from executed data.
    axis([0.278, 0.792, 0.078, 0.101], a["boundary"], "magma")
    text(0.317, 0.778, "classical filters", 9.7, ha="center")
    # Trained U-Net glyph: paired feature-map stacks, true U shape and skip paths.
    ax = axis([0.385, 0.797, 0.1, 0.094])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    levels = [
        (0.05, 0.7, 0.13, 0.28),
        (0.25, 0.42, 0.16, 0.22),
        (0.47, 0.14, 0.17, 0.16),
        (0.69, 0.42, 0.16, 0.22),
        (0.9, 0.7, 0.1, 0.28),
    ]
    for j, (x, y, w, h) in enumerate(levels):
        for shift in [0, 0.016, 0.032]:
            ax.add_patch(Rectangle((x + shift, y + shift), w, h, fc=PASTELS[j], ec=TEAL, lw=0.7))
    for j in range(4):
        u = levels[j]
        v = levels[j + 1]
        ax.annotate(
            "",
            (v[0], v[1] + v[3] / 2),
            (u[0] + u[2], u[1] + u[3] / 2),
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8},
        )
    ax.plot([0.18, 0.94], [0.98, 0.98], c=PURPLE, lw=0.8)
    ax.plot([0.4, 0.74], [0.7, 0.7], c=PURPLE, lw=0.8)
    ax.axis("off")
    text(0.436, 0.778, "trained U-Net", 9.7, ha="center")
    # Training supervision is a small actual label swatch linked to the model.
    axis([0.37, 0.861, 0.022, 0.029], label_rgb(a["truth_labels"][90:220, 90:220]))
    text(0.378, 0.891, "training", 8.5, GRAY, ha="center")
    arrow((0.392, 0.876), (0.405, 0.859), PURPLE, dashed=True, lw=0.9)
    text(0.436, 0.757, "6 heads", 8.5, GRAY, ha="center")
    # Actual shared-container evidence from the classical path; not neural output.
    ax = axis([0.306, 0.63, 0.145, 0.126], a["boundary"], "magma")
    ax.scatter(a["seed_points"][:, 1], a["seed_points"][:, 0], s=5, c="#80dfd4", edgecolor="white", linewidth=0.15)
    ax.set_xlim(0, 384)
    ax.set_ylim(384, 0)
    arrow((0.32, 0.766), (0.35, 0.755), TEAL, lw=1)
    arrow((0.43, 0.75), (0.415, 0.737), PURPLE, dashed=True, lw=1)
    text(0.381, 0.614, "boundary + seed evidence", 10, ha="center")
    text(0.381, 0.596, "shared proposal maps · six-head neural model", 9.2, GRAY, ha="center")

    heading(0.525, 0.903, "C", "Constrained reconstruction")
    ax = axis([0.523, 0.769, 0.087, 0.112], a["boundary"], "Greys")
    ax.scatter(a["seed_points"][:, 1], a["seed_points"][:, 0], s=6, c=TEAL)
    axis([0.633, 0.769, 0.087, 0.112], label_rgb(labels))
    text(0.566, 0.750, "seeds + flood mask", 9.5, ha="center")
    text(0.676, 0.750, "watershed", 9.5, ha="center")
    arrow((0.615, 0.824), (0.629, 0.824), TEAL)
    ax = axis([0.562, 0.628, 0.126, 0.105])
    complex_view(ax, cx, labels, width=0.45)
    text(0.625, 0.614, "cleanup → connected cell faces", 10, ha="center")
    text(0.625, 0.596, "external labels can enter here", 9.2, GRAY, ha="center")
    arrow((0.676, 0.742), (0.643, 0.728), TEAL, lw=1)

    heading(0.775, 0.903, "D", "Retain the physical complex")
    ax = axis([0.799, 0.745, 0.15, 0.14])
    complex_view(ax, cx, labels, width=0.75)
    text(0.874, 0.728, "faces · interfaces · vertices", 10, ha="center")
    tc = extract_complex(topo["cells_gap_outer"])
    ax = axis([0.775, 0.626, 0.094, 0.089])
    complex_view(ax, tc, topo["cells_gap_outer"], width=0.8)
    text(0.822, 0.611, "gap + exterior", 9.5, ha="center")
    ax = axis([0.89, 0.636, 0.076, 0.071])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    t = np.linspace(0, 1, 100)
    y = 0.5 + 0.11 * np.sin(t * 2 * np.pi)
    ax.fill_between(t, 0, y, color=PASTELS[0])
    ax.fill_between(t, y, 1, color=PASTELS[1])
    ax.plot(t, y, color=INK, lw=1.5)
    ax.annotate("", (0.8, 0.64), (0.2, 0.64), arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.6})
    ax.annotate("", (0.2, 0.36), (0.8, 0.36), arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.6})
    ax.axis("off")
    text(0.928, 0.611, "twin half-edges", 9.5, ha="center")
    text(0.873, 0.590, r"closed loops   ·   $B_1B_2=0$   ·   Euler identity", 10, GRAY, ha="center")

    for x1, x2 in [(0.238, 0.268), (0.486, 0.516), (0.725, 0.764)]:
        arrow((x1, 0.812), (x2, 0.812), TEAL, lw=1.7)

    fig.add_artist(plt.Line2D([0.025, 0.975], [0.574, 0.574], transform=fig.transFigure, color=LIGHT, lw=1))
    heading(0.025, 0.548, "E", "Optional candidate ensemble")
    heading(0.370, 0.548, "F", "Measure along interfaces")
    heading(0.775, 0.548, "G", "Export and evaluate")
    # A dashed connection makes the finite ensemble an optional decoder branch.
    fig.add_artist(
        plt.Line2D(
            [0.625, 0.625, 0.329], [0.593, 0.582, 0.582], transform=fig.transFigure, color=PURPLE, lw=1.1, ls="--"
        )
    )
    arrow((0.329, 0.582), (0.329, 0.512), PURPLE, dashed=True, lw=1.1)
    fig.add_artist(
        plt.Line2D([0.873, 0.748, 0.748], [0.582, 0.582, 0.507], transform=fig.transFigure, color=TEAL, lw=1.1)
    )
    arrow((0.748, 0.507), (0.727, 0.507), TEAL, lw=1.1)
    # Executed, energy-ranked candidates, with no calibrated-confidence claim.
    for j, idx in enumerate([0, 2, len(ensemble["weights"]) - 1]):
        labs = ensemble["labels"][idx]
        axis([0.025 + j * 0.099, 0.422, 0.088, 0.114], label_rgb(labs))
        text(0.069 + j * 0.099, 0.411, f"{np.count_nonzero(np.unique(labs) > 0)} cells", 9.3, ha="center")
    ax = fig.add_axes([0.039, 0.333, 0.123, 0.058])
    ax.bar(np.arange(len(ensemble["weights"])), ensemble["weights"], color=PURPLE, width=0.8)
    ax.set_ylabel("weight", fontsize=9)
    ax.set_xlabel("energy-ranked candidate", fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=8, length=2)
    ax.set_xticks([0, 5, 11], [1, 6, 12])
    ax.set_ylim(0, 0.36)
    text(0.039, 0.399, "perturb · merge · split", 9.3, GRAY)
    # Weighted contact evidence on the same seed-based cell identities.
    ax = axis([0.19, 0.33, 0.122, 0.079])
    probs = {tuple(pair): p for pair, p in zip(ensemble["contact_cells"], ensemble["contact_probabilities"])}
    mapcx = extract_complex(ensemble["labels"][0])
    smooth_complex(mapcx)
    for e in mapcx.cell_cell_edges():
        faces = mapcx.edge_faces[e]
        key = tuple(sorted(mapcx.face_label[faces]))
        p = probs.get(key, 0)
        geom = mapcx.edge_geometry(int(e))
        ax.plot(geom[:, 1], geom[:, 0], color=plt.cm.viridis(p), lw=0.8)
    ax.set_xlim(0, 384)
    ax.set_ylim(384, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    text(0.251, 0.314, "weighted contacts", 9.5, ha="center")
    cbax = fig.add_axes([0.317, 0.339, 0.004, 0.052])
    cbax.imshow(np.linspace(0, 1, 100)[:, None], cmap="viridis", aspect="auto", origin="lower")
    cbax.set_axis_off()
    text(0.326, 0.337, "0", 7, GRAY)
    text(0.326, 0.385, "1", 7, GRAY)
    text(0.025, 0.294, "finite sensitivity · ESS = 4 · uncalibrated", 9.3, GRAY)

    # Same actual interface is measured in the curve, unwrapped strip, and trace.
    p = a["strip_points"]
    n = a["strip_normals"]
    ax = axis([0.368, 0.344, 0.151, 0.185], a["junction"], "gray")
    ax.plot(p[:, 1], p[:, 0], c=AMBER, lw=1.4)
    ax.fill(
        np.r_[p[:, 1] + 5 * n[:, 1], (p[:, 1] - 5 * n[:, 1])[::-1]],
        np.r_[p[:, 0] + 5 * n[:, 0], (p[:, 0] - 5 * n[:, 0])[::-1]],
        color=TEAL,
        alpha=0.22,
    )
    for j in range(0, len(p), 12):
        ends = np.array([p[j] - 5 * n[j], p[j] + 5 * n[j]])
        ax.plot(ends[:, 1], ends[:, 0], c="#6cdcd1", lw=0.8)
    ax.set_xlim(p[:, 1].min() - 15, p[:, 1].max() + 15)
    ax.set_ylim(p[:, 0].max() + 18, p[:, 0].min() - 18)
    text(0.443, 0.328, "oriented sampling strip", 10, ha="center")
    ax = fig.add_axes([0.548, 0.451, 0.174, 0.067])
    ax.imshow(
        a["strip_values"].T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=140,
        extent=[a["strip_s"][0], a["strip_s"][-1], a["strip_r"][0], a["strip_r"][-1]],
    )
    ax.set_ylabel("normal r (px)", fontsize=8.5)
    ax.tick_params(labelsize=8, length=2)
    ax.set_xticks([])
    text(0.635, 0.525, "unwrapped molecular field", 9.5, ha="center")
    ax = fig.add_axes([0.548, 0.344, 0.174, 0.075])
    ax.plot(a["profile_arclength"], a["profile_max"], c=AMBER, lw=1.3, label="maximum")
    ax.plot(a["profile_arclength"], a["profile_mean"], c=TEAL, lw=1.3, label="mean")
    ax.axhline(38, c=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("arclength s (px)", fontsize=9, labelpad=2)
    ax.set_ylabel("signal", fontsize=9)
    ax.tick_params(labelsize=8, length=2)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    arrow((0.516, 0.472), (0.537, 0.472), TEAL)
    text(0.371, 0.294, "occupancy · continuity · width · sidedness", 9.3, GRAY)

    # Exports are depicted by their actual geometry and a small numerical excerpt.
    rgb = normalize(a["membrane"])[..., None] * np.array([0.30, 0.80, 0.75])
    ax = axis([0.775, 0.419, 0.097, 0.115], rgb)
    for e in cx.cell_cell_edges()[::2]:
        q = cx.edge_geometry(int(e))
        ax.plot(q[:, 1], q[:, 0], c="white", lw=0.32)
    text(0.823, 0.408, "labels + overlays", 9.5, ha="center")
    ax = axis([0.884, 0.432, 0.092, 0.091])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    table = [["cell", "area", "sides"]]
    for f in cx.cell_faces[:4]:
        table.append(
            [str(int(cx.face_label[f])), str(int((labels == cx.face_label[f]).sum())), str(cx.face_sides(int(f)))]
        )
    for row, values in enumerate(table):
        yy = 0.93 - row * 0.205
        if row == 0:
            ax.plot([0, 1], [yy - 0.095, yy - 0.095], color=TEAL, lw=1)
        for col, value in enumerate(values):
            ax.text(0.01 + col * 0.35, yy, value, fontsize=8.6, weight="bold" if row == 0 else "normal", va="center")
    ax.axis("off")
    text(0.929, 0.408, "structural tables", 9.5, ha="center")
    # Benchmark branch: source values are archived hCEC means, not this demo's scores.
    ax = fig.add_axes([0.807, 0.335, 0.151, 0.054])
    scores = np.array([[0.635, 0.680], [0.830, 0.867], [0.790, 0.816]])
    for i in range(3):
        ax.plot(scores[i], [i, i], color=LIGHT, lw=2)
        ax.scatter(scores[i], [i, i], s=20, c=[GRAY, TEAL], zorder=3)
    ax.set_yticks([0, 1, 2], ["vertex F1", "adjacency F1", "PQ"], fontsize=8)
    ax.set_xlim(0.6, 0.92)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.tick_params(labelsize=8, length=2)
    ax.invert_yaxis()
    text(0.873, 0.397, "held-out hCEC means", 9.1, GRAY, ha="center")
    text(0.817, 0.317, "● Cellpose", 8, GRAY)
    text(0.898, 0.317, "● PiMorph v6", 8, TEAL)
    text(0.775, 0.294, "provenance + reports · separate benchmark", 9.3, GRAY)
    arrow((0.73, 0.467), (0.765, 0.467), TEAL)

    fig.add_artist(plt.Line2D([0.025, 0.975], [0.276, 0.276], transform=fig.transFigure, color=LIGHT, lw=1))
    text(0.025, 0.260, "02  SEPARATE ANALYSIS MODULES", 10.5, TEAL, "bold")
    text(0.975, 0.260, "Additional channels, time series or volumes enter the relevant module", 9.4, GRAY, ha="right")
    fig.add_artist(
        plt.Line2D([0.982, 0.982, 0.132], [0.41, 0.282, 0.282], transform=fig.transFigure, color=GRAY, lw=0.8)
    )
    for branch_x in [0.132, 0.38, 0.63]:
        arrow((branch_x, 0.283), (branch_x, 0.272), GRAY, lw=0.8)
    heading(0.025, 0.232, "H", "Phenotypes and networks")
    heading(0.275, 0.232, "I", "Dynamics")
    heading(0.525, 0.232, "J", "Mechanics and transport")
    heading(0.775, 0.232, "K", "Volumetric extension")

    # H: real dual network from this executed reconstruction and conceptual markers.
    cent = {int(f): np.argwhere(labels == cx.face_label[f]).mean(axis=0) for f in cx.cell_faces}
    ax = axis([0.025, 0.081, 0.12, 0.139])
    for e in cx.cell_cell_edges():
        f, g = map(int, cx.edge_faces[e])
        r, s = cent[f], cent[g]
        ax.plot([r[1], s[1]], [r[0], s[0]], color=TEAL, lw=0.75)
    pts = np.array(list(cent.values()))
    ax.scatter(pts[:, 1], pts[:, 0], s=12, c=PURPLE, edgecolor="white", linewidth=0.35)
    ax.set_xlim(0, 384)
    ax.set_ylim(384, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    text(0.085, 0.068, "physical contact graph", 9.4, ha="center")
    ax = axis([0.158, 0.098, 0.084, 0.1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for j, col in enumerate([TEAL, PURPLE, AMBER]):
        for left, right in [[0.04, 0.24], [0.32 + j * 0.07, 0.52 + j * 0.07], [0.8, 0.96]]:
            ax.plot([left, right], [0.8 - j * 0.3] * 2, c=col, lw=6, solid_capstyle="round")
        ax.plot([0.02, 0.98], [0.8 - j * 0.3] * 2, c=col, lw=0.7, alpha=0.4)
    ax.axis("off")
    text(0.2, 0.205, "aligned markers", 9.4, ha="center")
    text(0.2, 0.068, "co-occupancy", 9.4, ha="center")
    text(0.025, 0.043, "geometry · joint morphology · null models", 9.3, GRAY)

    # I: event geometry, explicitly schematic, with identity colors preserved.
    geometric_patch(axis([0.278, 0.107, 0.081, 0.102]), True)
    geometric_patch(axis([0.403, 0.107, 0.081, 0.102]), False)
    arrow((0.363, 0.16), (0.396, 0.16), TEAL)
    text(0.319, 0.215, "t", 10, ha="center")
    text(0.444, 0.215, "t + 1", 10, ha="center")
    text(0.382, 0.09, "T1 contact exchange", 10, ha="center")
    text(0.275, 0.043, "tracking · division · extrusion · gap events", 9.3, GRAY)

    # J: equilibrium vectors at a junction, then a conductance-network glyph.
    ax = axis([0.533, 0.101, 0.095, 0.108])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    angles = np.deg2rad([15, 135, 255])
    points = np.c_[np.cos(angles), np.sin(angles)]
    for j, q in enumerate(points):
        ax.plot([0, q[0]], [0, q[1]], c=PASTELS[j], lw=12, solid_capstyle="round")
        ax.arrow(
            0,
            0,
            0.72 * q[0],
            0.72 * q[1],
            width=0.015,
            head_width=0.13,
            color=[TEAL, PURPLE, AMBER][j],
            length_includes_head=True,
            zorder=5,
        )
    ax.add_patch(Circle((0, 0), 0.05, color=INK))
    ax.set_aspect("equal")
    ax.axis("off")
    text(0.58, 0.215, "force balance", 9.4, ha="center")
    text(0.58, 0.084, "relative T, P", 9.4, ha="center")
    ax = axis([0.648, 0.101, 0.094, 0.108])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = np.array(
        [[0.1, 0.15], [0.1, 0.6], [0.4, 0.3], [0.4, 0.82], [0.68, 0.13], [0.68, 0.62], [0.93, 0.4], [0.93, 0.87]]
    )
    for i, j in [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (5, 7),
        (6, 7),
    ]:
        ax.plot(
            nodes[[i, j], 0],
            nodes[[i, j], 1],
            c=plt.cm.viridis((nodes[i, 0] + nodes[j, 0]) / 2),
            lw=1.1 + (i % 3) * 0.7,
        )
    ax.scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 0], cmap="viridis", s=25, zorder=5)
    ax.axis("off")
    text(0.695, 0.215, "conductance network", 9.1, ha="center")
    text(0.695, 0.084, "functional proxy", 9.4, ha="center")
    text(0.525, 0.043, "model-based readouts · external validation needed", 9.3, GRAY)

    # K: 3-D is entered through a separate volume, not an extrusion of the 2-D graph.
    ax = fig.add_axes([0.778, 0.071, 0.09, 0.139], projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for j, (xx, yy) in enumerate([(0, 0), (1.05, 0), (0, 1.05)]):
        z = 0.9 + (0.2 if j == 1 else 0)
        v = np.array(
            [
                [xx, yy, 0],
                [xx + 1, yy, 0],
                [xx + 1, yy + 1, 0],
                [xx, yy + 1, 0],
                [xx, yy, z],
                [xx + 1, yy, z],
                [xx + 1, yy + 1, z],
                [xx, yy + 1, z],
            ]
        )
        faces = [[v[k] for k in ids] for ids in [[0, 1, 5, 4], [1, 2, 6, 5], [4, 5, 6, 7], [2, 3, 7, 6]]]
        ax.add_collection3d(
            Poly3DCollection(faces, facecolors=PASTELS[j], edgecolors=TEAL, linewidths=0.65, alpha=0.85)
        )
    ax.set(xlim=(0, 2.1), ylim=(0, 2.1), zlim=(0, 1.3))
    ax.view_init(26, -52)
    ax.set_axis_off()
    ax = fig.add_axes([0.885, 0.071, 0.09, 0.139], projection="3d")
    for i in range(6):
        for j in range(4):
            th = np.linspace(-1.15 + i * 0.38, -1.15 + (i + 1) * 0.38, 9)
            z0 = j * 0.28
            top = np.c_[np.cos(th), np.sin(th), np.full(len(th), z0)]
            bot = np.c_[np.cos(th[::-1]), np.sin(th[::-1]), np.full(len(th), z0 + 0.28)]
            ax.add_collection3d(
                Poly3DCollection([np.r_[top, bot]], facecolors=PASTELS[(i + j) % 6], edgecolors=TEAL, linewidths=0.65)
            )
    ax.set(xlim=(0, 1.2), ylim=(-1, 1), zlim=(0, 1.2))
    ax.view_init(22, -52)
    ax.set_axis_off()
    text(0.825, 0.215, "volume labels", 9.4, ha="center")
    text(0.93, 0.215, "curved monolayer", 9.4, ha="center")
    text(0.775, 0.043, "interfaces · triple lines · junction points", 9.3, GRAY)
    text(
        0.975,
        0.018,
        "BioRender tissue artwork; all remaining geometry, charts and typography are reproducible source layers.",
        8.6,
        GRAY,
        ha="right",
    )

    OUT.mkdir(exist_ok=True, parents=True)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"methods_full_flow.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    svg = OUT / "methods_full_flow.svg"
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n")
    sources = [
        DATA / "methods_in_action_arrays.npz",
        DATA / "methods_full_flow_ensemble.npz",
        DATA / "methods_topology_labels.npz",
        OUT / "biology_biorender.png",
    ]
    provenance = {
        "script": "scripts/figures_full_flow.py",
        "figure": "methods_full_flow",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "biorender_crop_xyxy": list(bio_crop),
        "executed_path": "synthetic, classical proposer",
        "conceptual_panels": [
            "A tissue",
            "B neural architecture and training link",
            "D half-edge glyph",
            "H marker strips",
            "I T1 event",
            "J force and conductance glyphs",
            "K volume and curved surface",
        ],
        "ensemble": "12 actual finite hypotheses, uncalibrated; see methods_full_flow_ensemble.json",
        "benchmark_inset": {
            "dataset": "hCEC",
            "units": "5 held-out fields, not cultures",
            "rows": ["vertex_f1", "adjacency_f1", "pq"],
            "columns": ["Cellpose-SAM", "PiMorph tuned v6"],
            "values": scores.tolist(),
            "source": "EXPLAINER.md archived benchmark means",
        },
        "validation": "all synthetic structures validate; topology validity is not biological accuracy",
        "extensions": "separate APIs/scripts, not automatic reconstruct outputs; 3D has separate volume input",
    }
    (DATA / "methods_full_flow_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Created methods_full_flow PNG, SVG and PDF")


if __name__ == "__main__":
    main()
