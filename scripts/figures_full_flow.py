#!/usr/bin/env python3
"""Compose the full PiMorph flow from executed arrays and explicit schematics.

Run from the repository root after figures_methods.py and
figures_full_flow_ensemble.py. No neural inference or biological measurements
are invented: the executed path is classical and synthetic; module glyphs are
conceptual. BioRender artwork is embedded with its separate attribution.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
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
    fig = plt.figure(figsize=(22, 26), facecolor="white")

    # Tiered layout: local panel artwork is preserved; only its placement
    # changes. Equal physical x/y scaling keeps glyphs and microscopy undistorted.
    old_boxes = {
        "A": (0.025, 0.638, 0.251, 0.974),
        "B": (0.271, 0.64, 0.514, 0.974),
        "C": (0.54, 0.705, 0.7, 0.974),
        "D": (0.744, 0.712, 0.983, 0.974),
        "E": (0.023, 0.30, 0.359, 0.604),
        "F": (0.39, 0.312, 0.802, 0.604),
        "G": (0.819, 0.30, 0.989, 0.604),
        "H": (0.024, 0.04, 0.253, 0.28),
        "I": (0.279, 0.07, 0.503, 0.28),
        "J": (0.535, 0.065, 0.794, 0.28),
        "K": (0.821, 0.06, 0.993, 0.28),
    }
    # left, bottom, width; height follows the original physical aspect ratio.
    placements = {
        "A": (0.10, 0.762, 0.23),
        "B": (0.53, 0.754, 0.245),
        "C": (0.6522, 0.576, 0.13),
        "D": (0.20, 0.5831, 0.195),
        "E": (0.715, 0.392, 0.25),
        "F": (0.23874, 0.36, 0.40),
        "G": (0.30, 0.22, 0.17),
        "H": (0.036, 0.025, 0.22),
        "I": (0.303, 0.025, 0.205),
        "J": (0.549, 0.025, 0.235),
        "K": (0.824, 0.025, 0.158),
    }
    active = {"name": "A"}
    panel_axes = {name: [] for name in old_boxes}

    def panel_transform():
        x0, y0, x1, _ = old_boxes[active["name"]]
        left, bottom, width = placements[active["name"]]
        sx = width / (x1 - x0)
        sy = sx * 17 / 26
        return Affine2D().translate(-x0, -y0).scale(sx, sy).translate(left, bottom)

    def point(x, y):
        return panel_transform().transform((x, y))

    def mapped_axes(rect, **kwargs):
        x, y, w, h = rect
        lower = point(x, y)
        upper = point(x + w, y + h)
        ax = fig.add_axes([*lower, *(upper - lower)], **kwargs)
        panel_axes[active["name"]].append(ax)
        return ax

    def contained(points):
        x0, y0, x1, y1 = old_boxes[active["name"]]
        return all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in points)

    def text(x, y, s, size=11, color=INK, weight="normal", ha="left", **kw):
        return fig.text(*point(x, y), s, fontsize=size, color=color, weight=weight, ha=ha, **kw)

    def axis(rect, image=None, cmap=None, extent=None):
        ax = mapped_axes(rect)
        if image is not None:
            ax.imshow(image, cmap=cmap, interpolation="nearest", extent=extent)
        ax.set_axis_off()
        return ax

    def arrow(start, end, color=GRAY, dashed=False, bend=0, lw=1.4):
        if not contained([start, end]):
            return
        start, end = point(*start), point(*end)
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

    # All narrative titles and subtitles live in the external caption. The
    # figure retains panel letters, channel identities and quantitative axes.
    def letter(x, y, name):
        active["name"] = name
        text(x, y, name, 17, INK, "bold")

    def routed(points, color=TEAL, dashed=False):
        if not contained(points):
            return
        # Each bend lies in a reserved inter-panel gutter; the final segment
        # alone gets an arrowhead. No connector runs over a plot or caption.
        if len(points) > 2:
            fig.add_artist(
                plt.Line2D(
                    *zip(*points[:-1]),
                    transform=panel_transform() + fig.transFigure,
                    color=color,
                    linewidth=1.35,
                    linestyle="--" if dashed else "-",
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )
            )
        arrow(points[-2], points[-1], color, dashed=dashed, lw=1.35)

    # A: a complete biological sheet and three registered channel planes.
    letter(0.028, 0.966, "A")
    bio = Image.open(OUT / "biology_biorender.png").convert("RGB")
    bio_crop = (90, 330, 910, 1220)
    axis([0.027, 0.787, 0.102, 0.151], np.asarray(bio.crop(bio_crop)))
    for i, (key, name, col) in enumerate(
        [("membrane", "geometry", TEAL), ("nuclei", "nuclei", PURPLE), ("junction", "junction", AMBER)]
    ):
        yy = 0.857 - i * 0.099
        rgb = normalize(a[key])[..., None] * np.array(to_rgb(col))
        axis([0.16, yy, 0.065, 0.084], rgb)
        text(0.1925, yy - 0.013, name, 9, ha="center")
    arrow((0.135, 0.85), (0.151, 0.85), TEAL)
    # The channel stack has a common output rail in its own narrow gutter.
    fig.add_artist(
        plt.Line2D(
            [0.232, 0.242, 0.242, 0.232],
            [0.925, 0.925, 0.683, 0.683],
            transform=panel_transform() + fig.transFigure,
            color=GRAY,
            lw=1,
        )
    )

    # B: two alternative routes, a shared actual evidence-map output.
    letter(0.28, 0.966, "B")
    axis([0.278, 0.849, 0.066, 0.086], a["boundary"], "magma")
    text(0.311, 0.836, "classical", 9, ha="center")
    ax = axis([0.276, 0.723, 0.083, 0.088])
    ax.set_xlim(-0.09, 1.12)
    ax.set_ylim(-0.08, 1.12)
    levels = [
        (0.05, 0.7, 0.13, 0.28),
        (0.25, 0.42, 0.16, 0.22),
        (0.47, 0.14, 0.17, 0.16),
        (0.69, 0.42, 0.16, 0.22),
        (0.9, 0.7, 0.1, 0.28),
    ]
    for j, (x, y, w, h) in enumerate(levels):
        for shift in [0, 0.016, 0.032]:
            ax.add_patch(Rectangle((x + shift, y + shift), w, h, fc=PASTELS[j], ec=TEAL, lw=0.75))
    for j in range(4):
        u, v = levels[j], levels[j + 1]
        ax.annotate(
            "",
            (v[0], v[1] + v[3] / 2),
            (u[0] + u[2], u[1] + u[3] / 2),
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.85},
        )
    ax.plot([0.19, 0.9], [1.03, 1.03], c=PURPLE, lw=0.9)
    ax.plot([0.43, 0.68], [0.72, 0.72], c=PURPLE, lw=0.9)
    text(0.355, 0.747, "neural", 9, ha="left")
    axis([0.297, 0.652, 0.033, 0.043], label_rgb(a["truth_labels"][90:220, 90:220]))
    arrow((0.314, 0.699), (0.314, 0.718), PURPLE, dashed=True, lw=1)
    # Incoming roles fan into alternatives, without crossing labels.
    routed([(0.271, 0.835), (0.271, 0.894), (0.274, 0.894)], GRAY)
    routed([(0.271, 0.835), (0.271, 0.767), (0.273, 0.767)], GRAY)
    ax = axis([0.397, 0.79, 0.104, 0.135], a["boundary"], "magma")
    ax.scatter(a["seed_points"][:, 1], a["seed_points"][:, 0], s=6, c="#80dfd4", edgecolor="white", linewidth=0.2)
    ax.set_xlim(0, 384)
    ax.set_ylim(384, 0)
    routed([(0.352, 0.894), (0.372, 0.894), (0.389, 0.875)], TEAL)
    routed([(0.362, 0.764), (0.376, 0.764), (0.389, 0.817)], PURPLE, dashed=True)

    # C: the actual seeded watershed reconstruction. The training and mask
    # semantics are explained in the caption rather than repeated as prose.
    letter(0.547, 0.966, "C")
    axis([0.547, 0.787, 0.127, 0.164], label_rgb(labels))
    # A short magnification illustrates connected faces after cleanup.
    ax = axis([0.633, 0.716, 0.055, 0.057])
    complex_view(ax, cx, labels, width=0.45)
    ax.set_xlim(95, 255)
    ax.set_ylim(245, 85)

    # D: the full physical complex is the main output, with topology insets.
    letter(0.748, 0.966, "D")
    ax = axis([0.835, 0.772, 0.139, 0.18])
    complex_view(ax, cx, labels, width=0.85)
    tc = extract_complex(topo["cells_gap_outer"])
    ax = axis([0.75, 0.842, 0.07, 0.095])
    complex_view(ax, tc, topo["cells_gap_outer"], width=0.85)
    ax = axis([0.75, 0.723, 0.07, 0.075])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    t = np.linspace(0, 1, 100)
    y = 0.5 + 0.11 * np.sin(t * 2 * np.pi)
    ax.fill_between(t, 0, y, color=PASTELS[0])
    ax.fill_between(t, y, 1, color=PASTELS[1])
    ax.plot(t, y, color=INK, lw=1.6)
    ax.annotate("", (0.85, 0.73), (0.15, 0.73), arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.6})
    ax.annotate("", (0.15, 0.27), (0.85, 0.27), arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.6})

    # E: actual alternatives and their normalized weights; no title bars.
    letter(0.028, 0.595, "E")
    for j, idx in enumerate([0, 2, len(ensemble["weights"]) - 1]):
        labs = ensemble["labels"][idx]
        xx = 0.029 + j * 0.109
        axis([xx, 0.447, 0.096, 0.124], label_rgb(labs))
        text(xx + 0.048, 0.434, f"{np.count_nonzero(np.unique(labs) > 0)} cells", 9, ha="center")
    ax = mapped_axes([0.041, 0.333, 0.135, 0.068])
    ax.bar(np.arange(len(ensemble["weights"])), ensemble["weights"], color=PURPLE, width=0.8)
    ax.set_ylabel("weight", fontsize=9)
    ax.set_xlabel("candidate", fontsize=9, labelpad=3)
    ax.tick_params(labelsize=8, length=2)
    ax.set_xticks([0, 5, 11], [1, 6, 12])
    ax.set_ylim(0, 0.36)
    ax = axis([0.215, 0.331, 0.107, 0.09])
    probs = {tuple(pair): p for pair, p in zip(ensemble["contact_cells"], ensemble["contact_probabilities"])}
    mapcx = extract_complex(ensemble["labels"][0])
    smooth_complex(mapcx)
    for e in mapcx.cell_cell_edges():
        key = tuple(sorted(mapcx.face_label[mapcx.edge_faces[e]]))
        q = mapcx.edge_geometry(int(e))
        ax.plot(q[:, 1], q[:, 0], c=plt.cm.viridis(probs.get(key, 0)), lw=0.9)
    ax.set_xlim(-2, 386)
    ax.set_ylim(386, -2)
    ax.set_aspect("equal")
    ax.axis("off")
    cbax = mapped_axes([0.337, 0.343, 0.004, 0.056])
    cbax.imshow(np.linspace(0, 1, 100)[:, None], cmap="viridis", aspect="auto", origin="lower")
    cbax.set_axis_off()
    text(0.345, 0.341, "0", 8, GRAY)
    text(0.345, 0.394, "1", 8, GRAY)

    # F: the route arrives at the sampling strip, then unfolds toward the
    # heatmap. The intensity trace is aligned below with ample axis margins.
    letter(0.399, 0.595, "F")
    p, n = a["strip_points"], a["strip_normals"]
    ax = axis([0.399, 0.465, 0.172, 0.109], a["junction"], "gray")
    ax.plot(p[:, 1], p[:, 0], c=AMBER, lw=1.5)
    ax.fill(
        np.r_[p[:, 1] + 5 * n[:, 1], (p[:, 1] - 5 * n[:, 1])[::-1]],
        np.r_[p[:, 0] + 5 * n[:, 0], (p[:, 0] - 5 * n[:, 0])[::-1]],
        color=TEAL,
        alpha=0.24,
    )
    for j in range(0, len(p), 12):
        ends = np.array([p[j] - 5 * n[j], p[j] + 5 * n[j]])
        ax.plot(ends[:, 1], ends[:, 0], c="#6cdcd1", lw=0.9)
    ax.set_xlim(p[:, 1].min() - 15, p[:, 1].max() + 15)
    ax.set_ylim(p[:, 0].max() + 18, p[:, 0].min() - 18)
    ax = mapped_axes([0.623, 0.502, 0.155, 0.067])
    ax.imshow(
        a["strip_values"].T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=140,
        extent=[a["strip_s"][0], a["strip_s"][-1], a["strip_r"][0], a["strip_r"][-1]],
    )
    ax.set_yticks([-5, 0, 5])
    ax.tick_params(labelsize=8, length=2)
    ax.set_xticks([])
    text(0.622, 0.58, "r (px)", 9)
    arrow((0.578, 0.535), (0.604, 0.535), TEAL)
    ax = mapped_axes([0.623, 0.342, 0.155, 0.102])
    ax.plot(a["profile_arclength"], a["profile_max"], c=AMBER, lw=1.4, label="maximum")
    ax.plot(a["profile_arclength"], a["profile_mean"], c=TEAL, lw=1.4, label="mean")
    ax.axhline(38, c=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("s (px)", fontsize=9, labelpad=3)
    ax.set_ylabel("signal", fontsize=9, labelpad=3)
    ax.tick_params(labelsize=8, length=2)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)

    # G: numerical exports and the separate benchmark remain clearly distinct.
    letter(0.832, 0.595, "G")
    rgb = normalize(a["membrane"])[..., None] * np.array([0.30, 0.80, 0.75])
    ax = axis([0.832, 0.467, 0.079, 0.103], rgb)
    for e in cx.cell_cell_edges()[::2]:
        q = cx.edge_geometry(int(e))
        ax.plot(q[:, 1], q[:, 0], c="white", lw=0.4)
    ax = axis([0.922, 0.481, 0.06, 0.083])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    table = [["cell", "area", "n"]]
    for f in cx.cell_faces[:4]:
        table.append(
            [str(int(cx.face_label[f])), str(int((labels == cx.face_label[f]).sum())), str(cx.face_sides(int(f)))]
        )
    for row, values in enumerate(table):
        yy = 0.94 - row * 0.204
        if row == 0:
            ax.plot([0, 1], [yy - 0.095, yy - 0.095], color=TEAL, lw=1)
        for col, value in enumerate(values):
            ax.text(
                [0.0, 0.34, 0.91][col],
                yy,
                value,
                fontsize=8,
                weight="bold" if row == 0 else "normal",
                va="center",
                ha="left",
            )
    ax.axis("off")
    ax = mapped_axes([0.879, 0.348, 0.098, 0.063])
    scores = np.array([[0.635, 0.680], [0.830, 0.867], [0.790, 0.816]])
    for i in range(3):
        ax.plot(scores[i], [i, i], c=LIGHT, lw=2)
        ax.scatter(scores[i], [i, i], s=24, c=[GRAY, TEAL], zorder=3)
    ax.set_yticks([0, 1, 2], ["vertex F1", "adjacency F1", "PQ"], fontsize=8)
    ax.set_xlim(0.6, 0.92)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    ax.tick_params(labelsize=8, length=2)
    ax.invert_yaxis()
    text(0.865, 0.316, "● Cellpose", 8, GRAY)
    text(0.932, 0.316, "● v6", 8, TEAL)

    # H-K are separate modules, with their own local input illustrations.
    # No giant bus starts at a benchmark or falsely feeds a 3-D volume from 2-D.
    letter(0.028, 0.264, "H")
    ax = axis([0.061, 0.207, 0.048, 0.062])
    complex_view(ax, cx, labels, width=0.35, vertices=False)
    arrow((0.085, 0.199), (0.085, 0.184), GRAY, lw=1)
    cent = {int(f): np.argwhere(labels == cx.face_label[f]).mean(axis=0) for f in cx.cell_faces}
    ax = axis([0.031, 0.047, 0.112, 0.132])
    for e in cx.cell_cell_edges():
        f, g = map(int, cx.edge_faces[e])
        r, s = cent[f], cent[g]
        ax.plot([r[1], s[1]], [r[0], s[0]], c=TEAL, lw=0.8)
    pts = np.array(list(cent.values()))
    ax.scatter(pts[:, 1], pts[:, 0], s=14, c=PURPLE, ec="white", linewidth=0.3)
    ax.set_xlim(-5, 389)
    ax.set_ylim(389, -5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax = axis([0.165, 0.078, 0.081, 0.112])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for j, col in enumerate([TEAL, PURPLE, AMBER]):
        for left, right in [[0.04, 0.24], [0.32 + j * 0.07, 0.52 + j * 0.07], [0.8, 0.96]]:
            ax.plot([left, right], [0.8 - j * 0.3] * 2, c=col, lw=6, solid_capstyle="round")
        ax.plot([0.02, 0.98], [0.8 - j * 0.3] * 2, c=col, lw=0.7, alpha=0.4)
    ax.axis("off")

    letter(0.283, 0.264, "I")
    geometric_patch(axis([0.285, 0.079, 0.081, 0.111]), True)
    geometric_patch(axis([0.414, 0.079, 0.081, 0.111]), False)
    arrow((0.376, 0.132), (0.402, 0.132), TEAL)
    text(0.3255, 0.205, "t", 10, ha="center")
    text(0.4545, 0.205, "t + 1", 10, ha="center")

    letter(0.546, 0.264, "J")
    ax = axis([0.547, 0.071, 0.1, 0.132])
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
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
    ax = axis([0.682, 0.071, 0.096, 0.132])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
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
    ax.scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 0], cmap="viridis", s=28, zorder=5)
    ax.axis("off")

    letter(0.832, 0.264, "K")
    ax = mapped_axes([0.82, 0.063, 0.078, 0.139], projection="3d")
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
    ax = mapped_axes([0.911, 0.063, 0.078, 0.139], projection="3d")
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

    # G is a shallow export tier: image, numbers, then the independent
    # benchmark. Position the real table separately so its output port is clear.
    g_image, g_table, g_plot = panel_axes["G"]
    g_image.set_position([0.301, 0.229, 0.073, 0.082])
    g_table.set_position([0.539, 0.239, 0.088, 0.069])
    g_plot.set_position([0.79, 0.243, 0.12, 0.055])
    # Existing G text consists only of its letter and two method keys.
    for artist in list(fig.texts):
        if artist.get_text() in ["A", "B"]:
            artist.set_y(0.979)
        elif artist.get_text() in ["C", "D"]:
            artist.set_y(0.720)
        elif artist.get_text() in ["E", "F"]:
            artist.set_y(0.542)
        elif artist.get_text() == "G":
            artist.set_position((0.265, 0.317))
        elif artist.get_text() in ["H", "I", "J", "K"]:
            artist.set_y(0.175)
        elif artist.get_text() == "● Cellpose":
            artist.set_position((0.782, 0.221))
        elif artist.get_text() == "● v6":
            artist.set_position((0.869, 0.221))

    fig.canvas.draw()

    def port(ax, side):
        b = ax.get_position()
        return {
            "top": ((b.x0 + b.x1) / 2, b.y1),
            "bottom": ((b.x0 + b.x1) / 2, b.y0),
            "left": (b.x0, (b.y0 + b.y1) / 2),
            "right": (b.x1, (b.y0 + b.y1) / 2),
        }[side]

    routes = []

    def connect(name, points, color=TEAL, dashed=False):
        routes.append({"name": name, "points": points, "optional": dashed})
        if len(points) > 2:
            fig.add_artist(
                plt.Line2D(
                    *zip(*points[:-1]),
                    transform=fig.transFigure,
                    color=color,
                    lw=1.45,
                    ls="--" if dashed else "-",
                    solid_joinstyle="round",
                    solid_capstyle="round",
                )
            )
        fig.add_artist(
            FancyArrowPatch(
                points[-2],
                points[-1],
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.45,
                color=color,
                linestyle="--" if dashed else "-",
            )
        )

    active["name"] = "A"
    origin = tuple(point(0.242, 0.836))
    active["name"] = "B"
    destination = tuple(point(0.271, 0.835))
    connect("registered channels to proposal alternatives", [(origin[0], destination[1]), destination])
    origin = port(panel_axes["B"][3], "bottom")
    destination = port(panel_axes["C"][0], "top")
    connect(
        "proposal evidence to constrained decoder",
        [(destination[0], origin[1]), (destination[0], destination[1] + 0.007)],
    )
    origin = port(panel_axes["C"][0], "left")
    destination = port(panel_axes["D"][0], "right")
    connect(
        "decoded labels to embedded complex",
        [(origin[0] - 0.005, destination[1]), (destination[0] + 0.009, destination[1])],
    )
    origin = port(panel_axes["C"][0], "bottom")
    destination = port(panel_axes["E"][1], "top")
    connect(
        "decoder to optional ensemble",
        [origin, (origin[0], 0.556), (destination[0], 0.556), (destination[0], destination[1] + 0.007)],
        PURPLE,
        True,
    )
    origin = port(panel_axes["D"][0], "bottom")
    destination = port(panel_axes["F"][0], "top")
    connect(
        "physical interfaces to sampling strip",
        [(destination[0], origin[1]), (destination[0], destination[1] + 0.007)],
    )
    # A vertical output port sits to the right of the trace's axis title;
    # it lands on the numeric export rather than traversing the whole tier.
    origin = (0.583, panel_axes["F"][2].get_position().y0 - 0.013)
    destination = port(g_table, "top")
    connect("interface measurements to exports", [origin, (destination[0], destination[1] + 0.008)])
    # Only the table/complex exports feed the analysis branch, not the benchmark.
    origin = port(g_table, "bottom")
    fig.add_artist(
        plt.Line2D([origin[0], origin[0]], [origin[1] - 0.006, 0.196], transform=fig.transFigure, color=GRAY, lw=1.2)
    )
    routes.append(
        {"name": "structural tables to analysis tier", "points": [origin, (origin[0], 0.196)], "optional": False}
    )
    analysis_ports = [
        ("phenotypes", port(panel_axes["H"][0], "top"), 0.008),
        ("dynamics", port(panel_axes["I"][0], "top"), 0.022),
        ("mechanics", port(panel_axes["J"][0], "top"), 0.008),
        ("transport", port(panel_axes["J"][1], "top"), 0.008),
    ]
    fig.add_artist(
        plt.Line2D(
            [analysis_ports[0][1][0], analysis_ports[-1][1][0]],
            [0.196, 0.196],
            transform=fig.transFigure,
            color=GRAY,
            lw=1.2,
        )
    )
    for name, destination, clearance in analysis_ports:
        xx, yy = destination
        connect(name, [(xx, 0.196), (xx, yy + clearance)], GRAY)

    OUT.mkdir(exist_ok=True, parents=True)
    for suffix in ["png", "svg", "pdf"]:
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix, dpi=220, facecolor="white")
        (OUT / f"methods_full_flow.{suffix}").write_bytes(buffer.getvalue())
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
        "layout_revision": "five downward tiers; caption-only narrative; explicit diagram ports; no title or subtitles",
        "connector_routes": routes,
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
