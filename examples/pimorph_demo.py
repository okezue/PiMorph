"""Write a small, reproducible synthetic microscopy example for the README.

Run from the repository root: python examples/pimorph_demo.py
This creates simulated images and exact labels, not experimental observations.
"""

from pathlib import Path

import pandas as pd
import tifffile

from pimorph.synth import RenderParams, SynthTissueParams, generate_tissue, render_channels


def main() -> None:
    out = Path("output/demo_input")
    out.mkdir(parents=True, exist_ok=True)
    tissue = generate_tissue(SynthTissueParams(shape=(192, 192), n_cells=28, n_gaps=2, seed=17))
    channels = render_channels(tissue, RenderParams(seed=23, broken_fraction=0.10, nucleus_radius_px=4.0))
    for name in ("membrane", "nuclei", "junction"):
        tifffile.imwrite(out / f"{name}.tif", channels[name], photometric="minisblack")
    tifffile.imwrite(out / "labels_truth.tif", tissue.labels, photometric="minisblack")
    pd.DataFrame(
        [
            {
                "image_id": "synthetic_demo",
                "path_geometry": "membrane.tif",
                "path_nuclei": "nuclei.tif",
                "path_junction": "junction.tif",
                "geometry_source": "membrane_channel",
                "condition": "synthetic",
            }
        ]
    ).to_csv(out / "manifest.csv", index=False)
    print(f"Synthetic input and manifest: {out}")


if __name__ == "__main__":
    main()
