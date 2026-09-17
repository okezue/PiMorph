#!/usr/bin/env python3
"""Pair VE-strat VE-cadherin (w2) and nuclear (w4) TIFFs per site into one manifest row.

The original data/ve_strat/manifest.csv listed only the w4 file and called it
VE-cadherin. Visual inspection and a structure-thickness check on all 20 files show
w4 holds the nuclei (thick blobs) and w2 holds the junction mesh (thin lines), so the
legacy pipeline segmented nuclei as cells and found no contacts. Each site has two
single-channel 2048x2048 uint16 files whose names share the `_<well>_s<N>_` token.

Usage:
    python scripts/make_ve_strat_paired_manifest.py [--root data/ve_strat] [--out data/ve_strat/manifest_paired.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Plate well (e.g. D01, E05), site index within the well, wavelength index (w2 VE-cadherin, w4 nuclei).
SITE_RE = re.compile(r"_(?P<well>[A-H]\d{2})_s(?P<site>\d+)_w(?P<wave>\d)")


def pair_files(root: Path) -> list[dict]:
    rows: list[dict] = []
    for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        by_site: dict[str, dict[str, Path]] = {}
        for tif in sorted(cond_dir.glob("*.tif")):
            m = SITE_RE.search(tif.name)
            if not m:
                continue
            key = f"{m['well']}_{m['site']}"
            by_site.setdefault(key, {})[m["wave"]] = tif
        for key, waves in sorted(by_site.items(), key=lambda kv: int(kv[0].split("_")[1])):
            if "2" not in waves or "4" not in waves:
                continue
            well, site = key.split("_")
            rows.append(
                {
                    "image_id": f"{cond_dir.name}_s{site}",
                    "path_geometry": str(waves["2"].relative_to(root)),
                    "path_nuclei": str(waves["4"].relative_to(root)),
                    "path_junction": str(waves["2"].relative_to(root)),
                    "geometry_source": "junction_channel",
                    "condition": cond_dir.name,
                    "well": well,
                    "replicate": site,
                    "dataset": "ve_strat",
                    "pixel_size_um": "",
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ve_strat")
    ap.add_argument("--out", default="data/ve_strat/manifest_paired.csv")
    args = ap.parse_args()
    root = Path(args.root)
    rows = pair_files(root)
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} paired rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
