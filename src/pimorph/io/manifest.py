"""Manifests with explicit channel roles.

A row describes one field of view. Channels are given either as one multichannel
file with per-channel names (legacy S-BIAD1540 style: ``path`` plus ``channel_1..n``)
or as separate single-channel files (``path_geometry``, ``path_nuclei``,
``path_junction``). Roles are always explicit:

- ``geometry``: the channel used to place boundaries (membrane, or the junction
  channel when nothing independent exists; then ``geometry_source = junction_channel``
  is recorded because scoring junction continuity on boundaries found with the same
  signal is circular).
- ``nuclei``: seed channel.
- ``junction``: molecular channel(s) scored along interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .images import CalibratedImage, read_tiff, stack_channel_files

_JUNCTION_NAMES = ("ve-cadherin", "vecad", "ve_cadherin", "cdh5", "pecam", "zo-1", "zo1", "claudin", "occludin")
_NUCLEI_NAMES = ("dapi", "hoechst", "h2b", "nuclei", "nucleus", "dna", "sytox")
_MEMBRANE_NAMES = ("membrane", "cellmask", "wga", "cd31", "caax", "plasma")


@dataclass
class FieldSpec:
    image_id: str
    root: Path
    files: Dict[str, Path]  # role -> file (single-channel mode) or {"multi": path}
    channel_names: List[str]  # multichannel mode
    roles: Dict[str, str]  # role -> channel name or file role key
    geometry_source: str  # "membrane_channel" | "junction_channel" | "nuclei_only"
    pixel_size_um: Optional[float]
    condition: Optional[str] = None
    replicate: Optional[str] = None
    dataset: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    def load(self) -> CalibratedImage:
        if "multi" in self.files:
            im = read_tiff(self.files["multi"], channel_names=self.channel_names)
        else:
            im = stack_channel_files({role: p for role, p in self.files.items()}, pixel_size_um=self.pixel_size_um)
        if im.pixel_size_um is None and self.pixel_size_um is not None:
            im.pixel_size_um = float(self.pixel_size_um)
            im.pixel_size_source = "manifest"
        im.meta["roles"] = dict(self.roles)
        im.meta["geometry_source"] = self.geometry_source
        im.meta["image_id"] = self.image_id
        return im

    def role_channel(self, im: CalibratedImage, role: str):
        key = self.roles.get(role)
        if key is None:
            return None
        return im.channel(key)


def _guess_role(name: str, candidates) -> bool:
    n = name.lower()
    return any(c in n for c in candidates)


def parse_manifest(path: Path | str, root: Optional[Path | str] = None) -> List[FieldSpec]:
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    df = pd.read_csv(path, dtype=str).fillna("")
    specs: List[FieldSpec] = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        px = row.get("pixel_size_um", "")
        px_f = float(px) if px not in ("", None) else None
        cond = row.get("condition", "") or row.get("shear_stress", "") or None
        rep = row.get("replicate", "") or None
        ds = row.get("dataset", "") or None
        if "path_geometry" in df.columns or "path_nuclei" in df.columns:
            files: Dict[str, Path] = {}
            roles: Dict[str, str] = {}
            for role in ("geometry", "nuclei", "junction"):
                p = row.get(f"path_{role}", "")
                if p:
                    files[role] = root / p
                    roles[role] = role
            # identical geometry and junction files: keep one array, alias the role
            if "geometry" in files and "junction" in files and files["geometry"] == files["junction"]:
                del files["junction"]
                roles["junction"] = "geometry"
            gsrc = row.get("geometry_source", "") or (
                "junction_channel" if roles.get("junction") == "geometry" else "membrane_channel"
            )
            specs.append(FieldSpec(image_id, root, files, [], roles, gsrc, px_f, cond, rep, ds, dict(row)))
        else:
            names = [row[c] for c in df.columns if c.startswith("channel_") and row[c] not in ("", "-")]
            roles = {}
            for n in names:
                if _guess_role(n, _JUNCTION_NAMES) and "junction" not in roles:
                    roles["junction"] = n
                elif _guess_role(n, _NUCLEI_NAMES) and "nuclei" not in roles:
                    roles["nuclei"] = n
                elif _guess_role(n, _MEMBRANE_NAMES) and "geometry" not in roles:
                    roles["geometry"] = n
            if "geometry" in roles:
                gsrc = "membrane_channel"
            elif "junction" in roles:
                roles["geometry"] = roles["junction"]
                gsrc = "junction_channel"
            else:
                gsrc = "nuclei_only"
            # pad channel names to the file's channel count lazily; store known names
            specs.append(
                FieldSpec(
                    image_id, root, {"multi": root / row["path"]}, names, roles, gsrc, px_f, cond, rep, ds, dict(row)
                )
            )
    return specs
