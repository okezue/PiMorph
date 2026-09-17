from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

ChannelRef = Union[str, int]

# Channel roles are declared explicitly so the run records which signal defined
# cell geometry. Scoring junction continuity on boundaries found with the same
# junction signal is circular; resolve_channel_roles() warns in that case.
DEFAULTS: Dict[str, Any] = {
    "pixel_size_um": None,
    "channels": {
        "geometry": None,
        "nuclei": None,
        "junction": [],
    },
    "segmentation": {
        "method": "cellpose",
        "cellpose": {
            "model_type": "cyto2",
            "diameter": 30,
            "channels": {"cyto": None, "nuclei": None},
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
        },
        "watershed": {
            "nuclei": None,
            "membrane": None,
            "min_cell_area_px": 200,
        },
    },
    "junction_markers": {
        "AJ": {
            "channel": None,
            "threshold": "otsu",
            "min_occupancy": 0.05,
            "dilate_px": 2,
        }
    },
    "graph": {
        "min_contact_px": 10,
    },
    "qc": {
        "make_figures": True,
        "max_images": None,
        "random_seed": 0,
    },
}


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict")

    # Merge defaults (shallow + nested)
    merged = _deep_merge(DEFAULTS, cfg)

    # Required keys
    for k in ("manifest_csv", "output_dir"):
        if k not in merged or merged[k] in (None, ""):
            raise ValueError(f"Missing required config key: {k}")

    merged["pixel_size_um"] = _validate_pixel_size(merged.get("pixel_size_um"))
    merged["channels"] = _validate_channels(merged.get("channels"))

    return merged


def _validate_pixel_size(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pixel_size_um must be a positive number or null, got {value!r}")
    if value <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {value!r}")
    return float(value)


def _is_channel_ref(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _validate_channels(block: Any) -> Dict[str, Any]:
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(f"channels must be a mapping, got {type(block).__name__}")
    out: Dict[str, Any] = {}
    for role in ("geometry", "nuclei"):
        v = block.get(role)
        if v is not None and not _is_channel_ref(v):
            raise ValueError(f"channels.{role} must be a channel name, index, or null, got {v!r}")
        out[role] = v
    junction = block.get("junction", [])
    if junction is None:
        junction = []
    if _is_channel_ref(junction):
        junction = [junction]
    if not isinstance(junction, list) or not all(_is_channel_ref(j) for j in junction):
        raise ValueError(f"channels.junction must be a list of channel names or indices, got {junction!r}")
    out["junction"] = list(junction)
    for k in block:
        if k not in ("geometry", "nuclei", "junction"):
            raise ValueError(f"Unknown key channels.{k}; expected geometry, nuclei, junction")
    return out


def _channel_ref(spec: Any) -> Optional[ChannelRef]:
    """Normalize a legacy channel spec (name, index, or {channel_name/channel_index}) to a name or index."""
    if spec is None or isinstance(spec, bool):
        return None
    if isinstance(spec, (str, int)):
        return spec
    if isinstance(spec, dict):
        if spec.get("channel_name") is not None:
            return spec["channel_name"]
        if spec.get("channel_index") is not None:
            return int(spec["channel_index"])
        if spec.get("channel") is not None:
            return spec["channel"]
    return None


def _legacy_geometry_and_nuclei(cfg: Dict[str, Any]) -> tuple[Optional[ChannelRef], Optional[ChannelRef]]:
    seg = cfg.get("segmentation") or {}
    method = str(seg.get("method", "cellpose"))
    if method == "watershed":
        ws = seg.get("watershed") or {}
        geometry = _channel_ref(ws.get("membrane", ws.get("membrane_channel")))
        nuclei = _channel_ref(ws.get("nuclei", ws.get("nuclei_channel")))
    else:
        ch = (seg.get("cellpose") or {}).get("channels") or {}
        geometry = _channel_ref(ch.get("cyto"))
        nuclei = _channel_ref(ch.get("nuclei"))
    return geometry, nuclei


def _legacy_junction_channels(cfg: Dict[str, Any]) -> List[ChannelRef]:
    out: List[ChannelRef] = []
    for jcfg in (cfg.get("junction_markers") or {}).values():
        if not isinstance(jcfg, dict):
            continue
        ref = _channel_ref(jcfg.get("channel_name") or jcfg.get("channel"))
        if ref is None and jcfg.get("channel_index") is not None:
            ref = int(jcfg["channel_index"])
        if ref is not None and ref not in out:
            out.append(ref)
    return out


def resolve_channel_roles(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the channel used for each role and how the geometry channel was chosen.

    ``geometry_source`` is ``"junction_channel"`` when the geometry channel is one
    of the junction channels (circular: junction continuity is scored on
    boundaries found with the same signal), ``"membrane_channel"`` when an
    explicit ``channels.geometry`` is not a junction channel, and
    ``"unspecified"`` when no ``channels`` block is given and the roles are
    derived from the legacy segmentation / junction_markers settings.
    """
    block = _validate_channels(cfg.get("channels"))
    legacy_geometry, legacy_nuclei = _legacy_geometry_and_nuclei(cfg)

    explicit = block["geometry"] is not None
    geometry = block["geometry"] if explicit else legacy_geometry
    nuclei = block["nuclei"] if block["nuclei"] is not None else legacy_nuclei
    junction = list(block["junction"]) if block["junction"] else _legacy_junction_channels(cfg)

    if not explicit:
        geometry_source = "unspecified"
    elif geometry in junction:
        geometry_source = "junction_channel"
    else:
        geometry_source = "membrane_channel"

    if geometry_source == "junction_channel":
        warnings.warn(
            f"channels.geometry={geometry!r} is also a junction channel {junction!r}: cell boundaries are found "
            "with the same signal that is scored for junction continuity, so junction features are partly "
            "circular. geometry_source is recorded as 'junction_channel'; prefer a membrane or nuclei-based "
            "geometry channel when one is available.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "geometry": geometry,
        "nuclei": nuclei,
        "junction": junction,
        "geometry_source": geometry_source,
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if k in override:
            if isinstance(v, dict) and isinstance(override[k], dict):
                out[k] = _deep_merge(v, override[k])
            else:
                out[k] = override[k]
        else:
            out[k] = v
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out
