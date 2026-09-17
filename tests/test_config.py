import warnings
from pathlib import Path

import pytest
import yaml
from endopigraph.config import load_config, resolve_channel_roles, _deep_merge

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config_sbiad1540.yaml"


class TestDeepMerge:
    def test_flat(self):
        r = _deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert r == {"a": 1, "b": 3}

    def test_nested(self):
        r = _deep_merge({"x": {"a": 1, "b": 2}}, {"x": {"b": 99}})
        assert r["x"]["a"] == 1
        assert r["x"]["b"] == 99

    def test_new_keys(self):
        r = _deep_merge({"a": 1}, {"b": 2})
        assert r == {"a": 1, "b": 2}

    def test_override_dict_with_scalar(self):
        r = _deep_merge({"a": {"x": 1}}, {"a": 42})
        assert r["a"] == 42


class TestLoadConfig:
    def test_valid(self, tmp_path):
        cfg = {"manifest_csv": "m.csv", "output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        c = load_config(p)
        assert c["manifest_csv"] == "m.csv"
        assert "segmentation" in c

    def test_missing_required(self, tmp_path):
        cfg = {"output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ValueError, match="manifest_csv"):
            load_config(p)

    def test_defaults_applied(self, tmp_path):
        cfg = {"manifest_csv": "m.csv", "output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        c = load_config(p)
        assert c["segmentation"]["method"] == "cellpose"
        assert c["graph"]["min_contact_px"] == 10

    def test_not_dict_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError):
            load_config(p)


def _write(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    return p


class TestPixelSize:
    def test_default_none(self, tmp_path):
        c = load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/"}))
        assert c["pixel_size_um"] is None

    def test_parsed_as_float(self, tmp_path):
        c = load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/", "pixel_size_um": 0.4151}))
        assert c["pixel_size_um"] == pytest.approx(0.4151)
        assert isinstance(c["pixel_size_um"], float)

    def test_int_accepted(self, tmp_path):
        c = load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/", "pixel_size_um": 1}))
        assert c["pixel_size_um"] == 1.0

    @pytest.mark.parametrize("bad", ["0.4", -1.0, 0, True])
    def test_invalid_raises(self, tmp_path, bad):
        with pytest.raises(ValueError, match="pixel_size_um"):
            load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/", "pixel_size_um": bad}))


class TestChannelsBlock:
    def test_defaults_when_absent(self, tmp_path):
        c = load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/"}))
        assert c["channels"] == {"geometry": None, "nuclei": None, "junction": []}

    def test_explicit_block_kept(self, tmp_path):
        cfg = {
            "manifest_csv": "m.csv",
            "output_dir": "out/",
            "channels": {"geometry": "Membrane", "nuclei": 2, "junction": ["VE-cadherin", "ZO-1"]},
        }
        c = load_config(_write(tmp_path, cfg))
        assert c["channels"] == {"geometry": "Membrane", "nuclei": 2, "junction": ["VE-cadherin", "ZO-1"]}

    def test_single_junction_string_wrapped(self, tmp_path):
        cfg = {"manifest_csv": "m.csv", "output_dir": "out/", "channels": {"geometry": 0, "junction": "VE-cadherin"}}
        c = load_config(_write(tmp_path, cfg))
        assert c["channels"]["junction"] == ["VE-cadherin"]

    def test_null_block_is_defaults(self, tmp_path):
        c = load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/", "channels": None}))
        assert c["channels"]["geometry"] is None

    @pytest.mark.parametrize(
        "block",
        [
            "VE-cadherin",
            {"geometry": 1.5},
            {"geometry": True},
            {"junction": [{"channel_name": "x"}]},
            {"geometry": 0, "membrane": 1},
        ],
    )
    def test_invalid_block_raises(self, tmp_path, block):
        with pytest.raises(ValueError, match="channels"):
            load_config(_write(tmp_path, {"manifest_csv": "m.csv", "output_dir": "out/", "channels": block}))


class TestResolveChannelRoles:
    def test_explicit_block_membrane_geometry(self):
        cfg = {"channels": {"geometry": "Membrane", "nuclei": "DAPI", "junction": ["VE-cadherin"]}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            roles = resolve_channel_roles(cfg)
        assert roles == {
            "geometry": "Membrane",
            "nuclei": "DAPI",
            "junction": ["VE-cadherin"],
            "geometry_source": "membrane_channel",
        }

    def test_warns_when_geometry_is_junction_by_name(self):
        cfg = {"channels": {"geometry": "VE-cadherin", "nuclei": None, "junction": ["VE-cadherin"]}}
        with pytest.warns(UserWarning, match="circular"):
            roles = resolve_channel_roles(cfg)
        assert roles["geometry_source"] == "junction_channel"
        assert roles["geometry"] == "VE-cadherin"

    def test_warns_when_geometry_is_junction_by_index(self):
        cfg = {"channels": {"geometry": 0, "junction": [0, 2]}}
        with pytest.warns(UserWarning, match="geometry_source"):
            roles = resolve_channel_roles(cfg)
        assert roles["geometry_source"] == "junction_channel"

    def test_index_and_name_do_not_match(self):
        cfg = {"channels": {"geometry": 0, "junction": ["VE-cadherin"]}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            roles = resolve_channel_roles(cfg)
        assert roles["geometry_source"] == "membrane_channel"

    def test_backward_compatible_cellpose(self):
        cfg = {
            "segmentation": {
                "method": "cellpose",
                "cellpose": {"channels": {"cyto": {"channel_name": "VE-cadherin"}, "nuclei": {"channel_index": 2}}},
            },
            "junction_markers": {"AJ": {"channel_name": "VE-cadherin", "threshold": "otsu"}},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            roles = resolve_channel_roles(cfg)
        assert roles == {
            "geometry": "VE-cadherin",
            "nuclei": 2,
            "junction": ["VE-cadherin"],
            "geometry_source": "unspecified",
        }

    def test_backward_compatible_watershed(self):
        cfg = {
            "segmentation": {
                "method": "watershed",
                "watershed": {"nuclei": {"channel_index": 1}, "membrane": {"channel_index": 0}},
            },
            "junction_markers": {"AJ": {"channel_index": 0}},
        }
        roles = resolve_channel_roles(cfg)
        assert roles["geometry"] == 0
        assert roles["nuclei"] == 1
        assert roles["junction"] == [0]
        assert roles["geometry_source"] == "unspecified"

    def test_backward_compatible_via_load_config(self, tmp_path):
        cfg = {
            "manifest_csv": "m.csv",
            "output_dir": "out/",
            "segmentation": {"cellpose": {"channels": {"cyto": "Membrane"}}},
            "junction_markers": {"AJ": {"channel": "VE-cadherin"}},
        }
        roles = resolve_channel_roles(load_config(_write(tmp_path, cfg)))
        assert roles["geometry"] == "Membrane"
        assert roles["junction"] == ["VE-cadherin"]
        assert roles["geometry_source"] == "unspecified"

    def test_explicit_geometry_falls_back_to_marker_junctions(self):
        cfg = {
            "channels": {"geometry": "VE-cadherin"},
            "junction_markers": {"AJ": {"channel": "VE-cadherin"}},
        }
        with pytest.warns(UserWarning):
            roles = resolve_channel_roles(cfg)
        assert roles["junction"] == ["VE-cadherin"]
        assert roles["geometry_source"] == "junction_channel"

    def test_empty_config(self):
        roles = resolve_channel_roles({})
        assert roles == {"geometry": None, "nuclei": None, "junction": [], "geometry_source": "unspecified"}

    def test_example_sbiad1540_config_documents_circularity(self):
        with open(EXAMPLE_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["pixel_size_um"] == pytest.approx(0.4151)
        with pytest.warns(UserWarning, match="circular"):
            roles = resolve_channel_roles(cfg)
        assert roles["geometry"] == "VE-cadherin"
        assert roles["junction"] == ["VE-cadherin"]
        assert roles["geometry_source"] == "junction_channel"
