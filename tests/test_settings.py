"""Settings model: validation, clamping, pending/apply/cancel, and corruption fallback."""
from __future__ import annotations

import json
import os

from pycrossy import settings as st


def test_defaults_cover_all_value_specs():
    d = st.defaults()
    for sp in st.SPECS:
        if sp.kind in (st.INFO, st.ACTION):
            assert sp.key not in d
        else:
            assert sp.key in d, f"{sp.key} missing from defaults"
            # every default must survive its own clamp unchanged
            assert sp.clamp(sp.default) == sp.default, sp.key


def test_slider_clamps_and_snaps():
    cfg = st.Config()
    assert cfg.set("master_volume", 999) == 100        # clamp to hi
    assert cfg.set("master_volume", -50) == 0          # clamp to lo
    assert cfg.set("render_scale", 73) == 75           # snap to step 5
    assert isinstance(cfg.get("master_volume"), int)


def test_choice_rejects_invalid_keeps_default():
    cfg = st.Config()
    assert cfg.set("shadow_quality", "ultra") == "high"     # invalid -> default
    assert cfg.set("shadow_quality", "low") == "low"


def test_pending_apply_cancel():
    cfg = st.Config()
    base = cfg.get("brightness")
    cfg.set("brightness", base + 20)
    assert cfg.dirty
    assert cfg.get("brightness") == base + 20
    assert cfg.committed("brightness") == base          # not committed yet
    cfg.cancel()
    assert not cfg.dirty
    assert cfg.get("brightness") == base

    cfg.set("brightness", base + 20)
    changed = cfg.apply()
    assert "brightness" in changed
    assert cfg.committed("brightness") == base + 20
    assert not cfg.dirty


def test_setting_back_to_committed_clears_pending():
    cfg = st.Config()
    base = cfg.get("camera_zoom")
    cfg.set("camera_zoom", base + 10)
    assert cfg.dirty
    cfg.set("camera_zoom", base)
    assert not cfg.dirty


def test_needs_restart_detection():
    cfg = st.Config()
    cfg.set("brightness", 70)
    assert not cfg.needs_restart()
    cfg.set("vsync", not cfg.get("vsync"))
    assert cfg.needs_restart()


def test_cycle_choice_wraps():
    cfg = st.Config()
    vals = [c[0] for c in st.SPEC_BY_KEY["shadow_quality"].choices]
    start = cfg.get("shadow_quality")
    seen = [start]
    for _ in range(len(vals)):
        seen.append(cfg.cycle_choice("shadow_quality", 1))
    assert seen[-1] == start                            # wrapped fully around


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "settings.json")
    cfg = st.Config(path=path)
    cfg.set("shadow_quality", "low")
    cfg.set("master_volume", 35)
    cfg.set("key_up", "w")
    cfg.apply()

    cfg2 = st.load(path)
    assert cfg2.get("shadow_quality") == "low"
    assert cfg2.get("master_volume") == 35
    assert cfg2.get("key_up") == "w"


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        fh.write("{ this is not valid json ]]]")
    cfg = st.load(path)                                 # must not raise
    assert cfg.get("shadow_quality") == st.SPEC_BY_KEY["shadow_quality"].default


def test_partial_garbage_keeps_good_fields(tmp_path):
    path = str(tmp_path / "settings.json")
    with open(path, "w") as fh:
        json.dump({"version": 1, "values": {
            "shadow_quality": "nonsense",     # invalid -> default
            "master_volume": 9999,            # clamp -> 100
            "unknown_key": 5,                 # dropped
            "render_scale": 65,               # valid
        }}, fh)
    cfg = st.load(path)
    assert cfg.get("shadow_quality") == "high"
    assert cfg.get("master_volume") == 100
    assert cfg.get("render_scale") == 65
    assert "unknown_key" not in cfg.to_dict()


def test_missing_file_is_defaults(tmp_path):
    cfg = st.load(str(tmp_path / "does-not-exist.json"))
    assert cfg.to_dict() == st.defaults()


def test_export_import(tmp_path):
    src = st.Config(path=str(tmp_path / "a.json"))
    src.set("difficulty", "hard")
    src.set("brightness", 120)
    src.apply()
    out = str(tmp_path / "exported.json")
    assert src.export_to(out)

    dst = st.Config(path=str(tmp_path / "b.json"))
    assert dst.import_from(out)
    assert dst.get("difficulty") == "hard"       # staged as pending
    assert dst.get("brightness") == 120
    dst.apply()
    assert dst.committed("difficulty") == "hard"


def test_reset_to_defaults():
    cfg = st.Config()
    cfg.set("difficulty", "hard")
    cfg.set("brightness", 130)
    cfg.apply()
    cfg.reset_to_defaults()
    assert cfg.dirty
    cfg.apply()
    assert cfg.to_dict() == st.defaults()


def test_backup_creates_file(tmp_path):
    path = str(tmp_path / "settings.json")
    cfg = st.Config(path=path)
    cfg.save()
    dst = cfg.backup()
    assert dst and os.path.exists(dst)


def test_info_and_action_specs_not_settable():
    cfg = st.Config()
    assert cfg.set("gpu_info", "x") is None
    assert cfg.set("controls_reset", "x") is None
