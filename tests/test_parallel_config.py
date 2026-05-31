"""Tests for persisted parallel-grain tuning configuration."""

from __future__ import annotations

import json

import pytest
import reducers as rd


def test_parallel_grain_config_roundtrip(tmp_path):
    path = tmp_path / "parallel_grains.json"
    defaults = rd.get_default_parallel_grains()
    tuned = defaults | {"axis_scan_var": 1024, "minmax_1d": 16384}

    saved_path = rd.save_parallel_grains_config(tuned, path=path)

    assert saved_path == path
    assert rd.load_parallel_grains_config(path=path) == tuned

    rd.use_default_parallel_grains()
    assert rd.get_parallel_grains() == defaults

    assert rd.apply_parallel_grains_config(path=path) is True
    assert rd.get_parallel_grains() == tuned

    assert rd.clear_parallel_grains_config(path=path) is True
    assert not path.exists()
    assert rd.get_parallel_grains() == defaults


def test_missing_parallel_grain_config_is_noop(tmp_path):
    rd.use_default_parallel_grains()
    before = rd.get_parallel_grains()

    assert rd.apply_parallel_grains_config(path=tmp_path / "missing.json") is False
    assert rd.get_parallel_grains() == before


def test_save_parallel_grain_config_defaults_to_active_values(tmp_path):
    path = tmp_path / "parallel_grains.json"
    defaults = rd.get_default_parallel_grains()
    active = defaults | {"axis_scan_var": 2048}
    try:
        rd.set_parallel_grains(active)
        rd.save_parallel_grains_config(path=path)
        assert rd.load_parallel_grains_config(path=path) == active
    finally:
        rd.use_default_parallel_grains()


def test_parallel_grain_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "parallel_grains.json"
    path.write_text(json.dumps({"axis_scan_plain": 1024, "bogus": 2048}))

    with pytest.raises(ValueError, match="unknown parallel grain"):
        rd.load_parallel_grains_config(path=path)
