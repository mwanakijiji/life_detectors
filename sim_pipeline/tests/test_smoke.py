"""
Smoke-test mode: unit tests of the config helpers (always run), and an end-to-end
smoke run of the pipeline (marked 'smoke'; run with --run-smoke).
"""

import configparser
import time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from batch_process import run_single_calculation
from modules.utils.helpers.config_ops import (
    SMOKE_TEST_LIMIT_DEFAULTS,
    apply_smoke_test_overrides,
    smoke_test_enabled,
    smoke_test_limits,
)
from modules.utils.helpers.output_dirs import (
    COUNTS_TIME,
    INTERMEDIATE,
    OUTPUT_SUBDIRS,
)

MAIN_CONFIG = Path(__file__).resolve().parents[1] / "config" / "main_config.ini"


def _write_config(path: Path, enabled: str = "False") -> Path:
    cfg = configparser.ConfigParser()
    cfg.read_dict(
        {
            "detector": {"size_y": "200", "size_x": "1024", "spec_res": "100"},
            "observation": {"N_angles": "36"},
            "smoke_test": {
                "enabled": enabled,
                "detector.size_y": "64",
                "detector.spec_res": "20",
                "observation.N_angles": "2",
                "new_section.key": "value",
                "max_qe_values": "1",
                "max_planets": "2",
                "write_full_frame_example": "False",
            },
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)
    return path


def _read(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


class TestSmokeTestHelpers:
    def test_overrides_not_applied_when_disabled(self, tmp_path):
        path = _write_config(tmp_path / "c.ini", enabled="False")
        before = path.read_text()
        assert apply_smoke_test_overrides(str(path)) == []
        assert path.read_text() == before

    def test_overrides_applied_when_enabled(self, tmp_path, caplog):
        path = _write_config(tmp_path / "c.ini", enabled="True")
        with caplog.at_level("WARNING"):
            applied = apply_smoke_test_overrides(str(path))
        assert "SMOKE TEST MODE" in caplog.text
        cfg = _read(path)
        assert cfg["detector"]["size_y"] == "64"
        assert cfg["detector"]["size_x"] == "1024"  # not overridden
        assert cfg["detector"]["spec_res"] == "20"
        assert cfg["observation"]["N_angles"] == "2"
        assert cfg["new_section"]["key"] == "value"
        # plain keys are limits, not overrides
        assert ("smoke_test", "max_planets", "2") not in applied
        assert len(applied) == 4

    def test_force_applies_overrides_and_marks_config(self, tmp_path):
        path = _write_config(tmp_path / "c.ini", enabled="False")
        apply_smoke_test_overrides(str(path), force=True)
        cfg = _read(path)
        assert cfg["detector"]["size_y"] == "64"
        assert smoke_test_enabled(cfg)

    def test_limits_default_when_disabled_and_parsed_when_enabled(self, tmp_path):
        off = _read(_write_config(tmp_path / "off.ini", enabled="False"))
        assert smoke_test_limits(off) == SMOKE_TEST_LIMIT_DEFAULTS
        on = _read(_write_config(tmp_path / "on.ini", enabled="True"))
        assert smoke_test_limits(on) == {
            "max_qe_values": 1,
            "max_planets": 2,
            "write_full_frame_example": False,
        }
        # forced on (e.g. --smoke) reads the same limits
        assert smoke_test_limits(off, force=True)["max_planets"] == 2

    def test_limits_from_load_config_style_dict(self):
        config = {
            "smoke_test": {
                "enabled": "True",
                "max_qe_values": 1.0,  # load_config turns numbers into floats
                "max_planets": 3.0,
                "write_full_frame_example": "False",
            }
        }
        assert smoke_test_enabled(config)
        assert smoke_test_limits(config) == {
            "max_qe_values": 1,
            "max_planets": 3,
            "write_full_frame_example": False,
        }

    def test_main_config_smoke_section_is_off_and_consistent(self):
        cfg = _read(MAIN_CONFIG)
        assert not smoke_test_enabled(cfg), "commit main_config.ini with smoke off"
        # every override targets an existing section of the main config
        for key in cfg["smoke_test"]:
            if "." in key:
                assert cfg.has_section(key.split(".", 1)[0]), key


@pytest.mark.smoke
def test_smoke_run_writes_small_cubes_quickly(tmp_path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(MAIN_CONFIG)
    model_file = cfg["astrophysical_sources_library"]["exoplanet_model_10pc"]
    if not Path(model_file).exists():
        pytest.skip(f"model planet spectrum not found: {model_file}")
    cfg["system_options"]["systems_2_look_at"] = "read_in_one_planet_spectrum"
    cfg["tasks"]["generate_sims"] = "True"
    cfg["tasks"]["calculate_s2n_post_rotation"] = "False"
    config_path = tmp_path / "main_config.ini"
    with open(config_path, "w", encoding="utf-8") as f:
        cfg.write(f)
    # output_root given: it is used as this run's root (no extra run_<...>/ level)
    out_dir = tmp_path / "out"

    t_start = time.monotonic()
    ok = run_single_calculation(
        config_path=str(config_path),
        base_filename="smoke",
        sources_to_include=["star", "exoplanet_model_10pc"],
        qe=0.6,
        plot=False,
        output_root=str(out_dir),
        smoke_test=True,
    )
    elapsed_s = time.monotonic() - t_start

    assert ok
    products = out_dir / OUTPUT_SUBDIRS[COUNTS_TIME]
    intermediate = out_dir / OUTPUT_SUBDIRS[INTERMEDIATE]
    cubes = sorted(products.glob("extracted_cube_qe_*_dc_*.fits"))
    assert len(cubes) == 1  # one DC value in smoke mode
    # nothing is written loose in the run root
    assert not [p for p in out_dir.iterdir() if p.is_file() and p.suffix != ".ini"]
    n_angles = int(cfg["smoke_test"]["observation.N_angles"])
    n_read = int(cfg["smoke_test"]["observation.N_int_per_angle"])
    expected_bins = int(np.floor(np.log(18.0 / 4.0) / np.log(1.0 + 1.0 / 20.0)))
    with fits.open(cubes[0]) as hdul:
        assert hdul[0].data.shape == (expected_bins, n_angles * n_read, 4)
    assert not list(intermediate.glob("readouts_full_example_*.fits"))
    assert len(list(intermediate.glob("readouts_angle_*.fits"))) == n_angles
    assert elapsed_s < 120, f"smoke run took {elapsed_s:.0f} s"
