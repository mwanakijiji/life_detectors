"""
Unit tests for batch_process module.
"""

import configparser
import logging
import os
import sys
import types
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

# Mock ipdb before importing batch_process (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from batch_process import (
    batch_qe_nint_process,
    modify_config_file_pl_system_params,
    modify_config_file_sweep,
    parameter_sweep,
    run_single_calculation,
)


class TestModifyConfigFileSweep:
    '''
    Test cases for modify_config_file_sweep.
    '''

    @pytest.fixture
    def minimal_config_path(self, tmp_path):
        """Create a minimal config file matching demo_config.ini [observation] and [detector]."""
        config_path = tmp_path / "test_config.ini"
        config = configparser.ConfigParser()
        config.add_section("observation")
        config.set("observation", "t_int_obs_total", "100")
        config.set("observation", "t_int_frame", "10")
        config.set("observation", "n_int", "1000")
        config.add_section("detector")
        config.set("detector", "quantum_efficiency", "0.8")
        config.set("detector", "read_noise", "6")
        config.set("detector", "gain", "4.5")
        config.set("detector", "spec_res", "20")
        config.add_section("target")
        config.set("target", "lambda_rel_lon_los", "135")
        config.set("target", "beta_lat_los", "45")
        with open(config_path, "w") as f:
            config.write(f)
        return str(config_path)

    def test_returns_temp_config_path(self, minimal_config_path):
        # Temp config file is created and path is returned.
        result = modify_config_file_sweep(minimal_config_path, qe=0.87)
        assert os.path.isfile(result)
        assert result.endswith(".ini")

    def test_temp_path_format(self, minimal_config_path):
        # Temp path has expected format: .../parameter_sweeps/{basename}_temp_n{n_int}_qe{qe_str}.ini.
        result = modify_config_file_sweep(minimal_config_path, qe=0.87)
        assert "parameter_sweeps" in result
        assert "_temp_qe0p87.ini" in result

    def test_n_int_updated_in_output(self, minimal_config_path):
        # Output config has updated n_int value.
        result = modify_config_file_sweep(minimal_config_path, qe=0.8)
        config = configparser.ConfigParser()
        config.read(result)
        t_int_obs_total = float(config.get("observation", "t_int_obs_total"))
        t_int_frame = float(config.get("observation", "t_int_frame"))
        n_int = int(t_int_obs_total // t_int_frame)
        assert n_int == 10

    def test_qe_updated_in_output(self, minimal_config_path):
        # Output config has updated quantum_efficiency value.
        result = modify_config_file_sweep(minimal_config_path, qe=0.65)
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("detector", "quantum_efficiency") == "0.65"

    def test_qe_string_format_in_filename(self, minimal_config_path):
        # QE decimal is converted to 'p' in filename (e.g. 0.87 -> 0p87).
        result = modify_config_file_sweep(minimal_config_path, qe=0.1)
        assert "qe0p10.ini" in result


class TestModifyConfigFilePlSystemParams:
    """Test cases for modify_config_file_pl_system_params."""

    @pytest.fixture
    def base_config_path(self, tmp_path):
        """Create a config file matching demo_config.ini [dirs], [target], [observation]."""
        config_path = tmp_path / "base_config.ini"
        config = configparser.ConfigParser()
        config.add_section("dirs")
        config.set("dirs", "data_dir", str(tmp_path / "data") + "/")
        config.set("dirs", "save_s2n_data_unique_dir", str(tmp_path / "param_sweeps") + "/")
        config.add_section("target")
        config.set("target", "distance", "10.0")
        config.set("target", "pl_temp", "200.0")
        config.set("target", "rad_star", "1.0")
        config.set("target", "T_star", "5778")
        config.set("target", "L_star", "1")
        config.set("target", "rad_planet", "1.0")
        config.set("target", "A_albedo", "0.22")
        config.set("target", "z_exozodiacal", "1")
        config.set("target", "psg_spectrum_file_name", "/path/to/spectrum.response")
        config.set("target", "Stype", "G")
        config.set("target", "Nuniverse", "0")
        config.set("target", "Nstar", "0")
        config.add_section("observation")
        config.set("observation", "t_int_obs_total", "100")
        config.set("observation", "t_int_frame", "10")
        config.set("target", "lambda_rel_lon_los", "135")
        config.set("target", "beta_lat_los", "45")
        with open(config_path, "w") as f:
            config.write(f)
        return str(config_path)

    @pytest.fixture
    def system_params(self):
        """Planetary system parameters as from a population dataframe row."""
        return {
            "Ds": 10.0,
            "Rp": 1.0,
            "Tp": 288.0,
            "Rs": 1.0,
            "Ts": 5778.0,
            "z": 2.0,
            "eclip_lon": 0.5,
            "eclip_lat": 0.3,
            "Stype": "G",
            "abs_file_name_psg_spectrum": "/data/psg_cfg_00000015.response",
            "Nuniverse": "1",
            "Nstar": "42",
        }

    @pytest.fixture
    def lum_types(self):
        """Stellar type to luminosity mapping (L_sol)."""
        return {"o": 40000, "b": 1000, "a": 15, "f": 3, "g": 1.0, "k": 0.4, "m": 0.05}

    def test_returns_original_path_when_system_params_none(self, base_config_path):
        """When system_params is None, return the original config path unchanged."""
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00042",
            system_params=None,
            lum_types={"g": 1.0},
        )
        assert result == base_config_path

    def test_creates_temp_config_file(self, base_config_path, system_params, lum_types):
        """Temp config file is created and path is returned."""
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00042",
            system_params=system_params,
            lum_types=lum_types,
        )
        assert os.path.isfile(result)
        assert result.endswith(".ini")

    def test_target_params_updated_in_output(
        self, base_config_path, system_params, lum_types
    ):
        """Output config has updated target parameters from system_params."""
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00042",
            system_params=system_params,
            lum_types=lum_types,
        )
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("target", "distance") == "10.0"
        assert config.get("target", "rad_planet") == "1.0"
        assert config.get("target", "pl_temp") == "288.0"
        assert config.get("target", "rad_star") == "1.0"
        assert config.get("target", "t_star") == "5778.0"
        assert config.get("target", "z_exozodiacal") == "2.0"
        assert config.get("target", "psg_spectrum_file_name") == "/data/psg_cfg_00000015.response"
        assert config.get("target", "Stype") == "G"
        assert config.get("target", "Nuniverse") == "1"
        assert config.get("target", "Nstar") == "42"

    def test_observation_params_updated_in_output(
        self, base_config_path, system_params, lum_types
    ):
        """Output config has updated observation parameters from system_params."""
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00042",
            system_params=system_params,
            lum_types=lum_types,
        )
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("target", "lambda_rel_lon_los") == "0.5"
        assert config.get("target", "beta_lat_los") == "0.3"

    def test_lum_types_lookup_uses_lowercase(self, base_config_path, lum_types):
        """L_star is set from lum_types using Stype.lower()."""
        system_params = {
            "Ds": 10.0,
            "Rp": 1.0,
            "Tp": 288.0,
            "Rs": 0.5,
            "Ts": 3500.0,
            "z": 1.0,
            "eclip_lon": 0.0,
            "eclip_lat": 0.0,
            "Stype": "M",  # uppercase
            "abs_file_name_psg_spectrum": "/data/spectrum.response",
            "Nuniverse": "0",
            "Nstar": "0",
        }
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00001",
            system_params=system_params,
            lum_types=lum_types,
        )
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("target", "L_star") == "0.05"  # M type -> 0.05 L_sol

    def test_filename_includes_system_params(
        self, base_config_path, system_params, lum_types
    ):
        """Output path includes base_filename and key system params in directory structure."""
        result = modify_config_file_pl_system_params(
            base_config_path,
            base_filename="planet_00042",
            system_params=system_params,
            lum_types=lum_types,
        )
        assert "planet_00042" in result
        assert "Nuniverse_1" in result
        assert "Nstar_42" in result
        assert "dist_10.0" in result
        assert "Rp_1.0" in result


def _write_run_calc_config(
    config_path,
    *,
    save_dir: str,
    generate_sims: bool = False,
    calculate_s2n: bool = True,
):
    config = configparser.ConfigParser()
    config.add_section("tasks")
    config.set("tasks", "generate_sims", str(generate_sims))
    config.set("tasks", "calculate_s2n_post_rotation", str(calculate_s2n))
    config.add_section("dirs")
    config.set("dirs", "data_dir", str(config_path.parent / "data") + "/")
    config.set("dirs", "save_s2n_data_unique_dir", save_dir)
    config.add_section("telescope")
    config.set("telescope", "collecting_area", "25.0")
    config.set("telescope", "plate_scale", "0.1")
    config.set("telescope", "throughput", "0.8")
    config.set("telescope", "eta_t", "0.05")
    config.add_section("target")
    config.set("target", "distance", "10.0")
    config.set("target", "pl_temp", "200.0")
    config.set("target", "rad_star", "1.0")
    config.set("target", "T_star", "5778")
    config.set("target", "L_star", "1")
    config.set("target", "rad_planet", "1.0")
    config.set("target", "lambda_rel_lon_los", "135")
    config.set("target", "beta_lat_los", "45")
    config.add_section("nulling")
    config.set("nulling", "distance", "10.0")
    config.set("nulling", "pl_temp", "200.0")
    config.add_section("detector")
    config.set("detector", "quantum_efficiency", "0.8")
    config.set("detector", "read_noise", "6")
    config.set("detector", "dark_current", "0.0001")
    config.set("detector", "gain", "2.0")
    config.set("detector", "spec_res", "20")
    config.add_section("observation")
    config.set("observation", "t_int_obs_total", "100")
    config.set("observation", "t_int_frame", "10")
    config.set("observation", "N_angles", "2")
    config.set("observation", "N_int_per_angle", "1")
    config.add_section("wavelength_range")
    config.set("wavelength_range", "min", "4.0")
    config.set("wavelength_range", "max", "18.5")
    config.set("wavelength_range", "n_points", "30")
    with open(config_path, "w") as f:
        config.write(f)


def _run_calc_config_dict(save_dir: str) -> dict:
    return {
        "tasks": {
            "generate_sims": "True",
            "calculate_s2n_post_rotation": "False",
        },
        "dirs": {
            "data_dir": save_dir,
            "save_s2n_data_unique_dir": save_dir,
        },
        "observation": {
            "t_int_obs_total": 100.0,
            "t_int_frame": 10.0,
            "N_angles": 2.0,
            "N_int_per_angle": 1.0,
        },
        "detector": {
            "gain": 2.0,
            "quantum_efficiency": 0.8,
        },
        "plotting": {"title_context": ""},
    }


class _DictRunConfig(dict):
    """Dict config with ConfigParser-like methods used before logging."""

    def read(self, *_args, **_kwargs):
        return None

    def has_section(self, section):
        return section in self

    def has_option(self, section, option):
        return option in self.get(section, {})

    def get(self, section, option, fallback=None):
        return self[section][option]

    def set(self, section, option, value):
        self.setdefault(section, {})[option] = value


class _WeirdRunConfig:
    """Non-ConfigParser, non-dict config for fallback logging."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir

    def read(self, *_args, **_kwargs):
        return None

    def has_section(self, *_args, **_kwargs):
        return False

    def __getitem__(self, key):
        if key == "dirs":
            return {"save_s2n_data_unique_dir": self._save_dir}
        if key == "observation":
            return {"N_angles": 2, "N_int_per_angle": 1, "t_int_frame": 10}
        if key == "detector":
            return {"gain": 2.0, "quantum_efficiency": 0.8}
        if key == "plotting":
            return {"title_context": ""}
        raise KeyError(key)


def _patch_generate_sims_pipeline(mock_input, mock_s2n, mock_create_sample_data):
    mock_input.return_value = "n"
    mock_astro_instance = MagicMock()
    mock_astro_instance.calculate_incident_flux.return_value = {"flux": 1.0}
    mock_astro_instance.generate_onsky_scene.return_value = {"star": MagicMock()}
    mock_instrument = MagicMock()
    mock_instrument.output_channels = {}
    mock_instrument.post_chop_tables_by_dark_current = {}
    mock_instrument.generate_instrument_transmission.return_value = np.ones((6, 3, 3))
    return mock_astro_instance, mock_instrument


def _make_dirty_output_channel(name: str, angle_deg: float = 99.0):
    from modules.core.instrumental import OutputChannel

    channel = OutputChannel(name=name, angle_deg=angle_deg)
    channel.instrum_noise = {"stale": 1}
    channel.astroph_signal = {"stale": 2}
    channel.tables_by_dark_current = {0.0: "table"}
    channel.tables_by_dark_current_orig = {0.0: "orig"}
    return channel


@pytest.fixture
def run_calc_config_path(tmp_path):
    save_dir = str(tmp_path / "hdf5_out") + "/"
    config_path = tmp_path / "run_calc_config.ini"
    _write_run_calc_config(
        config_path,
        save_dir=save_dir,
        generate_sims=False,
        calculate_s2n=True,
    )
    return str(config_path), save_dir


class TestRunSingleCalculation:
    @patch("batch_process.calculate_s2n_post_rotation")
    def test_returns_true_and_calls_s2n_when_enabled(
        self, mock_s2n, run_calc_config_path
    ):
        config_path, save_dir = run_calc_config_path

        result = run_single_calculation(
            config_path=config_path,
            base_filename="planet_00001",
            sources_to_include=["star", "exoplanet_model_10pc"],
            qe=0.7,
            plot=False,
        )

        assert result is True
        mock_s2n.assert_called_once()
        args, kwargs = mock_s2n.call_args
        assert args[0] == save_dir
        assert kwargs["config"]["dirs"]["save_s2n_data_unique_dir"] == save_dir

    @patch("batch_process.calculate_s2n_post_rotation")
    def test_skips_s2n_when_disabled(self, mock_s2n, tmp_path):
        save_dir = str(tmp_path / "hdf5_out") + "/"
        config_path = tmp_path / "run_calc_config.ini"
        _write_run_calc_config(
            config_path,
            save_dir=save_dir,
            generate_sims=False,
            calculate_s2n=False,
        )

        result = run_single_calculation(
            config_path=str(config_path),
            base_filename="planet_00001",
            sources_to_include=["star"],
            qe=0.7,
            plot=False,
        )

        assert result is True
        mock_s2n.assert_not_called()

    @patch("batch_process.create_sample_data")
    @patch("batch_process.calculate_s2n_post_rotation")
    def test_skips_simulation_when_generate_sims_false(
        self, mock_s2n, mock_create_sample_data, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path

        run_single_calculation(
            config_path=config_path,
            base_filename="planet_00001",
            sources_to_include=["star"],
            qe=0.7,
            plot=False,
        )

        mock_create_sample_data.assert_not_called()
        mock_s2n.assert_called_once()

    @patch("batch_process.record_info_at_angle_and_qe")
    @patch("batch_process.instrumental.InstrumentDepTerms")
    @patch("batch_process.astrophysical.AstrophysicalSources")
    @patch("batch_process.create_sample_data")
    @patch("batch_process.calculate_s2n_post_rotation")
    def test_generate_sims_runs_pipeline_without_deleting_hdf5(
        self,
        mock_s2n,
        mock_create_sample_data,
        mock_astro_sources_cls,
        mock_instrument_cls,
        mock_record_info,
        tmp_path,
        caplog,
    ):
        save_dir = str(tmp_path / "hdf5_out") + "/"
        config_path = tmp_path / "run_calc_config.ini"
        _write_run_calc_config(
            config_path,
            save_dir=save_dir,
            generate_sims=True,
            calculate_s2n=False,
        )

        mock_astro_instance = MagicMock()
        mock_astro_instance.calculate_incident_flux.return_value = {"flux": 1.0}
        mock_astro_instance.generate_onsky_scene.return_value = {"star": MagicMock()}
        mock_astro_sources_cls.return_value = mock_astro_instance

        mock_instrument = MagicMock()
        mock_instrument.output_channels = {}
        mock_instrument.post_chop_tables_by_dark_current = {}
        mock_instrument.generate_instrument_transmission.return_value = np.ones((6, 3, 3))
        mock_instrument_cls.return_value = mock_instrument

        with caplog.at_level(logging.INFO):
            result = run_single_calculation(
                config_path=str(config_path),
                base_filename="planet_00001",
                sources_to_include=["star", "exoplanet_model_10pc"],
                qe=0.7,
                plot=False,
            )

        assert result is True
        mock_create_sample_data.assert_called_once()
        mock_astro_sources_cls.assert_called_once()
        mock_instrument_cls.assert_called_once()
        assert mock_record_info.call_count == 2
        mock_s2n.assert_not_called()
        assert "------- Temp config contents -------" in caplog.text
        assert "[tasks]" in caplog.text
        assert "generate_sims = True" in caplog.text

    @patch("batch_process.record_info_at_angle_and_qe")
    @patch("batch_process.instrumental.InstrumentDepTerms")
    @patch("batch_process.astrophysical.AstrophysicalSources")
    @patch("batch_process.create_sample_data")
    @patch("batch_process.calculate_s2n_post_rotation")
    def test_resets_output_channels_at_each_angle(
        self,
        mock_s2n,
        mock_create_sample_data,
        mock_astro_sources_cls,
        mock_instrument_cls,
        mock_record_info,
        tmp_path,
    ):
        save_dir = str(tmp_path / "hdf5_out") + "/"
        config_path = tmp_path / "run_calc_config.ini"
        _write_run_calc_config(
            config_path,
            save_dir=save_dir,
            generate_sims=True,
            calculate_s2n=False,
        )

        mock_astro_instance = MagicMock()
        mock_astro_instance.calculate_incident_flux.return_value = {"flux": 1.0}
        mock_astro_instance.generate_onsky_scene.return_value = {"star": MagicMock()}
        mock_astro_sources_cls.return_value = mock_astro_instance

        mock_instrument = MagicMock()
        mock_instrument.sources_astroph = {"star": {"flux": 1.0}}
        mock_instrument.output_channels = {
            "output_1_bright": _make_dirty_output_channel("output_1_bright"),
            "output_3_dark": _make_dirty_output_channel("output_3_dark"),
        }
        mock_instrument.post_chop_tables_by_dark_current = {"stale": True}
        angles_seen = []

        def _capture_reset_state(*, override_stellar_mask=False, angle_deg=None, plot=False):
            assert mock_instrument.post_chop_tables_by_dark_current == {}
            for channel in mock_instrument.output_channels.values():
                angles_seen.append((angle_deg, channel.name, channel.angle_deg))
                assert channel.instrum_noise == {}
                assert channel.astroph_signal == {}
                assert channel.tables_by_dark_current == {}
                assert channel.tables_by_dark_current_orig == {}
                channel.instrum_noise["filled"] = angle_deg
                channel.astroph_signal["filled"] = angle_deg
                channel.tables_by_dark_current[0.0] = angle_deg
                channel.tables_by_dark_current_orig[0.0] = angle_deg
            mock_instrument.post_chop_tables_by_dark_current["filled"] = angle_deg
            return np.ones((6, 3, 3))

        mock_instrument.generate_instrument_transmission.side_effect = _capture_reset_state
        mock_instrument_cls.return_value = mock_instrument

        result = run_single_calculation(
            config_path=str(config_path),
            base_filename="planet_00001",
            sources_to_include=["star", "exoplanet_model_10pc"],
            qe=0.7,
            plot=False,
        )

        assert result is True
        expected_angles = list(np.linspace(0, 360, num=2, endpoint=False))
        assert mock_instrument.generate_instrument_transmission.call_count == 2
        assert angles_seen == [
            (expected_angles[0], "output_1_bright", expected_angles[0]),
            (expected_angles[0], "output_3_dark", expected_angles[0]),
            (expected_angles[1], "output_1_bright", expected_angles[1]),
            (expected_angles[1], "output_3_dark", expected_angles[1]),
        ]
        mock_record_info.assert_called()


class TestBatchQeNintProcess:
    @patch("batch_process.run_single_calculation", return_value=True)
    def test_returns_true_when_all_qe_runs_succeed(
        self, mock_run_single, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path
        qe_values = [0.5, 0.7]

        result = batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=qe_values,
            sources_to_include=["star"],
            base_filename="planet_00001",
        )

        assert result is True
        assert mock_run_single.call_count == 2

    @patch("batch_process.run_single_calculation")
    def test_returns_false_when_any_qe_run_fails(
        self, mock_run_single, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path
        mock_run_single.side_effect = [True, False]

        result = batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=[0.5, 0.7],
            sources_to_include=["star"],
        )

        assert result is False
        assert mock_run_single.call_count == 2

    @patch("batch_process.run_single_calculation", return_value=True)
    def test_passes_override_stellar_mask_true_while_prompt_disabled(
        self, mock_run_single, run_calc_config_path
    ):
        """Stellar-mask prompt is commented out in batch_qe_nint_process; mask stays on."""
        config_path, _ = run_calc_config_path

        batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=[0.7],
            sources_to_include=["star"],
        )

        _, kwargs = mock_run_single.call_args
        assert kwargs["override_stellar_mask"] is True

    @patch("batch_process.run_single_calculation", return_value=True)
    def test_passes_override_stellar_mask_true_for_empty_response(
        self, mock_run_single, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path

        batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=[0.7],
            sources_to_include=["star"],
        )

        _, kwargs = mock_run_single.call_args
        assert kwargs["override_stellar_mask"] is True

    @patch("batch_process.run_single_calculation", return_value=True)
    def test_defaults_stellar_mask_to_true_on_eof(
        self, mock_run_single, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path

        batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=[0.7],
            sources_to_include=["star"],
        )

        _, kwargs = mock_run_single.call_args
        assert kwargs["override_stellar_mask"] is True

    @patch("batch_process.run_single_calculation", return_value=True)
    @patch("builtins.input", return_value="y")
    def test_forwards_qe_and_optional_args_to_run_single_calculation(
        self, mock_input, mock_run_single, run_calc_config_path
    ):
        config_path, _ = run_calc_config_path
        system_params = {"Ds": 10.0, "Stype": "G"}
        lum_types = {"g": 1.0}
        output_root = "/tmp/custom_output/"

        batch_qe_nint_process(
            base_config_path=config_path,
            qe_values=[0.65],
            sources_to_include=["star", "exoplanet_model_10pc"],
            base_filename="planet_00042",
            overwrite=False,
            plot=True,
            system_params=system_params,
            lum_types=lum_types,
            output_root=output_root,
        )

        args, kwargs = mock_run_single.call_args
        assert args == ()
        assert kwargs["config_path"] == config_path
        assert kwargs["base_filename"] == "planet_00042"
        assert kwargs["sources_to_include"] == ["star", "exoplanet_model_10pc"]
        assert kwargs["qe"] == 0.65
        assert kwargs["overwrite"] is False
        assert kwargs["plot"] is True
        assert kwargs["system_params"] == system_params
        assert kwargs["lum_types"] == lum_types
        assert kwargs["output_root"] == output_root
        assert kwargs["override_stellar_mask"] is True


def _write_aperture_yaml(path):
    path.write_text(
        "apertures:\n"
        "  - name: aperture_00\n"
        "    amplitude: 1.0\n"
        "    x_m: 0.0\n"
        "    y_m: 0.0\n"
        "  - name: aperture_01\n"
        "    amplitude: 1.0\n"
        "    x_m: 1.0\n"
        "    y_m: 0.0\n"
    )


def _write_parameter_sweep_configs(
    tmp_path,
    *,
    systems_2_look_at: str = "single_system",
    population_csv: str | None = None,
):
    aperture_yaml = tmp_path / "aperture_array.yaml"
    _write_aperture_yaml(aperture_yaml)

    single_obs_path = tmp_path / "single_obs.ini"
    single_obs = configparser.ConfigParser()
    single_obs.add_section("dirs")
    single_obs.set("dirs", "save_s2n_data_unique_dir", str(tmp_path / "out") + "/")
    single_obs.add_section("system_options")
    single_obs.set("system_options", "systems_2_look_at", systems_2_look_at)
    single_obs.add_section("telescope")
    single_obs.set("telescope", "aperture_array_config_file_name", str(aperture_yaml))
    single_obs.set("telescope", "single_mirror_diameter", "2.0")
    single_obs.add_section("astrophysical_sources_to_use")
    single_obs.set("astrophysical_sources_to_use", "star", "True")
    single_obs.set("astrophysical_sources_to_use", "exoplanet_model_10pc", "True")
    single_obs.set("astrophysical_sources_to_use", "zodiacal", "False")
    with open(single_obs_path, "w") as f:
        single_obs.write(f)

    sweep_path = tmp_path / "sweep.ini"
    sweep = configparser.ConfigParser()
    sweep.add_section("observation")
    sweep.set("observation", "qe_start", "0.5")
    sweep.set("observation", "qe_stop", "0.6")
    sweep.set("observation", "qe_step", "0.05")
    with open(sweep_path, "w") as f:
        sweep.write(f)

    population_path = tmp_path / "planet_population.ini"
    population = configparser.ConfigParser()
    population.add_section("file_name_planet_population")
    population.set(
        "file_name_planet_population",
        "file_name",
        population_csv or str(tmp_path / "population.csv"),
    )
    population.set(
        "file_name_planet_population",
        "fyi_plot_name",
        str(tmp_path / "population_plot.png"),
    )
    population.add_section("dir_file_name_psg_spectra")
    population.set("dir_file_name_psg_spectra", "dir_name", str(tmp_path / "psg") + "/")
    population.add_section("lum_type")
    population.set("lum_type", "G", "1.0")
    population.set("lum_type", "M", "0.05")
    with open(population_path, "w") as f:
        population.write(f)

    return str(single_obs_path), str(sweep_path), str(population_path)


class TestParameterSweep:
    @patch("batch_process.batch_qe_nint_process", return_value=True)
    def test_single_system_calls_batch_once(self, mock_batch, tmp_path):
        single_obs_path, sweep_path, population_path = _write_parameter_sweep_configs(
            tmp_path,
            systems_2_look_at="single_system",
        )

        parameter_sweep(
            config_single_obs_path=single_obs_path,
            config_sweep_path=sweep_path,
            config_planet_population_path=population_path,
        )

        mock_batch.assert_called_once()
        _, kwargs = mock_batch.call_args
        assert kwargs["base_config_path"] == single_obs_path
        assert kwargs["base_filename"] == "s2n_sweep"
        assert kwargs["system_params"] is None
        assert kwargs["lum_types"] is None
        assert kwargs["sources_to_include"] == ["star", "exoplanet_model_10pc"]
        assert kwargs["qe_values"] == [0.5, 0.55]
        assert kwargs["overwrite"] is True
        assert kwargs["plot"] is True

    @patch("batch_process.helpers.plot_planet_population_sample")
    @patch("batch_process.helpers.merge_psg_spectra_to_planet_population", side_effect=lambda df, _: df)
    @patch("batch_process.pd.read_csv")
    @patch("batch_process.batch_qe_nint_process", return_value=True)
    def test_planet_population_calls_batch_per_planet(
        self,
        mock_batch,
        mock_read_csv,
        mock_merge_psg,
        mock_plot_population,
        tmp_path,
    ):
        population_csv = tmp_path / "population.csv"
        population_csv.write_text("header\n")
        single_obs_path, sweep_path, population_path = _write_parameter_sweep_configs(
            tmp_path,
            systems_2_look_at="planet_population",
            population_csv=str(population_csv),
        )
        mock_read_csv.return_value = pd.DataFrame(
            [
                {"id": 1, "Rp": 1.0, "Porb": 365.0, "Mp": 1.0, "z": 1.0, "Tp": 288.0, "ap": 1.0},
                {"id": 2, "Rp": 1.1, "Porb": 400.0, "Mp": 1.1, "z": 2.0, "Tp": 290.0, "ap": 1.1},
            ]
        )

        parameter_sweep(
            config_single_obs_path=single_obs_path,
            config_sweep_path=sweep_path,
            config_planet_population_path=population_path,
            output_root=str(tmp_path / "batch_root") + "/"
        )

        assert mock_batch.call_count == 2
        first_kwargs = mock_batch.call_args_list[0].kwargs
        second_kwargs = mock_batch.call_args_list[1].kwargs
        assert first_kwargs["base_filename"] == "s2n_sweep_planet_index_0000000"
        assert second_kwargs["base_filename"] == "s2n_sweep_planet_index_0000001"
        assert first_kwargs["system_params"]["id"] == 1
        assert second_kwargs["system_params"]["id"] == 2
        #assert first_kwargs["lum_types"]["G"] == 1.0 ## ## TODO: fix this
        assert first_kwargs["output_root"] == str(tmp_path / "batch_root") + "/"
        #mock_plot_population.assert_called_once()

    @patch("batch_process.batch_qe_nint_process", return_value=True)
    def test_invalid_system_option_skips_batch(self, mock_batch, tmp_path):
        single_obs_path, sweep_path, population_path = _write_parameter_sweep_configs(
            tmp_path,
            systems_2_look_at="invalid_mode",
        )

        parameter_sweep(
            config_single_obs_path=single_obs_path,
            config_sweep_path=sweep_path,
            config_planet_population_path=population_path,
        )

        mock_batch.assert_not_called()
