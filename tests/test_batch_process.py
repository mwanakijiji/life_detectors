"""
Unit tests for batch_process module.
"""

import configparser
import os
import sys
import types

import pytest

# Mock ipdb before importing batch_process (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from batch_process import modify_config_file_sweep, modify_config_file_pl_system_params


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
        config.set("observation", "integration_time", "100")
        config.set("observation", "lambda_rel_lon_los", "135")
        config.set("observation", "beta_lat_los", "45")
        config.set("observation", "n_int", "1000")
        config.add_section("detector")
        config.set("detector", "quantum_efficiency", "0.8")
        config.set("detector", "read_noise", "6")
        config.set("detector", "gain", "4.5")
        config.set("detector", "spec_res", "20")
        with open(config_path, "w") as f:
            config.write(f)
        return str(config_path)

    def test_returns_temp_config_path(self, minimal_config_path):
        # Temp config file is created and path is returned.
        result = modify_config_file_sweep(minimal_config_path, n_int=5000, qe=0.87)
        assert os.path.isfile(result)
        assert result.endswith(".ini")

    def test_temp_path_format(self, minimal_config_path):
        # Temp path has expected format: .../parameter_sweeps/{basename}_temp_n{n_int}_qe{qe_str}.ini.
        result = modify_config_file_sweep(minimal_config_path, n_int=5000, qe=0.87)
        assert "parameter_sweeps" in result
        assert "_temp_n5000_qe0p87.ini" in result

    def test_n_int_updated_in_output(self, minimal_config_path):
        # Output config has updated n_int value.
        result = modify_config_file_sweep(minimal_config_path, n_int=25920, qe=0.8)
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("observation", "n_int") == "25920"

    def test_qe_updated_in_output(self, minimal_config_path):
        # Output config has updated quantum_efficiency value.
        result = modify_config_file_sweep(minimal_config_path, n_int=1000, qe=0.65)
        config = configparser.ConfigParser()
        config.read(result)
        assert config.get("detector", "quantum_efficiency") == "0.65"

    def test_qe_string_format_in_filename(self, minimal_config_path):
        # QE decimal is converted to 'p' in filename (e.g. 0.87 -> 0p87).
        result = modify_config_file_sweep(minimal_config_path, n_int=1000, qe=0.1)
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
        config.set("observation", "integration_time", "100")
        config.set("observation", "lambda_rel_lon_los", "135")
        config.set("observation", "beta_lat_los", "45")
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
        assert config.get("observation", "lambda_rel_lon_los") == "0.5"
        assert config.get("observation", "beta_lat_los") == "0.3"

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
