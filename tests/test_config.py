"""
Unit tests for modules.config.loader.
"""

import configparser
import logging
import os
import re
import sys
import types

import pytest

# Mock ipdb before importing config modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.config.loader import load_config, setup_logging
from modules.config.validator import ConfigValidator, validate_config


class TestLoadConfig:
    """Test cases for load_config."""

    @pytest.fixture
    def ini_config_path(self, tmp_path):
        """Create a minimal INI config file matching demo_config.ini structure."""
        config_path = tmp_path / "test_config.ini"
        config = configparser.ConfigParser()
        config.add_section("dirs")
        config.set("dirs", "data_dir", str(tmp_path / "data") + "/")
        config.set("dirs", "save_s2n_data_unique_dir", str(tmp_path / "param_sweeps") + "/")
        config.add_section("telescope")
        config.set("telescope", "collecting_area", "25.0")
        config.set("telescope", "plate_scale", "0.1")
        config.set("telescope", "throughput", "0.8")
        config.set("telescope", "eta_t", "0.05")
        config.set("telescope", "single_mirror_diameter", "2.0")
        config.add_section("target")
        config.set("target", "distance", "10.0")
        config.set("target", "pl_temp", "200.0")
        config.set("target", "rad_star", "1.0")
        config.set("target", "T_star", "5778")
        config.set("target", "L_star", "1")
        config.set("target", "rad_planet", "1.0")
        config.set("target", "A_albedo", "0.22")
        config.set("target", "z_exozodiacal", "1")
        config.set("target", "planet_source", "file")
        config.add_section("detector")
        config.set("detector", "quantum_efficiency", "0.8")
        config.add_section("observation")
        config.set("observation", "integration_time", "100")
        config.set("observation", "lambda_rel_lon_los", "135")
        config.set("observation", "beta_lat_los", "45")
        config.add_section("wavelength_range")
        config.set("wavelength_range", "min", "1.0")
        config.set("wavelength_range", "max", "20.0")
        config.set("wavelength_range", "n_points", "30")
        with open(config_path, "w") as f:
            config.write(f)
        return str(config_path)

    def test_returns_dict_with_sections(self, ini_config_path):
        """Config is loaded as dict with section keys."""
        result = load_config(ini_config_path, makedirs=False)
        assert isinstance(result, dict)
        assert "telescope" in result
        assert "target" in result
        assert "dirs" in result

    def test_numeric_values_converted_to_float(self, ini_config_path):
        """Numeric string values are converted to float."""
        result = load_config(ini_config_path, makedirs=False)
        assert result["telescope"]["collecting_area"] == 25.0
        assert result["telescope"]["plate_scale"] == 0.1
        assert result["telescope"]["eta_t"] == 0.05
        assert result["target"]["distance"] == 10.0
        assert result["target"]["pl_temp"] == 200.0
        assert result["target"]["rad_planet"] == 1.0
        assert result["wavelength_range"]["min"] == 1.0
        assert result["wavelength_range"]["max"] == 20.0

    def test_non_numeric_values_remain_strings(self, tmp_path):
        """Non-numeric values stay as strings."""
        config_path = tmp_path / "config.ini"
        config = configparser.ConfigParser()
        config.add_section("target")
        config.set("target", "planet_source", "file")
        config.set("target", "spectrum_path", "/path/to/spectrum.txt")
        with open(config_path, "w") as f:
            config.write(f)
        result = load_config(str(config_path), makedirs=False)
        assert result["target"]["planet_source"] == "file"
        assert result["target"]["spectrum_path"] == "/path/to/spectrum.txt"

    def test_makedirs_creates_dirs_section_paths(self, ini_config_path, tmp_path):
        """When makedirs=True, paths in [dirs] section are created."""
        load_config(ini_config_path, makedirs=True)
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "param_sweeps").is_dir()

    def test_makedirs_false_does_not_create_dirs(self, ini_config_path, tmp_path):
        """When makedirs=False, directories are not created."""
        load_config(ini_config_path, makedirs=False)
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "param_sweeps").exists()

    def test_empty_config_returns_empty_dict(self, tmp_path):
        """Config file with no sections returns empty dict."""
        config_path = tmp_path / "empty.ini"
        config_path.write_text("")
        result = load_config(str(config_path), makedirs=False)
        assert result == {}


class TestSetupLogging:
    """Test cases for setup_logging."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset root logger handlers after each test to avoid cross-test pollution."""
        yield
        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.WARNING)

    def test_returns_log_file_path(self, tmp_path):
        """Returns path to log file with detectorsim_YYYYMMDD_HHMMSS.log pattern."""
        log_dir = tmp_path / "logs"
        result = setup_logging(str(log_dir))
        assert result.endswith(".log")
        assert "detectorsim_" in result
        assert re.match(r"detectorsim_\d{8}_\d{6}\.log$", os.path.basename(result))

    def test_log_file_created_and_writable(self, tmp_path):
        """Log file is created and messages can be written to it."""
        log_dir = tmp_path / "logs"
        log_path = setup_logging(str(log_dir))
        assert os.path.isfile(log_path)
        logging.getLogger().info("test message")
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(log_path) as f:
            content = f.read()
        assert "test message" in content

    def test_uses_default_log_dir_when_no_arg(self):
        """Uses 'logs' as default when log_dir not provided."""
        result = setup_logging()
        assert "logs" in result
        assert result.endswith(".log")


class TestConfigValidator:
    """Test cases for ConfigValidator."""

    def test_init_sets_required_sections(self):
        """ConfigValidator has expected required_sections."""
        validator = ConfigValidator()
        assert "telescope" in validator.required_sections
        assert "target" in validator.required_sections
        assert "nulling" in validator.required_sections
        assert "detector" in validator.required_sections
        assert "observation" in validator.required_sections
        assert "wavelength_range" in validator.required_sections

    def test_init_sets_required_telescope_fields(self):
        """ConfigValidator has expected required_telescope_fields."""
        validator = ConfigValidator()
        assert validator.required_telescope_fields == [
            "collecting_area",
            "plate_scale",
            "throughput",
        ]

    def test_init_sets_required_target_fields(self):
        """ConfigValidator has expected required_target_fields."""
        validator = ConfigValidator()
        assert validator.required_target_fields == ["distance", "pl_temp"]

    def test_init_sets_required_detector_fields(self):
        """ConfigValidator has expected required_detector_fields."""
        validator = ConfigValidator()
        assert "read_noise" in validator.required_detector_fields
        assert "quantum_efficiency" in validator.required_detector_fields
        assert "spec_res" in validator.required_detector_fields

    def test_init_sets_required_observation_fields(self):
        """ConfigValidator has expected required_observation_fields."""
        validator = ConfigValidator()
        assert validator.required_observation_fields == [
            "integration_time",
            "n_int",
        ]

    def test_init_sets_required_wavelength_fields(self):
        """ConfigValidator has expected required_wavelength_fields."""
        validator = ConfigValidator()
        assert validator.required_wavelength_fields == ["min", "max", "n_points"]

    @pytest.fixture
    def valid_config(self):
        """Config dict that passes validation (matches demo_config structure)."""
        return {
            "telescope": {
                "collecting_area": 25.0,
                "plate_scale": 0.1,
                "throughput": 0.8,
            },
            "target": {"distance": 10.0, "pl_temp": 200.0},
            "nulling": {"null": True, "nulling_factor": 0.00001},
            "detector": {
                "read_noise": 6,
                "dark_current": "0.0, 70., 0.05",
                "gain": 4.5,
                "quantum_efficiency": 0.8,
                "spec_res": 20,
            },
            "observation": {"integration_time": 100, "n_int": 3000},
            "wavelength_range": {"min": 1.0, "max": 20.0, "n_points": 30},
        }

    def test_validate_config_complete_config_runs(self, valid_config):
        """validate_config runs without exception for complete valid config."""
        validator = ConfigValidator()
        validator.validate_config(valid_config)  # no exception

    def test_validate_config_missing_section_logs_error(self, valid_config, caplog):
        """Missing required section logs error."""
        del valid_config["telescope"]
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing required section" in caplog.text
        assert "telescope" in caplog.text

    def test_validate_config_invalid_telescope_value_logs_error(self, valid_config, caplog):
        """Invalid telescope value (<=0) logs error."""
        valid_config["telescope"]["collecting_area"] = -1.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Invalid telescope" in caplog.text or "must be positive" in caplog.text

    def test_validate_config_missing_telescope_field_logs_error(self, valid_config, caplog):
        """Missing telescope field logs error."""
        del valid_config["telescope"]["collecting_area"]
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing telescope field" in caplog.text

    def test_validate_config_missing_target_field_logs_error(self, valid_config, caplog):
        """Missing target field logs error."""
        del valid_config["target"]["distance"]
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing target field" in caplog.text
        assert "distance" in caplog.text

    def test_validate_config_invalid_target_value_logs_error(self, valid_config, caplog):
        """Invalid target value (<=0) logs error."""
        valid_config["target"]["pl_temp"] = 0.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Invalid target" in caplog.text or "must be positive" in caplog.text

    def test_validate_config_target_negative_value_logs_error(self, valid_config, caplog):
        """Target value < 0 logs error."""
        valid_config["target"]["distance"] = -5.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Invalid target" in caplog.text or "must be positive" in caplog.text

    def test_validate_config_target_non_numeric_raises(self, valid_config):
        """Target value that cannot convert to float raises ValueError."""
        valid_config["target"]["pl_temp"] = "not_a_number"
        validator = ConfigValidator()
        with pytest.raises(ValueError):
            validator.validate_config(valid_config)

    def test_validate_config_missing_wavelength_field_logs_error(self, valid_config, caplog):
        """Missing wavelength_range field logs error."""
        del valid_config["wavelength_range"]["min"]
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing wavelength_range field" in caplog.text
        assert "min" in caplog.text

    def test_validate_config_invalid_wavelength_value_logs_error(self, valid_config, caplog):
        """Invalid wavelength_range value (<=0) logs error."""
        valid_config["wavelength_range"]["min"] = 0.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Invalid wavelength_range" in caplog.text or "must be positive" in caplog.text

    def test_validate_config_wavelength_n_points_zero_logs_error(self, valid_config, caplog):
        """wavelength_range n_points <= 0 logs error."""
        valid_config["wavelength_range"]["n_points"] = 0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Invalid wavelength_range" in caplog.text or "must be positive" in caplog.text

    def test_validate_config_wavelength_min_ge_max_logs_error(self, valid_config, caplog):
        """wavelength_range min >= max logs error."""
        valid_config["wavelength_range"]["min"] = 20.0
        valid_config["wavelength_range"]["max"] = 1.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "min must be less than max" in caplog.text

    def test_validate_config_wavelength_min_equals_max_logs_error(self, valid_config, caplog):
        """wavelength_range min == max logs error."""
        valid_config["wavelength_range"]["min"] = 10.0
        valid_config["wavelength_range"]["max"] = 10.0
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "min must be less than max" in caplog.text

    def test_validate_config_missing_detector_field_logs_error(self, valid_config, caplog):
        """Missing detector field logs error."""
        del valid_config["detector"]["quantum_efficiency"]
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing detector field" in caplog.text

    def test_validate_config_astrophysical_sources_missing_source_logs_error(
        self, valid_config, caplog
    ):
        """Missing astrophysical source logs error."""
        valid_config["astrophysical_sources"] = {"star": "/path/star.txt"}
        validator = ConfigValidator()
        with caplog.at_level("ERROR"):
            validator.validate_config(valid_config)
        assert "Missing astrophysical source" in caplog.text

    def test_validate_config_returns_true_for_valid_config(self, valid_config):
        """validate_config returns True when config is valid."""
        assert validate_config(valid_config) is True

    def test_validate_config_returns_false_for_missing_section(self, valid_config):
        """validate_config returns False when required section is missing."""
        del valid_config["telescope"]
        assert validate_config(valid_config) is False

    def test_validate_config_returns_false_for_invalid_value(self, valid_config):
        """validate_config returns False when a value is invalid (e.g. <= 0)."""
        valid_config["telescope"]["collecting_area"] = -1.0
        assert validate_config(valid_config) is False

    def test_validate_config_returns_false_for_missing_field(self, valid_config):
        """validate_config returns False when required field is missing."""
        del valid_config["target"]["distance"]
        assert validate_config(valid_config) is False

    def test_validate_config_returns_false_for_wavelength_min_ge_max(self, valid_config):
        """validate_config returns False when wavelength min >= max."""
        valid_config["wavelength_range"]["min"] = 20.0
        valid_config["wavelength_range"]["max"] = 1.0
        assert validate_config(valid_config) is False

    def test_validate_config_returns_false_for_empty_config(self):
        """validate_config returns False for empty config."""
        assert validate_config({}) is False