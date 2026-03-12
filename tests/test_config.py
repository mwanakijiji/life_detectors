"""
Unit tests for modules.config.loader.
"""

import configparser
import os
import sys
import types

import pytest

# Mock ipdb before importing loader (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.config.loader import load_config


class TestLoadConfig:
    """Test cases for load_config."""

    @pytest.fixture
    def ini_config_path(self, tmp_path):
        """Create a minimal INI config file."""
        config_path = tmp_path / "test_config.ini"
        config = configparser.ConfigParser()
        config.add_section("telescope")
        config.set("telescope", "collecting_area", "25.0")
        config.set("telescope", "plate_scale", "0.1")
        config.add_section("target")
        config.set("target", "distance", "10.0")
        config.set("target", "rad_planet", "1.0")
        config.add_section("dirs")
        config.set("dirs", "data_dir", str(tmp_path / "data") + "/")
        config.set("dirs", "output_dir", str(tmp_path / "output") + "/")
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
        assert result["target"]["distance"] == 10.0
        assert result["target"]["rad_planet"] == 1.0

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
        assert (tmp_path / "output").is_dir()

    def test_makedirs_false_does_not_create_dirs(self, ini_config_path, tmp_path):
        """When makedirs=False, directories are not created."""
        load_config(ini_config_path, makedirs=False)
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "output").exists()

    def test_empty_config_returns_empty_dict(self, tmp_path):
        """Config file with no sections returns empty dict."""
        config_path = tmp_path / "empty.ini"
        config_path.write_text("")
        result = load_config(str(config_path), makedirs=False)
        assert result == {}