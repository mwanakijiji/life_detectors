"""
Unit tests for modules.utils.loader.
"""

import configparser
import logging
import os
import re
import sys
import types

import pytest

# Mock ipdb before importing loader (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.utils.loader import config_getboolean, load_config, setup_logging


class TestConfigGetBoolean:
    def test_configparser_reads_true_and_false(self):
        cfg = configparser.ConfigParser()
        cfg.add_section("nulling")
        cfg.set("nulling", "null", "true")
        cfg.set("nulling", "enabled", "false")

        assert config_getboolean(cfg, "nulling", "null") is True
        assert config_getboolean(cfg, "nulling", "enabled") is False

    def test_configparser_missing_section_or_key_returns_default(self):
        cfg = configparser.ConfigParser()
        cfg.add_section("nulling")
        cfg.set("nulling", "null", "true")

        assert config_getboolean(cfg, "missing", "null") is False
        assert config_getboolean(cfg, "nulling", "missing", default=True) is True

    def test_dict_reads_bool_and_string_values(self):
        cfg = {
            "nulling": {"null": True, "enabled": "false", "verbose": " TRUE "},
        }

        assert config_getboolean(cfg, "nulling", "null") is True
        assert config_getboolean(cfg, "nulling", "enabled") is False
        assert config_getboolean(cfg, "nulling", "verbose") is True

    def test_dict_invalid_value_returns_default(self):
        cfg = {"nulling": {"null": "maybe"}}

        assert config_getboolean(cfg, "nulling", "null", default=True) is True
        assert config_getboolean(cfg, "nulling", "missing") is False


class TestLoadConfig:
    @pytest.fixture
    def ini_config_path(self, tmp_path):
        config_path = tmp_path / "test_config.ini"
        cfg = configparser.ConfigParser()
        cfg.add_section("dirs")
        cfg.set("dirs", "data_dir", str(tmp_path / "data") + "/")
        cfg.add_section("telescope")
        cfg.set("telescope", "collecting_area", "25.0")
        cfg.set("telescope", "eta_t", "0.05")
        cfg.add_section("nulling")
        cfg.set("nulling", "null", "true")
        cfg.set("nulling", "enabled", "false")
        cfg.add_section("target")
        cfg.set("target", "planet_source", "BB")
        with open(config_path, "w") as f:
            cfg.write(f)
        return str(config_path)

    def test_returns_dict_with_typed_values(self, ini_config_path):
        result = load_config(ini_config_path, makedirs=False)

        assert isinstance(result, dict)
        assert result["telescope"]["collecting_area"] == 25.0
        assert result["nulling"]["null"] is True
        assert result["nulling"]["enabled"] is False
        assert result["target"]["planet_source"] == "BB"

    def test_makedirs_creates_dirs_section_paths(self, ini_config_path, tmp_path):
        load_config(ini_config_path, makedirs=True)
        assert (tmp_path / "data").is_dir()

    def test_makedirs_false_skips_directory_creation(self, ini_config_path, tmp_path):
        load_config(ini_config_path, makedirs=False)
        assert not (tmp_path / "data").exists()

    def test_empty_config_returns_empty_dict(self, tmp_path):
        config_path = tmp_path / "empty.ini"
        config_path.write_text("")
        assert load_config(str(config_path), makedirs=False) == {}

    def test_logs_loaded_sections(self, ini_config_path, caplog):
        with caplog.at_level(logging.INFO):
            load_config(ini_config_path, makedirs=False)

        assert "[telescope]" in caplog.text
        assert "collecting_area = 25.0" in caplog.text


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def reset_logging(self):
        yield
        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.WARNING)

    def test_returns_log_file_with_timestamp_and_tag(self, tmp_path):
        log_dir = tmp_path / "logs"
        result = setup_logging(str(log_dir), tag="unit_test")

        basename = os.path.basename(result)
        assert result.endswith(".log")
        assert basename.startswith("detectorsim_")
        assert basename.endswith("_unit_test.log")
        assert re.match(r"detectorsim_\d{8}_\d{6}_unit_test\.log$", basename)

    def test_uses_default_tag_when_not_provided(self, tmp_path):
        log_dir = tmp_path / "logs"
        result = setup_logging(str(log_dir))

        basename = os.path.basename(result)
        assert re.match(
            r"detectorsim_\d{8}_\d{6}_pid\d+_[0-9a-f]{8}\.log$",
            basename,
        )

    def test_log_file_created_and_writable(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_path = setup_logging(str(log_dir), tag="write_test")

        assert os.path.isfile(log_path)
        logging.getLogger().info("loader test message")
        for handler in logging.getLogger().handlers:
            handler.flush()

        with open(log_path) as f:
            assert "loader test message" in f.read()

    def test_creates_log_dir_when_missing(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        setup_logging(str(log_dir), tag="mkdir_test")
        assert log_dir.is_dir()
