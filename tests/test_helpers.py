"""
Unit tests for modules.utils.helpers.
"""

import configparser
import pytest
import modules.utils.helpers as helpers
import numpy as np
from astropy import units as u
import pandas as pd
import astropy.constants as const

from modules.utils.helpers import (
    _config_get,
    _config_set_plot_title_context,
    _get_plot_title_context,
    build_system_params_title,
    ensure_plot_title_context,
    format_plot_title,
    get_sweep_range, 
    validate_file_path,
    generate_star_spectrum,
    generate_planet_bb_spectrum,
    generate_zodiacal_spectrum,
    generate_exozodiacal_spectrum,
    create_sample_data,
)


class TestConfigHelpers:
    def test_config_get_from_configparser(self):
        cfg = configparser.ConfigParser()
        cfg.add_section("telescope")
        cfg.set("telescope", "collecting_area", "25.0")
        assert _config_get(cfg, "telescope", "collecting_area") == "25.0"
        assert _config_get(cfg, "telescope", "missing", "x") == "x"
        assert _config_get(cfg, "missing", "k", "d") == "d"

    def test_config_get_from_dict(self):
        cfg = {"telescope": {"collecting_area": "25.0"}}
        assert _config_get(cfg, "telescope", "collecting_area") == "25.0"
        assert _config_get(cfg, "telescope", "missing", "x") == "x"
        assert _config_get(cfg, "missing", "k", "d") == "d"

    def test_config_set_plot_title_context_configparser(self):
        cfg = configparser.ConfigParser()
        _config_set_plot_title_context(cfg, "hello")
        assert cfg.has_section("plotting")
        assert cfg["plotting"]["title_context"] == "hello"

    def test_config_set_plot_title_context_dict(self):
        cfg = {}
        _config_set_plot_title_context(cfg, "hello")
        assert cfg["plotting"]["title_context"] == "hello"

    def test_config_set_plot_title_context_ignores_empty(self):
        cfg = {"plotting": {"title_context": "keep"}}
        _config_set_plot_title_context(cfg, "")
        assert cfg["plotting"]["title_context"] == "keep"

    def test_get_plot_title_context_configparser_and_dict(self):
        cfg_p = configparser.ConfigParser()
        cfg_p.add_section("plotting")
        cfg_p.set("plotting", "title_context", " abc ")
        assert _get_plot_title_context(cfg_p) == "abc"

        cfg_d = {"plotting": {"title_context": " xyz "}}
        assert _get_plot_title_context(cfg_d) == "xyz"


class TestTitleBuilders:
    def test_build_system_params_title_contains_expected_lines(self):
        cfg = {
            "telescope": {"collecting_area": "25.0", "eta_t": "0.05"},
            "nulling": {"null": True, "nulling_factor": "1e-5"},
            "target": {
                "z_exozodiacal": "1",
                "A_albedo": "0.22",
                "L_star": "1",
                "rad_star": "1",
                "T_star": "5778",
                "rad_planet": "1",
                "pl_temp": "288",
                "distance": "10",
                "lambda_rel_lon_los": "135", 
                "beta_lat_los": "45",
            },
        }
        out = build_system_params_title(cfg)
        assert "collecting area = 25.00 m^2" in out
        assert "telescope throughput = 0.05" in out
        assert "nulling transmission = 1.0e-05" in out
        assert "distance = 10.0 pc" in out
        assert "pl_temp = 288.0 K" in out

    def test_ensure_plot_title_context_reuses_existing(self):
        cfg = {"plotting": {"title_context": "already here"}}
        out = ensure_plot_title_context(cfg)
        assert out == "already here"
        assert cfg["plotting"]["title_context"] == "already here"

    def test_ensure_plot_title_context_builds_and_sets_when_missing(self):
        cfg = {
            "telescope": {"collecting_area": "25.0"},
            "target": {"distance": "10"},
        }
        out = ensure_plot_title_context(cfg)
        assert "collecting area = 25.00 m^2" in out
        assert "distance = 10.0 pc" in out
        assert cfg["plotting"]["title_context"] == out

    def test_format_plot_title_with_and_without_context(self):
        cfg = {"plotting": {"title_context": "line1\nline2"}}
        out = format_plot_title("My Title", cfg)
        assert out.startswith("My Title\n========")
        assert "line1" in out and "line2" in out

        no_ctx = format_plot_title("My Title", {})
        assert no_ctx == "My Title"

        ctx_only = format_plot_title("", cfg)
        assert ctx_only == "line1\nline2"


class TestSweepRange:
    def test_get_sweep_range_includes_stop_with_step_extension(self, monkeypatch):
        monkeypatch.setattr(helpers.ipdb, "set_trace", lambda: None)
        obs = {
            "n_int_start": "1000",
            "n_int_stop": "3000",
            "n_int_step": "1000",
        }
        out = get_sweep_range(obs, "n_int")
        assert out == [1000.0, 2000.0, 3000.0]

    def test_get_sweep_range_non_divisible_stop_behavior(self, monkeypatch):
        monkeypatch.setattr(helpers.ipdb, "set_trace", lambda: None)
        obs = {
            "qe_start": "0.1",
            "qe_stop": "0.35",
            "qe_step": "0.1",
        }
        out = get_sweep_range(obs, "qe")
        # np.arange(start, stop + step, step) includes values up to <= stop+step
        assert out == [0.1, 0.2, 0.30000000000000004, 0.4]


def test_validate_file_path_cases(tmp_path):
    # 1) Missing path -> False
    missing = tmp_path / "missing.txt"
    assert validate_file_path(missing) is False
    # 2) Directory path -> False (exists but not a file)
    d = tmp_path / "adir"
    d.mkdir()
    assert validate_file_path(d) is False
    # 3) Empty file -> False (size == 0)
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    assert validate_file_path(empty_file) is False
    # 4) Non-empty file -> True
    nonempty_file = tmp_path / "data.txt"
    nonempty_file.write_text("hello")
    assert validate_file_path(nonempty_file) is True


class TestGenerateStarSpectrum:
    def _make_config(self, rad_star: float = 1.0, t_star: float = 5778.0):
        cfg = configparser.ConfigParser()
        cfg.add_section("target")
        cfg.set("target", "rad_star", str(rad_star))
        cfg.set("target", "T_star", str(t_star))
        return cfg

    def test_generate_star_spectrum_units_shape_and_positive_values(self):
        cfg = self._make_config(rad_star=1.0, t_star=5778.0)
        wavelength_um = np.array([1.0, 2.0, 5.0, 10.0]) * u.um

        luminosity_photons_star, luminosity_energy_star = generate_star_spectrum(
            config=cfg, wavelength_um=wavelength_um, plot=False
        )

        assert luminosity_photons_star.shape == wavelength_um.shape
        assert luminosity_energy_star.shape == wavelength_um.shape
        assert luminosity_photons_star.unit.is_equivalent(u.ph / (u.um * u.s))
        assert luminosity_energy_star.unit.is_equivalent(u.W / u.um)
        assert np.all(np.isfinite(luminosity_photons_star.value))
        assert np.all(np.isfinite(luminosity_energy_star.value))
        assert np.all(luminosity_photons_star.value > 0)
        assert np.all(luminosity_energy_star.value > 0)

    def test_generate_star_spectrum_scales_with_radius_squared(self):
        wavelength_um = np.array([1.5, 3.0, 6.0]) * u.um
        cfg_r1 = self._make_config(rad_star=1.0, t_star=5000.0)
        cfg_r2 = self._make_config(rad_star=2.0, t_star=5000.0)

        lph_r1, len_r1 = generate_star_spectrum(cfg_r1, wavelength_um, plot=False)
        lph_r2, len_r2 = generate_star_spectrum(cfg_r2, wavelength_um, plot=False)

        # Luminosity should scale with stellar surface area: L ~ R^2
        assert np.allclose((lph_r2 / lph_r1).value, 4.0)
        assert np.allclose((len_r2 / len_r1).value, 4.0)


class TestGeneratePlanetBBSpectrum:
    def _make_config(self, rad_planet: float = 1.0, pl_temp: float = 288.0):
        cfg = configparser.ConfigParser()
        cfg.add_section("target")
        cfg.set("target", "planet_source", "BB")
        cfg.set("target", "rad_planet", str(rad_planet))
        cfg.set("target", "pl_temp", str(pl_temp))
        return cfg

    def test_generate_planet_bb_spectrum_units_shape_and_positive_values(self):
        cfg = self._make_config(rad_planet=1.0, pl_temp=288.0)
        wavelength_um = np.array([4.0, 8.0, 12.0, 16.0]) * u.um

        luminosity_photons_planet, luminosity_energy_planet = generate_planet_bb_spectrum(
            config=cfg, wavelength_um=wavelength_um, plot=False
        )

        assert luminosity_photons_planet.shape == wavelength_um.shape
        assert luminosity_energy_planet.shape == wavelength_um.shape
        assert luminosity_photons_planet.unit.is_equivalent(u.ph / (u.um * u.s))
        assert luminosity_energy_planet.unit.is_equivalent(u.W / u.um)
        assert np.all(np.isfinite(luminosity_photons_planet.value))
        assert np.all(np.isfinite(luminosity_energy_planet.value))
        assert np.all(luminosity_photons_planet.value > 0)
        assert np.all(luminosity_energy_planet.value > 0)

    def test_generate_planet_bb_spectrum_scales_with_radius_squared(self):
        wavelength_um = np.array([5.0, 10.0, 15.0]) * u.um
        cfg_r1 = self._make_config(rad_planet=1.0, pl_temp=300.0)
        cfg_r2 = self._make_config(rad_planet=2.0, pl_temp=300.0)

        lph_r1, len_r1 = generate_planet_bb_spectrum(cfg_r1, wavelength_um, plot=False)
        lph_r2, len_r2 = generate_planet_bb_spectrum(cfg_r2, wavelength_um, plot=False)

        # Luminosity should scale with emitting area: L ~ R^2
        assert np.allclose((lph_r2 / lph_r1).value, 4.0)
        assert np.allclose((len_r2 / len_r1).value, 4.0)


class TestGenerateZodiacalSpectrum:
    def _make_config(self, tau_opt: float = 4e-8):
        cfg = configparser.ConfigParser()
        cfg.add_section("target")
        cfg.set("target", "tau_opt_zodiacal", str(tau_opt))
        cfg.set("target", "T_eff_zodiacal", "265.0")
        cfg.set("target", "A_albedo", "0.22")
        cfg.add_section("telescope")
        cfg.set("telescope", "single_mirror_diameter", "2.0")
        cfg.add_section("observation")
        cfg.set("target", "lambda_rel_lon_los", "135")
        cfg.set("target", "beta_lat_los", "45")
        return cfg

    def test_generate_zodiacal_spectrum_units_shape_and_finite(self):
        cfg = self._make_config(tau_opt=4e-8)
        wavelength_um = np.array([5.0, 10.0, 20.0]) * u.um

        photons_um_s_m2, energy_W_um_m2, energy_MJy_sr = generate_zodiacal_spectrum(
            config=cfg, wavelength_um=wavelength_um, plot=False
        )

        assert photons_um_s_m2.shape == wavelength_um.shape
        assert energy_W_um_m2.shape == wavelength_um.shape
        assert photons_um_s_m2.unit.is_equivalent(u.ph / (u.s * u.um * u.m**2))
        assert energy_W_um_m2.unit.is_equivalent(u.W / (u.um * u.m**2))
        assert np.all(np.isfinite(photons_um_s_m2.value))
        assert np.all(np.isfinite(energy_W_um_m2.value))
        assert np.all(photons_um_s_m2.value >= 0)
        assert np.all(energy_W_um_m2.value >= 0)

        # be within physical limits (see Dannert+ 2022, Fig. C1)
        background_05um = energy_MJy_sr[0].to(u.MJy / u.sr).value
        background_10um = energy_MJy_sr[1].to(u.MJy / u.sr).value
        background_20um = energy_MJy_sr[2].to(u.MJy / u.sr).value
        assert 0.3 < background_05um < 0.6
        assert 10 < background_10um < 16
        assert 10 < background_20um < 50

        # photons should match energy via E_photon = hc/lambda
        photons_from_energy = (
            energy_W_um_m2 * (wavelength_um / (const.h * const.c)) * u.ph
        ).to(u.ph / (u.s * u.um * u.m**2))
        assert np.allclose(photons_um_s_m2.value, photons_from_energy.value, rtol=1e-3)

        # third output is pre-FOV MJy/sr, so remove FOV from energy first
        single_mirror_diameter = float(cfg["telescope"]["single_mirror_diameter"]) * u.m
        hfov = (wavelength_um.to(u.m) / (2.0 * single_mirror_diameter)) * u.rad
        threshold_ampl = 1e-2
        radius_fov_effective = (4.0 / np.pi) * hfov * np.sqrt(-np.log(threshold_ampl))
        fov_effective = (np.pi * radius_fov_effective**2).to(u.sr)
        energy_surface = (energy_W_um_m2 / fov_effective).to(u.W / (u.um * u.m**2 * u.sr))
        energy_mjy_from_energy = ((energy_surface * wavelength_um**2) / const.c).to(u.MJy / u.sr)
        assert np.allclose(energy_MJy_sr.value, energy_mjy_from_energy.value, rtol=1e-3)


    def test_generate_zodiacal_spectrum_scales_linearly_with_tau_opt(self):
        wavelength_um = np.array([6.0, 12.0, 18.0]) * u.um
        cfg_lo = self._make_config(tau_opt=4e-8)
        cfg_hi = self._make_config(tau_opt=8e-8)

        ph_lo, en_lo, en_MJy_sr_lo = generate_zodiacal_spectrum(cfg_lo, wavelength_um, plot=False)
        ph_hi, en_hi, en_MJy_sr_hi = generate_zodiacal_spectrum(cfg_hi, wavelength_um, plot=False)

        # Model is linear in tau_opt, so doubling tau_opt doubles outputs.
        assert np.allclose((ph_hi / ph_lo).value, 2.0)
        assert np.allclose((en_hi / en_lo).value, 2.0)


class TestGenerateExozodiacalSpectrum:
    def _make_config(self, z_exozodiacal: float = 1.0, l_star: float = 1.0):
        cfg = configparser.ConfigParser()
        cfg.add_section("target")
        cfg.set("target", "z_exozodiacal", str(z_exozodiacal))
        cfg.set("target", "L_star", str(l_star))
        return cfg

    def test_generate_exozodiacal_spectrum_units_shape_and_finite(self):
        cfg = self._make_config(z_exozodiacal=1.0, l_star=1.0)
        wavelength_um = np.array([5.0, 10.0, 20.0]) * u.um

        photons, energy = generate_exozodiacal_spectrum(
            config=cfg, wavelength_um=wavelength_um, plot=False
        )

        assert photons.shape == wavelength_um.shape
        assert energy.shape == wavelength_um.shape
        assert photons.unit.is_equivalent(u.ph / (u.s * u.um))
        assert energy.unit.is_equivalent(u.W / u.um)
        assert np.all(np.isfinite(photons.value))
        assert np.all(np.isfinite(energy.value))
        assert np.all(photons.value >= 0)
        assert np.all(energy.value >= 0)

    def test_generate_exozodiacal_spectrum_scales_linearly_with_zodi(self):
        wavelength_um = np.array([6.0, 12.0, 18.0]) * u.um
        cfg_lo = self._make_config(z_exozodiacal=1.0, l_star=1.0)
        cfg_hi = self._make_config(z_exozodiacal=2.0, l_star=1.0)

        ph_lo, en_lo = generate_exozodiacal_spectrum(cfg_lo, wavelength_um, plot=False)
        ph_hi, en_hi = generate_exozodiacal_spectrum(cfg_hi, wavelength_um, plot=False)

        # Model is linear in number of zodis (z), so doubling z doubles output.
        assert np.allclose((ph_hi / ph_lo).value, 2.0)
        assert np.allclose((en_hi / en_lo).value, 2.0)


class TestCreateSampleData:
    def _make_config(self, data_dir: str):
        cfg = configparser.ConfigParser()
        cfg.add_section("dirs")
        cfg.set("dirs", "data_dir", data_dir)
        return cfg

    def test_create_sample_data_writes_expected_files(self, tmp_path, monkeypatch):
        cfg = self._make_config(str(tmp_path / "data"))
        n = 100
        wavelength_um = np.logspace(-1, 1.4, n) * u.um
        photons_um_s = np.ones(n) * u.ph / (u.um * u.s)
        energy_w_um = np.ones(n) * u.W / u.um
        photons_um_s_m2 = np.ones(n) * u.ph / (u.um * u.s * u.m**2)
        energy_w_um_m2 = np.ones(n) * u.W / (u.um * u.m**2)

        monkeypatch.setattr(
            helpers,
            "generate_star_spectrum",
            lambda config, wavelength_um, plot=False: (photons_um_s, energy_w_um),
        )
        monkeypatch.setattr(
            helpers,
            "generate_planet_bb_spectrum",
            lambda config, wavelength_um, plot=False: (2 * photons_um_s, 2 * energy_w_um),
        )
        monkeypatch.setattr(
            helpers,
            "generate_exozodiacal_spectrum",
            lambda config, wavelength_um, plot=False: (3 * photons_um_s, 3 * energy_w_um),
        )
        monkeypatch.setattr(
            helpers,
            "generate_zodiacal_spectrum",
            lambda config, wavelength_um, plot=False: (photons_um_s_m2, energy_w_um_m2),
        )

        create_sample_data(cfg, overwrite=True, plot=False)

        out_dir = (tmp_path / "data").resolve()
        expected_files = [
            "star_spectrum.txt",
            "exoplanet_bb_spectrum.txt",
            "exozodiacal_spectrum.txt",
            "zodiacal_spectrum.txt",
        ]
        for name in expected_files:
            path = out_dir / name
            assert path.exists()
            content = path.read_text()
            assert content.startswith("# wavelength_unit=um\n# luminosity_photons_unit=")
            # Read data rows (header starts at line 3 because first two are comments)
            df = pd.read_csv(path, sep=",", header=2)
            assert "wavel" in df.columns
            assert "luminosity_photons" in df.columns
            assert len(df) == n

