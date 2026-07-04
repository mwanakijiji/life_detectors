"""
Unit tests for modules.utils.helpers.
"""

import configparser
from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.constants as const
import numpy as np
import pandas as pd
import pytest
import yaml
from astropy import units as u
from astropy.table import QTable

import modules.utils.helpers as helpers
from modules.utils.helpers import (
    _config_get,
    _config_set_plot_title_context,
    _get_plot_title_context,
    _normalize_output_root,
    apply_output_root_override,
    build_astrophysical_sources_to_use_title,
    build_observation_detector_title,
    build_system_params_title,
    compute_collecting_area_m2,
    create_sample_data,
    ensure_plot_title_context,
    format_plot_title,
    generate_exozodiacal_spectrum,
    generate_planet_bb_spectrum,
    generate_star_spectrum,
    generate_zodiacal_spectrum,
    get_sweep_range,
    merge_psg_spectra_to_planet_population,
    modify_config_file_sweep,
    parse_sky_position_arcsec_yx,
    plot_planet_population_sample,
    record_info_at_angle_and_qe,
    validate_file_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
APERTURE_YAML = REPO_ROOT / "sim_pipeline/config/aperture_array_double_bracewell.yaml"


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
        assert "stellar nulling = True, nulling transmission = 1.0e-05" in out
        assert "distance = 10.0 pc" in out
        assert "pl_temp = 288.0 K" in out

    def test_build_observation_detector_title_contains_expected_lines(self):
        cfg = {
            "observation": {
                "t_int_frame": "10",
                "N_angles": "4",
                "N_int_per_angle": "5",
            },
            "detector": {
                "quantum_efficiency": "0.8",
                "dark_current": "0.0, 0.1",
                "read_noise": "5",
                "gain": "2",
                "spec_res": "20",
            },
        }
        out = build_observation_detector_title(cfg)
        assert "t_int_frame = 10.0 s" in out
        assert "N_angles = 4" in out
        assert "N_int_per_angle = 5" in out
        assert "t_int_total = 200 s" in out
        assert "QE = 0.8" in out
        assert "dark current sweep = 0.0, 0.1 e-/pix/s" in out
        assert "read noise = 5 e- rms" in out
        assert "gain = 2.0 e-/ADU" in out
        assert "spec_res R = 20.0" in out

    def test_ensure_plot_title_context_reuses_existing(self):
        cfg = {"plotting": {"title_context": "already here"}}
        out = ensure_plot_title_context(cfg)
        assert out == "already here"
        assert cfg["plotting"]["title_context"] == "already here"

    def test_ensure_plot_title_context_builds_and_sets_when_missing(self):
        cfg = {
            "telescope": {"collecting_area": "25.0"},
            "target": {"distance": "10"},
            "observation": {
                "t_int_frame": "1",
                "N_angles": "1",
                "N_int_per_angle": "1",
            },
            "detector": {"quantum_efficiency": "0.8"},
        }
        out = ensure_plot_title_context(cfg)
        assert "collecting area = 25.00 m^2" in out
        assert "distance = 10.0 pc" in out
        assert "QE = 0.8" in out
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

    def test_format_plot_title_two_column_astrophysical_sources(self):
        cfg = {
            "plotting": {"title_context": "collecting area = 25.00 m^2\ntelescope throughput = 0.05"},
            "astrophysical_sources_to_use": {
                "star": True,
                "exoplanet_bb": False,
                "zodiacal": False,
            },
        }
        out = format_plot_title("My Title", cfg)
        assert "collecting area = 25.00 m^2" in out
        assert "  star = True" in out
        assert "  exoplanet_bb = False" in out
        assert "  zodiacal = False" in out
        # Left and right columns share a row when both have content.
        assert "collecting area = 25.00 m^2    astrophysical sources:" in out

    def test_build_astrophysical_sources_to_use_title(self):
        cfg = {
            "astrophysical_sources_to_use": {
                "star": "True",
                "exozodiacal": "False",
            }
        }
        out = build_astrophysical_sources_to_use_title(cfg)
        assert out.splitlines()[0] == "astrophysical sources:"
        assert "  star = True" in out
        assert "  exozodiacal = False" in out

        assert build_astrophysical_sources_to_use_title({}) == ""


class TestSweepRange:
    def test_get_sweep_range_half_open_interval(self):
        obs = {
            "n_int_start": "1000",
            "n_int_stop": "3000",
            "n_int_step": "1000",
        }
        out = get_sweep_range(obs, "n_int")
        assert out == [1000.0, 2000.0]

    def test_get_sweep_range_non_divisible_stop_excludes_stop(self):
        obs = {
            "qe_start": "0.1",
            "qe_stop": "0.35",
            "qe_step": "0.1",
        }
        out = get_sweep_range(obs, "qe")
        assert out == [0.1, 0.2, 0.30000000000000004]


class TestParseSkyPosition:
    def test_parse_valid_position(self):
        assert parse_sky_position_arcsec_yx("1.5, -2.0") == (1.5, -2.0)

    def test_parse_rejects_wrong_number_of_values(self):
        with pytest.raises(ValueError, match="Expected two comma-separated values"):
            parse_sky_position_arcsec_yx("1.0")


class TestComputeCollectingAreaM2:
    def test_matches_aperture_yaml_and_mirror_diameter(self):
        with open(APERTURE_YAML) as f:
            n_apertures = len(yaml.safe_load(f)["apertures"])
        diameter_m = 2.0
        config = {
            "telescope": {
                "aperture_array_config_file_name": str(APERTURE_YAML),
                "single_mirror_diameter": str(diameter_m),
            }
        }
        expected = n_apertures * np.pi * (0.5 * diameter_m) ** 2
        assert compute_collecting_area_m2(config) == pytest.approx(expected)


class TestMergePsgSpectra:
    def test_left_joins_psg_response_files_by_planet_id(self, tmp_path):
        (tmp_path / "psg_cfg_00000015.response").write_text("dummy")
        (tmp_path / "psg_cfg_00000042.response").write_text("dummy")
        df = pd.DataFrame({"id": [15, 42, 99]})
        params = {"dir_file_name_psg_spectra": {"dir_name": str(tmp_path)}}

        merged = merge_psg_spectra_to_planet_population(df, params)

        assert len(merged) == 3
        assert merged.loc[merged["id"] == 15, "abs_file_name_psg_spectrum"].notna().all()
        assert merged.loc[merged["id"] == 42, "abs_file_name_psg_spectrum"].notna().all()
        assert merged.loc[merged["id"] == 99, "abs_file_name_psg_spectrum"].isna().all()


class TestPlotPlanetPopulationSample:
    @patch("modules.utils.helpers.plt.close")
    def test_saves_scatter_matrix_for_small_population(self, _mock_close, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out_path = tmp_path / "scatter.png"

        plot_planet_population_sample(df, ["a", "b"], str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestApplyOutputRootOverride:
    def test_normalize_output_root_adds_trailing_separator(self, tmp_path):
        normalized = _normalize_output_root(str(tmp_path))
        assert normalized.endswith("/")
        assert Path(normalized.rstrip("/")).resolve() == tmp_path.resolve()

    def test_apply_output_root_override_updates_config_file(self, tmp_path):
        config_path = tmp_path / "main_config.ini"
        output_root = tmp_path / "run_out"
        cfg = configparser.ConfigParser()
        cfg.add_section("dirs")
        cfg.set("dirs", "save_s2n_data_unique_dir", "/old/path/")
        with open(config_path, "w") as f:
            cfg.write(f)

        result_path = apply_output_root_override(str(config_path), str(output_root))

        assert result_path == str(config_path)
        updated = configparser.ConfigParser()
        updated.read(config_path)
        saved = updated["dirs"]["save_s2n_data_unique_dir"]
        assert saved.startswith(str(output_root.resolve()))
        assert saved.endswith("/")
        assert output_root.exists()

    def test_apply_output_root_override_noop_when_missing(self, tmp_path):
        config_path = tmp_path / "main_config.ini"
        config_path.write_text("[dirs]\nsave_s2n_data_unique_dir = /keep/\n")
        assert apply_output_root_override(str(config_path), None) == str(config_path)


class TestModifyConfigFileSweep:
    @pytest.fixture
    def minimal_config_path(self, tmp_path):
        config_path = tmp_path / "main_config.ini"
        cfg = configparser.ConfigParser()
        cfg.add_section("detector")
        cfg.set("detector", "quantum_efficiency", "0.8")
        cfg.set("detector", "read_noise", "6")
        with open(config_path, "w") as f:
            cfg.write(f)
        return str(config_path)

    def test_writes_temp_config_under_parameter_sweeps(self, minimal_config_path):
        result = modify_config_file_sweep(minimal_config_path, qe=0.87)

        assert Path(result).is_file()
        assert "parameter_sweeps" in result
        assert result.endswith("main_config_temp_qe0p87.ini")

    def test_updates_quantum_efficiency_in_output_file(self, minimal_config_path):
        result = modify_config_file_sweep(minimal_config_path, qe=0.65)

        updated = configparser.ConfigParser()
        updated.read(result)
        assert updated.get("detector", "quantum_efficiency") == "0.65"
        assert updated.get("detector", "read_noise") == "6"

    def test_appends_run_id_to_filename_when_provided(self, minimal_config_path):
        result = modify_config_file_sweep(
            minimal_config_path, qe=0.7, run_id="run42"
        )

        assert result.endswith("main_config_temp_qe0p70_run42.ini")


class TestRecordInfoAtAngleAndQe:
    def test_writes_hdf5_groups_for_each_dark_current(self, tmp_path):
        save_dir = str(tmp_path) + "/"
        n_bins = 2
        wavel = np.array([5.0, 10.0]) * u.um
        width = np.array([0.5, 0.5]) * u.um

        def _table(signal):
            return QTable(
                {
                    "wavel_bin_center": wavel,
                    "wavel_bin_width": width,
                    "signal_adu": np.ones(n_bins) * signal * u.adu,
                }
            )

        channel = MagicMock()
        channel.tables_by_dark_current = {0.0: _table(1.0), 0.1: _table(2.0)}
        output_channels = {
            "output_1_bright": channel,
            "output_2_bright": channel,
            "output_3_dark": channel,
            "output_4_dark": channel,
        }
        post_chop = {
            0.0: _table(3.0),
            0.1: _table(4.0),
        }
        post_chop[0.0]["chopped_planet_adu"] = np.ones(n_bins) * u.adu
        post_chop[0.1]["chopped_planet_adu"] = np.ones(n_bins) * u.adu

        record_info_at_angle_and_qe(
            angle_deg=45.0,
            qe=0.7,
            output_channels=output_channels,
            post_chop_tables_by_dark_current=post_chop,
            save_dir=save_dir,
            plot=False,
        )

        hdf5_path = tmp_path / "angle_45.hdf5"
        assert hdf5_path.exists()
        restored = QTable.read(hdf5_path, path="dc_0_qe_0.70/output_1_bright")
        assert len(restored) == n_bins
        assert restored.meta["angle_deg"] == 45.0
        assert restored.meta["qe"] == 0.7


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

    def test_generate_planet_bb_spectrum_physical_sense_10pc(self):
        cfg = self._make_config(rad_planet=1.0, pl_temp=300.0)
        wavelength_um = np.array([5.0, 10.0, 15.0]) * u.um
        # photons/s/um; W/um
        lph_5_um, len_5_um = generate_planet_bb_spectrum(cfg, wavelength_um=5.0 * u.um, plot=False)
        lph_10_um, len_10_um = generate_planet_bb_spectrum(cfg, wavelength_um=10.0 * u.um, plot=False)
        lph_15_um, len_15_um = generate_planet_bb_spectrum(cfg, wavelength_um=15.0 * u.um, plot=False)
        assert np.allclose(float(f"{lph_5_um.value:.2e}"), float(1.05e35), rtol=1e-2)
        assert np.allclose(float(f"{len_5_um.value:.2e}"), float(4.17e15), rtol=1e-2)
        assert np.allclose(float(f"{lph_10_um.value:.2e}"), float(8.01e35), rtol=1e-2)
        assert np.allclose(float(f"{len_10_um.value:.2e}"), float(1.59e16), rtol=1e-2)
        assert np.allclose(float(f"{lph_15_um.value:.2e}"), float(8.09e35), rtol=1e-2)
        assert np.allclose(float(f"{len_15_um.value:.2e}"), float(1.07e16), rtol=1e-2)


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
        cfg.set("dirs", "save_s2n_data_unique_dir", data_dir)
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
            lambda config, wavelength_um, plot=False: (
                photons_um_s_m2,
                energy_w_um_m2,
                np.ones(n) * u.MJy / u.sr,
            ),
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

        for source, filename in helpers.GENERATED_SPECTRA_FILENAMES.items():
            assert cfg.get("astrophysical_sources_library", source) == str(out_dir / filename)

