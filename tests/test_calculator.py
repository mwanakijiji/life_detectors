"""
Unit tests for the main noise calculator.
"""

import pytest
import numpy as np
import configparser
from types import SimpleNamespace
from unittest.mock import patch
from astropy import units as u

from modules.core.calculator import NoiseCalculator

class TestNoiseCalculator:
    """Test cases for NoiseCalculator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "telescope": {
                "collecting_area": 25.0,
                "plate_scale": 0.1,
                "throughput": 0.8,
            },
            "target": {
                "distance": 10.0,
                "nulling_factor": 0.01,
            },
            "detector": {
                "read_noise": 5.0,
                "dark_current": 0.1,
                "gain": 2.0,
            },
            "observation": {
                "t_int_obs_total": 3600,
                "t_int_frame": 10,
            },
            "astrophysical_sources": {
                "star": {
                    "spectrum_file": "test_star.txt",
                    "enabled": True,
                },
                "exoplanet": {
                    "spectrum_file": "test_exoplanet.txt",
                    "enabled": True,
                },
                "exozodiacal": {
                    "spectrum_file": "test_exozodiacal.txt",
                    "enabled": True,
                },
                "zodiacal": {
                    "spectrum_file": "test_zodiacal.txt",
                    "enabled": True,
                },
            },
            "instrumental_sources": {
                "dark_current": {
                    "enabled": True,
                },
                "read_noise": {
                    "enabled": True,
                },
            },
            "wavelength_range": {
                "min": 1.0,
                "max": 10.0,
                "n_points": 100,
            },
        }
    
    def test_init(self, mock_config, sources_all=['star', 'exoplanet_bb'], sources_to_include=['star', 'exoplanet_bb']):
        """Test initialization of NoiseCalculator."""
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config, sources_all, sources_to_include)
            assert calculator.config == mock_config
    

    def test_config_imported_correctly(self, mock_config, sources_all=['star', 'exoplanet_bb'], sources_to_include=['star', 'exoplanet_bb']):
        """Test wavelength grid generation."""
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config, sources_all, sources_to_include)
            
            # check these things exist
            assert calculator.config['wavelength_range']['n_points']
            assert calculator.config['wavelength_range']['min']
            assert calculator.config['wavelength_range']['max']

    def _build_minimal_s2n_inputs(self):
        """Shared minimal config/sources for s2n_val tests."""
        config = {
            "observation": {"n_int": "10", "t_int_obs_total": "100", "t_int_frame": "10"},
            "nulling": {"nulling_factor": "1e-5"},
            "detector": {"quantum_efficiency": "0.8"},
            "telescope": {"eta_t": "0.05"},
        }

        wavel_grid = np.array([5.0, 10.0, 15.0]) * u.um
        flux_grid = np.array([1.0, 2.0, 3.0]) * u.electron / (u.um * u.s)
        prop_dict = {
            "exoplanet_bb": {"wavel": wavel_grid, "flux_e_sec_um": flux_grid},
            "exoplanet_model_10pc": {"wavel": wavel_grid, "flux_e_sec_um": 2.0 * flux_grid},
            "exoplanet_psg": {"wavel": wavel_grid, "flux_e_sec_um": 3.0 * flux_grid},
            "star": {"wavel": wavel_grid, "flux_e_sec_um": flux_grid},
            "exozodiacal": {"wavel": wavel_grid, "flux_e_sec_um": flux_grid},
            "zodiacal": {"wavel": wavel_grid, "flux_e_sec_um": flux_grid},
        }
        sources_instrum = {
            "read_noise_e_pix-1": np.array([2.0]) * u.electron / u.pix,
            "dark_current_e_pix-1_sec-1": np.array([0.01]) * u.electron / (u.pix * u.s),
            "dark_current_e_pix-1": np.array([1.0]) * u.electron / u.pix,
        }
        sources_all = SimpleNamespace(prop_dict=prop_dict, sources_instrum=sources_instrum)
        del_lambda = np.array([1.0, 1.0, 1.0]) * u.um
        n_pix = np.array([4.0, 4.0, 4.0]) * u.pix
        return config, wavel_grid, sources_all, del_lambda, n_pix

    @pytest.mark.parametrize(
        "sources_to_include",
        [
            ["star", "exoplanet_bb", "exozodiacal", "zodiacal"],
            ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"],
            ["star", "exoplanet_psg", "exozodiacal", "zodiacal"],
        ],
    )
    def test_s2n_val_planet_model_branches_return_quantity(
        self, sources_to_include
    ):
        """Covers exoplanet_bb, exoplanet_model_10pc, and exoplanet_psg branches."""
        config, wavel_grid, sources_all, del_lambda, n_pix = self._build_minimal_s2n_inputs()
        calc = NoiseCalculator(
            config=config,
            sources_all=sources_all,
            sources_to_include=sources_to_include,
        )

        s2n = calc.s2n_val(
            wavel_bin_centers=wavel_grid,
            del_lambda_array=del_lambda,
            n_pix_array=n_pix,
            plot=False,
        )

        assert not hasattr(s2n, "unit") # should be unitless
        assert s2n.shape[-1] == len(wavel_grid)
        assert np.all(np.isfinite(s2n[0]))

    def test_s2n_val_no_planet_model_warns_then_errors(self, caplog):
        """Covers 'No planet model being used' warning branch."""
        config, wavel_grid, sources_all, del_lambda, n_pix = self._build_minimal_s2n_inputs()
        calc = NoiseCalculator(
            config=config,
            sources_all=sources_all,
            sources_to_include=["star", "exozodiacal", "zodiacal"],
        )

        with caplog.at_level("WARNING"):
            with pytest.raises(UnboundLocalError):
                calc.s2n_val(
                    wavel_bin_centers=wavel_grid,
                    del_lambda_array=del_lambda,
                    n_pix_array=n_pix,
                    plot=False,
                )
        assert "No planet model being used" in caplog.text

    def test_s2n_val_two_planet_models_exits(self):
        """Covers error branch when both exoplanet_bb and exoplanet_model_10pc are set."""
        config, wavel_grid, sources_all, del_lambda, n_pix = self._build_minimal_s2n_inputs()
        calc = NoiseCalculator(
            config=config,
            sources_all=sources_all,
            sources_to_include=["star", "exoplanet_bb", "exoplanet_model_10pc", "exozodiacal", "zodiacal"],
        )

        with pytest.raises(SystemExit):
            calc.s2n_val(
                wavel_bin_centers=wavel_grid,
                del_lambda_array=del_lambda,
                n_pix_array=n_pix,
                plot=False,
            )

    def test_s2n_e_writes_fits_and_returns_s2n(self, monkeypatch):
        """Test s2n_e path (excluding plotting) with mocked detector and FITS IO."""
        config = configparser.ConfigParser()
        config["dirs"] = {"save_s2n_data_unique_dir": "/tmp/"}
        config["saving"] = {"save_s2n_data_temp": "/tmp/s2n_temp.fits"}
        config["detector"] = {
            "spec_res": "2.0",
            "quantum_efficiency": "0.8",
            "dark_current": "0.0, 0.2, 0.1",
            "read_noise": "6.0",
            "gain": "4.5",
            "pix_per_wavel_bin": "2.2",
        }
        config["observation"] = {
            "n_int": "10",
            "t_int_obs_total": "100",
            "t_int_frame": "10",
            "lambda_rel_lon_los": "135",
            "beta_lat_los": "45",
        }
        config["wavelength_range"] = {"min": "1.0", "max": "4.0"}
        config["nulling"] = {"null": "True", "nulling_factor": "1e-5"}
        config["telescope"] = {"collecting_area": "25.0", "eta_t": "0.05"}
        config["target"] = {
            "T_star": "5778",
            "rad_star": "1.0",
            "distance": "10.0",
            "pl_temp": "300.0",
            "rad_planet": "1.0",
            "A_albedo": "0.3",
        }

        sources_all = SimpleNamespace(sources_astroph={"star": {"wavel": np.array([1.0, 2.0, 3.0])}})
        calc = NoiseCalculator(config=config, sources_all=sources_all, sources_to_include=["star", "exoplanet_bb"])

        class DummyDetector:
            def __init__(self, config, num_wavel_bins):
                self.num_wavel_bins = num_wavel_bins

            def footprint_spectral(self, file_name_plot, plot):
                return np.ones((self.num_wavel_bins, 2, 2))

        monkeypatch.setattr("modules.core.calculator.Detector", DummyDetector)

        s2n_mock = np.ones((2, 3))
        monkeypatch.setattr(
            calc,
            "s2n_val",
            lambda wavel_bin_centers, del_lambda_array, n_pix_array: s2n_mock,
        )

        class DummyHeader(dict):
            def add_blank(self, *args, **kwargs):
                return None

        class DummyHDU:
            def __init__(self):
                self.header = DummyHeader()
                self.data = None
                self.writes = []

            def writeto(self, path, overwrite=False):
                self.writes.append((path, overwrite))

        hdu_holder = {}

        def fake_primary_hdu():
            hdu = DummyHDU()
            hdu_holder["hdu"] = hdu
            return hdu

        monkeypatch.setattr("modules.core.calculator.fits.PrimaryHDU", fake_primary_hdu)
        monkeypatch.setattr("modules.core.calculator.fits.Card", lambda *args, **kwargs: None)

        out = calc.s2n_e(file_name_fits_unique="/tmp/s2n_unique.fits", plot=False)

        assert np.array_equal(out, s2n_mock)
        hdu = hdu_holder["hdu"]
        assert hdu.writes[0] == ("/tmp/s2n_temp.fits", True)
        assert hdu.writes[1] == ("/tmp/s2n_unique.fits", True)
        assert hdu.data.shape == (4, 2, 3)
