"""
Unit tests for the main noise calculator.
"""

import configparser
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

# Mock ipdb before importing project modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.core.calculator import NoiseCalculator


def _geometric_n_bins(lambda_min: float, lambda_max: float, spec_res: float) -> int:
    return int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / spec_res)))


@pytest.fixture
def unit_sources_all():
    return SimpleNamespace(
        sources_astroph={"star": {"wavel": np.linspace(1.0, 4.0, 8)}},
    )


@pytest.fixture
def s2n_e_config():
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
        "N_angles": "10",
        "t_int_obs_total": "100",
        "t_int_frame": "10",
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
        "lambda_rel_lon_los": "135",
        "beta_lat_los": "45",
    }
    config["plotting"] = {"s2n_levels_2d": "1, 5"}
    return config


class DummyDetector:
    instances = []

    def __init__(self, config, num_wavel_bins):
        self.config = config
        self.num_wavel_bins = num_wavel_bins
        self.footprint_calls = []
        self.systematics_calls = 0
        DummyDetector.instances.append(self)

    def footprint_spectral(self, file_name_plot, plot):
        self.footprint_calls.append((file_name_plot, plot))
        return np.ones((self.num_wavel_bins, 2, 2))

    def convert_2d_systematics_to_1d_vector(self):
        self.systematics_calls += 1
        return np.zeros(self.num_wavel_bins)


class TestNoiseCalculator:
    """Test cases for NoiseCalculator class."""

    @pytest.fixture
    def mock_config(self):
        config = configparser.ConfigParser()
        config["wavelength_range"] = {"min": "1.0", "max": "10.0", "n_points": "100"}
        config["target"] = {"distance": "10.0"}
        return config

    def test_init(self, mock_config, unit_sources_all):
        calculator = NoiseCalculator(
            mock_config,
            unit_sources_all,
            sources_to_include=["star", "exoplanet_bb"],
        )
        assert calculator.config is mock_config
        assert calculator.sources_all is unit_sources_all
        assert calculator.sources_to_include == ["star", "exoplanet_bb"]

    def test_config_imported_correctly(self, mock_config, unit_sources_all):
        calculator = NoiseCalculator(
            mock_config,
            unit_sources_all,
            sources_to_include=["star", "exoplanet_bb"],
        )
        assert calculator.config["wavelength_range"]["n_points"] == "100"
        assert calculator.config["wavelength_range"]["min"] == "1.0"
        assert calculator.config["wavelength_range"]["max"] == "10.0"


@pytest.mark.skipif(
    not hasattr(NoiseCalculator, "s2n_val"),
    reason="s2n_val is currently commented out in calculator.py",
)
class TestS2nVal:
    """Legacy tests for the vectorized s2n_val helper (currently disabled in production)."""

    def _build_minimal_s2n_inputs(self):
        config = {
            "observation": {"N_angles": "1", "t_int_obs_total": "100", "t_int_frame": "10"},
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
    @patch("modules.core.calculator.ipdb.set_trace")
    def test_planet_model_branches_return_unitless_array(
        self, _mock_set_trace, sources_to_include
    ):
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
            addl_systematics_vector=np.zeros(len(wavel_grid)),
            plot=False,
        )

        assert isinstance(s2n, np.ndarray)
        assert s2n.shape[-1] == len(wavel_grid)
        assert np.all(np.isfinite(s2n[0]))

    def test_no_planet_model_warns_then_errors(self, caplog):
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

    def test_two_planet_models_exits(self):
        config, wavel_grid, sources_all, del_lambda, n_pix = self._build_minimal_s2n_inputs()
        calc = NoiseCalculator(
            config=config,
            sources_all=sources_all,
            sources_to_include=[
                "star",
                "exoplanet_bb",
                "exoplanet_model_10pc",
                "exozodiacal",
                "zodiacal",
            ],
        )

        with pytest.raises(SystemExit):
            calc.s2n_val(
                wavel_bin_centers=wavel_grid,
                del_lambda_array=del_lambda,
                n_pix_array=n_pix,
                plot=False,
            )


class TestS2nE:
    def test_s2n_e_creates_four_detectors_and_reads_systematics(
        self, monkeypatch, s2n_e_config, unit_sources_all
    ):
        DummyDetector.instances = []
        monkeypatch.setattr("modules.core.calculator.Detector", DummyDetector)

        class DummyHeader(dict):
            def add_blank(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            "modules.core.calculator.fits.PrimaryHDU",
            lambda: SimpleNamespace(
                header=DummyHeader(),
                data=None,
                writeto=lambda *args, **kwargs: None,
            ),
        )
        monkeypatch.setattr("modules.core.calculator.fits.Card", lambda *args, **kwargs: None)

        calc = NoiseCalculator(
            config=s2n_e_config,
            sources_all=unit_sources_all,
            sources_to_include=["star", "exoplanet_bb"],
        )
        calc.s2n_e(file_name_fits_unique="/tmp/s2n_unique.fits", plot=False)

        assert len(DummyDetector.instances) == 4
        for detector in DummyDetector.instances:
            assert len(detector.footprint_calls) == 1
            assert detector.footprint_calls[0][1] is True
            assert detector.systematics_calls == 1

        n_bins = _geometric_n_bins(1.0, 4.0, 2.0)
        assert DummyDetector.instances[0].num_wavel_bins == n_bins

    def test_s2n_e_writes_fits_and_returns_s2n(self, monkeypatch, s2n_e_config, unit_sources_all):
        """Test s2n_e FITS output path with mocked detector and FITS IO."""
        DummyDetector.instances = []
        monkeypatch.setattr("modules.core.calculator.Detector", DummyDetector)

        calc = NoiseCalculator(
            config=s2n_e_config,
            sources_all=unit_sources_all,
            sources_to_include=["star", "exoplanet_bb"],
        )

        n_bins = _geometric_n_bins(1.0, 4.0, 2.0)
        n_dc = 2  # np.arange(0.0, 0.2, 0.1)

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

        assert out.shape == (n_dc, n_bins)
        hdu = hdu_holder["hdu"]
        assert hdu.writes[0] == ("/tmp/s2n_temp.fits", True)
        assert hdu.writes[1] == ("/tmp/s2n_unique.fits", True)
        assert hdu.data.shape == (4, n_dc, n_bins)
        assert hdu.header["N_INT"] == 10
        assert hdu.header["QE"] == "0.8"

    def test_s2n_e_plot_path_runs_with_mocked_matplotlib(
        self, monkeypatch, s2n_e_config, unit_sources_all
    ):
        DummyDetector.instances = []
        monkeypatch.setattr("modules.core.calculator.Detector", DummyDetector)

        class DummyHeader(dict):
            def add_blank(self, *args, **kwargs):
                return None

        monkeypatch.setattr(
            "modules.core.calculator.fits.PrimaryHDU",
            lambda: SimpleNamespace(
                header=DummyHeader(),
                data=None,
                writeto=lambda *args, **kwargs: None,
            ),
        )
        monkeypatch.setattr("modules.core.calculator.fits.Card", lambda *args, **kwargs: None)

        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.gca.return_value = mock_ax
        mock_plt.gcf.return_value = mock_fig
        monkeypatch.setattr("modules.core.calculator.plt", mock_plt)

        calc = NoiseCalculator(
            config=s2n_e_config,
            sources_all=unit_sources_all,
            sources_to_include=["star", "exoplanet_bb"],
        )
        calc.s2n_e(file_name_fits_unique="/tmp/s2n_unique.fits", plot=True)

        assert mock_plt.savefig.called
        assert mock_plt.close.called
