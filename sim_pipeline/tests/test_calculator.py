"""
Unit tests for the main noise calculator.
"""

import configparser
import pickle
import re
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

# Mock ipdb before importing project modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.core.calculator import (
    NoiseCalculator,
    S2NCube,
    calculate_s2n_post_rotation,
    load_s2n_cube,
    read_s2n_cube_hdf5,
    save_s2n_cube,
)


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
        calc.s2n_e(file_name_fits_unique="/tmp/s2n_unique.fits", plot=True)

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


PLANET_COL = "astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"
CHOPPED_PLANET_COL = f"chopped_{PLANET_COL}"
STAR_COL_OUT3 = "astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"
STAR_COL_3 = "output_3_dark_astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"
STAR_COL_4 = "output_4_dark_astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"


def _make_wavelength_meta(n_bins: int = 1):
    centers = np.linspace(10.0, 10.0 + n_bins - 1, n_bins) * u.um
    widths = np.ones(n_bins) * u.um
    edges = np.empty(n_bins + 1) * u.um
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * widths[0]
    edges[-1] = centers[-1] + 0.5 * widths[-1]
    return centers, widths, edges


def _base_table(n_bins: int = 1) -> QTable:
    centers, widths, edges = _make_wavelength_meta(n_bins)
    tbl = QTable()
    tbl["wavel_bin_num"] = np.arange(n_bins)
    tbl["wavel_bin_center"] = centers
    tbl["wavel_bin_width"] = widths
    tbl["n_pix_per_wavel_bin"] = np.full(n_bins, 100) * u.pix
    tbl.meta["wavel_bin_edges"] = edges
    tbl.meta["qe"] = 0.70
    tbl.meta["angle_deg"] = 0.0
    tbl.meta["dark_current_e_pix_s"] = 1e-4
    return tbl


def _write_angle_hdf5(
    path,
    *,
    angle_deg: float,
    dc_rate: float = 1e-4,
    qe: float = 0.70,
    planet_chopped,
    planet_out3,
    star_out3=None,
    star_out4=None,
    instrum_dark=0.5,
    instrum_read=0.3,
    n_bins: int = 1,
):
    """Write one angle_*.hdf5 file matching record_info_at_angle_and_qe layout."""
    dc_qe_str = f"dc_{dc_rate:06.3f}_qe_{qe:04.2f}"
    star_out3 = star_out3 if star_out3 is not None else np.zeros(n_bins)
    star_out4 = star_out4 if star_out4 is not None else np.zeros(n_bins)

    def _write_table(tbl, dataset_name, overwrite=False):
        tbl.meta["angle_deg"] = float(angle_deg)
        tbl.meta["qe"] = float(qe)
        tbl.write(
            path,
            path=f"{dc_qe_str}/{dataset_name}",
            serialize_meta=True,
            overwrite=overwrite,
            append=not overwrite,
        )

    first = True
    for ch_name in (
        "output_1_bright",
        "output_2_bright",
        "output_3_dark",
        "output_4_dark",
    ):
        tbl = _base_table(n_bins)
        if ch_name == "output_3_dark":
            tbl[PLANET_COL] = np.asarray(planet_out3) * u.adu
            tbl[STAR_COL_OUT3] = np.asarray(star_out3) * u.adu
        _write_table(tbl, ch_name, overwrite=first)
        first = False

    chopped = _base_table(n_bins)
    chopped[CHOPPED_PLANET_COL] = np.asarray(planet_chopped) * u.adu
    chopped[STAR_COL_3] = np.asarray(star_out3) * u.adu
    chopped[STAR_COL_4] = np.asarray(star_out4) * u.adu
    chopped["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"] = (
        np.full(n_bins, instrum_dark) * u.adu
    )
    chopped["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"] = (
        np.full(n_bins, instrum_read) * u.adu
    )
    _write_table(chopped, "chopped", overwrite=False)


def _expected_snr_lambda(
    *,
    planet_chopped_by_angle,
    planet_out3_by_angle,
    star_out3_by_angle,
    star_out4_by_angle=None,
    gain,
    instrum_dark,
    instrum_read,
):
    """Mirror _compute_snr_lambda_for_slot SNR math (Dannert+ 2022 Eqn. 19-20)."""
    del star_out4_by_angle  # not used in symmetric-noise term; kept for call-site clarity

    gain_val = float(gain)
    angles = sorted(planet_chopped_by_angle.keys())
    n_bins = len(next(iter(planet_chopped_by_angle.values())))

    cols_S_p_elec = [np.asarray(planet_chopped_by_angle[a]) * gain_val for a in angles]
    cols_S_p_3_elec = [np.asarray(planet_out3_by_angle[a]) * gain_val for a in angles]
    S_p_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_elec), 2), axis=1)
    S_p_3_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_3_elec), 2), axis=1)

    S_sym_noise_var_3_elec = np.zeros(n_bins, dtype=float)
    if star_out3_by_angle is not None:
        cols_sym_noise_var_3_elec = []
        for a in angles:
            sym_flux_elec = np.asarray(star_out3_by_angle[a]) * gain_val
            cols_sym_noise_var_3_elec.append(np.sqrt(sym_flux_elec))
        S_sym_noise_var_3_elec = np.mean(np.column_stack(cols_sym_noise_var_3_elec), axis=1)

    instrum_dark = np.asarray(instrum_dark, dtype=float)
    instrum_read = np.asarray(instrum_read, dtype=float)

    snr_bins = []
    for wavel_bin_num in range(n_bins):
        S_p_rms_phi = np.sqrt(S_p_sqd_arr_mean_elec[wavel_bin_num])
        S_p_3_rms_phi = np.sqrt(S_p_3_sqd_arr_mean_elec[wavel_bin_num])

        S_dark_noise_var = (instrum_dark[wavel_bin_num] * gain_val) ** 2
        S_read_noise_var = (instrum_read[wavel_bin_num] * gain_val) ** 2

        S_sym_3_var_this = S_sym_noise_var_3_elec[wavel_bin_num]
        astro_noise_term = 2 * (S_sym_3_var_this + S_p_3_rms_phi)
        instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)
        denominator = np.sqrt(astro_noise_term + instrum_noise_term)
        snr_bins.append(S_p_rms_phi / denominator)

    snr_bins = np.asarray(snr_bins)
    return snr_bins, float(np.sqrt(np.sum(np.power(snr_bins, 2))))


@pytest.fixture
def s2n_config(tmp_path):
    return {
        "dirs": {"save_s2n_data_unique_dir": str(tmp_path / "out") + "/"},
        "detector": {"gain": "2.0", "quantum_efficiency": "0.7"},
        "observation": {
            "t_int_frame": "10",
            "N_angles": "2",
            "N_int_per_angle": "1",
        },
        "plotting": {"title_context": ""},
    }


@pytest.fixture
def patch_plotting():
    with patch("modules.core.calculator.plt.savefig"), patch(
        "modules.core.calculator.plt.figure"
    ), patch("modules.core.calculator.plt.clf"), patch(
        "modules.core.calculator.plt.stairs"
    ), patch(
        "modules.core.calculator.plt.tight_layout"
    ):
        yield


class TestCalculateS2nPostRotation:
    def test_snr_matches_reference_implementation(
        self, tmp_path, s2n_config, patch_plotting, capsys
    ):
        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        _write_angle_hdf5(
            read_dir / "angle_0.hdf5",
            angle_deg=0.0,
            planet_chopped=[4.0],
            planet_out3=[3.0],
            star_out3=[3.0],
            star_out4=[1.0],
        )
        _write_angle_hdf5(
            read_dir / "angle_90.hdf5",
            angle_deg=90.0,
            planet_chopped=[-5.0],
            planet_out3=[1.0],
            star_out3=[3.0],
            star_out4=[1.0],
        )

        gain = float(s2n_config["detector"]["gain"])
        # note the inputs here are in photoelectrons, not ADU
        _, pipeline_s2n_tot = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [4.0], 90.0: [-5.0]},
            planet_out3_by_angle={0.0: [3.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [3.0], 90.0: [3.0]},
            star_out4_by_angle={0.0: [2.0], 90.0: [2.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )

        cube_test = calculate_s2n_post_rotation(str(read_dir), config=s2n_config)

        captured = capsys.readouterr().out
        match = re.search(r"SNR_tot for DC dc_00\.000_qe_0\.70: ([0-9.]+)", captured)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(pipeline_s2n_tot)

        # by hand (mirror _compute_snr_lambda_for_slot unit conventions)
        g = gain * u.electron / u.adu
        planet_chopped_adu = np.array([4, -5]) * u.adu
        planet_out3_adu = np.array([3, 1]) * u.adu
        star_out3_adu = np.array([3, 3]) * u.adu
        dark_rms_adu = 0.5 * u.adu
        read_rms_adu = 0.3 * u.adu

        S_p_rms = np.sqrt(np.mean(np.power(planet_chopped_adu * g, 2)))
        S_p3_rms = np.sqrt(np.mean(np.power(planet_out3_adu * g, 2)))
        S_sym = np.mean(np.sqrt((star_out3_adu * g).to_value(u.electron))) * u.electron
        astro_noise_term = 2 * (S_sym + S_p3_rms)

        S_dark_noise_var = np.power(dark_rms_adu * g, 2).value * u.electron
        S_read_noise_var = np.power(read_rms_adu * g, 2).value * u.electron
        instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)

        denominator = np.sqrt((astro_noise_term + instrum_noise_term).value) * u.electron
        manual_s2n_tot = (S_p_rms / denominator).decompose().value

        assert pipeline_s2n_tot == pytest.approx(manual_s2n_tot)

    def test_symmetric_noise_uses_all_angles_not_last_only(
        self, tmp_path, s2n_config, patch_plotting, capsys
    ):
        """Regression: star shot noise should use angle-averaged sigma^2, not last file only."""
        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        _write_angle_hdf5(
            read_dir / "angle_0.hdf5",
            angle_deg=0.0,
            planet_chopped=[1.0],
            planet_out3=[1.0],
            star_out3=[4.0],
            star_out4=[4.0],
        )
        _write_angle_hdf5(
            read_dir / "angle_90.hdf5",
            angle_deg=90.0,
            planet_chopped=[1.0],
            planet_out3=[1.0],
            star_out3=[0.0],
            star_out4=[0.0],
        )

        gain = float(s2n_config["detector"]["gain"])
        expected_with_avg, _ = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [1.0], 90.0: [1.0]},
            planet_out3_by_angle={0.0: [1.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [4.0], 90.0: [0.0]},
            star_out4_by_angle={0.0: [4.0], 90.0: [0.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )
        expected_last_only, _ = _expected_snr_lambda(
            planet_chopped_by_angle={0.0: [1.0], 90.0: [1.0]},
            planet_out3_by_angle={0.0: [1.0], 90.0: [1.0]},
            star_out3_by_angle={0.0: [0.0], 90.0: [0.0]},
            star_out4_by_angle={0.0: [0.0], 90.0: [0.0]},
            gain=gain,
            instrum_dark=np.array([0.5]),
            instrum_read=np.array([0.3]),
        )

        assert expected_with_avg[0] != pytest.approx(expected_last_only[0])
        assert expected_with_avg[0] < expected_last_only[0]

        calculate_s2n_post_rotation(str(read_dir), config=s2n_config)
        captured = capsys.readouterr().out
        match = re.search(r"SNR_tot for DC dc_00\.000_qe_0\.70: ([0-9.]+)", captured)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(expected_with_avg[0])


class TestReadHdf5Slots:
    def test_angle_keys_use_meta_canonical_value(self, tmp_path):
        from modules.core.calculator import read_hdf5_slots
        from modules.utils.helpers import canonical_angle_deg, format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()
        angle_linspace = float(np.linspace(0, 360, num=7, endpoint=False)[1])
        canonical = canonical_angle_deg(angle_linspace)

        _write_angle_hdf5(
            read_dir / format_angle_qe_hdf5_name(angle_linspace, 0.70),
            angle_deg=angle_linspace,
            planet_chopped=[1.0],
            planet_out3=[1.0],
        )

        slots = read_hdf5_slots(str(read_dir))
        slot = slots["dc_00.000_qe_0.70"]
        assert list(slot["S_p"].keys()) == [canonical]
        assert canonical != angle_linspace or angle_linspace == round(angle_linspace, 2)

    def test_read_hdf5_slots_filters_by_qe(self, tmp_path):
        from modules.core.calculator import build_s2n_cube_from_hdf5, read_hdf5_slots
        from modules.utils.helpers import format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        for qe in (0.60, 0.70):
            _write_angle_hdf5(
                read_dir / format_angle_qe_hdf5_name(0.0, qe),
                angle_deg=0.0,
                qe=qe,
                planet_chopped=[1.0],
                planet_out3=[1.0],
            )

        slots_070 = read_hdf5_slots(str(read_dir), qe=0.70)
        assert set(slots_070) == {"dc_00.000_qe_0.70"}

        slots_060 = read_hdf5_slots(str(read_dir), qe=0.60)
        assert set(slots_060) == {"dc_00.000_qe_0.60"}

    def test_build_s2n_cube_uses_config_qe_not_other_angle_files(
        self, tmp_path, s2n_config, patch_plotting
    ):
        from modules.core.calculator import build_s2n_cube_from_hdf5
        from modules.utils.helpers import format_angle_qe_hdf5_name

        read_dir = tmp_path / "hdf5"
        read_dir.mkdir()

        for qe in (0.60, 0.70):
            _write_angle_hdf5(
                read_dir / format_angle_qe_hdf5_name(0.0, qe),
                angle_deg=0.0,
                qe=qe,
                planet_chopped=[1.0],
                planet_out3=[1.0],
            )

        cube = build_s2n_cube_from_hdf5(str(read_dir), s2n_config)
        assert list(cube.qe) == [0.7]
        assert cube.snr.shape[2] == 1


def _make_sample_s2n_cube(read_dir: str = "/tmp/hdf5_in") -> S2NCube:
    wavelength = np.array([10.0, 11.0])
    wavel_bin_width = np.array([0.5, 0.5])
    wavel_bin_edges = np.array([9.75, 10.25, 11.25])
    dark_current = np.array([0.0, 0.1])
    qe = np.array([0.6, 0.7])
    snr = np.arange(8, dtype=float).reshape(2, 2, 2)
    snr_tot = np.array([[1.0, 2.0], [3.0, 4.0]])
    base_titles = np.array([["title_a", "title_b"], ["title_c", "title_d"]], dtype=object)
    return S2NCube(
        snr=snr,
        wavelength=wavelength,
        wavel_bin_width=wavel_bin_width,
        wavel_bin_edges=wavel_bin_edges,
        dark_current=dark_current,
        qe=qe,
        snr_tot=snr_tot,
        base_titles=base_titles,
        title_context="t_int_frame = 10.0 s",
        sources_context="star",
        read_dir=read_dir,
        n_angles=4,
        n_int_per_angle=1,
        t_int_frame=10.0,
        n_int_total=40.0,
        config={"detector": {"gain": "2.0"}, "plotting": {"title_context": ""}},
    )


class TestSaveS2nCube:
    def test_save_hdf5_roundtrip(self, tmp_path):
        cube = _make_sample_s2n_cube(read_dir=str(tmp_path / "hdf5_in"))
        out_path = tmp_path / "s2n_cube"

        saved = save_s2n_cube(cube, out_path, file_format="hdf5")

        assert saved == [str(out_path.with_suffix(".hdf5"))]
        restored = read_s2n_cube_hdf5(saved[0])
        assert np.allclose(restored.snr, cube.snr)
        assert np.allclose(restored.snr_tot, cube.snr_tot)
        assert np.allclose(restored.wavelength, cube.wavelength)
        assert np.allclose(restored.dark_current, cube.dark_current)
        assert np.allclose(restored.qe, cube.qe)
        assert restored.title_context == cube.title_context
        assert restored.read_dir == cube.read_dir
        assert restored.n_angles == cube.n_angles
        assert restored.config == cube.config

    def test_save_pickle_roundtrip(self, tmp_path):
        cube = _make_sample_s2n_cube()
        out_path = tmp_path / "s2n_cube.pkl"

        saved = save_s2n_cube(cube, out_path, file_format="pickle")

        assert saved == [str(out_path)]
        with open(saved[0], "rb") as handle:
            restored = pickle.load(handle)
        assert np.allclose(restored.snr, cube.snr)
        assert restored.base_titles.tolist() == cube.base_titles.tolist()

    def test_save_both_formats(self, tmp_path):
        cube = _make_sample_s2n_cube()
        out_path = tmp_path / "s2n_cube"

        saved = save_s2n_cube(cube, out_path, file_format="both")

        assert saved == [str(out_path.with_suffix(".pkl")), str(out_path.with_suffix(".hdf5"))]
        assert load_s2n_cube(saved[0]).snr.shape == cube.snr.shape
        assert load_s2n_cube(saved[1]).snr_tot.shape == cube.snr_tot.shape

    def test_hdf5_writes_meta_and_formatted_titles(self, tmp_path):
        cube = _make_sample_s2n_cube()
        hdf5_path = tmp_path / "s2n_cube.hdf5"
        save_s2n_cube(cube, hdf5_path, file_format="hdf5")

        with h5py.File(hdf5_path, "r") as handle:
            meta = handle["meta"]
            assert meta.attrs["axis_order"] == "wavelength, dark_current, qe"
            assert meta.attrs["n_angles"] == cube.n_angles
            assert "formatted_plot_titles" in meta
            formatted = meta["formatted_plot_titles"][:].astype(str)
            assert formatted.shape == cube.base_titles.shape
            assert formatted[0, 0].startswith("title_a")
