"""
Unit tests for instrumental noise calculations.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

# Mock ipdb before importing project modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.core.instrumental import Detector, InstrumentDepTerms, OutputChannel
from modules.data.units import UnitConverter

REPO_ROOT = Path(__file__).resolve().parents[2]
APERTURE_YAML = REPO_ROOT / "sim_pipeline/config/aperture_array_double_bracewell.yaml"


def _geometric_n_bins(lambda_min: float, lambda_max: float, spec_res: float) -> int:
    return int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / spec_res)))


@pytest.fixture
def unit_converter():
    return UnitConverter()


@pytest.fixture
def instrum_base_config():
    """Minimal config required by InstrumentDepTerms.__init__."""
    return {
        "detector": {"spec_res": "20"},
        "wavelength_range": {"min": "4.0", "max": "18.0"},
    }


@pytest.fixture
def detector_geometry_config():
    return {
        "detector": {
            "size": "400",
            "pitch_pix": "25",
            "pix_per_wavel_bin": "2.2",
            "pix_spectral_width": "3",
        }
    }


@pytest.fixture
def noise_calc_config(instrum_base_config):
    config = {
        **instrum_base_config,
        "detector": {
            **instrum_base_config["detector"],
            "read_noise": "2.0, 3.0",
            "dark_current": "0.0, 0.2, 0.1",
            "gain": "4.5",
            "quantum_efficiency": "0.8",
            "e_per_ph": "1.0",
            "photons_to_e": "1.0",
        },
        "observation": {"t_int_obs_total": "100", "t_int_frame": "10"},
        "telescope": {"eta_t": "0.8"},
        "dirs": {"save_s2n_data_unique_dir": "/tmp/"},
    }
    return config


def _make_flux_cube(n_wavel: int = 3, n_pix: int = 5, scale: float = 1.0):
    return np.ones((n_wavel, n_pix, n_pix)) * scale * u.ph / (u.um * u.m**2 * u.s)


def _output_flux_dict(cube):
    return {
        "output_1_bright": cube,
        "output_2_bright": cube,
        "output_3_dark": cube,
        "output_4_dark": cube,
    }


def _attach_detector_footprint(channel: OutputChannel, detector_geometry_config, n_bins: int = 3):
    det = Detector(config=detector_geometry_config, num_wavel_bins=n_bins)
    det.footprint_spectral(file_name_plot="/tmp/footprint.png", plot=False)
    channel.detector = det
    return det


def _make_astroph_signal(bin_centers, bin_widths, n_pix_per_bin, flux_um: float = 10.0, qe: float = 0.8):
    flux_ph_sec_um = np.full(len(bin_centers), flux_um) * u.ph / (u.um * u.s) * qe
    flux_ph_sec_wavel_bin = flux_ph_sec_um * bin_widths
    flux_ph_sec_pixel = flux_ph_sec_wavel_bin / n_pix_per_bin
    return {
        "wavel": bin_centers,
        "flux_astro_1d_interpolated_ph_sec_um": flux_ph_sec_um,
        "flux_astro_1d_interpolated_ph_sec_wavel_bin": flux_ph_sec_wavel_bin,
        "flux_astro_1d_interpolated_ph_sec_pixel": flux_ph_sec_pixel,
        "n_pix_per_wavel_bin": n_pix_per_bin,
    }


@pytest.fixture
def transmission_config(tmp_path):
    return {
        "detector": {"spec_res": "20"},
        "wavelength_range": {"min": "4.0", "max": "18.0"},
        "telescope": {"aperture_array_config_file_name": str(APERTURE_YAML)},
        "onsky_scene": {"n_pix": "1001", "pix_size_mas": "10.7", "half_pix": "1"},
        "nulling": {"nulling_factor": "0.001"},
        "dirs": {"save_s2n_data_unique_dir": str(tmp_path / "out") + "/"},
    }


@pytest.fixture
def disperse_config(instrum_base_config, detector_geometry_config):
    return {
        **instrum_base_config,
        **detector_geometry_config,
        "dirs": {"save_s2n_data_unique_dir": "/tmp/"},
        "observation": {"N_int_per_angle": "1"},
        "detector": {
            **instrum_base_config["detector"],
            **detector_geometry_config["detector"],
            "quantum_efficiency": "0.8",
        },
    }


class TestInstrumentDepTerms:
    def test_init_creates_output_channels_with_wavelength_bins(
        self, instrum_base_config, unit_converter
    ):
        instr = InstrumentDepTerms(
            instrum_base_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=["star"],
        )

        assert instr.config is instrum_base_config
        assert instr.unit_converter is unit_converter
        assert instr.sources_to_include == ["star"]
        assert isinstance(instr.sources_instrum, dict)
        assert isinstance(instr.prop_dict, dict)

        assert set(instr.output_channels.keys()) == {
            "output_1_bright",
            "output_2_bright",
            "output_3_dark",
            "output_4_dark",
        }
        n_bins = _geometric_n_bins(4.0, 18.0, 20.0)
        for channel in instr.output_channels.values():
            assert isinstance(channel, OutputChannel)
            assert channel.spec_R == 20.0
            assert len(channel.bin_centers) == n_bins
            assert len(channel.bin_edges) == n_bins + 1
            assert len(channel.bin_widths) == n_bins

    def test_calculate_intrinsic_instrumental_noise_parses_arrays(
        self, noise_calc_config, unit_converter
    ):
        instr = InstrumentDepTerms(
            noise_calc_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        instr.calculate_instrinsic_instrumental_noise()

        read_noise = instr.sources_instrum["read_noise_e_pix-1"]
        dc_rate = instr.sources_instrum["dark_current_e_pix-1_sec-1"]
        dc_total = instr.sources_instrum["dark_current_e_pix-1"]

        assert read_noise.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(read_noise.value, [2.0, 3.0])
        assert dc_rate.unit.is_equivalent(u.electron / (u.pix * u.s))
        assert np.allclose(dc_rate.value, [0.0, 0.1])
        assert dc_total.unit.is_equivalent(u.electron / u.pix)
        assert np.allclose(dc_total.value, [0.0, 1.0])  # t_int_frame * rate

        for channel in instr.output_channels.values():
            assert "read_noise_e_pix-1" in channel.instrum_noise
            assert "dark_current_e_pix-1_sec-1" in channel.instrum_noise

    def test_calculate_intrinsic_instrumental_noise_parses_single_values(
        self, unit_converter, instrum_base_config
    ):
        config = {
            **instrum_base_config,
            "detector": {
                **instrum_base_config["detector"],
                "read_noise": "6.0",
                "dark_current": "0.05",
                "gain": "4.5",
            },
            "observation": {"t_int_obs_total": "200", "t_int_frame": "20"},
        }
        instr = InstrumentDepTerms(config, unit_converter, sources_astroph={}, sources_to_include=[])

        instr.calculate_instrinsic_instrumental_noise()

        read_noise = instr.sources_instrum["read_noise_e_pix-1"]
        dc_rate = instr.sources_instrum["dark_current_e_pix-1_sec-1"]
        dc_total = instr.sources_instrum["dark_current_e_pix-1"]

        assert np.allclose(read_noise.value, [6.0])
        assert np.allclose(dc_rate.value, [0.05])
        assert np.allclose(dc_total.value, [1.0])  # 20 s * 0.05 e-/pix/s

    @patch("modules.core.instrumental.compute_collecting_area_m2", return_value=25.0)
    def test_pass_through_aperture_scales_flux_cubes_by_collecting_area_and_throughput(
        self, _mock_area, unit_converter, instrum_base_config
    ):
        config = {**instrum_base_config, "telescope": {"eta_t": "0.5"}}
        wavel = np.array([1.0, 2.0, 3.0]) * u.um
        cube = _make_flux_cube(scale=2.0)

        sources_astroph = {
            "star": {
                "wavel": wavel,
                "flux_cube_post_screen_ph_sec_um": _output_flux_dict(cube),
            }
        }

        instr = InstrumentDepTerms(
            config, unit_converter, sources_astroph=sources_astroph, sources_to_include=["star"]
        )
        instr.pass_through_aperture(plot=False)

        star = instr.prop_dict["star"]
        assert star["wavel"] is wavel
        post = star["flux_cube_post_screen_post_aperture_ph_sec_um"]["output_1_bright"]
        expected = 0.5 * (25.0 * u.m**2) * cube
        assert post.unit.is_equivalent(u.ph / (u.um * u.s))
        assert np.allclose(post.value, expected.value)

    def test_generate_instrument_transmission(
        self, unit_converter, instrum_base_config, transmission_config
        ):
        # make sure sum of transmission screens is 1.0 (except for very center where star is)

        Path(transmission_config["dirs"]["save_s2n_data_unique_dir"]).mkdir(parents=True, exist_ok=True)
        config = {**instrum_base_config, "dirs": {"save_s2n_data_unique_dir": "/tmp/"}}
        wavel = np.array([1.0, 2.0]) * u.um
        scene = np.ones((2, 4, 4)) * u.ph / (u.um * u.m**2 * u.s)

        instr = InstrumentDepTerms(
            transmission_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        #ipdb.set_trace()
        screens = instr.generate_instrument_transmission(wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False)
        
        # for debugging
        '''
        import matplotlib.pyplot as plt
        screens = instr.generate_instrument_transmission(
            wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False
        )
        extent = [
            screens[5].min(), screens[5].max(),  # x [arcsec]
            screens[4].min(), screens[4].max(),  # y [arcsec]
        ]
        names = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, i, name in zip(axes.ravel(), range(4), names):
            im = ax.imshow(screens[i], origin="lower", extent=extent, aspect="equal")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(transmission_config["dirs"]["save_s2n_data_unique_dir"] + "test_transmission_screens.png")
        plt.show()  # only if running interactively, not in headless pytest
        '''

        assert np.max(np.any(screens[0:5,:,:], axis=(0))) < 1.0 + 1e-6 # transmission < 1
        net = np.sum(screens[0:4], axis=0)
        assert np.allclose(net, 1.0)


    def test_pass_through_transmission_screens_multiplies_scene_by_each_output(
        self, unit_converter, instrum_base_config, transmission_config
    ):
        wavel = np.array([1.0, 2.0]) * u.um
        scene = np.zeros((2, 1001, 1001)) * u.ph / (u.um * u.m**2 * u.s)
        scene[:, 500:600, 200:300] = 1.0 * u.ph / (u.um * u.m**2 * u.s) # the 'star'
        Path(transmission_config["dirs"]["save_s2n_data_unique_dir"]).mkdir(parents=True, exist_ok=True)


        sources_astroph = {
            "star": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.ones(2) * u.ph / (u.um * u.m**2 * u.s),
            }
        }

        instr = InstrumentDepTerms(
            transmission_config, 
            unit_converter, 
            sources_astroph=sources_astroph, 
            sources_to_include=["star"]
        )

        screens = instr.generate_instrument_transmission(wavel_m=11e-6, override_stellar_mask=False, normalize=True, plot=False)

        instr.pass_through_transmission_screens(
            fyi_angle=0.0,
            source_dict_pre_screen={"star": scene},
            transmission_screens=screens,
            plot=False,
        )

        bright_1 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_um"]["output_1_bright"]
        bright_2 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_um"]["output_2_bright"]
        dark_1 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_um"]["output_3_dark"]
        dark_2 = instr.sources_astroph["star"]["flux_cube_post_screen_ph_sec_um"]["output_4_dark"]

        '''
        if np.logical_or(
            np.round(test_flux_1, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1),
            np.round(test_flux_2, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1)
        ):
        '''

        net_flux = bright_1 + bright_2 + dark_1 + dark_2

        # for debugging
        '''
        import matplotlib.pyplot as plt
        wavel_idx = 0
        panels = [
            ("scene", scene),
            ("output_1_bright", bright_1),
            ("output_2_bright", bright_2),
            ("output_3_dark", dark_1),
            ("output_4_dark", dark_2),
            ("net_flux", net_flux),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, (name, cube) in zip(axes.ravel(), panels):
            im = ax.imshow(cube[wavel_idx, :, :].value, origin="lower")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"post-screen flux (wavel index {wavel_idx})")
        plt.tight_layout()
        out_dir = transmission_config["dirs"]["save_s2n_data_unique_dir"]
        plt.show()
        #plt.savefig(out_dir + "test_post_screen_flux_panels.png")
        #plt.close(fig)
        '''

        # check flux conservation
        np.testing.assert_allclose(net_flux.value, scene.value)
   
        #assert np.allclose(dark[0, 0, 0].value, 0.1 * scene[0, 0, 0].value)


    '''
    def test_photons_to_e_converts_legacy_post_aperture_flux(self, unit_converter, instrum_base_config):
        config = {
            **instrum_base_config,
            "detector": {
                **instrum_base_config["detector"],
                "photons_to_e": "1.0",
                "quantum_efficiency": "0.8",
            },
        }
        instr = InstrumentDepTerms(config, unit_converter, sources_astroph={}, sources_to_include=[])

        wavel = np.array([1.0, 2.0, 3.0]) * u.um
        flux_ph = np.array([10.0, 20.0, 30.0]) * u.ph / (u.um * u.s)
        instr.prop_dict = {
            "star": {"wavel": wavel, "flux_post_aperture_ph_sec_um": flux_ph},
            "bad": {
                "wavel": wavel,
                "flux_post_aperture_ph_sec_um": np.array([1.0, 2.0, 3.0]) * u.W,
            },
        }

        instr.photons_to_e()

        expected = flux_ph * 0.8 * (u.electron / u.ph)
        got = instr.prop_dict["star"]["flux_e_sec_um"]
        assert got.unit.is_equivalent(u.electron / (u.um * u.s))
        assert np.allclose(got.value, expected.value)
        assert "flux_e_sec_um" not in instr.prop_dict["bad"]
    '''


    def test_chop_signal_builds_post_chop_tables(self, unit_converter, instrum_base_config):
        instr = InstrumentDepTerms(
            instrum_base_config, unit_converter, sources_astroph={}, sources_to_include=["star"]
        )

        def _table(astro_scale: float, instrum_scale: float):
            qt = QTable()
            qt["wavel_bin_num"] = [0, 1]
            qt["wavel_bin_center"] = np.array([5.0, 6.0]) * u.um
            qt["wavel_bin_width"] = np.array([0.1, 0.1]) * u.um
            qt["n_pix_per_wavel_bin"] = np.array([4.0, 4.0]) * u.pix
            qt["astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"] = (
                np.array([astro_scale, astro_scale + 1.0]) * u.adu
            )
            qt["instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"] = (
                np.array([instrum_scale, instrum_scale + 2.0]) * u.adu
            )
            qt["instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"] = (
                np.array([1.0, 1.0]) * u.adu
            )
            return qt

        for name, astro_scale, instrum_scale in [
            ("output_1_bright", 10.0, 1.0),
            ("output_2_bright", 11.0, 1.1),
            ("output_3_dark", 3.0, 2.0),
            ("output_4_dark", 1.0, 4.0),
        ]:
            instr.output_channels[name].tables_by_dark_current = {
                0.1: _table(astro_scale, instrum_scale)
            }

        instr.chop_signal(plot=False)

        chopped = instr.post_chop_tables_by_dark_current[0.1]
        assert "chopped_astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot" in chopped.colnames
        assert np.allclose(
            chopped["chopped_astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"].value,
            np.array([2.0, 2.0]),
        )


class TestBuildBaseAstroTable:
    def test_build_base_astro_table_includes_metadata_and_source_columns(
        self, unit_converter, instrum_base_config, detector_geometry_config
    ):
        instr = InstrumentDepTerms(
            instrum_base_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=["star", "exozodiacal"],
        )
        channel = instr.output_channels["output_1_bright"]
        n_bins = 3
        channel.bin_centers = np.array([5.0, 6.0, 7.0]) * u.um
        channel.bin_widths = np.array([0.1, 0.1, 0.1]) * u.um
        channel.bin_edges = np.array([4.95, 5.05, 6.05, 7.05]) * u.um

        det = _attach_detector_footprint(channel, detector_geometry_config, n_bins=n_bins)
        n_pix_per_bin = np.sum(det.footprint_cube, axis=(1, 2)) * u.pix
        channel.astroph_signal["star"] = _make_astroph_signal(
            channel.bin_centers, channel.bin_widths, n_pix_per_bin, flux_um=12.0
        )

        table = instr._build_base_astro_table(channel)

        assert len(table) == n_bins
        assert "wavel_bin_num" in table.colnames
        assert "n_pix_per_wavel_bin" in table.colnames
        assert "astro_star_flux_ph_sec_um" in table.colnames
        assert "astro_star_flux_ph_sec_wavel_bin" in table.colnames
        assert "astro_star_flux_ph_sec_pixel" in table.colnames
        assert "astro_exozodiacal_flux_ph_sec_um" not in table.colnames
        assert np.allclose(table["n_pix_per_wavel_bin"].value, n_pix_per_bin.value)
        assert table.meta["wavel_bin_edges"] is channel.bin_edges


class TestCombineAstroAndInstrumSignals:
    @pytest.fixture
    def combine_config(self, noise_calc_config):
        config = {**noise_calc_config}
        config["detector"]["read_noise"] = "6.0"
        config["observation"]["t_int_frame"] = "10"
        return config

    def _prepare_channels(self, instr, detector_geometry_config):
        for channel in instr.output_channels.values():
            channel.bin_centers = channel.bin_centers[:3]
            channel.bin_widths = channel.bin_widths[:3]
            channel.bin_edges = channel.bin_edges[:4]
            det = _attach_detector_footprint(channel, detector_geometry_config, n_bins=3)
            n_pix_per_bin = np.sum(det.footprint_cube, axis=(1, 2)) * u.pix
            channel.astroph_signal["star"] = _make_astroph_signal(
                channel.bin_centers,
                channel.bin_widths,
                n_pix_per_bin,
                flux_um=8.0,
                qe=1.0,
            )

    @patch("modules.core.instrumental.plt.close")
    @patch("modules.core.instrumental.plt.savefig")
    @patch("modules.core.instrumental.plt.subplots")
    @patch("builtins.print")
    def test_combine_builds_per_dc_tables_with_instrumental_and_astro_columns(
        self, _mock_print, mock_subplots, _mock_savefig, _mock_close,
        combine_config, unit_converter, detector_geometry_config,
    ):
        mock_subplots.return_value = (MagicMock(), MagicMock())

        instr = InstrumentDepTerms(
            combine_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=["star"],
        )
        instr.calculate_instrinsic_instrumental_noise()
        self._prepare_channels(instr, detector_geometry_config)

        instr.combine_astro_and_instrum_signals()

        channel = instr.output_channels["output_3_dark"]
        assert 0.0 in channel.tables_by_dark_current_orig
        assert 0.1 in channel.tables_by_dark_current_orig
        assert np.all(channel.tables_by_dark_current_orig[0.0]["qe"] == 0.8)

        final = channel.tables_by_dark_current[0.0]
        assert "instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot" in final.colnames
        assert "instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot" in final.colnames
        assert "astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot" in final.colnames
        assert len(final) == 3
        mock_subplots.assert_called()


class TestGenerateInstrumentTransmission:
    @patch("modules.core.instrumental.fits.writeto")
    def test_returns_six_slice_cube_and_writes_fits(
        self, mock_writeto, unit_converter, transmission_config
    ):
        instr = InstrumentDepTerms(
            transmission_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        cube = instr.generate_instrument_transmission(wavel_m=11e-6, plot=False)

        n_pix = 1001
        assert cube.shape == (6, n_pix, n_pix)
        assert np.all(cube[0] >= 0.0)
        assert np.max(cube[0]) <= 1.0 + 1e-12
        assert np.any(cube[4] != 0.0)
        assert np.any(cube[5] != 0.0)
        assert mock_writeto.call_count >= 5  # 4 outputs + differential dark

    @patch("modules.core.instrumental.fits.writeto")
    def test_override_stellar_mask_reduces_center_transmission(
        self, mock_writeto, unit_converter, transmission_config
    ):
        instr = InstrumentDepTerms(
            transmission_config,
            unit_converter,
            sources_astroph={},
            sources_to_include=[],
        )

        cube = instr.generate_instrument_transmission(
            wavel_m=11e-6, override_stellar_mask=True, plot=False
        )
        center = cube.shape[-1] // 2
        assert cube[0, center, center] == pytest.approx(0.001)


class TestDisperseAstroSignalsOnDetector:
    @patch("modules.core.instrumental.Detector")
    def test_disperse_populates_astroph_signal_per_output_channel(
        self, mock_detector_cls, unit_converter, disperse_config
    ):
        n_bins = 3
        footprint = np.ones((n_bins, 5, 5))
        n_pix_per_bin = np.sum(footprint, axis=(1, 2)) * u.pix

        mock_det = MagicMock()
        mock_det.footprint_cube = footprint
        mock_detector_cls.return_value = mock_det

        wavel = np.array([5.0, 10.0, 15.0]) * u.um
        post_cube = np.array([[[10.0]], [[20.0]], [[30.0]]]) * u.ph / (u.um * u.s)
        post_cube = np.broadcast_to(post_cube, (3, 5, 5)).copy()

        instr = InstrumentDepTerms(
            disperse_config,
            unit_converter,
            sources_astroph={"star": {"wavel": wavel}},
            sources_to_include=["star"],
        )
        instr.prop_dict["star"] = {
            "wavel": wavel,
            "flux_cube_post_screen_post_aperture_ph_sec_um": _output_flux_dict(post_cube),
        }
        for channel in instr.output_channels.values():
            channel.bin_centers = channel.bin_centers[:n_bins]
            channel.bin_widths = channel.bin_widths[:n_bins]

        instr.disperse_astro_signals_on_detector(plot=False)

        channel = instr.output_channels["output_1_bright"]
        assert "star" in channel.astroph_signal
        sig = channel.astroph_signal["star"]
        assert len(sig["flux_astro_1d_interpolated_ph_sec_um"]) == n_bins
        assert sig["flux_astro_1d_interpolated_ph_sec_um"].unit.is_equivalent(
            u.ph / (u.um * u.s)
        )
        expected_pixel = (
            sig["flux_astro_1d_interpolated_ph_sec_wavel_bin"][0] / n_pix_per_bin[0]
        )
        assert np.isclose(
            sig["flux_astro_1d_interpolated_ph_sec_pixel"][0].value,
            expected_pixel.value,
        )
        assert mock_det.footprint_spectral.call_count == 4


class TestDetector:
    def test_init_parses_detector_geometry(self, detector_geometry_config):
        det = Detector(config=detector_geometry_config, num_wavel_bins=17)
        assert det.side_length_pix == 400
        assert det.pitch_pix == 25.0
        assert det.pix_per_wavel_bin == 2.2
        assert det.pix_spectral_width == 3
        assert det.num_wavel_bins == 17

    def test_footprint_spectral_stores_expected_cube(self, detector_geometry_config):
        det = Detector(config=detector_geometry_config, num_wavel_bins=3)
        det.footprint_spectral(file_name_plot="/tmp/ignore.png", plot=False)
        cube = det.footprint_cube

        assert cube.shape == (3, 400, 400)
        assert cube.dtype == float
        assert np.min(cube) >= 0.0
        assert np.max(cube) <= 1.0

        expected_sum = det.pix_spectral_width * det.pix_per_wavel_bin
        for wbin in range(det.num_wavel_bins):
            assert np.isclose(cube[wbin].sum(), expected_sum)

        rows = slice(100, 100 + det.pix_spectral_width)
        assert np.allclose(cube[0, rows, 300], 1.0)
        assert np.allclose(cube[0, rows, 301], 1.0)
        assert np.allclose(cube[0, rows, 302], 0.2)

    def test_convert_2d_systematics_to_1d_vector_sums_within_footprint(
        self, detector_geometry_config
    ):
        det = Detector(config=detector_geometry_config, num_wavel_bins=2)
        det.footprint_spectral(file_name_plot="/tmp/ignore.png", plot=False)

        read_noise_map = np.zeros((400, 400))
        # Place map only under wavelength bin 0 footprint (cols 300-301), not bin 1 (starts ~302.2)
        read_noise_map[100:103, 300:302] = 2.0
        det.systematics_additive_dict["read_noise_map"] = read_noise_map

        vector = det.convert_2d_systematics_to_1d_vector()

        assert vector.shape == (2,)
        assert vector[0] > 0.0
        assert np.isclose(vector[1], 0.0)

    @patch("modules.core.instrumental.fits.getdata")
    def test_init_loads_enabled_2d_systematics_maps(
        self, mock_getdata, detector_geometry_config
    ):
        mock_getdata.return_value = np.ones((400, 400))
        config = {
            **detector_geometry_config,
            "detector_systematics": {
                "enable_read_noise_2d": "True",
                "read_noise_2d_file": "/tmp/read_noise.fits",
                "enable_dc_2d": "False",
                "dc_2d_file": "/tmp/dc.fits",
            },
        }

        det = Detector(config=config, num_wavel_bins=2)

        assert det.systematics_additive_dict["read_noise_map"] is not None
        assert det.systematics_additive_dict["bias_map"] is None
        mock_getdata.assert_called_once_with("/tmp/read_noise.fits")
