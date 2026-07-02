"""
Unit tests for astrophysical calculations in AstrophysicalSources.
"""

import configparser
import sys
import types
from unittest.mock import MagicMock, patch

# Mock ipdb before importing project modules (optional dev dependency)
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.utils.helpers import generate_star_spectrum, generate_planet_bb_spectrum

import numpy as np
import pandas as pd
import pytest
import astropy.constants as const
from astropy import units as u
from astropy.units import UnitConversionError

from modules.core.astrophysical import (
    AstrophysicalSources,
    _box_kernel,
    _exozodi_kernel,
    _zodiacal_kernel,
    generate_exoplanet_scene,
    generate_exozodi_scene,
    generate_star_scene,
    generate_zodiacal_scene,
)
from modules.data.spectra import SpectralData
from modules.data.units import UnitConverter


@pytest.fixture
def unit_converter():
    return UnitConverter()


class TestAstrophysicalSources:
    """Test cases for AstrophysicalSources class."""

    @pytest.fixture
    def config_with_sources(self):
        config = configparser.ConfigParser()
        config.add_section("nulling")
        config.set("nulling", "nulling_factor", "0.00001")
        config.add_section("astrophysical_sources_library")
        config.set("astrophysical_sources_library", "star", "/tmp/star.txt")
        config.set("astrophysical_sources_library", "exozodiacal", "/tmp/exozodi.txt")
        config.set("astrophysical_sources_library", "zodiacal", "/tmp/zodi.txt")
        return config

    @pytest.fixture
    def config_no_sources(self):
        config = configparser.ConfigParser()
        config.add_section("nulling")
        config.set("nulling", "nulling_factor", "0.00001")
        return config

    @pytest.fixture
    def sample_spectrum_zodiacal(self):
        wavelength = np.array([1.0, 2.0, 3.0])
        flux = np.array([10.0, 20.0, 30.0])
        return SpectralData(
            wavelength=wavelength,
            flux=flux,
            wavelength_unit="um",
            flux_unit="ph / (um m2 s)",
            source_name="test",
            metadata={},
        )

    @pytest.fixture
    def sample_spectrum_star(self):
        wavelength = np.array([1.0, 2.0, 3.0])
        flux = np.array([10.0, 20.0, 30.0])
        return SpectralData(
            wavelength=wavelength,
            flux=flux,
            wavelength_unit="um",
            flux_unit="ph / (um s)",
            source_name="test",
            metadata={},
        )

    @pytest.fixture
    def sample_spectrum_star_bb_spectrum(self):
        cfg = configparser.ConfigParser()
        cfg["target"] = {"rad_star": "1.0", "T_star": "5778"}
        cfg["dirs"] = {"save_s2n_data_unique_dir": "/tmp"}  # only needed if plot=True
        wavelength_um = np.logspace(0, np.log10(20), 100) * u.um
        luminosity_photons_star, _ = generate_star_spectrum(cfg, wavelength_um, plot=False)
        return SpectralData(
            wavelength=wavelength_um.value,
            flux=luminosity_photons_star.value,
            wavelength_unit="um",
            flux_unit=str(luminosity_photons_star.unit),  # "ph / (s um)"
            source_name="star_bb_test",
            metadata={"generator": "generate_star_spectrum"},
        )

    @pytest.fixture
    def sample_spectrum_bad_units(self):
        wavelength = np.array([1.0, 2.0, 3.0])
        flux = np.array([10.0, 20.0, 30.0])
        return SpectralData(
            wavelength=wavelength,
            flux=flux,
            wavelength_unit="um",
            flux_unit="W / (um m2)",
            source_name="test",
            metadata={},
        )
    def test_load_spectra_populates_dict(
        self, config_with_sources, unit_converter, sample_spectrum_star
    ):
        with patch(
            "modules.core.astrophysical.load_spectrum_from_file",
            return_value=sample_spectrum_star,
        ) as loader:
            sources = AstrophysicalSources(config_with_sources, unit_converter)

        assert set(sources.spectra.keys()) == {"star", "exozodiacal", "zodiacal"}
        assert loader.call_count == 3
        loader.assert_any_call("/tmp/star.txt")
        loader.assert_any_call("/tmp/exozodi.txt")
        loader.assert_any_call("/tmp/zodi.txt")

    def test_load_spectra_missing_section_logs_warning(self, config_no_sources, unit_converter, caplog):
        with caplog.at_level("WARNING"):
            sources = AstrophysicalSources(config_no_sources, unit_converter)

        assert sources.spectra == {}
        assert "No [astrophysical_sources_library] section found in config file." in caplog.text


    def test_calculate_flux_from_spectrum_zodiacal_no_distance(
        self, config_with_sources, unit_converter, sample_spectrum_zodiacal
    ):
        sources = AstrophysicalSources(config_with_sources, unit_converter)
        sources.spectra["zodiacal"] = sample_spectrum_zodiacal

        wavelength = np.array([1.5, 2.5]) * u.um
        flux = sources._calculate_flux_from_spectrum("zodiacal", wavelength, distance_set=10.0) # distance_set is ignored for zodiacal

        assert flux.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        assert np.allclose(flux.value, np.array([15.0, 25.0]))


    def test_calculate_flux_from_ad_hoc_spectrum_applies_distance(
        self, config_with_sources, unit_converter, sample_spectrum_star
    ):
        # checks if the distance correction is applied correctly (emitted spectrum is completely ad hoc)
        
        
        sources = AstrophysicalSources(config_with_sources, unit_converter)
        sources.spectra["star"] = sample_spectrum_star
        wavelength = np.array([1.5, 2.5]) * u.um

        # check distances < and > 10 pc
        for dist_check in [5.7, 21.6]:

            # function to predict incident flux
            flux_incident_predicted = sources._calculate_flux_from_spectrum("star", wavelength, distance_set=dist_check)

            # independently generate the incident flux
            spectrum = sample_spectrum_star
            interpolated_spectrum = spectrum.interpolate(wavelength)
            flux_unit_obj = u.Unit(interpolated_spectrum.flux_unit)
            flux_incident = interpolated_spectrum.flux * flux_unit_obj
            distance_correction = 1.0 / (4.0 * np.pi * (dist_check * u.pc) ** 2)
            flux_incident_expected = (flux_incident * distance_correction).to(u.ph / (u.um * u.m**2 * u.s))

            assert flux_incident_predicted.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
            assert flux_incident_expected.unit.is_equivalent(flux_incident_predicted.unit)
            assert np.allclose(flux_incident_predicted.value, flux_incident_expected.value)


    def test_calculate_flux_from_bb_spectrum_applies_distance(
        self, config_with_sources, unit_converter, sample_spectrum_star
    ):
        # checks if the distance correction is applied correctly (emitted spectrum is BB)

        # config with star like Sun
        cfg_ersatz = configparser.ConfigParser()
        cfg_ersatz["target"] = {"rad_star": "1.0", "T_star": "5778"}
        cfg_ersatz["dirs"] = {"save_s2n_data_unique_dir": "/tmp"}  # only needed if plot=True
        sources = AstrophysicalSources(cfg_ersatz, unit_converter)

        wavelength_grid = np.array([1.0, 5.0]) * u.um
        lum_phot_star, _ = generate_star_spectrum(cfg_ersatz, wavelength_grid, plot=False)
        sources.spectra["star"] = SpectralData(
            wavelength=wavelength_grid.value,
            flux=lum_phot_star.value,
            wavelength_unit="um",
            flux_unit=str(lum_phot_star.unit),  # ph / (s um)
            source_name="star_bb_test",
            metadata={},
        )

        
        dist_1 = 5.7 # pc
        dist_2 = 21.6 # pc
        incident_predicted_0 = sources._calculate_flux_from_spectrum("star", wavelength_grid, distance_set=dist_1)
        incident_predicted_1 = sources._calculate_flux_from_spectrum("star", wavelength_grid, distance_set=dist_2)

        # expected vals from Google Gemini, for 10 pc
        incident_expected_dist_10 = np.array([8.66e8, 2.37e7]) # units (u.ph / (u.um * u.m**2 * u.s))
        incident_expected_dist_5pt7 = incident_expected_dist_10 * (10.0/ dist_1) ** 2 # rescale for distance
        incident_expected_dist_21pt6 = incident_expected_dist_10 * (10.0/ dist_2) ** 2 # rescale for distance

        assert incident_predicted_0.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        assert incident_predicted_1.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        assert np.allclose(incident_predicted_0.value, incident_expected_dist_5pt7, rtol=1e-2) # 1% tolerance
        assert np.allclose(incident_predicted_1.value, incident_expected_dist_21pt6, rtol=1e-2) # 1% tolerance


    def test_calculate_flux_from_planet_bb_spectrum_at_10pc(
        self, unit_converter
    ):
        # checks BB exoplanet incident flux for R_earth, T=283K at 10 pc
        cfg_ersatz = configparser.ConfigParser()
        cfg_ersatz["target"] = {
            "rad_planet": "1.0",
            "pl_temp": "283.0",
            "planet_source": "BB",
        }
        cfg_ersatz["dirs"] = {"save_s2n_data_unique_dir": "/tmp"}  # only needed if plot=True
        sources = AstrophysicalSources(cfg_ersatz, unit_converter)

        wavelength_grid = np.array([1.0, 5.0]) * u.um
        lum_phot_planet, _ = generate_planet_bb_spectrum(cfg_ersatz, wavelength_grid, plot=False)
        sources.spectra["exoplanet_bb"] = SpectralData(
            wavelength=wavelength_grid.value,
            flux=lum_phot_planet.value,
            wavelength_unit="um",
            flux_unit=str(lum_phot_planet.unit),  # ph / (s um)
            source_name="planet_bb_test",
            metadata={},
        )

        dist_pc = 10.0
        incident_predicted = sources._calculate_flux_from_spectrum(
            "exoplanet_bb", wavelength_grid, distance_set=dist_pc
        )

        # expected vals from Google Gemini, for 10 pc
        incident_expected_dist_10 = np.array([6.7e-17, 4.9e-2]) # units (u.ph / (u.um * u.m**2 * u.s))

        assert incident_predicted.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        assert np.all(np.isfinite(incident_predicted.value))
        assert np.all(incident_predicted.value >= 0.0)
        assert np.allclose(incident_predicted.value, incident_expected_dist_10, rtol=1e-2)



    def test_calculate_flux_from_spectrum_star_bb_spectrum_at_source(
        self, unit_converter, sample_spectrum_star_bb_spectrum
    ):
        # check if the emitted stellar flux makes physical sense (note this is not the incident flux at Earth)

        # config with star like Sun
        cfg_ersatz = configparser.ConfigParser()
        cfg_ersatz["target"] = {"rad_star": "1.0", "T_star": "5778"}
        cfg_ersatz["dirs"] = {"save_s2n_data_unique_dir": "/tmp"}  # only needed if plot=True

        #sources = AstrophysicalSources(cfg_ersatz, unit_converter)
        #sources.spectra["star_bb_test"] = sample_spectrum_star_bb_spectrum
        wavelength = np.array([1.00, 5.00]) * u.um

        lum_phot_star, lum_energy_star = generate_star_spectrum(cfg_ersatz, wavelength, plot=False)

        assert lum_phot_star.unit.is_equivalent(u.ph / (u.um * u.s))
        assert lum_energy_star.unit.is_equivalent(u.W / u.um)

        l_phot_mine = np.array([float(f"{x:.2e}") for x in lum_phot_star.value])
        l_energy_mine = np.array([float(f"{x:.2e}") for x in lum_energy_star.value])
        l_phot_expected = np.array([1.04e45, 2.84e43]) # expected vals from Google Gemini
        l_energy_expected = np.array([2.06e26, 1.13e24])

        assert np.allclose(np.round(l_phot_mine, 2), l_phot_expected, rtol=1e-2) # 1% tolerance
        assert np.allclose(np.round(l_energy_mine, 2), l_energy_expected, rtol=1e-2)

        #assert np.allclose(lum_phot_star.value, np.array([6.59e14, 1.31e-4]))
        #assert np.allclose(lum_energy_star.value, np.array([2.31e11, 9.19e-9]))

        #flux = sources._calculate_flux_from_spectrum("star_bb_test", wavelength, distance_set=10.0)
        #assert flux.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        #assert np.allclose(flux.value, np.array([15.0, 25.0]))


    def test_calculate_flux_from_spectrum_logs_warning_for_inconsistent_units(
        self, config_with_sources, unit_converter, sample_spectrum_bad_units, caplog
    ):
        sources = AstrophysicalSources(config_with_sources, unit_converter)
        sources.spectra["zodiacal"] = sample_spectrum_bad_units
        wavelength = np.array([1.5, 2.5]) * u.um

        with caplog.at_level("WARNING"):
            with pytest.raises(UnitConversionError):
                sources._calculate_flux_from_spectrum(
                    "zodiacal", wavelength, distance_set=10.0
                )

        assert "Flux units not consistent for source: zodiacal" in caplog.text

    @pytest.fixture
    def config_for_incident_flux(self):
        """ConfigParser config for calculate_incident_flux source branches."""
        config = configparser.ConfigParser()
        config.add_section("nulling")
        config.set("nulling", "null", "True")
        config.set("nulling", "nulling_factor", "0.00001")
        config.add_section("wavelength_range")
        config.set("wavelength_range", "min", "1.0")
        config.set("wavelength_range", "max", "3.0")
        config.set("wavelength_range", "n_points", "3")
        config.add_section("target")
        config.set("target", "distance", "10.0")
        config.add_section("dirs")
        config.set("dirs", "save_s2n_data_unique_dir", "/tmp/")
        config.add_section("astrophysical_sources_library")
        config.set("astrophysical_sources_library", "star", "/tmp/star.txt")
        config.set("astrophysical_sources_library", "exoplanet_bb", "/tmp/exoplanet_bb.txt")
        config.set("astrophysical_sources_library", "exozodiacal", "/tmp/exozodiacal.txt")
        config.set("astrophysical_sources_library", "zodiacal", "/tmp/zodiacal.txt")
        return config

    @pytest.mark.parametrize("source_name", ["star", "exoplanet_bb", "exozodiacal", "zodiacal"])
    def test_calculate_incident_flux_source_branch_returns_expected_units_and_values(
        self, config_for_incident_flux, unit_converter, source_name
    ):
        expected_flux = np.array([11.0, 22.0, 33.0]) * u.ph / (u.um * u.m**2 * u.s)

        with patch("modules.core.astrophysical.load_spectrum_from_file"):
            sources = AstrophysicalSources(config_for_incident_flux, unit_converter)

        with patch.object(
            sources, "_calculate_flux_from_spectrum", return_value=expected_flux
        ) as flux_mock:
            incident = sources.calculate_incident_flux(source_name=source_name, plot=False)

        # Assert call contract at line 143.
        flux_mock.assert_called_once()
        call_args = flux_mock.call_args
        assert call_args.args[0] == source_name
        assert np.isclose(call_args.kwargs["distance_set"], 10.0)
        assert call_args.kwargs["null"] is True

        # Assert final answer and units: photons/sec/m^2/micron.
        final_flux = incident["pre_screen_astro_flux_ph_sec_m2_um"]
        assert final_flux.unit.is_equivalent(u.ph / (u.s * u.m**2 * u.micron))
        assert np.allclose(final_flux.to(u.ph / (u.s * u.m**2 * u.micron)).value, [11.0, 22.0, 33.0])

    def test_calculate_incident_flux_exoplanet_model_10pc(self, unit_converter):
        """Test exoplanet_model_10pc branch using a mocked model dataframe."""

        # ersatz config file
        config = configparser.ConfigParser()
        config.add_section("nulling")
        config.set("nulling", "null", "True")
        config.set("nulling", "nulling_factor", "0.00001")
        config.add_section("wavelength_range")
        config.set("wavelength_range", "min", "1.0")
        config.set("wavelength_range", "max", "3.0")
        config.set("wavelength_range", "n_points", "3")
        config.add_section("target")
        config.set("target", "distance", "20.0")
        config.add_section("astrophysical_sources_library")
        config.set("astrophysical_sources_library", "exoplanet_model_10pc", "/tmp/model_10pc.txt")

        df_model = pd.DataFrame(
            {
                "wavelength": [1.0, 2.0, 3.0],
                "flux": [1e-18, 2e-18, 3e-18],
                "err_flux": [1e-20, 1e-20, 1e-20],
            }
        )

        # temporarily replace some functions with mock functions
        with patch("modules.core.astrophysical.load_spectrum_from_file"), patch(
            "modules.core.astrophysical.pd.read_csv", return_value=df_model
        ):
            sources = AstrophysicalSources(config, unit_converter)
            incident = sources.calculate_incident_flux(
                source_name="exoplanet_model_10pc", plot=False
            )

        wavel = df_model["wavelength"].values * u.micron
        flux_nu_10pc = df_model["flux"].values * u.erg / (u.second * u.Hz * u.m**2)

        # use astropy to convert units independently
        flux_photons_10pc = flux_nu_10pc.to(
            u.ph / (u.s * u.m**2 * u.micron),
            equivalencies=u.spectral_density(wavel),
        )

        empirical = incident["pre_screen_astro_flux_ph_sec_m2_um"]

        # calculate_incident_flux always uses 100 wavelength points (ignores n_points in config)
        wavelength = np.linspace(1.0, 3.0, 100) * u.micron
        flux_photons_expected = flux_photons_10pc * (10.0 / 20.0) ** 2
        expected = np.interp(
            x=wavelength.value,
            xp=wavel.value,
            fp=flux_photons_expected.to_value(u.ph / (u.s * u.m**2 * u.micron)),
        )

        flux_photons_expected_bogus = flux_photons_10pc * (10.0 / 21.0) ** 2
        expected_bogus = np.interp(
            x=wavelength.value,
            xp=wavel.value,
            fp=flux_photons_expected_bogus.to_value(u.ph / (u.s * u.m**2 * u.micron)),
        )

        empirical_values = empirical.to_value(u.ph / (u.s * u.m**2 * u.micron))
        assert np.allclose(empirical_values, expected)
        assert not np.allclose(empirical_values, expected_bogus)

    def test_calculate_incident_flux_unknown_source_returns_empty_array(
        self, config_for_incident_flux, unit_converter
    ):
        with patch("modules.core.astrophysical.load_spectrum_from_file"):
            sources = AstrophysicalSources(config_for_incident_flux, unit_converter)

        result = sources.calculate_incident_flux(source_name="not_a_source", plot=False)

        assert isinstance(result, np.ndarray)
        assert result.size == 0

    @patch("modules.core.astrophysical.plt.savefig")
    @patch("modules.core.astrophysical.plt.plot")
    def test_calculate_incident_flux_plot_saves_figure(
        self, mock_plot, mock_savefig, config_for_incident_flux, unit_converter
    ):
        expected_flux = np.array([11.0, 22.0, 33.0]) * u.ph / (u.um * u.m**2 * u.s)

        with patch("modules.core.astrophysical.load_spectrum_from_file"):
            sources = AstrophysicalSources(config_for_incident_flux, unit_converter)

        with patch.object(
            sources, "_calculate_flux_from_spectrum", return_value=expected_flux
        ):
            incident = sources.calculate_incident_flux(source_name="star", plot=True)

        assert "pre_screen_astro_flux_ph_sec_m2_um" in incident
        mock_plot.assert_called()
        mock_savefig.assert_called_once()
        assert mock_savefig.call_args.args[0].endswith("incident_star.png")


class TestBoxKernel:
    def test_centered_kernel_shape_and_normalization(self):
        kernel = _box_kernel(n_pix=10, idx_y=5, idx_x=5, half_pix=1)

        assert kernel.shape == (10, 10)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.count_nonzero(kernel) == 9
        assert np.all(kernel[4:7, 4:7] == pytest.approx(1.0 / 9.0))
        assert kernel[0, 0] == 0.0

    def test_box_side_length_is_two_half_pix_plus_one_when_unclipped(self):
        kernel = _box_kernel(n_pix=100, idx_y=50, idx_x=50, half_pix=2)

        assert np.count_nonzero(kernel) == 25
        assert kernel[48:53, 48:53].sum() == pytest.approx(1.0)
        assert kernel[47, 50] == 0.0
        assert kernel[53, 50] == 0.0

    def test_clips_at_scene_edges(self):
        kernel = _box_kernel(n_pix=10, idx_y=0, idx_x=0, half_pix=2)

        assert np.count_nonzero(kernel) == 9
        assert kernel[0:3, 0:3].sum() == pytest.approx(1.0)
        assert kernel[3:, :].sum() == 0.0
        assert kernel[:, 3:].sum() == 0.0

    def test_clips_at_far_corner(self):
        kernel = _box_kernel(n_pix=10, idx_y=9, idx_x=9, half_pix=2)

        assert np.count_nonzero(kernel) == 9
        assert kernel[7:10, 7:10].sum() == pytest.approx(1.0)
        assert kernel[:7, :].sum() == 0.0


class TestExozodiKernel:
    def test_shape_and_normalization(self):
        kernel = _exozodi_kernel(
            n_pix=11,
            idx_y=5,
            idx_x=5,
            pix_size_arcsec=0.05,
            dist_pc=10.0,
        )

        assert kernel.shape == (11, 11)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.all(kernel >= 0.0)

    def test_center_is_peak_and_radial_falloff(self):
        kernel = _exozodi_kernel(
            n_pix=11,
            idx_y=5,
            idx_x=5,
            pix_size_arcsec=0.1,
            dist_pc=10.0,
            alpha=0.34,
        )
        center = 11 // 2

        # r=0 is non-finite in the power law; implementation assigns it the peak ring value.
        assert kernel[center, center] == pytest.approx(np.max(kernel))
        assert kernel[center, center] >= kernel[center, center + 1]
        assert kernel[center, center] >= kernel[center + 1, center]
        assert kernel[center, center] > kernel[0, 0]

    def test_doubling_z_exozodiacal_preserves_normalized_shape(self):
        base = _exozodi_kernel(
            n_pix=9,
            idx_y=4,
            idx_x=4,
            pix_size_arcsec=0.1,
            dist_pc=10.0,
            z_exozodiacal=1,
        )
        scaled = _exozodi_kernel(
            n_pix=9,
            idx_y=4,
            idx_x=4,
            pix_size_arcsec=0.1,
            dist_pc=10.0,
            z_exozodiacal=2,
        )

        assert np.allclose(base, scaled)


class TestZodiacalKernel:
    def test_uniform_normalized_kernel(self):
        n_pix = 13
        kernel = _zodiacal_kernel(n_pix=n_pix, pix_size_arcsec=0.05)

        assert kernel.shape == (n_pix, n_pix)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.all(kernel == pytest.approx(1.0 / n_pix**2))

    def test_pix_size_does_not_affect_kernel(self):
        kernel_a = _zodiacal_kernel(n_pix=7, pix_size_arcsec=0.01)
        kernel_b = _zodiacal_kernel(n_pix=7, pix_size_arcsec=1.0)

        assert np.allclose(kernel_a, kernel_b)


@pytest.fixture
def sample_flux_1d():
    return np.array([1.0, 2.0, 3.0]) * u.ph / (u.um * u.m**2 * u.s)


def _assert_quantity_close(actual, expected, rtol=1e-10):
    assert actual.unit.is_equivalent(expected.unit)
    assert np.isclose(actual.to(expected.unit).value, expected.value, rtol=rtol)


class TestGenerateStarScene:
    def test_spreads_flux_with_normalized_box_kernel(self, sample_flux_1d):
        scene = generate_star_scene(
            sample_flux_1d, n_pix=11, idx_y_star=5, idx_x_star=5, half_pix=1
        )

        assert scene.shape == (3, 11, 11)
        assert scene.unit == sample_flux_1d.unit
        for i in range(3):
            _assert_quantity_close(scene[i].sum(), sample_flux_1d[i])
        assert scene[0, 5, 5] > scene[0, 0, 0]


class TestGenerateExoplanetScene:
    def test_spreads_flux_onto_planet_pixel(self, sample_flux_1d):
        scene = generate_exoplanet_scene(
            sample_flux_1d, n_pix=11, idx_y_planet=6, idx_x_planet=4, half_pix=1
        )

        assert scene.shape == (3, 11, 11)
        assert scene[0, 6, 4] > scene[0, 0, 0]
        _assert_quantity_close(scene[0].sum(), sample_flux_1d[0])


class TestGenerateExozodiScene:
    def test_spreads_flux_with_exozodi_kernel(self, sample_flux_1d):
        scene = generate_exozodi_scene(
            sample_flux_1d,
            n_pix=11,
            idx_y_exozodi=5,
            idx_x_exozodi=5,
            pix_size_arcsec=0.1,
            dist_pc=10.0,
            z_exozodiacal=1.0,
        )

        assert scene.shape == (3, 11, 11)
        _assert_quantity_close(scene[0].sum(), sample_flux_1d[0])
        _assert_quantity_close(scene[0, 5, 5], np.max(scene[0]))


class TestGenerateZodiacalScene:
    def test_spreads_flux_uniformly(self, sample_flux_1d):
        n_pix = 9
        scene = generate_zodiacal_scene(
            sample_flux_1d, n_pix=n_pix, pix_size_arcsec=0.05
        )

        assert scene.shape == (3, n_pix, n_pix)
        expected_pixel = sample_flux_1d[0] / n_pix**2
        _assert_quantity_close(scene[0].flat[0], expected_pixel)
        _assert_quantity_close(scene[0].sum(), sample_flux_1d[0])


class TestGenerateOnskyScene:
    @staticmethod
    def _mock_subplots_axes(n_cols: int = 5):
        """Mimic plt.subplots(..., squeeze=False): shape (1, n_cols) object array of Axes."""
        fig = MagicMock()
        axes = np.empty((1, n_cols), dtype=object)
        for j in range(n_cols):
            ax = MagicMock(name=f"ax_{j}")
            ax.imshow.return_value = MagicMock(name=f"im_{j}")
            axes[0, j] = ax
        return fig, axes

    @pytest.fixture
    def onsky_config(self, tmp_path):
        config = configparser.ConfigParser()
        config.add_section("onsky_scene")
        config.set("onsky_scene", "n_pix", "11")
        config.set("onsky_scene", "pix_size_mas", "100")
        config.set("onsky_scene", "pos_star_arcsec", "0.0, 0.0")
        config.set("onsky_scene", "pos_planet_arcsec", "0.1, 0.1")
        config.set("onsky_scene", "half_pix", "1")
        config.add_section("target")
        config.set("target", "distance", "10.0")
        config.set("target", "z_exozodiacal", "1.0")
        config.add_section("dirs")
        config.set("dirs", "save_s2n_data_unique_dir", str(tmp_path / "out") + "/")
        return config

    @pytest.fixture
    def incident_dict(self):
        wavel = np.linspace(4.0, 18.0, 5) * u.um
        flux_unit = u.ph / (u.um * u.m**2 * u.s)
        return {
            "star": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.ones(5) * flux_unit,
            },
            "exoplanet_model_10pc": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.full(5, 2.0) * flux_unit,
            },
            "exozodiacal": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.full(5, 3.0) * flux_unit,
            },
            "zodiacal": {
                "wavel": wavel,
                "pre_screen_astro_flux_ph_sec_m2_um": np.full(5, 4.0) * flux_unit,
            },
        }

    @patch("modules.core.astrophysical.plt.close")
    @patch("modules.core.astrophysical.plt.colorbar")
    @patch("modules.core.astrophysical.fits.HDUList.writeto")
    @patch("modules.core.astrophysical.plt.savefig")
    @patch("modules.core.astrophysical.plt.subplots")
    def test_builds_scene_for_all_sources(
        self, mock_subplots, mock_savefig, mock_writeto, mock_colorbar, mock_close,
        onsky_config, incident_dict, unit_converter
    ):
        mock_subplots.return_value = self._mock_subplots_axes()

        with patch("modules.core.astrophysical.load_spectrum_from_file"):
            sources = AstrophysicalSources(onsky_config, unit_converter)

        scene = sources.generate_onsky_scene(incident_dict, plot=False)

        assert set(scene.keys()) == {
            "star",
            "exoplanet_model_10pc",
            "exozodiacal",
            "zodiacal",
        }
        for cube in scene.values():
            assert cube.shape == (5, 11, 11)

    @patch("modules.core.astrophysical.fits.HDUList.writeto")
    @patch("modules.core.astrophysical.plt.savefig")
    @patch("modules.core.astrophysical.plt.subplots")
    def test_raises_when_source_flux_units_differ(
        self, mock_subplots, mock_savefig, mock_writeto, onsky_config, incident_dict, unit_converter
    ):
        mock_subplots.return_value = self._mock_subplots_axes()
        incident_dict["exozodiacal"]["pre_screen_astro_flux_ph_sec_m2_um"] = (
            np.ones(5) * u.ph / (u.um * u.s)
        )

        with patch("modules.core.astrophysical.load_spectrum_from_file"):
            sources = AstrophysicalSources(onsky_config, unit_converter)

        with pytest.raises(ValueError, match="flux units differ"):
            sources.generate_onsky_scene(incident_dict, plot=False)
