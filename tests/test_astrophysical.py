"""
Unit tests for astrophysical calculations in AstrophysicalSources.
"""

import configparser
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.modules["ipdb"].set_trace = lambda: None

from modules.core.astrophysical import AstrophysicalSources
from modules.data.spectra import SpectralData
from modules.data.units import UnitConverter


class TestAstrophysicalSources:
    """Test cases for AstrophysicalSources class."""

    @pytest.fixture
    def unit_converter(self):
        return UnitConverter()

    @pytest.fixture
    def config_with_sources(self):
        config = configparser.ConfigParser()
        config.add_section("nulling")
        config.set("nulling", "nulling_factor", "0.00001")
        config.add_section("astrophysical_sources")
        config.set("astrophysical_sources", "star", "/tmp/star.txt")
        config.set("astrophysical_sources", "exozodiacal", "/tmp/exozodi.txt")
        config.set("astrophysical_sources", "zodiacal", "/tmp/zodi.txt")
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
    '''
    def test_load_spectra_populates_dict(self, config_with_sources, unit_converter, sample_spectrum):
        with patch("modules.core.astrophysical.load_spectrum_from_file", return_value=sample_spectrum) as loader:
            sources = AstrophysicalSources(config_with_sources, unit_converter)

        assert "star" in sources.spectra
        assert "exozodiacal" in sources.spectra
        assert loader.call_count == 2
        loader.assert_any_call("/tmp/star.txt")
        loader.assert_any_call("/tmp/exozodi.txt")
    '''

    def test_load_spectra_missing_section_logs_warning(self, config_no_sources, unit_converter, caplog):
        with caplog.at_level("WARNING"):
            sources = AstrophysicalSources(config_no_sources, unit_converter)

        assert sources.spectra == {}
        assert "No [astrophysical_sources] section found in config file." in caplog.text


    def test_calculate_flux_from_spectrum_zodiacal_no_distance(
        self, config_with_sources, unit_converter, sample_spectrum_zodiacal
    ):
        sources = AstrophysicalSources(config_with_sources, unit_converter)
        sources.spectra["zodiacal"] = sample_spectrum_zodiacal

        wavelength = np.array([1.5, 2.5]) * u.um
        flux = sources._calculate_flux_from_spectrum("zodiacal", wavelength, distance_set=10.0) # distance_set is ignored for zodiacal

        assert flux.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))
        assert np.allclose(flux.value, np.array([15.0, 25.0]))


    def test_calculate_flux_from_spectrum_applies_distance(
        self, config_with_sources, unit_converter, sample_spectrum_star
    ):
        sources = AstrophysicalSources(config_with_sources, unit_converter)
        sources.spectra["star"] = sample_spectrum_star

        wavelength = np.array([1.5, 2.5]) * u.um
        flux = sources._calculate_flux_from_spectrum("star", wavelength, distance_set=10.0) # distance_set is ignored for zodiacal

        assert flux.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s))