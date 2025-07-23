"""
Unit tests for astrophysical noise calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from modules.core.astrophysical import AstrophysicalSources
from modules.data.units import UnitConverter
from modules.data.spectra import SpectralData

class TestAstrophysicalSources:
    """Test cases for AstrophysicalSources class."""
    
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
                "gain": 2.0,
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
            },
        }
    
    @pytest.fixture
    def unit_converter(self):
        """Create a unit converter for testing."""
        return UnitConverter()
    
    @pytest.fixture
    def sample_spectrum(self):
        """Create a sample spectrum for testing."""
        wavelength = np.linspace(1.0, 10.0, 100)
        flux = 1e10 * np.exp(-wavelength / 2.0)  # Simple exponential
        return SpectralData(wavelength=wavelength, flux=flux, source_name="test")
    
    def test_init(self, mock_config, unit_converter):
        """Test initialization of AstrophysicalSources."""
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
            noise_calc = AstrophysicalSources(mock_config, unit_converter)
            assert noise_calc.config == mock_config
            assert noise_calc.unit_converter == unit_converter
    
    def test_calculate_source_flux(self, mock_config, unit_converter, sample_spectrum):
        """Test calculation of source flux."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        flux = noise_calc.calculate_source_flux("star", wavelength)
        
        assert isinstance(flux, np.ndarray)
        assert flux.shape == wavelength.shape
        assert np.all(flux >= 0)  # Flux should be non-negative
    
    def test_calculate_total_astrophysical_flux(self, mock_config, unit_converter, sample_spectrum):
        """Test calculation of total astrophysical flux."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        noise_calc.spectra["exoplanet"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        total_flux = noise_calc.calculate_total_astrophysical_flux(wavelength)
        
        assert isinstance(total_flux, np.ndarray)
        assert total_flux.shape == wavelength.shape
        assert np.all(total_flux >= 0)
    
    def test_calculate_detector_illumination(self, mock_config, unit_converter, sample_spectrum):
        """Test calculation of detector illumination."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        illumination = noise_calc.calculate_detector_illumination(wavelength)
        
        assert isinstance(illumination, np.ndarray)
        assert illumination.shape == wavelength.shape
        assert np.all(illumination >= 0)
    
    def test_calculate_astrophysical_noise_electrons(self, mock_config, unit_converter, sample_spectrum):
        """Test calculation of astrophysical noise in electrons."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        integration_time = 3600.0
        
        noise = noise_calc.calculate_astrophysical_noise_electrons(wavelength, integration_time)
        
        assert isinstance(noise, np.ndarray)
        assert noise.shape == wavelength.shape
        assert np.all(noise >= 0)
    
    def test_calculate_astrophysical_noise_adu(self, mock_config, unit_converter, sample_spectrum):
        """Test calculation of astrophysical noise in ADU."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        integration_time = 3600.0
        
        noise = noise_calc.calculate_astrophysical_noise_adu(wavelength, integration_time)
        
        assert isinstance(noise, np.ndarray)
        assert noise.shape == wavelength.shape
        assert np.all(noise >= 0)
    
    def test_get_source_contributions(self, mock_config, unit_converter, sample_spectrum):
        """Test getting source contributions."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        noise_calc.spectra["star"] = sample_spectrum
        noise_calc.spectra["exoplanet"] = sample_spectrum
        
        wavelength = np.array([2.0, 5.0, 8.0])
        contributions = noise_calc.get_source_contributions(wavelength)
        
        assert isinstance(contributions, dict)
        assert "star" in contributions
        assert "exoplanet" in contributions
        assert all(isinstance(flux, np.ndarray) for flux in contributions.values())
    
    def test_missing_spectrum(self, mock_config, unit_converter):
        """Test behavior when spectrum is missing."""
        noise_calc = AstrophysicalSources(mock_config, unit_converter)
        
        wavelength = np.array([2.0, 5.0, 8.0])
        flux = noise_calc.calculate_source_flux("missing_source", wavelength)
        
        assert isinstance(flux, np.ndarray)
        assert np.all(flux == 0)  # Should return zeros for missing spectrum 