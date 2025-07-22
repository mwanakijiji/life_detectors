"""
Unit tests for the main noise calculator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from life_detectors.core.calculator import NoiseCalculator

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
                "integration_time": 3600,
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
    
    def test_init(self, mock_config):
        """Test initialization of NoiseCalculator."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            assert calculator.config == mock_config
            assert len(calculator.wavelength) == 100
    
    def test_generate_wavelength_grid(self, mock_config):
        """Test wavelength grid generation."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            
            assert len(calculator.wavelength) == 100
            assert np.min(calculator.wavelength) >= 1.0
            assert np.max(calculator.wavelength) <= 10.0
            assert np.all(np.diff(calculator.wavelength) > 0)  # Monotonically increasing
    
    def test_calculate_snr(self, mock_config):
        """Test SNR calculation."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            results = calculator.calculate_snr()
            
            assert isinstance(results, dict)
            assert "wavelength" in results
            assert "signal_to_noise" in results
            assert "integrated_snr" in results
            assert "total_noise_adu" in results
            assert "astrophysical_noise_adu" in results
            assert "instrumental_noise_adu" in results
    
    def test_calculate_noise_budget(self, mock_config):
        """Test noise budget calculation."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            budget = calculator.calculate_noise_budget()
            
            assert isinstance(budget, dict)
            assert "wavelength" in budget
            assert "total_astrophysical_noise" in budget
            assert "total_instrumental_noise" in budget
            assert "total_noise" in budget
            assert "astrophysical_breakdown" in budget
            assert "instrumental_breakdown" in budget
    
    def test_calculate_optimal_parameters(self, mock_config):
        """Test optimal parameter calculation."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            optimal_params = calculator.calculate_optimal_parameters(target_snr=10.0)
            
            assert isinstance(optimal_params, dict)
            assert "current_snr" in optimal_params
            assert "target_snr" in optimal_params
            assert "required_integration_time" in optimal_params
            assert "optimal_integration_time" in optimal_params
    
    def test_get_summary(self, mock_config):
        """Test summary generation."""
        with patch('life_detectors.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            summary = calculator.get_summary()
            
            assert isinstance(summary, dict)
            assert "integrated_snr" in summary
            assert "wavelength_range" in summary
            assert "integration_time" in summary
            assert "total_astrophysical_noise" in summary
            assert "total_instrumental_noise" in summary
            assert "total_noise" in summary
    
    def test_invalid_config(self):
        """Test behavior with invalid configuration."""
        invalid_config = {
            "telescope": {
                "collecting_area": -1.0,  # Invalid negative value
            },
        }
        
        with pytest.raises(ValueError):
            NoiseCalculator(invalid_config)
    
    def test_missing_required_sections(self):
        """Test behavior with missing required configuration sections."""
        incomplete_config = {
            "telescope": {
                "collecting_area": 25.0,
                "plate_scale": 0.1,
                "throughput": 0.8,
            },
            # Missing other required sections
        }
        
        with pytest.raises(ValueError):
            NoiseCalculator(incomplete_config)
    
    def test_wavelength_range_validation(self):
        """Test wavelength range validation."""
        config = {
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
                "integration_time": 3600,
            },
            "astrophysical_sources": {
                "star": {"spectrum_file": "test.txt", "enabled": True},
                "exoplanet": {"spectrum_file": "test.txt", "enabled": True},
                "exozodiacal": {"spectrum_file": "test.txt", "enabled": True},
                "zodiacal": {"spectrum_file": "test.txt", "enabled": True},
            },
            "instrumental_sources": {
                "dark_current": {"enabled": True},
                "read_noise": {"enabled": True},
            },
            "wavelength_range": {
                "min": 10.0,  # Invalid: min > max
                "max": 1.0,
                "n_points": 100,
            },
        }
        
        with pytest.raises(ValueError):
            NoiseCalculator(config) 