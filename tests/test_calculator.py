"""
Unit tests for the main noise calculator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

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

    
    
    '''
    def test_calculate_noise_budget(self, mock_config):
        """Test noise budget calculation."""
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
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
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
            calculator = NoiseCalculator(mock_config)
            optimal_params = calculator.calculate_optimal_parameters(target_snr=10.0)
            
            assert isinstance(optimal_params, dict)
            assert "current_snr" in optimal_params
            assert "target_snr" in optimal_params
            assert "required_integration_time" in optimal_params
            assert "optimal_integration_time" in optimal_params
    
    def test_get_summary(self, mock_config):
        """Test summary generation."""
        with patch('modules.core.astrophysical.load_spectrum_from_file'):
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


    def test_init_sets_config_and_sources_references(self):
        
        config = {"observation": {"n_int": 10, "integration_time": 100}}
        sources_all = object()
        sources_to_include = ["star", "exoplanet_bb"]
        calc = NoiseCalculator(
            config=config,
            sources_all=sources_all,
            sources_to_include=sources_to_include,
        )
        assert calc.config is config
        assert calc.sources_all is sources_all
        assert calc.sources_to_include is sources_to_include

    
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
        '''