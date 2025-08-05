"""
Unit tests for instrumental noise calculations.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from modules.core.instrumental import InstrumentalSources
from modules.data.units import UnitConverter

class TestInstrumentalSources:
    """Test cases for InstrumentalSources class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "detector": {
                "read_noise": 5.0,
                "dark_current": 0.1,
                "gain": 2.0,
            },
            "instrumental_sources": {
                "dark_current": {
                    "enabled": True,
                },
                "read_noise": {
                    "enabled": True,
                },
            },
        }
    
    @pytest.fixture
    def unit_converter(self):
        """Create a unit converter for testing."""
        return UnitConverter()
    
    def test_init(self, mock_config, unit_converter):
        """Test initialization of InstrumentalSources."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        assert noise_calc.config == mock_config
        assert noise_calc.unit_converter == unit_converter
    
    def test_calculate_dark_current_electrons(self, mock_config, unit_converter):
        """Test calculation of dark current noise in electrons."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_dark_current_electrons(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        # Should be sqrt(dark_current_rate * integration_time)
        expected = np.sqrt(0.1 * 3600.0)
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_dark_current_adu(self, mock_config, unit_converter):
        """Test calculation of dark current noise in ADU."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_dark_current_adu(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        # Should be (sqrt(dark_current_rate * integration_time)) / gain
        expected = np.sqrt(0.1 * 3600.0) / 2.0
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_read_noise_electrons(self, mock_config, unit_converter):
        """Test calculation of read noise in electrons."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        noise = noise_calc.calculate_read_noise_electrons()
        
        assert isinstance(noise, float)
        assert noise == 5.0  # Should match config value
    
    def test_calculate_read_noise_adu(self, mock_config, unit_converter):
        """Test calculation of read noise in ADU."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        noise = noise_calc.calculate_read_noise_adu()
        
        assert isinstance(noise, float)
        assert noise == 5.0 / 2.0  # Should be read_noise / gain
    
    def test_calculate_total_instrumental_noise_electrons(self, mock_config, unit_converter):
        """Test calculation of total instrumental noise in electrons."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_total_instrumental_noise_electrons(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        
        # Should be sqrt(dark_noise^2 + read_noise^2)
        dark_noise = np.sqrt(0.1 * 3600.0)
        read_noise = 5.0
        expected = np.sqrt(dark_noise**2 + read_noise**2)
        assert abs(noise - expected) < 1e-10
    
    def test_calculate_total_instrumental_noise_adu(self, mock_config, unit_converter):
        """Test calculation of total instrumental noise in ADU."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        noise = noise_calc.calculate_total_instrumental_noise_adu(integration_time)
        
        assert isinstance(noise, float)
        assert noise >= 0
        
        # Should be total_electron_noise / gain
        dark_noise = np.sqrt(0.1 * 3600.0)
        read_noise = 5.0
        total_electron_noise = np.sqrt(dark_noise**2 + read_noise**2)
        expected = total_electron_noise / 2.0
        assert abs(noise - expected) < 1e-10
    
    def test_get_noise_breakdown_electrons(self, mock_config, unit_converter):
        """Test getting noise breakdown in electrons."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        breakdown = noise_calc.get_noise_breakdown_electrons(integration_time)
        
        assert isinstance(breakdown, dict)
        assert "dark_current" in breakdown
        assert "read_noise" in breakdown
        assert all(isinstance(noise, float) for noise in breakdown.values())
        assert all(noise >= 0 for noise in breakdown.values())
    
    def test_get_noise_breakdown_adu(self, mock_config, unit_converter):
        """Test getting noise breakdown in ADU."""
        noise_calc = InstrumentalSources(mock_config, unit_converter)
        
        integration_time = 3600.0
        breakdown = noise_calc.get_noise_breakdown_adu(integration_time)
        
        assert isinstance(breakdown, dict)
        assert "dark_current" in breakdown
        assert "read_noise" in breakdown
        assert all(isinstance(noise, float) for noise in breakdown.values())
        assert all(noise >= 0 for noise in breakdown.values())
    
    def test_disabled_noise_sources(self):
        """Test behavior when noise sources are disabled."""
        config = {
            "detector": {
                "read_noise": 5.0,
                "dark_current": 0.1,
                "gain": 2.0,
            },
            "instrumental_sources": {
                "dark_current": {
                    "enabled": False,
                },
                "read_noise": {
                    "enabled": False,
                },
            },
        }
        
        noise_calc = InstrumentalSources(config, UnitConverter())
        
        integration_time = 3600.0
        total_noise = noise_calc.calculate_total_instrumental_noise_electrons(integration_time)
        
        assert total_noise == 0.0  # Should be zero when all sources disabled
    
    def test_partial_disabled_sources(self):
        """Test behavior when some noise sources are disabled."""
        config = {
            "detector": {
                "read_noise": 5.0,
                "dark_current": 0.1,
                "gain": 2.0,
            },
            "instrumental_sources": {
                "dark_current": {
                    "enabled": False,
                },
                "read_noise": {
                    "enabled": True,
                },
            },
        }
        
        noise_calc = InstrumentalSources(config, UnitConverter())
        
        integration_time = 3600.0
        total_noise = noise_calc.calculate_total_instrumental_noise_electrons(integration_time)
        
        assert total_noise == 5.0  # Should only include read noise 