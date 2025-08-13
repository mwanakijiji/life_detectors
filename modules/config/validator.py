"""
Configuration validator for the modules package.

This module validates configuration files to ensure all required
parameters are present and have valid values.
"""

from typing import Dict, Any, List, Optional
import logging
import ipdb
import numpy as np

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration dictionaries for the modules package."""
    
    def __init__(self):
        """Initialize the validator with required fields."""
        self.required_sections = [
            "telescope",
            "target", 
            "nulling",
            "detector",
            "observation",
            "wavelength_range"
        ]
        
        self.required_telescope_fields = [
            "collecting_area",
            "plate_scale", 
            "throughput"
        ]
        
        self.required_target_fields = [
            "distance",
            "pl_temp"
        ]

        self.required_nulling_fields = [
            "distance",
            "pl_temp"
        ]
        
        self.required_detector_fields = [
            "read_noise",
            "dark_current",
            "gain",
            "quantum_efficiency",
            "spec_res"
        ]

        self.required_observation_fields = [
            "integration_time",
            "n_int"
        ]
        
        self.required_wavelength_fields = [
            "min",
            "max",
            "n_points"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        
        # Check for required sections
        for section in self.required_sections:
            if section not in config:
                logging.error(f"Missing required section: {section}")
        
        # Validate telescope section
        if "telescope" in config:
            self._validate_telescope(config["telescope"])
        ipdb.set_trace()
        
        # Validate target section
        if "target" in config:
            self._validate_target(config["target"])
        ipdb.set_trace()
        
        # Validate detector section
        if "detector" in config:
            self._validate_detector(config["detector"])
        ipdb.set_trace()
        
        # Validate wavelength range
        if "wavelength_range" in config:
            self._validate_wavelength_range(config["wavelength_range"])
        ipdb.set_trace()
        
        # Validate astrophysical sources
        if "astrophysical_sources" in config:
            self._validate_astrophysical_sources(config["astrophysical_sources"])
        ipdb.set_trace()
        
        return
    
    def _validate_telescope(self, telescope_config: Dict[str, Any]) -> List[str]:
        """Validate telescope configuration section."""
        
        for field in self.required_telescope_fields:
            if field not in telescope_config:
                logging.error(f"Missing telescope field: {field}")
            else:
                value = telescope_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    # all values here should be convertable to floats
                    logging.error(f"Invalid telescope {field}: must be positive number")
        
        return
    
    def _validate_target(self, target_config: Dict[str, Any]) -> List[str]:
        """Validate target configuration section."""
        
        for field in self.required_target_fields:
            if field not in target_config:
                logging.error(f"Missing target field: {field}")
            else:
                value = target_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    # all values here should be convertable to floats
                    logging.error(f"Invalid target {field}: must be positive number")
        
        return
    
    def _validate_detector(self, detector_config: Dict[str, Any]) -> List[str]:
        """Validate detector configuration section."""
        
        for field in self.required_detector_fields:
            if field not in detector_config:
                logging.error(f"Missing detector field: {field}")
        
        return
    
    def _validate_wavelength_range(self, wavelength_config: Dict[str, Any]) -> List[str]:
        """Validate wavelength range configuration section."""
        
        for field in self.required_wavelength_fields:
            if field not in wavelength_config:
                logging.error(f"Missing wavelength_range field: {field}")
            else:
                value = wavelength_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    logging.error(f"Invalid wavelength_range {field}: must be positive number")
        
        # Check that min < max
        if "min" in wavelength_config and "max" in wavelength_config:
            if float(wavelength_config["min"]) >= float(wavelength_config["max"]):
                logging.error("wavelength_range min must be less than max")
        
        return
    
    def _validate_astrophysical_sources(self, sources_config: Dict[str, Any]) -> List[str]:
        """Validate astrophysical sources configuration section."""
        
        expected_sources = ["star", "exoplanet", "exozodiacal", "zodiacal"]
        
        for source in expected_sources:
            if source not in sources_config:
                logging.error(f"Missing astrophysical source: {source}")
        
        return


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If configuration is invalid (with detailed error messages)
    """
    validator = ConfigValidator()
    errors = validator.validate_config(config)
    
    #if errors:
    #    error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
    #     logging.warning(error_msg)
    
    return