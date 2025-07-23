"""
Configuration validator for the modules package.

This module validates configuration files to ensure all required
parameters are present and have valid values.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration dictionaries for the modules package."""
    
    def __init__(self):
        """Initialize the validator with required fields."""
        self.required_sections = [
            "telescope",
            "target", 
            "detector",
            "astrophysical_sources",
            "instrumental_sources",
            "wavelength_range"
        ]
        
        self.required_telescope_fields = [
            "collecting_area",
            "plate_scale", 
            "throughput"
        ]
        
        self.required_target_fields = [
            "distance",
            "nulling_factor"
        ]
        
        self.required_detector_fields = [
            "read_noise",
            "dark_current",
            "gain",
            "integration_time"
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
        errors = []
        
        # Check for required sections
        for section in self.required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate telescope section
        if "telescope" in config:
            errors.extend(self._validate_telescope(config["telescope"]))
        
        # Validate target section
        if "target" in config:
            errors.extend(self._validate_target(config["target"]))
        
        # Validate detector section
        if "detector" in config:
            errors.extend(self._validate_detector(config["detector"]))
        
        # Validate wavelength range
        if "wavelength_range" in config:
            errors.extend(self._validate_wavelength_range(config["wavelength_range"]))
        
        # Validate astrophysical sources
        if "astrophysical_sources" in config:
            errors.extend(self._validate_astrophysical_sources(config["astrophysical_sources"]))
        
        # Validate instrumental sources
        if "instrumental_sources" in config:
            errors.extend(self._validate_instrumental_sources(config["instrumental_sources"]))
        
        return errors
    
    def _validate_telescope(self, telescope_config: Dict[str, Any]) -> List[str]:
        """Validate telescope configuration section."""
        errors = []
        
        for field in self.required_telescope_fields:
            if field not in telescope_config:
                errors.append(f"Missing telescope field: {field}")
            else:
                value = telescope_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"Invalid telescope {field}: must be positive number")
        
        return errors
    
    def _validate_target(self, target_config: Dict[str, Any]) -> List[str]:
        """Validate target configuration section."""
        errors = []
        
        for field in self.required_target_fields:
            if field not in target_config:
                errors.append(f"Missing target field: {field}")
            else:
                value = target_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"Invalid target {field}: must be positive number")
        
        return errors
    
    def _validate_detector(self, detector_config: Dict[str, Any]) -> List[str]:
        """Validate detector configuration section."""
        errors = []
        
        for field in self.required_detector_fields:
            if field not in detector_config:
                errors.append(f"Missing detector field: {field}")
            else:
                value = detector_config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Invalid detector {field}: must be non-negative number")
        
        return errors
    
    def _validate_wavelength_range(self, wavelength_config: Dict[str, Any]) -> List[str]:
        """Validate wavelength range configuration section."""
        errors = []
        
        for field in self.required_wavelength_fields:
            if field not in wavelength_config:
                errors.append(f"Missing wavelength_range field: {field}")
            else:
                value = wavelength_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"Invalid wavelength_range {field}: must be positive number")
        
        # Check that min < max
        if "min" in wavelength_config and "max" in wavelength_config:
            if wavelength_config["min"] >= wavelength_config["max"]:
                errors.append("wavelength_range min must be less than max")
        
        return errors
    
    def _validate_astrophysical_sources(self, sources_config: Dict[str, Any]) -> List[str]:
        """Validate astrophysical sources configuration section."""
        errors = []
        
        expected_sources = ["star", "exoplanet", "exozodiacal", "zodiacal"]
        
        for source in expected_sources:
            if source not in sources_config:
                errors.append(f"Missing astrophysical source: {source}")
            else:
                source_config = sources_config[source]
                if not isinstance(source_config, dict):
                    errors.append(f"Invalid {source} configuration: must be dictionary")
                else:
                    # Check for required fields
                    if "enabled" not in source_config:
                        errors.append(f"Missing 'enabled' field for {source}")
                    elif not isinstance(source_config["enabled"], bool):
                        errors.append(f"Invalid 'enabled' field for {source}: must be boolean")
                    
                    if "spectrum_file" not in source_config:
                        errors.append(f"Missing 'spectrum_file' field for {source}")
                    elif not isinstance(source_config["spectrum_file"], str):
                        errors.append(f"Invalid 'spectrum_file' field for {source}: must be string")
        
        return errors
    
    def _validate_instrumental_sources(self, sources_config: Dict[str, Any]) -> List[str]:
        """Validate instrumental sources configuration section."""
        errors = []
        
        expected_sources = ["dark_current", "read_noise"]
        
        for source in expected_sources:
            if source not in sources_config:
                errors.append(f"Missing instrumental source: {source}")
            else:
                source_config = sources_config[source]
                if not isinstance(source_config, dict):
                    errors.append(f"Invalid {source} configuration: must be dictionary")
                else:
                    if "enabled" not in source_config:
                        errors.append(f"Missing 'enabled' field for {source}")
                    elif not isinstance(source_config["enabled"], bool):
                        errors.append(f"Invalid 'enabled' field for {source}: must be boolean")
        
        return errors

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
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    return True 