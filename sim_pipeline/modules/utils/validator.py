"""
Configuration validator for the modules package.

This module validates configuration files to ensure all required
parameters are present and have valid values.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration dictionaries for the modules package."""

    def __init__(self) -> None:
        """Initialize the validator with required fields."""
        self.required_sections = [
            "telescope",
            "target",
            "nulling",
            "detector",
            "observation",
            "wavelength_range",
        ]

        self.required_telescope_fields = [
            "collecting_area",
            "plate_scale",
            "throughput",
        ]

        self.required_target_fields = ["distance", "pl_temp"]

        self.required_nulling_fields = ["distance", "pl_temp"]

        self.required_detector_fields = [
            "read_noise",
            "dark_current",
            "gain",
            "quantum_efficiency",
            "spec_res",
        ]

        self.required_observation_fields = ["t_int_obs_total", "t_int_frame"]

        self.required_wavelength_fields = ["min", "max", "n_points"]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Problems are logged at ERROR level (nothing is returned); a value that cannot
        be converted to float raises ValueError.
        """

        # Check for required sections
        for section in self.required_sections:
            if section not in config:
                logging.error("Missing required section: %s", section)

        # Validate telescope section
        if "telescope" in config:
            self._validate_telescope(config["telescope"])

        # Validate target section
        if "target" in config:
            self._validate_target(config["target"])

        # Validate detector section
        if "detector" in config:
            self._validate_detector(config["detector"])

        # Validate wavelength range
        if "wavelength_range" in config:
            self._validate_wavelength_range(config["wavelength_range"])

        # Validate astrophysical sources
        if "astrophysical_sources" in config:
            self._validate_astrophysical_sources(config["astrophysical_sources"])

    def _validate_telescope(self, telescope_config: Dict[str, Any]) -> None:
        """Validate telescope configuration section."""

        for field in self.required_telescope_fields:
            if field not in telescope_config:
                logging.error("Missing telescope field: %s", field)
            else:
                value = telescope_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    # all values here should be convertable to floats
                    logging.error(
                        "Invalid telescope %s: must be positive number", field
                    )

    def _validate_target(self, target_config: Dict[str, Any]) -> None:
        """Validate target configuration section."""

        for field in self.required_target_fields:
            if field not in target_config:
                logging.error("Missing target field: %s", field)
            else:
                value = target_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    # all values here should be convertable to floats
                    logging.error("Invalid target %s: must be positive number", field)

    def _validate_detector(self, detector_config: Dict[str, Any]) -> None:
        """Validate detector configuration section."""

        for field in self.required_detector_fields:
            if field not in detector_config:
                logging.error("Missing detector field: %s", field)

    def _validate_wavelength_range(self, wavelength_config: Dict[str, Any]) -> None:
        """Validate wavelength range configuration section."""

        for field in self.required_wavelength_fields:
            if field not in wavelength_config:
                logging.error("Missing wavelength_range field: %s", field)
            else:
                value = wavelength_config[field]
                if not isinstance(float(value), (int, float)) or float(value) <= 0:
                    logging.error(
                        "Invalid wavelength_range %s: must be positive number", field
                    )

        # Check that min < max
        if "min" in wavelength_config and "max" in wavelength_config:
            if float(wavelength_config["min"]) >= float(wavelength_config["max"]):
                logging.error("wavelength_range min must be less than max")

    def _validate_astrophysical_sources(self, sources_config: Dict[str, Any]) -> None:
        """Validate astrophysical sources configuration section."""

        expected_sources = ["star", "exoplanet", "exozodiacal", "zodiacal"]

        for source in expected_sources:
            if source not in sources_config:
                logging.error("Missing astrophysical source: %s", source)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator()

    # Capture validation errors via logging
    errors: List[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: errors.append(record.getMessage())
    handler.setLevel(logging.ERROR)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        validator.validate_config(config)
        return len(errors) == 0
    except (ValueError, TypeError):
        return False
    finally:
        root_logger.removeHandler(handler)
