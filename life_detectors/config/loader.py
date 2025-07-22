"""
Configuration loader for the life_detectors package.

This module handles loading and saving YAML configuration files
that define all parameters for noise calculations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        filepath: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Configuration file is empty")
        
        logger.info(f"Loaded configuration from {filepath}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {filepath}: {e}")

def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        filepath: Path where to save the configuration file
    """
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {filepath}")
        
    except Exception as e:
        raise IOError(f"Error saving configuration to {filepath}: {e}")

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "telescope": {
            "collecting_area": 25.0,  # m^2
            "plate_scale": 0.1,       # arcsec/pixel
            "throughput": 0.8,        # dimensionless
        },
        "target": {
            "distance": 10.0,         # parsecs
            "nulling_factor": 0.01,   # dimensionless
        },
        "detector": {
            "read_noise": 5.0,        # e-/pixel
            "dark_current": 0.1,      # e-/pixel/sec
            "gain": 2.0,              # e-/ADU
            "integration_time": 3600,  # seconds
        },
        "astrophysical_sources": {
            "star": {
                "spectrum_file": "data/star_spectrum.txt",
                "enabled": True,
            },
            "exoplanet": {
                "spectrum_file": "data/exoplanet_spectrum.txt",
                "enabled": True,
            },
            "exozodiacal": {
                "spectrum_file": "data/exozodiacal_spectrum.txt",
                "enabled": True,
            },
            "zodiacal": {
                "spectrum_file": "data/zodiacal_spectrum.txt",
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
            "min": 1.0,   # microns
            "max": 10.0,  # microns
            "n_points": 1000,
        },
    }

def load_config_with_defaults(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or create default if file doesn't exist.
    
    Args:
        filepath: Path to configuration file (optional)
        
    Returns:
        Configuration dictionary
    """
    if filepath is None:
        return create_default_config()
    
    try:
        return load_config(filepath)
    except FileNotFoundError:
        logger.warning(f"Configuration file {filepath} not found, using defaults")
        return create_default_config() 