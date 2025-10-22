"""
Configuration loader for the modules package.

This module handles loading and saving YAML configuration files
that define all parameters for noise calculations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import configparser
import os
from datetime import datetime
import ipdb

logger = logging.getLogger(__name__)

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


def load_config(config_file: str, makedirs: bool = False) -> dict:
    """Load configuration from a file.
    
    Args:
        config_file (str): Path to the configuration file
        makedirs (bool, optional): Create directories if they don't exist.
        
    Returns:
        dict: Dictionary containing configuration parameters
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    logger = logging.getLogger(__name__)
    
    # Convert ConfigParser to regular dictionary
    config_dict = {}
    
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config[section].items():
            # Try to convert to float if possible
            try:
                config_dict[section][key] = float(value)
            except ValueError:
                config_dict[section][key] = value

    # Log all sections and their parameters
    for section in config_dict:
        logger.info(f"[{section}]")
        for key, value in config_dict[section].items():
            logger.info(f"  {key} = {value}")

    if makedirs and 'dirs' in config_dict:
        for key, value in config_dict['dirs'].items():
            os.makedirs(value, exist_ok=True)
            logger.info(f"Created directory: {value}")

    return config_dict


def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'detectorsim_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file