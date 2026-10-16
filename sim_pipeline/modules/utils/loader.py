"""
Configuration loader for the modules package.

This module handles loading and saving YAML configuration files
that define all parameters for noise calculations.
"""

import configparser
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def config_getboolean(
    config: Union[configparser.ConfigParser, Dict[str, Dict[str, Any]]],
    section: str,
    key: str,
    default: bool = False,
) -> bool:
    """Read a True/False INI flag from ConfigParser or a dict from load_config."""
    if isinstance(config, configparser.ConfigParser):
        if not config.has_section(section):
            return default
        return config.getboolean(section, key, fallback=default)
    sec = config.get(section, {})
    if key not in sec:
        return default
    val = sec[key]
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        lower = val.strip().lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return default


# def save_config(config: Dict[str, Any], filepath: str) -> None:
#     """
#     Save configuration to a YAML file.
#
#     Args:
#         config: Configuration dictionary to save
#         filepath: Path where to save the configuration file
#     """
#     filepath = Path(filepath)
#
#     # Create directory if it doesn't exist
#     filepath.parent.mkdir(parents=True, exist_ok=True)
#
#     try:
#         with open(filepath, 'w') as f:
#             yaml.dump(config, f, default_flow_style=False, indent=2)
#
#         logger.info(f"Saved configuration to {filepath}")
#
#     except Exception as e:
#         raise IOError(f"Error saving configuration to {filepath}: {e}")


def load_config(config_file: str, makedirs: bool = True) -> dict:
    """Load configuration from a file.

    Args:
        config_file (str): Path to the configuration file
        makedirs (bool, optional): Create directories if they don't exist.

    Returns:
        dict: Dictionary containing configuration parameters
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    # Convert ConfigParser to regular dictionary
    config_dict = {}

    for section in config.sections():
        config_dict[section] = {}
        for key, value in config[section].items():
            # Try to convert to float if possible
            try:
                config_dict[section][key] = float(value)
            except ValueError:
                lower = value.strip().lower()
                if lower == "true":
                    config_dict[section][key] = True
                elif lower == "false":
                    config_dict[section][key] = False
                else:
                    config_dict[section][key] = value

    # Log all sections and their parameters
    for section, params in config_dict.items():
        logger.info("[%s]", section)
        for key, value in params.items():
            logger.info("  %s = %s", key, value)

    if makedirs and "dirs" in config_dict:
        for key, value in config_dict["dirs"].items():
            os.makedirs(value, exist_ok=True)
            logger.info("Created directory: %s", value)

    return config_dict


def setup_logging(log_dir="logs", tag: Optional[str] = None):
    """Log INFO to the console and to a timestamped file in log_dir; return its path."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_tag = tag or f"pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
    log_file = os.path.join(log_dir, f"detectorsim_{timestamp}_{log_tag}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return log_file
