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
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Logs always go to sim_pipeline/logs/, wherever the pipeline is run from
# (this file is sim_pipeline/modules/utils/loader.py).
DEFAULT_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")
# Environment variable that overrides DEFAULT_LOG_DIR (the tests point it at a
# temporary directory so pytest never writes into sim_pipeline/logs/).
LOG_DIR_ENV_VAR = "DETECTORSIM_LOG_DIR"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


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


def default_log_dir() -> str:
    """Log directory: $DETECTORSIM_LOG_DIR if set, else sim_pipeline/logs/."""
    return os.environ.get(LOG_DIR_ENV_VAR) or DEFAULT_LOG_DIR


def _new_log_file_path(log_dir: str, tag: Optional[str] = None) -> str:
    """Create log_dir if needed; return a timestamped, unique log file path in it."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_tag = tag or f"pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
    return os.path.join(log_dir, f"detectorsim_{timestamp}_{log_tag}.log")


def writes_to_log_file(handler: logging.Handler) -> bool:
    """True for a FileHandler writing to a real file (pytest attaches one on
    os.devnull, which does not count)."""
    return isinstance(handler, logging.FileHandler) and os.path.abspath(
        handler.baseFilename
    ) != os.path.abspath(os.devnull)


def setup_logging(log_dir: Optional[str] = None, tag: Optional[str] = None) -> str:
    """Log INFO to the console and to a timestamped file; return the file's path.

    Replaces any existing root-logger handlers. log_dir defaults to
    default_log_dir() (sim_pipeline/logs/, independent of the working directory).
    """
    log_file = _new_log_file_path(log_dir or default_log_dir(), tag)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return log_file


def ensure_logging(tag: Optional[str] = None) -> Optional[str]:
    """Start logging to a file unless the root logger already writes to one.

    For entry points other than batch_run.py (notebooks, direct calls to
    run_single_calculation / parameter_sweep). Unlike setup_logging, existing
    handlers are kept; a console handler is added only if there are none.

    Returns:
        The new log file's path, or None if file logging was already set up.
    """
    root = logging.getLogger()
    if any(writes_to_log_file(h) for h in root.handlers):
        return None
    log_file = _new_log_file_path(default_log_dir(), tag)
    formatter = logging.Formatter(LOG_FORMAT)
    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)
    logger.info("Log file: %s", log_file)
    return log_file
