"""Shared type aliases for annotations across the pipeline."""

import configparser
from typing import Any, Dict, Union

# A pipeline config: a ConfigParser, or the dict-of-dicts returned by
# modules.utils.loader.load_config
ConfigLike = Union[configparser.ConfigParser, Dict[str, Dict[str, Any]]]
