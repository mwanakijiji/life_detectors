"""
Life Detectors - Infrared Detector Noise Calculator

A Python package for calculating total noise in infrared detectors on telescopes observing stars.
"""

__version__ = "0.1.0"
__author__ = "Life Team"

from .core.calculator import NoiseCalculator
from .config.loader import load_config

__all__ = ["NoiseCalculator", "load_config"] 