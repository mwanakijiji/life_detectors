"""
Life Detectors - Infrared Detector Noise Calculator

A Python package for calculating total noise in infrared detectors on telescopes observing stars.
"""

__version__ = "0.1.0"
__author__ = "Life Team"

from .core.calculator import NoiseCalculator
from .config.loader import load_config
from astropy import units as u

__all__ = ["NoiseCalculator", "load_config"] 

#const_c = 2.99792458e8 * u.m / u.s
#const_h = 6.62607015e-34 * u.J * u.s