"""
Unit definitions and conversions for the modules package.

This module provides unit conversion utilities and defines standard units
for wavelength, flux, and noise calculations.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Standard unit definitions
WAVELENGTH_UNITS = {
    "um": "microns",
    "nm": "nanometers", 
    "angstrom": "angstroms",
    "m": "meters"
}

FLUX_UNITS = {
    "photon_sec_m2_um": "photons/second/meter^2/micron",
    "photon_sec_m2_nm": "photons/second/meter^2/nanometer",
    "erg_sec_cm2_angstrom": "ergs/second/cm^2/angstrom",
    "watt_m2_um": "watts/meter^2/micron"
}

NOISE_UNITS = {
    "e_per_pixel": "electrons per pixel",
    "e_per_pixel_sec": "electrons per pixel per second",
    "adu_per_pixel": "ADU per pixel",
    "adu_per_pixel_sec": "ADU per pixel per second"
}

@dataclass
class UnitConverter:
    """Handles unit conversions for astronomical calculations."""
    
    def __post_init__(self):
        """Initialize conversion factors."""
        self._wavelength_conversions = {
            ("um", "nm"): 1000.0,
            ("um", "angstrom"): 10000.0,
            ("um", "m"): 1e-6,
            ("nm", "um"): 0.001,
            ("nm", "angstrom"): 10.0,
            ("nm", "m"): 1e-9,
            ("angstrom", "um"): 0.0001,
            ("angstrom", "nm"): 0.1,
            ("angstrom", "m"): 1e-10,
            ("m", "um"): 1e6,
            ("m", "nm"): 1e9,
            ("m", "angstrom"): 1e10,
        }
        
        self._flux_conversions = {
            ("photon_sec_m2_um", "photon_sec_m2_nm"): 0.001,
            ("photon_sec_m2_um", "erg_sec_cm2_angstrom"): self._photon_to_erg_conversion,
            ("photon_sec_m2_um", "watt_m2_um"): self._photon_to_watt_conversion,
        }
    
    def convert_wavelength(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert wavelength between different units.
        
        Args:
            value: Wavelength value to convert
            from_unit: Source unit (e.g., 'um', 'nm', 'angstrom', 'm')
            to_unit: Target unit
            
        Returns:
            Converted wavelength value
            
        Raises:
            ValueError: If conversion is not supported
        """
        if from_unit == to_unit:
            return value
            
        key = (from_unit, to_unit)
        if key in self._wavelength_conversions:
            return value * self._wavelength_conversions[key]
        elif (to_unit, from_unit) in self._wavelength_conversions:
            return value / self._wavelength_conversions[(to_unit, from_unit)]
        else:
            raise ValueError(f"Unsupported wavelength conversion: {from_unit} to {to_unit}")
    
    def convert_flux(self, value: float, wavelength: float, from_unit: str, to_unit: str) -> float:
        """
        Convert flux between different units.
        
        Args:
            value: Flux value to convert
            wavelength: Wavelength in microns (for photon-energy conversions)
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted flux value
        """
        if from_unit == to_unit:
            return value
            
        key = (from_unit, to_unit)
        if key in self._flux_conversions:
            conversion_func = self._flux_conversions[key]
            if callable(conversion_func):
                return conversion_func(value, wavelength)
            else:
                return value * conversion_func
        else:
            raise ValueError(f"Unsupported flux conversion: {from_unit} to {to_unit}")
    
    
    def electrons_to_adu(self, electrons: float, gain: float) -> float:
        """
        Convert electrons to ADU (Analog-to-Digital Units).
        
        Args:
            electrons: Number of electrons
            gain: Detector gain in e-/ADU
            
        Returns:
            ADU value
        """
        return electrons / gain
    
    
    def adu_to_electrons(self, adu: float, gain: float) -> float:
        """
        Convert ADU to electrons.
        
        Args:
            adu: ADU value
            gain: Detector gain in e-/ADU
            
        Returns:
            Number of electrons
        """
        return adu * gain 