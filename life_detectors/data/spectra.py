"""
Spectral data handling for the life_detectors package.

This module provides classes and functions for loading and managing
spectral data from files and handling different spectral formats.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

@dataclass
class SpectralData:
    """
    Container for spectral data with wavelength and flux arrays.
    
    Attributes:
        wavelength: Wavelength array in microns
        flux: Flux array in photons/sec/m^2/micron
        wavelength_unit: Unit of wavelength data
        flux_unit: Unit of flux data
        source_name: Name of the spectral source
        metadata: Additional metadata dictionary
    """
    wavelength: np.ndarray
    flux: np.ndarray
    wavelength_unit: str = "um"
    flux_unit: str = "photon_sec_m2_um"
    source_name: str = "unknown"
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        """Validate and initialize spectral data."""
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure arrays are numpy arrays
        self.wavelength = np.asarray(self.wavelength)
        self.flux = np.asarray(self.flux)
        
        # Validate array shapes
        if self.wavelength.shape != self.flux.shape:
            raise ValueError("Wavelength and flux arrays must have the same shape")
        
        # Sort by wavelength if not already sorted
        if not np.all(np.diff(self.wavelength) >= 0):
            sort_idx = np.argsort(self.wavelength)
            self.wavelength = self.wavelength[sort_idx]
            self.flux = self.flux[sort_idx]
    
    def interpolate(self, new_wavelength: np.ndarray) -> 'SpectralData':
        """
        Interpolate spectral data to new wavelength grid.
        
        Args:
            new_wavelength: New wavelength grid in microns
            
        Returns:
            New SpectralData object with interpolated flux
        """
        from scipy.interpolate import interp1d
        
        # Create interpolation function
        interp_func = interp1d(
            self.wavelength, 
            self.flux, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        # Interpolate to new wavelength grid
        new_flux = interp_func(new_wavelength)
        
        return SpectralData(
            wavelength=new_wavelength,
            flux=new_flux,
            wavelength_unit="um",
            flux_unit=self.flux_unit,
            source_name=self.source_name,
            metadata=self.metadata
        )
    
    def integrate_flux(self, wavelength_min: float, wavelength_max: float) -> float:
        """
        Integrate flux over a wavelength range.
        
        Args:
            wavelength_min: Minimum wavelength in microns
            wavelength_max: Maximum wavelength in microns
            
        Returns:
            Integrated flux in photons/sec/m^2
        """
        # Find indices within wavelength range
        mask = (self.wavelength >= wavelength_min) & (self.wavelength <= wavelength_max)
        
        if not np.any(mask):
            return 0.0
        
        # Integrate using trapezoidal rule
        wavelength_subset = self.wavelength[mask]
        flux_subset = self.flux[mask]
        
        return np.trapz(flux_subset, wavelength_subset)
    
    def get_flux_at_wavelength(self, wavelength: float) -> float:
        """
        Get flux at a specific wavelength using linear interpolation.
        
        Args:
            wavelength: Wavelength in microns
            
        Returns:
            Flux value at the specified wavelength
        """
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(
            self.wavelength, 
            self.flux, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        return float(interp_func(wavelength))

def load_spectrum_from_file(filepath: Union[str, Path]) -> SpectralData:
    """
    Load spectral data from a file.
    
    Currently supports ASCII files with two columns (wavelength, flux).
    
    Args:
        filepath: Path to the spectral data file
        
    Returns:
        SpectralData object containing the loaded spectrum
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Spectral data file not found: {filepath}")
    
    # Try to determine file format and load data
    try:
        # Load as ASCII file with two columns
        data = np.loadtxt(filepath, comments='#')
        
        if data.shape[1] < 2:
            raise ValueError(f"File must have at least 2 columns: {filepath}")
        
        wavelength = data[:, 0]
        flux = data[:, 1]
        
        # Try to read header for units
        wavelength_unit = "um"  # Default
        flux_unit = "photon_sec_m2_um"  # Default
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') and 'wavelength_unit' in line:
                        wavelength_unit = line.split('=')[1].strip()
                    elif line.startswith('#') and 'flux_unit' in line:
                        flux_unit = line.split('=')[1].strip()
                    elif not line.startswith('#'):
                        break
        except:
            pass  # Use defaults if header parsing fails
        
        return SpectralData(
            wavelength=wavelength,
            flux=flux,
            wavelength_unit=wavelength_unit,
            flux_unit=flux_unit,
            source_name=filepath.stem,
            metadata={"filepath": str(filepath)}
        )
        
    except Exception as e:
        raise ValueError(f"Failed to load spectral data from {filepath}: {e}")

def create_blackbody_spectrum(
    temperature: float, 
    wavelength_range: Tuple[float, float], 
    n_points: int = 1000
) -> SpectralData:
    """
    Create a blackbody spectrum.
    
    Args:
        temperature: Temperature in Kelvin
        wavelength_range: (min_wavelength, max_wavelength) in microns
        n_points: Number of wavelength points
        
    Returns:
        SpectralData object with blackbody spectrum
    """
    from scipy import constants
    
    wavelength_min, wavelength_max = wavelength_range
    wavelength = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_points)
    
    # Convert wavelength from microns to meters
    wavelength_m = wavelength * 1e-6
    
    # Planck's law: B_λ(T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    h = constants.h  # Planck's constant
    c = constants.c  # Speed of light
    k = constants.k  # Boltzmann constant
    
    # Calculate blackbody flux in W/m²/m
    exp_term = np.exp(h * c / (wavelength_m * k * temperature))
    bb_flux_watt = (2 * h * c**2 / wavelength_m**5) / (exp_term - 1)
    
    # Convert to photons/sec/m²/micron
    # E = hc/λ, so photon flux = energy flux / energy per photon
    energy_per_photon = h * c / wavelength_m
    photon_flux = bb_flux_watt / energy_per_photon
    
    # Convert from per meter to per micron
    photon_flux_per_um = photon_flux * 1e-6
    
    return SpectralData(
        wavelength=wavelength,
        flux=photon_flux_per_um,
        wavelength_unit="um",
        flux_unit="photon_sec_m2_um",
        source_name=f"blackbody_{temperature}K",
        metadata={"temperature": temperature}
    ) 