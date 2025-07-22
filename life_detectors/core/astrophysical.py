"""
Astrophysical noise calculations for the life_detectors package.

This module handles calculations of astrophysical noise sources including
stars, exoplanets, exozodiacal disks, and zodiacal backgrounds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..data.spectra import SpectralData, load_spectrum_from_file
from ..data.units import UnitConverter

logger = logging.getLogger(__name__)

@dataclass
class AstrophysicalNoise:
    """
    Calculates astrophysical noise sources for telescope observations.
    
    This class handles the calculation of photon flux from various
    astrophysical sources and converts them to detector noise.
    """
    
    def __init__(self, config: Dict, unit_converter: UnitConverter):
        """
        Initialize astrophysical noise calculator.
        
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion utility
        """
        self.config = config
        self.unit_converter = unit_converter
        self.spectra = {}
        self._load_spectra()
    
    def _load_spectra(self) -> None:
        """Load spectral data for all astrophysical sources."""
        sources_config = self.config.get("astrophysical_sources", {})
        
        for source_name, source_config in sources_config.items():
            if source_config.get("enabled", True):
                try:
                    spectrum_file = source_config.get("spectrum_file")
                    if spectrum_file:
                        self.spectra[source_name] = load_spectrum_from_file(spectrum_file)
                        logger.info(f"Loaded spectrum for {source_name}: {spectrum_file}")
                    else:
                        logger.warning(f"No spectrum file specified for {source_name}")
                except Exception as e:
                    logger.error(f"Failed to load spectrum for {source_name}: {e}")
    
    def calculate_source_flux(self, source_name: str, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate flux from a specific astrophysical source.
        
        Args:
            source_name: Name of the source (star, exoplanet, etc.)
            wavelength: Wavelength array in microns
            
        Returns:
            Flux array in photons/sec/m^2/micron
        """
        if source_name not in self.spectra:
            logger.warning(f"Spectrum not available for {source_name}")
            return np.zeros_like(wavelength)
        
        spectrum = self.spectra[source_name]
        
        # Interpolate to the requested wavelength grid
        interpolated_spectrum = spectrum.interpolate(wavelength)
        
        # Apply distance correction
        distance = self.config["target"]["distance"]  # parsecs
        distance_correction = 1.0 / (distance ** 2)  # 1/r^2 law
        
        # Apply nulling factor for on-axis sources
        nulling_factor = self.config["target"]["nulling_factor"]
        if source_name in ["star"]:  # Apply nulling to star only
            flux = interpolated_spectrum.flux * distance_correction * nulling_factor
        else:
            flux = interpolated_spectrum.flux * distance_correction
        
        return flux
    
    def calculate_total_astrophysical_flux(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate total astrophysical flux from all sources.
        
        Args:
            wavelength: Wavelength array in microns
            
        Returns:
            Total flux array in photons/sec/m^2/micron
        """
        total_flux = np.zeros_like(wavelength)
        
        sources_config = self.config.get("astrophysical_sources", {})
        
        for source_name in sources_config.keys():
            if sources_config[source_name].get("enabled", True):
                source_flux = self.calculate_source_flux(source_name, wavelength)
                total_flux += source_flux
                logger.debug(f"Added {source_name} flux: {np.sum(source_flux):.2e}")
        
        return total_flux
    
    def calculate_detector_illumination(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate total illumination at the detector.
        
        This includes telescope collecting area, throughput, and plate scale.
        
        Args:
            wavelength: Wavelength array in microns
            
        Returns:
            Detector illumination in photons/sec/pixel/micron
        """
        # Get telescope parameters
        collecting_area = self.config["telescope"]["collecting_area"]  # m^2
        throughput = self.config["telescope"]["throughput"]  # dimensionless
        plate_scale = self.config["telescope"]["plate_scale"]  # arcsec/pixel
        
        # Calculate total astrophysical flux
        total_flux = self.calculate_total_astrophysical_flux(wavelength)
        
        # Convert to detector illumination
        # photons/sec/m^2/micron -> photons/sec/pixel/micron
        pixel_area = (plate_scale ** 2) * (np.pi / (180 * 3600)) ** 2  # steradians
        detector_illumination = total_flux * collecting_area * throughput * pixel_area
        
        return detector_illumination
    
    def calculate_astrophysical_noise_electrons(self, wavelength: np.ndarray, integration_time: float) -> np.ndarray:
        """
        Calculate astrophysical noise in electrons per pixel.
        
        Args:
            wavelength: Wavelength array in microns
            integration_time: Integration time in seconds
            
        Returns:
            Noise in electrons per pixel
        """
        # Get detector illumination
        illumination = self.calculate_detector_illumination(wavelength)
        
        # Convert to electrons (assuming 1 photon = 1 electron for simplicity)
        # In practice, quantum efficiency would be applied here
        ## ## TO DO: MAKE THIS MORE REALISTIC
        electrons_per_pixel = illumination * integration_time
        
        # Calculate noise (shot noise: sqrt(N))
        noise_electrons = np.sqrt(electrons_per_pixel)
        
        return noise_electrons
    
    def calculate_astrophysical_noise_adu(self, wavelength: np.ndarray, integration_time: float) -> np.ndarray:
        """
        Calculate astrophysical noise in ADU per pixel.
        
        Args:
            wavelength: Wavelength array in microns
            integration_time: Integration time in seconds
            
        Returns:
            Noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_astrophysical_noise_electrons(wavelength, integration_time)
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def get_source_contributions(self, wavelength: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual source contributions to total flux.
        
        Args:
            wavelength: Wavelength array in microns
            
        Returns:
            Dictionary mapping source names to their flux contributions
        """
        contributions = {}
        
        sources_config = self.config.get("astrophysical_sources", {})
        
        for source_name in sources_config.keys():
            if sources_config[source_name].get("enabled", True):
                source_flux = self.calculate_source_flux(source_name, wavelength)
                contributions[source_name] = source_flux
        
        return contributions 