"""
Astrophysical noise calculations for the modules package.

This module handles calculations of astrophysical noise sources including
stars, exoplanets, exozodiacal disks, and zodiacal backgrounds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import ipdb
import configparser
import matplotlib.pyplot as plt

from ..data.spectra import SpectralData, load_spectrum_from_file
from ..data.units import UnitConverter

logger = logging.getLogger(__name__)


class AstrophysicalSources:
    """
    Calculates photon flux from astrophysical sources (incl. noise)
    """
    
    def __init__(self, config: configparser.ConfigParser, unit_converter: UnitConverter):
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
        #sources_config = self.config.get("astrophysical_sources", {})
        
        sources_section = "astrophysical_sources"
        if self.config.has_section(sources_section):
            for source_name in self.config.options(sources_section):

                try:
                    # Get the source name from the section
                    spectrum_file_name = self.config[sources_section][source_name]
                    self.spectra[source_name] = load_spectrum_from_file(spectrum_file_name)
                    logger.info(f"Loaded spectrum for {source_name}: {spectrum_file_name}")

                except Exception as e:
                    logger.error(f"Failed to load spectrum for {source_name}: {e}")
        else:
            logger.warning("No [astrophysical_sources] section found in config file.")
    

    def calculate_incident_flux(self, source_name: str, null: bool = False, plot: bool = False) -> np.ndarray:
        """
        Calculate local (at Earth) flux from an emitted spectrum at a given distance
        
        Args:
            source_name: Name of the source (star, exoplanet, etc.)
            null: apply the nulling factor? (only applies to star target)
            
        Returns:
            Flux array in photons/sec/m^2/micron
        """

        incident_dict = {}

        if source_name not in self.spectra:
            logger.warning(f"Spectrum not available for {source_name}")
            return np.array([])
        
        wavelength = np.linspace(float(self.config['wavelength_range']['min']), 
                               float(self.config['wavelength_range']['max']),
                               int(self.config['wavelength_range']['n_points']))
        
        spectrum = self.spectra[source_name]
        
        # Interpolate to the requested wavelength grid
        # (note this is not integrating over wavelength for each interpolated data point) 
        interpolated_spectrum = spectrum.interpolate(wavelength)
        
        # Apply distance correction
        distance = float(self.config["target"]["distance"])  # parsecs
        distance_correction = 1.0 / (distance ** 2)  # 1/r^2 law
        
        # Apply nulling factor for on-axis sources
        nulling_factor = self.config["nulling"]["nulling_factor"]
        if null and (source_name in ["star"]):  # Apply nulling to star only
            flux = interpolated_spectrum.flux * distance_correction * float(nulling_factor)
            logger.info(f"Applying nulling factor of {nulling_factor} to {source_name}")
        else:
            flux = interpolated_spectrum.flux * distance_correction
            logger.info(f"No nulling factor applied to {source_name}.")

        incident_dict['wavel'] = wavelength
        # units ph/um/sec * (1/pc^2) * (pc / 3.086e16 m)^2 <-- last term is for unit consistency
        # = ph/um/m^2/sec
        incident_dict['astro_flux_ph_sec_m2_um'] = flux * distance_correction * (1.0 / (3.086e16)**2)

        if plot:
            plt.scatter(incident_dict['wavel'], incident_dict['astro_flux_ph_sec_m2_um'])
            plt.yscale('log')
            plt.xlabel(f"Wavelength ({spectrum.wavelength_unit})")
            plt.ylabel(f"Flux (ph/um/m^2/sec)")
            plt.title(f"Incident flux from {source_name}")
            file_name_plot = "/Users/eckhartspalding/Downloads/" + f"incident_{source_name}.png"
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux to " + file_name_plot)
        
        return incident_dict
    
    '''
    def convert_adu(self, source_name: str, null: bool = False, plot: bool = False) -> np.ndarray:
        # Converts photons to e and ADU

        pass
    '''

    '''
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
    '''