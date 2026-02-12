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
from astropy import units as u
import astropy.constants as const
import pandas as pd


from ..data.spectra import SpectralData, load_spectrum_from_file
from ..data.units import UnitConverter
from ..utils.helpers import format_plot_title

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
    

    #def _load_model_spectrum(self, source_name: str) -> None:
    #    """Load model spectrum for an exoplanet."""
    #    pass
    

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
                    logger.warning(f"Did not load self-generated spectrum for {source_name}: {e}; either missing, or will need to read in from other file")
        else:
            logger.warning("No [astrophysical_sources] section found in config file.")

    def _calculate_flux_from_spectrum(self, source_name: str, wavelength: u.Quantity, null: bool) -> u.Quantity:
        """
        Helper function to interpolate a stored spectrum and apply distance/nulling corrections.

        Args:
            source_name: Name of the source (star, exoplanet_bb, exozodiacal, zodiacal)
            wavelength: Wavelength grid (with units)
            null: Whether to apply nulling for the star

        Returns:
            Flux array with units ph / (um m^2 s)
        """
        spectrum = self.spectra[source_name]

        # Interpolate to the requested wavelength grid
        # (note this is not integrating over wavelength for each interpolated data point)
        interpolated_spectrum = spectrum.interpolate(wavelength)

        # Apply distance correction, if the source is not zodiacal (which is already in brightness units as seen from Earth)
        if source_name != "zodiacal":
            distance = float(self.config["target"]["distance"]) * u.pc  # parsecs
            distance_correction = 1.0 / (distance ** 2)  # 1/r^2 law
        else:
            distance_correction = 1.0

        # Apply nulling factor for on-axis sources
        nulling_factor = self.config["nulling"]["nulling_factor"]

        # Convert flux_unit string to astropy unit object
        flux_unit_obj = u.Unit(interpolated_spectrum.flux_unit)

        # treatment of units and nulling depending on the source
        if source_name == "zodiacal":
            # no distance correction and no nulling
            flux_incident = interpolated_spectrum.flux * flux_unit_obj
        elif null and (source_name == "star"):
            # apply nulling to star only
            flux_incident = (
                interpolated_spectrum.flux * float(nulling_factor) * distance_correction * flux_unit_obj
            )
            logger.info(f"Applying nulling transmission of {nulling_factor} to {source_name}")
        else:
            # no nulling
            flux_incident = interpolated_spectrum.flux * distance_correction * flux_unit_obj
            logger.info(f"No nulling factor applied to {source_name}.")

        return flux_incident.to(u.ph / (u.um * u.m**2 * u.s))
    

    def calculate_incident_flux(self, source_name: str, plot: bool = False) -> np.ndarray:
        """
        Calculate local (at Earth) flux from an emitted spectrum at a given distance
        
        Args:
            source_name: Name of the source (star, zodiacal, exozodiacal, exoplanet_bb, exoplanet_model_10pc, exoplanet_psg)
            null: apply the nulling factor? (only applies to star target)
            
        Returns:
            Flux array, with units # in photons/sec/m^2/micron
        """

        incident_dict = {}

        # should star be nulled?
        null = bool(self.config["nulling"]["null"])
        logger.info(f"Nulling of star: {null}")
        
        wavelength = np.linspace(float(self.config['wavelength_range']['min']),
                               float(self.config['wavelength_range']['max']),
                               int(self.config['wavelength_range']['n_points'])) * u.um

        if source_name in ["star", "exoplanet_bb", "exozodiacal", "zodiacal"]:

            flux_incident = self._calculate_flux_from_spectrum(source_name, wavelength, null)

        elif source_name == "exoplanet_model_10pc":

            # check that the desired distance is 10 pc; if not, will have to update this
            if float(self.config["target"]["distance"]) != 10.0:
                logger.info(f"Distance {float(self.config['target']['distance'])} pc is not 10 pc; rescaling planet spectrum with true distance.")

            # this is a model spectrum from a file with different units, formatting
            file_name_exoplanet_model_10pc = self.config['astrophysical_sources']['exoplanet_model_10pc']
            df = pd.read_csv(file_name_exoplanet_model_10pc, delim_whitespace=True, names=['wavelength', 'flux', 'err_flux'])
            logger.info(f"Loaded model exoplanetspectrum for {source_name}: {file_name_exoplanet_model_10pc}")

            wavel = df['wavelength'].values * u.micron
            flux_nu_10pc = df['flux'].values * u.erg / (u.second * u.Hz * u.m**2)
            err_flux_nu_10pc = df['err_flux'].values * u.erg / (u.second * u.Hz * u.m**2)

            # convert to F_lambda
            flux_lambda_10pc = flux_nu_10pc * (const.c / wavel**2)
            flux_lambda_10pc = flux_lambda_10pc.to(u.W / (u.m**2 * u.micron))

            # convert to photon flux
            flux_photons_10pc = flux_lambda_10pc * (wavel / (const.h * const.c)) * u.ph
            flux_photons_10pc = flux_photons_10pc.to(u.ph / (u.micron * u.s * u.m**2))

            # rescale this flux (from a source at 10 pc) to the desired distance
            flux_photons = flux_photons_10pc * (10.0 / float(self.config["target"]["distance"])) ** 2

            # interpolate
            flux_incident = np.interp(x = wavelength, 
                                            xp = wavel, 
                                            fp = flux_photons)

        elif source_name == "exoplanet_psg":
            
            # read in the NASA PSG spectrum file name associated with the planets in the population
            df = pd.read_csv(self.config['target']['psg_spectrum_file_name'], names=['wavel', 'flux_total', 'flux_noise', 'flux_planet'], skiprows=15, sep='\s+')

            wavel = df['wavel'].values * u.micron

            # note source is already at the desired distance; should not be rescaled as if it were at 10 pc
            flux_nu = df['flux_planet'].values * u.erg / (u.second * u.Hz * u.m**2)
            err_flux_nu = df['flux_noise'].values * u.erg / (u.second * u.Hz * u.m**2)

            # convert to F_lambda
            flux_lambda = flux_nu * (const.c / wavel**2)
            flux_lambda = flux_lambda.to(u.W / (u.m**2 * u.micron))

            flux_photons = flux_lambda * (wavel / (const.h * const.c)) * u.ph
            flux_photons = flux_photons.to(u.ph / (u.micron * u.s * u.m**2))

            # incident flux from PSG spectrum
            flux_psg_incident = np.interp(x = wavelength, 
                                            xp = wavel, 
                                            fp = flux_photons)

            print('!!!------- FLUX CONVERSION FROM PSG FILE BEING DONE INCORRECTLY; CORRECT LATER -------!!!') ## ## TODO


            ## ## TODO: MAKE SURE SCALING, UNITS ARE RIGHT
            # get BB spectrum for making a rough rescaling of the PSG spectrum
            
            flux_bb_incident = self._calculate_flux_from_spectrum(source_name="exoplanet_bb", wavelength=wavelength, null=False)
            #flux_incident_junk = self._calculate_flux_from_spectrum(source_name="exoplanet_model_10pc", wavelength=wavelength, null=False)

            # integrate the BB spectrum over the wavelength grid
            flux_incident_bb_integrated = np.trapz(y=flux_bb_incident, x=wavelength)

            # integrate the PSG spectrum over the wavelength grid
            flux_incident_psg_integrated = np.trapz(y=flux_psg_incident, x=wavelength)

            # rescale the PSG spectrum to the BB spectrum
            ratio_incident_psg_over_bb = (flux_incident_psg_integrated / flux_incident_bb_integrated).value

            flux_incident = flux_psg_incident / ratio_incident_psg_over_bb
            

        else:
            logger.warning(f"Spectrum not available for {source_name}")
            return np.array([])

        incident_dict['wavel'] = wavelength
        # units ph/um/sec * (1/pc^2) * (pc / 3.086e16 m)^2 <-- last term is for unit consistency
        # = ph/um/m^2/sec

        incident_dict['astro_flux_ph_sec_m2_um'] = flux_incident

        if plot:
            plt.clf()
            plt.figure(figsize=(8, 8)) 
            plt.plot(incident_dict['wavel'], incident_dict['astro_flux_ph_sec_m2_um'])
            plt.yscale('log')
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-3, 1e9]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({incident_dict['wavel'].unit})")
            plt.ylabel(f"Flux (" + str(incident_dict['astro_flux_ph_sec_m2_um'].unit) + ")")
            plt.title(
                format_plot_title(
                    f"Incident flux from {source_name} (at Earth, rescaled for distance {float(self.config['target']['distance'])} pc)",
                    self.config,
                )
            )
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"incident_{source_name}.png"
            plt.tight_layout()
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