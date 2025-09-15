"""
Instrumental noise calculations for the modules package.

This module handles calculations of instrumental noise sources including
dark current, read noise, and other detector effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import ipdb
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from ..data.units import UnitConverter


class InstrumentDepTerms:
    # Provides the effects of the instrument (including astro flux passed through the telescope aperture)

    def __init__(self, config: Dict, unit_converter: UnitConverter, sources_astroph: dict):
        '''
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion object
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
        '''

        self.config = config
        self.unit_converter = unit_converter ## ## TODO: DO I NEED THIS?
        self.sources_astroph = sources_astroph # all sources of astrophysical flux, as are incident on the instrument

        # initialize dict to carry intrinsic instrumental terms (independent of astrophysics)
        self.sources_instrum = {}

        # initialize dict to carry propagated astrophysicalterms (i.e., intensity levels on the detector, after instrument effects)
        self.prop_dict = {}
        # assume wavelengths are the same for the star and planet
        #self.prop_dict['wavel'] = self.star_flux['wavel']


    def calculate_instrinsic_instrumental_noise(self):
        # calculate intrinsic instrumental noise, and update self.sources_instrum

        gain = float(self.config["detector"]["gain"]) * u.electron / u.adu  # e-/ADU

        # read noise
        # e-/pix rms
        #self.instrum_dict['read_noise_e_rms'] = float(self.config["detector"]["read_noise"])
        # e-/pix rms -> ADU rms
        logging.info(f'Finding instrumental noise sources...')
        read_noise_e_rms = float(self.config["detector"]["read_noise"]) * u.electron / u.pix
        self.sources_instrum['read_noise_e_pix-1'] = read_noise_e_rms
        logging.info(f'Read noise is {read_noise_e_rms} rms')
        #self.sources_instrum['read_noise_adu'] = read_noise_e_rms / gain
        #read_noise_adu_rms = self.sources_instrum['read_noise_adu']
        #logging.info(f'Read noise is {read_noise_adu_rms} rms')

        # dark current rate 
        # e/pix/sec
        dark_current_str = self.config["detector"]["dark_current"]
        ## ## TODO: DO I WANT TO MAKE THIS MORE FLEXIBLE TO ALLOW OTHER TERMS TO BE IN THE FORM OF ARRAYS TOO?
        dark_current_rate_e_pix_sec = np.fromstring(dark_current_str, sep=',') * u.electron / (u.pix * u.second) # in case it's an array
        logging.info(f'Dark current is {dark_current_rate_e_pix_sec} e-/pix/sec')

        # total dark current in e-, based on integration time
        # e/pix/sec -> e/pix
        integration_time = float(self.config["observation"]["integration_time"]) * u.second  # seconds
        self.sources_instrum['dark_current_e_pix-1_sec-1'] = dark_current_rate_e_pix_sec
        self.sources_instrum['dark_current_e_pix-1'] = dark_current_rate_e_pix_sec * integration_time

        # total dark current in ADU
        # e/pix -> ADU/pix
        #self.sources_instrum['dark_current_adu_pix-1'] = self.sources_instrum['dark_current_e_pix-1'] / gain

        return 


    def pass_through_aperture(self, plot: bool = False):
        # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec

        for source_name, source_val in self.sources_astroph.items():
            # if name is right and units are right
            if ('astro_flux_ph_sec_m2_um' in source_val) and (source_val['astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                dict_this = {source_name: {'wavel': source_val['wavel'], 'flux_post_aperture_ph_sec_um': np.multiply( float(self.config["telescope"]["collecting_area"])*u.m**2, source_val['astro_flux_ph_sec_m2_um'] )}}
                self.prop_dict.update(dict_this)

        # overplot all the sources
        title_lines = [
            "Photoelectrons, post-aperture",
            "\n",
            f"collecting area = {float(self.config['telescope']['collecting_area']):.2f} m²",
            f"throughput = {float(self.config['telescope']['throughput']):.2f}",
            f"stellar nulling = {bool(self.config['nulling']['null'])}, nulling transmission = {float(self.config['nulling']['nulling_factor']):.2f}",
            fr"galactic $\lambda_{{\rm rel}}$ = {float(self.config['observation']['lambda_rel_lon_los']):.2f} deg, $\beta$ = {float(self.config['observation']['beta_lat_los']):.2f} deg"
        ]
        if plot:
            plt.clf()
            for source_name, source_val in self.prop_dict.items():
                plt.plot(source_val['wavel'], source_val['flux_post_aperture_ph_sec_um'], label=source_name)
            plt.yscale('log')
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-8, 1e12]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({source_val['wavel'].unit})")
            plt.ylabel(f"Flux (" + str(source_val['flux_post_aperture_ph_sec_um'].unit) + ")")
            plt.legend()
            plt.title("\n".join(title_lines))
            file_name_plot = "/Users/eckhartspalding/Downloads/" + f"photoelectrons_all_sources.png"
            plt.tight_layout()
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux to " + file_name_plot)

        logging.info(f'Passed astrophysical flux through telescope aperture...')

        return


    def photons_to_e(self):
        # convert photons to e-s, using e-to-photon relation and QE, and update prop_dict with the converted terms

        ## ## TODO: INCORPORATE REAL RESPONSE CURVES

        for source_name, source_val in self.prop_dict.items():
            if ('flux_post_aperture_ph_sec_um' in source_val) and (source_val['flux_post_aperture_ph_sec_um'].unit == u.ph / (u.um * u.s)):
                source_val['flux_e_sec_um'] = float(self.config["detector"]["photons_to_e"]) * (u.electron/u.ph) * np.multiply(float(self.config["detector"]["quantum_efficiency"]), source_val['flux_post_aperture_ph_sec_um'])

        return


    def e_to_adu(self):

        for source_name, source_val in self.prop_dict.items():
            if ('flux_e_sec_um' in source_val) and (source_val['flux_e_sec_um'].unit == u.electron / (u.um * u.s)):
                source_val['flux_adu_sec_um'] = np.divide(source_val['flux_e_sec_um'], float(self.config["detector"]["gain"])) * (u.adu/u.electron)

        logging.info(f'Converted e-s to ADUs...')

        return


class Detector:
    # Detector object that contains the illumination footprint and possibly fancier noise sources

    def __init__(self, config: Dict, num_wavel_bins: int):
        '''
        Args:
            config: Configuration dictionary
            num_wavel_bins: Number of wavelength bins
        '''

        self.side_length_pix = int(config["detector"]["size"])
        self.pitch_pix = float(config["detector"]["pitch_pix"])
        self.pix_per_wavel_bin = float(config["detector"]["pix_per_wavel_bin"])
        self.num_wavel_bins = num_wavel_bins
        self.pix_spectral_width = int(config["detector"]["pix_spectral_width"])

        #self.gain = config["detector"]["gain"]
        #self.quantum_efficiency = config["detector"]["quantum_efficiency"]
        #self.photons_to_e = config["detector"]["photons_to_e"]
        #self.read_noise = config["detector"]["read_noise"]
        #self.dark_current = config["detector"]["dark_current"]

        # initialize dict to carry intrinsic instrumental terms (independent of astrophysics)
        #self.sources_instrum = {}

        # initialize dict to carry propagated astrophysicalterms (i.e., intensity levels on the detector, after instrument effects)
        #self.prop_dict = {}
        # assume wavelengths are the same for the star and planet
        #self.prop_dict['wavel'] = self.star_flux['wavel']


    def footprint_spectral(self, plot: bool = True):
        # return a boolean array of the detector, where the footprint is 1 (or a fraction, if the footprint edges do not cover a whole number of pixels)
        # plot: whether to make an FYI plot of the footprint

        # lower-left corner of the footprint
        # (keep this coordinate whole numbers for now)
        starting_pixel = np.array([100,300])

        footprint_cube = np.full((self.num_wavel_bins, self.side_length_pix, self.side_length_pix), 0.0, dtype=float)

        # for each wavelength bin, make the footprint for that bin alone
        for wavel_bin_num in range(0, self.num_wavel_bins):
            # assumes horizontal spectra

            # initialize the footprint
            footprint_this = np.full((self.side_length_pix, self.side_length_pix), 0.0, dtype=float)

            # get the starting lower-left pixel for this wavelength bin
            # (note this can be a float)
            starting_pixel_this = starting_pixel + np.array([0,wavel_bin_num * self.pix_per_wavel_bin])

            # get the footprint for this wavelength bin
            pixel_ceil_start_x = int(np.ceil(starting_pixel_this[1]))
            pixel_frac_start_x = pixel_ceil_start_x-starting_pixel_this[1]
            pixel_floor_end_x = int(np.floor(starting_pixel_this[1] + self.pix_per_wavel_bin))
            pixel_frac_end_x = (starting_pixel_this[1] + self.pix_per_wavel_bin) - pixel_floor_end_x
            # where whole pixels are under the footprint, set them to 1
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_ceil_start_x):int(pixel_floor_end_x)] = 1.0

            # where partial pixels are under the footprint, set their values to the fraction of the pixel that is under the footprint
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_ceil_start_x)-1] = pixel_frac_start_x
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_floor_end_x)] = pixel_frac_end_x

            # add to the cube
            footprint_cube[wavel_bin_num,:,:] = footprint_this

        # FYI FITS file
        # fits.writeto(f"/Users/eckhartspalding/Downloads/footprint_cube.fits", footprint_cube, overwrite=True)

        footprint_sum = np.sum(footprint_cube, axis=0)

        logging.info(f"Detector footprint is {footprint_sum.sum()} pixels")

        if plot:
            plt.clf()
            plt.title(f"Detector spectral footprint (True)")
            plt.imshow(footprint_sum, origin='lower', cmap='gray')
            plt.xlabel(f"Pixel")
            plt.ylabel(f"Pixel")
            plt.gca().set_aspect('equal', adjustable='box')
            file_name_plot = "/Users/eckhartspalding/Downloads/footprint_bool.png"
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of detector footprint containing all wavelength bins to {file_name_plot}")


        return footprint_cube

'''
@dataclass
class InstrumentalNoise:
    """
    Calculates instrumental noise sources for telescope observations.
    
    This class handles the calculation of detector noise including
    dark current, read noise, and other instrumental effects.
    """
    
    def __init__(self, config: Dict, unit_converter: UnitConverter):
        """
        Initialize instrumental noise calculator.
        
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion utility
        """
        self.config = config
        self.unit_converter = unit_converter
    
    def calculate_dark_current_electrons(self, integration_time: float) -> float:
        """
        Calculate dark current noise in electrons per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dark current noise in electrons per pixel
        """
        dark_current_rate = self.config["detector"]["dark_current"]  # e-/pixel/sec
        
        # Dark current is a Poisson process, so noise = sqrt(N)
        dark_electrons = dark_current_rate * integration_time
        dark_noise = np.sqrt(dark_electrons)
        
        return dark_noise
    
    def calculate_dark_current_adu(self, integration_time: float) -> float:
        """
        Calculate dark current noise in ADU per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dark current noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_dark_current_electrons(integration_time)
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def calculate_read_noise_electrons(self) -> float:
        """
        Calculate read noise in electrons per pixel.
        
        Returns:
            Read noise in electrons per pixel
        """
        read_noise = self.config["detector"]["read_noise"]  # e-/pixel
        
        # Read noise is typically Gaussian, so we use the value directly
        return read_noise
    
    def calculate_read_noise_adu(self) -> float:
        """
        Calculate read noise in ADU per pixel.
        
        Returns:
            Read noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_read_noise_electrons()
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def calculate_total_instrumental_noise_electrons(self, integration_time: float) -> float:
        """
        Calculate total instrumental noise in electrons per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Total instrumental noise in electrons per pixel
        """
        total_noise_squared = 0.0
        
        sources_config = self.config.get("instrumental_sources", {})
        
        # Add dark current noise
        if sources_config.get("dark_current", {}).get("enabled", True):
            dark_noise = self.calculate_dark_current_electrons(integration_time)
            total_noise_squared += dark_noise ** 2
            logger.debug(f"Dark current noise: {dark_noise:.2f} e-/pixel")
        
        # Add read noise
        if sources_config.get("read_noise", {}).get("enabled", True):
            read_noise = self.calculate_read_noise_electrons()
            total_noise_squared += read_noise ** 2
            logger.debug(f"Read noise: {read_noise:.2f} e-/pixel")
        
        # Add other instrumental noise sources here as needed
        # For example: thermal noise, quantization noise, etc.
        
        total_noise = np.sqrt(total_noise_squared)
        
        return total_noise
    
    def calculate_total_instrumental_noise_adu(self, integration_time: float) -> float:
        """
        Calculate total instrumental noise in ADU per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Total instrumental noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_total_instrumental_noise_electrons(integration_time)
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def get_noise_breakdown_electrons(self, integration_time: float) -> Dict[str, float]:
        """
        Get breakdown of instrumental noise sources in electrons.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dictionary mapping noise source names to their contributions
        """
        breakdown = {}
        
        sources_config = self.config.get("instrumental_sources", {})
        
        # Dark current
        if sources_config.get("dark_current", {}).get("enabled", True):
            breakdown["dark_current"] = self.calculate_dark_current_electrons(integration_time)
        
        # Read noise
        if sources_config.get("read_noise", {}).get("enabled", True):
            breakdown["read_noise"] = self.calculate_read_noise_electrons()
        
        return breakdown
    
    def get_noise_breakdown_adu(self, integration_time: float) -> Dict[str, float]:
        """
        Get breakdown of instrumental noise sources in ADU.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dictionary mapping noise source names to their contributions
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        breakdown_electrons = self.get_noise_breakdown_electrons(integration_time)
        breakdown_adu = {}
        
        for source, noise_electrons in breakdown_electrons.items():
            breakdown_adu[source] = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return breakdown_adu
'''