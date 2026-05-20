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
from astropy.visualization import ZScaleInterval, ImageNormalize


from ..data.units import UnitConverter
from ..utils.helpers import format_plot_title
from ..utils.loader import config_getboolean


class InstrumentDepTerms:
    # Provides the effects of the instrument (including astro flux passed through the telescope aperture)

    def __init__(self, config: Dict, unit_converter: UnitConverter, sources_astroph: dict, sources_to_include: list):
        '''
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion object
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
            sources_to_include: List of sources to actuallyinclude in the S/N calculation (and plots of incident fluxes)
        '''

        self.config = config
        self.unit_converter = unit_converter ## ## TODO: DO I NEED THIS?
        self.sources_astroph = sources_astroph # all sources of astrophysical flux, as are incident on the instrument
        self.sources_to_include = sources_to_include

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
        #read_noise_e_rms = float(self.config["detector"]["read_noise"]) * u.electron / u.pix

        read_noise_str = self.config["detector"]["read_noise"]
        read_noise_e_rms = np.fromstring(read_noise_str, sep=',') * u.electron / u.pix # in case it's an array
        self.sources_instrum['read_noise_e_pix-1'] = read_noise_e_rms
        logging.info(f'Read noise is {read_noise_e_rms} rms')
        #self.sources_instrum['read_noise_adu'] = read_noise_e_rms / gain
        #read_noise_adu_rms = self.sources_instrum['read_noise_adu']
        #logging.info(f'Read noise is {read_noise_adu_rms} rms')

        # dark current rate 
        # e/pix/sec
        dark_current_str = self.config["detector"]["dark_current"]
        if ',' in dark_current_str:
            parts = [float(x.strip()) for x in dark_current_str.split(',')]
            dark_current_rate_e_pix_sec = np.arange(parts[0], parts[1], parts[2]) * u.electron / (u.pix * u.second)
        else:
            dark_current_rate_e_pix_sec = np.fromstring(dark_current_str, sep=',') * u.electron / (u.pix * u.second) # in case it's an array confirming to (start, stop, step)
        #dark_current_rate_e_pix_sec = np.fromstring(dark_current_str, sep=',') * u.electron / (u.pix * u.second) # in case it's an array

        logging.info(f'Dark current is {dark_current_rate_e_pix_sec} e-/pix/sec')

        # total dark current in e-, based on integration time
        # e/pix/sec -> e/pix
        integration_time = float(self.config["observation"]["t_int_obs_total"]) * u.second  # seconds
        self.sources_instrum['dark_current_e_pix-1_sec-1'] = dark_current_rate_e_pix_sec
        self.sources_instrum['dark_current_e_pix-1'] = dark_current_rate_e_pix_sec * integration_time

        # total dark current in ADU
        # e/pix -> ADU/pix
        #self.sources_instrum['dark_current_adu_pix-1'] = self.sources_instrum['dark_current_e_pix-1'] / gain

        return 

    def generate_transmission_screen(self):

        logging.info(f'Generating ersatz transmission screen...')

        # make array of pixels 10 mas on a side, centered at 0
        n_pix = 1001 # odd number to simplify centering
        pix_size_mas = 10  # milliarcseconds
        pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
        axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
        xx_arcsec, yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')
        self.screen_xx_arcsec = xx_arcsec
        self.screen_yy_arcsec = yy_arcsec
        return self.transmission_screen_2d

    '''
    def pass_through_transmission_screen(self, plot: bool = False):
        # pass each astrophysical source through the transmission screen, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec/m^2

        # make array of pixels 10 mas on a side, centered at 0
        n_pix = 1001 # odd number to simplify centering
        pix_size_mas = 10  # milliarcseconds
        pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
        axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
        xx_arcsec, yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')

        # ersatz 2D scene contributions
        ersatz_star_2D = np.zeros((n_pix, n_pix), dtype=float)
        ersatz_planet_2D = np.zeros((n_pix, n_pix), dtype=float)

        # build star
        radius_arcsec_scene = 0.1
        center_x_arcsec = 0.0
        center_y_arcsec = 0.0
        r2 = (xx_arcsec - center_x_arcsec)**2 + (yy_arcsec - center_y_arcsec)**2
        ersatz_star_2D[r2 <= radius_arcsec_scene**2] = 1.0

        # build planet
        center_x_arcsec = 3.0
        center_y_arcsec = 3.0
        radius_arcsec_balloon = 1.0
        r2 = (xx_arcsec - center_x_arcsec)**2 + (yy_arcsec - center_y_arcsec)**2
        ersatz_planet_2D[r2 <= radius_arcsec_balloon**2] = 0.2

        # cobble together a scene, with a circle of radius 0.1" at the center
        scene_no_screen = ersatz_star_2D + ersatz_planet_2D
 
        # make a sin**2 screen
        screen_transmission_ersatz = np.sin(xx_arcsec)**2

        # full scene, with screen
        scene_with_screen = scene_no_screen * screen_transmission_ersatz

        for source_name, source_val in self.sources_astroph.items():

            # pass scene elements separately through the screen 
            # kludge; remove later
            if source_name == 'star':
                #source_2D_no_screen = ersatz_star_2D * source_val['astro_flux_ph_sec_m2_um'] / ersatz_star_2D.sum()
                #source_2D_with_screen = source_2D_no_screen * screen_transmission_ersatz

                kernel_star = ersatz_star_2D / ersatz_star_2D.sum()  # (ny, nx), normalized, unitless
                flux = source_val['astro_flux_ph_sec_m2_um']         # (n_lambda,), Quantity
                # (n_lambda, ny, nx): one 2D image per wavelength bin
                source_cube_no_screen = flux[:, None, None] * kernel_star[None, :, :]
                # apply screen to each wavelength plane
                source_cube_with_screen = source_cube_no_screen * screen_transmission_ersatz[None, :, :]
            elif source_name == 'exoplanet_bb':
                kernel_planet = ersatz_planet_2D / ersatz_planet_2D.sum()  # (ny, nx), normalized, unitless
                flux = source_val['astro_flux_ph_sec_m2_um']         # (n_lambda,), Quantity
                # (n_lambda, ny, nx): one 2D image per wavelength bin
                source_cube_no_screen = flux[:, None, None] * kernel_planet[None, :, :]
                # apply screen to each wavelength plane
                source_cube_with_screen = source_cube_no_screen * screen_transmission_ersatz[None, :, :]
            else:

                kernel_other = np.ones_like(ersatz_star_2D) / np.ones_like(ersatz_star_2D).sum()  # (ny, nx), normalized, unitless
                flux = source_val['astro_flux_ph_sec_m2_um']         # (n_lambda,), Quantity
                # (n_lambda, ny, nx): one 2D image per wavelength bin
                source_cube_no_screen = flux[:, None, None] * kernel_other[None, :, :]
                # apply screen to each wavelength plane
                source_cube_with_screen = source_cube_no_screen * screen_transmission_ersatz[None, :, :]
                #source_2D_no_screen = np.ones_like(ersatz_star_2D) * source_val['astro_flux_ph_sec_m2_um'] / np.ones_like(ersatz_star_2D).sum()
                #source_2D_with_screen = source_2D_no_screen * screen_transmission_ersatz

            # sanity check: sum the pre-screen flux over the scene, and compare to the input
            ## ## TODO: PUT THIS INTO UNIT TEST
            flux_pre_screen_integrated = source_cube_no_screen.sum(axis=(1,2)) # 1D vector of pre-screen flux, per wavelength bin
            if not np.allclose(flux_pre_screen_integrated.value, source_val['astro_flux_ph_sec_m2_um'].value):
                logging.error(f"Pre-screen flux integrated over scene does not match input flux for {source_name}")
                exit()

            flux_transmitted_integrated = source_cube_with_screen.sum(axis=(1,2)) # 1D vector of transmitted flux, per wavelength bin

            # if name is right and units are right
            if ('astro_flux_ph_sec_m2_um' in source_val) and (source_val['astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                dict_this = {source_name: {
                    'wavel': source_val['wavel'], 
                    'scene_2D_no_screen': scene_no_screen, 
                    'scene_2D_with_screen': scene_with_screen,
                    'source_cube_no_screen': source_cube_no_screen,
                    'source_cube_with_screen': source_cube_with_screen,
                    'transmission_screen_2D': screen_transmission_ersatz,
                    'flux_pre_screen_ph_sec_m2_um': source_val['astro_flux_ph_sec_m2_um'],
                    'flux_post_screen_ph_sec_m2_um': source_val['astro_flux_ph_sec_m2_um']
                    }}
                self.prop_dict.update(dict_this)

            if plot:
                idx = 15 # wavelength slice index
                source_img = source_cube_no_screen[idx, :, :].value
                source_units = source_cube_no_screen[idx, :, :].unit.to_string()
                transmission_img = screen_transmission_ersatz
                source_times_transmission_img = source_cube_with_screen[idx, :, :].value
                source_times_transmission_units = source_cube_with_screen[idx, :, :].unit.to_string()

                fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
                im0 = axs[0].imshow(source_img, origin='lower', cmap='gray')
                axs[0].set_title(f"Source ({source_name})")
                axs[0].set_xlabel(f"x (pixel)")
                axs[0].set_ylabel(f"y (pixel)")
                fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label=f"{source_units}") ## ## TODO: MAY NEED TO ADD /ARCSEC**2 TO UNITS, ONCE ARCSEC ARE FULLY INCORPORATED HERE

                im1 = axs[1].imshow(transmission_img, origin='lower', cmap='gray')
                axs[1].set_title("Transmission")
                axs[1].set_xlabel(f"x (pixel)")
                axs[1].set_ylabel(f"y (pixel)")
                fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label=f"transmission")

                im2 = axs[2].imshow(source_times_transmission_img, origin='lower', cmap='gray')
                axs[2].set_title("Source * Transmission")
                axs[2].set_xlabel(f"x (pixel)")
                axs[2].set_ylabel(f"y (pixel)")
                fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label=f"{source_times_transmission_units}")

                file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"source_{source_name}_triptych.png"
                fig.savefig(file_name_plot) ## ## TODO: PEG THE FILE NAMES TO CONFIG
                plt.close(fig)
                logging.info(f"Saved plot of source {source_name} transmission triptych to {file_name_plot}")

        return
    '''


    def pass_through_aperture(self, plot: bool = False):
        # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec

        for source_name, source_val in self.sources_astroph.items():
            # if name is right and units are right
            if ('astro_flux_ph_sec_m2_um' in source_val) and (source_val['astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                dict_this = {source_name: {'wavel': source_val['wavel'], 
                'flux_post_aperture_ph_sec_um': np.multiply( float(self.config["telescope"]["collecting_area"])*u.m**2, source_val['astro_flux_ph_sec_m2_um'] )}}
                self.prop_dict.update(dict_this)
            ''' ## ## TODO: PUT THIS BACK IN ONCE READY FOR TRANSMISSION SCREEN
            if ('astro_flux_ph_sec_m2_um' in source_val) and (source_val['astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                dict_this = {source_name: {'wavel': source_val['wavel'], 
                'flux_post_aperture_ph_sec_um': np.multiply( float(self.config["telescope"]["collecting_area"])*u.m**2, source_val['astro_flux_ph_sec_m2_um'] ),
                'flux_pre_aperture_ph_sec_m2_um': source_val['flux_post_screen_ph_sec_um']}}
                self.prop_dict.update(dict_this)
            '''

        '''
        # overplot all the sources
        if plot: # pragma: no cover
            # pre-aperture fluxes
            plt.clf()
            plt.figure(figsize=(8, 8))
            for source_name, source_val in self.prop_dict.items():
                if source_name in self.sources_to_include:
                    plt.plot(source_val['wavel'], source_val['flux_pre_aperture_ph_sec_m2_um'], label=source_name)
            plt.yscale('log')
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-3, 1e10]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({source_val['wavel'].unit})")
            for source_name, source_val in self.prop_dict.items():
                plt.ylabel(f"Flux (" + str(source_val['flux_pre_aperture_ph_sec_m2_um'].unit) + ")")
            plt.legend()
            plt.title(format_plot_title("Photoelectrons, pre-aperture (no nulling yet)", self.config), loc='left')
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"photoelectrons_all_sources_pre_aperture.png"
            plt.tight_layout()
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux pre-aperture to " + file_name_plot)

            # post-aperture fluxes
            plt.clf()
            plt.figure(figsize=(8, 8))
            for source_name, source_val in self.prop_dict.items():
                if source_name in self.sources_to_include:
                    plt.plot(source_val['wavel'], source_val['flux_post_aperture_ph_sec_um'], label=source_name)
            plt.yscale('log')
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-3, 1e10]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({source_val['wavel'].unit})")
            plt.ylabel(f"Flux (" + str(source_val['flux_post_aperture_ph_sec_um'].unit) + ")")
            plt.legend()
            plt.title(format_plot_title("Photoelectrons, post-aperture", self.config), loc='left')
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"photoelectrons_all_sources_post_aperture.png"
            plt.tight_layout()
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux post-aperture to " + file_name_plot)
        '''

        logging.info(f'Passed astrophysical flux through telescope aperture...')

        return


    def photons_to_e(self):
        # convert photons to e-s, using e-to-photon relation and QE, and update prop_dict with the converted terms

        ## ## TODO: INCORPORATE REAL RESPONSE CURVES

        for source_name, source_val in self.prop_dict.items():
            if ('flux_post_aperture_ph_sec_um' in source_val) and (source_val['flux_post_aperture_ph_sec_um'].unit == u.ph / (u.um * u.s)):
                source_val['flux_e_sec_um'] = float(self.config["detector"]["photons_to_e"]) * (u.electron/u.ph) * np.multiply(float(self.config["detector"]["quantum_efficiency"]), source_val['flux_post_aperture_ph_sec_um'])

        return


    '''
    def e_to_adu(self):

        for source_name, source_val in self.prop_dict.items():
            if ('flux_e_sec_um' in source_val) and (source_val['flux_e_sec_um'].unit == u.electron / (u.um * u.s)):
                source_val['flux_adu_sec_um'] = np.divide(source_val['flux_e_sec_um'], float(self.config["detector"]["gain"])) * (u.adu/u.electron)

        logging.info(f'Converted e-s to ADUs...')

        return
    '''


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


        self.config = config

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

        # load 2D systematics (INI True/False must use config_getboolean, not bare if on strings)
        sys_section = "detector_systematics"

        def _load_systematic_map(enable_key: str, file_key: str):
            if not config_getboolean(self.config, sys_section, enable_key):
                return None
            section = self.config[sys_section]
            file_path = section[file_key] if isinstance(section, dict) else section.get(file_key)
            return fits.getdata(file_path)

        read_noise_map = _load_systematic_map("enable_read_noise_2d", "read_noise_2d_file")
        bias_map = _load_systematic_map("enable_dc_2d", "dc_2d_file")
        cosmic_rays_map = _load_systematic_map("enable_cosmic_rays_2d", "cosmic_rays_2d_file")
        hot_pixels_map = _load_systematic_map("enable_hot_pixels_2d", "hot_pixels_2d_file")

        # systematics: additive
        ## ## TODO: ARE THESE ALL ADDIITVE?
        self.systematics_additive_dict = {
            'read_noise_map': read_noise_map,
            'bias_map': bias_map,
            'cosmic_rays_map': cosmic_rays_map,
            'hot_pixels_map': hot_pixels_map
        }
        # multiplicative systematics
        self.systematics_multiplicative_dict = {}
        logging.info('Loaded detector systematics')

        return

    def footprint_spectral(self, file_name_plot: str, plot: bool = True):
        # return a boolean array of the detector, where the footprint is 1 (or a fraction, if the footprint edges do not cover a whole number of pixels)
        # file_name_plot: name of the file to save the plot to
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
            logging.info(f"Wavelength bin {wavel_bin_num} detector footprint is {footprint_this.sum()} pixels")

        # FYI FITS file
        # fits.writeto(f"/Users/eckhartspalding/Downloads/footprint_cube.fits", footprint_cube, overwrite=True)

        # generate the array with the spectrum only (no detector noise yet)
        footprint_sum = np.sum(footprint_cube, axis=0)

        logging.info(f"Total detector footprint is {footprint_sum.sum()} pixels")

        if plot: # pragma: no cover
            plt.clf()
            plt.title(format_plot_title("Detector spectral footprint (True)", self.config))
            norm = ImageNormalize(footprint_sum, interval=ZScaleInterval())
            plt.imshow(footprint_sum, origin='lower', cmap='gray', norm=norm)
            plt.xlabel(f"Pixel")
            plt.ylabel(f"Pixel")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of detector footprint containing all wavelength bins to {file_name_plot}")

        self.footprint_cube = footprint_cube

        return footprint_cube

    def convert_2d_systematics_to_1d_vector(self) -> np.ndarray:
        # convert the 2D systematics to a 1D vector, based on the footprint_cube which sets where each wavelength bin is on the detector
        # returns a 1D vector of the systematics

        # make blank canvas onto which we will add the systematics
        canvas_systematics = np.zeros((self.footprint_cube[0,:,:].shape), dtype=float)

        # read in the 2D systematics
        if len(self.systematics_additive_dict) > 0: # if there are any listed
            for key, value in self.systematics_additive_dict.items():
                if value is not None: # if there is a 2D array
                    ipdb.set_trace()
                    logging.info(f"Applying {key} systematic (additive) to the detector")
                    canvas_systematics = canvas_systematics + value
        # apply the 2D multiplicative systematics to the spectrum
        if len(self.systematics_multiplicative_dict) > 0: # if there are any listed
            for key, value in self.systematics_multiplicative_dict.items():
                if value is not None:# if there is a 2D array
                    logging.info(f"Applying {key} systematic (multiplicative) to the detector")
                    canvas_systematics = canvas_systematics * value
        
        # for reference/debugging
        #self.canvas_systematics = canvas_systematics
        
        # use the footprint_cube to sum the 2D systematics within each wavelength bin
        systematics_vector_1d = np.zeros(self.num_wavel_bins)

        for wavel_bin_num in range(0, self.num_wavel_bins):
            # for a given wavelength bin, sum the systematics across the pixels corresponding to that wavelength bin
            # self.footprint_cube[wavel_bin_num,:,:] is just a boolean array, so we multiply it
            systematics_vector_1d[wavel_bin_num] = np.sum( self.footprint_cube[wavel_bin_num,:,:] * canvas_systematics )
        ipdb.set_trace()

        '''
        # apply the 2D additive systematics to the spectrum

        systematics_vector = np.zeros(self.num_wavel_bins)
        for wavel_bin_num in range(0, self.num_wavel_bins):
            systematics_vector[wavel_bin_num] = systematics_dict[wavel_bin_num].sum()
        '''
        return systematics_vector_1d


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