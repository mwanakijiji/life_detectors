"""
Instrumental noise calculations for the modules package.

This module handles calculations of instrumental noise sources including
dark current, read noise, and other detector effects.

Sky / aperture coordinate convention (use everywhere in this repo):
  - 2D sky angle arrays: index 0 = y, index 1 = x (arcsec or rad).
  - 3D cubes (3, Ny, Nx): slice 0 = science; slice 1 = y; slice 2 = x.
  - Aperture position vectors pos_vec_m: [y_m, x_m] per aperture.
  - Config pos_*_arcsec strings: "y, x" (y first, x second).
"""

from socket import IPV6_DONTFRAG
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import ipdb
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import yaml
from pathlib import Path



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
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'pre_screen_astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
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

        #########################################################################################################################
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

        #########################################################################################################################
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


    def generate_instrument_transmission(self, wavel_m: float = 11e-6, normalize: bool = True, plot: bool = False):
        # phi_dc_vec_rad, theta_vec_2d_asec, 
        # instrument transmission respose over the sky (R_theta_vec,Dannert 2025 Eqn. B12, ignoring polarization for now)

        '''
        INPUTS:
        A_vec: array of amplitudes (ex. np.array([1, 1, 1]))
        phi_dc_vec_rad: phase offsets per aperture [rad]
        wavel_m: wavelength in meters (ex. 1e-6)
        normalize: if True, normalize the transmission to 1 (otherwise, for N identical apertures, max transmission is N)
        plot: if True, plot and write FITS cubes

        OUTPUT:
        transmission_instrument_response: (3, Ny, Nx) cube
        [0] = on-sky transmission (instrument response)
        [1] = y on-sky [arcsec]
        [2] = x on-sky [arcsec]
        '''

        # read in array parameters from config file
        aperture_array_definition_file_name = self.config["telescope"]["aperture_array_config_file_name"]
        with open(aperture_array_definition_file_name, 'r') as file:
            aperture_array_definition = yaml.safe_load(file)

        # construct vectors
        A_vec = [] # amplitudes
        phi_dc_vec_rad = [] # relative phase offests of each arm
        pos_vec_m = []  # [y_m, x_m] per aperture
        for aperture in aperture_array_definition['apertures']:
            A_vec.append(aperture['amplitude'])
            phi_dc_vec_rad.append(np.deg2rad(aperture['phi_dc_deg']))
            pos_vec_m.append([aperture['y_m'], aperture['x_m']])
        A_vec = np.array(A_vec)
        phi_dc_vec_rad = np.array(phi_dc_vec_rad)
        pos_vec_m = np.array(pos_vec_m)

        n_pix = int(self.config['onsky_scene']['n_pix'])
        pix_size_mas = float(self.config['onsky_scene']['pix_size_mas'])  # milliarcseconds
        pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
        axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
        xx_arcsec, yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')
        sky_xx_arcsec = xx_arcsec
        sky_yy_arcsec = yy_arcsec
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        theta_vec_rad_array = np.zeros((2, n_pix, n_pix), dtype=float)
        theta_vec_rad_array[0] = sky_yy_arcsec * arcsec_to_rad  # θ_y [rad]
        theta_vec_rad_array[1] = sky_xx_arcsec * arcsec_to_rad  # θ_x [rad]

        # Initialize the instrument response array
        R_theta_vec = np.zeros(np.shape(theta_vec_rad_array[0]))  # shape: (Ny, Nx)

        ipdb.set_trace()
        
        # Calculate total number of baselines (unique pairs of apertures)
        N_apertures = len(aperture_array_definition['apertures'])
        logging.info(f'Number of apertures: {N_apertures}')
        # For N apertures, number of unique baselines = N*(N-1)/2
        N_baselines = N_apertures * (N_apertures - 1) // 2
        logging.info(f'Total number of baselines: {N_baselines}')

        #if incl_comp_transmission:
        #    cube_canvas = np.zeros((N_baselines+3, N, N))

        cube_canvas = np.zeros((3, n_pix, n_pix))

        # Sum over all pairs of apertures (j, k) where j < k
        for j in range(N_apertures):
            for k in range(N_apertures):

                # Differential phase between apertures j and k [rad]
                del_phi_dc_jk_rad = phi_dc_vec_rad[k] - phi_dc_vec_rad[j]
                
                # Baseline from aperture j to aperture k [m]; del_x_jk[0]=Δy, del_x_jk[1]=Δx
                del_x_jk = pos_vec_m[k] - pos_vec_m[j]

                # Compute phase term for all sky positions at once using broadcasting
                # theta_vec_rad_array has shape (2, Ny, Nx), del_x_jk has shape (2,)
                # We want to compute dot(del_x_jk, theta_vec_rad_array) for all positions
                # This gives shape (Ny, Nx)
                phase_term = (2 * np.pi / wavel_m) * (
                    del_x_jk[0] * theta_vec_rad_array[0] +  # Δy · θ_y
                    del_x_jk[1] * theta_vec_rad_array[1]    # Δx · θ_x
                )
                
                # Use cosine addition formula: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                # This is more efficient than computing cos and sin separately
                # Eqn. B12 in Dannert 2025
                # Eqn. 3 in Lay 2004
                response_jk = A_vec[j] * A_vec[k] * np.cos(del_phi_dc_jk_rad + phase_term)
                
                # Add contribution from this pair to the total response
                '''
                if plot:
                    plt.clf()
                    plt.imshow(response_jk)
                    plt.title(f'Baseline {j}-{k}')
                    plt.colorbar()
                    plt.show()
                '''
                
                '''
                # if incl_comp_transmission, add this as a separate slice
                if incl_comp_transmission:
                '''

                # Add contribution from this pair to the total response
                R_theta_vec += response_jk        

        # cube_canvas[0,:,:] = R_theta_vec
        cube_canvas[0, :, :] = R_theta_vec
        cube_canvas[1, :, :] = sky_yy_arcsec  # y [arcsec]
        cube_canvas[2, :, :] = sky_xx_arcsec  # x [arcsec]

        # conceptual point here! this response to photons is real, not complex! See Lay Eqn. (3): it's the rr*
        complex_instrument_response = cube_canvas

        # now for the actual transmission
        transmission_instrument_response = np.zeros(np.shape(cube_canvas))
        #transmission_instrument_response[0,:,:] = np.abs(complex_instrument_response[0,:,:])**2 # on-sky transmission
        transmission_instrument_response[0,:,:] = R_theta_vec # on-sky transmission

        #transmission_instrument_response[0,:,:] /= np.max(transmission_instrument_response[0,:,:]) # normalize (TODO: is this right?)
        transmission_instrument_response[1:3,:,:] = cube_canvas[1:3,:,:] # replicate coordinates

        if plot:
            plt.clf()

            ipdb.set_trace()

            arcsec_to_rad = np.pi / (180.0 * 3600.0)

            y_sky_asec = transmission_instrument_response[1, :, :]  # y at each pixel
            x_sky_asec = transmission_instrument_response[2, :, :]  # x at each pixel
            x_sky_rad = x_sky_asec * arcsec_to_rad
            y_sky_rad = y_sky_asec * arcsec_to_rad
            #y_sky_asec = y_sky_rad * 206265
            #x_sky_asec = x_sky_rad * 206265

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            extent = [
                x_sky_asec.min(), x_sky_asec.max(),   # left, right
                y_sky_asec.min(), y_sky_asec.max(),   # bottom, top
            ]

            im0 = axes[0].imshow(
                transmission_instrument_response[0, :, :],
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            #axes[0].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            #axes[0].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            axes[0].set_xlabel("x [arcsec]")
            axes[0].set_ylabel("y [arcsec]")
            axes[0].set_title("Net on-sky transmission")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # make what the HOSTS screen should look like
            hosts_trans = np.power( np.sin(np.pi * x_sky_rad * 14.4 / 11e-6), 2 )

            im1 = axes[1].imshow(
                hosts_trans,
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            #axes[1].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            #axes[1].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            axes[1].set_xlabel("x [arcsec]")
            axes[1].set_ylabel("y [arcsec]")
            axes[1].set_title("HOSTS transmission (l/B=0.158asec)")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            ipdb.set_trace()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"transmission_instrument_response.png"
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of transmission instrument response to {file_name_plot}")
            plt.close(fig)

            save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])
            transmission_cube_3d = np.stack(
                (transmission_instrument_response[0, :, :], y_sky_asec, x_sky_asec),
                axis=0,
            )
            hosts_cube_3d = np.stack((hosts_trans, y_sky_asec, x_sky_asec), axis=0)
            fits.writeto(
                save_dir + "transmission_instrument_response_cube.fits",
                transmission_cube_3d,
                overwrite=True,
            )
            fits.writeto(
                save_dir + "hosts_transmission_cube.fits",
                hosts_cube_3d,
                overwrite=True,
            )
            logging.info(
                f"Saved 3D FITS cubes: {save_dir}transmission_instrument_response_cube.fits, "
                f"{save_dir}hosts_transmission_cube.fits"
            )
            
        if normalize: 
            transmission_instrument_response /= np.max(transmission_instrument_response)

        return transmission_instrument_response


    def pass_through_transmission_screen(self, source_dict_pre_screen: dict, transmission_screen: np.ndarray, plot: bool = False):
        '''
        Pass each astrophysical source through the transmission screen, and update prop_dict with the propagated terms
        photons/sec/m^2 -> photons/sec/m^2

        INPUTS:
            source_cube_no_screen (dict of Quantities): on-sky scene before transmission screen; for each key (astro source), value (Quantitiy array) has shape (n_wavel, n_pix, n_pix)
            transmission_screen (np.ndarray): transmission screen, shape (n_pix, n_pix)
            plot (bool): whether to plot the scene
        '''        

        source_dict_post_screen = {}
        for source_name, source_val in source_dict_pre_screen.items():
            source_dict_post_screen[source_name] = source_val * transmission_screen[None, :, :]
        
        # collapse the sources into a single 3D array (wavel, x, y), for plotting
        source_cube_post_screen = np.stack([source_dict_post_screen[source_name] for source_name in source_dict_post_screen.keys()], axis=0)
        source_collapsed_cube_post_screen_sum = np.sum(source_cube_post_screen, axis=0)

        # integrate over 2D sky to get total flux from each source
        # update the sources
        source_integrated_dict_post_screen = {}
        for source_name, source_val in source_dict_post_screen.items():
            source_val_integrated = np.sum(source_val, axis=(1,2))
            self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'] = source_val_integrated
            self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_um'] = source_dict_post_screen[source_name]
            logging.info(f'Flux of {source_name} passed through transmission screen')

        # if name is right and units are right
        '''
        if ('pre_screen_astro_flux_ph_sec_m2_um' in source_val) and (source_val['pre_screen_astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
            dict_this = {source_name: {
                'wavel': source_val['wavel'], 
                'scene_2D_no_screen': scene_no_screen, 
                'scene_2D_with_screen': scene_with_screen,
                'source_cube_no_screen': source_cube_no_screen,
                'source_cube_with_screen': source_cube_with_screen,
                'transmission_screen_2D': screen_transmission_ersatz,
                'flux_pre_screen_ph_sec_m2_um': source_val['pre_screen_astro_flux_ph_sec_m2_um'],
                'flux_post_screen_ph_sec_m2_um': source_val['pre_screen_astro_flux_ph_sec_m2_um']
                }}
        self.prop_dict.update(dict_this)
        '''
        if plot:
            idx = 15 # wavelength slice index
            for source_name, source_val in source_dict_post_screen.items():
                source_img = source_dict_pre_screen[source_name][idx, :, :].value
                source_units = source_val[idx, :, :].unit.to_string()
                transmission_img = transmission_screen
                source_times_transmission_img = source_val[idx, :, :].value
                source_times_transmission_units = source_val[idx, :, :].unit.to_string()

                fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
                im0 = axs[0].imshow(source_img, origin='lower', cmap='gray')

                fig.suptitle(f"Source: {source_name}, idx_wavel: {idx}")
                axs[0].set_title("Source")
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
                fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label=f"{source_times_transmission_units}")

                file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"source_transmission_map_triptych_{source_name}.png"
                fig.savefig(file_name_plot)
                plt.close(fig)
                logging.info(f"Saved plot of source, transmission, source * transmission triptych to {file_name_plot}")

        return


    def pass_through_aperture(self, plot: bool = False):
        # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec

        for source_name, source_val in self.sources_astroph.items():
            # if name is right and units are right
            if ('flux_cube_post_screen_ph_sec_um' in source_val) and (source_val['flux_cube_post_screen_ph_sec_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                dict_this = {source_name: {
                                        'wavel': source_val['wavel'], 
                                        'flux_cube_post_screen_pre_aperture_ph_sec_m2_um': source_val['flux_cube_post_screen_ph_sec_um'],
                                        'flux_cube_post_screen_post_aperture_ph_sec_um': np.multiply( float(self.config["telescope"]["collecting_area"])*u.m**2, source_val['flux_cube_post_screen_ph_sec_um'] )
                                        }}
                self.prop_dict.update(dict_this)

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