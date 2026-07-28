import logging

import numpy as np
import yaml
import astropy.io.fits as fits
from scipy import ndimage

from ...viz.transmission import (
    plot_flux_through_screens,
    plot_pre_aperture_all_sources,
    plot_source_transmission_triptych,
)

class TransmissionMixin:
    def generate_instrument_transmission(self, wavel_m: float = 11e-6, override_stellar_mask = False, normalize: bool = True, plot: bool = False, angle_deg: float = 0):
        # phi_dc_vec_rad, theta_vec_2d_asec, 
        # instrument transmission respose over the sky (R_theta_vec,Dannert 2025 Eqn. B12, ignoring polarization for now)

        '''
        INPUTS:
        # wavel_m (float): Wavelength in meters (e.g., 1e-6).
        # normalize (bool): If True, normalize transmission to unity maximum. If False, max transmission is N for N identical apertures.
        # plot (bool): If True, generate plots and write FITS cubes.
        # angle_deg (float): Rotation angle of the transmission screens in degrees.

        OUTPUT:
        # Returns:
        #   transmission_instrument_response: np.ndarray of shape (6, Ny, Nx)
        #     [0]: transmission, bright output 1
        #     [1]: transmission, bright output 2
        #     [2]: transmission, dark output 3
        #     [3]: transmission, dark output 4
        #     [4]: y-coordinates on sky [arcsec]
        #     [5]: x-coordinates on sky [arcsec]
        '''

        # read in array parameters from config file
        aperture_array_definition_file_name = self.config["telescope"]["aperture_array_config_file_name"]
        with open(aperture_array_definition_file_name, 'r') as file:
            aperture_array_definition = yaml.safe_load(file)

        # construct vectors
        A_vec = [] # amplitudes
        phi_dc_vec_rad = [] # relative phase offests of each arm (one vector per aperture)
        pos_vec_m = []  # [y_m, x_m] per aperture
        phase_vector_rad_array = [] # phase vector for each output
        for aperture in aperture_array_definition['apertures']:
            A_vec.append(aperture['amplitude'])
            pos_vec_m.append([aperture['y_m'], aperture['x_m']])
        for output in aperture_array_definition['outputs']:
            phase_vector_deg = output['phase_vector_deg']
            phase_vector_rad = np.deg2rad(phase_vector_deg)
            phase_vector_rad_array.append(phase_vector_rad)
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
        
        # Calculate total number of baselines (unique pairs of apertures)
        N_apertures = len(aperture_array_definition['apertures'])
        logging.info(f'Number of apertures: {N_apertures}')
        # For N apertures, number of unique baselines = N*(N-1)/2
        N_baselines = N_apertures * (N_apertures - 1) // 2
        logging.info(f'Total number of baselines: {N_baselines}')

        #if incl_comp_transmission:
        #    cube_canvas = np.zeros((N_baselines+3, N, N))

        cube_canvas = np.zeros((3, n_pix, n_pix))

        # dict to hold bright and dark outputs
        output_all_responses = {}

        # Sum over all pairs of apertures (j, k) where j < k
        def R_m(phase_vector_rad: np.ndarray, wavel_m: float, output_name: str):
            # response of output m
            # N_apertures: number of apertures N
            # phase_vector_deg: phase vector for output m (total number of outputs is not nec. same as apertures N)
            # return: response of output m

            # convert phase vector to radians
            #phase_vector_rad = np.deg2rad(phase_vector_deg)
            R_theta_vec = np.zeros(np.shape(theta_vec_rad_array[0]))  # shape: (Ny, Nx)

            # sum over all baselines (i.e., over apertures over apertures)
            for j in range(N_apertures):
                for k in range(N_apertures):

                    # Differential phase between apertures j and k [rad]
                    del_phi_dc_jk_rad = phase_vector_rad[k] - phase_vector_rad[j]
                    
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
                    # phase_term: 2pi/lambda * b dot theta (in some notations)
                    # del_phi_dc_jk_rad: phase offset between apertures j and k [rad]
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

            # amplitude can be >1 due to addition of aperture amplitudes
            if normalize: 
                max_field_amplitude = 0
                # find the total field amplitude of the apertures, then square for transmission
                for aperture in aperture_array_definition['apertures']:
                    field_amplitude = aperture['amplitude']
                    max_field_amplitude += field_amplitude
                max_response = max_field_amplitude**2
                transmission_instrument_response /= max_response
                logging.info(f'Normalized transmission instrument response to unity, based on way bookkeeping is done downstream')


            # rotate the transmission screens
            if angle_deg != 0:   
                transmission_screens_only_rot = ndimage.rotate(transmission_instrument_response[0:4,:,:], angle_deg, axes=(1,2), reshape=False) # rotate the screens, but not the sky coordinates
                transmission_instrument_response[0:4,:,:] = transmission_screens_only_rot # reasssign the rotated screens to the original transmission_screens

            # a small override mask is put over the star for now to avoid geometrical leakage ## ## TODO: remove this once the geometry is properly implemented
            # (important to do this after rotation, to avoid numerical errors)
            # Apply only to dark outputs; bright ports should keep their geometric response.
            if override_stellar_mask and output_name in ("output_3_dark", "output_4_dark"):
                nulling_factor = float(self.config['nulling']['nulling_factor'])
                logging.info(
                    f'Star is manually being nulled to {nulling_factor} on {output_name}'
                )
                # mask a central circular region over the star
                mask_radius_pix = 4
                cy, cx = transmission_instrument_response.shape[1] // 2, transmission_instrument_response.shape[2] // 2
                y_idx, x_idx = np.ogrid[:transmission_instrument_response.shape[1], :transmission_instrument_response.shape[2]]
                circular_mask = (y_idx - cy) ** 2 + (x_idx - cx) ** 2 <= mask_radius_pix ** 2
                transmission_instrument_response[0, circular_mask] = nulling_factor

            return transmission_instrument_response


        for output in aperture_array_definition['outputs']:
            phase_vector_rad = np.deg2rad(output['phase_vector_deg'])
            transmission_instrument_response = R_m(
                phase_vector_rad=phase_vector_rad,
                wavel_m=wavel_m,
                output_name=output['name'],
            )
            output_all_responses[output['name']] = transmission_instrument_response


        '''
        # check: double Bracewell should look like HOSTS
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
        '''

        if plot:
            # save all responses to FITS files
            # output_all_responses contains output_1_bright, output_2_bright, output_3_dark, output_4_dark
            save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])
            for output_name, transmission_instrument_response in output_all_responses.items():
                fits.writeto(
                    save_dir + f"transmission_instrument_response_{output_name}.fits",
                    transmission_instrument_response,
                    overwrite=True,
                )
                logging.info(f"Saved transmission instrument response for {output_name} to {save_dir}transmission_instrument_response_{output_name}.fits")
            # save the differential dark
            differential_dark = output_all_responses['output_3_dark'] - output_all_responses['output_4_dark']
            fits.writeto(
                save_dir + f"differential_dark.fits",
                differential_dark,
                overwrite=True,
            )
            logging.info(f"Saved differential dark to {save_dir}differential_dark.fits")

        # arrange outputs into a cube, shape (4, n_pix, n_pix), slices order 0 = output_1_bright, 1 = output_2_bright, 2 = output_3_dark, 3 = output_4_dark
        # for-loop to preserve order
        keys_ordered = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark', 'yy', 'xx']
        transmission_instrument_response_cube = np.zeros((6, n_pix, n_pix))
        for t in range(4):
            transmission_instrument_response_cube[t, :, :] = output_all_responses[keys_ordered[t]][0, :, :] # screens
        transmission_instrument_response_cube[4, :, :] = output_all_responses['output_1_bright'][1, :, :] # y vals
        transmission_instrument_response_cube[5, :, :] = output_all_responses['output_1_bright'][2, :, :] # x vals

        return transmission_instrument_response_cube



    def pass_through_transmission_screens(self, fyi_angle, source_dict_pre_screen: dict, transmission_screens: np.ndarray, plot: bool = False):
        '''
        Pass each astrophysical source through the transmission screens, and update prop_dict with the propagated terms
        photons/sec/m^2 -> photons/sec/m^2

        INPUTS:
            fyi_angle (float): angle of the transmission screen (for plotting strings only)
            source_cube_no_screen (dict of Quantities): on-sky scene before transmission screen; for each key (astro source), value (Quantity array) has shape (n_wavel, n_pix, n_pix)
            transmission_screen (np.ndarray): transmission screen, shape (n_pix, n_pix)
            plot (bool): whether to plot the scene
        '''        

        transmission_screen_order = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark'] ## ## TODO: insert check to ensure always consistent
        transmission_screens = transmission_screens[0:4,:,:] # just keep the transmission slices for now ## ## TODO: include the yy and xx slices as a check somehow

        # put all the post-screen fluxes (for each source and from each channel) into a single dict
        # first check that transmission screens add up to one (energy conservation)
        net_transmission_screen = np.sum(transmission_screens, axis=0)
        source_dict_post_screen = {}
        for source_name, source_val in source_dict_pre_screen.items():
            source_dict_post_screen[source_name] = {}

            for transmission_screen_name in transmission_screen_order:
                source_dict_post_screen[source_name][transmission_screen_name] = source_val * transmission_screens[transmission_screen_order.index(transmission_screen_name), :, :]
                # collapse the sources into a single 3D array (wavel, x, y), for plotting
                # source_dict_post_screen[source_name][transmission_screen_name + '_collapsed'] = np.sum(source_dict_post_screen[source_name][transmission_screen_name], axis=(1,2))
        # there should be a cube for each output (4 cubes total)

        # collapse the sources into a single 3D array (wavel, x, y), for plotting
        # there should be a cube for each output (4 cubes total)
        #for transmission_screen_name in transmission_screen_order:

        #source_cube_post_screen = np.stack([source_dict_post_screen[source_name] for source_name in source_dict_post_screen.keys()], axis=0)
        #source_collapsed_cube_post_screen_sum = np.sum(source_cube_post_screen, axis=0)

        # integrate over 2D sky to get total flux from each source
        # update the sources
        source_integrated_dict_post_screen = {}
        for source_name, source_val in source_dict_post_screen.items():
            # source_val has 4 different screens, so integrate them separately
            self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'] = {} # will contain flux corresponding to each screen
            self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_m2_um'] = {} # will contain flux cube corresponding to each screen
            #test_flux_1 = 0 # to check flux conservation
            #test_flux_2 = 0 # to check flux conservation
            for transmission_screen_name in transmission_screen_order:
                source_val_integrated = np.sum(source_val[transmission_screen_name], axis=(1,2))
                self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'][transmission_screen_name] = source_val_integrated
                self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_m2_um'][transmission_screen_name] = source_val[transmission_screen_name]
                logging.info(f'Flux of {source_name} passed through transmission screen {transmission_screen_name}')

                # to check flux conservation, add up all the light transmitted through each screen
                #test_flux_1 += source_val_integrated
                #test_flux_2 += np.sum(source_val[transmission_screen_name], axis=(1,2))
            # if total flux after transmission is same as the input
            '''
            if np.logical_or(
                np.round(test_flux_1, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1),
                np.round(test_flux_2, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1)
            ):
                logging.error(f'Flux conservation check failed for {source_name} at angle {fyi_angle:06.2f}')
                ipdb.set_trace()
            '''
            if plot: # pragma: no cover
                file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"flux_of_{source_name}_passed_through_transmission_screens_angle_{fyi_angle:06.2f}.png"
                plot_flux_through_screens(
                    self.sources_astroph[source_name]['wavel'],
                    self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'],
                    self.sources_astroph[source_name]['pre_screen_astro_flux_ph_sec_m2_um'],
                    source_name,
                    file_name_plot,
                )

                idx = 15 # wavelength slice index (for plotting only)

                for transmission_screen_name in transmission_screen_order:
                    source_img = source_dict_pre_screen[source_name][idx, :, :].value
                    source_units = source_val[transmission_screen_name][idx, :, :].unit.to_string()
                    transmission_img = transmission_screens[transmission_screen_order.index(transmission_screen_name), :, :]
                    file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"source_transmission_map_triptych_{source_name}_angle_{fyi_angle:06.2f}_output_{transmission_screen_name}.png"
                    plot_source_transmission_triptych(
                        source_img,
                        transmission_img,
                        source_name,
                        transmission_screen_name,
                        idx,
                        source_units,
                        file_name_plot,
                    )

        if plot: # pragma: no cover
            flux_by_source = {
                source_name: (
                    self.sources_astroph[source_name]['wavel'],
                    self.sources_astroph[source_name]['pre_screen_astro_flux_ph_sec_m2_um'],
                )
                for source_name in self.sources_to_include
            }
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"photoelectrons_all_sources_pre_aperture.png"
            plot_pre_aperture_all_sources(flux_by_source, self.config, file_name_plot)

        return transmission_screens


