"""
Main noise calculator for the modules package.

This module provides the primary interface for calculating total noise
and signal-to-noise ratios for infrared detector observations.
"""

import glob
import logging
import os
from pathlib import Path

import astropy.units as u
import configparser
import h5py
import ipdb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy.io import fits
from astropy.table import QTable
from dataclasses import dataclass
from ipaddress import ip_network
from typing import Any, Dict, List, Optional, Tuple

from .astrophysical import AstrophysicalSources
from .instrumental import Detector, InstrumentDepTerms
from ..data.units import UnitConverter
from ..utils.helpers import ensure_plot_title_context, format_plot_title
from ..utils.validator import validate_config

logger = logging.getLogger(__name__)

'''
def calculate_astrophysical_noise_adu(total_astro_adu: float) -> np.ndarray:
    """
    Calculate astrophysical noise in electrons per pixel.
    
    Args:
        total_astro_adu: the total astrophysical flux contribution to the readout in ADU
        
    Returns:
        Noise in electrons per pixel
    """
    
    # photon noise: sqrt(N)
    noise_adu = np.sqrt(total_astro_adu)
    
    return noise_adu
'''


def calculate_s2n_post_rotation(read_dir, config):
    """
    Calculate the S/N of the chopped dark outputs.

    Args:
        read_dir: dir containing the HDF5 files
        config: configuration dictionary

    Returns:
        S_p_sqd_phi_mean: mean of the squared signal of the chopped dark outputs
        S_p_3_sqd_phi_mean: mean of the squared signal of the chopped dark output 3
        SNR_lambda_array: SNR for each wavelength bin
        SNR_tot: total SNR
    """

    hdf5_files = glob.glob(os.path.join(read_dir, '*.hdf5'))
    by_dc_qe = {}
    ensure_plot_title_context(config)

    # read in one HDF5 file per angle and QE value
    for hdf5_file in hdf5_files:
        angle = float(Path(hdf5_file).stem.removeprefix("angle_"))

        with h5py.File(hdf5_file, "r") as f:
            # for all DC, QE values

            for dc_qe_str in f.keys():
                # for all outputs
                for ch in f[dc_qe_str].keys():
                    if ch.endswith(".__table_column_meta__"):
                        continue
                    tbl = QTable.read(hdf5_file, path=f"{dc_qe_str}/{ch}")

                qe = tbl.meta['qe']

                # calculate the S/N of the chopped dark outputs
                # need S_p and S_p_3; see Dannert+ 2022 Eqn. 20
                chopped = QTable.read(hdf5_file, path=f"{dc_qe_str}/chopped")
                S_p = chopped["chopped_astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"]
                out3 = QTable.read(hdf5_file, path=f"{dc_qe_str}/output_3_dark")
                S_p_3 = out3["astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"]
                wavel = chopped["wavel_bin_center"].value
                wavel_bin_edges = chopped.meta["wavel_bin_edges"]

                slot = by_dc_qe.setdefault(dc_qe_str,
                                        {"wavel": wavel, "S_p": {}, "S_p_3": {},
                                        "chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot": {},
                                        "chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot": {}})
                slot["S_p"][angle] = S_p
                slot["S_p_3"][angle] = S_p_3

                #slot["instrum_dark_current_chopped"][angle] = # self.output_channels["output_3_dark"].instrum_noise["dark_current_e_pix-1_sec-1"] * chopped["n_pix_per_wavel_bin"] * t_int_frame / gain chopped["instrum_dark_current_chopped"]
                slot["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"][angle] = chopped["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"]
                slot["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"][angle] = chopped["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"]
                slot["wavel_bin_width"] = chopped["wavel_bin_width"] ## ## TODO: use the proper bin edges

                # symmetric sources: per-output flux on output_3_dark (note this is NOT chopped!); Dannert+ 2022 Eqn. 20
                SYM_TAGS = ('star', 'exozodiacal', 'zodiacal')
                slot.setdefault('sources_sym', {})
                for source_name in SYM_TAGS:
                    col = f'astro_{source_name}_flux_adu_sec_for_wavel_bin_and_integration_tot'
                    if col not in out3.colnames:
                        continue
                    slot['sources_sym'].setdefault(source_name, {'Ssym_dark_3': {}})
                    slot['sources_sym'][source_name]['Ssym_dark_3'][angle] = chopped['output_3_dark_astro_'+source_name+'_flux_adu_sec_for_wavel_bin_and_integration_tot']

    for dc_qe_str, slot in by_dc_qe.items():
        gain = float(config['detector']['gain']) * u.electron / u.adu  # e-/ADU
        angles = sorted(slot["S_p"].keys())
        N_angles = len(angles)
        # (n_bins, n_angles) — use .value if columns are Quantity

        cols_S_p_adu = []
        cols_S_p_3_adu = []
        cols_S_p_elec = []
        cols_S_p_3_elec = []
        for a in angles:
            # get photoelectron quantities to find S/N
            ## ## TODO: if not 1 electron per photon, this breaks reasoning
            cols_S_p_adu.append(np.asarray(slot["S_p"][a])) # adu units are lost here, but will be restored later
            cols_S_p_3_adu.append(np.asarray(slot["S_p_3"][a])) # adu units are lost here, but will be restored later
            cols_S_p_elec.append(np.asarray(slot["S_p"][a] * gain) ) # photoelectrons units are lost here, but will be restored later
            cols_S_p_3_elec.append(np.asarray(slot["S_p_3"][a]* gain) ) # photoelectrons units are lost here, but will be restored later
        S_p_arr_elec = np.column_stack(cols_S_p_elec) * u.electron
        S_p_sqd_arr_elec = np.power(S_p_arr_elec, 2)
        S_p_3_arr_elec = np.column_stack(cols_S_p_3_elec) * u.electron
        S_p_3_sqd_arr_elec = np.power(S_p_3_arr_elec, 2)

        S_p_sqd_arr_mean_elec = S_p_sqd_arr_elec.mean(axis=1) # Dannert+ 2022 Eqn. (20)
        S_p_3_sqd_arr_mean_elec = S_p_3_sqd_arr_elec.mean(axis=1)  # Dannert+ 2022 Eqn. (20)

        #N_sym_adu = 0 * u.adu # symmetric sources
        #gain = float(config['detector']['gain'])
        #S_sym_shot_adu = np.sqrt(N_sym_adu * gain)   # = sqrt(sigma_sym_noise) / gain

        # symmetric noise sources (angle-averaged shot noise; Dannert+ 2022 Eqn. 20)
        sources_sym = slot.get('sources_sym', {})
        S_sym_noise_var_3_elec = None
        logging.info(f'Astrophysical sources considered to be symmetric: {[source_name for source_name in sources_sym.keys()]}')
        for source_name, source_dict in sources_sym.items():
            cols_sym_noise_var_3_elec = [] # will contain the noise of symmetric sources for each angle
            for a in angles:
                # absolute signal from S3 (in photoelectrons) is the same as the noise var of output 3
                # note just using S3 here; effect of S4 (which is symmetric) is included downstream with sqrt(2)
                sym_noise_var_this_source_this_angle_dark_3_elec = source_dict['Ssym_dark_3'][a] * gain
                # for averaging symmetric noise vars across angles (kind of redundant, because the symmetric sources are not expected to change)
                cols_sym_noise_var_3_elec.append(np.sqrt(sym_noise_var_this_source_this_angle_dark_3_elec.value) * u.electron)
            # symmetric noise var averaged across angles for this source
            #sym_sigma_mean_3_elec = np.mean(np.column_stack(cols_sym_noise_var_3_elec), axis=1)
            # symmetric noise var averaged across angles for this source
            sym_noise_var_mean_3_elec = np.mean(np.column_stack(cols_sym_noise_var_3_elec).value * u.electron, axis=1)
            # add chopped photon noise from all the astrophysical sources in quadrature
            S_sym_noise_var_3_elec = sym_noise_var_mean_3_elec if S_sym_noise_var_3_elec is None else S_sym_noise_var_3_elec + sym_noise_var_mean_3_elec
        #ipdb.set_trace()
        # noise sigma and var from all symmetric sources
        if S_sym_noise_var_3_elec.unit == u.electron:
            S_sym_3_sigma_elec = np.sqrt(S_sym_noise_var_3_elec.value) * u.electron if S_sym_noise_var_3_elec is not None else np.zeros(len(slot["wavel_bin_width"])) * u.electron
        else:
            logger.error('Unit inconsistency in symmetric astrophysical noise sources!')
            exit()
        #S_sym_3_var = np.power(S_sym_3_sigma, 2)

        SNR_lambda_array = []
        for wavel_bin_num in range(len(slot["wavel_bin_width"])):
            #wavel_start = slot["wavel_bin_edges"][wavel_bin_num].value
            #wavel_stop = slot["wavel_bin_edges"][wavel_bin_num + 1].value

            d_wavel = slot["wavel_bin_width"][wavel_bin_num]

            # astrophysical signals (all must be in photoelectrons)
            # Dannert+ 2022 Eqn. (19)-(20): sqrt of angle-averaged squared signals per bin
            S_p_rms_phi = np.sqrt(S_p_sqd_arr_mean_elec[wavel_bin_num])
            S_p_3_rms_phi = np.sqrt(S_p_3_sqd_arr_mean_elec[wavel_bin_num])

            # term for dark current for this wavelength bin
            # note this is not 2*sqrt(N_angles) etc. because the dark current term is already from the CHOPPED signal, not from each of the 2 dark outputs
            # note also that the dark current is assumed to be independent of viewing angle (so just use angle 0.0 here)
            S_dark_noise_var = np.power(slot["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"][0.0][wavel_bin_num] * gain, 2).value * u.electron
            # again, so 2*sqrt(N_angles) etc. because the net read noise for this wavelength bin is from the chopped signal; also consider independent of viewing angle
            S_read_noise_var = np.power(slot["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"][0.0][wavel_bin_num] * gain, 2).value * u.electron
            #S_instrumental_var = S_dark_noise_var + S_read_noise_var
            #S_instrumental_sigma = np.sqrt(S_instrumental_var).value * u.electron

            # symmetric astrophysical noise sources
            S_sym_3_var_this = S_sym_noise_var_3_elec[wavel_bin_num]
            S_sym_3_sigma_this = S_sym_3_sigma_elec[wavel_bin_num]

            # put it all together
            # note that integration over wavelengths of this wavelength bin is already included (i.e., they are bin totals),
            # since the signals were already multiplied by the wavelength bin further upstream
            numerator_ = S_p_rms_phi
            # note S_sym_3_var_this is same as S_sym_3 this, assuming Poisson noise
            astro_noise_term = 2 * (S_sym_3_var_this + S_p_3_rms_phi)
            instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)
            #instrum_noise = S_instrumental_sigma # note there is no sqrt(2) (detector noise from 2 detectors) because it is aleady being added in quadrature further upstream
            # add noise terms; note astronoise is already the quadrature term, so no **2 on it
            # debugging:
            #instrum_noise_term = 0 * u.electron # this is just for testing
            denominator_ = np.sqrt(astro_noise_term + instrum_noise_term).value * u.electron

            SNR_lambda = numerator_ / denominator_
            SNR_lambda_array.append(SNR_lambda.value)

        SNR_tot = np.sqrt(np.sum(np.power(SNR_lambda_array, 2)))
        print(f'SNR_tot for DC {dc_qe_str}: {SNR_tot}')

        t_int_frame = float(config['observation']['t_int_frame'])
        n_angles_cfg = int(float(config['observation']['N_angles']))
        n_int_per_angle = int(float(config['observation']['N_int_per_angle']))
        n_int_total = n_angles_cfg * n_int_per_angle * t_int_frame
        qe_val = float(config['detector']['quantum_efficiency'])

        # plot SNR
        if True:  # pragma: no cover
            fig = plt.figure(figsize=(8, 8), constrained_layout=True)
            plt.clf()
            plt.stairs(SNR_lambda_array, edges=wavel_bin_edges.value)
            plt.xlim([4, 18.5])
            plt.yscale('log')
            plt.grid(True)
            plt.xlabel(f'Wavelength ({slot["wavel_bin_width"][wavel_bin_num].unit})')
            plt.ylabel('SNR')
            base_title = (
                f"SNR for DC {dc_qe_str}  |  SNR_tot = {SNR_tot:.4g}  |  "
                f"N_angles = {n_angles_cfg}  |  N_int_per_angle = {n_int_per_angle}  |  "
                f"N_int tot = {n_int_total} sec"
            )
            plt.title(format_plot_title(base_title, config), fontsize=8, loc='left')
            file_name_plot = (
                str(config['dirs']['save_s2n_data_unique_dir'])
                + f"SNR_vs_wavelength_{dc_qe_str}"
                + f"_Nang_{n_angles_cfg}_Nintpa_{n_int_per_angle}_Ninttot_{n_int_total}.png"
            )
            plt.tight_layout()
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of SNR vs wavelength for {dc_qe_str} to {file_name_plot}")

    return


class NoiseCalculator:
    """
    Puts astrophysical and instrumental sources together.
    """
    
    def __init__(self, config: Dict, sources_all, sources_to_include: list):
        """
        Initialize the noise calculator.
        
        Args:
            config: Configuration dictionary containing all parameters
            sources_all: the object including the various fluxes and noise contributions, from astro and instrumental sources
            sources_to_include: list of sources to include in the S/N calculation
            
        Raises:
            ValueError: If configuration is invalid
        """

        self.config = config

        # the object that is the 'origin' of the various noise contributions for the calculations to follow
        self.sources_all = sources_all
        self.sources_to_include = sources_to_include

    def s2n_e(self, file_name_fits_unique, plot: bool = False):
        '''
        Find S/N using photoelectrons

        Ref.: s_to_n_logic_life_detectors.pdf
        '''

        ## everything in units of photoelectrons
        
        #wavel_abcissa = self.noise_origin.prop_dict['wavel']
        logging.info("Calculating S/N ...")

        ## ## TO DO: MAKE ALL WAVELS TO BE ON AN EXPLICIT COMMON BASIS ACCORDING TO THE BINNING
        wavel_abcissa = self.sources_all.sources_astroph['star']['wavel']

        # a constant pixel spacing for each wavelength bin is assumed (but wavelength spacing within each bin is not constant)

        # map wavelengths-bins
        R = float(self.config["detector"]["spec_res"]) # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation lambda_i = lambda_{0} * (1 + 1/R)**i
        lambda_min, lambda_max = float(self.config["wavelength_range"]["min"]) * u.um, float(self.config["wavelength_range"]["max"])  * u.um
        # number of bins that fit fully in [lmin, lmax]
        n_bins = int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / R)))

        # geometric bin edges and centers
        bin_edges = lambda_min * (1.0 + 1.0 / R) ** np.arange(n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = bin_edges[1:]-bin_edges[:-1] # removed units for plotting

        # instantiate detector objects for each of the channels
        detector_bright_1 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_bright_2 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_dark_3 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_dark_4 = Detector(config=self.config, num_wavel_bins=n_bins)

        # get the boolean illumination footprint (cube where each slice is the footprint for one wavelength bin)
        ## ## NOTE THIS IS KIND OF REDUNDANT RIGHT NOW, SINCE THE NUMBER OF PIXELS PER WAVELENGTH BIN IS CONSTANT AS CALCULATED BELOW; MIGHT CHANGE THIS LATER IF THE DISPERSION IS NOT CONSTANT
        footprint_spec_cube_bright_1 = detector_bright_1.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_bright_1.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_bright_2 = detector_bright_2.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_bright_2.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_dark_3 = detector_dark_3.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_dark_3.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_dark_4 = detector_dark_4.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_dark_4.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT

        # integration time for 1 frame
        #t_int = float(self.config["observation"]["t_int_obs_total"]) * u.second
        n_int = int(self.config["observation"]["N_angles"]) # number of frames
        logging.info(f'Number of frames: {n_int:d}')

        # the number of pixels for each wavelength bin
        # (in practice the number of pixels is the same for all wavelength bins, but this can be updated later if the dispersion is not constant)
        #n_pix_array_reshaped = np.tile( np.sum(footprint_spec_cube[0,:,:]) * np.ones(len(wavel_bin_centers)), (len(D_tot), 1) ) * u.pix # shape (N_dark_current, N_wavel)
        n_pix_array = u.Quantity([]) * u.pix
        for wavel_bin_num in range(0, n_bins):
            val = np.sum(footprint_spec_cube_bright_1[wavel_bin_num, :, :]) * u.pix
            n_pix_array = u.Quantity(np.append(n_pix_array, val))

        # retrieve the addl systematics vector
        addl_systematics_vector_bright_1 = detector_bright_1.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_bright_2 = detector_bright_2.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_dark_3 = detector_dark_3.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_dark_4 = detector_dark_4.convert_2d_systematics_to_1d_vector()

        #ipdb.set_trace()
        
        # Now the calculation will broadcast to (N_dark_current, N_wavel)
        # return S/N; and the variable values that are either the dark current or read noise (whichever has length >1)
        #ipdb.set_trace()
        #s2n_dark_3 = self.s2n_val(wavel_bin_centers=bin_centers, del_lambda_array=bin_widths, n_pix_array=n_pix_array, addl_systematics_vector=addl_systematics_vector_dark_3)


    
        # write the S/N data to a FITS file, with the config data in the header
        file_name_fits_temp = self.config['saving']['save_s2n_data_temp'] # this is just a file that is repeatedly written over in the case of a batch job, but it handy if I want to check the last written thing
        hdu = fits.PrimaryHDU()  # s2n will be packed into this later

        # add the sweeped parameters to the header (these are what are used to make the axes of the big 4D cubes)
        hdu.header['N_INT'] = n_int
        hdu.header['QE'] = self.config["detector"]["quantum_efficiency"]

        # Add config data to header
        relevant_sections = ['telescope', 'target', 'nulling', 'detector', 'observation', 'wavelength_range']
        for section_name, _ in self.config.items():

            if section_name not in relevant_sections:
                continue

            # Add a blank line before each section (except the first)
            if len(hdu.header) > 0:
                hdu.header.add_blank('')
            
            # Add section header as a comment
            #hdu.header.add_comment(f"===== {section_name.upper()} =====")
            
            # Add each key-value pair in the section
            for key, value in self.config[section_name].items():
                # Create a hierarchical keyword using HIERARCH for long names
                # FITS keywords are limited to 8 characters, but HIERARCH allows arbitrary length
                hierarch_key = f"HIERARCH {section_name}.{key}"

                value_str = str(value)
                if len(value_str) > 68:
                    logger.warning(
                        f"Skipping FITS header entry (value too long): {hierarch_key}={value_str!r}"
                    )
                    continue
                # Validate keyword/value length before writing 
                fits.Card(hierarch_key, value)
                hdu.header[hierarch_key] = value
            

        # stuff for plots and FITS file too
        # parse dark current values (can be comma-separated list)
        dark_current_str = self.config['detector']['dark_current']
        if ',' in dark_current_str:
            #dark_current_values = [float(x.strip()) for x in dark_current_str.split(',')] * u.electron / (u.pix * u.second)
            #dark_current_display = ', '.join([f"{val:.2f}" for val in dark_current_values.value])
            parts = [float(x.strip()) for x in dark_current_str.split(',')]
            dark_current_values = np.arange(parts[0], parts[1], parts[2]) * u.electron / (u.pix * u.second)
            dark_current_display = ', '.join([f"{val:.2f}" for val in dark_current_values.value])
        else:
            dark_current_values = float(dark_current_str) * u.electron / (u.pix * u.second)
            dark_current_display = f"{float(dark_current_str):.2f}"
        # do the same for read noise
        read_noise_str = self.config['detector']['read_noise']
        if ',' in read_noise_str:
            read_noise_values = [float(x.strip()) for x in read_noise_str.split(',')] * u.electron
            read_noise_display = ', '.join([f"{val:.2f}" for val in read_noise_values.value])
        else:
            read_noise_display = f"{float(read_noise_str):.2f}"
            read_noise_values = float(read_noise_str) * u.electron

        # TODO: wire per-channel s2n_val calls for bright/dark outputs
        n_s2n_rows = int(np.size(dark_current_values)) if np.size(dark_current_values) > 1 else 1
        s2n = np.ones((n_s2n_rows, n_bins))

        # pack stuff into FITS file and save
        # add two slices denoting the wavelength bin centers, wavelength bin widths, and the dark current values
        s2n_wavel_bin_centers = np.zeros(s2n.shape)
        s2n_wavel_bin_widths = np.zeros(s2n.shape)
        s2n_dc = np.zeros(s2n.shape)
        # s2n_wavel is function of x only, s2n_dc is function of y only
        s2n_wavel_bin_centers[:,:] = np.tile(bin_centers.value, (s2n.shape[0], 1))
        s2n_wavel_bin_widths[:,:] = np.tile(bin_widths.value, (s2n.shape[0], 1))
        s2n_dc[:,:] = np.tile(dark_current_values.value.reshape(-1, 1), (1, s2n.shape[1]))
        s2n_complete = np.stack((s2n, s2n_wavel_bin_centers, s2n_wavel_bin_widths, s2n_dc), axis=0)
        hdu.data = s2n_complete
        hdu.writeto(file_name_fits_temp, overwrite=True)
        logger.info(f"Wrote S/N, wavelength, and dark current data to overwriteable file {file_name_fits_temp}")
        hdu.writeto(file_name_fits_unique, overwrite=True)
        logger.info(f"Wrote S/N, wavelength, and dark current data to unique file {file_name_fits_unique}")

        # pragma: no cover
        # Prepare two left-aligned columns for figure metadata
        total_integration_time = float(self.config['observation']['t_int_frame']) * float(self.config['observation']['N_angles']) * u.second
        instrumental_lines = [
            "INSTRUMENTAL:",
            "\n",
            f"collecting area = {float(self.config['telescope']['collecting_area']):.2f} m²",
            f"telescope throughput = {float(self.config['telescope']['eta_t']):.2f}",
            f"stellar nulling = {bool(self.config['nulling']['null'])}",
            f"nulling transmission = {float(self.config['nulling']['nulling_factor']):.2e}",
            f"quantum efficiency = {float(self.config['detector']['quantum_efficiency']):.2f}",
            f"dark current = {dark_current_display} e-/pix/sec",
            f"read noise = {read_noise_display} e- rms",
            f"gain = {float(self.config['detector']['gain']):.2f} e-/ADU",
            f"pix per wavel bin = {float(self.config['detector']['pix_per_wavel_bin']):.2f}",
            f"integration time, total for obs. = {total_integration_time} sec",
            f"integration time per readout = {float(self.config['observation']['t_int_frame']):.2f} sec",
            f"number of readouts = {int(n_int)}"
        ]
        astrophysical_lines = [
            "ASTROPHYSICAL:",
            "\n",
            f"stellar temperature = {float(self.config['target']['T_star']):.2f} K",
            f"stellar radius = {float(self.config['target']['rad_star']):.2f} solar radii",
            f"distance = {float(self.config['target']['distance']):.2f} pc",
            f"planet temperature = {float(self.config['target']['pl_temp']):.2f} K",
            f"planet radius = {float(self.config['target']['rad_planet']):.2f} Earth radii",
            f"planet albedo = {float(self.config['target']['A_albedo']):.2f}",
            fr"galactic $\lambda_{{\rm rel}}$ = {float(self.config['target']['lambda_rel_lon_los']):.2f} deg, $\beta$ = {float(self.config['target']['beta_lat_los']):.2f} deg",
        ]


        # pragma: no cover
        ############################
        # 2D plot of S/N vs wavelength and dark current
        if plot: # pragma: no cover
            plt.close()

            param_name = "Dark current" if np.size(dark_current_values) > 1 else "Read noise"
            param_units_string = 'e_pix-1_sec-1' if np.size(dark_current_values) > 1 else "e rms"
            param_values = dark_current_values if np.asarray(dark_current_values).size > 1 else read_noise_values
            #param_values = np.asarray(param_values, dtype=float)  
            N = len(param_values)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Heatmap of S/N with contours overlaid
            # make edges of the y-axis
            y_c = param_values.value
            y_edges = np.concatenate((
                [y_c[0] - 0.5*(y_c[1] - y_c[0])],
                0.5*(y_c[:-1] + y_c[1:]),
                [y_c[-1] + 0.5*(y_c[-1] - y_c[-2])]
            ))
            # Align imshow with physical axes using wavelength edges and parameter range
            im = ax.pcolormesh(
                bin_edges.value, y_edges, s2n,
                cmap='viridis', shading='flat'
                )
            levels_2d = self.config['plotting']['s2n_levels_2d']
            levels_2d = [float(x.strip()) for x in levels_2d.split(',')]
            contour = ax.contour(
                bin_centers.value, param_values.value, s2n,
                levels=levels_2d,
                colors=['white', 'white'],
                linewidths=2,
                linestyles=['dashed', 'solid']
            )
            '''
            ax.clabel(contour, inline=True, fmt='%g', fontsize=9)
            fig.colorbar(im, ax=ax, label='S/N')
            ax.set_xlabel(f"Wavelength ({bin_centers.unit})")
            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_title(format_plot_title("S/N", self.config))
            plt.tight_layout()
            plt.show()
            im = ax.imshow(
                s2n,
                extent=[bin_edges[0].value, bin_edges[-1].value,
                        np.min(param_values.value), np.max(param_values.value)],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            contour = ax.contour(
                bin_centers.value, param_values.value, s2n,
                levels=[1, 5],
                colors=['white', 'white'],
                linewidths=2,
                linestyles=['dashed', 'solid']
            )
            '''
            ax.clabel(contour, inline=True, fmt='%g', fontsize=9)
            fig.colorbar(im, ax=ax, label='S/N')
            ax.set_xlabel(f"Wavelength ({bin_centers.unit})")
            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_title(format_plot_title("S/N", self.config))
            plt.tight_layout()
            #plt.show()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"2d_s2n_vs_wavelength_and_dark_current.png"
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")

            '''
            tick_idx = np.linspace(0, N - 1, num=min(N, 10), dtype=int)
            ax.set_yticks(tick_idx)
            labels = [f"{param_values[i].value:g}" for i in tick_idx]
            ax.set_yticklabels(labels)

            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_xlabel("Wavelength bin (" + str(wavel_abcissa.unit) + ")")
            fig.colorbar(contour, ax=ax, label="S/N")
            # Keep a concise axes title and add two figure-level columns
            ax.set_title(format_plot_title("S/N", self.config))
            fig.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            #plt.tight_layout()
            plt.subplots_adjust(top=0.6,right=0.8)
            plt.show()
            '''

            #plt.imshow(s2n,aspect='auto', origin='lower')
            #plt.imshow(s2n, origin='lower')
            #plt.colorbar()

            # Set y-axis ticks to match D_rate_reshaped[:,0]
            #plt.yticks(range(len(D_rate_reshaped[:,0])), D_rate_reshaped[:,0])

            # Set bottom x-axis: wavelength
            plt.clf()
            # Set x-ticks at intervals of every three values along the x axis, and rotate the labels 30 degrees
            interval = 3
            # Set x-ticks at intervals of every two values along the x axis
            x_tick_indices = np.arange(0, len(wavel_abcissa), interval)

            ax_bottom = plt.gca()
            ax_bottom.set_xticks(x_tick_indices)
            ax_bottom.set_xticklabels(np.round(wavel_abcissa[x_tick_indices], 2), rotation=30)
            ax_bottom.set_xlabel('Wavelength (um)')

            # Add contour lines for S/N = 1 (dashed) and S/N = 5 (solid)
            X, Y = np.meshgrid(np.arange(s2n.shape[1]), np.arange(s2n.shape[0]))
            contour = ax_bottom.contour(
                X, Y, s2n, 
                levels=[1, 5], 
                colors=['white', 'white'],  
                linewidths=2, 
                linestyles=['dashed', 'solid']
            )
            ax_bottom.clabel(contour, inline=True, fmt='%.1f', fontsize=9)
            plt.ylabel(param_name + " (" + param_units_string + ")")
            # Keep a concise axes title and add two figure-level columns
            ax_bottom.set_title(format_plot_title("S/N", self.config))
            fig2 = plt.gcf()
            fig2.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig2.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            plt.tight_layout()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"2d_s2n_vs_wavelength_and_dark_current.png"
            #plt.show()
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")
            plt.close()

            

            '''
            for plot_num in range(0,len(s2n[:,0])):
                # draw a histogram-like plot of S/N using step plot that respects bin widths
                # Create x-coordinates that include both bin edges for proper step plotting
                x_step = np.repeat(wavel_bin_edges_lower, 2)
                x_step[1::2] = wavel_bin_edges_upper  # Replace every other element with upper edges
                y_step = np.repeat(s2n[plot_num,:], 2)  # Repeat y-values for step effect
                plt.plot(x_step, y_step, label=f'Dark current {plot_num}', linewidth=2)
                
                #plt.bar(wavel_bin_edges_lower.value, s2n[plot_num,:], width=bin_widths, align='edge', edgecolor='black', linewidth=0)
                #plt.plot(wavel_abcissa, s2n[plot_num,:], label=str(int(plot_num)))
            '''
            plt.figure(figsize=(10, 8))
            for plot_num in range(0,len(s2n[:,0])):
                plt.scatter(bin_centers, s2n[plot_num,:], alpha=0.5)
            # Add vertical lines at bin edges for reference
            #for line in wavel_bin_edges_lower:
            #     plt.axvline(x=line, color='gray', linestyle='--', alpha=0.5)
            cmap = cm.viridis  # or any other colormap
            norm = mcolors.Normalize(vmin=np.min(param_values.value), vmax=np.max(param_values.value))

            for plot_num in range(0,len(s2n[:,0])):
                color = cmap(norm(param_values[plot_num].value))
                plt.scatter(bin_centers, s2n[plot_num,:], label= param_name + ": " + f"{param_values[plot_num].value:g}" + " (" + param_units_string + ")", color=color)
            plt.axhline(y=1, color='gray', linestyle='--')
            plt.axhline(y=5, color='gray', linestyle='-')
            # Annotate S/N = 1 and S/N = 5 on the plot
            plt.annotate('S/N = 1', xy=(6, 1), xytext=(-10, 5), textcoords='offset points',
                        ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
            plt.annotate('S/N = 5', xy=(6, 5), xytext=(-10, 5), textcoords='offset points',
                        ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
            #plt.yscale('log')
            plt.xlim([4, 18])
            plt.ylabel('S/N per wavelength bin')
            plt.xlabel('Wavelength (um)')

            ax = plt.gca()
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(param_name + " (" + param_units_string + ")")
            # Keep a concise axes title and add two figure-level columns
            plt.title(format_plot_title("S/N", self.config))
            fig3 = plt.gcf()
            fig3.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig3.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"1d_s2n_vs_wavelength_and_dark_current_per_wavelength_bin.png"
            plt.subplots_adjust(top=0.6,right=0.8)
            #plt.show()
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")


        return s2n