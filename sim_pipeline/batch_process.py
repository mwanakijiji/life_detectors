#!/usr/bin/env python3
"""
Batch processing script for Life Detectors package.

This script allows you to run multiple calculations with different n_int values
and save the results as FITS files with different names.
"""

import os
from socket import IPV6_DONTFRAG
import sys
import configparser
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import ipdb
import pandas as pd
import numpy as np
import uuid
import copy
import h5py
import glob
from datetime import datetime
from scipy import ndimage
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
import yaml
from astropy.table import QTable

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core import calculator, astrophysical, instrumental
from modules.utils import loader, validator
from modules.utils.helpers import (
    _normalize_output_root,
    apply_output_root_override,
    create_sample_data,
    ensure_plot_title_context,
    modify_config_file_pl_system_params,
    modify_config_file_sweep,
    record_info_at_angle,
)
from modules.data.units import UnitConverter
from modules.utils import helpers



# Module-level logger so it's available everywhere in this file
logger = logging.getLogger(__name__)

def calculate_s2n_post_rotation(read_dir):
    """
    Calculate the S/N of the chopped dark outputs.

    Args:
        read_dir: dir containing the HDF5 files

    Returns:
        S_p_sqd_phi_mean: mean of the squared signal of the chopped dark outputs
        S_p_3_sqd_phi_mean: mean of the squared signal of the chopped dark output 3
        SNR_lambda_array: SNR for each wavelength bin
        SNR_tot: total SNR
    """
    

    hdf5_files = glob.glob(os.path.join(read_dir, '*.hdf5'))
    by_dc = {}

    ipdb.set_trace()
    # read in one HDF5 file per angle
    for hdf5_file in hdf5_files:
        angle = float(Path(hdf5_file).stem.removeprefix("angle_"))
        
        with h5py.File(hdf5_file, "r") as f:
            # for all DC values
            for dc_rate in f.keys():
                # for all outputs
                for ch in f[dc_rate].keys():
                    if ch.endswith(".__table_column_meta__"):
                        continue
                    tbl = QTable.read(hdf5_file, path=f"{dc_rate}/{ch}")

                # calculate the S/N of the chopped dark outputs
                # need S_p and S_p_3; see Dannert+ 2022 Eqn. 20
                chopped = QTable.read(hdf5_file, path=f"{dc_rate}/chopped")
                S_p = chopped["chopped_astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"]
                out3 = QTable.read(hdf5_file, path=f"{dc_rate}/output_3_dark")
                S_p_3 = out3["astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"]
                wavel = chopped["wavel_bin_center"].value
                wavel_bin_edges = chopped.meta["wavel_bin_edges"]
                

                slot = by_dc.setdefault(dc_rate, {"wavel": wavel, "S_p": {}, "S_p_3": {}})
                slot["S_p"][angle] = S_p
                slot["S_p_3"][angle] = S_p_3
                slot["wavel_bin_width"] = chopped["wavel_bin_width"] ## ## TODO: use the proper bin edges

    for dc_rate, slot in by_dc.items():
        angles = sorted(slot["S_p"].keys())
        # (n_bins, n_angles) — use .value if columns are Quantity

        cols_S_p = []
        cols_S_p_3 = []
        for a in angles:
            cols_S_p.append(np.asarray(slot["S_p"][a])) # units are lost here, but will be restored later
            cols_S_p_3.append(np.asarray(slot["S_p_3"][a])) # units are lost here, but will be restored later
        S_p_arr = np.column_stack(cols_S_p)
        S_p_sqd_arr = np.power(S_p_arr, 2)
        S_p_3_arr = np.column_stack(cols_S_p_3)

        S_p_sqd_arr_mean = S_p_sqd_arr.mean(axis=1) * slot["S_p"][a].unit # Dannert+ 2022 Eqn. (19)
        S_p_3_sqd_arr_mean = S_p_3_arr.mean(axis=1) * slot["S_p_3"][a].unit # Dannert+ 2022 Eqn. (19)
        S_sym_3 = 0 # placeholder ## ## TODO: implement this

        SNR_lambda_array = []
        for wavel_bin_num in range(len(slot["wavel_bin_width"])): 
            #wavel_start = slot["wavel_bin_edges"][wavel_bin_num].value
            #wavel_stop = slot["wavel_bin_edges"][wavel_bin_num + 1].value

            

            d_wavel = slot["wavel_bin_width"][wavel_bin_num]

            # Dannert+ 2022 Eqn. (19)-(20): sqrt of angle-averaged squared signals per bin
            S_p_rms_phi = np.sqrt(S_p_sqd_arr_mean[wavel_bin_num])
            S_p_3_rms_phi = np.sqrt(S_p_3_sqd_arr_mean[wavel_bin_num])

            numerator_ = S_p_rms_phi * d_wavel
            noise_term = S_sym_3 + S_p_3_rms_phi ## ## CONTINUE HERE; INSERT INSTRUMENTAL TERMS
            denominator_ = np.sqrt(2 * noise_term * d_wavel)

            SNR_lambda = numerator_ / denominator_
            SNR_lambda_array.append(SNR_lambda.value)

        SNR_tot = np.sqrt(np.sum(np.power(SNR_lambda_array, 2)))
        print(f'SNR_tot for DC {dc_rate}: {SNR_tot}')

        # plot SNR 
        plt.clf()
        plt.stairs(SNR_lambda_array, edges=wavel_bin_edges.value)
        plt.xlim([4, 18.5])
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel(f'Wavelength ({slot["wavel_bin_width"][wavel_bin_num].unit})')
        plt.ylabel('SNR')
        plt.title(f'SNR for DC {dc_rate}\nSNR_tot: {SNR_tot}')
        plt.show()

        

        #numerator_ = np.sqrt(S_p_sqd_arr_mean) * slot["wavel_bin_width"] # effectively an integral over in Dannert+ 2022 Eqn. (20)
        ipdb.set_trace()
    




    ipdb.set_trace()
    # now find S/N by using all observing angles

    ipdb.set_trace()
    ## ## CONTINUE HERE: calculate the S/N of the chopped dark outputs

    '''
    for angle in angles_sorted: # note pandas doesn't like units
        # note units (see Dannert+ 2022 Eqn (17)); 
        # results[angle]['exoplanet_model_10pc']['spectrum_chopped_dark_outputs'] is already the same as the integral of ( transmission * flux * collecting_area ) over solid angle of the FOV. 
        # ... we still need to multiply by 
        #      t: integration time of one readout (corresponding to one angle)
        #      eta: detector efficiency, in practice this is throughput * QE
        # units are
        # [t] * [eta] * [ results[angle]['exoplanet_model_10pc']['spectrum_chopped_dark_outputs'] ] = s * 1 * ph / (micron s) = ph / micron
        # 
        t_times_eta = float(config['observation']['t_int_frame']) * float(config['detector']['quantum_efficiency'])
        data_S_p[f'angle_{int(angle)}_deg'] = results[angle]['exoplanet_model_10pc']['spectrum_chopped_dark_outputs'].value * t_times_eta
        data_S_p_3[f'angle_{int(angle)}_deg'] = results[angle]['exoplanet_model_10pc']['spectrum_output_3_dark'].value * t_times_eta
        data_S_p_sqd[f'angle_{int(angle)}_deg'] = np.power(results[angle]['exoplanet_model_10pc']['spectrum_chopped_dark_outputs'].value * t_times_eta, 2)
        data_S_p_3_sqd[f'angle_{int(angle)}_deg'] = np.power(results[angle]['exoplanet_model_10pc']['spectrum_output_3_dark'].value * t_times_eta, 2)
    '''

    # tack wavelength back on
    #df_S_p['wavel'] = wavel
    #df_S_p_3['wavel'] = wavel

    ipdb.set_trace()

    # find the quadratic mean of the modulated signal S_p (Dannert+ 2022 Eqn. (19))

    #'wavel': wavel.value
    df_S_p = pd.DataFrame(S_p, columns=["S_p"])
    df_S_p_sqd = pd.DataFrame(np.power(S_p, 2), columns=["S_p_sqd"])
    df_S_p_3 = pd.DataFrame(S_p_3, columns=["S_p_3"])
    df_S_p_3_sqd = pd.DataFrame(np.power(S_p_3, 2), columns=["S_p_3_sqd"])

    # now insert mean column, averaging over angles; Dannert+ 2022 Eqn. (19)
    df_S_p_sqd['avg_S_p_sqd_phi'] = df_S_p_sqd.mean(axis=1)
    #df_S_p_3 = pd.DataFrame(data_S_p_3)
    df_S_p_3_sqd['avg_S_p_3_sqd_phi'] = df_S_p_3_sqd.mean(axis=1)
    # now reconstitute units, which pandas does not like
    ipdb.set_trace()
    unit = results[angles_sorted[0]]['exoplanet_model_10pc']['spectrum_output_3_dark'].unit ** 2 # note the squared
    S_p_sqd_phi_mean = df_S_p_sqd['avg_S_p_sqd_phi'].values * unit
    S_p_3_sqd_phi_mean = df_S_p_3_sqd['avg_S_p_3_sqd_phi'].values * unit
    
    # SNR_lambda for each wavelength bin; see Dannert+ 2022 Eqn. (20)
    # note integration is over the wavelength bin only; not the entire spectrum
    ## ## TODO: make sure wavelength bin sizes are consistent with calcs further upstream
    SNR_lambda_array = []

    # find SNR for each wavelength bin for this rotation angle
    for wavel_bin_num in range(len(wavel)-1):

        print(f"wavel_bin_num: {wavel_bin_num}")
        wavel_start = wavel[wavel_bin_num].value
        wavel_stop = wavel[wavel_bin_num + 1].value
        d_wavel = wavel_stop - wavel_start

        wavel_range = np.array([wavel_start, wavel_stop]) ## ## TODO: make this denser later? np.linspace(wavel_start, wavel_stop, 100)

        # use simple rectangles to integrate
        term_1 = np.sqrt(S_p_sqd_phi_mean[wavel_bin_num]) * d_wavel

        term_2 = np.sqrt(S_p_3_sqd_phi_mean[wavel_bin_num]) * d_wavel

        term_3 = np.sqrt(2. * term_2)

        #term_1 = np.trapz(y=np.sqrt(S_p_sqd_phi_mean[wavel_bin_num]), dx=d_wavel)
        #term_2 = np.trapz(y=np.sqrt(S_p_3_sqd_phi_mean[wavel_bin_num]), dx=d_wavel)
        #term_3 = np.sqrt(2. * term_2)
    

        # SNR for that wavelength bin
        SNR_lambda = term_1 / term_3 ## ## TODO: update to include S_sym term
        SNR_lambda_array.append(SNR_lambda.value)

    SNR_lambda_array = u.Quantity(SNR_lambda_array)
    SNR_tot = np.sqrt( 
                    np.sum(np.power(SNR_lambda_array,2)) 
                    )

    ipdb.set_trace()
    #abcissa_values = np.array(abcissa_values).flatten()
    #ordinate_values = np.array(ordinate_values).flatten()
    plt.clf()
    with quantity_support():
        plt.plot(abcissa_values, ordinate_values, linestyle='--', marker='o',
                label='total planet signal (one dark output)')
    plt.legend()
    output_string = 'chopped_dark_outputs'
    plt.title('Planet flux as function of rotation angle, ' + output_string + ' output')
    plt.xlabel(f'Angle (deg)')
    ipdb.set_trace()
    plt.ylabel(f"Total flux ({instrument_dep_terms.prop_dict['exoplanet_model_10pc']['flux_cube_post_screen_post_aperture_ph_sec_um'][output_string].unit})")
    file_name_plot = str(config['dirs']['save_s2n_data_unique_dir']) + f"planet_flux_as_function_of_rotation_angle.png"
    plt.savefig(file_name_plot)
    logging.info(f"Saved plot of planet flux as function of rotation angle to {file_name_plot}")
    plt.show()

    return



def run_single_calculation(
    config_path: str,
    base_filename: str,
    sources_to_include: List[str],
    qe: float,
    override_stellar_mask: bool = True,
    overwrite: bool = True,
    plot: bool = False,
    system_params: Optional[dict] = None,
    lum_types: Optional[dict] = None,
    output_root: Optional[str] = None,
) -> bool:
    """
    Run a single calculation with specified parameters.
    
    Args:
        config_path: Path to the configuration file
        base_filename: string to distinguish individual planets (index number from df of planet population data)
        sources_to_include: List of sources to include in calculation
        n_int: Number of integrations
        qe: quantum efficiency
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional[dict] = None: the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        output_path: Path of the output FITS file
    """

    # get config things sorted out
    base_config = configparser.ConfigParser()
    base_config.read(config_path)
    generate_sims = loader.config_getboolean(base_config, "tasks", "generate_sims")
    calculate_s2n = loader.config_getboolean(base_config, "tasks", "calculate_s2n_post_rotation")

    # Build a per-run token so parallel jobs do not clobber temp files.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"

    # Create temporary config file with modified values: use new n_int, QE values
    ## TO DO: CHECK THIS FOR CASE WHEN PLANET POPULATION IS NOT BEING DONE; DOES THE ABSENCE OF A BASE_FILENAME CAUSE PROBLEMS?
    temp_config_path_nint_qe = modify_config_file_sweep(config_path, qe, run_id=run_id)
    temp_config_path_nint_qe = apply_output_root_override(temp_config_path_nint_qe, output_root)
    # modify again, for a given planetary system 
    if system_params is not None:
        temp_config_path = modify_config_file_pl_system_params(
            config_path=temp_config_path_nint_qe,
            base_filename=base_filename,
            system_params=system_params,
            lum_types=lum_types,
            run_id=run_id,
        )
    else:
        temp_config_path = temp_config_path_nint_qe

    # First log entry: which config file we're using (original and temp)
    logger.info(f"--------------------------------")
    logger.info(f"Config path (base): {config_path}")
    logger.info(f"Config path (temporary one for this case, for batch processing): {temp_config_path}")

    # Load config in two forms:
    # - dict (for validation + directory creation)
    # - ConfigParser (for downstream code that expects .has_section/.options)
    # this is necessary for vestigial reasons while using only one load_config() function
    config_dict = loader.load_config(config_file=temp_config_path, makedirs=True)
    validator.validate_config(config_dict)
    config = configparser.ConfigParser()
    config.read(temp_config_path)
    ensure_plot_title_context(config)

    # Make the overwriteable scratch FITS file unique per run.
    if config.has_section("saving") and config.has_option("saving", "save_s2n_data_temp"):
        temp_fits_path = config.get("saving", "save_s2n_data_temp")
        temp_root, temp_ext = os.path.splitext(temp_fits_path)
        config.set("saving", "save_s2n_data_temp", f"{temp_root}_{run_id}{temp_ext}")

    # should we simulate the observations and generate the HDF5 files?
    if generate_sims:

        logger.info("Generating simulations.")

        # S/N results will be written to this file
        # Insert QE and n_int into the filename before .fits
        fname_base = temp_config_path.replace('.ini', '')
        output_fits_file_abs_path = f"{fname_base}_qe_{qe:.4f}_{run_id}.fits"

        # Useful debug info about the loaded config: list actual INI-style sections and key-value pairs
        try:
            import configparser as _cp  # local import to avoid circulars
        except ImportError:
            _cp = None


        logger.info("------- Temp config contents -------")

        # log stuff from the config file
        if _cp is not None and isinstance(config, _cp.ConfigParser):
            # Config is a ConfigParser: iterate its sections and items
            for section_name in config.sections():
                logger.info(f"[{section_name}]")
                for key, value in config[section_name].items():
                    logger.info(f"  {key} = {value}")
        elif isinstance(config, dict):
            # Config is a dict-of-dicts (as returned by some loaders)
            logger.info(f"Top-level keys: {list(config.keys())}")
            for section_name, section_dict in config.items():
                logger.info(f"[{section_name}]")
                if isinstance(section_dict, dict):
                    for key, value in section_dict.items():
                        logger.info(f"  {key} = {value}")
                else:
                    logger.info(f"  (section value) {section_dict}")
        else:
            # Fallback: just log the raw object
            logger.info(f"(Unrecognized config type {type(config)}; raw repr follows)")
            logger.info(repr(config))
        logger.info(f"Running calculation with output file name = {output_fits_file_abs_path}")
        
        # Generate sample spectral data
        logger.info("Creating sample spectral data...")
        create_sample_data(config, overwrite=overwrite, plot=plot, read_sample_file=False)
        
        # instantiate astrophysical flux calculator
        logger.info("Calculating astrophysical flux...")
        astrophysical_sources = astrophysical.AstrophysicalSources(config, unit_converter=UnitConverter())
        
        # Calculate incident flux for each source, and add on 2D positions as projected on sky
        sources_astroph = {}
        for source_name in sources_to_include:
            if source_name in ["star", "star_psg", "exoplanet_bb", "exoplanet_bb_psg", "exoplanet_model_10pc", "exoplanet_psg", "exozodiacal", "exozodiacal_psg", "zodiacal"]:
                sources_astroph[source_name] = astrophysical_sources.calculate_incident_flux(
                    source_name=source_name, plot=plot, system_params=system_params
                )
        
        # put all the objects into the scene
        logger.info("Generating on-sky scene...")
        astro_scene_perfect_no_screen = astrophysical_sources.generate_onsky_scene(incident_dict=sources_astroph, plot=plot)

        # instantiate instrument effects, OutputChannel objects
        logger.info("Passing astrophysical flux through telescope aperture...")
        instrument_dep_terms = instrumental.InstrumentDepTerms(
            config, 
            unit_converter=UnitConverter(),
            sources_astroph=sources_astroph,
            sources_to_include=sources_to_include
        )

        # Pristine copy after incident flux, before any screen/aperture steps
        sources_astroph_pristine = copy.deepcopy(instrument_dep_terms.sources_astroph)

        # make 1 full rotation
        angles_deg = np.linspace(0, 360, num=int(config['observation']['N_angles']), endpoint=False)
        results = {}  # dict to hold results for each angle

        for angle_deg in angles_deg:
            # Reset mutable pipeline state
            instrument_dep_terms.sources_astroph = copy.deepcopy(sources_astroph_pristine)
            instrument_dep_terms.prop_dict = {} # dict to hold astrophysical terms as propagated through the instrument

            # reset the OutputChannel objects for this new angle
            for ch in instrument_dep_terms.output_channels.values():
                ch.angle_deg = angle_deg
                ch.instrum_noise.clear()
                ch.astroph_signal.clear()
                ch.tables_by_dark_current.clear()
                if hasattr(ch, 'tables_by_dark_current_orig'):
                    ch.tables_by_dark_current_orig.clear()
            instrument_dep_terms.post_chop_tables_by_dark_current = {}

            # generate the transmission screens (one per output)
            logger.info("Generating transmission screens...")
            # no rotation yet
            transmission_screens = instrument_dep_terms.generate_instrument_transmission(override_stellar_mask=override_stellar_mask, plot=plot)

            # rotate the transmission screens
            transmission_screens_only_rot = ndimage.rotate(transmission_screens[0:4,:,:], angle_deg, axes=(1,2), reshape=False) # rotate the screens, but not the sky coordinates
            transmission_screens[0:4,:,:] = transmission_screens_only_rot # reasssign the rotated screens to the original transmission_screens

            # pass astrophysical scene through transmission screens
            logger.info("Passing astrophysical flux through transmission screens ...")
            instrument_dep_terms.pass_through_transmission_screens(
                fyi_angle = angle_deg,
                source_dict_pre_screen = astro_scene_perfect_no_screen, 
                transmission_screens = transmission_screens, 
                plot=plot)

            # Pass through telescope aperture
            logger.info("Converting photons to photo-electrons ...")
            instrument_dep_terms.pass_through_aperture(plot=plot)

            # set instrumental noise terms and update the OutputChannel objects
            logger.info("Assigning intrinsic instrumental noise ...")
            instrument_dep_terms.calculate_instrinsic_instrumental_noise()

            # disperse astrophysical signals on the detector (i.e., update the OutputChannel objects; note that input astrophysical signals should still be photons, not electrons)
            logger.info("Dispersing signals on channel detectors ...")
            instrument_dep_terms.disperse_astro_signals_on_detector(plot=plot)

            # pack the signals together (and convert photons to electrons)
            instrument_dep_terms.combine_astro_and_instrum_signals()


            # chop the signal between dark outputs
            instrument_dep_terms.chop_signal(plot=plot)

            # write condensed information at this transmission screen angle (to avoid mem leak)
            # see plot of chopped planet flux: instrument_dep_terms.prop_dict['exoplanet_model_10pc']['flux_cube_post_screen_post_aperture_ph_sec_um']['chopped_dark_outputs'][15,:,:].value
            record_info_at_angle(
                angle_deg=angle_deg,
                output_channels=instrument_dep_terms.output_channels,
                post_chop_tables_by_dark_current=instrument_dep_terms.post_chop_tables_by_dark_current,
                save_dir=str(config['dirs']['save_s2n_data_unique_dir']),
                plot=plot,
            )

    else:
        logger.info("Skipping simulation and calculating S/N from HDF5 files...")

        '''

        
        # Calculate S/N
        logger.info("Calculating signal-to-noise ratio...")
        noise_calc = calculator.NoiseCalculator(
            config,
            sources_all=instrument_dep_terms, 
            sources_to_include=sources_to_include
        )
        
        # This will automatically save the FITS file 
        s2n = noise_calc.s2n_e(file_name_fits_unique = output_fits_file_abs_path)
        
        #logger.info(f"Successfully completed calculation with n_int={n_int}")
        logger.info(f"Results saved to: {output_fits_file_abs_path}")
        ipdb.set_trace()
        '''

    if calculate_s2n:
        logger.info("Calculating S/N from HDF5 files.")
        calculate_s2n_post_rotation(config['dirs']['save_s2n_data_unique_dir'])

    else:
        logger.info("Not calculating S/N from HDF5 files.")
        
    return True


def batch_qe_nint_process(base_config_path: str, 
                qe_values: List[float],
                  sources_to_include: List[str],
                  base_filename: str = "s2n", 
                  overwrite: bool = True, 
                  plot: bool = False, 
                system_params: Optional[dict] = None, 
                lum_types: Optional[dict] = None,
                output_root: Optional[str] = None) -> List[Tuple[int, str, bool]]:
    """
    Run batch processing with multiple QE and n_int values.
    
    Args:
        base_config_path: Path to the base configuration file
        n_int_values: List of n_int values to process
        qe_values: List of qe values to process
        sources_to_include: List of sources to include in calculations
        base_filename: Base filename for output files (without extension)
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional dictionary of the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        List of tuples (n_int, output_path, success)
    """
    # Create output directory if it doesn't exist
    #Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []

    # initialize for checking if all calculations for the QE, n_int run were successful
    success_all = True

    # make a manual mask over the star?
    override_stellar_mask = bool(True)
    if override_stellar_mask:
        try:
            raw_response = input(
                "! ------ Inserting a manual stellar mask for ALL outputs. Proceed? (n -> discard manual mask) ------- ! "
            )
        except EOFError:
            raw_response = "y"
        response = str(raw_response).strip().lower()
        override_stellar_mask = response in {"y", "yes", ""}
    
    for qe in qe_values:
        # Create output filename
        qe_pct = int(round(qe * 100))  # e.g. 0.87 -> 87
        #output_filename = f"{base_filename}_n{n_int:08d}_qe{qe_pct:03d}.fits"
        #output_path = os.path.join(output_dir, output_filename)
        
        logging.info(f"--------------------------------")
        logging.info(f"--------------------------------")
        logging.info(f"Processing single calculation:")
        logging.info(f"Parameter qe = {qe}")

        # run single calculation, for range of rotation angles
        # this could involve either or both of:
        # 1. generating simulations and writing out the HDF5 files
        # 2. calculating the S/N from the HDF5 files
        success = run_single_calculation(
            config_path=base_config_path,
            base_filename = base_filename,
            sources_to_include=sources_to_include,
            qe=qe,
            override_stellar_mask=override_stellar_mask,
            overwrite=overwrite,
            plot=plot,
            system_params=system_params, 
            lum_types=lum_types,
            output_root=output_root,
        )
                    
        if success:
            logging.info(f"  ✓ Success for the following planetary system parameters:")
            logging.info(system_params)
        else:
            logging.info(f"  ✗ Failed for qe={qe}")
        success_all = success_all and bool(success) # was this calculation successful too?
    
    return success_all

'''
def example_custom_sources():
    """Example 4: Batch processing with different source combinations."""
    print("\nExample 4: Different source combinations")
    print("-" * 40)
    
    config_path = "modules/config/demo_config.ini"
    n_int_values = [3000, 6000]
    
    # Different source combinations
    source_combinations = [
        (["star", "exoplanet_model_10pc"], "planet_only"),
        (["star", "exozodiacal", "zodiacal"], "background_only"),
        (["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"], "all_sources")
    ]
    
    all_results = []
    
    for sources, name_suffix in source_combinations:
        print(f"\nProcessing sources: {sources}")
        output_dir = f"source_combinations/{name_suffix}"
        
        results = batch_qe_nint_process(
            config_path=config_path,
            n_int_values=n_int_values,
            output_dir=output_dir,
            sources_to_include=sources,
            base_filename=f"s2n_{name_suffix}",
            overwrite=True,
            plot=True
        )
        
        all_results.extend(results)
    
    return all_results
'''
'''
def example_simple_batch():
    """Example 1: Simple batch processing with a few n_int values."""
    print("Example 1: Simple batch processing")
    print("-" * 40)
    
    config_path = "modules/config/demo_config.ini"
    n_int_values = [1000, 3000, 6000, 9000]
    output_dir = "batch_output"
    sources = ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]
    
    results = batch_qe_nint_process(
        config_path=config_path,
        n_int_values=n_int_values,
        output_dir=output_dir,
        sources_to_include=sources,
        base_filename="s2n_simple",
        overwrite=True,
        plot=True
    )
    
    print(f"Processed {len(results)} calculations")
    return results
'''
'''
def example_single_calculation():
    """Example 2: Run a single calculation with custom parameters."""
    print("\nExample 2: Single calculation")
    print("-" * 40)
    
    config_path = "modules/config/demo_config.ini"
    n_int = 5000
    output_path = "single_output/s2n_n5000.fits"
    sources = ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]
    
    # Create output directory
    os.makedirs("single_output", exist_ok=True)
    
    success = run_single_calculation(
        config_path=config_path,
        sources_to_include=sources,
        n_int=n_int,
        output_path=output_path,
        overwrite=True,
        plot=True
    )
    
    if success:
        print(f"✓ Successfully created: {output_path}")
    else:
        print(f"✗ Failed to create: {output_path}")
    
    return success
'''
''' # command line option
    print("\nTo run the batch processor from command line:")
    print("python batch_qe_nint_process.py --config modules/config/demo_config.ini \\")
    print("                       --n-int 1000 3000 6000 9000 \\")
    print("                       --output-dir my_batch_output \\")
    print("                       --sources star exoplanet_model_10pc exozodiacal zodiacal")
'''

def parameter_sweep(
    config_single_obs_path: str,
    config_sweep_path: str,
    config_planet_population_path: str,
    output_root: Optional[str] = None,
) -> None:
    """
    Run a QE/n_int parameter sweep for one system or a planet population.
    """
    print("\nExample 3: Parameter sweep")
    print("-" * 40)

    # load in all config files once, and don't load them again
    config_single_obs = loader.load_config(config_file=config_single_obs_path)
    config_sweep = loader.load_config(config_file=config_sweep_path)
    config_planet_population = loader.load_config(config_file=config_planet_population_path)
    lum_types = None

    if output_root:
        logging.info(f"Top-level batch output root override: {_normalize_output_root(output_root)}")

    aperture_array_config_path = config_single_obs["telescope"]["aperture_array_config_file_name"]
    with open(aperture_array_config_path, 'r') as file:
        aperture_array_definition = yaml.safe_load(file)
    n_apertures = len(aperture_array_definition['apertures'])
    logging.info(f"Number of apertures: {n_apertures}")

    # calculate the total collecting area of the telescope
    collecting_area = helpers.compute_collecting_area_m2(config_single_obs) * u.m**2
    logging.info(f"Total collecting area of telescope: {collecting_area}")

    # does the user want to look at a single planet or a planet population?
    systems_2_look_at = config_single_obs["system_options"]["systems_2_look_at"]
    if systems_2_look_at == "planet_population":
        logging.info("Applying parameter sweep to an entire planet population")
    elif systems_2_look_at == "single_system":
        logging.info("Applying parameter sweep to a single system")
    else:
        logging.error(f"Invalid system option: {systems_2_look_at}")
        return

    # if applying a parameter sweep to every planet in a population
    if systems_2_look_at == "planet_population":
        logging.info("Applying parameter sweep to an entire planet population")

        file_name_planet_population = config_planet_population["file_name_planet_population"]["file_name"]
        logging.info(f"Reading planet population from {file_name_planet_population}")
        lum_types = config_planet_population["lum_type"]  # map luminosities with stellar types
        df_planet_population = pd.read_csv(file_name_planet_population, skiprows=1, sep=r"\s+")
        df_planet_population = helpers.merge_psg_spectra_to_planet_population(
            df_planet_population, config_planet_population
        )

        cols_to_plot = ["Rp", "Porb", "Mp", "z", "Tp", "ap"]
        fyi_plot_path = config_planet_population["file_name_planet_population"]["fyi_plot_name"]
        helpers.plot_planet_population_sample(df_planet_population, cols_to_plot, fyi_plot_path)
        logging.info(f"FYI plot of planet population saved to {fyi_plot_path}")

    elif systems_2_look_at == "single_system":
        logging.info("Applying parameter sweep to a single planetary system")
        df_planet_population = [None]  # wrap in list for length 1

    else:
        logging.error(f"Invalid system option (must be 'planet_population' or 'single_system'): {systems_2_look_at}")
        return

    # parameter sweep
    obs = config_sweep["observation"]
    qe_values = helpers.get_sweep_range(obs, "qe")

    # get the astrophysical sources to include from the config file
    sources_to_include = [
        source
        for source, include in config_single_obs["astrophysical_sources_to_use"].items()
        if include in [True, "True", "true", 1, "1"]
    ]

    # loop over all the planets (note there can be multiple planets in a system)
    for pl_num in range(len(df_planet_population)):
        if isinstance(df_planet_population, pd.DataFrame):
            system_params = df_planet_population.iloc[pl_num]
            logging.info(f"Processing system {pl_num} with parameters: {system_params}")
            base_filename = f"s2n_sweep_planet_index_{pl_num:07d}"
        else:
            system_params = None
            logging.info("No planet population; doing parameter sweep for a single system")
            base_filename = "s2n_sweep"

        # wraps around calculations for a single set of detector parameters
        success_all = batch_qe_nint_process(
            base_config_path=config_single_obs_path,
            qe_values=qe_values,
            sources_to_include=sources_to_include,
            base_filename=base_filename,
            overwrite=True,
            plot=True,
            system_params=system_params,
            lum_types=lum_types,
            output_root=output_root,
        )
        logging.info(f"Parameter sweep completed: all calculations successful = {success_all}")
