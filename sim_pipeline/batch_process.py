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
from datetime import datetime
from scipy import ndimage
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
import yaml




# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core import calculator, astrophysical, instrumental
from modules.utils import loader, validator
from modules.utils.helpers import create_sample_data, ensure_plot_title_context
from modules.data.units import UnitConverter
from modules.utils import helpers

# Module-level logger so it's available everywhere in this file
logger = logging.getLogger(__name__)

DARK_OUTPUT_PANELS = [
    ('output_3_dark', 'spectrum_output_3_dark'),
    ('output_4_dark', 'spectrum_output_4_dark'),
]


OUTPUT_FLUX_KEYS = [
    ('illumination_integrated_dark_3', 'output_3_dark', None),
    ('illumination_integrated_dark_4', 'output_4_dark', None),
    ('spectrum_output_1_bright', 'output_1_bright', (1, 2)),
    ('spectrum_output_2_bright', 'output_2_bright', (1, 2)),
    ('spectrum_output_3_dark', 'output_3_dark', (1, 2)),
    ('spectrum_output_4_dark', 'output_4_dark', (1, 2)),
    ('spectrum_chopped_dark_outputs', 'chopped_dark_outputs', (1, 2)),
]



def record_info_at_angle(
    prop_dict: dict,
    angle_deg: float,
    save_dir: str,
    *,
    star_source: str = 'star',
    planet_source: str = 'exoplanet_model_10pc',
    plot: bool = True,
) -> dict:
    """
    Snapshot prop_dict for one rotation angle and plot star vs planet spectra
    for the two dark outputs.
    """
    angle_results = {}
    for source_name, source_val in prop_dict.items():
        cubes = source_val['flux_cube_post_screen_post_aperture_ph_sec_um']
        source_snapshot = {'wavel': source_val['wavel']}

        #OUTPUT_FLUX_KEYS = list(source_val['flux_cube_post_screen_post_aperture_ph_sec_um'].keys())
        for result_key, output_name, sum_axes in OUTPUT_FLUX_KEYS:
            cube = cubes[output_name]
            source_snapshot[result_key] = (
                np.sum(cube) if sum_axes is None else np.sum(cube, axis=sum_axes)
            )

        angle_results[source_name] = source_snapshot

    logger.info(f"Recorded results for angle {angle_deg}.")


    if plot: 

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        wavel = angle_results[star_source]['wavel']
        flux_unit = angle_results[star_source]['spectrum_output_3_dark'].unit

        for ax, (output_label, spectrum_key) in zip(axes, DARK_OUTPUT_PANELS):
            ax.plot(wavel, angle_results[star_source][spectrum_key], label='star')
            ax.plot(wavel, angle_results[planet_source][spectrum_key], label='planet')
            ax.set_yscale('log')
            ax.set_xlim(4.0, 18.0)
            ax.set_ylim(1e-3, 1e9)
            ax.axhline(y=1e-1, color='gray', linestyle='--')
            ax.axhline(y=1e7, color='gray', linestyle='--')
            ax.set_title(output_label)
            ax.set_xlabel(f'Wavelength ({wavel.unit})')
            ax.set_ylabel(f'Flux ({flux_unit})')
            ax.legend()

        fig.suptitle(f'Stellar and planet flux vs wavelength at angle {angle_deg}')
        file_name_plot = (
            f"{save_dir}stellar_and_planet_flux_vs_wavelength_at_angle_{angle_deg}.png"
        )
        fig.savefig(file_name_plot)
        plt.close(fig)
        logger.info(
            f"Saved plot of stellar and planet flux vs wavelength at angle {angle_deg}: {file_name_plot}"
        )

    return angle_results


def _normalize_output_root(output_root: str) -> str:
    """Return an absolute output directory with a trailing separator."""
    normalized = str(Path(output_root).expanduser().resolve())
    return normalized if normalized.endswith(os.sep) else normalized + os.sep


def apply_output_root_override(config_path: str, output_root: Optional[str]) -> str:
    """
    Override the batch output root in a config file.

    The file is updated in place so downstream temporary config generation
    inherits the run-specific root directory.
    """
    if not output_root:
        return config_path

    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section("dirs"):
        config.add_section("dirs")

    normalized_root = _normalize_output_root(output_root)
    os.makedirs(normalized_root, exist_ok=True)
    config.set("dirs", "save_s2n_data_unique_dir", normalized_root)

    with open(config_path, "w") as f:
        config.write(f)

    logger.info(f"Using overridden output root: {normalized_root}")
    return config_path

def modify_config_file_sweep(config_path: str, qe: float, run_id: Optional[str] = None) -> str:
    """
    Create a modified configuration file with new n_int and output path values.
    
    Args:
        config_path: Path to the original configuration file
        
    Returns:
        Path to the temporary modified configuration file
    """
    # Load the original config
    config = configparser.ConfigParser()
    config.read(config_path)
    

    # Modify the values
    config.set('detector', 'quantum_efficiency', str(qe))
    #config.set('saving', 'save_s2n_data', output_path)
    
    # Create a temporary config file
    temp_config_dir = os.path.dirname(config_path) + '/parameter_sweeps/'
    qe_str = f"{qe:.2f}".replace('.', 'p') # for making better string (since it's a decimal)
    run_suffix = f"_{run_id}" if run_id else ""
    temp_config_path = temp_config_dir + os.path.basename(config_path).replace('.ini', f'_temp_qe{qe_str}{run_suffix}.ini')
    if not os.path.exists(temp_config_dir):
        os.makedirs(temp_config_dir, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    return temp_config_path


def modify_config_file_pl_system_params(
    config_path: str,
    base_filename: str,
    system_params: dict,
    lum_types: dict,
    run_id: Optional[str] = None,
) -> str:
    """
    Create a modified configuration file which takes a planet from a population model and overwrites planetary system parameters from a config file.
    
    Args:
        config_path: Path to the original configuration file, which will be modified here
        base_filename: string to distinguish individual planets (index number from df of planet population data)
        system_params: Optional[dict] = None: the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        Path to the temporary modified configuration file
    """

    if system_params is not None:

        # Load the original config
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Modify the values
        config.set('target', 'distance', str(system_params['Ds'])) # distance to the star (pc)
        config.set('target', 'rad_planet', str(system_params['Rp'])) # planet radius (Earth radii)
        config.set('target', 'pl_temp', str(system_params['Tp'])) # planet temp (K)
        config.set('target', 'rad_star', str(system_params['Rs'])) # stellar radius (solar radii)
        config.set('target', 't_star', str(system_params['Ts'])) # stellar temperature (K)
        config.set('target', 'z_exozodiacal', str(system_params['z'])) # stellar temperature (K)
        config.set('target', 'lambda_rel_lon_los', str(system_params['eclip_lon'])) # ecliptic latitude (rad)
        config.set('target', 'beta_lat_los', str(system_params['eclip_lat'])) # ecliptic longitude (rad)

        # this is a kludge to map stellar-type/luminosity in case only the stellar type is input
        config.set('target', 'L_star', str(lum_types[system_params['Stype'].lower()])) # stellar luminosity (L_sol) based on the type

        config.set('target', 'psg_spectrum_file_name', str(system_params['abs_file_name_psg_spectrum'])) # NASA PSG spectrum file name
        logging.info(f"NASA PSG spectrum file name: {system_params['abs_file_name_psg_spectrum']}")
        
        # for strings only
        config.set('target', 'Stype', str(system_params['Stype']))
        config.set('target', 'Nuniverse', str(system_params['Nuniverse']))
        config.set('target', 'Nstar', str(system_params['Nstar']))
        
        # Create a temporary config file, set up directory to contain stuff

        # Compose parts of the file name for readability
        nuniverse_part = f"Nuniverse_{config['target']['Nuniverse']}"
        nstar_part = f"Nstar_{config['target']['Nstar']}"
        dist_part = f"dist_{config['target']['distance']}"
        rp_part = f"Rp_{config['target']['rad_planet']}"
        rs_part = f"Rs_{config['target']['rad_star']}"
        ts_part = f"Ts_{config['target']['t_star']}"
        l_part = f"L_{config['target']['L_star']}"
        z_part = f"z_{config['target']['z_exozodiacal']}"
        eclip_lon_part = f"eclip_lon_{config['target']['lambda_rel_lon_los']}"
        eclip_lat_part = f"eclip_lat_{config['target']['beta_lat_los']}"
        stype_part = f"Stype_{config['target']['Stype']}"

        # use this string to 
        # 1. make a subdir (which will contain all the files for the different values of QE, n_int)
        # 2. name the files in the subdir as well
        file_basename_string = (
            f"temp_{base_filename}_"
            f"{nuniverse_part}_"
            f"{nstar_part}_"
            f"{dist_part}_"
            f"{rp_part}_"
            f"{rs_part}_"
            f"{ts_part}_"
            f"{l_part}_"
            f"{z_part}_"
            f"{eclip_lon_part}_"
            f"{eclip_lat_part}_"
            f"{stype_part}"
        )

        # make a long string of all the system parameters for FYI plots
        #ipdb.set_trace()
        #ensure_plot_title_context(config)

        # save all stuff (FYI plots, SNR results, etc.) in the subdir
        #ipdb.set_trace()
        config.set('dirs', 'save_s2n_data_unique_dir', config['dirs']['save_s2n_data_unique_dir'] + file_basename_string + '/')

        #qe_str = f"{qe:.2f}".replace('.', 'p') # for making better string (since it's a decimal)
        run_suffix = f"_{run_id}" if run_id else ""
        temp_config_path = str(config['dirs']['save_s2n_data_unique_dir']) + file_basename_string + f'{run_suffix}.ini'

        # Ensure the directory exists before writing the temporary config file
        temp_config_dir = os.path.dirname(temp_config_path)
        os.makedirs(temp_config_dir, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            config.write(f)
        logger.info(f"Created temporary config file for one planetary system: {temp_config_path}")

    else:
        # just return the original config path
        return config_path
    
    return temp_config_path


def run_single_calculation(config_path: str, 
                            base_filename: str,
                            sources_to_include: List[str], 
                          qe: float,
                          overwrite: bool = True, 
                          plot: bool = False, 
                          system_params: Optional[dict] = None, 
                          lum_types: Optional[dict] = None,
                          output_root: Optional[str] = None) -> bool:
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

    if True:
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

        # instantiate instrument effects
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
        angles_deg = np.linspace(0, 360, num=int(config['observation']['N_angles']), endpoint=False)  # e.g. step=1 or 5
        results = {}  # dict to hold results for each angle

        override_stellar_mask = bool(True)
        if override_stellar_mask:
            input("! ------ Inserting a manual stellar mask for ALL outputs. Press Enter if you're OK with this ------- !")

        for angle_deg in angles_deg:
            # Reset mutable pipeline state
            instrument_dep_terms.sources_astroph = copy.deepcopy(sources_astroph_pristine)
            instrument_dep_terms.prop_dict = {}

            # generate the transmission screens (one per output)
            logger.info("Generating transmission screens...")
            # no rotation yet
            transmission_screens = instrument_dep_terms.generate_instrument_transmission(override_stellar_mask=override_stellar_mask, plot=plot)

            # rotate the transmission screens
            transmission_screens_only_rot = ndimage.rotate(transmission_screens[0:4,:,:], angle_deg, axes=(1,2), reshape=False) # rotate the screens, but not the sky coordinates
            transmission_screens[0:4,:,:] = transmission_screens_only_rot

            # pass astrophysical scene through transmission screens (and chop)
            logger.info("Passing astrophysical flux through transmission screens...")

            instrument_dep_terms.pass_through_transmission_screens(
                fyi_angle = angle_deg,
                source_dict_pre_screen = astro_scene_perfect_no_screen, 
                transmission_screens = transmission_screens, 
                chop = True,
                plot=plot)

            # Pass through telescope aperture
            logger.info("Converting photons to photo-electrons...")
            instrument_dep_terms.pass_through_aperture(plot=plot)

            # record condensed information at this angle (to avoid mem leak)
            # see plot of chopped planet flux: instrument_dep_terms.prop_dict['exoplanet_model_10pc']['flux_cube_post_screen_post_aperture_ph_sec_um']['chopped_dark_outputs'][15,:,:].value
            results[angle_deg] = record_info_at_angle(
                prop_dict = instrument_dep_terms.prop_dict,
                angle_deg = angle_deg,
                save_dir = str(config['dirs']['save_s2n_data_unique_dir']),
                star_source = 'star',
                planet_source = 'exoplanet_model_10pc',
                plot=plot,
            )

            '''
            # on detector: convert photons to electrons
            instrument_dep_terms.photons_to_e()
            instrument_dep_terms.calculate_instrinsic_instrumental_noise()
            
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

        # calculate the S/N of the chopped dark outputs; need S_p and S_p_3; see Dannert+ 2022 Eqn. 20

        # build a pandas dataframe of the chopped signal (and keep wavelength)
        angles_sorted = sorted(results.keys())
        wavel = results[angles_sorted[0]]['exoplanet_model_10pc']['wavel']
        data_S_p = {} # modulated signal between dark outputs 3 and 4
        data_S_p_3 = {} # signal from dark output 3
        data_S_p_sqd = {} # squared
        data_S_p_3_sqd = {}

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

        # tack wavelength back on
        #df_S_p['wavel'] = wavel
        #df_S_p_3['wavel'] = wavel

        #'wavel': wavel.value
        df_S_p = pd.DataFrame(data_S_p)
        df_S_p_sqd = pd.DataFrame(data_S_p_sqd)
        df_S_p_3 = pd.DataFrame(data_S_p_3)
        df_S_p_3_sqd = pd.DataFrame(data_S_p_3_sqd)

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
        success = run_single_calculation(
            config_path=base_config_path,
            base_filename = base_filename,
            sources_to_include=sources_to_include,
            qe=qe,
            overwrite=overwrite,
            plot=plot,
            system_params=system_params, 
            lum_types=lum_types,
            output_root=output_root,
        )

        ipdb.set_trace()
                    
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
