#!/usr/bin/env python3
"""
Batch processing script for Life Detectors package.

This script allows you to run multiple calculations with different n_int values
and save the results as FITS files with different names.
"""

import os
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
import time
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
from modules.core.calculator import calculate_s2n_post_rotation
from modules.utils import loader, validator
from modules.utils.helpers import (
    _normalize_output_root,
    apply_output_root_override,
    create_sample_data,
    ensure_plot_title_context,
    modify_config_file_pl_system_params,
    modify_config_file_sweep,
    record_info_at_angle_and_qe,
)
from modules.data.units import UnitConverter
from modules.utils import helpers



# Module-level logger so it's available everywhere in this file
logger = logging.getLogger(__name__)


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

        # kludge for now
        # laptop
        dir_temp_hdf5_files = '/Users/eckhartspalding/Documents/git.repos/life_detectors/hdf5_testing/temp_s2n_sweep_planet_index_0000000_Nuniverse_1_Nstar_1_dist_10_Rp_1_Rs_1_Ts_5778_L_1.0_z_3_eclip_lon_135_eclip_lat_45_Stype_G/'
        # node
        #dir_temp_hdf5_files = '/home/eckhart/Documents/git.repos/life_detectors/hdf5_testing/temp_s2n_sweep_planet_index_0000000_Nuniverse_1_Nstar_1_dist_10_Rp_1_Rs_1_Ts_5778_L_1.0_z_3_eclip_lon_135_eclip_lat_45_Stype_G/'
        hdf5_files = glob.glob(os.path.join(dir_temp_hdf5_files, '*.hdf5'))
        # Ask the user if they want to delete the files in this directory.
        response = input(f"Found directory containing temp HDF5 files:\n    {dir_temp_hdf5_files}\nFound {len(hdf5_files)} HDF5 files in this directory.\nDo you want to delete HDF5 files in this directory? [y/N]: ").strip().lower()
        if response == "y" or response == "yes":
            
            for file_path in hdf5_files:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted: {file_path}")
                except Exception as ex:
                    logger.warning(f"Failed to delete {file_path}: {ex}")
            logger.info("All temp HDF5 files deleted.")
        else:
            logger.info("Keeping temp HDF5 files.")
 

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
        astro_scene_perfect_no_screen = astrophysical_sources.generate_onsky_scene(
                                                                    incident_dict=sources_astroph, 
                                                                    plot=plot)

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
            transmission_screens = instrument_dep_terms.generate_instrument_transmission(
                override_stellar_mask = override_stellar_mask, 
                angle_deg = angle_deg,
                plot = plot
                )
            



            
            ###### START DEBUGGING (pre-transmission-screen)
            # Plot each of the transmission screens
            if plot:
                fig_ts, axs_ts = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
                transmission_screen_order = [
                    "output_1_bright",
                    "output_2_bright",
                    "output_3_dark",
                    "output_4_dark",
                ]
                for i, ch_name in enumerate(transmission_screen_order):
                    im = axs_ts[i].imshow(transmission_screens[i, :, :], origin='lower', aspect='auto')
                    axs_ts[i].set_title(f"{ch_name}")
                    plt.colorbar(im, ax=axs_ts[i], fraction=0.046, pad=0.04)
                fig_ts.suptitle(f"Transmission screens (angle {angle_deg:g}°)")
                save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                file_name_plot = f"{save_dir}transmission_screens_angle_{angle_deg:g}.png"
                #fig_ts.savefig(file_name_plot)
                plt.show()
                logger.info(f"Plotted transmission screens for angle {angle_deg:g}: {file_name_plot}")
    
            # FYI: star flux per output from scene × rotated transmission screen (before pass_through)
            plt.clf()
            if plot and "star" in instrument_dep_terms.sources_astroph:
                star_cube = astro_scene_perfect_no_screen.get("star")
                ipdb.set_trace()
                if star_cube is not None:
                    star_src = instrument_dep_terms.sources_astroph["star"]
                    wavel_vals = (
                        star_src["wavel"].value
                        if hasattr(star_src["wavel"], "value")
                        else star_src["wavel"]
                    )
                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    transmission_screen_order = [
                        "output_1_bright",
                        "output_2_bright",
                        "output_3_dark",
                        "output_4_dark",
                    ]
                    ipdb.set_trace()
                    for i, ch_name in enumerate(transmission_screen_order):
                        screened = star_cube * transmission_screens[i, :, :]
                        ipdb.set_trace()
                        flux_1d = np.sum(screened, axis=(1, 2))
                        flux_vals = flux_1d.value if hasattr(flux_1d, "value") else flux_1d
                        ax.plot(wavel_vals, flux_vals, label=ch_name)
                    pre_screen = star_src["pre_screen_astro_flux_ph_sec_m2_um"]
                    pre_vals = pre_screen.value if hasattr(pre_screen, "value") else pre_screen
                    ax.plot(
                        wavel_vals,
                        pre_vals,
                        label="pre-screen (incident 1D)",
                        linestyle="--",
                        color="0.5",
                    )
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ph/s/m^2/um, integrated over field)")
                    ax.set_title(
                        f"Star flux per output (scene × screen, pre-pass_through), angle {angle_deg:g}°"
                    )
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_all_outputs_pre_transmission_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    #fig.savefig(file_name_plot)
                    plt.show()
                    logger.info(f"Saved pre-transmission star flux per output: {file_name_plot}")
            ###### END DEBUGGING (pre-transmission-screen)

            # pass astrophysical scene through transmission screens
            logger.info("Passing astrophysical flux through transmission screens ...")
            instrument_dep_terms.pass_through_transmission_screens(
                fyi_angle = angle_deg,
                source_dict_pre_screen = astro_scene_perfect_no_screen, 
                transmission_screens = transmission_screens, 
                plot=plot
                )
            ipdb.set_trace()

            # debugging
            '''
            integrated_post_screen_spectrum = None
            for output_channel_name in instrument_dep_terms.output_channels.keys():
                if integrated_post_screen_spectrum is None:
                    integrated_post_screen_spectrum = instrument_dep_terms.sources_astroph['star']['flux_integrated_post_screen_ph_sec_m2_um'][output_channel_name]
                else:
                    integrated_post_screen_spectrum += instrument_dep_terms.sources_astroph['star']['flux_integrated_post_screen_ph_sec_m2_um'][output_channel_name]
            status_bool = np.allclose(integrated_post_screen_spectrum, instrument_dep_terms.sources_astroph["star"]["pre_screen_astro_flux_ph_sec_m2_um"])
            ipdb.set_trace()
            if not status_bool:
                logger.error("Flux conservation check failed")
                exit()
            '''

            ###### START DEBUGGING (post-transmission-screen, pre-aperture)
            # FYI: star flux in each output channel after transmission screens
            plt.clf()
            if plot and "star" in instrument_dep_terms.sources_astroph:
                star_src = instrument_dep_terms.sources_astroph["star"]
                wavel = star_src["wavel"]
                wavel_vals = wavel.value if hasattr(wavel, "value") else wavel
                flux_by_output = star_src.get("flux_integrated_post_screen_ph_sec_m2_um", {})
                if flux_by_output:
                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    for ch_name in instrument_dep_terms.output_channels.keys():
                        if ch_name not in flux_by_output:
                            continue
                        flux_1d = flux_by_output[ch_name]
                        flux_vals = flux_1d.value if hasattr(flux_1d, "value") else flux_1d
                        ax.plot(wavel_vals, flux_vals, label=ch_name)
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ph/s/m^2/um, integrated over field)")
                    ax.set_title(
                        f"Star flux per output (post-screen, pre-aperture), angle {angle_deg:g}°"
                    )
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_all_outputs_post_screen_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    #fig.savefig(file_name_plot)
                    plt.show()
                    logger.info(f"Saved post-screen star flux per output: {file_name_plot}")
            ###### END DEBUGGING (post-transmission-screen, pre-aperture)

            # Pass through telescope aperture
            logger.info("Passing through telescope aperture (incl. telescope throughput)...")
            instrument_dep_terms.pass_through_aperture(plot=plot)

            ###### START DEBUGGING (post-aperture, pre-disperse)
            # FYI: star flux in each output channel after aperture (photons, pre-detector bins)
            plt.clf()
            if plot:
                star_prop = instrument_dep_terms.prop_dict.get("star")
                if star_prop is not None:
                    wavel = star_prop["wavel"]
                    wavel_vals = wavel.value if hasattr(wavel, "value") else wavel
                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    flux_by_output = star_prop["flux_cube_post_screen_post_aperture_ph_sec_um"]
                    for ch_name in instrument_dep_terms.output_channels.keys():
                        if ch_name not in flux_by_output:
                            continue
                        flux_1d = np.sum(flux_by_output[ch_name], axis=(1, 2))
                        flux_vals = flux_1d.value if hasattr(flux_1d, "value") else flux_1d
                        ax.plot(wavel_vals, flux_vals, label=ch_name)
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ph/s per um, integrated over field)")
                    ax.set_title(
                        f"Star flux per output (post-aperture, pre-disperse), angle {angle_deg:g}°"
                    )
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_all_outputs_post_aperture_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    #fig.savefig(file_name_plot)
                    plt.show()
                    logger.info(f"Saved post-aperture star flux per output: {file_name_plot}")
            ###### END DEBUGGING (post-aperture, pre-disperse)

            # set instrumental noise terms and update the OutputChannel objects
            logger.info("Assigning intrinsic instrumental noise ...")
            instrument_dep_terms.calculate_instrinsic_instrumental_noise()

            # disperse astrophysical signals on the detector (i.e., update the OutputChannel objects; note that input astrophysical signals should still be photons, not electrons)
            logger.info("Dispersing signals on channel detectors ...")
            instrument_dep_terms.disperse_astro_signals_on_detector(plot=plot)

            ###### START DEBUGGING (post-disperse, pre-combine)
            # FYI: star flux in each output channel after disperse (photons, before ADU conversion)
            plt.clf()
            if plot and instrument_dep_terms.output_channels:
                ref_ch = instrument_dep_terms.output_channels["output_3_dark"]
                if (
                    ref_ch.bin_centers is not None
                    and "star" in ref_ch.astroph_signal
                ):
                    wavel_center = (
                        ref_ch.bin_centers.value
                        if hasattr(ref_ch.bin_centers, "value")
                        else ref_ch.bin_centers
                    )
                    wavel_width = (
                        ref_ch.bin_widths.value
                        if hasattr(ref_ch.bin_widths, "value")
                        else ref_ch.bin_widths
                    )
                    wavel_edges = np.empty(len(wavel_center) + 1)
                    wavel_edges[1:-1] = 0.5 * (wavel_center[:-1] + wavel_center[1:])
                    wavel_edges[0] = wavel_center[0] - 0.5 * wavel_width[0]
                    wavel_edges[-1] = wavel_center[-1] + 0.5 * wavel_width[-1]

                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    for ch_name, ch in instrument_dep_terms.output_channels.items():
                        star_sig = ch.astroph_signal.get("star")
                        if star_sig is None:
                            continue
                        y_col = star_sig["flux_astro_1d_interpolated_ph_sec_wavel_bin"]
                        y_vals = y_col.value if hasattr(y_col, "value") else y_col
                        ax.stairs(y_vals, edges=wavel_edges, label=ch_name)
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ph/s per wavelength bin)")
                    ax.set_title(
                        f"Star flux per output (post-disperse, pre-combine), angle {angle_deg:g}°"
                    )
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_all_outputs_post_disperse_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    #fig.savefig(file_name_plot)
                    plt.show()
                    logger.info(f"Saved post-disperse star flux per output: {file_name_plot}")
            ###### END DEBUGGING (post-disperse, pre-combine)

            # pack the signals together (and convert photons to electrons)
            instrument_dep_terms.combine_astro_and_instrum_signals()

            ###### START DEBUGGING (pre-chop)
            # FYI: star flux in each output channel before chopping
            plt.clf()
            if plot and instrument_dep_terms.output_channels:
                dc_rate = next(
                    iter(instrument_dep_terms.output_channels["output_3_dark"].tables_by_dark_current)
                )
                star_col = "astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"
                ref_tbl = instrument_dep_terms.output_channels["output_3_dark"].tables_by_dark_current[dc_rate]
                if star_col in ref_tbl.colnames:
                    wavel_center = ref_tbl["wavel_bin_center"].value
                    wavel_width = ref_tbl["wavel_bin_width"].value
                    wavel_edges = np.empty(len(wavel_center) + 1)
                    wavel_edges[1:-1] = 0.5 * (wavel_center[:-1] + wavel_center[1:])
                    wavel_edges[0] = wavel_center[0] - 0.5 * wavel_width[0]
                    wavel_edges[-1] = wavel_center[-1] + 0.5 * wavel_width[-1]

                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    for ch_name, ch in instrument_dep_terms.output_channels.items():
                        out_tbl = ch.tables_by_dark_current.get(dc_rate)
                        if out_tbl is None or star_col not in out_tbl.colnames:
                            continue
                        ax.stairs(out_tbl[star_col].value, edges=wavel_edges, label=ch_name)
                    ax.set_yscale("symlog", linthresh=1.0)
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ADU)")
                    ax.set_title(
                        f"Star flux per output (pre-chop), angle {angle_deg:g}°, dc={dc_rate:g}"
                    )
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_all_outputs_pre_chop_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    #fig.savefig(file_name_plot)
                    plt.show()
                    logger.info(f"Saved pre-chop star flux per output: {file_name_plot}")
            ###### END DEBUGGING (pre-chop)

            # chop the signal between dark outputs
            instrument_dep_terms.chop_signal(plot=plot)

            ###### START DEBUGGING
            # FYI: star flux on dark outputs 3 and 4 (from chopped table)
            plt.clf()
            if plot and instrument_dep_terms.post_chop_tables_by_dark_current:
                dc_rate = next(iter(instrument_dep_terms.post_chop_tables_by_dark_current))
                chopped_tbl = instrument_dep_terms.post_chop_tables_by_dark_current[dc_rate]
                star_col = "astro_star_flux_adu_sec_for_wavel_bin_and_integration_tot"
                col3 = f"output_3_dark_{star_col}"
                col4 = f"output_4_dark_{star_col}"
                col_chop = f"chopped_{star_col}"
                if col3 in chopped_tbl.colnames and col4 in chopped_tbl.colnames:
                    wavel_center = chopped_tbl["wavel_bin_center"].value
                    wavel_width = chopped_tbl["wavel_bin_width"].value
                    wavel_edges = np.empty(len(wavel_center) + 1)
                    wavel_edges[1:-1] = 0.5 * (wavel_center[:-1] + wavel_center[1:])
                    wavel_edges[0] = wavel_center[0] - 0.5 * wavel_width[0]
                    wavel_edges[-1] = wavel_center[-1] + 0.5 * wavel_width[-1]
                    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                    ax.stairs(chopped_tbl[col3].value, edges=wavel_edges, label="output_3_dark (star)")
                    ax.stairs(chopped_tbl[col4].value, edges=wavel_edges, label="output_4_dark (star)")
                    if col_chop in chopped_tbl.colnames:
                        ax.stairs(chopped_tbl[col_chop].value, edges=wavel_edges, label="chopped (star)")
                    ax.set_yscale("symlog", linthresh=1.0)  # symlog handles negative leakage
                    ax.set_xlim(4.0, 18.5)
                    ax.set_xlabel("Wavelength (um)")
                    ax.set_ylabel("Flux (ADU)")
                    ax.set_title(f"Star flux: post-chop dark outputs 3 vs 4, angle {angle_deg:g}°, dc={dc_rate:g}")
                    ax.legend(fontsize=8, loc="best")
                    save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])
                    file_name_plot = (
                        f"{save_dir}star_flux_output3_vs_output4_angle_{angle_deg:g}_QE_{qe:.2f}.png"
                    )
                    plt.show()
                    logger.info(f"Saved star output 3/4 comparison: {file_name_plot}")

            ###### END DEBUGGING

            # write condensed information at this transmission screen angle (to avoid mem leak)
            # see plot of chopped planet flux: instrument_dep_terms.prop_dict['exoplanet_model_10pc']['flux_cube_post_screen_post_aperture_ph_sec_um']['chopped_dark_outputs'][15,:,:].value
            record_info_at_angle_and_qe(
                angle_deg=angle_deg,
                qe=qe,
                output_channels=instrument_dep_terms.output_channels,
                post_chop_tables_by_dark_current=instrument_dep_terms.post_chop_tables_by_dark_current,
                save_dir=str(config['dirs']['save_s2n_data_unique_dir']),
                plot=plot,
            )

    else:
        logger.info("Skipping simulation and calculating S/N from HDF5 files...")

    if calculate_s2n:
        logger.info("Calculating S/N from HDF5 files.")
        calculate_s2n_post_rotation(config['dirs']['save_s2n_data_unique_dir'], config=config)

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
        logging.info(f"--------------------------------")
        logging.info(f"--------------------------------")
        logging.info(f"Processing single calculation:")
        logging.info(f"Parameter qe = {qe}")


        # run single calculation, for range of rotation angles
        # this could involve either or both of:
        # 1. generating simulations and writing out the HDF5 files
        # 2. calculating the S/N from the HDF5 files
        time_0 = time.time()
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
        time_1 = time.time()
        logging.info(f"Time taken for run_single_calculation(): {time_1 - time_0:.2f} seconds")
                    
        if success:
            logging.info(f"  ✓ Success for the following planetary system parameters:")
            logging.info(system_params)
        else:
            logging.info(f"  ✗ Failed for qe={qe}")
        success_all = success_all and bool(success) # was this calculation successful too?
    
    return success_all


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

    # delete preexisting HDF5 output files?
    ## ## TODO: make this work; will need to enable deletion of HdF5 files in each output dir determined by the temp config file
    '''
    generate_sims = loader.config_getboolean(config_single_obs, "tasks", "generate_sims")
    if generate_sims:
        save_dir = Path(config['dirs']['save_s2n_data_unique_dir'])
        existing_hdf5 = sorted(save_dir.glob("angle_*.hdf5"))
        if existing_hdf5:
            print(f"\nFound {len(existing_hdf5)} HDF5 file(s) in {save_dir}:")
            for p in existing_hdf5[:5]:
                print(f"  - {p.name}")
            if len(existing_hdf5) > 5:
                print(f"  ... and {len(existing_hdf5) - 5} more")
            try:
                raw = input("Delete existing angle_*.hdf5 files before running? (y/N): ")
            except EOFError:
                raw = "n"
            if str(raw).strip().lower() in {"y", "yes"}:
                for p in existing_hdf5:
                    p.unlink()
                logger.info(f"Deleted {len(existing_hdf5)} HDF5 file(s) from {save_dir}")
            else:
                logger.info("Keeping existing HDF5 files")
    '''

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
