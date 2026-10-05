#!/usr/bin/env python3
"""
Parameter sweep for LIFE detectors
"""

import argparse
from batch_process import parameter_sweep
import logging
from modules.utils import loader
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a LIFE detector parameter sweep with explicit config paths."
    )
    parser.add_argument(
        "--config-single-obs",
        default="config/main_config.ini",
        help="Base single-observation config file.",
    )
    parser.add_argument(
        "--config-sweep",
        default="config/sweep_config.ini",
        help="Sweep config file.",
    )
    parser.add_argument(
        "--config-planet-population",
        default="config/planet_population_earth_10pc_only.ini",
        help="Planet population config file.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional run-specific output root to keep simultaneous runs isolated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # set up logging
    log_file = loader.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Life Detectors - Batch Processing")
    logger.info(f"Log file: {log_file}")

    #########################################################
    ###### BEGIN USER INPUTS
    # starting config for a single observation
    # (parameters being swept will be overwritten)
    config_single_obs_path = args.config_single_obs
    # config file for making a parameter sweep; this effectively make a batch job
    config_sweep_path = args.config_sweep
    # if you want to apply the batch job with the above ini settings to an entire planet population, use this config file
    # if not using a planet population, use a placeholder file
    config_planet_population_path = args.config_planet_population
    output_root = args.output_root
    ###### END USER INPUTS
    #########################################################

    # kludge for now
    # laptop
    dir_temp_hdf5_files = "/Users/eckhartspalding/Documents/git.repos/life_detectors/hdf5_testing/temp_s2n_sweep_planet_index_0000000_Nuniverse_1_Nstar_1_dist_10_Rp_1_Rs_1_Ts_5778_L_1.0_z_3_eclip_lon_135_eclip_lat_45_Stype_G/"
    # node
    # dir_temp_hdf5_files = '/home/eckhart/Documents/git.repos/life_detectors/hdf5_testing/temp_s2n_sweep_planet_index_0000000_Nuniverse_1_Nstar_1_dist_10_Rp_1_Rs_1_Ts_5778_L_1.0_z_3_eclip_lon_135_eclip_lat_45_Stype_G/'
    hdf5_files = glob.glob(os.path.join(dir_temp_hdf5_files, "*.hdf5"))
    # Ask the user if they want to delete the files in this directory.
    response = (
        input(
            f"Found directory containing temp HDF5 files:\n    {dir_temp_hdf5_files}\nFound {len(hdf5_files)} HDF5 files in this directory.\nDo you want to delete HDF5 files in this directory? [y/N]: "
        )
        .strip()
        .lower()
    )
    # response = "y"
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

    # parameter sweep over QE, dark current
    parameter_sweep(
        config_single_obs_path=config_single_obs_path,
        config_sweep_path=config_sweep_path,
        config_planet_population_path=config_planet_population_path,
        output_root=output_root,
    )

    # this writes out HDF5 files with

    # One file per angle and QE: angle_{angle:06.2f}_qe_{qe:04.2f}.hdf5
    # Example hierarchical structure of output HDF5 files:
    #   angle_000.00_qe_0.80.hdf5
    #       └── dc_00.200_qe_0.80/          # one group per dark-current rate (e-/pix/s)
    #           ├── output_1_bright         # per-output table
    #           ├── output_2_bright
    #           ├── output_3_dark
    #           └── output_4_dark
    #       └── __extracted_qe_0.80/        # per-readout spectra box-summed from the readout FITS
    #
    # plus, per QE: extracted_cube_qe_{qe}_dc_{dc}.fits with (wavelength, readout, channel) cubes

    # One S/N cube per QE (not written for now: S/N is disabled while chopping is removed)
    # qe_{qe:04.2f}_s2n_cube.hdf5
    # holds plain h5py datasets, not QTables, and a .pkl copy is saved alongside it:
    # Example contents of the S/N cube HDF5 files (qe_{qe:04.2f}_s2n_cube.hdf5):
    #   - snr:               shape (n_bins, n_dc, n_qe)      -- S/N for each wavelength bin
    #   - snr_tot:           shape (n_dc, n_qe)              -- Total S/N
    #   - wavelength:        shape (n_bins,)                  -- Wavelength bin centres
    #   - wavel_bin_width:   shape (n_bins,)                  -- Wavelength bin widths
    #   - wavel_bin_edges:   shape (n_bins+1,)                -- Wavelength bin edges
    #   - dark_current:      shape (n_dc,)                    -- Dark current values
    #   - qe:                shape (n_qe,)                    -- Quantum efficiency values
    #   - base_titles:       shape (n_dc, n_qe)               -- Base plot titles
    #   - meta/formatted_plot_titles: shape (n_dc, n_qe)      -- Formatted plot titles

    logger.info("Complete.")


if __name__ == "__main__":
    main()
