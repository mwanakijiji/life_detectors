#!/usr/bin/env python3
"""
Parameter sweep for LIFE detectors
"""

import argparse
import logging

from batch_process import parameter_sweep
from modules.utils import loader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (config paths, output root, smoke test)."""
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
        help=(
            "Optional directory under which this run's new run_<YYYYMMDDHHMMSS>/ "
            "output tree is made (default: [dirs] save_s2n_data_unique_dir)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: a small, fast run using the [smoke_test] overrides in "
            "the single-obs config (not a science run)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run a parameter sweep from the command line."""
    args = parse_args()

    # set up logging
    log_file = loader.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Life Detectors - Batch Processing")
    logger.info("Log file: %s", log_file)
    if args.smoke:
        logger.warning("*** SMOKE TEST MODE *** requested with --smoke")

    #########################################################
    # ## BEGIN USER INPUTS
    # starting config for a single observation
    # (parameters being swept will be overwritten)
    config_single_obs_path = args.config_single_obs
    # config file for making a parameter sweep; this effectively make a batch job
    config_sweep_path = args.config_sweep
    # if you want to apply the batch job with the above ini settings to an entire planet
    # population, use this config file
    # if not using a planet population, use a placeholder file
    config_planet_population_path = args.config_planet_population
    output_root = args.output_root
    # ## END USER INPUTS
    #########################################################

    # (no clean-up of old files needed: every run writes into its own new
    # run_<YYYYMMDDHHMMSS>/ tree, so previous runs' outputs are never mixed in)

    # parameter sweep over QE, dark current
    parameter_sweep(
        config_single_obs_path=config_single_obs_path,
        config_sweep_path=config_sweep_path,
        config_planet_population_path=config_planet_population_path,
        output_root=output_root,
        smoke_test=args.smoke,
    )

    # Output tree of one run (see modules/utils/helpers/output_dirs.py):
    #
    #   <[dirs] save_s2n_data_unique_dir or --output-root>/run_<YYYYMMDDHHMMSS>/
    #       [temp_<planet...>/]               per-planet folder (planet populations)
    #           fyi_plots/                    FYI plots (if print_fyi_plots = True)
    #           intermediate/                 readout FITS, transmission FITS, scene
    #                                         FITS, sample spectra, wavelength
    #                                         solution, and one HDF5 per angle and QE:
    #                                         angle_{angle:06.2f}_qe_{qe:04.2f}.hdf5
    #                                           └── __extracted_qe_{qe}/ (per-readout
    #                                               spectra, bin geometry, wavelength
    #                                               solution, electrons)
    #           products_counts_vs_time/      extracted_cube_qe_{qe}_dc_{dc}.fits and
    #                                         counts_vs_<x>_qe_{qe}_dc_{dc}_*.fits
#                                         (always) and .png (if plotting)
    #           products_snr/                 S/N cubes and plots (S/N is disabled
    #                                         for now, while chopping is removed)
    #
    # S/N cube contents, once re-enabled (qe_{qe:04.2f}_s2n_cube.hdf5, plus a .pkl):
    #   - snr:               shape (n_bins, n_dc, n_qe)      -- S/N for each wavelength
    # bin
    #   - snr_tot:           shape (n_dc, n_qe)              -- Total S/N
    #   - wavelength:        shape (n_bins,)                  -- Wavelength bin centres
    #   - wavel_bin_width:   shape (n_bins,)                  -- Wavelength bin widths
    #   - wavel_bin_edges:   shape (n_bins+1,)                -- Wavelength bin edges
    #   - dark_current:      shape (n_dc,)                    -- Dark current values
    #   - qe:                shape (n_qe,)                    -- Quantum efficiency
    # values
    #   - base_titles:       shape (n_dc, n_qe)               -- Base plot titles
    #   - meta/formatted_plot_titles: shape (n_dc, n_qe)      -- Formatted plot titles

    logger.info("Complete.")


if __name__ == "__main__":
    main()
