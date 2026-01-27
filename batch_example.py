#!/usr/bin/env python3
"""
Example script showing how to use the batch processing functionality.

This demonstrates different ways to run batch jobs with varying n_int values.
"""

import os
from pathlib import Path
from batch_process import batch_qe_nint_process, run_single_calculation
import logging
from modules.config import loader
import ipdb
import numpy as np
import pandas as pd

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

def example_parameter_sweep(planet_population: bool = False):
    """
    Example 3: Parameter sweep with many n_int values.

    planet_population: bool = False
        If True, the parameter sweep will be applied to an entire planet population.
        If False, the parameter sweep will be applied to a single observation.
    """
    print("\nExample 3: Parameter sweep")
    print("-" * 40)


    
    # starting config for a single observation
    # (parameters being swept will be overwritten)
    config_single_obs_path = "modules/config/demo_config.ini" 
    # config file for making a parameter sweep; this effectively make a batch job
    config_sweep_path = "modules/config/sweep_config.ini"
    # if you want to apply the batch job with the above ini settings to an entire planet population, use this config file
    config_planet_population_path = "modules/config/planet_population_config.ini"

    # read in the sweeped parameters
    sweeped_params = loader.load_config(config_file=config_sweep_path)

    # if applying a parameter sweep to every planet in a population
    if planet_population:
        logging.info("Applying parameter sweep to an entire planet population")
        # for planet population, we need to read in the planet population file name
        planet_population_params = loader.load_config(config_file=config_planet_population_path)
        file_name_planet_population = planet_population_params['file_name_planet_population']['file_name']
        lum_types = planet_population_params['lum_type'] # to map luminosities with stellar types
        # read in the planet population
        df_planet_population = pd.read_csv(file_name_planet_population, skiprows=1, delim_whitespace=True)
    else:
        logging.info("Applying parameter sweep to a single planetary system")
        df_planet_population = [None] # need to wrap in a list for length 1

    # parameter sweep: create a range of n_int values
    # for month-long integration of 100sec integrations, n_int = 2592000/100 = 25920
    n_int_values = list[float](np.arange(float(sweeped_params['observation']['n_int_start']), float(sweeped_params['observation']['n_int_stop']), float(sweeped_params['observation']['n_int_step'])))  # 1000, 2000, ..., 10000
    qe_values = list[float](np.arange(float(sweeped_params['observation']['qe_start']), float(sweeped_params['observation']['qe_stop']), float(sweeped_params['observation']['qe_step'])))
    #output_dir = "parameter_sweep/20251105_R20_4pix_wide_footprint_2pt2pixperwavelelement_2month_observation"
    output_dir = "parameter_sweep/junk"
    sources = ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]

    
    # loop over all the planetary systems
    for sys_num in range(len(df_planet_population)):

        if isinstance(df_planet_population, pd.DataFrame):
            system_params = df_planet_population.iloc[sys_num]
            logging.info(f"Processing system {sys_num} with parameters: {system_params}")
            base_filename = f"s2n_sweep_sys_{sys_num:03d}"
            
        else:
            system_params = None
            logging.info(f"No planet population; doing parameter sweep for a single system")
            base_filename = "s2n_sweep"
    
        # do parameter sweep over n_int and qe values for a single planetary system
        results = batch_qe_nint_process(
            base_config_path=config_single_obs_path,
            n_int_values=n_int_values,
            qe_values=qe_values,
            output_dir=output_dir,
            sources_to_include=sources,
            base_filename=base_filename,
            overwrite=True,
            plot=True, 
            system_params=system_params, 
            lum_types=lum_types
        )
    
        # Print summary
        successful = sum(1 for _, _, success in results if success)
        print(f"Parameter sweep completed: {successful}/{len(results)} successful")
    
    return

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

def main():
    """Run all examples."""
    print("Life Detectors - Batch Processing Examples")
    print("=" * 50)

    log_file = loader.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Life Detectors - Batch Processing Examples")
    logger.info(f"Log file: {log_file}")
    
    # Run examples
    #example_simple_batch()
    #example_single_calculation()
    example_parameter_sweep(planet_population = True)
    #example_custom_sources()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo run the batch processor from command line:")
    print("python batch_qe_nint_process.py --config modules/config/demo_config.ini \\")
    print("                       --n-int 1000 3000 6000 9000 \\")
    print("                       --output-dir my_batch_output \\")
    print("                       --sources star exoplanet_model_10pc exozodiacal zodiacal")

if __name__ == "__main__":
    main()
