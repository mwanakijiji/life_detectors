#!/usr/bin/env python3
"""
Example script showing how to use the batch processing functionality.

This demonstrates different ways to run batch jobs with varying n_int values.
"""

import os
from pathlib import Path
from batch_process import batch_process, run_single_calculation
import logging
from modules.config import loader
import ipdb
import numpy as np

def example_simple_batch():
    """Example 1: Simple batch processing with a few n_int values."""
    print("Example 1: Simple batch processing")
    print("-" * 40)
    
    config_path = "modules/config/demo_config.ini"
    n_int_values = [1000, 3000, 6000, 9000]
    output_dir = "batch_output"
    sources = ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]
    
    results = batch_process(
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

def example_parameter_sweep():
    """Example 3: Parameter sweep with many n_int values."""
    print("\nExample 3: Parameter sweep")
    print("-" * 40)


    
    # starting config for a single observation
    # (parameters being swept will be overwritten)
    config_single_obs_path = "modules/config/demo_config.ini" 
    config_sweep_path = "modules/config/sweep_config.ini"

    # read in the sweeped parameters
    sweeped_params = loader.load_config(config_file=config_sweep_path)

    # Create a range of n_int values
    # for month-long integration of 100sec integrations, n_int = 2592000/100 = 25920
    n_int_values = list[float](np.arange(float(sweeped_params['observation']['n_int_start']), float(sweeped_params['observation']['n_int_stop']), float(sweeped_params['observation']['n_int_step'])))  # 1000, 2000, ..., 10000
    qe_values = list[float](np.arange(float(sweeped_params['observation']['qe_start']), float(sweeped_params['observation']['qe_stop']), float(sweeped_params['observation']['qe_step'])))
    #output_dir = "parameter_sweep/20251105_R20_4pix_wide_footprint_2pt2pixperwavelelement_2month_observation"
    output_dir = "parameter_sweep/junk"
    sources = ["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]
    
    results = batch_process(
        config_path=config_single_obs_path,
        n_int_values=n_int_values,
        qe_values=qe_values,
        output_dir=output_dir,
        sources_to_include=sources,
        base_filename="s2n_sweep",
        overwrite=True,
        plot=True
    )
    
    # Print summary
    successful = sum(1 for _, _, success in results if success)
    print(f"Parameter sweep completed: {successful}/{len(results)} successful")
    
    return results

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
        
        results = batch_process(
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
    example_parameter_sweep()
    #example_custom_sources()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo run the batch processor from command line:")
    print("python batch_process.py --config modules/config/demo_config.ini \\")
    print("                       --n-int 1000 3000 6000 9000 \\")
    print("                       --output-dir my_batch_output \\")
    print("                       --sources star exoplanet_model_10pc exozodiacal zodiacal")

if __name__ == "__main__":
    main()
