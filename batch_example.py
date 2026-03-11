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
import numpy as np
import pandas as pd
import glob
import corner
import copy
import matplotlib.pyplot as plt
import ipdb

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


def merge_psg_spectra_to_planet_population(
    df_planet_population: pd.DataFrame, planet_population_params: dict
) -> pd.DataFrame:
    """
    Merge PSG spectrum file names into the planet population DataFrame.

    Reads all *.response files from the configured directory, parses planet IDs
    from filenames (e.g. psg_cfg_00000015.response), and left-joins with the
    planet population. Missing PSG spectra are indicated as NaN.
    """
    dir_name_psg_spectra = planet_population_params['dir_file_name_psg_spectra']['dir_name']
    file_name_psg_spectra = glob.glob(os.path.join(dir_name_psg_spectra, '*.response'))
    df_psg_spectra_names = pd.DataFrame({
        'abs_file_name_psg_spectrum': file_name_psg_spectra,
    })
    df_psg_spectra_names['id'] = df_psg_spectra_names['abs_file_name_psg_spectrum'].apply(
        lambda x: int(x.split('psg_cfg_')[1].split('.')[0])
    )
    return df_planet_population.merge(df_psg_spectra_names, on='id', how='left')


def plot_planet_population_sample(
    df_planet_population: pd.DataFrame,
    cols_to_plot: list[str],
    output_path: str,
    max_sample_size: int = 10000,
) -> None:
    """
    Create and save a scatter matrix plot of the planet population.

    If the population exceeds max_sample_size, a random sample is used for plotting.
    """
    if len(df_planet_population) > max_sample_size:
        df_sample = copy.deepcopy(df_planet_population[cols_to_plot].sample(n=max_sample_size, replace=False))
    else:
        df_sample = copy.deepcopy(df_planet_population[cols_to_plot])
    title_string = f"Sample of {len(df_sample)} planets from population of {len(df_planet_population)}"
    axes = pd.plotting.scatter_matrix(df_sample, figsize=(10, 8))
    fig = axes[0, 0].figure
    fig.suptitle(title_string)
    fig.savefig(output_path)
    plt.close(fig)


def get_sweep_range(obs: dict, prefix: str) -> list[float]:
    """
    Build [start, start+step, ..., stop] from obs[prefix_start], obs[prefix_stop], obs[prefix_step].

    The stop value is included by extending the range by one step.
    """
    start = float(obs[f'{prefix}_start'])
    stop = float(obs[f'{prefix}_stop'])
    step = float(obs[f'{prefix}_step'])
    return np.arange(start, stop + step, step).tolist()


def example_parameter_sweep(config_single_obs_path, 
                            config_sweep_path, 
                            config_planet_population_path, 
                            planet_population: bool = False):
    """
    Example 3: Parameter sweep with many n_int values.

    config_single_obs_path: str
        Path to the configuration file for a single observation.
    config_sweep_path: str
        Path to the configuration file for the parameter sweep.
    config_planet_population_path: str
        Path to the configuration file for the planet population.
    planet_population: bool = False
        If True, the parameter sweep will be applied to an entire planet population.
        If False, the parameter sweep will be applied to a single observation.
    """
    print("\nExample 3: Parameter sweep")
    print("-" * 40)

    # load in all config files once, and don't load them again
    config_single_obs = loader.load_config(config_file=config_single_obs_path) # parameters of the observation
    config_sweep = loader.load_config(config_file=config_sweep_path) # parameters of the parameter sweep
    config_planet_population = loader.load_config(config_file=config_planet_population_path) # parameters of the planet population

    # if applying a parameter sweep to every planet in a population
    if planet_population:
        logging.info("Applying parameter sweep to an entire planet population")
        # for planet population, we need to read in the planet population file name
        config_planet_population = loader.load_config(config_file=config_planet_population_path)
        
        # the file that contains the parameters of the planet population
        file_name_planet_population = config_planet_population['file_name_planet_population']['file_name']
        lum_types = config_planet_population['lum_type'] # to map luminosities with stellar types
        # read in the planet population and merge PSG file names to it
        df_planet_population = pd.read_csv(file_name_planet_population, skiprows=1, sep='\s+')
        df_planet_population = merge_psg_spectra_to_planet_population(
            df_planet_population, config_planet_population
        )

        cols_to_plot = ['Rp', 'Porb', 'Mp', 'z', 'Tp', 'ap']
        fyi_plot_path = config_planet_population['file_name_planet_population']['fyi_plot_name']
        plot_planet_population_sample(df_planet_population, cols_to_plot, fyi_plot_path)
        logging.info(f"FYI plot of planet population saved to {fyi_plot_path}")

    else:
        logging.info("Applying parameter sweep to a single planetary system")
        df_planet_population = [None] # need to wrap in a list for length 1

    # parameter sweep in n_int and QE: create ranges from config
    # for month-long integration of 100sec integrations, n_int = 2592000/100 = 25920
    obs = config_sweep['observation']
    n_int_values = get_sweep_range(obs, 'n_int')
    qe_values = get_sweep_range(obs, 'qe')

    ipdb.set_trace()
    # get the astrophysical sources to include from the config file
    sources_to_include = [
        source
        for source, include in config_single_obs['astrophysical_sources_to_use'].items()
        if include in [True, "True", "true", 1, "1"]
    ]

    ipdb.set_trace()
    
    # loop over all the planetary systems
    for sys_num in range(len(df_planet_population)):

        if isinstance(df_planet_population, pd.DataFrame):
            system_params = df_planet_population.iloc[sys_num]
            logging.info(f"Processing system {sys_num} with parameters: {system_params}")
            base_filename = f"s2n_sweep_planet_index_{sys_num:07d}"
            
        else:
            system_params = None
            logging.info(f"No planet population; doing parameter sweep for a single system")
            base_filename = "s2n_sweep"

        # do parameter sweep over n_int and qe values for a single planetary system
        success_all = batch_qe_nint_process(
            base_config_path=config_single_obs_path,
            n_int_values=n_int_values,
            qe_values=qe_values,
            sources_to_include=sources,
            base_filename=base_filename,
            overwrite=True,
            plot=True, 
            system_params=system_params, 
            lum_types=lum_types
        )
    
        # Print summary
        logging.info(f"Parameter sweep completed: all calculations successful = {success_all}")

    
    return

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

def main():
    """Run all examples."""
    print("Life Detectors - Batch Processing Examples")
    print("=" * 50)

    log_file = loader.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Life Detectors - Batch Processing Examples")
    logger.info(f"Log file: {log_file}")

    ###### BEGIN USER INPUTS
    # starting config for a single observation
    # (parameters being swept will be overwritten)
    config_single_obs_path = "modules/config/demo_config.ini" 
    # config file for making a parameter sweep; this effectively make a batch job
    config_sweep_path = "modules/config/sweep_config.ini"
    # if you want to apply the batch job with the above ini settings to an entire planet population, use this config file
    # if not using a planet population, use a placeholder file
    config_planet_population_path = "modules/config/planet_population_config.ini"
    ###### END USER INPUTS
    
    # Run examples
    #example_simple_batch()
    #example_single_calculation()
    example_parameter_sweep(config_single_obs_path = config_single_obs_path, 
                            config_sweep_path = config_sweep_path, 
                            config_planet_population_path = config_planet_population_path, 
                            planet_population = True)
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
