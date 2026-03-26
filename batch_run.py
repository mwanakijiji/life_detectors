#!/usr/bin/env python3
"""
Parameter sweep for LIFE detectors
"""

import os
from pathlib import Path
from batch_process import parameter_sweep
import logging
from modules.config import loader
import ipdb

def main():

    # set up logging
    log_file = loader.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("========================================")
    logger.info("Life Detectors - Batch Processing Examples")
    logger.info(f"Log file: {log_file}")

    #########################################################
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
    #########################################################

    parameter_sweep(config_single_obs_path = config_single_obs_path, 
                            config_sweep_path = config_sweep_path, 
                            config_planet_population_path = config_planet_population_path, 
                            planet_population = True)
    
    logger.info("Complete.")

if __name__ == "__main__":
    main()
