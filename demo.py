#!/usr/bin/env python3
"""
Demonstration script for the Life Detectors package.

This script shows how to use the package to calculate infrared detector noise
and signal-to-noise ratios for telescope observations.
"""

import numpy as np
from pathlib import Path
import ipdb
import logging
import matplotlib.pyplot as plt

from modules.core import calculator, astrophysical #, instrumental
from modules.config import loader, validator
from modules.utils.helpers import create_sample_data, load_config
from modules.data.units import UnitConverter

def main(config_abs_file_name: str):
    """Run the demonstration."""
    print("Life Detectors - Infrared Detector Noise Calculator")
    print("=" * 60)

    # Initialize logging, make directories
    log_file = loader.setup_logging()
    logging.info("Starting Life Detectors demonstration")

    # load config file
    logging.info("Loading config file...")
    config = load_config(config_file=config_abs_file_name)
    validator.validate_config(config)

    # Generate sample spectral data
    logging.info("Creating sample spectral data...")
    create_sample_data(config, overwrite=True, plot=True)

    # Calculate the astrophysical flux incident on the instrument (no nulling yet)
    logging.info("Calculating astrophysical flux incident on the primary mirror...")
    astrophysical_sources = astrophysical.AstrophysicalSources(config, unit_converter=UnitConverter())
    incident_star = astrophysical_sources.calculate_incident_flux(source_name = "star", plot=True)

    # Calculate the flux incident on the detector after passing through the telescope

    # Calculate the detector outputs in ADU

    # Find the S/N
    
    # Initialize the noise calculator

    print("Initializing noise calculator...")
    test = calculator.NoiseCalculator(config, incident_flux=incident_star)

    # Calculate signal-to-noise
    print("Calculating signal-to-noise...")
    results = calculator.calculate_snr()

    '''
    
    # Print summary
    summary = calculator.get_summary()
    print("\nResults Summary:")
    print(f"  Integrated SNR: {summary['integrated_snr']:.2f}")
    print(f"  Wavelength Range: {summary['wavelength_range']['min']:.1f} - {summary['wavelength_range']['max']:.1f} microns")
    print(f"  Integration Time: {summary['integration_time']:.0f} seconds")
    print(f"  Total Noise: {summary['total_noise']:.2f} ADU/pixel")
    print(f"  Detection Limit: {summary['detection_limit']:.2f} ADU/pixel")
    
    # Calculate noise budget
    print("\nCalculating noise budget...")
    budget = calculator.calculate_noise_budget()
    
    print("\nNoise Budget Breakdown:")
    print("  Astrophysical Sources:")
    for source, flux in budget["astrophysical_breakdown"].items():
        avg_flux = np.mean(flux)
        print(f"    {source:15s}: {avg_flux:.2e} photons/sec/m²/micron")
    
    print("  Instrumental Sources:")
    for source, noise in budget["instrumental_breakdown"].items():
        print(f"    {source:15s}: {noise:.2f} ADU/pixel")
    
    # Calculate optimal parameters
    print("\nCalculating optimal parameters...")
    optimal_params = calculator.calculate_optimal_parameters(target_snr=10.0)
    
    print(f"  Current SNR: {optimal_params['current_snr']:.2f}")
    print(f"  Target SNR: {optimal_params['target_snr']:.2f}")
    print(f"  Required Integration Time: {optimal_params['required_integration_time']:.0f} seconds")
    
    if np.isfinite(optimal_params['optimal_integration_time']):
        print(f"  Optimal Integration Time: {optimal_params['optimal_integration_time']:.0f} seconds")
    else:
        print("  Optimal Integration Time: Not achievable with current parameters")
    
    print("\nDemonstration complete!")
    print("\nTo run with your own configuration:")
    print("  python -m modules.cli --config your_config.yaml")
    print("\nTo create a default configuration file:")
    print("  python -m modules.cli --create-config my_config.yaml")

    '''
if __name__ == "__main__":
    main(config_abs_file_name = "/Users/eckhartspalding/Documents/git.repos/life_detectors/modules/config/demo_config.ini") 