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

from modules.core import calculator, astrophysical, instrumental
from modules.config import loader, validator
from modules.utils.helpers import create_sample_data, load_config
from modules.data.units import UnitConverter

def main(config_abs_file_name: str):
    """Run the demonstration."""

    # set up logging
    log_file = loader.setup_logging()
    logging.info("Life Detectors - Infrared Detector Noise Calculator")
    logging.info("=" * 60)

    # load config file
    logging.info("Loading config file...")
    config = load_config(config_file=config_abs_file_name)
    validator.validate_config(config)

    # Generate sample spectral data
    logging.info("Creating sample spectral data...")
    
    create_sample_data(config, overwrite=True, plot=True, read_sample_file=False)

    # Calculate the astrophysical flux incident on the instrument (post-null, if null=True)
    logging.info("Calculating astrophysical flux...")
    astrophysical_sources = astrophysical.AstrophysicalSources(config, unit_converter=UnitConverter()) ## ## UnitConverter is unused at the moment 
    incident_astro_star = astrophysical_sources.calculate_incident_flux(source_name = "star", null=True, plot=True)
    incident_astro_exoplanet = astrophysical_sources.calculate_incident_flux(source_name = "exoplanet", plot=True)

    # pass the astrophysical flux through the telescope aperture to the detector plane and into detector units
    logging.info("Passing astrophysical flux through telescope aperture to detector plane...")
    instrumental_effects = instrumental.InstrumentalSources(config, 
                                                            unit_converter=UnitConverter(), 
                                                            star_flux=incident_astro_star,
                                                            exoplanet_flux=incident_astro_exoplanet)
    _pass_aperture = instrumental_effects.pass_through_aperture()
    
    # convert photons to electrons
    logging.info("Converting photons to counts...")
    _phot_2_e = instrumental_effects.photons_to_e()
    # convert electrons to ADU
    _e_2_adu = instrumental_effects.e_to_adu()

    # intrinsic instrumental noise contributions in ADU
    logging.info("Calculating the instrumental-only noise sources...")
    _instrinsic_instrum = instrumental_effects.calculate_instrinsic_instrumental_noise()

    # find the noise
    logging.info("Calculating noise...")
    noise_calc = calculator.NoiseCalculator(config, 
                                            noise_origin = instrumental_effects)

    # pass astro signal through the nuller and find contribution to readout in ADU
    s2n = noise_calc.s2n_e()
    ## ## compare the above later with calculate_snr() below
    #snr = noise_calc.calculate_snr(contrib_astro = total_astro, contrib_instrum = incident_instrum) # find S/N
    
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