#!/usr/bin/env python3
"""
Demonstration script for the Life Detectors package.

This script shows how to use the package to calculate infrared detector noise
and signal-to-noise ratios for telescope observations.
"""

import numpy as np
from pathlib import Path
import ipdb

#from life_detectors.core import NoiseCalculator
#from life_detectors.config import create_default_config, save_config
from life_detectors.utils.helpers import create_sample_data

def main():
    """Run the demonstration."""
    print("Life Detectors - Infrared Detector Noise Calculator")
    print("=" * 60)

    # Initialize logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Life Detectors demonstration")
    
    # Create sample data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample spectral data
    print("Creating sample spectral data...")
    create_sample_data(data_dir, overwrite=True, plot=True)
    
    ipdb.set_trace()

    # Create a default configuration
    print("Creating default configuration...")
    config = create_default_config()
    
    # Save configuration for inspection
    config_file = "demo_config.yaml"
    save_config(config, config_file)
    print(f"Configuration saved to: {config_file}")
    
    # Initialize the noise calculator
    print("Initializing noise calculator...")
    calculator = NoiseCalculator(config)
    
    # Calculate signal-to-noise
    print("Calculating signal-to-noise...")
    results = calculator.calculate_snr()
    
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
    print("  python -m life_detectors.cli --config your_config.yaml")
    print("\nTo create a default configuration file:")
    print("  python -m life_detectors.cli --create-config my_config.yaml")

if __name__ == "__main__":
    main() 