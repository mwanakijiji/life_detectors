"""
Command line interface for the modules package.

This module provides a command-line interface for running noise calculations
and generating reports.
"""

import argparse
import json
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ..core import NoiseCalculator
from ..config import load_config, create_default_config, save_config

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Life Detectors - Infrared Detector Noise Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  life-detectors

  # Run with custom configuration file
  life-detectors --config my_config.yaml --output results.json

  # Create a default configuration file
  life-detectors --create-config default_config.yaml

  # Run with verbose output
  life-detectors --verbose
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output file (JSON format)"
    )
    
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default configuration file at the specified path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary only (not full results)"
    )
    
    parser.add_argument(
        "--noise-budget",
        action="store_true",
        help="Calculate and display noise budget breakdown"
    )
    
    parser.add_argument(
        "--optimal-params",
        type=float,
        metavar="TARGET_SNR",
        help="Calculate optimal parameters for target SNR"
    )
    
    return parser.parse_args()

def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the calculation results."""
    summary = results.get("summary", {})
    
    print("\n" + "="*60)
    print("LIFE DETECTORS - NOISE CALCULATION SUMMARY")
    print("="*60)
    
    print(f"Integrated Signal-to-Noise: {summary['integrated_snr']:.2f}")
    print(f"Wavelength Range: {summary['wavelength_range']['min']:.1f} - {summary['wavelength_range']['max']:.1f} {summary['wavelength_range']['units']}")
    print(f"Integration Time: {summary['integration_time']:.0f} seconds")
    print(f"Total Astrophysical Noise: {summary['total_astrophysical_noise']:.2f} ADU/pixel")
    print(f"Total Instrumental Noise: {summary['total_instrumental_noise']:.2f} ADU/pixel")
    print(f"Total Noise: {summary['total_noise']:.2f} ADU/pixel")
    print(f"Detection Limit: {summary['detection_limit']:.2f} ADU/pixel")
    print(f"SNR Range: {summary['min_snr']:.2f} - {summary['max_snr']:.2f}")
    print("="*60)

def print_noise_budget(budget: Dict[str, Any]) -> None:
    """Print noise budget breakdown."""
    print("\n" + "="*60)
    print("NOISE BUDGET BREAKDOWN")
    print("="*60)
    
    print("Astrophysical Sources:")
    for source, flux in budget["astrophysical_breakdown"].items():
        avg_flux = np.mean(flux)
        print(f"  {source:15s}: {avg_flux:.2e} photons/sec/m²/micron")
    
    print("\nInstrumental Sources:")
    for source, noise in budget["instrumental_breakdown"].items():
        print(f"  {source:15s}: {noise:.2f} ADU/pixel")
    
    print(f"\nTotal Astrophysical Noise: {np.mean(budget['total_astrophysical_noise']):.2f} ADU/pixel")
    print(f"Total Instrumental Noise: {budget['total_instrumental_noise']:.2f} ADU/pixel")
    print(f"Total Noise: {np.mean(budget['total_noise']):.2f} ADU/pixel")
    print("="*60)

def print_optimal_parameters(optimal_params: Dict[str, Any]) -> None:
    """Print optimal parameter calculations."""
    print("\n" + "="*60)
    print("OPTIMAL PARAMETERS")
    print("="*60)
    
    print(f"Current SNR: {optimal_params['current_snr']:.2f}")
    print(f"Target SNR: {optimal_params['target_snr']:.2f}")
    print(f"SNR Ratio (target/current): {optimal_params['snr_ratio']:.2f}")
    print(f"Current Integration Time: {optimal_params['current_integration_time']:.0f} seconds")
    print(f"Required Integration Time: {optimal_params['required_integration_time']:.0f} seconds")
    
    if np.isfinite(optimal_params['optimal_integration_time']):
        print(f"Optimal Integration Time: {optimal_params['optimal_integration_time']:.0f} seconds")
    else:
        print("Optimal Integration Time: Not achievable with current parameters")
    
    print("="*60)

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main() -> None:
    """Main CLI entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Handle configuration file creation
        if args.create_config:
            config = create_default_config()
            save_config(config, args.create_config)
            print(f"Default configuration saved to: {args.create_config}")
            return
        
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = create_default_config()
            logger.info("Using default configuration")
        
        # Initialize calculator
        calculator = NoiseCalculator(config)
        
        # Perform calculations
        results = calculator.calculate_snr()
        
        # Add summary to results
        results["summary"] = calculator.get_summary()
        
        # Handle different output modes
        if args.summary:
            print_summary(results)
        elif args.noise_budget:
            budget = calculator.calculate_noise_budget()
            print_noise_budget(budget)
        elif args.optimal_params:
            optimal_params = calculator.calculate_optimal_parameters(args.optimal_params)
            print_optimal_parameters(optimal_params)
        else:
            # Print summary by default
            print_summary(results)
        
        # Save results if output file specified
        if args.output:
            save_results(results, args.output)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 