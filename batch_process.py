#!/usr/bin/env python3
"""
Batch processing script for Life Detectors package.

This script allows you to run multiple calculations with different n_int values
and save the results as FITS files with different names.
"""

import os
import sys
import configparser
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core import calculator, astrophysical, instrumental
from modules.config import loader, validator
from modules.utils.helpers import create_sample_data, load_config
from modules.data.units import UnitConverter

def modify_config_file(config_path: str, n_int: int, output_path: str) -> str:
    """
    Create a modified configuration file with new n_int and output path values.
    
    Args:
        config_path: Path to the original configuration file
        n_int: New value for n_int
        output_path: New path for the FITS output file
        
    Returns:
        Path to the temporary modified configuration file
    """
    # Load the original config
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Modify the values
    config.set('observation', 'n_int', str(n_int))
    config.set('saving', 'save_s2n_data', output_path)
    
    # Create a temporary config file
    temp_config_path = config_path.replace('.ini', f'_temp_n{n_int}.ini')
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    return temp_config_path

def run_single_calculation(config_path: str, sources_to_include: List[str], 
                          n_int: int, output_path: str, 
                          overwrite: bool = True, plot: bool = False) -> bool:
    """
    Run a single calculation with specified parameters.
    
    Args:
        config_path: Path to the configuration file
        sources_to_include: List of sources to include in calculation
        n_int: Number of integrations
        output_path: Path for the output FITS file
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temporary config file with modified values
        temp_config_path = modify_config_file(config_path, n_int, output_path)
        
        # Set up logging
        log_file = loader.setup_logging()
        logger = logging.getLogger(__name__)
        logger.info(f"Running calculation with n_int={n_int}, output={output_path}")
        
        # Load the modified config
        config = load_config(config_file=temp_config_path)
        validator.validate_config(config)
        
        # Generate sample spectral data
        logger.info("Creating sample spectral data...")
        create_sample_data(config, overwrite=overwrite, plot=plot, read_sample_file=False)
        
        # Calculate astrophysical flux
        logger.info("Calculating astrophysical flux...")
        astrophysical_sources = astrophysical.AstrophysicalSources(config, unit_converter=UnitConverter())
        
        # Calculate incident flux for each source
        sources_astroph = {}
        for source_name in sources_to_include:
            if source_name in ["star", "exoplanet_bb", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]:
                sources_astroph[source_name] = astrophysical_sources.calculate_incident_flux(
                    source_name=source_name, plot=plot
                )
        
        # Pass through instrument
        logger.info("Passing astrophysical flux through telescope aperture...")
        instrument_dep_terms = instrumental.InstrumentDepTerms(
            config, 
            unit_converter=UnitConverter(),
            sources_astroph=sources_astroph,
            sources_to_include=sources_to_include
        )
        
        # Convert photons to electrons
        logger.info("Converting photons to photo-electrons...")
        instrument_dep_terms.pass_through_aperture(plot=plot)
        instrument_dep_terms.photons_to_e()
        instrument_dep_terms.calculate_instrinsic_instrumental_noise()
        
        # Calculate S/N
        logger.info("Calculating signal-to-noise ratio...")
        noise_calc = calculator.NoiseCalculator(
            config,
            sources_all=instrument_dep_terms, 
            sources_to_include=sources_to_include
        )
        
        # This will automatically save the FITS file 
        s2n = noise_calc.s2n_e()
        
        logger.info(f"Successfully completed calculation with n_int={n_int}")
        logger.info(f"Results saved to: {output_path}")
        
        # Clean up temporary config file
        os.remove(temp_config_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in calculation with n_int={n_int}: {e}")
        # Clean up temporary config file if it exists
        if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return False

def batch_process(config_path: str, n_int_values: List[int], 
                  output_dir: str, sources_to_include: List[str],
                  base_filename: str = "s2n", overwrite: bool = True, 
                  plot: bool = False) -> List[Tuple[int, str, bool]]:
    """
    Run batch processing with multiple n_int values.
    
    Args:
        config_path: Path to the base configuration file
        n_int_values: List of n_int values to process
        output_dir: Directory to save output FITS files
        sources_to_include: List of sources to include in calculations
        base_filename: Base filename for output files (without extension)
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        
    Returns:
        List of tuples (n_int, output_path, success)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for n_int in n_int_values:
        # Create output filename
        output_filename = f"{base_filename}_n{n_int:08d}.fits"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing n_int = {n_int}...")
        success = run_single_calculation(
            config_path=config_path,
            sources_to_include=sources_to_include,
            n_int=n_int,
            output_path=output_path,
            overwrite=overwrite,
            plot=plot
        )
        
        results.append((n_int, output_path, success))
        
        if success:
            print(f"  ✓ Success: {output_path}")
        else:
            print(f"  ✗ Failed: {output_path}")
    
    return results

def main():
    """Main entry point for the batch processing script."""
    parser = argparse.ArgumentParser(description="Batch process Life Detectors calculations")
    parser.add_argument("--config", "-c", required=True, 
                       help="Path to the base configuration file")
    parser.add_argument("--n-int", "-n", nargs="+", type=int, required=True,
                       help="List of n_int values to process")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Output directory for FITS files")
    parser.add_argument("--sources", "-s", nargs="+", 
                       default=["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"],
                       help="Sources to include in calculations")
    parser.add_argument("--base-filename", "-f", default="s2n",
                       help="Base filename for output files")
    parser.add_argument("--overwrite", action="store_true", default=True,
                       help="Overwrite existing files")
    parser.add_argument("--plot", action="store_true", default=False,
                       help="Generate plots")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    print("Life Detectors - Batch Processing")
    print("=" * 40)
    print(f"Config file: {args.config}")
    print(f"n_int values: {args.n_int}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sources: {args.sources}")
    print(f"Base filename: {args.base_filename}")
    print()
    
    # Run batch processing
    results = batch_process(
        config_path=args.config,
        n_int_values=args.n_int,
        output_dir=args.output_dir,
        sources_to_include=args.sources,
        base_filename=args.base_filename,
        overwrite=args.overwrite,
        plot=args.plot
    )
    
    # Print summary
    print("\nBatch Processing Summary:")
    print("-" * 30)
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for n_int, output_path, success in results:
        status = "✓" if success else "✗"
        print(f"{status} n_int={n_int:4d}: {output_path}")
    
    print(f"\nCompleted: {successful}/{total} calculations successful")
    
    if successful < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
