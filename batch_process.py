#!/usr/bin/env python3
"""
Batch processing script for Life Detectors package.

This script allows you to run multiple calculations with different n_int values
and save the results as FITS files with different names.
"""

import os
from socket import IPV6_DONTFRAG
import sys
import configparser
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import ipdb



# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core import calculator, astrophysical, instrumental
from modules.config import loader, validator
from modules.utils.helpers import create_sample_data
from modules.data.units import UnitConverter

# Module-level logger so it's available everywhere in this file
logger = logging.getLogger(__name__)

def modify_config_file_sweep(config_path: str, n_int: int, qe: float) -> str:
    """
    Create a modified configuration file with new n_int and output path values.
    
    Args:
        config_path: Path to the original configuration file
        n_int: New value for n_int
        
    Returns:
        Path to the temporary modified configuration file
    """
    # Load the original config
    config = configparser.ConfigParser()
    config.read(config_path)
    

    # Modify the values
    config.set('observation', 'n_int', str(n_int))
    config.set('detector', 'quantum_efficiency', str(qe))
    #config.set('saving', 'save_s2n_data', output_path)
    
    # Create a temporary config file
    temp_config_dir = os.path.dirname(config_path) + '/parameter_sweeps/'
    qe_str = f"{qe:.2f}".replace('.', 'p') # for making better string (since it's a decimal)
    temp_config_path = temp_config_dir + os.path.basename(config_path).replace('.ini', f'_temp_n{n_int}_qe{qe_str}.ini')
    if not os.path.exists(temp_config_dir):
        os.makedirs(temp_config_dir, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        config.write(f)
    
    return temp_config_path


def modify_config_file_pl_system_params(config_path: str, base_filename: str, system_params: dict, lum_types: dict) -> str:
    """
    Create a modified configuration file which overwrites new planetary system parameters.
    
    Args:
        config_path: Path to the original configuration file
        base_filename: string to distinguish individual planets (index number from df of planet population data)
        system_params: Optional[dict] = None: the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        Path to the temporary modified configuration file
    """

    if system_params is not None:

        # Load the original config
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Modify the values
        config.set('target', 'distance', str(system_params['Ds'])) # distance to the star (pc)
        config.set('target', 'rad_planet', str(system_params['Rp'])) # planet radius (Earth radii)
        config.set('target', 'pl_temp', str(system_params['Tp'])) # planet temp (K)
        config.set('target', 'rad_star', str(system_params['Rs'])) # stellar radius (solar radii)
        config.set('target', 't_star', str(system_params['Ts'])) # stellar temperature (K)
        ## ## TO DO: make sure the modified luminosity is being used right, if it is being used at all
        config.set('target', 'L_star', str(lum_types[system_params['Stype'].lower()])) # stellar luminosity (L_sol) based on the type
        
        # for strings only
        config.set('target', 'Stype', str(system_params['Stype']))
        config.set('target', 'Nuniverse', str(system_params['Nuniverse']))
        config.set('target', 'Nstar', str(system_params['Nstar']))


        
        # Create a temporary config file

        # Compose parts of the file name for readability
        nuniverse_part = f"Nuniverse_{config['target']['Nuniverse']}"
        nstar_part = f"Nstar_{config['target']['Nstar']}"
        dist_part = f"dist_{config['target']['distance']}"
        rp_part = f"Rp_{config['target']['rad_planet']}"
        rs_part = f"Rs_{config['target']['rad_star']}"
        ts_part = f"Ts_{config['target']['t_star']}"
        l_part = f"L_{config['target']['L_star']}"
        stype_part = f"Stype_{config['target']['Stype']}"

        # use this string to 
        # 1. make a subdir (which will contain all the files for the different values of QE, n_int)
        # 2. name the files in the subdir as well
        file_basename_string = (
            f"temp_{base_filename}_"
            f"{nuniverse_part}_"
            f"{nstar_part}_"
            f"{dist_part}_"
            f"{rp_part}_"
            f"{rs_part}_"
            f"{ts_part}_"
            f"{l_part}_"
            f"{stype_part}"
        )
        #qe_str = f"{qe:.2f}".replace('.', 'p') # for making better string (since it's a decimal)
        temp_config_path = config_path.replace('.ini', file_basename_string + '/' + file_basename_string + '.ini')

        # Ensure the directory exists before writing the temporary config file
        temp_config_dir = os.path.dirname(temp_config_path)
        os.makedirs(temp_config_dir, exist_ok=True)
        with open(temp_config_path, 'w') as f:
            config.write(f)
        logger.info(f"Created temporary config file for one planetary system: {temp_config_path}")

    else:
        # just return the original config path
        return config_path
    
    return temp_config_path


def run_single_calculation(config_path: str, 
                            base_filename: str,
                            sources_to_include: List[str], 
                          n_int: int, 
                          qe: float,
                          overwrite: bool = True, 
                          plot: bool = False, 
                          system_params: Optional[dict] = None, 
                          lum_types: Optional[dict] = None) -> bool:
    """
    Run a single calculation with specified parameters.
    
    Args:
        config_path: Path to the configuration file
        base_filename: string to distinguish individual planets (index number from df of planet population data)
        sources_to_include: List of sources to include in calculation
        n_int: Number of integrations
        qe: quantum efficiency
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional[dict] = None: the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        output_path: Path of the output FITS file
    """

    if True:
        # Create temporary config file with modified values: use new n_int, QE values
        ## TO DO: CHECK THIS FOR CASE WHEN PLANET POPULATION IS NOT BEING DONE; DOES THE ABSENCE OF A BASE_FILENAME CAUSE PROBLEMS?
        temp_config_path_nint_qe = modify_config_file_sweep(config_path, n_int, qe)
        # modify again, for a given planetary system 
        if system_params is not None:
            temp_config_path = modify_config_file_pl_system_params(config_path = temp_config_path_nint_qe, base_filename = base_filename, system_params = system_params, lum_types = lum_types)
        else:
            temp_config_path = temp_config_path_nint_qe

        # First log entry: which config file we're using (original and temp)
        logger.info(f"--------------------------------")
        logger.info(f"Config path (base): {config_path}")
        logger.info(f"Config path (temporary one for this case, for batch processing): {temp_config_path}")

        # Load config in two forms:
        # - dict (for validation + directory creation)
        # - ConfigParser (for downstream code that expects .has_section/.options)
        # this is necessary for vestigial reasons while using only one load_config() function
        config_dict = loader.load_config(config_file=temp_config_path, makedirs=True)
        validator.validate_config(config_dict)
        config = configparser.ConfigParser()
        config.read(temp_config_path)

        # S/N results will be written to this file
        # Insert QE and n_int into the filename before .fits
        fname_base = temp_config_path.replace('.ini', '')
        output_fits_file_abs_path = f"{fname_base}_nint_{n_int}_qe_{qe:.4f}.fits"

        # Useful debug info about the loaded config: list actual INI-style sections and key-value pairs
        try:
            import configparser as _cp  # local import to avoid circulars
        except ImportError:
            _cp = None

        logger.info("------- Temp config contents -------")

        if _cp is not None and isinstance(config, _cp.ConfigParser):
            # Config is a ConfigParser: iterate its sections and items
            for section_name in config.sections():
                logger.info(f"[{section_name}]")
                for key, value in config[section_name].items():
                    logger.info(f"  {key} = {value}")
        elif isinstance(config, dict):
            # Config is a dict-of-dicts (as returned by some loaders)
            logger.info(f"Top-level keys: {list(config.keys())}")
            for section_name, section_dict in config.items():
                logger.info(f"[{section_name}]")
                if isinstance(section_dict, dict):
                    for key, value in section_dict.items():
                        logger.info(f"  {key} = {value}")
                else:
                    logger.info(f"  (section value) {section_dict}")
        else:
            # Fallback: just log the raw object
            logger.info(f"(Unrecognized config type {type(config)}; raw repr follows)")
            logger.info(repr(config))
        logger.info(f"Running calculation with n_int={n_int}, output={output_fits_file_abs_path}")
        
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
        s2n = noise_calc.s2n_e(file_name_fits_unique = output_fits_file_abs_path)
        
        #logger.info(f"Successfully completed calculation with n_int={n_int}")
        logger.info(f"Results saved to: {output_fits_file_abs_path}")
        
        # Clean up temporary config file
        #os.remove(temp_config_path)
        
        return True
        
    #except Exception as e:
    #    logger.error(f"Error in calculation with n_int={n_int}: {e}")
    #    # Clean up temporary config file if it exists
    #    if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
   #         os.remove(temp_config_path)
   #     return False

def batch_qe_nint_process(base_config_path: str, 
                n_int_values: List[float], 
                qe_values: List[float],
                  sources_to_include: List[str],
                  base_filename: str = "s2n", 
                  overwrite: bool = True, 
                  plot: bool = False, 
                system_params: Optional[dict] = None, 
                lum_types: Optional[dict] = None) -> List[Tuple[int, str, bool]]:
    """
    Run batch processing with multiple n_int values.
    
    Args:
        base_config_path: Path to the base configuration file
        n_int_values: List of n_int values to process
        qe_values: List of qe values to process
        sources_to_include: List of sources to include in calculations
        base_filename: Base filename for output files (without extension)
        overwrite: Whether to overwrite existing files
        plot: Whether to generate plots
        system_params: Optional dictionary of the planetary system parameters
        lum_types: Optional dictionary mapping the luminosities to the stellar types
        
    Returns:
        List of tuples (n_int, output_path, success)
    """
    # Create output directory if it doesn't exist
    #Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []

    # initialize for checking if all calculations for the QE, n_int run were successful
    success_all = True
    
    for qe in qe_values:
        for n_int in n_int_values:
            # Create output filename
            n_int = int(n_int)
            qe_pct = int(round(qe * 100))  # e.g. 0.87 -> 87
            #output_filename = f"{base_filename}_n{n_int:08d}_qe{qe_pct:03d}.fits"
            #output_path = os.path.join(output_dir, output_filename)
            
            logging.info(f"--------------------------------")
            logging.info(f"--------------------------------")
            logging.info(f"Processing single calculation:")
            logging.info(f"Parameter n_int = {n_int}")
            logging.info(f"Parameter qe = {qe}")

            success = run_single_calculation(
                config_path=base_config_path,
                base_filename = base_filename,
                sources_to_include=sources_to_include,
                n_int=n_int,
                qe=qe,
                overwrite=overwrite,
                plot=plot,
                system_params=system_params, 
                lum_types=lum_types
            )
                        
            if success:
                logging.info(f"  ✓ Success for n_int={n_int}, qe={qe} and the following planetary system parameters:")
                logging.info(system_params)
            else:
                logging.info(f"  ✗ Failed for n_int={n_int}, qe={qe}")
            success_all = success_all and bool(success) # was this calculation successful too?
    
    return success_all


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
    print(f"QE values: {args.qe}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sources: {args.sources}")
    print(f"Base filename: {args.base_filename}")
    print()
    
    # Run batch processing
    results = batch_process(
        config_path=args.config,
        n_int_values=args.n_int,
        qe_values=args.qe,
        output_dir=args.output_dir,
        sources_to_include=args.sources,
        base_filename=args.base_filename,
        overwrite=args.overwrite,
        plot=args.plot,
        system_params=args.system_params
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
