"""
Helper utility functions for the modules package.

This module provides various utility functions used throughout
the package for data formatting, validation, and sample data generation.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import configparser
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
from astropy.visualization import quantity_support

logger = logging.getLogger(__name__)

def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number with appropriate precision and scientific notation.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) < 1e-3 or abs(value) > 1e6:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def validate_file_path(filepath: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable.
    
    Args:
        filepath: Path to validate
        
    Returns:
        True if file exists and is readable
    """
    try:
        path = Path(filepath)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def create_sample_data(config: configparser.ConfigParser, overwrite: bool = False, plot: bool = False) -> None:
    """
    Create sample spectral data files for testing.
    
    Args:
        config: ConfigParser object
        overwrite: Whether to overwrite existing files
        plot: Whether to plot the data

    Returns:
        None (writes to file)
    """

    output_dir = Path(config['dirs']['data_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wavelength grid
    wavelength = np.logspace(0, 1, 100)  # 1-10 microns
    wavelength_um = wavelength * u.um

    ipdb.set_trace()
    # fluxes
    # stellar spectrum
    bb_star = BlackBody(temperature=5778*u.K)
    flux_star = bb_star(wavelength_um)
    # planet spectrum
    bb_planet = BlackBody(temperature=400*u.K)
    flux_planet = bb_planet(wavelength_um)

    ipdb.set_trace()
    
    # Sample data for different sources
    ## ## TODO: add zodiacal stuff
    sample_data = {
        "star_spectrum.txt": {
            "description": "Blackbody spectrum for star",
            "wavelength_um": wavelength_um,
            "flux": flux_star,  # Approximate blackbody
            "plot_name": "star_spectrum.png"
        },
        "exoplanet_spectrum.txt": {
            "description": "Exoplanet spectrum",
            "wavelength_um": wavelength_um,
            "flux": flux_planet,  # Cooler blackbody
            "plot_name": "exoplanet_spectrum.png"
        },
        "exozodiacal_spectrum.txt": {
            "description": "Exozodiacal dust spectrum",
            "wavelength_um": wavelength_um,
            "flux": 1e8 * wavelength_um ** (-1.5),  # Power law
            "plot_name": "exozodiacal_spectrum.png"
        },
        "zodiacal_spectrum.txt": {
            "description": "Zodiacal dust spectrum",
            "wavelength_um": wavelength_um,
            "flux": 1e9 * wavelength_um ** (-1.2),  # Power law
            "plot_name": "zodiacal_spectrum.png"
        }
    }
    
    for filename, data in sample_data.items():
        filepath = output_dir / filename
        
        if filepath.exists() and not overwrite:
            logger.info(f"Skipping {filename} (already exists)")
            continue
        
        ipdb.set_trace()
        # Create dataframe and write to CSV
        df = pd.DataFrame({
            'wavel': data['wavelength_um'],
            'flux': data['flux']
        })
        
        # add header with units
        with open(filepath, 'w') as f:
            f.write('# wavelength_unit=um\n')
            f.write('# flux_unit=photon_sec_m2_um\n')
        df.to_csv(filepath, mode='a', index=False)
        
        logger.info(f"Created sample data: {filepath}")

        if plot:
            ipdb.set_trace()
            plt.plot(data['wavelength_um'], data['flux'])
            plt.xlabel(fr"$\lambda$ [{wavelength_um.unit}]")
            plt.ylabel(fr"$F(\lambda)$ [{flux_star.unit}]")
            plt.title(data['description'])
            file_name_plot = output_dir / data['plot_name']
            plt.tight_layout()
            ipdb.set_trace()
            plt.savefig(file_name_plot)
            plt.close()
            logger.info(f"Wrote plot {file_name_plot}")



def calculate_photon_energy(wavelength_um: float) -> float:
    """
    Calculate photon energy in Joules.
    
    Args:
        wavelength_um: Wavelength in microns
        
    Returns:
        Photon energy in Joules
    """
    h = 6.626e-34  # Planck's constant (J⋅s)
    c = 3e8  # Speed of light (m/s)
    wavelength_m = wavelength_um * 1e-6  # Convert to meters
    
    return h * c / wavelength_m

def calculate_blackbody_flux(temperature: float, wavelength_um: float) -> float:
    """
    Calculate blackbody flux at a given temperature and wavelength.
    
    Args:
        temperature: Temperature in Kelvin
        wavelength_um: Wavelength in microns
        
    Returns:
        Flux in W/m²/micron
    """
    h = 6.626e-34  # Planck's constant (J⋅s)
    c = 3e8  # Speed of light (m/s)
    k = 1.381e-23  # Boltzmann constant (J/K)
    wavelength_m = wavelength_um * 1e-6  # Convert to meters
    
    # Planck's law
    exp_term = np.exp(h * c / (wavelength_m * k * temperature))
    flux = (2 * h * c**2 / wavelength_m**5) / (exp_term - 1)
    
    # Convert from per meter to per micron
    flux_per_um = flux * 1e-6
    
    return flux_per_um

def convert_flux_to_photons(energy_flux_watt: float, wavelength_um: float) -> float:
    """
    Convert energy flux to photon flux.
    
    Args:
        energy_flux_watt: Energy flux in W/m²/micron
        wavelength_um: Wavelength in microns
        
    Returns:
        Photon flux in photons/sec/m²/micron
    """
    photon_energy = calculate_photon_energy(wavelength_um)
    photon_flux = energy_flux_watt / photon_energy
    
    return photon_flux 