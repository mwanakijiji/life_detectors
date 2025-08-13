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
from scipy.interpolate import interp1d
#from astropy import constants as const

logger = logging.getLogger(__name__)

const_c = 2.99792458e8 * u.m / u.s
const_h = 6.62607015e-34 * u.J * u.s

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


def generate_star_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray) -> np.ndarray:

    # stellar radius
    rad_star = 69.6340 * 1e9 * u.cm

    # fluxes
    # stellar BB spectrum, BB_nu
    # units ergs/(cm^2 Hz sec sr)
    #bb_star_nu = BlackBody(temperature=5778*u.K)
    # convert BB_nu to BB_lambda
    bb_star_lambda = BlackBody(temperature=5778*u.K,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))
    # stellar surface flux, Fs_nu: multiply by pi steradians
    # W / (micron m2 sr) --> W / (micron m2)
    flux_star = np.pi*u.sr * bb_star_lambda(wavelength_um)
    # stellar luminosity in terms of energy, L_nu: rate at which a planet radiates energy in all directions
    # W / (micron m2) --> W / micron
    luminosity_energy_star = 4 * np.pi * (rad_star**2) * flux_star 
    luminosity_energy_star = luminosity_energy_star.to(u.W / u.micron) # consistent units
    # stellar luminosity in terms of photons, L_gamma_nu (divide by energy units E=hc/lambda)
    # W / micron --> photons / (um sec)
    luminosity_photons_star = luminosity_energy_star / (const_h * const_c / wavelength_um)
    luminosity_photons_star = luminosity_photons_star.to(1 / u.micron / u.s) # consistent units

    return luminosity_photons_star, flux_star


def generate_planet_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, read_sample_file: bool = False) -> np.ndarray:

    # planet radius
    rad_planet = 0.637 * 1e9 * u.cm

    # planet BB spectrum
    temp_bb_planet = float(config['target']['pl_temp'])
    bb_planet_lambda = BlackBody(temperature=temp_bb_planet*u.K,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))
    # planet surface flux
    if read_sample_file:
        ## ## NOT WORKING YET
        df = pd.read_csv('/Users/eckhartspalding/Documents/job_science/postdoc_eth/life/example_spectrum.txt', delim_whitespace=True, names=('wavel','flux'))
        # units of df['flux'] are u.photon / (u.micron * u.s * u.m**2 * u.sr)
        # so total units of test_photons below are sr * (above) = u.photon / (u.micron * u.s * u.m**2)
        test_photons = np.pi*u.sr * df['flux'].values * u.photon / (u.micron * u.s * u.m**2 * u.sr)
        
        # resample the spectrum onto the wavelength grid we gave it above
        #test_photons = test_photons.to(1 / u.micron / u.s)
        # Create interpolation function
        interp_func = interp1d(
            df['wavel'].values, 
            test_photons, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        # Interpolate to new wavelength grid
        new_flux = interp_func(wavelength_um)
        emission_photons = new_flux * u.photon / (u.micron * u.s * u.m**2)

        # current units are u.ph / (u.micron * u.s * u.m**2)
        # want units of u.ph / (u.micron * u.s)

        luminosity_photons_planet = 4 * np.pi * (rad_planet**2) * emission_photons
        luminosity_photons_planet = luminosity_photons_planet.to(u.ph / (u.micron * u.s)) * (1/u.ph)  # last bit is to remove the photon units

        # convert photons to energy by multiplying by hc/lambda
        #test_energy = test_photons * (const_h * const_c / wavelength_um) * (1/u.photon) # last bit is to remove the photon units
        # convert energy to W/micron by dividing by 4piR^2
        #luminosity_energy_planet = 4 * np.pi * (rad_planet**2) * test_energy
        #ipdb.set_trace()
        #luminosity_energy_planet = luminosity_energy_planet.to(u.W / u.micron) # consistent units
        #luminosity_photons_planet = luminosity_energy_planet / (const_h * const_c / wavelength_um)
        #luminosity_photons_planet = luminosity_photons_planet.to(1 / u.micron / u.s) # consistent units

        #flux_planet = flux_planet.to(u.W / (u.micron * u.m**2 * u.sr))
        #wavelength_um = df['wavel'].values * u.um
        #luminosity_photons_planet = df['luminosity_photons'].values * 1 / u.micron / u.s
    else:
        # bb_planet_lambda() units are W / (micron sr m2)
        # so total units of flux_planet are sr * (above) = W / (micron m2)
        flux_planet = np.pi*u.sr * bb_planet_lambda(wavelength_um)
        # planet luminosity
        luminosity_energy_planet = 4 * np.pi * (rad_planet**2) * flux_planet
        luminosity_energy_planet = luminosity_energy_planet.to(u.W / u.micron) # consistent units
        luminosity_photons_planet = luminosity_energy_planet / (const_h * const_c / wavelength_um)
        luminosity_photons_planet = luminosity_photons_planet.to(1 / u.micron / u.s) # consistent units

    return luminosity_photons_planet, flux_planet


def generate_exozodiacal_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray) -> np.ndarray:
    # exozodiacal dust spectrum
    #flux_exozodi = 1e8 * wavelength_um ** (-1.5)
    #luminosity_photons_exozodi = flux_exozodi * (const_h * const_c / wavelength_um)
    #return luminosity_photons_exozodi, flux_exozodi
    return


def create_sample_data(config: configparser.ConfigParser, overwrite: bool = False, plot: bool = False, read_sample_file: bool = False) -> None:
    """
    Create sample spectral data files for testing.
    
    Args:
        config: ConfigParser object
        overwrite: Whether to overwrite existing files
        plot: Whether to plot the data
        read_sample_file: Whether to read the sample file that LIFEsim uses

    Returns:
        None (writes to file)
    """

    output_dir = Path(config['dirs']['data_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create wavelength grid
    wavelength = np.logspace(0, 1.4, 100)  # 1-20 microns
    wavelength_um = wavelength * u.um


    luminosity_photons_star, flux_star = generate_star_spectrum(config, wavelength_um)
    luminosity_photons_planet, flux_planet = generate_planet_spectrum(config, wavelength_um, read_sample_file=False)
    luminosity_photons_exozodi, flux_exozodi = generate_exozodiacal_spectrum(config, wavelength_um)

    # Sample data for different sources
    ## ## TODO: add zodiacal stuff
    sample_data = {
        "star_spectrum.txt": {
            "description": "Blackbody spectrum for star",
            "wavelength_um": wavelength_um,
            "flux": flux_star,
            "luminosity_photons": luminosity_photons_star,
            "plot_name": "star_spectrum.png"
        },
        "exoplanet_spectrum.txt": {
            "description": "Exoplanet spectrum",
            "wavelength_um": wavelength_um,
            "flux": flux_planet,
            "luminosity_photons": luminosity_photons_planet,
            "plot_name": "exoplanet_spectrum.png"
        }
    }
    '''
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
    '''
    
    for filename, data in sample_data.items():
        filepath = output_dir / filename
        
        if filepath.exists() and not overwrite:
            logger.info(f"Skipping {filename} (already exists)")
            continue
        
        # Create dataframe and write to CSV
        df = pd.DataFrame({
            'wavel': data['wavelength_um'],
            'luminosity_photons': data['luminosity_photons']
        })
        
        # add header with units
        with open(filepath, 'w') as f:
            f.write('# wavelength_unit=um\n')
            f.write('# luminosity_photons_unit=photon/um/sec\n')
        df.to_csv(filepath, mode='a', index=False)
        
        logger.info(f"Created sample data: {filepath}")

        if plot:
            # individual plot
            plt.plot(data['wavelength_um'], data['luminosity_photons'])
            plt.xlabel(fr"$\lambda$ [{wavelength_um.unit}]")
            plt.ylabel(fr"$L_photons(\lambda)$ [{luminosity_photons_planet.unit}]")
            plt.title(data['description'])
            file_name_plot = output_dir / data['plot_name']
            plt.tight_layout()
            plt.savefig(file_name_plot)
            plt.close()
            logger.info(f"Wrote plot {file_name_plot}")


'''
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
'''