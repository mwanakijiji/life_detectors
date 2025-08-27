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
from astropy import constants as const


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


def generate_star_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, plot: bool = False) -> np.ndarray:

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
    # stellar luminosity in terms of energy, L_nu: rate at which the star radiates energy in all directions
    # W / (micron m2) --> W / micron
    luminosity_energy_star = 4 * np.pi * (rad_star**2) * flux_star 
    luminosity_energy_star = luminosity_energy_star.to(u.W / u.micron) # consistent units
    # stellar luminosity in terms of photons, L_gamma_nu (divide by energy units E=hc/lambda)
    # W / micron --> photons / (um sec)

    luminosity_photons_star = luminosity_energy_star / (const.h * const.c / wavelength_um)
    luminosity_photons_star = luminosity_photons_star.to(1 / (u.micron * u.s)) # consistent units

    ipdb.set_trace()

    if plot:
        plt.clf()
        plt.plot(wavelength_um, luminosity_photons_star)
        plt.xlabel(fr"$\lambda$ [{wavelength_um.unit}]")
        plt.ylabel(fr"$L_photons(\lambda)$ [{luminosity_photons_star.unit}]")
        plt.title("Star spectrum")
        plt.tight_layout()
        file_name_plot = "star_spectrum.png"
        plt.savefig(file_name_plot)
        print(f"Wrote stellar emission plot {file_name_plot}")

    return luminosity_photons_star, flux_star


def generate_planet_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, read_sample_file: bool = False, plot: bool = False) -> np.ndarray:

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
        #test_energy = test_photons * (const.h * const.c / wavelength_um) * (1/u.photon) # last bit is to remove the photon units
        # convert energy to W/micron by dividing by 4piR^2
        #luminosity_energy_planet = 4 * np.pi * (rad_planet**2) * test_energy
        #ipdb.set_trace()
        #luminosity_energy_planet = luminosity_energy_planet.to(u.W / u.micron) # consistent units
        #luminosity_photons_planet = luminosity_energy_planet / (const.h * const.c / wavelength_um)
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
        luminosity_photons_planet = luminosity_energy_planet / (const.h * const.c / wavelength_um)
        luminosity_photons_planet = luminosity_photons_planet.to(1 / u.micron / u.s) # consistent units

    if plot:
        plt.clf()
        plt.plot(wavelength_um, luminosity_photons_planet)
        plt.xlabel(fr"$\lambda$ [{wavelength_um.unit}]")
        plt.ylabel(fr"$L_photons(\lambda)$ [{luminosity_photons_planet.unit}]")
        plt.title("Planet spectrum")
        plt.tight_layout()
        file_name_plot = "planet_spectrum.png"
        plt.savefig(file_name_plot)
        print(f"Wrote planet emission plot {file_name_plot}")

    return luminosity_photons_planet, flux_planet


def generate_exozodiacal_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, plot: bool = False) -> np.ndarray:
    # exozodiacal dust spectrum
    #flux_exozodi = 1e8 * wavelength_um ** (-1.5)
    #luminosity_photons_exozodi = flux_exozodi * (const.h * const.c / wavelength_um)
    #return luminosity_photons_exozodi, flux_exozodi


    # surface brightness profile is
    # Ref.: Eqn. 1 in Kennedy 2015 ApJSS 216:23
    #
    # S_disk = Sigma_m * BB(lambda, T(r))
    #
    # where 
    # 
    # Sigma_m = z * Sigma_m_0 * (r/r0)^(-alpha), where z is number of zodis and Sigma_m_0 normalizes the surface brightness
    # T(r) = 278.3K * Ls^(0.25) * r^-(0.5)


    def T_temp(Ls, r):
        # Ls: luminosity of star (units L_sol)
        # r: radius in disk (units AU)

        T = 278.3*u.K * (Ls**0.25) * (r**-0.5)

        return T


    def Sigma_m(r, r0, alpha, Ls, z, Sigma_m_0):
        # r: radius in disk (units AU)
        # r0: reference radius (units AU)
        # alpha: power law index
        # z: number of zodis
        # Sigma_m_0: normalization factor
        #   Kennedy: 'is to be set at some r0 (in AU) such that the surface density is in units of zodis z (see Section 2.2.3).'

        Sigma_m = z * Sigma_m_0 * (r/r0)**(-alpha)

        return Sigma_m


    # spectral surface brightness profile I(lambda, r)
    # Eqn. 16 in Dannert+ 2022 A&A 664:A22 
    # Slight variation in notation: Eqn. 1 in Kennedy+ 2015 ApJSS 216:23
    # N.b. Kennedy uses 'S' instead of 'I' 
    def I_disk_lambda_r(r, r0, alpha, Ls, z, Sigma_m_0, wavel_array):

        bb = BlackBody(temperature=T_temp(Ls=Ls, r=r),  scale=1.0*u.W/(u.m**2*u.micron*u.sr))

        return Sigma_m(r=r, r0=r0, alpha=alpha, Ls=Ls, z=z, Sigma_m_0=Sigma_m_0) * bb(wavel_array)


    # surface brightness as function of wavelength I(lambda): I_disk_lambda_r integrated over dA = r dr dtheta
    def I_disk_lambda(r_array, r0, alpha, Ls, z, Sigma_m_0, wavel_array):

        # don't give r_array units, because it messes up the list comprehension & np.array below
        #r_array = r_array * u.AU
        
        # Integrate over r * I_disk_lambda_r()
        # units:
        # r: AU
        # I_disk_lambda_r(): (W / (micron sr m2))
        # r * I_disk_lambda_r(): AU * (W / (micron sr m2))
        #ipdb.set_trace()
        #r_array[0] * I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        #test =  I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        #test2 = r_array[0] * I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        integrand = np.array( [radius * np.pi * I_disk_lambda_r(radius, r0, alpha, Ls, z, Sigma_m_0, wavel_array) for radius in r_array] ) # this loses units in np.array() operation
        # factor of pi steradians comes from integrating over dtheta (one hemisphere of the disk)
        
        # integrate over r in r_array using the trapezoidal rule (logarithmic spacing), to leave array dimensions (1, lambda)
        # tack on (W / (micron sr m2)) * AU^2
        # (W / (micron sr m2)) comes from integrand
        # AU^2 units come from rdr in units of AU
        # final units here should be W / (micron m2)
        I_lambda = 2 * np.pi * np.trapezoid(integrand, x=r_array, axis=0) * (u.W / (u.um * u.m**2)) * u.AU**2 
        I_lambda = I_lambda.to(u.W / u.um)

        return I_lambda


    # scale emission for distance from Earth (effectively doing I_disk_lambda, except that integral is over d_Omega = r dr dtheta / D**2)

    def I_disk_lambda_Earth(I_disk_lambda_array, D):

        return I_disk_lambda_array * (1 / (D * u.pc))**2 * (u.pc / (206265. * u.AU))**2 * u.sr


    ## ## TODO: read in the below params from config file
    ## ## TODO: weave in the units from the beginning, rather than tacking them on at the end
    ## ## TODO: pass fluxes through telescope aperture with the same function

    # set up some basic params
    r_array = np.arange(0.1, 10, 0.1)
    r0 = 1
    alpha = 0.5
    z = 3
    Sigma_m_0 = 1
    Ls = 1

    T_array = T_temp(Ls=Ls, r=r_array)
    wavel_array = np.arange(2., 20, 0.1) * u.um

    # units W / um
    radiance_disk_lambda = I_disk_lambda(r_array=r_array, r0=r0, alpha=alpha, Ls=Ls, z=z, Sigma_m_0=Sigma_m_0, wavel_array=wavel_array)

    # convert to photons
    # units 1 / (um sec)
    luminosity_photons_exozodi_disk = radiance_disk_lambda * u.photon / (const.h * const.c / wavel_array)
    luminosity_photons_exozodi_disk = luminosity_photons_exozodi_disk.to(u.photon / (u.micron * u.s))

    if plot:
        plt.clf()
        plt.plot(wavel_array, luminosity_photons_exozodi_disk)
        plt.xlabel(fr"$\lambda$ [{wavel_array.unit}]")
        plt.ylabel(fr"$I(\lambda)$ [{luminosity_photons_exozodi_disk.unit}]")
        plt.title("Exozodiacal disk spectrum")
        plt.tight_layout()
        file_name_plot = "exozodiacal_spectrum.png"
        plt.savefig(file_name_plot)
        print(f"Wrote exozodiacal emission plot {file_name_plot}")

    return luminosity_photons_exozodi_disk, radiance_disk_lambda


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
    wavelength = np.logspace(-1, 1.4, 100)  # 1-20 microns
    wavelength_um = wavelength * u.um


    # units 1/(um sec),  W / (um m2)
    luminosity_photons_star, flux_star = generate_star_spectrum(config, wavelength_um, plot=plot)
    luminosity_photons_planet, flux_planet = generate_planet_spectrum(config, wavelength_um, read_sample_file=False, plot=plot)
    luminosity_photons_exozodi, flux_exozodi = generate_exozodiacal_spectrum(config, wavelength_um, plot=plot)

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

        '''
        if plot:
            # individual plot
            plt.plot(data['wavelength_um'], data['luminosity_photons'])
            plt.xlabel(fr"$\lambda$ [{wavelength_um.unit}]")
            plt.ylabel(fr"$L_photons(\lambda)$ [{luminosity_photons_planet.unit}]")
            plt.title(data['description'])
            file_name_plot = output_dir + data['plot_name']
            plt.tight_layout()
            plt.savefig(file_name_plot)
            plt.close()
            logger.info(f"Wrote plot {file_name_plot}")
        '''


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