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
from matplotlib.colors import LogNorm


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

    # host star radius
    rad_star = float(config['target']['rad_star']) * 69.6340 * 1e9 * u.cm

    # fluxes
    # stellar BB spectrum, BB_nu
    # units ergs/(cm^2 Hz sec sr)
    #bb_star_nu = BlackBody(temperature=5778*u.K)
    # convert BB_nu to BB_lambda
    bb_star_lambda = BlackBody(temperature=float(config['target']['T_star'])*u.K,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))
    # stellar surface flux, Fs_nu: multiply by pi steradians
    # W / (micron m2 sr) --> W / (micron m2)
    flux_star = np.pi * u.sr * bb_star_lambda(wavelength_um)
    # stellar luminosity in terms of energy, L_nu: rate at which the star radiates energy in all directions
    # W / (micron m2) --> W / micron
    luminosity_energy_star = 4 * np.pi * (rad_star**2) * flux_star 
    luminosity_energy_star = luminosity_energy_star.to(u.W / u.micron) # consistent units
    # stellar luminosity in terms of photons, L_gamma_nu (divide by energy units E=hc/lambda)
    # W / micron --> photons / (um sec)

    luminosity_photons_star = luminosity_energy_star * u.ph / (const.h * const.c / wavelength_um)
    luminosity_photons_star = luminosity_photons_star.to(u.ph / (u.micron * u.s)) # consistent units

    if plot:
        plt.clf()
        fig, ax1 = plt.subplots()
        
        # Primary y-axis for luminosity_photons_star
        color1 = 'tab:blue'
        ax1.set_xlabel(fr"$\lambda$ ({wavelength_um.unit})")
        ax1.set_ylabel(fr"$L_{{\lambda}}$ (ph * {luminosity_photons_star.unit})", color=color1)
        line1 = ax1.plot(wavelength_um, luminosity_photons_star, color=color1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary y-axis for flux_star
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(fr"$F(\lambda)$ ({flux_star.unit})", color=color2)
        line2 = ax2.plot(wavelength_um, flux_star, color=color2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title("Star spectrum (no distance correction)")
        plt.tight_layout()
        file_name_plot = "star_spectrum.png"
        plt.savefig(file_name_plot)
        print(f"Wrote stellar emission plot {file_name_plot}")

    return luminosity_photons_star, luminosity_energy_star


def generate_planet_bb_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, plot: bool = False) -> np.ndarray:

    # what should serve as the source of the planet spectrum? blackbody or file?
    planet_source = str(config['target']['planet_source'])
    logging.info(f"Planet source: {planet_source}")

    # planet radius
    rad_planet = float(config['target']['rad_planet']) * 0.637 * 1e9 * u.cm

    # planet BB spectrum
    temp_bb_planet = float(config['target']['pl_temp'])
    bb_planet_lambda = BlackBody(temperature=temp_bb_planet*u.K,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))
    # planet surface flux
    '''
    if planet_source == 'file':

        # this input file appears to be linearly-sampled in wavelength, with flux in units of u.photon / (u.s * u.m**2 * u.um)
        file_to_read_in = '/Users/eckhartspalding/Documents/git.repos/life_detectors/data/example_planet_spectrum.txt'
        df = pd.read_csv(file_to_read_in, delim_whitespace=True, names=('wavel','flux'))
        logging.info(f"Read in planet spectrum from {file_to_read_in}")

        # units of df['flux'] are u.photon / (u.s * u.m**2 * u.um)
        # so total units of test_photons below are sr * (above) = u.photon / (u.micron * u.s * u.m**2)
        test_luminosity = np.pi*u.sr * df['flux'].values * u.W/(u.m**2 * u.um) 
        
        
        # resample the spectrum onto the wavelength grid we gave it above
        #test_photons = test_photons.to(1 / u.micron / u.s)
        # Create interpolation function
        interp_func = interp1d(
            df['wavel'].values, 
            test_luminosity, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        # Interpolate to new wavelength grid
        new_flux = interp_func(wavelength_um)
        luminosity_energy_earth = new_flux * u.W/(u.m**2 * u.um)

        emission_photons = emission_energy * (wavelength_um / (const.h * const.c)) * u.photon
        emission_photons = emission_photons.to(u.photon / (u.um * u.s * u.m**2)) # units ph / (s um m2)

        test_photons = test_luminosity * (const.h * const.c / wavelength_um)

        # current units are u.ph / (u.micron * u.s * u.m**2)
        # want units of u.ph / (u.micron * u.s)

        luminosity_photons_planet = 4 * np.pi * (rad_planet**2) * emission_photons
        luminosity_photons_planet = luminosity_photons_planet.to(u.ph / (u.micron * u.s)) * (1/u.ph)  # last bit is to remove the photon units

        # convert photons to energy by multiplying by hc/lambda
        #test_energy = test_photons * (const.h * const.c / wavelength_um) * (1/u.photon) # last bit is to remove the photon units
        # convert energy to W/micron by dividing by 4piR^2
        #luminosity_energy_planet = 4 * np.pi * (rad_planet**2) * test_energy

        #luminosity_energy_planet = luminosity_energy_planet.to(u.W / u.micron) # consistent units
        #luminosity_photons_planet = luminosity_energy_planet / (const.h * const.c / wavelength_um)
        #luminosity_photons_planet = luminosity_photons_planet.to(1 / u.micron / u.s) # consistent units

        #flux_planet = flux_planet.to(u.W / (u.micron * u.m**2 * u.sr))
        #wavelength_um = df['wavel'].values * u.um
        #luminosity_photons_planet = df['luminosity_photons'].values * 1 / u.micron / u.s
    '''
    # bb_planet_lambda() units are W / (micron sr m2)
    # so total units of flux_planet are sr * (above) = W / (micron m2)
    flux_planet = np.pi*u.sr * bb_planet_lambda(wavelength_um)
    # planet luminosity
    luminosity_energy_planet = 4 * np.pi * (rad_planet**2) * flux_planet
    luminosity_energy_planet = luminosity_energy_planet.to(u.W / u.micron) # consistent units
    luminosity_photons_planet = luminosity_energy_planet * u.ph / (const.h * const.c / wavelength_um)
    luminosity_photons_planet = luminosity_photons_planet.to(u.ph / u.micron / u.s) # consistent units

    if plot:
        plt.clf()
        fig, ax1 = plt.subplots()
        
        # Primary y-axis for luminosity_photons_planet
        color1 = 'tab:blue'
        ax1.set_xlabel(fr"$\lambda$ ({wavelength_um.unit})")
        ax1.set_ylabel(fr"$L_{{\lambda}}$ (ph * {luminosity_photons_planet.unit})", color=color1)
        line1 = ax1.plot(wavelength_um, luminosity_photons_planet, color=color1)
        #ax1.set_xscale('log')
        #ax1.set_xlim(4., 18.)
        #ax1.set_ylim(1e-4, 1e0)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary y-axis for flux_planet
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(fr"$F(\lambda)$ ({flux_planet.unit})", color=color2)
        line2 = ax2.plot(wavelength_um, flux_planet, color=color2)
        #ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title("Planet spectrum (no distance correction)")
        plt.tight_layout()
        file_name_plot = "planet_spectrum.png"

        plt.savefig(file_name_plot)
        print(f"Wrote planet emission plot {file_name_plot}")


    return luminosity_photons_planet, luminosity_energy_planet


def generate_zodiacal_spectrum(config: configparser.ConfigParser, wavelength_um: np.ndarray, nulling_baseline: float, plot: bool = False) -> np.ndarray:

    # Generates a zodiacal background
    # See Sec. 2.2.3 in Dannert+ 2022

    # Inputs:
    # wavelength_um: wavelength in microns
    # nulling_baseline (m): baseline nulling distance 
    # plot: whether to plot the result and FYI plot of the whole background

    # set some parameters
    tau_opt = float(config['target']['tau_opt_zodiacal'])
    T_eff_zodiacal = float(config['target']['T_eff_zodiacal']) * u.K
    T_sol = 5778.0 * u.K # of Sun; does not change!
    rad_sol = 1 # of Sun, in normalized units; does not change!
    A_albedo = float(config['target']['A_albedo'])
    rad_sol = float(rad_sol) * 69.6340 * 1e9 * (1./1.496e13) # radius of Sun in AU (keep unitless for this function to work)
    single_mirror_diameter = float(config['telescope']['single_mirror_diameter']) * u.m

    lambda_rel_lon_los = float(config["observation"]["lambda_rel_lon_los"]) 
    beta_lat_los = float(config["observation"]["beta_lat_los"])

    # Inputs:
    # wavel: wavelength in microns (unitless for parts of this function to work)
    # tau_opt: optical depth
    # lambda_rel_lon_array: array of relative longitudes
    # beta_lat_array: array of latitudes

    bb_1 = BlackBody(temperature=T_eff_zodiacal,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))
    bb_2 = BlackBody(temperature=T_sol,  scale=1.0*u.W/(u.m**2*u.micron*u.sr))

    # the BB term of the zodiacal background
    # Ref. Eqn. 5.2 in Kelsall+ 2005, 'The DARWINsim science simulator', Sci-A/2005/297/Darwin/DMS
    term_i_los = bb_1(wavelength_um) + A_albedo * bb_2(wavelength_um) * ( rad_sol / 1.5 ) ** 2
    # the second term, for single value of the background along the line-of-sight
    term_ii_los =  ( np.pi/np.arccos(np.cos(lambda_rel_lon_los * np.pi/180.) * np.cos(beta_lat_los * np.pi/180.)) ) / ( np.sqrt( (np.sin(beta_lat_los * np.pi/180.) ** 2.) + 0.6 * (wavelength_um / (11.*u.um))**(-0.4) * np.cos(beta_lat_los * np.pi/180.) ** 2.) )

    # make FYI quantities in terms of photons, for debugging (note term_ii_los is just geometric)
    fyi_term_i_los = term_i_los * u.ph / ((const.h * const.c) / wavelength_um)
    fyi_term_i_los = fyi_term_i_los.to(u.ph / ( u.micron * u.sr * u.second * u.m**2))

    # for FYI 2D plot of the whole background
    N_beta = 100 # number of latitude points
    N_rel_lon = 200 # number of longitude points
    beta_lat_grid, lambda_rel_lon_grid = np.meshgrid(
        np.linspace(-90, 90, N_beta),
        np.linspace(90, 270, N_rel_lon),
        indexing='ij'
    )
    # for 2D FYI plot
    wavelengths_to_plot_2d = [5, 10, 20] * u.um  # microns (unitless; this is just for plotting)
    I_lambda_2d_energy = {} # dict to contain the 2D arrays showing emission at each wavelength
    I_nu_2d_energy = {} # dict to contain the 2D arrays showing emission at each wavelength
    I_lambda_2d_photons = {} # dict to contain the 2D arrays showing emission at each wavelength

    for wavel_this in wavelengths_to_plot_2d:
        # See Eq. 14 in Dannert+ 2022, in form tau * term_i * term_ii

        # units W / (um * sr * m2)
        term_i_2d = bb_1(wavel_this) + A_albedo * bb_2(wavel_this) * ( rad_sol / 1.5 ) ** 2
        # unitless; note the wavel_this/u.um is necessary to avoid math errors
        term_ii_2d = ( np.pi/np.arccos(np.cos(lambda_rel_lon_grid * np.pi/180.) * np.cos(beta_lat_grid * np.pi/180.)) ) / ( np.sqrt( (np.sin(beta_lat_grid * np.pi/180.) ** 2.) + 0.6 * (wavel_this / (11.*u.um))**(-0.4) * np.cos(beta_lat_grid * np.pi/180.) ** 2.) )
        
        I_lambda_2d_energy[str(wavel_this)] = tau_opt * term_i_2d * term_ii_2d # for units W  / (micron sr m2)
        I_nu_2d_energy[str(wavel_this)] = (I_lambda_2d_energy[str(wavel_this)] * (wavel_this)**2 / const.c).to(u.MJy / u.sr) # for units MJy/sr; note the plotted wavel_this is unitless, so have to tack on units here
        #I_lambda_2d_photons[str(wavel_this)] = I_lambda_2d_energy[str(wavel_this)] * u.photon / (const.h * const.c / wavel_this).to # for units 1 / (micron s)
    
    # make a full spectrum of the emission along the line-of-sight
    # note these line-of-sight units are still in terms of surface brightness: they include units of 'per steradian'
    I_lambda_los_array_energy = tau_opt * term_i_los * term_ii_los

    I_nu_los_array_energy = (I_lambda_los_array_energy * wavelength_um**2 / const.c).to(u.W / (u.m**2 * u.Hz * u.sr)) # intermediate units, if desired
    I_nu_los_array_energy_MJy = I_nu_los_array_energy.to(u.MJy / u.sr) # these units make for easier comparison to published numbers

    #radiance_nu_zodiacal_los_energy = I_lambda_los_array_energy
    # convert to photons
    I_lambda_los_array_photons = I_lambda_los_array_energy * u.ph / ((const.h * const.c) / wavelength_um)
    I_lambda_los_array_photons = I_lambda_los_array_photons.to(u.ph / (u.um * u.second * u.sr * u.m**2)) # units ph / ( um * sec * sr * m**2 )

    # now collect all the photons within (lambda_avg/B)**2 (a rough FOV) to find the total energy & photons along the line-of-sight
    ## ## TODO: this is still kind of hackneyed; find better way of evaluating total number of photons
    #fov_effective = ( (np.mean(wavelength_um) / (nulling_baseline * u.m)) * u.rad ) ** 2 

    # replicates concept in FD's code, using 'half FOV'
    # TODO: this is open to debate!
    hfov = (wavelength_um.to(u.m) / (2. * single_mirror_diameter)) * u.rad
    threshold_ampl =  1e-2 # this amplitude of the Gaussian is used to define the boundary of the effective FOV
    radius_fov_effective = (4./np.pi) * hfov * np.sqrt( -np.log(threshold_ampl) )
    fov_effective = np.pi * radius_fov_effective ** 2
    
    fov_effective = fov_effective.to(u.sr)

    I_lambda_los_array_photons = I_lambda_los_array_photons * fov_effective 
    I_lambda_los_array_energy = I_lambda_los_array_energy * fov_effective
    I_lambda_los_array_photons = I_lambda_los_array_photons.to(u.ph / (u.second * u.um * u.m**2 ))
    I_lambda_los_array_energy = I_lambda_los_array_energy.to(u.W / (u.um * u.m**2 ))

    if plot:
        plt.clf()
        # Plot three 2D subplots of zodiacal emission as fcn of beta and lambda, each for a different wavelength
        fig, axes = plt.subplots(3, 1, figsize=(9, 15))
        for i, wl in enumerate(wavelengths_to_plot_2d):
            # Find the index in wavelength_um closest to wl
            idx = np.abs(wavelength_um/u.um - wl/u.um).argmin()
            '''
            im = axes[i].imshow(I_nu_2d_energy[str(wl)].value, origin='lower', 
                               extent=[np.min(lambda_rel_lon_grid), np.max(lambda_rel_lon_grid), np.min(beta_lat_grid), np.max(beta_lat_grid)], aspect='auto')
            '''
            # colorscale like in Kendall+ 2005
            im = axes[i].imshow(
                I_nu_2d_energy[str(wl)].value,
                origin='lower',
                extent=[np.min(lambda_rel_lon_grid), np.max(lambda_rel_lon_grid), np.min(beta_lat_grid), np.max(beta_lat_grid)],
                aspect='auto',
                norm=LogNorm(),
                cmap='rainbow'
                
            )

            axes[i].set_title(f'Zodiacal background\nat {wavelength_um[idx]/u.um:.1f} μm')
            axes[i].set_xlabel(r'Relative Longitude to Sun, $\lambda_{\rm rel}$ (deg)')
            axes[i].set_ylabel(r'Latitude $\beta$ (deg)')
            
            # Set reasonable tick intervals
            axes[i].set_xticks([90, 135, 180, 225, 270])
            axes[i].set_yticks([-90, -45, 0, 45, 90])
            
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label(str(I_nu_2d_energy[str(wl)].unit))
        plt.tight_layout()
        file_name = 'zodiacal_emission_2d.png'
        plt.savefig(file_name)
        print(f"Wrote zodiacal emission 2D plot {file_name}")

        # spectrum of zodiacal light along the line-of-sight
        plt.clf()
        fig, ax1 = plt.subplots()
        
        # Primary y-axis for luminosity_photons_star
        color1 = 'tab:blue'
        ax1.set_xlabel(fr"$\lambda$ ({wavelength_um.unit})")
        ax1.set_ylabel(fr"$L_{{\lambda}}$ ({I_lambda_los_array_photons.unit})", color=color1)
        line1 = ax1.plot(wavelength_um, I_lambda_los_array_photons, color=color1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary y-axis for flux_star
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(fr"$F(\lambda)$ ({I_lambda_los_array_energy.unit})", color=color2)
        line2 = ax2.plot(wavelength_um, I_lambda_los_array_energy, color=color2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"Zodiacal spectrum (no distance correction)\n"
                  fr"$\lambda_{{\rm rel}}$={lambda_rel_lon_los}, $\beta$={beta_lat_los}")
        plt.tight_layout()
        file_name_plot = "zodiacal_spectrum_los.png"
        plt.savefig(file_name_plot)
        print(f"Wrote zodiacal emission spectrum plot {file_name_plot}")

    # I_nu should be
    # ~5 um: ~0.1s of MJy/sr
    # ~10 um: ~10 MJy/sr
    # ~20 um: ~10s to 100 MJy/sr

    return I_lambda_los_array_photons, I_lambda_los_array_energy


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


    def T_temp(r):
        # r: radius in disk (units AU)
        
        ## ## TODO: peg this to the stellar temp, if possible
        Ls = float(config['target']['L_star'])

        T = 278.3*u.K * ((Ls)**0.25) * ((r/u.au)**-0.5)

        return T


    def Sigma_m(r, r0, alpha, z, Sigma_m_0):
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
    def I_disk_lambda_r(r, r0, alpha, z, Sigma_m_0, wavel_array):

        bb = BlackBody(temperature=T_temp(r=r),  scale=1.0*u.W/(u.m**2*u.micron*u.sr))

        return Sigma_m(r=r, r0=r0, alpha=alpha, z=z, Sigma_m_0=Sigma_m_0) * bb(wavel_array)


    # surface brightness as function of wavelength I(lambda): I_disk_lambda_r integrated over dA = r dr dtheta
    def I_disk_lambda(r_array, r0, alpha, z, Sigma_m_0, wavel_array):

        # don't give r_array units, because it messes up the list comprehension & np.array below
        #r_array = r_array * u.AU
        
        # Integrate over r * I_disk_lambda_r()
        # units:
        # r: AU
        # I_disk_lambda_r(): (W / (micron sr m2))
        # r * I_disk_lambda_r(): AU * (W / (micron sr m2))
        #r_array[0] * I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        #test =  I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        #test2 = r_array[0] * I_disk_lambda_r(1, r0, alpha, Ls, z, Sigma_m_0, np.array([3.3,3.7]))
        integrand = np.array( [radius * np.pi * I_disk_lambda_r(radius, r0, alpha, z, Sigma_m_0, wavel_array) for radius in r_array] ) # this loses units in np.array() operation
        # factor of pi steradians comes from integrating over dtheta (one hemisphere of the disk)
        
        # integrate over r in r_array using the trapezoidal rule (logarithmic spacing), to leave array dimensions (1, lambda)
        # tack on (W / (micron sr m2)) * AU^2
        # (W / (micron sr m2)) comes from integrand
        # AU^2 units come from rdr in units of AU
        # final units here should be W / (micron m2)
        I_lambda = 2 * np.pi * np.trapz(integrand, x=r_array, axis=0) * (u.W / (u.um * u.m**2)) * u.AU # u.au comes from dr
        I_lambda = I_lambda.to(u.W / u.um)

        return I_lambda


    # scale emission for distance from Earth (effectively doing I_disk_lambda, except that integral is over d_Omega = r dr dtheta / D**2)
    #def I_disk_lambda_Earth(I_disk_lambda_array, D):
    #
    #    return I_disk_lambda_array * (1 / (D * u.pc))**2 * (u.pc / (206265. * u.AU))**2 * u.sr


    ## ## TODO: read in the below params from config file
    ## ## TODO: weave in the units from the beginning, rather than tacking them on at the end
    ## ## TODO: pass fluxes through telescope aperture with the same function

    # set up some basic params
    r_array = np.arange(0.034, 10, 0.1) * u.au
    
    # see Kennedy 2015 Table 1
    alpha = 0.34
    z = float(config['target']['z_exozodiacal'])
    Sigma_m_0 = 7.12e-8
    Ls = float(config['target']['L_star'])
    r0 = np.sqrt(Ls) * u.au # see Kennedy 2015 ApJSS, sec. 2.2.3

    T_array = T_temp(r=r_array)

    # units W / um
    luminosity_energy_disk_lambda = I_disk_lambda(r_array=r_array, r0=r0, alpha=alpha, z=z, Sigma_m_0=Sigma_m_0, wavel_array=wavelength_um)

    # convert to photons
    # units 1 / (um sec)
    luminosity_photons_exozodi_disk = luminosity_energy_disk_lambda * u.ph / (const.h * const.c / wavelength_um)
    luminosity_photons_exozodi_disk = luminosity_photons_exozodi_disk.to(u.ph / (u.micron * u.s))

    if plot:
        plt.clf()
        fig, ax1 = plt.subplots()
        
        color1 = 'tab:blue'
        ax1.set_xlabel(fr"$\lambda$ ({wavelength_um.unit})")
        ax1.set_ylabel(fr"$L_{{\lambda}}$ ({luminosity_photons_exozodi_disk.unit})", color=color1)
        line1 = ax1.plot(wavelength_um, luminosity_photons_exozodi_disk, color=color1)
        #ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(fr"$F(\lambda)$ ({luminosity_energy_disk_lambda.unit})", color=color2)
        line2 = ax2.plot(wavelength_um, luminosity_energy_disk_lambda, color=color2)
        #ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title("Exozodiacal disk spectrum (no distance correction)")
        file_name_plot = "exozodiacal_spectrum.png"
        plt.savefig(file_name_plot)
        print(f"Wrote exozodiacal emission plot {file_name_plot}")


    return luminosity_photons_exozodi_disk, luminosity_energy_disk_lambda


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


    # for unresolved sources, units ph /(um sec),  W / um
    # note these are independent of distance from Earth
    luminosity_photons_star, luminosity_energy_star = generate_star_spectrum(config, wavelength_um, plot=plot) # unresolved; just a BB
    luminosity_photons_planet, luminosity_energy_planet = generate_planet_bb_spectrum(config, wavelength_um, plot=plot) # unresolved; just a BB
    luminosity_photons_exozodi, luminosity_energy_exozodi = generate_exozodiacal_spectrum(config, wavelength_um, plot=plot) # unresolved
    # notes on zodiacal units:
    # 1. the zodiacal background is resolved, so within the function we deal with the extra 1/sr in the units by considering a crude FOV
    # 2. the output units here include 1/m**2, because the quantity is a surface brightness seen from Earth (i.e., there is no downstream distance correction that brings in 1/m**2)
    luminosity_photons_zodiacal, luminosity_energy_zodiacal = generate_zodiacal_spectrum(config, wavelength_um, nulling_baseline=12, plot=plot) # resolved

    # Sample data for different sources
    ## ## TODO: add zodiacal stuff
    sample_data = {
        "star_spectrum.txt": {
            "description": "Blackbody spectrum for star",
            "wavelength_um": wavelength_um,
            "luminosity_energy": luminosity_energy_star,
            "luminosity_energy_units": str(luminosity_energy_star.unit),
            "luminosity_photons": luminosity_photons_star,
            "luminosity_photons_units": str(luminosity_photons_star.unit),
            "plot_name": "star_BB_spectrum.png"
        },
        "exoplanet_bb_spectrum.txt": {
            "description": "Blackbody exoplanet spectrum",
            "wavelength_um": wavelength_um,
            "luminosity_energy": luminosity_energy_planet,
            "luminosity_energy_units": str(luminosity_energy_planet.unit),
            "luminosity_photons": luminosity_photons_planet,
            "luminosity_photons_units": str(luminosity_photons_planet.unit),
            "plot_name": "exoplanet_bb_spectrum.png"
        },
            "exozodiacal_spectrum.txt": {
            "description": "Exozodiacal dust spectrum",
            "wavelength_um": wavelength_um,
            "luminosity_energy": luminosity_energy_exozodi,
            "luminosity_energy_units": str(luminosity_energy_exozodi.unit),
            "luminosity_photons": luminosity_photons_exozodi,
            "luminosity_photons_units": str(luminosity_photons_exozodi.unit),
            "plot_name": "exozodiacal_spectrum.png"
        },
        "zodiacal_spectrum.txt": {
            "description": "Zodiacal dust spectrum",
            "wavelength_um": wavelength_um,
            "luminosity_energy": luminosity_energy_zodiacal,
            "luminosity_energy_units": str(luminosity_energy_zodiacal.unit),
            "luminosity_photons": luminosity_photons_zodiacal,
            "luminosity_photons_units": str(luminosity_photons_zodiacal.unit),
            "plot_name": "zodiacal_spectrum.png"
        }
    }

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
            f.write('# wavelength_unit=' + str(data['wavelength_um'].unit) + '\n')
            f.write('# luminosity_photons_unit=' + str(data['luminosity_photons_units']) + '\n')
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