"""
Astrophysical noise calculations for the modules package.

This module handles calculations of astrophysical noise sources including
stars, exoplanets, exozodiacal disks, and zodiacal backgrounds.

Sky coordinate convention: y first, x second (see instrumental.py module doc).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import ipdb
import configparser
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import units as u
import astropy.constants as const
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval


from ..data.spectra import SpectralData, load_spectrum_from_file
from ..data.units import UnitConverter
from ..utils.helpers import format_plot_title, parse_sky_position_arcsec_yx

logger = logging.getLogger(__name__)


def _box_kernel(n_pix: int, idx_y: int, idx_x: int, half_pix: int) -> np.ndarray:
    """Normalized square kernel centered on (idx_y, idx_x), clipped to the scene."""
    y0 = max(0, idx_y - half_pix)
    y1 = min(n_pix, idx_y + half_pix + 1)
    x0 = max(0, idx_x - half_pix)
    x1 = min(n_pix, idx_x + half_pix + 1)
    kernel = np.zeros((n_pix, n_pix))
    kernel[y0:y1, x0:x1] = 1.0
    kernel /= kernel.sum()
    return kernel


def _circle_kernel(n_pix: int, idx_y: int, idx_x: int, half_pix: int) -> np.ndarray:
    """Normalized circular kernel centered on (idx_y, idx_x); half_pix is the radius in pixels."""
    y_idx, x_idx = np.ogrid[:n_pix, :n_pix]
    kernel = ((y_idx - idx_y) ** 2 + (x_idx - idx_x) ** 2 <= half_pix ** 2).astype(float)
    kernel /= kernel.sum()
    return kernel


def _exozodi_kernel(n_pix: int, 
                    idx_y: int, 
                    idx_x: int, 
                    pix_size_arcsec: float,
                    dist_pc: float,
                    r0_au = 1 * u.au, 
                    alpha = 0.34, 
                    z_exozodiacal = 1, 
                    Sigma_m_0 = 7.12e-08) -> np.ndarray:
    '''
    Conical exozodi kernel

    INPUTS:
        n_pix: number of pixels in the scene
        idx_y: y index of the center of the disk
        idx_x: x index of the center of the disk
        pix_size_mas: pixel size in arcseconds
        dist_pc: distance to the target in parsecs
        r0_au: reference radius in AU
        alpha: power law index
        z: number of zodis
        Sigma_m_0: normalization factor

    RETURNS:
        kernel: 2D array of the exozodi
    '''

    # Ref.: Kennedy+ 2015 ApJSS 216:23, Eqn. (3)
    # Sigma_m(r) = z * Sigma_m_0 * (r/r0)^(-alpha)

    # note that a single spectrum is just spread across this resolved shape according to surface
    # density, for simplicity at the moment

    ## ## TODO: peg exozodi parameters to the model that generates the spectrum

    r0_arsec = (r0_au / dist_pc).value * u.arcsec

    # make a slice representing au from the center of the disk
    kernel = np.zeros((n_pix, n_pix))

    #n_pix = int(self.config['onsky_scene']['n_pix']) # should be odd number to simplify centering
    axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
    sky_xx_arcsec, sky_yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')

    # distances in arcsec from the center of the disk in pixels
    distances_arcsec = np.sqrt(sky_xx_arcsec**2 + sky_yy_arcsec**2)

    # surface density profile
    Sigma_m_2d = (z_exozodiacal * Sigma_m_0 * np.divide(distances_arcsec, r0_arsec) **(-alpha)).value

    # pixel at distance zero is a nan; just make it the max value
    Sigma_m_2d[~np.isfinite(Sigma_m_2d)] = np.max(Sigma_m_2d[np.isfinite(Sigma_m_2d)])

    kernel = np.copy(Sigma_m_2d)
    kernel /= kernel.sum()

    return kernel


def _zodiacal_kernel(n_pix: int, pix_size_arcsec: float) -> np.ndarray:
    '''
    Zodiacal background kernel
    '''
    kernel = np.ones((n_pix, n_pix))
    kernel /= kernel.sum()

    return kernel


def generate_star_scene(
    flux_star: u.Quantity,
    n_pix: int,
    idx_y_star: int,
    idx_x_star: int,
    half_pix: int,
) -> u.Quantity:
    """
    Spread star flux onto a 3D on-sky canvas (n_wavel, n_pix, n_pix).

    Returns a Quantity with the same units as flux_star, typically ph/(um m^2 s).
    """
    kernel_star = _circle_kernel(n_pix, idx_y_star, idx_x_star, half_pix)

    return flux_star[:, None, None] * kernel_star[None, :, :]


def generate_exoplanet_scene(
    flux_planet: u.Quantity,
    n_pix: int,
    idx_y_planet: int,
    idx_x_planet: int,
    half_pix: int,
) -> u.Quantity:
    """
    Spread exoplanet flux onto a 3D on-sky canvas (n_wavel, n_pix, n_pix).

    Returns a Quantity with the same units as flux_planet, typically ph/(um m^2 s).
    """

    kernel_planet = _circle_kernel(n_pix, idx_y_planet, idx_x_planet, half_pix)

    return flux_planet[:, None, None] * kernel_planet[None, :, :]


def generate_exozodi_scene(
    flux_exozodi: u.Quantity,
    n_pix: int,
    idx_y_exozodi: int,
    idx_x_exozodi: int,
    pix_size_arcsec: float,
    dist_pc: float,
    z_exozodiacal: float,
    ) -> u.Quantity:
    '''
    Spread exozodi flux onto a 3D on-sky canvas (n_wavel, n_pix, n_pix).
    '''

    kernel_exozodi = _exozodi_kernel(dist_pc=dist_pc, 
                                        n_pix=n_pix, 
                                        idx_y=idx_y_exozodi, 
                                        idx_x=idx_x_exozodi, 
                                        pix_size_arcsec=pix_size_arcsec, 
                                        z_exozodiacal=z_exozodiacal)

    return flux_exozodi[:, None, None] * kernel_exozodi[None, :, :]


def generate_zodiacal_scene(flux_zodiacal: u.Quantity, 
                            n_pix: int, 
                            pix_size_arcsec: float) -> u.Quantity:
    '''
    Spread zodiacal flux onto a 3D on-sky canvas (n_wavel, n_pix, n_pix).
    '''

    # kernel is just a simple screen
    kernel_zodiacal = _zodiacal_kernel(n_pix=n_pix, pix_size_arcsec=pix_size_arcsec)

    return flux_zodiacal[:, None, None] * kernel_zodiacal[None, :, :]


class AstrophysicalSources:
    """
    Calculates photon flux from astrophysical sources (incl. noise)
    """
    
    def __init__(self, config: configparser.ConfigParser, unit_converter: UnitConverter):
        """
        Initialize astrophysical noise calculator.
        
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion utility
        """

        self.config = config
        self.unit_converter = unit_converter
        self.spectra = {}
        self._load_spectra()
    

    #def _load_model_spectrum(self, source_name: str) -> None:
    #    """Load model spectrum for an exoplanet."""
    #    pass
    

    def _load_spectra(self) -> None:
        """Load spectral data for all astrophysical sources."""
        #sources_config = self.config.get("astrophysical_sources", {})

        #output_dir = Path(config['dirs']['save_s2n_data_unique_dir']).resolve()
        
        sources_section = "astrophysical_sources_library"
        if self.config.has_section(sources_section):
            for source_name in self.config.options(sources_section):

                try:
                    # Get the source name from the section
                    spectrum_file_name = self.config[sources_section][source_name]
                    self.spectra[source_name] = load_spectrum_from_file(spectrum_file_name)
                    logger.info(f"Loaded spectrum for {source_name}: {spectrum_file_name}")

                except Exception as e:
                    logger.warning(f"Did not load self-generated spectrum for {source_name}: {e}; either missing, or will need to read in from other file")
        else:
            logger.warning("No [astrophysical_sources_library] section found in config file.")

    def _calculate_flux_from_spectrum(self, source_name: str, wavelength: u.Quantity, distance_set: float = None, null: bool = False) -> u.Quantity:
        """
        Helper function to interpolate a stored spectrum and apply distance corrections (but not nulling yet!)

        Args:
            source_name: Name of the source (star, exoplanet_bb, exozodiacal, zodiacal)
            wavelength: Wavelength grid (with units)
            distance_set: Distance to the source (pc); ignored if source is zodiacal
            null: Whether to apply nulling for the star (vestigial; null is applied in the S/N calculator)

        Returns:
            Flux array with units ph / (um m^2 s)
        """
        
        spectrum = self.spectra[source_name]

        # Interpolate to the requested wavelength grid
        # (note this is not integrating over wavelength for each interpolated data point)
        interpolated_spectrum = spectrum.interpolate(wavelength)

        # Apply distance correction, if the source is not zodiacal (which is already in brightness units as seen from Earth)
        if source_name != "zodiacal":
            distance = distance_set * u.pc  # parsecs
            distance_correction = 1.0 / (4 * np.pi * distance ** 2)  # 1/r^2 law
            logging.info(f"Distance of object set to: {distance_set} pc")
        else:
            distance_correction = 1.0

        # Convert flux_unit string to astropy unit object
        flux_unit_obj = u.Unit(interpolated_spectrum.flux_unit)

        flux_incident = interpolated_spectrum.flux * flux_unit_obj
        if source_name != "zodiacal":
            flux_incident = flux_incident * distance_correction

        if flux_incident.unit.is_equivalent(u.ph / (u.um * u.m**2 * u.s)):
            logger.info(f'Flux units consistent for source: {source_name}')
        else:
            logger.warning(f'Flux units not consistent for source: {source_name}')

        return flux_incident.to(u.ph / (u.um * u.m**2 * u.s))
    

    
    def calculate_incident_flux(self, source_name: str, plot: bool = False, system_params: dict = None) -> np.ndarray:
        """
        Calculate local (at Earth) flux from an emitted spectrum at a given distance
        
        Args:
            source_name: Name of the source (star, zodiacal, exozodiacal, exoplanet_bb, exoplanet_model_10pc, exoplanet_psg)
            null: apply the nulling factor? (only applies to star target)
            system_params: system parameters dictionary (optional; use in case of planet population)
            
        Returns:
            Flux array, with units # in photons/sec/m^2/micron
        """

        incident_dict = {}

        # should star be nulled?
        null = self.config.getboolean("nulling", "null")
        logger.info(f"Nulling of star: {null}")
        
        wavelength_incident_cube_points = np.linspace(float(self.config['wavelength_range']['min']),
                               float(self.config['wavelength_range']['max']),
                               int(100)) * u.um

        if source_name in ["star", "exoplanet_bb", "exozodiacal", "zodiacal"]:

            # note distance is being set by the config file; this is just one object
            flux_incident = self._calculate_flux_from_spectrum(source_name, wavelength_incident_cube_points, distance_set=float(self.config["target"]["distance"]), null=null)

        elif source_name == "exoplanet_model_10pc":

            # check that the desired distance is 10 pc; if not, will have to update this
            if float(self.config["target"]["distance"]) != 10.0:
                logger.info(f"Distance {float(self.config['target']['distance'])} pc is not 10 pc; rescaling planet spectrum with true distance.")

            # this is a model spectrum from a file with different units, formatting
            file_name_exoplanet_model_10pc = self.config['astrophysical_sources_library']['exoplanet_model_10pc']
            df = pd.read_csv(file_name_exoplanet_model_10pc, delim_whitespace=True, names=['wavelength', 'flux', 'err_flux'])
            logger.info(f"Loaded model exoplanet spectrum for {source_name}: {file_name_exoplanet_model_10pc}")

            wavel = df['wavelength'].values * u.micron
            flux_nu_10pc = df['flux'].values * u.erg / (u.second * u.Hz * u.m**2)
            err_flux_nu_10pc = df['err_flux'].values * u.erg / (u.second * u.Hz * u.m**2)

            # convert to F_lambda
            flux_lambda_10pc = flux_nu_10pc * (const.c / wavel**2)
            flux_lambda_10pc = flux_lambda_10pc.to(u.W / (u.m**2 * u.micron))

            # convert to photon flux
            flux_photons_10pc = flux_lambda_10pc * (wavel / (const.h * const.c)) * u.ph
            flux_photons_10pc = flux_photons_10pc.to(u.ph / (u.micron * u.s * u.m**2))

            # rescale this flux (from a source at 10 pc) to the desired distance
            flux_photons = flux_photons_10pc * (10.0 / float(self.config["target"]["distance"])) ** 2

            # interpolate
            flux_incident = np.interp(x = wavelength_incident_cube_points, 
                                            xp = wavel, 
                                            fp = flux_photons)

        elif source_name in ["star_psg", "exoplanet_bb_psg", "exoplanet_psg","exozodiacal_psg"]:
            
            # read in the NASA PSG spectrum file name associated with the planets in the population
            df = pd.read_csv(self.config['target']['psg_spectrum_file_name'], names=['wavel', 'flux_total', 'flux_noise', 'flux_planet'], skiprows=15, sep=r'\s+')
            
            logger.info(f"!!! --- OVERWRITING PSG PLANET SPECTRUM FILE WITH A BLACKBODY; FIX LATER --- !!!")
            flux_incident = self._calculate_flux_from_spectrum(
                                                        source_name=source_name, 
                                                        wavelength=wavelength_incident_cube_points, 
                                                        distance_set=float(system_params['Ds']), 
                                                        null=False)


            wavel = df['wavel'].values * u.micron

            # note source is already at the desired distance; should not be rescaled as if it were at 10 pc
            flux_nu = df['flux_planet'].values * u.erg / (u.second * u.Hz * u.m**2)
            err_flux_nu = df['flux_noise'].values * u.erg / (u.second * u.Hz * u.m**2)

            # convert to F_lambda
            flux_lambda = flux_nu * (const.c / wavel**2)
            flux_lambda = flux_lambda.to(u.W / (u.m**2 * u.micron))

            flux_photons = flux_lambda * (wavel / (const.h * const.c)) * u.ph
            flux_photons = flux_photons.to(u.ph / (u.micron * u.s * u.m**2))

            # incident flux from PSG spectrum
            flux_psg_incident = np.interp(x = wavelength_incident_cube_points, 
                                            xp = wavel, 
                                            fp = flux_photons)

            print('!!!------- FLUX CONVERSION FROM PSG FILE BEING DONE INCORRECTLY; CORRECT LATER -------!!!') ## ## TODO


            ## ## TODO: MAKE SURE SCALING, UNITS ARE RIGHT
            # get BB spectrum for making a rough rescaling of the PSG spectrum
            
            flux_bb_incident = self._calculate_flux_from_spectrum(source_name="exoplanet_bb", wavelength=wavelength_incident_cube_points, null=False)
            #flux_incident_junk = self._calculate_flux_from_spectrum(source_name="exoplanet_model_10pc", wavelength=wavelength, null=False)

            # integrate the BB spectrum over the wavelength grid
            flux_incident_bb_integrated = np.trapz(y=flux_bb_incident, x=wavelength_incident_cube_points)

            # integrate the PSG spectrum over the wavelength grid
            flux_incident_psg_integrated = np.trapz(y=flux_psg_incident, x=wavelength_incident_cube_points)

            # rescale the PSG spectrum to the BB spectrum
            ratio_incident_psg_over_bb = (flux_incident_psg_integrated / flux_incident_bb_integrated).value

            flux_incident = flux_psg_incident / ratio_incident_psg_over_bb
            

        else:
            logger.warning(f"Spectrum not available for {source_name}")
            return np.array([])

        incident_dict['wavel'] = wavelength_incident_cube_points
        # units ph/um/sec * (1/pc^2) * (pc / 3.086e16 m)^2 <-- last term is for unit consistency
        # = ph/um/m^2/sec

        incident_dict['pre_screen_astro_flux_ph_sec_m2_um'] = flux_incident

        
        if plot: # pragma: no cover
            plt.clf()
            plt.figure(figsize=(8, 8)) 
            plt.plot(incident_dict['wavel'], incident_dict['pre_screen_astro_flux_ph_sec_m2_um'])
            plt.yscale('log')
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-3, 1e9]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({incident_dict['wavel'].unit})")
            plt.ylabel(f"Flux (" + str(incident_dict['pre_screen_astro_flux_ph_sec_m2_um'].unit) + ")")
            plt.title(
                format_plot_title(
                    f"Incident flux from {source_name} (at Earth, rescaled for distance {float(self.config['target']['distance'])} pc)",
                    self.config,
                )
            )
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"incident_{source_name}.png"
            #ipdb.set_trace()
            plt.tight_layout()
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux to " + file_name_plot)
        
        return incident_dict
    

    def generate_onsky_scene(self, incident_dict: dict, plot: bool = False):
        '''
        Construct the on-sky scene from the incident flux dictionary and positions of objects as set in the config file.

        INPUTS:
            incident_dict: dictionary of incident flux for each source
            plot: whether to plot the on-sky scene

        OUTPUTS:
            dict_source_layered_scene: dictionary of the on-sky scene for each source
                - star: 3D array of the on-sky scene for the star
                - planet: 3D array of the on-sky scene for the planet
        '''

        n_pix = int(self.config['onsky_scene']['n_pix']) # should be odd number to simplify centering
        pix_size_mas = float(self.config['onsky_scene']['pix_size_mas'])  # milliarcseconds
        pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
        axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
        xx_arcsec, yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')
        sky_xx_arcsec = xx_arcsec
        sky_yy_arcsec = yy_arcsec

        # point sources only for now
        ## ## TODO: ADD RESOLVED SOURCES
        '''
        for source_name in incident_dict:
            if source_name is "star":
                x_star_arcsec, y_star_arcsec = (float(v.strip()) for v in incident_dict[source_name]["pos_star_arcsec"].split(","))
            elif source_name is "exoplanet_model_10pc":
                x_planet_arcsec, y_planet_arcsec = (float(v.strip()) for v in incident_dict[source_name]["pos_star_arcsec"].split(","))
            else:
                continue
        '''
        y_star_arcsec, x_star_arcsec = parse_sky_position_arcsec_yx(
            self.config['onsky_scene']['pos_star_arcsec']
        )
        y_planet_arcsec, x_planet_arcsec = parse_sky_position_arcsec_yx(
            self.config['onsky_scene']['pos_planet_arcsec']
        )

        # make flux vectors and check consistency in units
        flux_dict = {}
        for source_name in incident_dict.keys():
            flux_dict[source_name] = incident_dict[source_name]['pre_screen_astro_flux_ph_sec_m2_um']
            flux_star = incident_dict['star']['pre_screen_astro_flux_ph_sec_m2_um'] # to check units
            if flux_dict[source_name].unit != flux_star.unit:
                raise ValueError(
                    f"{source_name} flux units differ: {flux_dict[source_name].unit} vs {flux_star.unit}"
                )
            else:
                logger.info(f"Constructing scene. {source_name} flux units are consistent: {flux_dict[source_name].unit}")

        #flux_star = incident_dict['star']['pre_screen_astro_flux_ph_sec_m2_um']   # (n_wavel,) Quantity
        #flux_planet = incident_dict['exoplanet_model_10pc']['pre_screen_astro_flux_ph_sec_m2_um'] # (n_wavel,) Quantity
        #flux_exozodi = incident_dict['exozodiacal']['pre_screen_astro_flux_ph_sec_m2_um']
        #flux_zodiacal = incident_dict['zodiacal']['pre_screen_astro_flux_ph_sec_m2_um']

        idx_y_star = (np.abs(sky_yy_arcsec[:, 0] - y_star_arcsec)).argmin()
        idx_x_star = (np.abs(sky_xx_arcsec[0, :] - x_star_arcsec)).argmin()
        idx_y_planet = (np.abs(sky_yy_arcsec[:, 0] - y_planet_arcsec)).argmin()
        idx_x_planet = (np.abs(sky_xx_arcsec[0, :] - x_planet_arcsec)).argmin()
        idx_y_exozodi = idx_y_star # exozodi is centered on the star
        idx_x_exozodi = idx_x_star

        # make star and planet small boxes (so I can see them)
        half_pix = int(self.config['onsky_scene']['half_pix'])
        dist_pc = float(self.config['target']['distance']) * u.pc
        pix_size_arcsec = (float(self.config['onsky_scene']['pix_size_mas']) / 1000.0) * u.arcsec
        z_exozodiacal = float(self.config['target']['z_exozodiacal'])
        canvas_3D_dict = {}

        for source_name in incident_dict.keys():
            if source_name == 'star':
                canvas_3D_dict[source_name] = generate_star_scene(
                    flux_dict[source_name], n_pix, idx_y_star, idx_x_star, half_pix
                )
            elif source_name == 'exoplanet_model_10pc':
                canvas_3D_dict[source_name] = generate_exoplanet_scene(
                    flux_dict[source_name], n_pix, idx_y_planet, idx_x_planet, half_pix
                )
            elif source_name == 'exozodiacal':
                canvas_3D_dict[source_name] = generate_exozodi_scene(
                    flux_dict[source_name], n_pix, idx_y_exozodi, idx_x_exozodi, pix_size_arcsec, dist_pc, z_exozodiacal
                )
            elif source_name == 'zodiacal':
                canvas_3D_dict[source_name] = generate_zodiacal_scene(
                    flux_dict[source_name], n_pix, pix_size_arcsec
                )

        # collapse the sources into a single cube (wavel, y, x)
        source_collapsed_scene_no_screen = None
        for source_array in canvas_3D_dict.values():
            if source_collapsed_scene_no_screen is None:
                source_collapsed_scene_no_screen = source_array.copy()
            else:
                source_collapsed_scene_no_screen += source_array

        dict_source_layered_scene = canvas_3D_dict # vestigial

        # Prepare list of all per-source 3D cubes
        scene_cubes = list(canvas_3D_dict.values())
        scene_names = list(canvas_3D_dict.keys())
        # The total/collapsed scene as the last plot
        if plot:  # pragma: no cover
            n_rows = 1
            n_cols = len(scene_cubes) + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
            # Compute the 2D collapsed maps and store for the summary
            for i, (source_cube, name) in enumerate(zip(scene_cubes, scene_names)):
                ax = axes[0, i]
                summed = np.sum(source_cube, axis=0)
                im = ax.imshow(
                    summed.value,
                    origin='lower',
                    norm=LogNorm(),
                    aspect='equal',
                    interpolation='none'
                )
                ax.set_title(f"{name}")
                try:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                except:
                    logger.warning(f"No colorbar for {name}")
            # Last subplot: sum across total/collapsed scene (z-scaled)
            ax = axes[0, -1]
            total_summed = np.sum(source_collapsed_scene_no_screen, axis=0)
            total_data = np.asarray(total_summed.value, dtype=float)
            vmin, vmax = ZScaleInterval().get_limits(total_data)
            im = ax.imshow(
                total_data,
                origin='lower',
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='none',
            )
            ax.set_title("Total (z-scale)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.suptitle("FYI: On-Sky Scene for Each Source (Sum Across Wavelength)", y=1.02)
            plt.subplots_adjust(top=0.85)
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"/scene_no_screen_fyi.png"
            plt.savefig(file_name_plot, bbox_inches='tight')
            logger.info(f"FYI scene plot written to {file_name_plot}")
            plt.close(fig)


        def _cube_values(q):
            '''
            Strip units (if any) for saving as FITS file
            '''
            return np.asarray(q.value if hasattr(q, "value") else q, dtype=float)

        # save the scene to FITS file
        wavel_array = incident_dict['exoplanet_model_10pc']['wavel'] ## ## TODO: GENERALIZE THIS
        file_name_fits = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"scene_no_screen.fits"
        hdul = fits.HDUList([
            fits.PrimaryHDU(_cube_values(source_collapsed_scene_no_screen)),
            fits.ImageHDU(_cube_values(dict_source_layered_scene["star"]), name="STAR"),
            fits.ImageHDU(_cube_values(dict_source_layered_scene["exoplanet_model_10pc"]), name="PLANET"),
            fits.ImageHDU(sky_yy_arcsec, name="YY_ARCSEC"),
            fits.ImageHDU(sky_xx_arcsec, name="XX_ARCSEC"),
            fits.ImageHDU(wavel_array.value, name="WAVEL_UM"),
        ])
        hdul.writeto(file_name_fits, overwrite=True)
        logger.info(f"Saved scene to {file_name_fits}")
    
        return dict_source_layered_scene