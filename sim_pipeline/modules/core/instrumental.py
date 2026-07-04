"""
Instrumental noise calculations for the modules package.

This module handles calculations of instrumental noise sources including
dark current, read noise, and other detector effects.

Sky / aperture coordinate convention (use everywhere in this repo):
  - 2D sky angle arrays: index 0 = y, index 1 = x (arcsec or rad).
  - 3D cubes (3, Ny, Nx): slice 0 = science; slice 1 = y; slice 2 = x.
  - Aperture position vectors pos_vec_m: [y_m, x_m] per aperture.
  - Config pos_*_arcsec strings: "y, x" (y first, x second).
"""

from socket import IPV6_DONTFRAG
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import logging
import ipdb
import astropy.units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import yaml
from pathlib import Path
import logging
from modules.utils.helpers import enable_plot_units



from ..data.units import UnitConverter
from ..utils.helpers import format_plot_title, compute_collecting_area_m2
from ..utils.loader import config_getboolean

logger = logging.getLogger(__name__)


class Detector:
    # Detector object that contains the illumination footprint and possibly fancier noise sources

    def __init__(self, config: Dict, num_wavel_bins: int):
        '''
        Args:
            config: Configuration dictionary
            num_wavel_bins: Number of wavelength bins
        '''

        self.side_length_pix = int(config["detector"]["size"])
        self.pitch_pix = float(config["detector"]["pitch_pix"])
        self.pix_per_wavel_bin = float(config["detector"]["pix_per_wavel_bin"])
        self.num_wavel_bins = num_wavel_bins
        self.pix_spectral_width = int(config["detector"]["pix_spectral_width"])


        self.config = config

        #self.gain = config["detector"]["gain"]
        #self.quantum_efficiency = config["detector"]["quantum_efficiency"]
        #self.photons_to_e = config["detector"]["photons_to_e"]
        #self.read_noise = config["detector"]["read_noise"]
        #self.dark_current = config["detector"]["dark_current"]

        # initialize dict to carry intrinsic instrumental terms (independent of astrophysics)
        #self.sources_instrum = {}

        # initialize dict to carry propagated astrophysicalterms (i.e., intensity levels on the detector, after instrument effects)
        #self.prop_dict = {}
        # assume wavelengths are the same for the star and planet
        #self.prop_dict['wavel'] = self.star_flux['wavel']

        # load 2D systematics (INI True/False must use config_getboolean, not bare if on strings)
        sys_section = "detector_systematics"

        def _load_systematic_map(enable_key: str, file_key: str):
            if not config_getboolean(self.config, sys_section, enable_key):
                return None
            section = self.config[sys_section]
            file_path = section[file_key] if isinstance(section, dict) else section.get(file_key)
            return fits.getdata(file_path)

        read_noise_map = _load_systematic_map("enable_read_noise_2d", "read_noise_2d_file")
        bias_map = _load_systematic_map("enable_dc_2d", "dc_2d_file")
        cosmic_rays_map = _load_systematic_map("enable_cosmic_rays_2d", "cosmic_rays_2d_file")
        hot_pixels_map = _load_systematic_map("enable_hot_pixels_2d", "hot_pixels_2d_file")

        # systematics: additive
        ## ## TODO: ARE THESE ALL ADDIITVE?
        self.systematics_additive_dict = {
            'read_noise_map': read_noise_map,
            'bias_map': bias_map,
            'cosmic_rays_map': cosmic_rays_map,
            'hot_pixels_map': hot_pixels_map
        }
        # multiplicative systematics
        self.systematics_multiplicative_dict = {}
        logging.info('Loaded detector systematics')

        return

    def footprint_spectral(self, file_name_plot: str, plot: bool = True):
        '''
        # return a boolean array of the detector, where the footprint is 1 (or a fraction, if the footprint edges do not cover a whole number of pixels)
        # file_name_plot: name of the file to save the plot to
        # plot: whether to make an FYI plot of the footprint

        # lower-left corner of the footprint
        # (keep this coordinate whole numbers for now)

        INPUTS:


        OUTPUTS:
        None; the footprint is stored in self.footprint_cube.

        '''

        starting_pixel = np.array([100,300])

        footprint_cube = np.full((self.num_wavel_bins, self.side_length_pix, self.side_length_pix), 0.0, dtype=float)

        # for each wavelength bin, make the footprint for that bin alone
        for wavel_bin_num in range(0, self.num_wavel_bins):
            # assumes horizontal spectra

            # initialize the footprint
            footprint_this = np.full((self.side_length_pix, self.side_length_pix), 0.0, dtype=float)

            # get the starting lower-left pixel for this wavelength bin
            # (note this can be a float)
            starting_pixel_this = starting_pixel + np.array([0,wavel_bin_num * self.pix_per_wavel_bin])

            # get the footprint for this wavelength bin
            pixel_ceil_start_x = int(np.ceil(starting_pixel_this[1]))
            pixel_frac_start_x = pixel_ceil_start_x-starting_pixel_this[1]
            pixel_floor_end_x = int(np.floor(starting_pixel_this[1] + self.pix_per_wavel_bin))
            pixel_frac_end_x = (starting_pixel_this[1] + self.pix_per_wavel_bin) - pixel_floor_end_x
            # where whole pixels are under the footprint, set them to 1
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_ceil_start_x):int(pixel_floor_end_x)] = 1.0

            # where partial pixels are under the footprint, set their values to the fraction of the pixel that is under the footprint
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_ceil_start_x)-1] = pixel_frac_start_x
            footprint_this[int(starting_pixel_this[0]):int(starting_pixel_this[0]+self.pix_spectral_width), int(pixel_floor_end_x)] = pixel_frac_end_x

            # add to the cube
            footprint_cube[wavel_bin_num,:,:] = footprint_this
            logging.info(f"Wavelength bin {wavel_bin_num} detector footprint is {footprint_this.sum()} pixels")

        # FYI FITS file
        # fits.writeto(f"/Users/eckhartspalding/Downloads/footprint_cube.fits", footprint_cube, overwrite=True)

        # generate the array with the spectrum only (no detector noise yet)
        footprint_sum = np.sum(footprint_cube, axis=0)

        logging.info(f"Total detector footprint is {footprint_sum.sum()} pixels")

        if plot: # pragma: no cover
            plt.clf()
            plt.title(format_plot_title("Detector spectral footprint (True)", self.config))
            norm = ImageNormalize(footprint_sum, interval=ZScaleInterval())
            plt.imshow(footprint_sum, origin='lower', cmap='gray', norm=norm)
            plt.xlabel(f"Pixel")
            plt.ylabel(f"Pixel")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of detector footprint containing all wavelength bins to {file_name_plot}")

        self.footprint_cube = footprint_cube

        return

    def convert_2d_systematics_to_1d_vector(self) -> np.ndarray:
        # convert the 2D systematics to a 1D vector, based on the footprint_cube which sets where each wavelength bin is on the detector
        # returns a 1D vector of the systematics

        # make blank canvas onto which we will add the systematics
        canvas_systematics = np.zeros((self.footprint_cube[0,:,:].shape), dtype=float)

        # read in the 2D systematics
        if len(self.systematics_additive_dict) > 0: # if there are any listed
            for key, value in self.systematics_additive_dict.items():
                if value is not None: # if there is a 2D array
                    logging.info(f"Applying {key} systematic (additive) to the detector")
                    canvas_systematics = canvas_systematics + value
        # apply the 2D multiplicative systematics to the spectrum
        if len(self.systematics_multiplicative_dict) > 0: # if there are any listed
            for key, value in self.systematics_multiplicative_dict.items():
                if value is not None:# if there is a 2D array
                    logging.info(f"Applying {key} systematic (multiplicative) to the detector")
                    canvas_systematics = canvas_systematics * value
        
        # for reference/debugging
        #self.canvas_systematics = canvas_systematics
        
        # use the footprint_cube to sum the 2D systematics within each wavelength bin
        systematics_vector_1d = np.zeros(self.num_wavel_bins)

        for wavel_bin_num in range(0, self.num_wavel_bins):
            # for a given wavelength bin, sum the systematics across the pixels corresponding to that wavelength bin
            # self.footprint_cube[wavel_bin_num,:,:] is just a boolean array, so we multiply it
            systematics_vector_1d[wavel_bin_num] = np.sum( self.footprint_cube[wavel_bin_num,:,:] * canvas_systematics )

        '''
        # apply the 2D additive systematics to the spectrum

        systematics_vector = np.zeros(self.num_wavel_bins)
        for wavel_bin_num in range(0, self.num_wavel_bins):
            systematics_vector[wavel_bin_num] = systematics_dict[wavel_bin_num].sum()
        '''
        return systematics_vector_1d


'''
@dataclass
class InstrumentalNoise:
    """
    Calculates instrumental noise sources for telescope observations.
    
    This class handles the calculation of detector noise including
    dark current, read noise, and other instrumental effects.
    """
    
    def __init__(self, config: Dict, unit_converter: UnitConverter):
        """
        Initialize instrumental noise calculator.
        
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion utility
        """
        self.config = config
        self.unit_converter = unit_converter
    
    def calculate_dark_current_electrons(self, integration_time: float) -> float:
        """
        Calculate dark current noise in electrons per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dark current noise in electrons per pixel
        """
        dark_current_rate = self.config["detector"]["dark_current"]  # e-/pixel/sec
        
        # Dark current is a Poisson process, so noise = sqrt(N)
        dark_electrons = dark_current_rate * integration_time
        dark_noise = np.sqrt(dark_electrons)
        
        return dark_noise
    
    def calculate_dark_current_adu(self, integration_time: float) -> float:
        """
        Calculate dark current noise in ADU per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dark current noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_dark_current_electrons(integration_time)
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def calculate_read_noise_electrons(self) -> float:
        """
        Calculate read noise in electrons per pixel.
        
        Returns:
            Read noise in electrons per pixel
        """
        read_noise = self.config["detector"]["read_noise"]  # e-/pixel
        
        # Read noise is typically Gaussian, so we use the value directly
        return read_noise
    
    def calculate_read_noise_adu(self) -> float:
        """
        Calculate read noise in ADU per pixel.
        
        Returns:
            Read noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_read_noise_electrons()
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def calculate_total_instrumental_noise_electrons(self, integration_time: float) -> float:
        """
        Calculate total instrumental noise in electrons per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Total instrumental noise in electrons per pixel
        """
        total_noise_squared = 0.0
        
        sources_config = self.config.get("instrumental_sources", {})
        
        # Add dark current noise
        if sources_config.get("dark_current", {}).get("enabled", True):
            dark_noise = self.calculate_dark_current_electrons(integration_time)
            total_noise_squared += dark_noise ** 2
            logger.debug(f"Dark current noise: {dark_noise:.2f} e-/pixel")
        
        # Add read noise
        if sources_config.get("read_noise", {}).get("enabled", True):
            read_noise = self.calculate_read_noise_electrons()
            total_noise_squared += read_noise ** 2
            logger.debug(f"Read noise: {read_noise:.2f} e-/pixel")
        
        # Add other instrumental noise sources here as needed
        # For example: thermal noise, quantization noise, etc.
        
        total_noise = np.sqrt(total_noise_squared)
        
        return total_noise
    
    def calculate_total_instrumental_noise_adu(self, integration_time: float) -> float:
        """
        Calculate total instrumental noise in ADU per pixel.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Total instrumental noise in ADU per pixel
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        # Calculate noise in electrons
        noise_electrons = self.calculate_total_instrumental_noise_electrons(integration_time)
        
        # Convert to ADU
        noise_adu = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return noise_adu
    
    def get_noise_breakdown_electrons(self, integration_time: float) -> Dict[str, float]:
        """
        Get breakdown of instrumental noise sources in electrons.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dictionary mapping noise source names to their contributions
        """
        breakdown = {}
        
        sources_config = self.config.get("instrumental_sources", {})
        
        # Dark current
        if sources_config.get("dark_current", {}).get("enabled", True):
            breakdown["dark_current"] = self.calculate_dark_current_electrons(integration_time)
        
        # Read noise
        if sources_config.get("read_noise", {}).get("enabled", True):
            breakdown["read_noise"] = self.calculate_read_noise_electrons()
        
        return breakdown
    
    def get_noise_breakdown_adu(self, integration_time: float) -> Dict[str, float]:
        """
        Get breakdown of instrumental noise sources in ADU.
        
        Args:
            integration_time: Integration time in seconds
            
        Returns:
            Dictionary mapping noise source names to their contributions
        """
        gain = self.config["detector"]["gain"]  # e-/ADU
        
        breakdown_electrons = self.get_noise_breakdown_electrons(integration_time)
        breakdown_adu = {}
        
        for source, noise_electrons in breakdown_electrons.items():
            breakdown_adu[source] = self.unit_converter.electrons_to_adu(noise_electrons, gain)
        
        return breakdown_adu
'''


@dataclass
class OutputChannel:
    '''
    Holds signals as detected in each output channel
    name: name of the output channel
    snr: signal-to-noise ratio of the signal in the channel
    '''
    name: str
    detector: Optional[Detector] = None
    instrum_noise: dict = field(default_factory=dict)   # instrumental terms for this channel
    astroph_signal: dict = field(default_factory=dict)   # astrophysical signals for this channel
    snr: Optional[float] = None
    spec_R: float | None = None # spectral R
    angle_deg: float | None = None # rotation angle of transmission screen
    bin_edges: np.ndarray | None = None # wavelength bins
    bin_centers: np.ndarray | None = None # wavelength bins
    bin_widths: np.ndarray | None = None # wavelength bins
    tables_by_dark_current: dict[float, QTable] = field(default_factory=dict) # stores all the data relevant to S/N calculations downstream (one entry for each value of DC)

class InstrumentDepTerms:
    # Provides the effects of the instrument (including astro flux passed through the telescope aperture)

    def __init__(self, config: Dict, unit_converter: UnitConverter, sources_astroph: dict, sources_to_include: list):
        '''
        Args:
            config: Configuration dictionary
            unit_converter: Unit conversion object
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'pre_screen_astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
            sources_to_include: List of sources to actuallyinclude in the S/N calculation (and plots of incident fluxes)
        '''

        self.config = config
        self.unit_converter = unit_converter ## ## TODO: DO I NEED THIS?
        self.sources_astroph = sources_astroph # all sources of astrophysical flux, as are incident on the instrument
        self.sources_to_include = sources_to_include

        # initialize dict to carry intrinsic instrumental terms (independent of astrophysics)
        self.sources_instrum = {}

        # initialize dict to carry propagated astrophysical terms (i.e., intensity levels on the detector, after instrument effects)
        self.prop_dict = {}
        # assume wavelengths are the same for the star and planet
        #self.prop_dict['wavel'] = self.star_flux['wavel']

        # initialize output channels
        self.output_channels = {
            name: OutputChannel(name=name)
            for name in ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark']
        }

        # for each output channel, set the detection wavelength bins (same for all channels for now)
        R = float(self.config["detector"]["spec_res"]) # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation lambda_i = lambda_{0} * (1 + 1/R)**i
        lambda_min, lambda_max = float(self.config["wavelength_range"]["min"]) * u.um, float(self.config["wavelength_range"]["max"])  * u.um
        # number of bins that fit fully in [lmin, lmax]
        n_bins = int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / R)))

        # geometric bin edges and centers
        bin_edges = lambda_min * (1.0 + 1.0 / R) ** np.arange(n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = bin_edges[1:]-bin_edges[:-1] # removed units for plotting
        for output_channel in self.output_channels.values():
            output_channel.spec_R = R
            output_channel.bin_edges = bin_edges
            output_channel.bin_centers = bin_centers
            output_channel.bin_widths = bin_widths


    def _build_base_astro_table(self, output_channel) -> QTable:
        # builds a table that consolidates astrophysical signals, before including instrumental noise
        qt = QTable()
        qt['wavel_bin_num'] = np.arange(len(output_channel.bin_centers))
        qt['wavel_bin_center'] = output_channel.bin_centers
        qt['wavel_bin_width'] = output_channel.bin_widths
        qt.meta['wavel_bin_edges'] = output_channel.bin_edges
        qt['n_pix_per_wavel_bin'] = np.sum(
            output_channel.detector.footprint_cube, axis=(1, 2)
        ) * u.pix
        for source_name in self.sources_to_include:
            if source_name not in output_channel.astroph_signal:
                continue
            sig = output_channel.astroph_signal[source_name]
            qt[f'astro_{source_name}_flux_ph_sec_um'] = sig['flux_astro_1d_interpolated_ph_sec_um']
            qt[f'astro_{source_name}_flux_ph_sec_wavel_bin'] = sig['flux_astro_1d_interpolated_ph_sec_wavel_bin']
            qt[f'astro_{source_name}_flux_ph_sec_pixel'] = sig['flux_astro_1d_interpolated_ph_sec_pixel']
        return qt



    def calculate_instrinsic_instrumental_noise(self):
        # calculate intrinsic instrumental noise, and update self.sources_instrum

        gain = float(self.config["detector"]["gain"]) * u.electron / u.adu  # e-/ADU

        #########################################################################################################################
        # read noise
        # e-/pix rms
        #self.instrum_dict['read_noise_e_rms'] = float(self.config["detector"]["read_noise"])
        # e-/pix rms -> ADU rms
        logging.info(f'Finding instrumental noise sources...')
        #read_noise_e_rms = float(self.config["detector"]["read_noise"]) * u.electron / u.pix

        read_noise_str = self.config["detector"]["read_noise"]
        read_noise_e_rms = np.fromstring(read_noise_str, sep=',') * u.electron / u.pix # sep in case it's an array
        self.sources_instrum['read_noise_e_pix-1'] = read_noise_e_rms
        logging.info(f'Read noise is {read_noise_e_rms} rms')
        #self.sources_instrum['read_noise_adu'] = read_noise_e_rms / gain
        #read_noise_adu_rms = self.sources_instrum['read_noise_adu']
        #logging.info(f'Read noise is {read_noise_adu_rms} rms')

        #########################################################################################################################
        # dark current rate 
        # e/pix/sec
        dark_current_str = self.config["detector"]["dark_current"]
        if ',' in dark_current_str:
            parts = [float(x.strip()) for x in dark_current_str.split(',')]
            dark_current_rate_e_pix_sec = np.arange(parts[0], parts[1], parts[2]) * u.electron / (u.pix * u.second)
        else:
            dark_current_rate_e_pix_sec = np.fromstring(dark_current_str, sep=',') * u.electron / (u.pix * u.second) # in case it's an array confirming to (start, stop, step)
        #dark_current_rate_e_pix_sec = np.fromstring(dark_current_str, sep=',') * u.electron / (u.pix * u.second) # in case it's an array

        logging.info(f'Dark current array is {dark_current_rate_e_pix_sec} e-/pix/sec')

        # total dark current in e-, based on integration time
        # e/pix/sec -> e/pix
        integration_time_per_frame = float(self.config["observation"]["t_int_frame"]) * u.second  # seconds
        self.sources_instrum['dark_current_e_pix-1_sec-1'] = dark_current_rate_e_pix_sec
        self.sources_instrum['dark_current_e_pix-1'] = dark_current_rate_e_pix_sec * integration_time_per_frame

        # total dark current in ADU
        # e/pix -> ADU/pix
        #self.sources_instrum['dark_current_adu_pix-1'] = self.sources_instrum['dark_current_e_pix-1'] / gain

        # assign all these noise terms to the output channels
        for output_name, output_channel in self.output_channels.items():
            output_channel.instrum_noise['dark_current_e_pix-1_sec-1'] = dark_current_rate_e_pix_sec
            output_channel.instrum_noise['read_noise_e_pix-1'] = read_noise_e_rms

        return 


    def combine_astro_and_instrum_signals(self):
        # combines astrophysical signals and instrumental noise

        t_frame = float(self.config['observation']['t_int_frame']) * u.second
        read_noise_scalar = self.sources_instrum['read_noise_e_pix-1'] # just one value here
        dc_rates = self.sources_instrum['dark_current_e_pix-1_sec-1']  # (n_dc,)
        gain = float(self.config["detector"]["gain"]) * u.electron / u.adu
        e_per_ph = float(self.config["detector"]["e_per_ph"]) * u.electron / u.ph

        # loop over output channels and make a set of tables (one for each value of dark current)
        for output_name, output_channel in self.output_channels.items():

            base = self._build_base_astro_table(output_channel) # build table that includes astrophysical signals
            n_bins = len(output_channel.bin_centers)
            tables_by_dc = {}

            # loop over dark current values
            for dc_rate_this in dc_rates: 
                
                logging.info(f'Combining astrophysical signals and instrumental noise for output {output_name} at dark current {dc_rate_this}')
                qt = QTable(base.copy())
                # metadata (scalar, for bookkeeping)
                #qt.meta['dark_current_rate_e_pix_sec'] = dc_rate
                # instrumental columns: constant across wavelength bins at this DC
                #qt['instrum_dark_current_e_pix_sec'] = dc_rate * np.ones(n_bins)
                #qt['instrum_dark_current_e_pix'] = np.full(n_bins, dc_rate * t_frame)
                #qt['instrum_read_noise_e_pix'] = read_noise * np.ones(n_bins)
                qt['t_int_frame'] = t_frame # integration time of one frame ## ## TODO: enable multiple reads
                qt['qe'] = float(self.config['detector']['quantum_efficiency'])
                tables_by_dc[float(dc_rate_this.value)] = qt

            output_channel.tables_by_dark_current_orig = tables_by_dc # _orig meaning that we have not modified the units here

            ##########################################
            # begin debug: summary function
            for name, ch in self.output_channels.items():
                print(f"\n=== {name} ===")
                print(f"  angle_deg: {ch.angle_deg}")
                print(f"  spec_R: {ch.spec_R}")
                print(f"  n_bins: {len(ch.bin_centers) if ch.bin_centers is not None else None}")
                print(f"  instrum_noise keys: {list(ch.instrum_noise.keys())}")
                print(f"  astroph_signal keys: {list(ch.astroph_signal.keys())}")
                print(f"  tables_by_dark_current DC rates: {list(ch.tables_by_dark_current.keys())}")
                for dc, tbl in ch.tables_by_dark_current.items():
                    print(f"    dc={dc:g}: {len(tbl)} rows, columns:")
                    for col in tbl.colnames:
                        print(f"      - {col}")
            # end debug: summary function
            ##########################################

        # loop over each of the tables and make a new table that keeps some of the columns for bookkeeping,
        # and then multiplies others by the appropriate factor to get the total signal in electrons
        for output_name, output_channel in self.output_channels.items():

            for dc_rate, table in output_channel.tables_by_dark_current_orig.items():

                # one table of signals for a permutation of 
                # 1) output
                # 2) dark current
                # 3) rotation angle
                final_table = QTable()
                final_table['wavel_bin_num'] = table['wavel_bin_num']
                final_table['wavel_bin_center'] = table['wavel_bin_center']
                final_table['wavel_bin_width'] = table['wavel_bin_width']
                final_table['n_pix_per_wavel_bin'] = table['n_pix_per_wavel_bin']
                # total dark current within the wavelength bin for the entire integration

                # 'dark current' is an additive pedestal value, not an rms term
                # 'dark current rms' is what we calculate here by taking the square root, so that we can propagate the noise as if dark-subtraction were already being carried out
                # dark current 'pedestal' to make clear that this is not an rms term, but a constant offset
                # total dark current for wavelength bin: multiply rate of single pixel by sqrt(dc_rate × N_pix × t_frame) for N_pix in the wavelength bin
                final_table['instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot'] = np.sqrt((dc_rate * u.electron/u.pix) * table['n_pix_per_wavel_bin'] * t_frame).value*u.electron / gain 
                # total read noise within the wavelength bin for the entire integration
                # the .value*u.pix is to avoid resulting in sqrt(pix)
                final_table['instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot'] = read_noise_scalar * np.sqrt(table['n_pix_per_wavel_bin']).value*u.pix / gain

                # loop over astrophysical sources:
                for source_name in self.sources_to_include:
                    astro_sig = output_channel.astroph_signal[source_name]
                    final_table[f'astro_{source_name}_flux_adu_sec_for_wavel_bin_and_integration_tot'] = astro_sig['flux_astro_1d_interpolated_ph_sec_pixel'] * table['n_pix_per_wavel_bin'] * t_frame * (e_per_ph) * (1./gain)

                final_table.meta.update(table.meta)

                # store the final table for this permutation of output, dark current, and rotation angle
                output_channel.tables_by_dark_current[float(dc_rate)] = final_table

                # plot of final signal in the detector
                if True:  # pragma: no cover
                    wavel_bin_center = final_table['wavel_bin_center']
                    wavel_bin_width = final_table['wavel_bin_width']
                    wavel_bin_edges = output_channel.bin_edges
                    fig, ax = plt.subplots(figsize=(10, 5))
                    debug_cols = [
                        'instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot',
                        'instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot',
                    ]
                    for col_name in debug_cols:
                        y_col = final_table[col_name]
                        y_vals = y_col.value if hasattr(y_col, "value") else y_col
                        ax.stairs(y_vals, edges=wavel_bin_edges.value if hasattr(wavel_bin_edges, "value") else wavel_bin_edges, label=col_name)
                        if hasattr(y_col, "unit"):
                            y_unit = y_col.unit
                    for source_name in self.sources_to_include:
                        col_name = f'astro_{source_name}_flux_adu_sec_for_wavel_bin_and_integration_tot'
                        if col_name in final_table.colnames:
                            y_col = final_table[col_name]
                            y_vals = y_col.value if hasattr(y_col, "value") else y_col
                            ax.stairs(y_vals, edges=wavel_bin_edges.value if hasattr(wavel_bin_edges, "value") else wavel_bin_edges, label=col_name)
                            if hasattr(y_col, "unit"):
                                y_unit = y_col.unit
                    ax.set_xlim(4.0, 18.5)
                    ax.set_title(f"Debug final_table: {output_name}, dc={dc_rate:.3f} e/pix/s")
                    ax.set_xlabel(f"Wavelength bin center ({wavel_bin_center.unit})")
                    ax.set_ylabel(f"Flux ({y_unit})")
                    ax.set_yscale('log')
                    ax.legend(fontsize=8, loc='best')
                    fig.tight_layout()
                    file_name_plot = (
                        str(self.config['dirs']['save_s2n_data_unique_dir'])
                        + f"debug_final_table_{output_name}_dc_{dc_rate:.3f}.png"
                    )
                    #plt.show()
                    fig.savefig(file_name_plot)
                    plt.close(fig)
                    logging.info(f"Saved plot of binned fluxes from output {output_name} at dark current {dc_rate:.3f} e/pix/s to {file_name_plot}")
         

    def generate_instrument_transmission(self, wavel_m: float = 11e-6, override_stellar_mask = False, normalize: bool = True, plot: bool = False):
        # phi_dc_vec_rad, theta_vec_2d_asec, 
        # instrument transmission respose over the sky (R_theta_vec,Dannert 2025 Eqn. B12, ignoring polarization for now)

        '''
        INPUTS:
        # wavel_m (float): Wavelength in meters (e.g., 1e-6).
        # normalize (bool): If True, normalize transmission to unity maximum. If False, max transmission is N for N identical apertures.
        # plot (bool): If True, generate plots and write FITS cubes.

        OUTPUT:
        # Returns:
        #   transmission_instrument_response: np.ndarray of shape (6, Ny, Nx)
        #     [0]: transmission, bright output 1
        #     [1]: transmission, bright output 2
        #     [2]: transmission, dark output 3
        #     [3]: transmission, dark output 4
        #     [4]: y-coordinates on sky [arcsec]
        #     [5]: x-coordinates on sky [arcsec]
        '''

        # read in array parameters from config file
        aperture_array_definition_file_name = self.config["telescope"]["aperture_array_config_file_name"]
        with open(aperture_array_definition_file_name, 'r') as file:
            aperture_array_definition = yaml.safe_load(file)

        # construct vectors
        A_vec = [] # amplitudes
        phi_dc_vec_rad = [] # relative phase offests of each arm (one vector per aperture)
        pos_vec_m = []  # [y_m, x_m] per aperture
        phase_vector_rad_array = [] # phase vector for each output
        for aperture in aperture_array_definition['apertures']:
            A_vec.append(aperture['amplitude'])
            pos_vec_m.append([aperture['y_m'], aperture['x_m']])
        for output in aperture_array_definition['outputs']:
            phase_vector_deg = output['phase_vector_deg']
            phase_vector_rad = np.deg2rad(phase_vector_deg)
            phase_vector_rad_array.append(phase_vector_rad)
        A_vec = np.array(A_vec)
        phi_dc_vec_rad = np.array(phi_dc_vec_rad)
        pos_vec_m = np.array(pos_vec_m)

        n_pix = int(self.config['onsky_scene']['n_pix'])
        pix_size_mas = float(self.config['onsky_scene']['pix_size_mas'])  # milliarcseconds
        pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
        axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
        xx_arcsec, yy_arcsec = np.meshgrid(axis_arcsec, axis_arcsec, indexing='xy')
        sky_xx_arcsec = xx_arcsec
        sky_yy_arcsec = yy_arcsec
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        theta_vec_rad_array = np.zeros((2, n_pix, n_pix), dtype=float)
        theta_vec_rad_array[0] = sky_yy_arcsec * arcsec_to_rad  # θ_y [rad]
        theta_vec_rad_array[1] = sky_xx_arcsec * arcsec_to_rad  # θ_x [rad]
        
        # Calculate total number of baselines (unique pairs of apertures)
        N_apertures = len(aperture_array_definition['apertures'])
        logging.info(f'Number of apertures: {N_apertures}')
        # For N apertures, number of unique baselines = N*(N-1)/2
        N_baselines = N_apertures * (N_apertures - 1) // 2
        logging.info(f'Total number of baselines: {N_baselines}')

        #if incl_comp_transmission:
        #    cube_canvas = np.zeros((N_baselines+3, N, N))

        cube_canvas = np.zeros((3, n_pix, n_pix))

        # dict to hold bright and dark outputs
        output_all_responses = {}

        # Sum over all pairs of apertures (j, k) where j < k
        def R_m(phase_vector_rad: np.ndarray, wavel_m: float):
            # response of output m
            # N_apertures: number of apertures N
            # phase_vector_deg: phase vector for output m (total number of outputs is not nec. same as apertures N)
            # return: response of output m

            # convert phase vector to radians
            #phase_vector_rad = np.deg2rad(phase_vector_deg)
            R_theta_vec = np.zeros(np.shape(theta_vec_rad_array[0]))  # shape: (Ny, Nx)

            # sum over all baselines (i.e., over apertures over apertures)
            for j in range(N_apertures):
                for k in range(N_apertures):

                    # Differential phase between apertures j and k [rad]
                    del_phi_dc_jk_rad = phase_vector_rad[k] - phase_vector_rad[j]
                    
                    # Baseline from aperture j to aperture k [m]; del_x_jk[0]=Δy, del_x_jk[1]=Δx
                    del_x_jk = pos_vec_m[k] - pos_vec_m[j]

                    # Compute phase term for all sky positions at once using broadcasting
                    # theta_vec_rad_array has shape (2, Ny, Nx), del_x_jk has shape (2,)
                    # We want to compute dot(del_x_jk, theta_vec_rad_array) for all positions
                    # This gives shape (Ny, Nx)
                    phase_term = (2 * np.pi / wavel_m) * (
                        del_x_jk[0] * theta_vec_rad_array[0] +  # Δy · θ_y
                        del_x_jk[1] * theta_vec_rad_array[1]    # Δx · θ_x
                    )
                    
                    # Use cosine addition formula: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                    # This is more efficient than computing cos and sin separately
                    # Eqn. B12 in Dannert 2025
                    # Eqn. 3 in Lay 2004
                    # phase_term: 2pi/lambda * b dot theta (in some notations)
                    # del_phi_dc_jk_rad: phase offset between apertures j and k [rad]
                    response_jk = A_vec[j] * A_vec[k] * np.cos(del_phi_dc_jk_rad + phase_term)
                    
                    # Add contribution from this pair to the total response
                    '''
                    if plot:
                        plt.clf()
                        plt.imshow(response_jk)
                        plt.title(f'Baseline {j}-{k}')
                        plt.colorbar()
                        plt.show()
                    '''
                    
                    '''
                    # if incl_comp_transmission, add this as a separate slice
                    if incl_comp_transmission:
                    '''

                    # Add contribution from this pair to the total response
                    R_theta_vec += response_jk        

            # cube_canvas[0,:,:] = R_theta_vec
            cube_canvas[0, :, :] = R_theta_vec
            cube_canvas[1, :, :] = sky_yy_arcsec  # y [arcsec]
            cube_canvas[2, :, :] = sky_xx_arcsec  # x [arcsec]

            # conceptual point here! this response to photons is real, not complex! See Lay Eqn. (3): it's the rr*
            complex_instrument_response = cube_canvas

            # now for the actual transmission
            transmission_instrument_response = np.zeros(np.shape(cube_canvas))
            #transmission_instrument_response[0,:,:] = np.abs(complex_instrument_response[0,:,:])**2 # on-sky transmission
            transmission_instrument_response[0,:,:] = R_theta_vec # on-sky transmission

            #transmission_instrument_response[0,:,:] /= np.max(transmission_instrument_response[0,:,:]) # normalize (TODO: is this right?)
            transmission_instrument_response[1:3,:,:] = cube_canvas[1:3,:,:] # replicate coordinates

            # amplitude can be >1 due to addition of aperture amplitudes
            if normalize: 
                max_field_amplitude = 0
                # find the total field amplitude of the apertures, then square for transmission
                for aperture in aperture_array_definition['apertures']:
                    field_amplitude = aperture['amplitude']
                    max_field_amplitude += field_amplitude
                max_response = max_field_amplitude**2
                transmission_instrument_response /= max_response
                logging.info(f'Normalized transmission instrument response to unity, based on way bookkeeping is done downstream')

            # a small override mask is put over the star for now to avoid geometrical leakage ## ## TODO: remove this once the geometry is properly implemented
            if override_stellar_mask:
                nulling_factor = float(self.config['nulling']['nulling_factor'])
                logging.info(f'Star is manually being nulled to {nulling_factor}')
                # mask the central NxN pixels
                N_mask = int(2) + int(2) * int(self.config['onsky_scene']['half_pix']) # extra 2 to make sure we cover the resolved star
                transmission_instrument_response[0, 
                                                transmission_instrument_response.shape[1]//2-int(0.5*N_mask):transmission_instrument_response.shape[1]//2+int(0.5*N_mask), 
                                                transmission_instrument_response.shape[2]//2-int(0.5*N_mask):transmission_instrument_response.shape[2]//2+int(0.5*N_mask)
                                                ] = nulling_factor

            return transmission_instrument_response


        for output in aperture_array_definition['outputs']:
            phase_vector_rad = np.deg2rad(output['phase_vector_deg'])
            transmission_instrument_response = R_m(phase_vector_rad=phase_vector_rad, wavel_m=wavel_m)
            output_all_responses[output['name']] = transmission_instrument_response


        '''
        # check: double Bracewell should look like HOSTS
        if plot:
            plt.clf()

            ipdb.set_trace()

            arcsec_to_rad = np.pi / (180.0 * 3600.0)

            y_sky_asec = transmission_instrument_response[1, :, :]  # y at each pixel
            x_sky_asec = transmission_instrument_response[2, :, :]  # x at each pixel
            x_sky_rad = x_sky_asec * arcsec_to_rad
            y_sky_rad = y_sky_asec * arcsec_to_rad
            #y_sky_asec = y_sky_rad * 206265
            #x_sky_asec = x_sky_rad * 206265

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            extent = [
                x_sky_asec.min(), x_sky_asec.max(),   # left, right
                y_sky_asec.min(), y_sky_asec.max(),   # bottom, top
            ]

            im0 = axes[0].imshow(
                transmission_instrument_response[0, :, :],
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            #axes[0].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            #axes[0].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            axes[0].set_xlabel("x [arcsec]")
            axes[0].set_ylabel("y [arcsec]")
            axes[0].set_title("Net on-sky transmission")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # make what the HOSTS screen should look like
            hosts_trans = np.power( np.sin(np.pi * x_sky_rad * 14.4 / 11e-6), 2 )

            im1 = axes[1].imshow(
                hosts_trans,
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            #axes[1].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            #axes[1].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
            axes[1].set_xlabel("x [arcsec]")
            axes[1].set_ylabel("y [arcsec]")
            axes[1].set_title("HOSTS transmission (l/B=0.158asec)")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            ipdb.set_trace()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"transmission_instrument_response.png"
            plt.savefig(file_name_plot)
            logging.info(f"Saved plot of transmission instrument response to {file_name_plot}")
            plt.close(fig)

            save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])
            transmission_cube_3d = np.stack(
                (transmission_instrument_response[0, :, :], y_sky_asec, x_sky_asec),
                axis=0,
            )
            hosts_cube_3d = np.stack((hosts_trans, y_sky_asec, x_sky_asec), axis=0)
            fits.writeto(
                save_dir + "transmission_instrument_response_cube.fits",
                transmission_cube_3d,
                overwrite=True,
            )
            fits.writeto(
                save_dir + "hosts_transmission_cube.fits",
                hosts_cube_3d,
                overwrite=True,
            )
            logging.info(
                f"Saved 3D FITS cubes: {save_dir}transmission_instrument_response_cube.fits, "
                f"{save_dir}hosts_transmission_cube.fits"
            )
        '''

        # save all responses to FITS files
        # output_all_responses contains output_1_bright, output_2_bright, output_3_dark, output_4_dark
        for output_name, transmission_instrument_response in output_all_responses.items():
            save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])
            fits.writeto(
                save_dir + f"transmission_instrument_response_{output_name}.fits",
                transmission_instrument_response,
                overwrite=True,
            )
            logging.info(f"Saved transmission instrument response for {output_name} to {save_dir}transmission_instrument_response_{output_name}.fits")
        # save the differential dark
        differential_dark = output_all_responses['output_3_dark'] - output_all_responses['output_4_dark']
        fits.writeto(
            save_dir + f"differential_dark.fits",
            differential_dark,
            overwrite=True,
        )
        logging.info(f"Saved differential dark to {save_dir}differential_dark.fits")

        # arrange outputs into a cube, shape (4, n_pix, n_pix), slices order 0 = output_1_bright, 1 = output_2_bright, 2 = output_3_dark, 3 = output_4_dark
        # for-loop to preserve order
        keys_ordered = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark', 'yy', 'xx']
        transmission_instrument_response_cube = np.zeros((6, n_pix, n_pix))
        for t in range(4):
            transmission_instrument_response_cube[t, :, :] = output_all_responses[keys_ordered[t]][0, :, :] # screens
        transmission_instrument_response_cube[4, :, :] = output_all_responses['output_1_bright'][1, :, :] # y vals
        transmission_instrument_response_cube[5, :, :] = output_all_responses['output_1_bright'][2, :, :] # x vals

        return transmission_instrument_response_cube


    def disperse_astro_signals_on_detector(self, plot: bool = False):
        '''
        Disperse signals on the detector, using the detector object
        '''
        ## ## TODO: allow different dispersion laws for each output channel

        # disperse in each channel
        for output_channel in self.output_channels.values():
            if output_channel.detector is None:
                output_channel.detector = Detector(config=self.config, num_wavel_bins=len(output_channel.bin_centers))
            logging.info(f"Dispersing signals on detector for {output_channel.name}")
            #output_channel.detector.disperse_signals(output_channel.signal_by_source)

            # generate the spectral footprint (shape (n_bins, n_pix, n_pix)) and tack it to output_channel.detector.footprint_cube
            output_channel.detector.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + f'footprint_bool_{output_channel.name}.png', plot=True)

            # dump the photons from each source in each wavelength bin into the output_channel.detector.footprint_cube
            # integration time for 1 frame
            #t_int = float(self.config["observation"]["t_int_obs_total"]) * u.second
            n_int_this_angle = int(self.config["observation"]["N_int_per_angle"]) # number of frames for each angle position ## ## TODO: Make this be able to be != 1
            logging.info(f'Number of frames at this angle position: {n_int_this_angle:d}')

            # the number of pixels for each wavelength bin
            # (in practice the number of pixels is the same for all wavelength bins, but this can be updated later if the dispersion is not constant)
            #n_pix_array_reshaped = np.tile( np.sum(footprint_spec_cube[0,:,:]) * np.ones(len(wavel_bin_centers)), (len(D_tot), 1) ) * u.pix # shape (N_dark_current, N_wavel)
            n_pix_wavel_bin_array = u.Quantity([]) * u.pix

            # loop over each wavelength bin
            '''
            for wavel_bin_num in range(0, len(output_channel.bin_centers)):
                for source_name, source_val in self.sources_astroph.items():

                    # put all the astrophysical photons in this wavelength bin 
                    n_pix_in_wavel_bin_this = np.sum(output_channel.detector.footprint_cube[wavel_bin_num, :, :]) * u.pix # number of pixels in this wavelength bin
                    n_pix_wavel_bin_array = u.Quantity(np.append(n_pix_wavel_bin_array, n_pix_in_wavel_bin_this))
            '''
            # tack on to the output_channel.detector.n_pix_array

            footprint_cube = output_channel.detector.footprint_cube  # (n_bins, ny, nx)
            footprint_pixel_count = np.sum(footprint_cube, axis=(1, 2))
            n_pix_per_wavel_bin = (
                footprint_pixel_count.to(u.pix)
                if hasattr(footprint_pixel_count, "unit")
                else footprint_pixel_count * u.pix
            )

            # dump the astrophysical photons from each source in each wavelength bin into the output_channel.detector.footprint_cube
            for source_name, source_val in self.sources_astroph.items():

                # flatten the 3D flux cube into a 1D array (spatial information is lost)
                flux_astro_1d_ph_sec_um = np.sum(self.prop_dict[source_name]['flux_cube_post_screen_post_aperture_ph_sec_um'][output_channel.name], axis=(1,2))
                # interpolate the flux to fit the wavelength bins on the detector
                flux_unit = (
                    flux_astro_1d_ph_sec_um.unit
                    if hasattr(flux_astro_1d_ph_sec_um, "unit")
                    else u.ph / (u.um * u.s)
                )
                wavel_bins = output_channel.bin_centers
                wavel_pts = self.prop_dict[source_name]["wavel"]
                flux_astro_1d_interpolated_ph_sec_um = (
                    np.interp(
                        x=wavel_bins.value if hasattr(wavel_bins, "value") else wavel_bins,
                        xp=wavel_pts.value if hasattr(wavel_pts, "value") else wavel_pts,
                        fp=flux_astro_1d_ph_sec_um.value if hasattr(flux_astro_1d_ph_sec_um, "value") else flux_astro_1d_ph_sec_um,
                    )
                    * flux_unit
                )
                flux_astro_1d_interpolated_ph_sec_wavel_bin = flux_astro_1d_interpolated_ph_sec_um * output_channel.bin_widths
                flux_astro_1d_interpolated_ph_sec_pixel = flux_astro_1d_interpolated_ph_sec_wavel_bin / n_pix_per_wavel_bin

                # multiply by the detector QE
                flux_astro_1d_interpolated_ph_sec_pixel *= float(self.config['detector']['quantum_efficiency'])
                flux_astro_1d_interpolated_ph_sec_wavel_bin *= float(self.config['detector']['quantum_efficiency'])
                flux_astro_1d_interpolated_ph_sec_um *= float(self.config['detector']['quantum_efficiency'])

                #output_channel.signal_by_source = {}
                output_channel.astroph_signal[source_name] = {
                    'wavel': output_channel.bin_centers,           # (n_bins,) Quantity
                    'flux_astro_1d_interpolated_ph_sec_um': flux_astro_1d_interpolated_ph_sec_um.decompose(),
                    'flux_astro_1d_interpolated_ph_sec_wavel_bin': flux_astro_1d_interpolated_ph_sec_wavel_bin.decompose(),
                    'flux_astro_1d_interpolated_ph_sec_pixel': flux_astro_1d_interpolated_ph_sec_pixel.decompose(),
                    'n_pix_per_wavel_bin': n_pix_per_wavel_bin.decompose(),                # (n_bins,) pix — from footprint
                }

        return


    def chop_signal(self, plot: bool = False):

        self.post_chop_tables_by_dark_current = {}
        for dc_rate, t3 in self.output_channels['output_3_dark'].tables_by_dark_current.items():
            t1 = self.output_channels['output_1_bright'].tables_by_dark_current[dc_rate]
            t2 = self.output_channels['output_2_bright'].tables_by_dark_current[dc_rate]
            t4 = self.output_channels['output_4_dark'].tables_by_dark_current[dc_rate]
            chopped = QTable()

            # copy wavelength metadata once
            for col in ('wavel_bin_num', 'wavel_bin_center', 'wavel_bin_width', 'n_pix_per_wavel_bin'):
                chopped[col] = t3[col]

            # keep outputs, but add the chopped signal
            # consolidate signals from dark outputs, and the chopped signal
            for col in t3.colnames:
                if col.startswith(('astro_')):
                    chopped[f'output_1_bright_{col}'] = t1[col]
                    chopped[f'output_2_bright_{col}'] = t2[col]
                    chopped[f'output_3_dark_{col}'] = t3[col]
                    chopped[f'output_4_dark_{col}'] = t4[col]
                    chopped[f'chopped_{col}'] = t3[col] - t4[col]
                if col.startswith(('instrum_')):
                    # copy over instrumental terms from the dark outputs
                    chopped[f'instrum_output_3_dark_{col}'] = t3[col]
                    chopped[f'instrum_output_4_dark_{col}'] = t4[col]
                    if 'dark_current' in col:
                        chopped[f'chopped_{col}'] = np.sqrt(t3[col]**2 + t4[col]**2) ## ## TODO: MAKE SURE THIS IS CORRECT
                    if 'read_noise' in col:
                        chopped[f'chopped_{col}'] = np.sqrt(t3[col]**2 + t4[col]**2) ## ## TODO: MAKE SURE THIS IS CORRECT

            chopped.meta.update(t3.meta)

            self.post_chop_tables_by_dark_current[dc_rate] = chopped


    def pass_through_transmission_screens(self, fyi_angle, source_dict_pre_screen: dict, transmission_screens: np.ndarray, plot: bool = False):
        '''
        Pass each astrophysical source through the transmission screens, and update prop_dict with the propagated terms
        photons/sec/m^2 -> photons/sec/m^2

        INPUTS:
            fyi_angle (float): angle of the transmission screen (for plotting strings only)
            source_cube_no_screen (dict of Quantities): on-sky scene before transmission screen; for each key (astro source), value (Quantity array) has shape (n_wavel, n_pix, n_pix)
            transmission_screen (np.ndarray): transmission screen, shape (n_pix, n_pix)
            plot (bool): whether to plot the scene
        '''        

        transmission_screen_order = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark'] ## ## TODO: insert check to ensure always consistent
        transmission_screens = transmission_screens[0:4,:,:] # just keep the transmission slices for now ## ## TODO: include the yy and xx slices as a check somehow

        # put all the post-screen fluxes (for each source and from each channel) into a single dict
        # first check that transmission screens add up to one (energy conservation)
        net_transmission_screen = np.sum(transmission_screens, axis=0)
        source_dict_post_screen = {}
        for source_name, source_val in source_dict_pre_screen.items():
            source_dict_post_screen[source_name] = {}

            for transmission_screen_name in transmission_screen_order:
                source_dict_post_screen[source_name][transmission_screen_name] = source_val * transmission_screens[transmission_screen_order.index(transmission_screen_name), :, :]
                # collapse the sources into a single 3D array (wavel, x, y), for plotting
                # source_dict_post_screen[source_name][transmission_screen_name + '_collapsed'] = np.sum(source_dict_post_screen[source_name][transmission_screen_name], axis=(1,2))
        # there should be a cube for each output (4 cubes total)

        # collapse the sources into a single 3D array (wavel, x, y), for plotting
        # there should be a cube for each output (4 cubes total)
        #for transmission_screen_name in transmission_screen_order:

        #source_cube_post_screen = np.stack([source_dict_post_screen[source_name] for source_name in source_dict_post_screen.keys()], axis=0)
        #source_collapsed_cube_post_screen_sum = np.sum(source_cube_post_screen, axis=0)

        # integrate over 2D sky to get total flux from each source
        # update the sources
        source_integrated_dict_post_screen = {}
        for source_name, source_val in source_dict_post_screen.items():
            # source_val has 4 different screens, so integrate them separately
            self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'] = {} # will contain flux corresponding to each screen
            self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_um'] = {} # will contain flux cube corresponding to each screen
            test_flux_1 = 0 # to check flux conservation
            test_flux_2 = 0 # to check flux conservation
            for transmission_screen_name in transmission_screen_order:
                source_val_integrated = np.sum(source_val[transmission_screen_name], axis=(1,2))
                self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'][transmission_screen_name] = source_val_integrated
                self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_um'][transmission_screen_name] = source_val[transmission_screen_name]
                logging.info(f'Flux of {source_name} passed through transmission screen {transmission_screen_name}')

                # to check flux conservation, add up all the light transmitted through each screen
                test_flux_1 += source_val_integrated
                test_flux_2 += np.sum(source_val[transmission_screen_name], axis=(1,2))
            # if total flux after transmission is same as the input
            if np.logical_or(
                np.round(test_flux_1, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1),
                np.round(test_flux_2, 1) != np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1)
            ):
                logging.error(f'Flux conservation check failed for {source_name} at angle {int(fyi_angle)}')
                ipdb.set_trace()

            ipdb.set_trace()
            if plot: # pragma: no cover

                # flux vs wavelength post-screen, separated by screen
                plt.clf()
                plt.figure(figsize=(12, 4))
                for transmission_screen_name in transmission_screen_order:
                    plt.plot(self.sources_astroph[source_name]['wavel'], self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'][transmission_screen_name], label=transmission_screen_name)
                plt.plot(self.sources_astroph[source_name]['wavel'], self.sources_astroph[source_name]['pre_screen_astro_flux_ph_sec_m2_um'], label='pre-screen')
                plt.xlim([4.,18.])
                plt.legend()
                plt.title(f'Flux of {source_name} passed through transmission screens')
                plt.xlabel('Wavelength')
                plt.ylabel('Flux (ph/s/m^2/um)')
                file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"flux_of_{source_name}_passed_through_transmission_screens_angle_{int(fyi_angle)}.png"
           
                plt.savefig(file_name_plot)
                logging.info(f"Saved plot of flux of {source_name} passed through transmission screens at angle {int(fyi_angle)}: {file_name_plot}")
                plt.close()

                # source_val_integrated = np.sum(source_val, axis=(1,2))
                # self.sources_astroph[source_name]['flux_integrated_post_screen_ph_sec_m2_um'] = source_val_integrated
                # self.sources_astroph[source_name]['flux_cube_post_screen_ph_sec_um'] = source_dict_post_screen[source_name]
                # logging.info(f'Flux of {source_name} passed through transmission screen')
            
                # if name is right and units are right
                '''
                if ('pre_screen_astro_flux_ph_sec_m2_um' in source_val) and (source_val['pre_screen_astro_flux_ph_sec_m2_um'].unit == u.ph / (u.um * u.m**2 * u.s)):
                    dict_this = {source_name: {
                        'wavel': source_val['wavel'], 
                        'scene_2D_no_screen': scene_no_screen, 
                        'scene_2D_with_screen': scene_with_screen,
                        'source_cube_no_screen': source_cube_no_screen,
                        'source_cube_with_screen': source_cube_with_screen,
                        'transmission_screen_2D': screen_transmission_ersatz,
                        'flux_pre_screen_ph_sec_m2_um': source_val['pre_screen_astro_flux_ph_sec_m2_um'],
                        'flux_post_screen_ph_sec_m2_um': source_val['pre_screen_astro_flux_ph_sec_m2_um']
                        }}
                self.prop_dict.update(dict_this)
                '''

                idx = 15 # wavelength slice index (for plotting only)

                # tryptichs of source, transmission, source * transmission
                for transmission_screen_name in transmission_screen_order:
                    source_img = source_dict_pre_screen[source_name][idx, :, :].value
                    source_units = source_val[transmission_screen_name][idx, :, :].unit.to_string()
                    transmission_img = transmission_screens[transmission_screen_order.index(transmission_screen_name), :, :]
                    source_times_transmission_img = source_img * source_val[transmission_screen_name][idx, :, :].value
                    source_times_transmission_units = source_val[transmission_screen_name][idx, :, :].unit.to_string()

                    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
                    im0 = axs[0].imshow(source_img, origin='lower', cmap='gray')

                    fig.suptitle(f"Source: {source_name}, idx_wavel: {idx}")
                    axs[0].set_title("Source")
                    axs[0].set_xlabel(f"x (pixel)")
                    axs[0].set_ylabel(f"y (pixel)")
                    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label=f"{source_units}") ## ## TODO: MAY NEED TO ADD /ARCSEC**2 TO UNITS, ONCE ARCSEC ARE FULLY INCORPORATED HERE
                    im1 = axs[1].imshow(transmission_img, origin='lower', cmap='gray')
                    axs[1].set_title("Transmission")
                    axs[1].set_xlabel(f"x (pixel)")
                    axs[1].set_ylabel(f"y (pixel)")
                    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label=f"transmission")
                    im2 = axs[2].imshow(source_times_transmission_img, origin='lower', cmap='gray')
                    axs[2].set_title(f"Source * Transmission ({transmission_screen_name})\n({np.sum(source_times_transmission_img)/np.sum(source_img)*100:.2f}% transmitted; not chopped)")
                    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label=f"{source_times_transmission_units}")

                    # for debugging
                    #if source_name == 'exoplanet_model_10pc':
                    #    ipdb.set_trace()
                    file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"source_transmission_map_triptych_{source_name}_angle_{int(fyi_angle)}_output_{transmission_screen_name}.png"
                    fig.savefig(file_name_plot)
                    plt.close(fig)
                    logging.info(f"Saved plot of source, transmission, source * transmission triptych at angle {int(fyi_angle)} to {file_name_plot}")

        if plot: # pragma: no cover

            # pre-screen, pre-aperture incident fluxes
            plt.clf()
            plt.figure(figsize=(8, 8))

            for source_name in self.sources_to_include:
                plt.plot(
                    self.sources_astroph[source_name]['wavel'], 
                    self.sources_astroph[source_name]['pre_screen_astro_flux_ph_sec_m2_um'], 
                    label=source_name)
            plt.yscale('log')
            plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid pattern to plot
            plt.xlim([4, 18]) # for comparison with Dannert
            plt.ylim([1e-3, 1e10]) # for comparison with Dannert
            plt.xlabel(f"Wavelength ({self.sources_astroph[source_name]['wavel'].unit})")
            plt.ylabel(f"Flux (" + str(self.sources_astroph[source_name]['pre_screen_astro_flux_ph_sec_m2_um'].unit) + ")")
            plt.legend()
            plt.title(format_plot_title("Photoelectrons, pre-aperture (no nulling yet)", self.config), loc='left')
            plt.tight_layout()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"photoelectrons_all_sources_pre_aperture.png"
            plt.savefig(file_name_plot)
            logging.info("Saved plot of incident flux pre-aperture to " + file_name_plot)

        return transmission_screens


    def pass_through_aperture(self, plot: bool = False):
        # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec

        transmission_screen_order = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark']
        collecting_area = compute_collecting_area_m2(self.config) * u.m**2

        # telescope throughput
        eta_t = float(self.config['telescope']['eta_t'])

        # apply throughput and collecting area to the flux
        for source_name, source_val in self.sources_astroph.items():

            post_aperture_flux_by_output = {
                output_name: eta_t * np.multiply(collecting_area, source_val['flux_cube_post_screen_ph_sec_um'][output_name])
                for output_name in transmission_screen_order
            }

            # note telescope throughput is incorporated at the stage of passing through the aperture
            ## ## TODO; make a separate module for throughput, and add other terms (telescope background, etc.)
            self.prop_dict[source_name] = {
                'wavel': source_val['wavel'],
                'flux_cube_post_screen_pre_aperture_ph_sec_m2_um': source_val['flux_cube_post_screen_ph_sec_um'],
                'flux_cube_post_screen_post_aperture_ph_sec_um': post_aperture_flux_by_output, # includes chop signal if enabled
            }


        # overplot all the sources
        if plot: # pragma: no cover

            def _plot_integrated_flux_by_output(cube_key, title, file_stem):
                save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])
                for output_name in transmission_screen_order:
                    fig, ax = plt.subplots(figsize=(10, 12))
                    flux_unit = None
                    wavel_unit = None
                    for source_name, source_val in self.prop_dict.items():
                        if source_name not in self.sources_to_include:
                            continue
                        flux_integrated = np.sum(
                            source_val[cube_key][output_name], axis=(1, 2)
                        )
                        ax.plot(
                            source_val['wavel'],
                            flux_integrated,
                            label=source_name,
                        )
                        if flux_unit is None:
                            flux_unit = flux_integrated.unit
                            wavel_unit = source_val['wavel'].unit
                    ax.set_yscale('log')
                    ax.set_xlim([4, 18])  # for comparison with Dannert
                    ax.set_ylim([1e-3, 1e10])  # for comparison with Dannert
                    ax.set_xlabel(f'Wavelength ({wavel_unit})')
                    ax.set_ylabel(f'Flux ({flux_unit})')
                    ax.legend(fontsize=8, loc='best')
                    ax.set_title(
                        format_plot_title(f'{title} — {output_name}', self.config),
                        loc='left',
                        fontsize=8,
                    )
                    file_name = f'{file_stem}_{output_name}.png'
                    fig.tight_layout()
                    fig.savefig(save_dir + file_name)
                    plt.close(fig)
                    logging.info(f'Saved plot to {save_dir}{file_name}')

            _plot_integrated_flux_by_output(
                'flux_cube_post_screen_pre_aperture_ph_sec_m2_um',
                'Post-screen, pre-aperture flux (all sources)',
                'flux_all_sources_post_screen_pre_aperture',
            )

            _plot_integrated_flux_by_output(
                'flux_cube_post_screen_post_aperture_ph_sec_um',
                'Post-screen, post-aperture flux (all sources)',
                'flux_all_sources_post_screen_post_aperture',
            )
        logging.info(f'Passed astrophysical flux through telescope aperture...')

        return


    def photons_to_e(self):
        # convert photons to e-s, using e-to-photon relation and QE, and update prop_dict with the converted terms

        ## ## TODO: INCORPORATE REAL RESPONSE CURVES

        for source_name, source_val in self.prop_dict.items():
            if ('flux_post_aperture_ph_sec_um' in source_val) and (source_val['flux_post_aperture_ph_sec_um'].unit == u.ph / (u.um * u.s)):
                source_val['flux_e_sec_um'] = float(self.config["detector"]["photons_to_e"]) * (u.electron/u.ph) * np.multiply(float(self.config["detector"]["quantum_efficiency"]), source_val['flux_post_aperture_ph_sec_um'])

        return


    '''
    def e_to_adu(self):

        for source_name, source_val in self.prop_dict.items():
            if ('flux_e_sec_um' in source_val) and (source_val['flux_e_sec_um'].unit == u.electron / (u.um * u.s)):
                source_val['flux_adu_sec_um'] = np.divide(source_val['flux_e_sec_um'], float(self.config["detector"]["gain"])) * (u.adu/u.electron)

        logging.info(f'Converted e-s to ADUs...')

        return
    '''