"""
Main noise calculator for the modules package.

This module provides the primary interface for calculating total noise
and signal-to-noise ratios for infrared detector observations.
"""

from ipaddress import ip_network
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import configparser
import ipdb
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sns

from .astrophysical import AstrophysicalSources
from .instrumental import InstrumentDepTerms, Detector
from .conversions import ConversionEngine
from ..data.units import UnitConverter
from ..config.validator import validate_config

logger = logging.getLogger(__name__)

'''
def calculate_astrophysical_noise_adu(total_astro_adu: float) -> np.ndarray:
    """
    Calculate astrophysical noise in electrons per pixel.
    
    Args:
        total_astro_adu: the total astrophysical flux contribution to the readout in ADU
        
    Returns:
        Noise in electrons per pixel
    """
    
    # photon noise: sqrt(N)
    noise_adu = np.sqrt(total_astro_adu)
    
    return noise_adu
'''


class NoiseCalculator:
    """
    Puts astrophysical and instrumental sources together.
    """
    
    def __init__(self, config: Dict, sources_all):
        """
        Initialize the noise calculator.
        
        Args:
            config: Configuration dictionary containing all parameters
            sources_all: the object including the various fluxes and noise contributions, from astro and instrumental sources
            
        Raises:
            ValueError: If configuration is invalid
        """

        self.config = config

        # the object that is the 'origin' of the various noise contributions for the calculations to follow
        self.sources_all = sources_all

        #logging.info("Calculating noise...")
        #self.prop_dict = noise_origin.prop_dict

        #self.unit_converter = UnitConverter()
        #self.conversion_engine = ConversionEngine(self.unit_converter)
        
        # Store pre-calculated incident flux if provided
        #self.incident_flux = incident_flux
        
        # Initialize noise calculators
        #self.astrophysical_sources = AstrophysicalSources(config, self.unit_converter)
        #self.instrumental_noise = InstrumentDepTerms(config, self.unit_converter)
        
        # Generate wavelength grid
        #self.wavelength = self._generate_wavelength_grid()

    def s2n_val(self, wavel_bin_centers, del_lambda_array, n_pix):
        """ _star_planet_only
        Vectorized S/N function. Any single input variable can be an array while others remain scalars.
        Note this equation is for the case where there is only a planet and star signal, and no exozodiacal dust etc.
        
        INPUTS:
        wavel_bin_centers: wavelength bin centers (um)
        del_lambda_array: wavelength bin widths (um)
        n_int: number of integrations
        t_int: time for 1 integration (sec)
        n_pix: total number of pixels under the wavelength element footprint (pix)
        del_Np_prime_del_t: planet signal rate (photo-electrons / sec) 
        del_Nstar_prime_del_t: star signal rate (photo-electrons / sec)
        null: nulling factor
        R: read noise (photo-electrons rms)
        D_rate: dark current rate (photo-electrons / (pix * sec))
        eta_qm: quantum efficiency

        OUTPUTS:
        s2n: S/N (unitless); can be 2D
        """

        # self.sources_all.prop_dict['star']

        # reinterpolate the fluxes onto the binned wavelength grid
        ## ## TODO: IS THERE A BETTER WAY TO DO THIS, OTHER THAN INTERPOLATING?
        exoplanet_flux_e_sec_um = np.interp(wavel_bin_centers, 
                                            self.sources_all.prop_dict['exoplanet']['wavel'].value, 
                                            self.sources_all.prop_dict['exoplanet']['flux_e_sec_um'].value) * u.electron / (u.um * u.s)
        star_flux_e_sec_um = np.interp(wavel_bin_centers, 
                                            self.sources_all.prop_dict['star']['wavel'].value, 
                                            self.sources_all.prop_dict['star']['flux_e_sec_um'].value) * u.electron / (u.um * u.s)
        exozodiacal_flux_e_sec_um = np.interp(wavel_bin_centers, 
                                            self.sources_all.prop_dict['exozodiacal']['wavel'].value, 
                                            self.sources_all.prop_dict['exozodiacal']['flux_e_sec_um'].value) * u.electron / (u.um * u.s)
        zodiacal_flux_e_sec_um = np.interp(wavel_bin_centers, 
                                            self.sources_all.prop_dict['zodiacal']['wavel'].value, 
                                            self.sources_all.prop_dict['zodiacal']['flux_e_sec_um'].value) * u.electron / (u.um * u.s)


        # find count rates by approximating an integral over lambda (final units are e/sec/um * um = e/sec)
        del_Np_prime_del_t = exoplanet_flux_e_sec_um * del_lambda_array 
        del_Ns_prime_del_t = star_flux_e_sec_um * del_lambda_array
        del_Nez_prime_del_t = exozodiacal_flux_e_sec_um * del_lambda_array
        del_Nz_prime_del_t = zodiacal_flux_e_sec_um * del_lambda_array 
        
        # Convert all inputs to numpy arrays to enable broadcasting
        # This ensures consistent shapes for vectorized operations
        n_int = np.asarray(int(self.config['observation']['n_int']))
        t_int = np.asarray(float(self.config['observation']['integration_time'])) * u.second
        n_pix = np.asarray(n_pix) * u.pix
        del_Np_prime_del_t = np.asarray(del_Np_prime_del_t) * u.electron / u.second
        del_Ns_prime_del_t = np.asarray(del_Ns_prime_del_t) * u.electron / u.second
        null = np.asarray(float(self.config['nulling']['nulling_factor']))
        
        # Handle the case where read_noise may be a comma-separated list (array) or a single value
        '''
        read_noise_str = self.config['detector']['read_noise']
        try:
            R_vals = np.fromstring(read_noise_str, sep=',')
            if R_vals.size == 1:
                R = R_vals[0] * u.electron
            else:
                R = R_vals * u.electron
        except Exception:
            R = float(read_noise_str) * u.electron
        '''
        R = self.sources_all.sources_instrum['read_noise_e_pix-1']
        D_rate = self.sources_all.sources_instrum['dark_current_e_pix-1_sec-1']
        D_tot = self.sources_all.sources_instrum['dark_current_e_pix-1']
        # Handle the case where dark_current may be a comma-separated list (array) or a single value
        '''
        dark_current_str = self.config['detector']['dark_current']
        try:
            # Try to parse as array (comma-separated)
            D_vals = np.fromstring(dark_current_str, sep=',')
            if D_vals.size == 1:
                D = D_vals[0] * u.electron / (u.pix * u.second)
            else:
                D = D_vals * u.electron / (u.pix * u.second)
        except Exception:
            # Fallback: try to parse as float
            D = float(dark_current_str) * u.electron / (u.pix * u.second)
        '''
        
        eta_qm = np.asarray(float(self.config['detector']['quantum_efficiency']))
        eta_t = np.asarray(float(self.config['telescope']['eta_t']))

        # Reshape arrays for broadcasting
        del_Np_prime_del_t_reshaped = np.tile( del_Np_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        del_Ns_prime_del_t_reshaped = np.tile( del_Ns_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        del_Nez_prime_del_t_reshaped = np.tile( del_Nez_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        del_Nz_prime_del_t_reshaped = np.tile( del_Nz_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)

        # note: either D_rate or R can be an array of length >1, but not both
        if len(D_rate) > 1:
            D_rate_reshaped = np.tile(D_rate, (len(wavel_bin_centers), 1) ).T # shape (N_dark_current, N_wavel)
            R_reshaped = R * np.ones((len(D_rate), len(wavel_bin_centers))) # shape (N_dark_current, N_wavel)
        elif len(R) > 1:
            D_rate_reshaped = D_rate * np.ones((len(R), len(wavel_bin_centers))) # shape (N_dark_current, N_wavel)
            R_reshaped = np.tile(R, (len(wavel_bin_centers), 1) ).T # shape (N_R, N_wavel)
        else:
            D_rate_reshaped = D_rate
            R_reshaped = R

        # term in front
        term_1 = np.sqrt(n_int)

        # numerator
        term_2 = eta_t * eta_qm * t_int * del_Np_prime_del_t

        # first term under square root in the denominator
        term_3 = eta_t * eta_qm * t_int * (del_Np_prime_del_t_reshaped + del_Nez_prime_del_t + del_Nz_prime_del_t + null * del_Ns_prime_del_t_reshaped)

        # second term under square root in the denominator
        term_4 = n_pix * (( R_reshaped**2/(u.electron / u.pix) ) + t_int * D_rate_reshaped)



        return ( term_1 * term_2 / np.sqrt(term_3 + term_4) ) / u.electron**0.5


    def s2n_e(self):
        '''
        Find S/N using photoelectrons

        Ref.: s_to_n_logic_life_detectors.pdf
        '''

        ## everything in units of photoelectrons
        
        #wavel_abcissa = self.noise_origin.prop_dict['wavel']
        logging.info("Calculating S/N ...")

        ## ## TO DO: MAKE ALL WAVELS TO BE ON AN EXPLICIT COMMON BASIS ACCORDING TO THE BINNING
        ## ## TO DO: INCORPORATE NYQUIST SAMPLING
        wavel_abcissa = self.sources_all.sources_astroph['star']['wavel']

        # a constant pixel spacing for each wavelength bin is assumed

        # map wavelengths to pixels
        res_spec = float(self.config["detector"]["spec_res"]) # spectral resolution (lambda/del_lambda)
        # find bin edges

        #del_lambda_array = wavel_abcissa / res_spec # size of wavelength bins (um)
        #wavel_abcissa_array = wavel_abcissa - del_lambda_array/2 # center of wavelength bins (um)

        # choose 4 um as the center of one bin, and stairstep from there
        wavel_bin_edges_lower = np.array([])
        wavel_bin_edges_upper = np.array([])
        wavel_bin_centers = np.array([])
        del_lambda_array = np.array([])
        del_pixel_array = np.array([0]) # start with 0 pixel relative displacement (this array will effectively be the bid edges of each wavelength element in pixel space)

        bin_edge_lower_this = 4.0
        bin_edge_upper_this = 0.0
        ## ## TO DO: SHIFT STARSTEP TO HAVE CENTERS OF BINS AT THE LOWER EDGE (BECAUSE THE BIN WIDTH IS SET BY THE RESOLUTION AT THE LOWER EDGE)
        while bin_edge_upper_this < 20: # at wavelengths <20 um
             del_lambda_this = bin_edge_lower_this / res_spec
             bin_edge_upper_this = bin_edge_lower_this + del_lambda_this
             wavel_bin_edges_lower = np.append(wavel_bin_edges_lower, bin_edge_lower_this)
             wavel_bin_edges_upper = np.append(wavel_bin_edges_upper, bin_edge_upper_this)
             wavel_bin_centers = np.append(wavel_bin_centers, (bin_edge_lower_this + bin_edge_upper_this) / 2)
             del_lambda_array = np.append(del_lambda_array, del_lambda_this)
             bin_edge_lower_this = bin_edge_upper_this
             del_pixel_array = np.append(del_pixel_array, del_pixel_array[-1] + float(self.config["detector"]["pix_per_wavel_bin"]))
        del_lambda_array = del_lambda_array * u.um # attach units

        # instantiate detector object
        detector = Detector(config=self.config, num_wavel_bins=len(del_lambda_array))
        # get the illumination footprint (cube where each slice is the footprint for one wavelength bin)
        ## ## NOTE THIS IS KIND OF REDUNDANT RIGHT NOW, SINCE THE NUMBER OF PIXELS PER WAVELENGTH BIN IS CONSTANT AS CALCULATED BELOW; MIGHT CHANGE THIS LATER IF THE DISPERSION IS NOT CONSTANT
        footprint_spec_cube = detector.footprint_spectral(plot=True)

        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = [(hi - lo) for lo, hi in zip(wavel_bin_edges_lower, wavel_bin_edges_upper)] # removed units for plotting

        #n_pix = 1 / disp # pixels per micron(pix/um)
        #pix_abcissa = n_pix*wavel_abcissa - np.min(n_pix*wavel_abcissa) # remove offset

        # integration time for 1 frame
        #t_int = float(self.config["observation"]["integration_time"]) * u.second
        n_int = float(self.config["observation"]["n_int"])

        # total science (planet) signal (note this is not a function of time)
        # _prime denotes it is not measured directly (i.e., photoelectrons and not ADU)

        ## ## REMOVED HERE

        #star_flux_e_sec_um = self.sources_all.prop_dict['star']['flux_e_sec_um'].interpolate(wavel_bin_centers)



        # quantum efficiency
        #eta = float(self.config["detector"]["quantum_efficiency"])
        # null
        #nulling_factor = float(self.config["nulling"]["nulling_factor"])

        # read noise
        #R = self.sources_all.sources_instrum['read_noise_e_pix-1'] # in e- per pixel rms
        # dark current in one pixel


 
        
        # the number of pixels for each wavelength bin
        ## ## NOTE np.sum(footprint_spec_cube[0,:,:]) MEANS THAT THE NUMBER OF PIXELS IS THE SAME FOR ALL WAVELENGTH BINS
        #n_pix_array_reshaped = np.tile( np.sum(footprint_spec_cube[0,:,:]) * np.ones(len(wavel_bin_centers)), (len(D_tot), 1) ) * u.pix # shape (N_dark_current, N_wavel)
        n_pix = np.sum(footprint_spec_cube[0,:,:]) * u.pix
        
        # Now the calculation will broadcast to (N_dark_current, N_wavel)
        s2n = self.s2n_val(wavel_bin_centers=wavel_bin_centers, del_lambda_array=del_lambda_array, n_pix=n_pix)



        '''
        # old formulation
        s2n = np.sqrt(n_int) * np.divide(eta * t_int * del_Np_prime_del_t_reshaped, 
                        np.sqrt(eta * t_int * ( del_Np_prime_del_t_reshaped + nulling_factor * del_Ns_prime_del_t_reshaped ) + 
                            n_pix_array_reshaped * (( R**2/(u.electron / u.pix) ) + t_int * D_rate_reshaped))) # the R**2/(u.electron / u.pix) is necessary to make the units consistent 
        '''

        #ipdb.set_trace()
        s2n = s2n.value # get rid of the sqrt(e-) units for plotting


        # stuff for plots
        # parse dark current values (can be comma-separated list)
        dark_current_str = self.config['detector']['dark_current']
        if ',' in dark_current_str:
            dark_current_values = [float(x.strip()) for x in dark_current_str.split(',')]
            dark_current_display = ', '.join([f"{val:.2f}" for val in dark_current_values])
        else:
            dark_current_values = float(dark_current_str)
            dark_current_display = f"{float(dark_current_str):.2f}"
        # do the same for read noise
        read_noise_str = self.config['detector']['read_noise']
        if ',' in read_noise_str:
            read_noise_values = [float(x.strip()) for x in read_noise_str.split(',')]
            read_noise_display = ', '.join([f"{val:.2f}" for val in read_noise_values])
        else:
            read_noise_display = f"{float(read_noise_str):.2f}"
            read_noise_values = float(read_noise_str)

        # Prepare two left-aligned columns for figure metadata
        instrumental_lines = [
            "INSTRUMENTAL:",
            f"collecting area = {float(self.config['telescope']['collecting_area']):.2f} m²",
            f"telescope throughput = {float(self.config['telescope']['eta_t']):.2f}",
            f"quantum efficiency = {float(self.config['detector']['quantum_efficiency']):.2f}",
            f"dark current = {dark_current_display} e-/pix/sec",
            f"read noise = {read_noise_display} e- rms",
            f"gain = {float(self.config['detector']['gain']):.2f} e-/ADU",
            f"pix per wavel bin = {float(self.config['detector']['pix_per_wavel_bin']):.2f}",
            f"integration time = {float(self.config['observation']['integration_time']):.2f} sec",
            f"number of integrations = {float(self.config['observation']['n_int']):.2f}"
        ]
        astrophysical_lines = [
            "ASTROPHYSICAL:",
            f"stellar nulling = {bool(self.config['nulling']['null'])}",
            f"nulling transmission = {float(self.config['nulling']['nulling_factor']):.2f}",
            f"distance = {float(self.config['target']['distance']):.2f} pc",
            f"planet temperature = {float(self.config['target']['pl_temp']):.2f} K",
            fr"galactic $\lambda_{{\rm rel}}$ = {float(self.config['observation']['lambda_rel_lon_los']):.2f} deg, $\beta$ = {float(self.config['observation']['beta_lat_los']):.2f} deg",
        ]


        
        ############################
        # 2D plot of S/N vs wavelength and dark current
        plt.close()

        param_name = "Dark current" if np.size(dark_current_values) > 1 else "Read noise"
        param_units_string = 'e_pix-1'
        param_values = dark_current_values if np.size(dark_current_values) > 1 else read_noise_values

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(s2n, aspect='auto', origin='lower')
        # Add contour lines for S/N = 1 (dashed) and S/N = 5 (solid)
        ax.contour(s2n, levels=[1, 5], colors=['white', 'white'], linewidths=2, linestyles=['dashed', 'solid'])

        # Choose up to ~10 ticks to keep labels readable
        param_values = dark_current_values if np.asarray(dark_current_values).size > 1 else read_noise_values
        param_values = np.asarray(param_values, dtype=float)  
        N = len(param_values)
        tick_idx = np.linspace(0, N - 1, num=min(N, 10), dtype=int)
        ax.set_yticks(tick_idx)
        labels = [f"{param_values[i]:g}" for i in tick_idx]
        ax.set_yticklabels(labels)

        ax.set_ylabel(param_name + " (" + param_units_string + ")")
        ax.set_xlabel("Wavelength bin (" + str(wavel_abcissa.unit) + ")")
        fig.colorbar(im, ax=ax, label="S/N")
        # Keep a concise axes title and add two figure-level columns
        ax.set_title("S/N")
        fig.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
        fig.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
        #plt.tight_layout()
        plt.subplots_adjust(top=0.7,right=0.8)
        plt.show()

        #plt.imshow(s2n,aspect='auto', origin='lower')
        #plt.imshow(s2n, origin='lower')
        #plt.colorbar()

        # Set y-axis ticks to match D_rate_reshaped[:,0]
        #plt.yticks(range(len(D_rate_reshaped[:,0])), D_rate_reshaped[:,0])

        # Set bottom x-axis: wavelength
        plt.clf()
        # Set x-ticks at intervals of every three values along the x axis, and rotate the labels 30 degrees
        interval = 3
        # Set x-ticks at intervals of every two values along the x axis
        x_tick_indices = np.arange(0, len(wavel_abcissa), interval)

        ax_bottom = plt.gca()
        ax_bottom.set_xticks(x_tick_indices)
        ax_bottom.set_xticklabels(np.round(wavel_abcissa[x_tick_indices], 2), rotation=30)
        ax_bottom.set_xlabel('Wavelength (um)')

        # Add contour lines for S/N = 1 (dashed) and S/N = 5 (solid)
        X, Y = np.meshgrid(np.arange(s2n.shape[1]), np.arange(s2n.shape[0]))
        contour = ax_bottom.contour(
            X, Y, s2n, 
            levels=[1, 5], 
            colors=['white', 'white'],  
            linewidths=2, 
            linestyles=['dashed', 'solid']
        )
        ax_bottom.clabel(contour, inline=True, fmt='%.1f', fontsize=9)
        plt.ylabel('Dark current (e- sec-1 pix-1)')
        # Keep a concise axes title and add two figure-level columns
        ax_bottom.set_title("S/N")
        fig2 = plt.gcf()
        fig2.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
        fig2.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
        plt.tight_layout()
        file_name_plot = "/Users/eckhartspalding/Downloads/" + f"2d_s2n_vs_wavelength_and_dark_current.png"
        #plt.show()
        plt.savefig(file_name_plot)
        logger.info(f"Wrote plot {file_name_plot}")
        plt.close()

        

        '''
        for plot_num in range(0,len(s2n[:,0])):
            # draw a histogram-like plot of S/N using step plot that respects bin widths
            ipdb.set_trace()
            # Create x-coordinates that include both bin edges for proper step plotting
            x_step = np.repeat(wavel_bin_edges_lower, 2)
            x_step[1::2] = wavel_bin_edges_upper  # Replace every other element with upper edges
            y_step = np.repeat(s2n[plot_num,:], 2)  # Repeat y-values for step effect
            plt.plot(x_step, y_step, label=f'Dark current {plot_num}', linewidth=2)
            
            #plt.bar(wavel_bin_edges_lower.value, s2n[plot_num,:], width=bin_widths, align='edge', edgecolor='black', linewidth=0)
            #plt.plot(wavel_abcissa, s2n[plot_num,:], label=str(int(plot_num)))
        '''
        # Create seaborn histogram with custom bin edges
        plt.figure(figsize=(10, 6))
        #sns.histplot(data=s2n[0,:], bins=wavel_bin_edges_lower, alpha=0.7, edgecolor='black', linewidth=1)
        for plot_num in range(0,len(s2n[:,0])):
            plt.scatter(wavel_bin_centers, s2n[plot_num,:], alpha=0.5)
        # Add vertical lines at bin edges for reference
        #for line in wavel_bin_edges_lower:
        #     plt.axvline(x=line, color='gray', linestyle='--', alpha=0.5)

        plt.scatter(wavel_bin_centers, s2n[0,:], color='black', alpha=0.5)
        plt.scatter(wavel_bin_centers, s2n[1,:], color='red', alpha=0.5)
        plt.axhline(y=1, color='gray', linestyle='--')
        plt.axhline(y=5, color='gray', linestyle='-')
        # Annotate S/N = 1 and S/N = 5 on the plot
        plt.annotate('S/N = 1', xy=(6, 1), xytext=(-10, 5), textcoords='offset points',
                     ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
        plt.annotate('S/N = 5', xy=(6, 5), xytext=(-10, 5), textcoords='offset points',
                     ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
        #plt.yscale('log')
        plt.xlim([4, 18])
        plt.ylabel('S/N per wavelength bin')
        plt.xlabel('Wavelength (um)')
        # Keep a concise axes title and add two figure-level columns
        plt.title("S/N")
        fig3 = plt.gcf()
        fig3.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
        fig3.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
        plt.legend()
        plt.tight_layout()
        file_name_plot = "/Users/eckhartspalding/Downloads/" + f"1d_s2n_vs_wavelength_and_dark_current_per_wavelength_bin.png"
        plt.show()
        #plt.savefig(file_name_plot)
        logger.info(f"Wrote plot {file_name_plot}")
        '''

        s2n = np.divide(eta * Np_prime, 
                        np.sqrt( eta * (Np_prime + nulling_factor * Ns_prime) + n_pix_array * (R**2 + D_tot) )
                        )
        '''




        return s2n

    '''
    def total_astro_detector_adu(self):
        
        # Astrophysical flux after passing through the telescope, and integrated over wavelength

        # Returns:
        #     incident_dict: dictionary which now also contains the astrophysical flux in ADU in a readout

        # integrate astrophysical flux over wavelength to get total flux
        # photons/sec/m^2/micron -> photons/sec/m^2
        self.incident_astro['astro_flux_ph_sec_m2'] = np.trapz(self.incident_astro['astro_flux_ph_sec_m2_um'], self.incident_astro['wavel'])

        # pass through the telescope aperture
        # photons/sec/m^2 -> photons/sec
        self.incident_astro['astro_flux_ph_sec'] = float(self.incident_astro['astro_flux_ph_sec_m2']) * float(self.config["telescope"]["collecting_area"])

        # Convert to electrons ## ## TODO: make 1 photon = 1 electron more realistic
        # photons/sec -> electrons/sec
        qe = float(self.config["detector"]["quantum_efficiency"])
        self.incident_astro['astro_electrons_sec'] = self.incident_astro['astro_flux_ph_sec'] * qe

        # Convert to ADU (Analog-to-Digital Units)
        # electrons/sec -> ADU/sec
        gain = float(self.config["detector"]["gain"])  # e-/ADU
        self.incident_astro['astro_adu_sec'] = float(self.incident_astro['astro_electrons_sec']) / gain

        # For integration time calculations
        integration_time = float(self.config["observation"]["integration_time"])  # seconds
        # incident_dict['astro_electrons_total'] = incident_dict['astro_electrons_sec'] * integration_time
        self.incident_astro['astro_adu_total'] = self.incident_astro['astro_adu_sec'] * integration_time

        return self.incident_astro
    '''
    

    '''
    def add_fluxes(self):

        Add the astrophysical and instrumental fluxes together, in units of ADU

        Args:
            astro_contrib: dict incluting astrophysical flux in ADU
            instr_contrib: dict including instrumental flux in ADU

        Returns:

        ipdb.set_trace()

        # add the astrophysical and instrumental fluxes together, in units of ADU
        self.total_signal_adu = self.incident_astro['astro_adu_total'] + self.incident_instrum['dark_current_total_adu'] + self.incident_instrum['read_noise_adu']
        
        #incident_astro['astro_adu_total'] = 
    '''
        
    
    '''
    def _generate_wavelength_grid(self) -> np.ndarray:
        """Generate wavelength grid based on configuration."""
        wavelength_config = self.config["wavelength_range"]
        min_wavelength = wavelength_config["min"]  # microns
        max_wavelength = wavelength_config["max"]  # microns
        n_points = wavelength_config["n_points"]
        
        # Use logarithmic spacing for better spectral coverage
        wavelength = np.logspace(np.log10(min_wavelength), np.log10(max_wavelength), n_points)
        
        return wavelength
    '''
    
    def calculate_snr(self, contrib_astro: Dict, contrib_instrum: Dict) -> Dict[str, Any]:
        """
        Calculate S/N
        
        Returns:
            Dictionary containing all calculation results
        """
        logger.info("Starting SNR calculation")
        
        
        '''
        # astrophysical photon noise
        #astro_sources = AstrophysicalSources(config=self.config, unit_converter=UnitConverter())
        #noise_adu = astro_sources.calculate_astrophysical_noise_adu(total_astro_adu)
        astrophysical_noise_adu = calculate_astrophysical_noise_adu(total_astro_adu = self.incident_astro)

        #noise_adu = AstrophysicalSources.calculate_astrophysical_noise_adu(total_astro_adu = self.total_astro_detector_adu)
        
        # Calculate instrumental noise
        instrumental_noise_adu = self.instrumental_noise.calculate_total_instrumental_noise_adu(
            integration_time
        )
        
        # Calculate total noise
        total_noise_adu = self.conversion_engine.calculate_total_noise_adu(
            astrophysical_noise_adu, instrumental_noise_adu
        )
        
        # Calculate signal (assuming exoplanet is the signal of interest)
        exoplanet_flux = self.astrophysical_sources.calculate_source_flux("exoplanet", self.wavelength)
        exoplanet_illumination = self.astrophysical_sources.calculate_detector_illumination(self.wavelength)
        exoplanet_signal_adu = self.astrophysical_sources.calculate_astrophysical_noise_adu(
            self.wavelength, integration_time
        )
        
        # Calculate SNR
        snr = self.conversion_engine.calculate_signal_to_noise(exoplanet_signal_adu, total_noise_adu)
        
        # Calculate integrated SNR
        integrated_snr = self.conversion_engine.calculate_integrated_snr(snr, self.wavelength)
        
        # Calculate detection limit
        detection_limit = self.conversion_engine.calculate_detection_limit(total_noise_adu)
        
        # Get noise breakdowns
        astrophysical_breakdown = self.astrophysical_sources.get_source_contributions(self.wavelength)
        instrumental_breakdown = self.instrumental_noise.get_noise_breakdown_adu(integration_time)
        
        results = {
            "wavelength": self.wavelength,
            "integration_time": integration_time,
            "astrophysical_noise_adu": astrophysical_noise_adu,
            "instrumental_noise_adu": instrumental_noise_adu,
            "total_noise_adu": total_noise_adu,
            "exoplanet_signal_adu": exoplanet_signal_adu,
            "signal_to_noise": snr,
            "integrated_snr": integrated_snr,
            "detection_limit": detection_limit,
            "astrophysical_breakdown": astrophysical_breakdown,
            "instrumental_breakdown": instrumental_breakdown,
            "config": self.config,
        }
        
        logger.info(f"SNR calculation complete. Integrated SNR: {integrated_snr:.2f}")
        '''

        # astro photon noise
        noise_astro = np.sqrt(contrib_astro['astro_adu_total'])
        noise_instrum = contrib_instrum['dark_current_total_adu'] + contrib_instrum['read_noise_adu']

        snr = contrib_astro['astro_adu_total'] / (noise_astro + noise_instrum)

        return snr
    
    
    '''
    def calculate_optimal_parameters(self, target_snr: float = 5.0) -> Dict[str, Any]:
        """
        Calculate optimal observation parameters.
        
        Args:
            target_snr: Target signal-to-noise ratio
            
        Returns:
            Dictionary containing optimal parameters
        """
        # Calculate current SNR
        current_results = self.calculate_snr()
        current_snr = current_results["integrated_snr"]
        
        # Calculate required integration time for target SNR
        current_integration_time = self.config["detector"]["integration_time"]
        required_integration_time = current_integration_time * (target_snr / current_snr) ** 2
        
        # Calculate optimal integration time
        astrophysical_noise_rate = np.mean(current_results["astrophysical_noise_adu"]) / current_integration_time
        instrumental_noise = current_results["instrumental_noise_adu"]
        optimal_integration_time = self.conversion_engine.calculate_optimal_integration_time(
            astrophysical_noise_rate, instrumental_noise, target_snr
        )
        
        optimal_params = {
            "current_snr": current_snr,
            "target_snr": target_snr,
            "current_integration_time": current_integration_time,
            "required_integration_time": required_integration_time,
            "optimal_integration_time": optimal_integration_time,
            "snr_ratio": target_snr / current_snr,
        }
        
        return optimal_params
    
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current calculation results.
        
        Returns:
            Dictionary containing summary information
        """
        results = self.calculate_snr()
        
        summary = {
            "integrated_snr": results["integrated_snr"],
            "wavelength_range": {
                "min": np.min(self.wavelength),
                "max": np.max(self.wavelength),
                "units": "microns"
            },
            "integration_time": results["integration_time"],
            "total_astrophysical_noise": np.mean(results["astrophysical_noise_adu"]),
            "total_instrumental_noise": results["instrumental_noise_adu"],
            "total_noise": np.mean(results["total_noise_adu"]),
            "detection_limit": np.mean(results["detection_limit"]),
            "max_snr": np.max(results["signal_to_noise"]),
            "min_snr": np.min(results["signal_to_noise"]),
        }
        
        return summary 

    def calculate_snr_with_incident_flux(self, integration_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate SNR using pre-calculated incident flux.
        
        Args:
            integration_time: Integration time in seconds (uses config default if None)
            
        Returns:
            Dictionary containing SNR calculation results
        """
        if self.incident_flux is None:
            raise ValueError("No incident flux provided. Use calculate_snr() instead.")
        
        if integration_time is None:
            integration_time = float(self.config["detector"]["integration_time"])
        
        # Extract incident flux data
        incident_wavelength = self.incident_flux['wavel']
        incident_flux = self.incident_flux['flux']
        
        # Calculate detector illumination using incident flux
        collecting_area = float(self.config["telescope"]["collecting_area"])
        throughput = float(self.config["telescope"]["throughput"])
        plate_scale = float(self.config["telescope"]["plate_scale"])
        
        # Convert incident flux to detector illumination
        pixel_area = (plate_scale ** 2) * (np.pi / (180 * 3600)) ** 2
        detector_illumination = incident_flux * collecting_area * throughput * pixel_area
        
        # Calculate signal in electrons
        signal_electrons = detector_illumination * integration_time
        
        # Calculate noise (shot noise: sqrt(N))
        noise_electrons = np.sqrt(signal_electrons)
        
        # Convert to ADU
        gain = float(self.config["detector"]["gain"])
        signal_adu = signal_electrons / gain
        noise_adu = noise_electrons / gain
        
        # Calculate SNR
        snr = signal_adu / noise_adu
        integrated_snr = np.sqrt(np.sum(snr**2))
        
        return {
            'wavelength': incident_wavelength,
            'signal_adu': signal_adu,
            'noise_adu': noise_adu,
            'signal_to_noise': snr,
            'integrated_snr': integrated_snr,
            'integration_time': integration_time,
            'incident_flux': incident_flux,
            'detector_illumination': detector_illumination
        } 
    '''