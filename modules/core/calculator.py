"""
Main noise calculator for the modules package.

This module provides the primary interface for calculating total noise
and signal-to-noise ratios for infrared detector observations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import configparser
import ipdb
import matplotlib.pyplot as plt

from .astrophysical import AstrophysicalSources
from .instrumental import InstrumentalSources
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
    
    def __init__(self, config: Dict, noise_origin: InstrumentalSources):
        """
        Initialize the noise calculator.
        
        Args:
            config: Configuration dictionary containing all parameters
            noise_origin: the object including the various noise contributions
            
        Raises:
            ValueError: If configuration is invalid
        """

        self.config = config

        # the object that is the 'origin' of the various noise contributions for the calculations to follow
        self.noise_origin = noise_origin

        #self.unit_converter = UnitConverter()
        #self.conversion_engine = ConversionEngine(self.unit_converter)
        
        # Store pre-calculated incident flux if provided
        #self.incident_flux = incident_flux
        
        # Initialize noise calculators
        #self.astrophysical_sources = AstrophysicalSources(config, self.unit_converter)
        #self.instrumental_noise = InstrumentalSources(config, self.unit_converter)
        
        # Generate wavelength grid
        #self.wavelength = self._generate_wavelength_grid()

    def s2n_e(self):
        '''
        Find S/N using photoelectrons

        Ref.: s_to_n_logic_life_detectors.pdf
        '''

        ## everything in units of photoelectrons

        wavel_abcissa = self.noise_origin.prop_dict['wavel']
        # map wavelengths to pixels
        res_spec = float(self.config["detector"]["spec_res"]) # spectral resolution (del_lambda/lambda)
        del_lambda_array = wavel_abcissa / res_spec # size of wavelength bins (um)
        #wavel_abcissa_array = wavel_abcissa - del_lambda_array/2 # center of wavelength bins (um)
        # Edges of the wavelength bins.
        # The bin edges are at wavel_abcissa - del_lambda_array/2 and wavel_abcissa + del_lambda_array/2
        wavel_bin_edges_lower = wavel_abcissa - del_lambda_array/2
        wavel_bin_edges_upper = wavel_abcissa + del_lambda_array/2
        #n_pix = 1 / disp # pixels per micron(pix/um)
        #pix_abcissa = n_pix*wavel_abcissa - np.min(n_pix*wavel_abcissa) # remove offset

        # integration time for 1 frame
        t_int = float(self.config["observation"]["integration_time"])
        n_int = float(self.config["observation"]["n_int"])

        # total science (planet) signal (note this is not a function of time)
        # _prime denotes it is not measured directly (i.e., photoelectrons and not ADU)
        del_Np_prime_del_t = self.noise_origin.prop_dict['exoplanet_flux_e_sec_um'] * del_lambda_array # approximates an integral over lambda (final units are e/sec/um * um = e/sec)
        # stellar signal
        del_Ns_prime_del_t = self.noise_origin.prop_dict['star_flux_e_sec_um'] * del_lambda_array # approximates an integral over lambda 

        # quantum efficiency
        eta = float(self.config["detector"]["quantum_efficiency"])
        # null
        nulling_factor = float(self.config["nulling"]["nulling_factor"])

        # read noise
        R = self.noise_origin.instrum_dict['read_noise_e_rms']
        # dark current in one pixel
        D_rate = self.noise_origin.instrum_dict['dark_current_e_pix-1_sec-1']
        D_tot = self.noise_origin.instrum_dict['dark_current_total_e']

        # Reshape arrays for broadcasting
        del_Np_prime_del_t_reshaped = np.tile( del_Np_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        del_Ns_prime_del_t_reshaped = np.tile( del_Ns_prime_del_t, (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        
        # the number of pixels for each wavelength bin is 1-to-1 for now ## ## TODO: change later
        n_pix_array_reshaped = np.tile( np.ones(len(wavel_abcissa)), (len(D_tot), 1) ) # shape (N_dark_current, N_wavel)
        
        D_rate_reshaped = np.tile(D_rate, (len(wavel_abcissa), 1) ).T # shape (N_dark_current, N_wavel)

        # Now the calculation will broadcast to (N_dark_current, N_wavel)
        s2n = np.sqrt(n_int) * np.divide(eta * t_int * del_Np_prime_del_t_reshaped, 
                        np.sqrt(eta * t_int * ( del_Np_prime_del_t_reshaped + nulling_factor * del_Ns_prime_del_t_reshaped ) + 
                            n_pix_array_reshaped * (R**2 + t_int * D_rate_reshaped)))

        # 2D plot of S/N vs wavelength and dark current
        plt.close()
        plt.imshow(s2n,aspect='auto', origin='lower')
        plt.colorbar()

        # Set y-axis ticks to match D_rate_reshaped[:,0]
        plt.yticks(range(len(D_rate_reshaped[:,0])), D_rate_reshaped[:,0])

        # Set bottom x-axis: wavelength
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
        plt.title('S/N, int time = ' + str(int(t_int)) + ' sec, n_int = ' + str(int(n_int)))
        plt.tight_layout()
        file_name_plot = "/Users/eckhartspalding/Downloads/" + f"2d_s2n_vs_wavelength_and_dark_current.png"
        plt.savefig(file_name_plot)
        logger.info(f"Wrote plot {file_name_plot}")

        plt.close()
        for plot_num in range(0,len(s2n[:,0])):
            plt.plot(wavel_abcissa, s2n[plot_num,:], label=str(int(plot_num)))
        #plt.yscale('log')
        plt.ylabel('S/N per wavelength bin')
        plt.xlabel('Wavelength (um)')
        plt.title('S/N for different dark currents')
        plt.legend()
        file_name_plot = "/Users/eckhartspalding/Downloads/" + f"1d_s2n_vs_wavelength_and_dark_current_per_wavelength_bin.png"
        plt.savefig(file_name_plot)
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