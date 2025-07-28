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

from .astrophysical import AstrophysicalSources
from .instrumental import InstrumentalNoise
from .conversions import ConversionEngine
from ..data.units import UnitConverter
from ..config.validator import validate_config

logger = logging.getLogger(__name__)


class NoiseCalculator:
    """
    Main calculator for infrared detector noise analysis.
    
    This class orchestrates all noise calculations including
    astrophysical and instrumental sources, and provides
    comprehensive signal-to-noise analysis.
    """
    
    def __init__(self, config: configparser.ConfigParser, incident_flux: Dict):
        """
        Initialize the noise calculator.
        
        Args:
            config: Configuration dictionary containing all parameters
            incident_flux: Optional pre-calculated incident flux dictionary
                          with keys 'wavel' and 'flux'
            
        Raises:
            ValueError: If configuration is invalid
        """

        ipdb.set_trace()
        self.config = config
        self.unit_converter = UnitConverter()
        self.conversion_engine = ConversionEngine(self.unit_converter)
        
        # Store pre-calculated incident flux if provided
        self.incident_flux = incident_flux
        
        # Initialize noise calculators
        self.astrophysical_sources = AstrophysicalSources(config, self.unit_converter)
        #self.instrumental_noise = InstrumentalNoise(config, self.unit_converter)
        
        # Generate wavelength grid
        #self.wavelength = self._generate_wavelength_grid()
        
        logger.info("NoiseCalculator initialized successfully")
        ipdb.set_trace()
    
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
    
    def calculate_snr(self) -> Dict[str, Any]:
        """
        Calculate comprehensive signal-to-noise analysis.
        
        Returns:
            Dictionary containing all calculation results
        """
        logger.info("Starting SNR calculation")
        
        # Get integration time
        integration_time = float(self.config["detector"]["integration_time"])

        ipdb.set_trace()
        
        # Calculate astrophysical noise
        astrophysical_noise_adu = self.astrophysical_sources.calculate_astrophysical_noise_electrons(
            self.wavelength, integration_time
        )
        '''
        astrophysical_noise_adu = self.astrophysical_sources.calculate_astrophysical_noise_adu(
            self.wavelength, integration_time
        )
        '''
        
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
        
        return results
    
    
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