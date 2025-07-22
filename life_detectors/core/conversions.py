"""
Conversion engine for the life_detectors package.

This module handles unit conversions and signal-to-noise calculations
for the noise analysis pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..data.units import UnitConverter

logger = logging.getLogger(__name__)

@dataclass
class ConversionEngine:
    """
    Handles unit conversions and signal-to-noise calculations.
    
    This class provides utilities for converting between different
    units and calculating signal-to-noise ratios.
    """
    
    def __init__(self, unit_converter: UnitConverter):
        """
        Initialize conversion engine.
        
        Args:
            unit_converter: Unit conversion utility
        """
        self.unit_converter = unit_converter
    
    def calculate_signal_to_noise(
        self, 
        signal_adu: np.ndarray, 
        total_noise_adu: np.ndarray
    ) -> np.ndarray:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            signal_adu: Signal in ADU per pixel
            total_noise_adu: Total noise in ADU per pixel
            
        Returns:
            Signal-to-noise ratio (dimensionless)
        """
        # Avoid division by zero
        noise_safe = np.where(total_noise_adu > 0, total_noise_adu, np.inf)
        snr = signal_adu / noise_safe
        
        # Set SNR to 0 where noise is infinite
        snr = np.where(np.isinf(noise_safe), 0.0, snr)
        
        return snr
    
    def calculate_total_noise_adu(
        self, 
        astrophysical_noise_adu: np.ndarray, 
        instrumental_noise_adu: float
    ) -> np.ndarray:
        """
        Calculate total noise in ADU by combining astrophysical and instrumental sources.
        
        Args:
            astrophysical_noise_adu: Astrophysical noise in ADU per pixel (array)
            instrumental_noise_adu: Instrumental noise in ADU per pixel (scalar)
            
        Returns:
            Total noise in ADU per pixel
        """
        # Add noise sources in quadrature
        total_noise_squared = astrophysical_noise_adu ** 2 + instrumental_noise_adu ** 2
        total_noise = np.sqrt(total_noise_squared)
        
        return total_noise
    
    def calculate_detection_limit(
        self, 
        total_noise_adu: np.ndarray, 
        snr_threshold: float = 5.0
    ) -> np.ndarray:
        """
        Calculate detection limit based on noise and SNR threshold.
        
        Args:
            total_noise_adu: Total noise in ADU per pixel
            snr_threshold: Minimum SNR for detection (default: 5.0)
            
        Returns:
            Detection limit in ADU per pixel
        """
        detection_limit = total_noise_adu * snr_threshold
        
        return detection_limit
    
    def convert_flux_to_electrons(
        self, 
        flux_photons_per_sec_m2_um: np.ndarray, 
        collecting_area: float, 
        throughput: float, 
        pixel_area_sr: float, 
        integration_time: float
    ) -> np.ndarray:
        """
        Convert flux to electrons at the detector.
        
        Args:
            flux_photons_per_sec_m2_um: Flux in photons/sec/m^2/micron
            collecting_area: Telescope collecting area in m^2
            throughput: Optical throughput (dimensionless)
            pixel_area_sr: Pixel solid angle in steradians
            integration_time: Integration time in seconds
            
        Returns:
            Electrons per pixel
        """
        # Convert to photons per pixel per second
        photons_per_pixel_per_sec = (
            flux_photons_per_sec_m2_um * 
            collecting_area * 
            throughput * 
            pixel_area_sr
        )
        
        # Convert to electrons (assuming 1 photon = 1 electron)
        ## ## TO DO: UPDATE THIS
        electrons_per_pixel = photons_per_pixel_per_sec * integration_time
        
        return electrons_per_pixel
    
    def convert_electrons_to_adu(
        self, 
        electrons_per_pixel: np.ndarray, 
        gain: float
    ) -> np.ndarray:
        """
        Convert electrons to ADU.
        
        Args:
            electrons_per_pixel: Electrons per pixel
            gain: Detector gain in e-/ADU
            
        Returns:
            ADU per pixel
        """
        adu_per_pixel = self.unit_converter.electrons_to_adu(electrons_per_pixel, gain)
        
        return adu_per_pixel
    
    def calculate_integrated_snr(
        self, 
        snr_per_wavelength: np.ndarray, 
        wavelength: np.ndarray
    ) -> float:
        """
        Calculate integrated signal-to-noise over wavelength range.
        
        Args:
            snr_per_wavelength: SNR at each wavelength
            wavelength: Wavelength array in microns
            
        Returns:
            Integrated SNR
        """
        # Integrate SNR^2 over wavelength (SNR adds in quadrature)
        snr_squared_integrated = np.trapz(snr_per_wavelength ** 2, wavelength)
        integrated_snr = np.sqrt(snr_squared_integrated)
        
        return integrated_snr
    
    def calculate_optimal_integration_time(
        self, 
        astrophysical_noise_rate: float, 
        instrumental_noise: float,
        target_snr: float = 5.0
    ) -> float:
        """
        Calculate optimal integration time for a target SNR.
        
        Args:
            astrophysical_noise_rate: Astrophysical noise rate (ADU/pixel/sec)
            instrumental_noise: Instrumental noise (ADU/pixel)
            target_snr: Target signal-to-noise ratio
            
        Returns:
            Optimal integration time in seconds
        """
        # For shot noise limited case: SNR = signal / sqrt(signal + instrumental_noise^2)
        # Solving for integration time when SNR = target_snr
        
        if astrophysical_noise_rate <= 0:
            return np.inf  # No astrophysical signal
        
        # Solve quadratic equation for integration time
        # SNR^2 = (signal * t)^2 / (signal * t + instrumental_noise^2)
        # This gives: t = (SNR^2 * instrumental_noise^2) / (signal * (SNR^2 - signal))
        
        signal_rate = astrophysical_noise_rate ** 2  # Convert noise rate to signal rate
        
        if signal_rate <= 0:
            return np.inf
        
        optimal_time = (target_snr ** 2 * instrumental_noise ** 2) / (signal_rate * (target_snr ** 2 - signal_rate))
        
        # Ensure positive integration time
        if optimal_time <= 0:
            return np.inf
        
        return optimal_time 