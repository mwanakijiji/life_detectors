"""
Instrumental noise calculations for the modules package.

This module handles calculations of instrumental noise sources including
dark current, read noise, and other detector effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..data.units import UnitConverter

logger = logging.getLogger(__name__)

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