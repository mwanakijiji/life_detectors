"""Detector: pixel grid, QE, and 2D systematics (not the spectrograph dispersion law)."""

from __future__ import annotations

import logging
from typing import Dict

import astropy.io.fits as fits
import numpy as np

from ...utils.loader import config_getboolean

logger = logging.getLogger(__name__)


class Detector:
    """Sensor model: geometry, quantum efficiency, and optional 2D systematics maps."""

    def __init__(self, config: Dict, num_wavel_bins: int):
        self.side_length_pix = int(config["detector"]["size"])
        self.pitch_pix = float(config["detector"]["pitch_pix"])
        self.num_wavel_bins = num_wavel_bins
        self.config = config
        self.quantum_efficiency = float(config["detector"]["quantum_efficiency"])
        self.footprint_cube: np.ndarray | None = None

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

        self.systematics_additive_dict = {
            "read_noise_map": read_noise_map,
            "bias_map": bias_map,
            "cosmic_rays_map": cosmic_rays_map,
            "hot_pixels_map": hot_pixels_map,
        }
        self.systematics_multiplicative_dict = {}
        logging.info("Loaded detector systematics")

    def set_footprint(self, footprint_cube: np.ndarray) -> None:
        """Attach an illumination footprint from a DispersionLaw."""
        self.footprint_cube = footprint_cube

    def apply_qe(self, flux):
        """Apply detector quantum efficiency to a photon-rate quantity or array."""
        return flux * self.quantum_efficiency

    def convert_2d_systematics_to_1d_vector(self) -> np.ndarray:
        """Sum 2D systematics within each wavelength-bin footprint."""
        if self.footprint_cube is None:
            raise RuntimeError(
                "Detector.footprint_cube is unset; call set_footprint() with a DispersionLaw result first"
            )

        canvas_systematics = np.zeros(self.footprint_cube[0, :, :].shape, dtype=float)

        if len(self.systematics_additive_dict) > 0:
            for key, value in self.systematics_additive_dict.items():
                if value is not None:
                    logging.info("Applying %s systematic (additive) to the detector", key)
                    canvas_systematics = canvas_systematics + value

        if len(self.systematics_multiplicative_dict) > 0:
            for key, value in self.systematics_multiplicative_dict.items():
                if value is not None:
                    logging.info(
                        "Applying %s systematic (multiplicative) to the detector", key
                    )
                    canvas_systematics = canvas_systematics * value

        systematics_vector_1d = np.zeros(self.num_wavel_bins)
        for wavel_bin_num in range(self.num_wavel_bins):
            systematics_vector_1d[wavel_bin_num] = np.sum(
                self.footprint_cube[wavel_bin_num, :, :] * canvas_systematics
            )

        return systematics_vector_1d
