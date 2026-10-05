"""Detector: pixel grid, QE, gain and noise parameters (not the spectrograph dispersion law)."""

from __future__ import annotations

import logging
from typing import Dict

import astropy.units as u
import numpy as np

from .dispersion import DispersionLaw
from .effects import DetectorFrames, _config_get

logger = logging.getLogger(__name__)

# key in Detector.extract() output for non-astrophysical charge
SYSTEMATICS_KEY = "instrum_systematics"


def _parse_dark_current_rates(dark_current_str: str) -> np.ndarray:
    if "," in dark_current_str:
        parts = [float(x.strip()) for x in dark_current_str.split(",")]
        # Historical tables.py behavior: any comma-separated list used as
        # np.arange(start, stop, step) when it has three parts.
        if len(parts) == 3:
            return np.arange(parts[0], parts[1], parts[2])
        return np.asarray(parts, dtype=float)
    return np.fromstring(dark_current_str, sep=",")


class Detector:
    """Sensor model: geometry, quantum efficiency, gain, read noise and dark current."""

    def __init__(self, config: Dict, num_wavel_bins: int):
        self.side_length_pix = int(config["detector"]["size"])
        self.pitch_pix = float(config["detector"]["pitch_pix"])
        self.num_wavel_bins = num_wavel_bins
        self.config = config
        self.quantum_efficiency = float(config["detector"]["quantum_efficiency"])
        self.gain = (
            float(_config_get(config, "detector", "gain", "1.0")) * u.electron / u.adu
        )
        self.e_per_ph = (
            float(_config_get(config, "detector", "e_per_ph", "1.0"))
            * u.electron
            / u.ph
        )

        read_noise_str = _config_get(config, "detector", "read_noise", "0")
        self.read_noise_e_rms = (
            np.fromstring(str(read_noise_str), sep=",") * u.electron / u.pix
        )
        logger.info("Read noise is %s rms", self.read_noise_e_rms)

        dark_current_str = str(_config_get(config, "detector", "dark_current", "0"))
        self.dark_current_rate_e_pix_sec = (
            _parse_dark_current_rates(dark_current_str)
            * u.electron
            / (u.pix * u.second)
        )
        logger.info(
            "Dark current array is %s e-/pix/sec", self.dark_current_rate_e_pix_sec
        )

        self.footprint_cube: np.ndarray | None = None
        self._inv_footprint_image: np.ndarray | None = None

    def set_footprint(self, footprint_cube: np.ndarray) -> None:
        """Attach an illumination footprint from a DispersionLaw."""
        self.footprint_cube = footprint_cube
        # 1 / (footprint summed over bins), zero off the footprint; used for flux-conserving extraction
        footprint_image = DispersionLaw.footprint_image(footprint_cube)
        self._inv_footprint_image = np.divide(
            1.0,
            footprint_image,
            out=np.zeros_like(footprint_image, dtype=float),
            where=footprint_image > 0,
        )

    def apply_qe(self, flux):
        """Apply detector quantum efficiency to a photon-rate quantity or array."""
        return flux * self.quantum_efficiency

    def _require_footprint(self) -> None:
        if self.footprint_cube is None:
            raise RuntimeError(
                "Detector.footprint_cube is unset; call set_footprint() with a DispersionLaw result first"
            )

    def expose(self, astroph_signal: dict, t_frame: u.Quantity) -> DetectorFrames:
        """Paint each source's per-bin photon rate onto the pixel grid for one readout (e-/pix)."""
        self._require_footprint()
        source_e = {}
        for source_name, sig in astroph_signal.items():
            e_pix_per_bin = (
                sig["flux_astro_1d_interpolated_ph_sec_pixel"] * t_frame * self.e_per_ph
            ).to_value(u.electron / u.pix)
            source_e[source_name] = np.tensordot(
                e_pix_per_bin, self.footprint_cube, axes=(0, 0)
            )
        systematics_e = np.zeros(
            (self.side_length_pix, self.side_length_pix), dtype=float
        )
        return DetectorFrames(source_e=source_e, systematics_e=systematics_e)

    def extract(self, frames: DetectorFrames) -> Dict[str, u.Quantity]:
        """
        Per-bin charge (e-) for each source plus ``instrum_systematics``.

        Uses flux-conserving weights footprint / footprint.sum(axis=0), so a pixel
        shared by two bins is split between them and the total charge is conserved.
        """
        self._require_footprint()

        def _per_bin(image_e: np.ndarray) -> u.Quantity:
            return (
                DispersionLaw.weighted_sum_per_bin(
                    self.footprint_cube, image_e * self._inv_footprint_image
                )
                * u.electron
            )

        extracted = {name: _per_bin(img) for name, img in frames.source_e.items()}
        extracted[SYSTEMATICS_KEY] = _per_bin(frames.systematics_e)
        return extracted
