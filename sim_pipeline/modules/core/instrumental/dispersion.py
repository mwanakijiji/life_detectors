"""Optical dispersion: map wavelength bins onto a pixel grid (independent of detector QE/noise)."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)


class DispersionLaw:
    """Maps spectral bins to detector-pixel footprints.

    Physically this is a spectrograph / illumination law. It needs a pixel-grid
    geometry to realize the map, but it does not own QE, gain, or noise.
    """

    def __init__(
        self,
        config: Dict,
        *,
        starting_pixel: Sequence[float] | None = None,
    ):
        self.config = config
        self.pix_per_wavel_bin = float(config["detector"]["pix_per_wavel_bin"])
        self.pix_spectral_width = int(config["detector"]["pix_spectral_width"])
        # lower-left corner of the first wavelength-bin footprint (y, x)
        self.starting_pixel = np.asarray(
            starting_pixel if starting_pixel is not None else (100.0, 300.0),
            dtype=float,
        )

    def make_footprint(self, side_length_pix: int, num_wavel_bins: int) -> np.ndarray:
        """Return footprint cube of shape (n_bins, n_pix, n_pix).

        Values are 1 for fully covered pixels and fractional coverage on edges.
        """
        footprint_cube = np.full(
            (num_wavel_bins, side_length_pix, side_length_pix), 0.0, dtype=float
        )

        for wavel_bin_num in range(num_wavel_bins):
            # assumes horizontal spectra
            footprint_this = np.full((side_length_pix, side_length_pix), 0.0, dtype=float)
            starting_pixel_this = self.starting_pixel + np.array(
                [0.0, wavel_bin_num * self.pix_per_wavel_bin]
            )

            pixel_ceil_start_x = int(np.ceil(starting_pixel_this[1]))
            pixel_frac_start_x = pixel_ceil_start_x - starting_pixel_this[1]
            pixel_floor_end_x = int(np.floor(starting_pixel_this[1] + self.pix_per_wavel_bin))
            pixel_frac_end_x = (starting_pixel_this[1] + self.pix_per_wavel_bin) - pixel_floor_end_x

            y0 = int(starting_pixel_this[0])
            y1 = int(starting_pixel_this[0] + self.pix_spectral_width)
            footprint_this[y0:y1, int(pixel_ceil_start_x) : int(pixel_floor_end_x)] = 1.0
            footprint_this[y0:y1, int(pixel_ceil_start_x) - 1] = pixel_frac_start_x
            footprint_this[y0:y1, int(pixel_floor_end_x)] = pixel_frac_end_x

            footprint_cube[wavel_bin_num, :, :] = footprint_this
            logging.info(
                "Wavelength bin %s dispersion footprint is %s pixels",
                wavel_bin_num,
                footprint_this.sum(),
            )

        footprint_sum = np.sum(footprint_cube, axis=0)
        logging.info("Total dispersion footprint is %s pixels", footprint_sum.sum())
        return footprint_cube

    @staticmethod
    def n_pix_per_bin(footprint_cube: np.ndarray) -> u.Quantity:
        """Pixels (or fractional pixels) illuminated in each wavelength bin."""
        footprint_pixel_count = np.sum(footprint_cube, axis=(1, 2))
        if hasattr(footprint_pixel_count, "unit"):
            return footprint_pixel_count.to(u.pix)
        return footprint_pixel_count * u.pix
