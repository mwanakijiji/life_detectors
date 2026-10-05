"""Optical dispersion: map wavelength bins onto a pixel grid.

Independent of detector QE/noise.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Sequence

import astropy.units as u
import numpy as np
import scipy.sparse

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

    def make_footprint(
        self, side_length_pix: int, num_wavel_bins: int, method: str = "sparse"
    ) -> np.ndarray | scipy.sparse.csr_matrix:
        """Return the wavelength-bin footprint on the detector.

        With method="sparse" (default), returns a CSR matrix of shape
        (n_bins, n_pix * n_pix), each row a row-major (y, x) flattening of that
        bin's footprint. With method="dense", returns a cube of shape
        (n_bins, n_pix, n_pix). Values are 1 for fully covered pixels and
        fractional coverage on edges.
        """

        if method == "dense":
            # old, inefficient way
            footprint_cube = np.full(
                (num_wavel_bins, side_length_pix, side_length_pix), 0.0, dtype=float
            )

            for wavel_bin_num in range(num_wavel_bins):
                # assumes horizontal spectra
                footprint_this = np.full(
                    (side_length_pix, side_length_pix), 0.0, dtype=float
                )
                starting_pixel_this = self.starting_pixel + np.array(
                    [0.0, wavel_bin_num * self.pix_per_wavel_bin]
                )

                pixel_ceil_start_x = int(np.ceil(starting_pixel_this[1]))
                pixel_frac_start_x = pixel_ceil_start_x - starting_pixel_this[1]
                pixel_floor_end_x = int(
                    np.floor(starting_pixel_this[1] + self.pix_per_wavel_bin)
                )
                pixel_frac_end_x = (
                    starting_pixel_this[1] + self.pix_per_wavel_bin
                ) - pixel_floor_end_x

                y0 = int(starting_pixel_this[0])
                y1 = int(starting_pixel_this[0] + self.pix_spectral_width)
                footprint_this[
                    y0:y1, int(pixel_ceil_start_x) : int(pixel_floor_end_x)
                ] = 1.0
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

        elif method == "sparse":
            # shape (num_wavel_bins, side_length_pix * side_length_pix)
            # vectorized: compute all bin start/end columns at once (they're a linear
            # function of wavel_bin_num), then build row/col/data arrays for COO
            # construction in one shot — no per-bin Python-level array allocation.
            side = side_length_pix
            bin_idx = np.arange(num_wavel_bins)
            x_start = self.starting_pixel[1] + bin_idx * self.pix_per_wavel_bin
            x_end = x_start + self.pix_per_wavel_bin

            # same edge logic as the dense loop
            ceil_start = np.ceil(x_start).astype(int)
            floor_end = np.floor(x_end).astype(int)
            frac_start = ceil_start - x_start
            frac_end = x_end - floor_end

            # each bin touches columns ceil_start-1 (frac_start),
            # [ceil_start, floor_end) (1.0), and floor_end (frac_end); the count of
            # full columns can differ by 1 between bins
            n_cols = (floor_end - ceil_start) + 2
            col_bin = np.repeat(bin_idx, n_cols)
            offset = np.arange(n_cols.sum()) - np.repeat(
                np.cumsum(n_cols) - n_cols, n_cols
            )
            col_x = ceil_start[col_bin] - 1 + offset
            col_val = np.where(
                offset == 0,
                frac_start[col_bin],
                np.where(offset == n_cols[col_bin] - 1, frac_end[col_bin], 1.0),
            )

            # broadcast across the spectral-width rows (constant for all bins)
            y0 = int(self.starting_pixel[0])
            y1 = int(self.starting_pixel[0] + self.pix_spectral_width)
            ys = np.arange(y0, y1)
            rows = np.repeat(col_bin, ys.size)
            cols = (ys[None, :] * side + col_x[:, None]).ravel()
            data = np.repeat(col_val, ys.size)
            col_x_rep = np.repeat(col_x, ys.size)

            # drop zero-weight edges and anything off the detector
            keep = (data != 0) & (col_x_rep >= 0) & (col_x_rep < side)
            footprint_sparse = scipy.sparse.coo_matrix(
                (data[keep], (rows[keep], cols[keep])),
                shape=(num_wavel_bins, side * side),
            ).tocsr()
            logging.info(
                "Total dispersion footprint is %s pixels", footprint_sparse.sum()
            )
            return footprint_sparse

        else:
            raise ValueError(
                f"Unknown footprint method {method!r}; use 'dense' or 'sparse'"
            )

        return footprint_cube

    @staticmethod
    def n_pix_per_bin(
        footprint_cube: np.ndarray | scipy.sparse.csr_matrix,
    ) -> u.Quantity:
        """Pixels (or fractional pixels) illuminated in each wavelength bin."""
        if scipy.sparse.issparse(footprint_cube):
            footprint_pixel_count = np.asarray(footprint_cube.sum(axis=1)).ravel()
        else:
            footprint_pixel_count = np.sum(footprint_cube, axis=(1, 2))
        if hasattr(footprint_pixel_count, "unit"):
            return footprint_pixel_count.to(u.pix)
        return footprint_pixel_count * u.pix

    @staticmethod
    def footprint_image(
        footprint: np.ndarray | scipy.sparse.csr_matrix,
    ) -> np.ndarray:
        """Footprint summed over wavelength bins, as a 2D (n_pix, n_pix) image."""
        if scipy.sparse.issparse(footprint):
            side = math.isqrt(footprint.shape[1])
            return np.asarray(footprint.sum(axis=0)).reshape(side, side)
        return np.sum(footprint, axis=0)

    @staticmethod
    def weighted_sum_per_bin(
        footprint: np.ndarray | scipy.sparse.csr_matrix, image_2d: np.ndarray
    ) -> np.ndarray:
        """Sum of footprint * image_2d within each wavelength bin."""
        if scipy.sparse.issparse(footprint):
            return np.asarray(footprint @ np.ravel(image_2d)).ravel()
        return np.tensordot(footprint, image_2d, axes=([1, 2], [0, 1]))
