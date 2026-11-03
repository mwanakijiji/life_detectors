"""
Sharp & Birchall (2010, PASA 27, 91) optimal extraction of spectra from a 2D readout.

Ported from glint-control/sharp_birchall_extraction/utils/backbone_classes.py
(``Extractor``), keeping the matrix maths and the per-column solve, without the
file I/O, plotting and multiprocessing of the original.

Notation (as in the original): for each detector column k, the pixels along the
column (index j) are modelled as a sum of spectral profiles phi_i (index i) with
fluxes eta_i. Dispersion runs along x, so each column is one spectral element.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _replace_non_finite_with_median(array: np.ndarray) -> np.ndarray:
    # as in the original: let the solve go through instead of failing on infs/NaNs
    finite = np.isfinite(array)
    if not finite.all() and finite.any():
        array = array.copy()
        array[~finite] = np.median(array[finite])
    return array


def floor_variance(variance: np.ndarray) -> np.ndarray:
    """Replace zero/negative/non-finite variances with
    the median of the positive ones.
    """
    variance = np.asarray(variance, dtype=float).copy()
    good = np.isfinite(variance) & (variance > 0)
    if not good.any():
        raise ValueError("Variance image has no positive finite values")
    variance[~good] = np.median(variance[good])
    return variance


class SharpBirchallExtractor:
    """
    Readout-independent parts of the extraction, built once for a profile set and
    a variance image; ``extract`` then needs only the data frame.

    Args:
        profiles: (n_spec, ny, nx) spectral profiles, each normalized so it sums to
            1 down each illuminated column (so eta is the column's total flux).
        variance: (ny, nx) per-pixel variance, in the data's units squared.
        read_noise: read noise (data units, rms), used only for vark (Eqn 19).
    """

    def __init__(
        self, profiles: np.ndarray, variance: np.ndarray, read_noise: float = 0.0
    ):
        self.phi = np.asarray(profiles, dtype=float)  # (i, j, k) = (spec, y, x)
        if self.phi.ndim != 3:
            raise ValueError("profiles must have shape (n_spec, ny, nx)")
        self.num_spec = self.phi.shape[0]
        self.variance = floor_variance(variance)
        self.n_rd = float(read_noise)

        # S^-2
        self.S_inv_squared = 1.0 / self.variance
        phi_S = self.phi * self.S_inv_squared
        phi_T = np.transpose(self.phi, (1, 0, 2))

        # Sharp+ Eqn. 9: C_il = sum_j phi_ij phi_lj / S_j^2, per column k
        self.c_matrix = _replace_non_finite_with_median(
            np.einsum("ijk,jlk->ilk", phi_S, phi_T)
        )
        # Sharp+ Eqn. 19: C'_il = sum_j phi_ij phi_lj ; b'_i = sum_j phi_ij (S_j^2 -
        # n_rd^2)
        self.c_mat_prime = np.einsum("ijk,jlk->ilk", self.phi, phi_T)
        self.b_mat_prime = np.einsum(
            "ijk,jk->ik", self.phi, self.variance - self.n_rd**2
        )

        # Per-column pseudo-inverses (minimum-norm least-squares solutions, as the
        # original's per-column lstsq), stacked as (n_cols, n_spec, n_spec)
        self._c_pinv = self._pinv_per_column(self.c_matrix)
        c_prime_pinv = self._pinv_per_column(self.c_mat_prime)

        # 1-sigma^2 on eta: diag(C^-1); includes every term in the variance image
        self.var_flux = np.einsum("kii->ik", self._c_pinv)  # (n_spec, n_cols)
        # vark (Eqn 19): estimated signal variance of each spectrum (read noise
        # removed);
        # independent of the data frame, so solved once here
        self.vark = np.einsum("kil,lk->ik", c_prime_pinv, self.b_mat_prime)

    def _pinv_per_column(self, matrix: np.ndarray) -> np.ndarray:
        stacked = np.moveaxis(matrix, -1, 0)  # (n_cols, n_spec, n_spec)
        try:
            return np.linalg.pinv(stacked)
        except np.linalg.LinAlgError:
            out = np.full_like(stacked, np.nan)
            for k, mat in enumerate(stacked):
                try:
                    out[k] = np.linalg.pinv(mat)
                except np.linalg.LinAlgError:
                    logger.warning("Sharp & Birchall solve failed in column %d", k)
            return out

    def extract(self, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the spectra from one data frame.

        Args:
            D: (ny, nx) data frame, or (n_frames, ny, nx) frames sharing this variance.

        Returns:
            flux: (n_spec, nx), or (n_frames, n_spec, nx)
            var_flux: (n_spec, nx) variance of flux, diag(C^-1)
            vark: (n_spec, nx) Eqn 19 signal-variance estimate
        """
        D = np.asarray(D, dtype=float)
        single = D.ndim == 2
        frames = D[np.newaxis] if single else D

        # Sharp+ Eqn. 10: b_i = sum_j phi_ij D_j / S_j^2, per column k
        b = np.einsum("ijk,fjk->fik", self.phi, frames * self.S_inv_squared)
        b = np.stack([_replace_non_finite_with_median(b_f) for b_f in b])
        flux = np.einsum("kil,flk->fik", self._c_pinv, b)

        return (flux[0] if single else flux), self.var_flux, self.vark


def rebin_columns_to_bins(
    fraction: np.ndarray, flux: np.ndarray, var: np.ndarray | None = None
):
    """
    Rebin per-column values into wavelength bins.

    Args:
        fraction: (n_bins, nx) share of each column's footprint belonging to each bin
        flux: (..., nx) per-column flux
        var: (..., nx) per-column variance, or None

    Returns: flux_b = fraction @ flux, and var_b = fraction**2 @ var (or None)
    """
    flux_b = np.asarray(flux) @ fraction.T
    var_b = None if var is None else np.asarray(var) @ (fraction**2).T
    return flux_b, var_b
