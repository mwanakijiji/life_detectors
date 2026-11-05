"""Read the spectral wavelength solution (written by DispersionLaw.write_wavel_solution)
and attach it to extracted spectra."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict

import astropy.units as u
import numpy as np
from astropy.io import fits

from ...utils.helpers.output_dirs import INTERMEDIATE, resolve_in_run_dir
from ...utils.types import ConfigLike
from .effects import _config_get

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WavelSolution:
    """
    Wavelength of each extracted bin (all (n_bins,) Quantities in um), and, if known:
    each bin's start/end position along the detector x axis (continuous pixels;
    pixel k spans [k, k + 1)), and the wavelength at the centre of every detector
    column (x = k + 0.5; (n_pix_x,) um, NaN off the trace).
    """

    wavel_lo: u.Quantity
    wavel_cen: u.Quantity
    wavel_hi: u.Quantity
    width: u.Quantity
    source: str  # file it was read from, or a note on where it came from
    x_lo: u.Quantity | None = None  # (n_bins,) pix
    x_hi: u.Quantity | None = None  # (n_bins,) pix
    wavel_col: u.Quantity | None = None  # (n_pix_x,) um

    @property
    def n_bins(self) -> int:
        """Number of wavelength bins."""
        return int(self.wavel_cen.size)

    @classmethod
    def from_bin_edges(
        cls,
        bin_edges: u.Quantity,
        source: str,
        x_edges: np.ndarray | u.Quantity | None = None,
        wavel_col: u.Quantity | None = None,
    ) -> "WavelSolution":
        """Build from bin edges (n_bins + 1,), with geometric-mean bin centres, and
        optionally the bins' x edges on the detector (n_bins + 1,) in pixels and the
        per-column wavelengths (n_pix_x,)."""
        edges = u.Quantity(bin_edges).to(u.um)
        x_lo = x_hi = None
        if x_edges is not None:
            x_edges = u.Quantity(x_edges, u.pix)
            x_lo, x_hi = x_edges[:-1], x_edges[1:]
        return cls(
            wavel_lo=edges[:-1],
            wavel_cen=np.sqrt(edges[:-1] * edges[1:]),
            wavel_hi=edges[1:],
            width=edges[1:] - edges[:-1],
            source=source,
            x_lo=x_lo,
            x_hi=x_hi,
            wavel_col=None if wavel_col is None else u.Quantity(wavel_col, u.um),
        )


def wavel_solution_file_from_config(config: ConfigLike) -> str | None:
    """
    Path of the wavelength-solution FITS to read: [extraction]
    file_name_wavel_solution_read if set, else file_name_wavel_solution_write (the
    file this run writes in define_dispersion_on_detectors); None if neither is set.

    A bare file name (no directory) refers to this run's intermediate/ directory, as
    when it is written; a full path is used as given.
    """
    for key in ("file_name_wavel_solution_read", "file_name_wavel_solution_write"):
        value = _config_get(config, "extraction", key)
        if value is not None and str(value).strip():
            value = str(value).strip()
            if os.path.dirname(value):
                return value
            try:
                return resolve_in_run_dir(config, value, INTERMEDIATE)
            except (KeyError, TypeError):
                # no [dirs] output root (e.g. a minimal test config)
                return value
    return None


def read_wavel_solution(file_name: str) -> WavelSolution:
    """Read the BINS table (and WAVEL_COL, if present) of a wavelength-solution FITS
    file."""
    with fits.open(file_name) as hdul:
        bins = hdul["BINS"].data
        names = bins.columns.names
        has_x = "X_START_PIX" in names and "X_END_PIX" in names
        wavel_col = (
            np.asarray(hdul["WAVEL_COL"].data, dtype=float) * u.um
            if "WAVEL_COL" in hdul
            else None
        )
        solution = WavelSolution(
            wavel_lo=np.asarray(bins["WAVEL_LO_UM"], dtype=float) * u.um,
            wavel_cen=np.asarray(bins["WAVEL_CEN_UM"], dtype=float) * u.um,
            wavel_hi=np.asarray(bins["WAVEL_HI_UM"], dtype=float) * u.um,
            width=np.asarray(bins["WIDTH_UM"], dtype=float) * u.um,
            source=str(file_name),
            x_lo=(
                np.asarray(bins["X_START_PIX"], dtype=float) * u.pix if has_x else None
            ),
            x_hi=(
                np.asarray(bins["X_END_PIX"], dtype=float) * u.pix if has_x else None
            ),
            wavel_col=wavel_col,
        )
    logger.info(
        "Read wavelength solution (%d bins, %.3f-%.3f um) from %s",
        solution.n_bins,
        solution.wavel_lo[0].value,
        solution.wavel_hi[-1].value,
        file_name,
    )
    return solution


def apply_wavel_solution(
    extracted_spectra_adu: Dict[str, Dict[str, u.Quantity]],
    solution: WavelSolution,
    extracted_errors_adu: Dict[str, Dict[str, u.Quantity]] | None = None,
) -> Dict[str, dict]:
    """
    Put the extracted spectra on the wavelength solution.

    Args:
        extracted_spectra_adu: {channel: {component: (n_dc, n_readouts, n_bins) ADU}}
        solution: wavelength of each bin
        extracted_errors_adu: optional {channel: {"error": (n_dc, n_readouts,
            n_bins) ADU, ...}} (Sharp & Birchall only)

    Returns:
        {channel: {
            "wavel", "wavel_lo", "wavel_hi", "width": (n_bins,) um,
            "x_lo_pix", "x_hi_pix": (n_bins,) detector x of each bin's ends, or None,
            "spectra_adu": {component: (n_dc, n_readouts, n_bins) ADU per bin},
            "spectra_adu_per_um": {component: same / bin width, ADU / um},
            "error_adu", "error_adu_per_um": total-spectrum 1-sigma, or None,
        }}

    Raises ValueError if a channel's number of bins differs from the solution's.
    """
    extracted_errors_adu = extracted_errors_adu or {}
    out: Dict[str, dict] = {}
    for channel_name, spectra in extracted_spectra_adu.items():
        for component, spectrum in spectra.items():
            if spectrum.shape[-1] != solution.n_bins:
                raise ValueError(
                    f"{channel_name}/{component} has {spectrum.shape[-1]} wavelength "
                    f"bins, but the wavelength solution ({solution.source}) has "
                    f"{solution.n_bins}"
                )
        error = extracted_errors_adu.get(channel_name, {}).get("error")
        out[channel_name] = {
            "wavel": solution.wavel_cen,
            "wavel_lo": solution.wavel_lo,
            "wavel_hi": solution.wavel_hi,
            "width": solution.width,
            "x_lo_pix": solution.x_lo,
            "x_hi_pix": solution.x_hi,
            "spectra_adu": spectra,
            "spectra_adu_per_um": {
                component: spectrum / solution.width
                for component, spectrum in spectra.items()
            },
            "error_adu": error,
            "error_adu_per_um": None if error is None else error / solution.width,
        }
    return out
