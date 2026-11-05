"""HDF5 recording helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from ...core.instrumental.dispersion import DispersionLaw
from ...pipeline_registry import pipeline_stage
from .keys import (
    canonical_angle_deg,
    canonical_qe,
    format_angle_qe_hdf5_name,
)

if TYPE_CHECKING:
    from ...core.instrumental.wavel_solution import WavelSolution

logger = logging.getLogger(__name__)


# per-channel datasets that are not signal components (leading "__")
ERROR_TOTAL_KEY = "__error_total"
VARK_TOTAL_KEY = "__vark_total"
# per-channel subgroup holding the same spectra in electrons (leading "__", so readers
# that list a channel's components skip it)
ELECTRONS_KEY = "__electrons"
# group-level datasets / attribute recording the wavelength solution
X_START_KEY = "x_start_pix"
X_END_KEY = "x_end_pix"
WAVEL_PER_COLUMN_KEY = "wavel_per_column"
WAVEL_SOLUTION_FILE_ATTR = "wavel_solution_file"


def _extracted_group_name(qe: float) -> str:
    # leading "__" so readers that walk dc_*_qe_* groups (snr_from_hdf5) skip it
    return f"__extracted_qe_{canonical_qe(qe):04.2f}"


def _to_value(quantity: u.Quantity | np.ndarray, unit: str) -> np.ndarray:
    return np.asarray(
        quantity.to_value(unit) if hasattr(quantity, "to_value") else quantity
    )


def _write_wavel_solution(
    group: h5py.Group, wavel_solution: WavelSolution | None
) -> None:
    """Record the wavelength solution used for these spectra in the extracted group."""
    if wavel_solution is None:
        group.attrs[WAVEL_SOLUTION_FILE_ATTR] = "(none given)"
        return
    group.attrs[WAVEL_SOLUTION_FILE_ATTR] = str(wavel_solution.source)
    if wavel_solution.x_lo is not None and wavel_solution.x_hi is not None:
        for key, values, what in (
            (X_START_KEY, wavel_solution.x_lo, "start"),
            (X_END_KEY, wavel_solution.x_hi, "end"),
        ):
            dset = group.create_dataset(key, data=_to_value(values, "pix"))
            dset.attrs["unit"] = "pix"
            dset.attrs["description"] = (
                f"Detector x at the {what} of each wavelength bin (continuous; "
                "pixel column k spans [k, k+1))"
            )
    if wavel_solution.wavel_col is not None:
        dset = group.create_dataset(
            WAVEL_PER_COLUMN_KEY, data=_to_value(wavel_solution.wavel_col, "um")
        )
        dset.attrs["unit"] = "um"
        dset.attrs["description"] = (
            "Wavelength at the centre (x = k + 0.5) of each detector column k; NaN "
            "off the trace"
        )


def _write_extracted_spectra(
    file_name_hdf5: str,
    *,
    angle_deg: float,
    qe: float,
    output_channels: dict,
    t_frame: u.Quantity,
    extracted_spectra_adu: dict,
    extracted_errors_adu: dict | None,
    extraction_method: str,
    wavel_solution: WavelSolution | None = None,
) -> None:
    """Write per-readout extracted spectra (and errors), bin geometry and the
    wavelength solution to an angle HDF5 file, in ADU and (in each channel's
    __electrons subgroup) in electrons.
    """
    ref_channel = output_channels["output_3_dark"]
    det = ref_channel.detector
    dc_rates = np.atleast_1d(
        det.dark_current_rate_e_pix_sec.to_value(u.electron / (u.pix * u.s))
    )
    with h5py.File(file_name_hdf5, "w") as handle:
        group = handle.create_group(_extracted_group_name(qe))
        group.attrs["angle_deg"] = float(angle_deg)
        group.attrs["qe"] = float(qe)
        group.attrs["t_frame_s"] = float(t_frame.to_value(u.s))
        group.attrs["gain_e_per_adu"] = float(det.gain.to_value(u.electron / u.adu))
        group.attrs["extraction_method"] = str(extraction_method)
        group.create_dataset("dc_rates", data=dc_rates)
        group.create_dataset(
            "npix", data=DispersionLaw.n_pix_per_bin(det.footprint_cube).to_value(u.pix)
        )
        group.create_dataset("center", data=_to_value(ref_channel.bin_centers, "um"))
        group.create_dataset("width", data=_to_value(ref_channel.bin_widths, "um"))
        group.create_dataset("edges", data=_to_value(ref_channel.bin_edges, "um"))
        _write_wavel_solution(group, wavel_solution)
        n_readouts = None
        for ch_name, spectra in extracted_spectra_adu.items():
            ch_group = group.create_group(ch_name)
            e_group = ch_group.create_group(ELECTRONS_KEY)
            # this channel's own gain (e-/ADU), to convert ADU -> electrons
            gain = float(
                output_channels[ch_name].detector.gain.to_value(u.electron / u.adu)
            )
            e_group.attrs["gain_e_per_adu"] = gain
            for component, spectrum in spectra.items():
                values = np.asarray(
                    spectrum.to_value("adu")
                )  # (n_dc, n_readouts, n_bins)
                if values.shape[0] != len(dc_rates):
                    raise ValueError(
                        f"{ch_name}/{component}: {values.shape[0]} DC slices, "
                        f"expected {len(dc_rates)}"
                    )
                n_readouts = values.shape[1]
                dset = ch_group.create_dataset(component, data=values)
                dset.attrs["unit"] = "adu"
                dset = e_group.create_dataset(component, data=values * gain)
                dset.attrs["unit"] = "electron"
            errors = (extracted_errors_adu or {}).get(ch_name)
            if errors is not None:
                error_adu = np.asarray(errors["error"].to_value("adu"))
                vark_adu2 = np.asarray(errors["vark"].to_value("adu2"))
                dset = ch_group.create_dataset(ERROR_TOTAL_KEY, data=error_adu)
                dset.attrs["unit"] = "adu"
                dset = ch_group.create_dataset(VARK_TOTAL_KEY, data=vark_adu2)
                dset.attrs["unit"] = "adu2"
                dset = e_group.create_dataset(ERROR_TOTAL_KEY, data=error_adu * gain)
                dset.attrs["unit"] = "electron"
                dset = e_group.create_dataset(
                    VARK_TOTAL_KEY, data=vark_adu2 * gain**2
                )
                dset.attrs["unit"] = "electron2"
        group.attrs["n_readouts"] = int(n_readouts or 0)


@pipeline_stage(depends_on=("extract_spectra_from_readouts",))
def record_info_at_angle_and_qe(
    *,
    angle_deg: float,
    qe: float,
    output_channels: dict,
    save_dir: str,
    t_frame: u.Quantity,
    extracted_spectra_adu: dict,
    extracted_errors_adu: dict | None = None,
    extraction_method: str = "sharp_birchall",
    wavel_solution: WavelSolution | None = None,
) -> None:
    """
    Save one HDF5 file for this rotation angle and QE (any existing file is replaced)

    File structure::

        angle_{angle}_qe_{qe}.hdf5
          __extracted_qe_{qe}/
            dc_rates                    dark-current values (e-/pix/s), one per DC slice
            npix, center, width, edges  bin geometry (pix, um)
            x_start_pix, x_end_pix      (n_bins,) detector x of each bin's start / end
                                        (continuous; column k spans k..k+1)
            wavel_per_column            (n_pix_x,) wavelength (um) at the centre of each
                                        detector column (x = k + 0.5); NaN off trace
                                        (these three only if the wavelength solution
                                        gives them)

            {channel}/{component}       (n_dc, n_readouts, n_bins) ADU per frame
            {channel}/__error_total     (n_dc, n_readouts, n_bins) 1-sigma on the total,
            ADU
            {channel}/__vark_total      (n_dc, n_readouts, n_bins) Sharp & Birchall Eqn
            19 vark, ADU^2
                                        (the two __ datasets only for Sharp & Birchall)
            {channel}/__electrons/{component}
                                        (n_dc, n_readouts, n_bins) electrons per frame
                                        (= ADU per frame x the channel's gain)
            {channel}/__electrons/__error_total, __vark_total
                                        the same, in e- and e-^2 (Sharp & Birchall only)
            {channel}/__electrons attrs: gain_e_per_adu
            attrs: angle_deg, qe, t_frame_s, gain_e_per_adu, n_readouts,
                extraction_method, wavel_solution_file (path of the wavelength-
                solution FITS used, "channel bin edges" if none was found, or
                "(none given)")
    """

    file_name_hdf5 = f"{save_dir}{format_angle_qe_hdf5_name(angle_deg, qe)}"
    angle_deg = canonical_angle_deg(angle_deg)
    qe = canonical_qe(qe)
    Path(file_name_hdf5).parent.mkdir(parents=True, exist_ok=True)

    _write_extracted_spectra(
        file_name_hdf5,
        angle_deg=angle_deg,
        qe=qe,
        output_channels=output_channels,
        t_frame=t_frame,
        extracted_spectra_adu=extracted_spectra_adu,
        extracted_errors_adu=extracted_errors_adu,
        extraction_method=extraction_method,
        wavel_solution=wavel_solution,
    )
    logger.info(
        "Recorded angle %s: extracted spectra in %s:%s",
        angle_deg,
        file_name_hdf5,
        _extracted_group_name(qe),
    )
