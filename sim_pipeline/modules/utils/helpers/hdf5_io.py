"""HDF5 recording helpers."""

import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)


# per-channel datasets that are not signal components (leading "__")
ERROR_TOTAL_KEY = "__error_total"
VARK_TOTAL_KEY = "__vark_total"


def _extracted_group_name(qe: float) -> str:
    # leading "__" so readers that walk dc_*_qe_* groups (snr_from_hdf5) skip it
    return f"__extracted_qe_{canonical_qe(qe):04.2f}"


def _to_value(quantity, unit: str) -> np.ndarray:
    return np.asarray(
        quantity.to_value(unit) if hasattr(quantity, "to_value") else quantity
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
) -> None:
    """Write per-readout extracted spectra (and errors) and bin geometry to an angle HDF5 file."""
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
        n_readouts = None
        for ch_name, spectra in extracted_spectra_adu.items():
            ch_group = group.create_group(ch_name)
            for component, spectrum in spectra.items():
                values = np.asarray(spectrum.to_value("adu"))  # (n_dc, n_readouts, n_bins)
                if values.shape[0] != len(dc_rates):
                    raise ValueError(
                        f"{ch_name}/{component}: {values.shape[0]} DC slices, "
                        f"expected {len(dc_rates)}"
                    )
                n_readouts = values.shape[1]
                dset = ch_group.create_dataset(component, data=values)
                dset.attrs["unit"] = "adu"
            errors = (extracted_errors_adu or {}).get(ch_name)
            if errors is not None:
                dset = ch_group.create_dataset(
                    ERROR_TOTAL_KEY, data=np.asarray(errors["error"].to_value("adu"))
                )
                dset.attrs["unit"] = "adu"
                dset = ch_group.create_dataset(
                    VARK_TOTAL_KEY, data=np.asarray(errors["vark"].to_value("adu2"))
                )
                dset.attrs["unit"] = "adu2"
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
    plot: bool = True,
) -> dict:
    """
    Save one HDF5 file for this rotation angle and QE (any existing file is replaced)

    File structure::

        angle_{angle}_qe_{qe}.hdf5
          __extracted_qe_{qe}/
            dc_rates                    dark-current values (e-/pix/s), one per DC slice
            npix, center, width, edges  bin geometry (pix, um)
            {channel}/{component}       (n_dc, n_readouts, n_bins) ADU per frame
            {channel}/__error_total     (n_dc, n_readouts, n_bins) 1-sigma on the total, ADU
            {channel}/__vark_total      (n_dc, n_readouts, n_bins) Sharp & Birchall Eqn 19 vark, ADU^2
                                        (the two __ datasets only for Sharp & Birchall)
            attrs: angle_deg, qe, t_frame_s, gain_e_per_adu, n_readouts, extraction_method
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
    )
    logger.info(
        f"Recorded angle {angle_deg}: extracted spectra in "
        f"{file_name_hdf5}:{_extracted_group_name(qe)}"
    )

    return
