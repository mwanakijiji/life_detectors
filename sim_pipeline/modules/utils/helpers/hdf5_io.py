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
) -> None:
    """Write per-readout extracted spectra and bin geometry to an angle HDF5 file."""
    ref_channel = output_channels["output_3_dark"]
    det = ref_channel.detector
    with h5py.File(file_name_hdf5, "w") as handle:
        group = handle.create_group(_extracted_group_name(qe))
        group.attrs["angle_deg"] = float(angle_deg)
        group.attrs["qe"] = float(qe)
        group.attrs["t_frame_s"] = float(t_frame.to_value(u.s))
        group.attrs["gain_e_per_adu"] = float(det.gain.to_value(u.electron / u.adu))
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
                values = np.asarray(spectrum.to_value("adu"))  # (n_readouts, n_bins)
                n_readouts = values.shape[0]
                dset = ch_group.create_dataset(component, data=values)
                dset.attrs["unit"] = "adu"
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
    plot: bool = True,
) -> dict:
    """
    Save one HDF5 file for this rotation angle and QE (any existing file is replaced)

    File structure::

        angle_{angle}_qe_{qe}.hdf5
          __extracted_qe_{qe}/
            npix, center, width, edges  bin geometry (pix, um)
            {channel}/{component}       (n_readouts, n_bins) ADU per frame, box-summed from readouts
            attrs: angle_deg, qe, t_frame_s, gain_e_per_adu, n_readouts
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
    )
    logger.info(
        f"Recorded angle {angle_deg}: extracted spectra in "
        f"{file_name_hdf5}:{_extracted_group_name(qe)}"
    )

    return
