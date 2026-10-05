"""HDF5 recording helpers."""

import logging
from pathlib import Path

import h5py
import numpy as np
from astropy.table import QTable

from ...pipeline_registry import pipeline_stage
from .keys import (
    canonical_angle_deg,
    canonical_dc_rate,
    canonical_qe,
    format_angle_qe_hdf5_name,
)

logger = logging.getLogger(__name__)


def _extracted_group_name(qe: float) -> str:
    # leading "__" so readers that walk the dc_*_qe_* groups skip it
    return f"__extracted_qe_{canonical_qe(qe):04.2f}"


def _write_extracted_spectra(
    file_name_hdf5: str,
    *,
    angle_deg: float,
    qe: float,
    output_channels: dict,
    extracted_spectra_adu: dict,
) -> None:
    """Append per-readout extracted spectra and bin geometry to an angle HDF5 file."""
    ref_channel = output_channels["output_3_dark"]
    ref_table = next(iter(ref_channel.tables_by_dark_current.values()))
    with h5py.File(file_name_hdf5, "a") as handle:
        group = handle.create_group(_extracted_group_name(qe))
        group.attrs["angle_deg"] = float(angle_deg)
        group.attrs["qe"] = float(qe)
        group.attrs["t_frame_s"] = float(ref_table["t_int_frame"][0].to_value("s"))
        group.attrs["gain_e_per_adu"] = float(
            ref_channel.detector.gain.to_value("electron / adu")
        )
        group.create_dataset("npix", data=np.asarray(ref_table["npix"].value))
        group.create_dataset(
            "center", data=np.asarray(ref_table["center"].to_value("um"))
        )
        group.create_dataset(
            "width", data=np.asarray(ref_table["width"].to_value("um"))
        )
        group.create_dataset(
            "edges", data=np.asarray(ref_table.meta["wavel_bin_edges"].to_value("um"))
        )
        n_readouts = None
        for ch_name, spectra in extracted_spectra_adu.items():
            ch_group = group.create_group(ch_name)
            for component, spectrum in spectra.items():
                values = np.asarray(spectrum.to_value("adu"))  # (n_readouts, n_bins)
                n_readouts = values.shape[0]
                dset = ch_group.create_dataset(component, data=values)
                dset.attrs["unit"] = "adu"
        group.attrs["n_readouts"] = int(n_readouts or 0)


@pipeline_stage(
    depends_on=("combine_astro_and_instrum_signals", "extract_spectra_from_readouts")
)
def record_info_at_angle_and_qe(
    *,
    angle_deg: float,
    qe: float,
    output_channels: dict,
    save_dir: str,
    extracted_spectra_adu: dict | None = None,
    plot: bool = True,
) -> dict:
    """
    Save one HDF5 file for this rotation angle and QE

    File structure::

        angle_{angle}_qe_{qe}.hdf5
          dc_{dc}_qe_{qe}/
            output_1_bright   QTable, all columns from tables_by_dark_current
            output_2_bright
            output_3_dark
            output_4_dark
          __extracted_qe_{qe}/          (only if extracted_spectra_adu is given)
            npix, center, width, edges  bin geometry (pix, um)
            {channel}/{component}       (n_readouts, n_bins) ADU per frame, box-summed from readouts
            attrs: angle_deg, qe, t_frame_s, gain_e_per_adu, n_readouts
    """

    file_name_hdf5 = f"{save_dir}{format_angle_qe_hdf5_name(angle_deg, qe)}"
    angle_deg = canonical_angle_deg(angle_deg)
    qe = canonical_qe(qe)
    hdf5_path = Path(file_name_hdf5)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if hdf5_path.exists():
        hdf5_path.unlink()
    hdf5_paths = []
    first_dataset = True
    dc_rates = list(output_channels["output_3_dark"].tables_by_dark_current)

    for dc_rate in dc_rates:
        dc_rate_key = canonical_dc_rate(dc_rate)
        dc_group = f"dc_{dc_rate_key:06.3f}"
        qe_group = f"qe_{qe:04.2f}"
        dc_qe_str = f"{dc_group}_{qe_group}"

        # write out the tables for each output channel
        for ch_name, ch in output_channels.items():
            out_tbl = ch.tables_by_dark_current[dc_rate_key].copy()
            out_tbl.meta["angle_deg"] = float(angle_deg)
            out_tbl.meta["dark_current_e_pix_s"] = float(dc_rate_key)
            out_tbl.meta["qe"] = float(qe)
            hdf5_path = dc_qe_str + f"/{ch_name}"
            if first_dataset:
                out_tbl.write(
                    file_name_hdf5,
                    path=hdf5_path,
                    serialize_meta=True,
                    overwrite=True,
                )
                first_dataset = False
            else:
                out_tbl.write(
                    file_name_hdf5,
                    path=hdf5_path,
                    serialize_meta=True,
                    append=True,
                )
            hdf5_paths.append(hdf5_path)
            logger.info(f"Wrote {file_name_hdf5}:{hdf5_path}")

    if extracted_spectra_adu:
        _write_extracted_spectra(
            file_name_hdf5,
            angle_deg=angle_deg,
            qe=qe,
            output_channels=output_channels,
            extracted_spectra_adu=extracted_spectra_adu,
        )
        hdf5_paths.append(_extracted_group_name(qe))
        logger.info(
            f"Wrote extracted spectra to {file_name_hdf5}:{_extracted_group_name(qe)}"
        )

    snapshot = {
        "angle_deg": angle_deg,
        "hdf5_file": file_name_hdf5,
        "hdf5_paths": hdf5_paths,
        "dark_currents": dc_rates,
    }

    logger.info(f"Recorded angle {angle_deg} to {file_name_hdf5}")

    return
