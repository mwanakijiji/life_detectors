"""Assemble per-angle extracted spectra into (wavelength, readout, channel) cubes."""

from __future__ import annotations

import glob
import logging
import os
from typing import List

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.hdf5_io import ERROR_TOTAL_KEY, _extracted_group_name
from ...utils.helpers.keys import (
    canonical_dc_rate,
    canonical_qe,
    format_extracted_cube_name,
    hdf5_path_matches_qe,
    parse_angle_from_hdf5_path,
)
from ..instrumental.channels import FITS_CHANNEL_NAMES

logger = logging.getLogger(__name__)

DARK_CURRENT_KEY = "dark_current"
ERROR_HDU_NAME = "ERROR"


@pipeline_stage(depends_on=("record_info_at_angle_and_qe",))
def assemble_extracted_spectra_cubes(
    read_dir: str, qe: float, save_dir: str
) -> List[str]:
    """
    Build one FITS cube file per (QE, DC) from the extracted spectra stored in the
    angle HDF5 files, reading one angle file at a time.

    Readout axis: angle_index * n_readouts_per_angle + readout_index, angles sorted.

    Each file ``extracted_cube_qe_{qe}_dc_{dc}.fits`` holds:
        PRIMARY      total counts, numpy shape (n_bins, n_readouts_total, n_channels),
        ADU
        <component>  one image HDU per component (each source, instrum_systematics,
                     dark_current), same shape
        ERROR        1-sigma on the total, same shape, ADU (Sharp & Birchall extraction
        only)
        WAVELENGTH   table: center, width, lower_edge, upper_edge (um), npix
        READOUTS     table: angle_deg, angle_index, readout_index

    The DC values are the ``dc_rates`` stored in the angle files (one extracted
    slice per DC). The dark_current component is the mean pedestal
    dc * npix * t_frame / gain; the ERROR already includes dark-current noise.

    Returns: the written file paths.
    """
    group_name = _extracted_group_name(qe)
    hdf5_files = [
        f
        for f in glob.glob(os.path.join(read_dir, "angle_*.hdf5"))
        if hdf5_path_matches_qe(f, qe)
    ]
    hdf5_files = sorted(hdf5_files, key=parse_angle_from_hdf5_path)
    if not hdf5_files:
        logger.warning(
            "No angle HDF5 files for QE %s in %s; no cubes written", qe, read_dir
        )
        return []

    # geometry and layout from the first file
    with h5py.File(hdf5_files[0], "r") as handle:
        group = handle[group_name]
        channels = [k for k in group if isinstance(group[k], h5py.Group)]
        # "__" datasets (errors, vark) are not signal components
        components = [k for k in group[channels[0]].keys() if not k.startswith("__")]
        has_error = ERROR_TOTAL_KEY in group[channels[0]]
        dc_rates = np.asarray(group["dc_rates"])
        extraction_method = str(group.attrs.get("extraction_method", "unknown"))
        n_read = int(group.attrs["n_readouts"])
        t_frame_s = float(group.attrs["t_frame_s"])
        gain = float(group.attrs["gain_e_per_adu"])
        npix = np.asarray(group["npix"])
        center = np.asarray(group["center"])
        width = np.asarray(group["width"])
        edges = np.asarray(group["edges"])
    n_bins = len(center)
    n_total = len(hdf5_files) * n_read

    # (n_dc, n_bins, n_readouts_total, n_channels) per component
    cube_shape = (len(dc_rates), n_bins, n_total, len(channels))
    cubes = {comp: np.zeros(cube_shape) for comp in components}
    error = np.zeros(cube_shape) if has_error else None
    angle_deg = np.zeros(n_total)
    angle_index = np.zeros(n_total, dtype=int)
    readout_index = np.zeros(n_total, dtype=int)

    for i_angle, hdf5_file in enumerate(hdf5_files):
        sl = slice(i_angle * n_read, (i_angle + 1) * n_read)
        with h5py.File(hdf5_file, "r") as handle:
            group = handle[group_name]
            if int(group.attrs["n_readouts"]) != n_read:
                raise ValueError(
                    f"{hdf5_file} has {group.attrs['n_readouts']} readouts per angle, "
                    f"expected {n_read}"
                )
            for i_ch, ch_name in enumerate(channels):
                for comp in components:
                    # stored (n_dc, n_readouts, n_bins) -> cube (n_dc, n_bins,
                    # n_readouts)
                    cubes[comp][:, :, sl, i_ch] = np.swapaxes(
                        np.asarray(group[ch_name][comp]), 1, 2
                    )
                if has_error:
                    error[:, :, sl, i_ch] = np.swapaxes(
                        np.asarray(group[ch_name][ERROR_TOTAL_KEY]), 1, 2
                    )
            angle_deg[sl] = float(group.attrs["angle_deg"])
        angle_index[sl] = i_angle
        readout_index[sl] = np.arange(n_read)

    wavelength_hdu = fits.BinTableHDU(
        Table(
            {
                "center": center,
                "width": width,
                "lower_edge": edges[:-1],
                "upper_edge": edges[1:],
                "npix": npix,
            },
            units={
                "center": "um",
                "width": "um",
                "lower_edge": "um",
                "upper_edge": "um",
                "npix": "pix",
            },
        ),
        name="WAVELENGTH",
    )
    readouts_hdu = fits.BinTableHDU(
        Table(
            {
                "angle_deg": angle_deg,
                "angle_index": angle_index,
                "readout_index": readout_index,
            },
            units={"angle_deg": "deg"},
        ),
        name="READOUTS",
    )

    written = []
    for i_dc, dc_rate in enumerate(dc_rates):
        dc_rate = canonical_dc_rate(dc_rate)
        dark_adu = dc_rate * npix * t_frame_s / gain  # (n_bins,)
        components_dc = {comp: cube[i_dc] for comp, cube in cubes.items()}
        components_dc[DARK_CURRENT_KEY] = np.broadcast_to(
            dark_adu[:, None, None], (n_bins, n_total, len(channels))
        )
        total = sum(components_dc.values())

        primary = fits.PrimaryHDU(data=total)
        header = primary.header
        header["BUNIT"] = "adu"
        header["QE"] = (canonical_qe(qe), "Quantum efficiency")
        header["DC"] = (dc_rate, "Dark current [e-/pix/s]")
        header["TFRAME"] = (t_frame_s, "Integration time of one frame [s]")
        header["GAIN"] = (gain, "Detector gain [e-/ADU]")
        header["EXTRACT"] = (extraction_method, "Spectral extraction method")
        header["NREADANG"] = (n_read, "Readouts per angle")
        header["NANGLES"] = (len(hdf5_files), "Number of angles")
        header["AXISWAVE"] = (3, "FITS axis: wavelength bin (numpy axis 0)")
        header["AXISREAD"] = (2, "FITS axis: readout (numpy axis 1)")
        header["AXISCHAN"] = (1, "FITS axis: channel (numpy axis 2)")
        for i_ch, ch_name in enumerate(channels):
            header[f"CHAN{i_ch}"] = FITS_CHANNEL_NAMES[ch_name]
        for i_comp, comp in enumerate(components_dc):
            header[f"COMP{i_comp}"] = comp

        hdus = [primary]
        for comp, data in components_dc.items():
            hdu = fits.ImageHDU(data=np.ascontiguousarray(data), name=comp.upper())
            hdu.header["BUNIT"] = "adu"
            hdu.header["COMPNAME"] = comp
            hdus.append(hdu)
        if has_error:
            hdu = fits.ImageHDU(
                data=np.ascontiguousarray(error[i_dc]), name=ERROR_HDU_NAME
            )
            hdu.header["BUNIT"] = "adu"
            hdu.header["COMMENT"] = "1-sigma uncertainty on the total (PRIMARY)"
            hdus.append(hdu)
        hdus += [wavelength_hdu.copy(), readouts_hdu.copy()]

        file_name = os.path.join(save_dir, format_extracted_cube_name(qe, dc_rate))
        fits.HDUList(hdus).writeto(file_name, overwrite=True)
        written.append(file_name)
        logger.info(
            "Wrote extracted-spectra cube %s, shape (wavelength, readout, channel) = "
            "%s",
            file_name,
            total.shape,
        )
    return written


def read_extracted_cube(file_name: str) -> dict:
    """
    Read an ``extracted_cube_qe_*_dc_*.fits`` file back into arrays.

    Returns a dict with:
        total_adu            (n_bins, n_readouts_total, n_channels) ADU
        components_adu       {component name: same shape, ADU}
        error_adu            same shape, or None (no ERROR HDU)
        channels             FITS channel names in axis-2 order (e.g. "CH1BR")
        wavel_lo, wavel_cen, wavel_hi   (n_bins,) um
        angle_deg, angle_index, readout_index   (n_readouts_total,)
        gain (e-/ADU), t_frame_s, qe, dc (e-/pix/s), n_angles, n_readouts_per_angle,
        extraction_method, file_name
    """
    with fits.open(file_name) as hdul:
        header = hdul[0].header
        n_comp = sum(1 for key in header if key.startswith("COMP"))
        component_names = [header[f"COMP{i}"] for i in range(n_comp)]
        n_chan = sum(1 for key in header if key.startswith("CHAN"))
        channels = [header[f"CHAN{i}"] for i in range(n_chan)]
        wavelength = hdul["WAVELENGTH"].data
        readouts = hdul["READOUTS"].data
        return {
            "total_adu": np.asarray(hdul[0].data, dtype=float),
            "components_adu": {
                comp: np.asarray(hdul[comp.upper()].data, dtype=float)
                for comp in component_names
            },
            "error_adu": (
                np.asarray(hdul[ERROR_HDU_NAME].data, dtype=float)
                if ERROR_HDU_NAME in hdul
                else None
            ),
            "channels": channels,
            "wavel_lo": np.asarray(wavelength["lower_edge"], dtype=float),
            "wavel_cen": np.asarray(wavelength["center"], dtype=float),
            "wavel_hi": np.asarray(wavelength["upper_edge"], dtype=float),
            "angle_deg": np.asarray(readouts["angle_deg"], dtype=float),
            "angle_index": np.asarray(readouts["angle_index"], dtype=int),
            "readout_index": np.asarray(readouts["readout_index"], dtype=int),
            "gain": float(header["GAIN"]),
            "t_frame_s": float(header["TFRAME"]),
            "qe": float(header["QE"]),
            "dc": float(header["DC"]),
            "n_angles": int(header["NANGLES"]),
            "n_readouts_per_angle": int(header["NREADANG"]),
            "extraction_method": str(header.get("EXTRACT", "unknown")),
            "file_name": str(file_name),
        }
