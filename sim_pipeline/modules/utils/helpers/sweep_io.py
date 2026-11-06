"""Read/write the labeled sweep Dataset (NetCDF4/HDF5 via xarray + h5netcdf).

Schema (one file per planet; one ``sweep_partial_qe_*.nc`` per QE before combining):

    dims:   qe, dark_current, readout, channel, wavelength, component
    coords: qe, dark_current [e-/pix/s], readout (epoch index),
            angle_deg / angle_index / readout_index (readout),
            channel (+ channel_fits_name), component,
            wavelength [um] (+ wavel_width, wavel_lower_edge, wavel_upper_edge, npix)
    data:   counts        (qe, dark_current, readout, channel, wavelength, component) adu
            dark_pedestal (dark_current, wavelength)                                 adu
            counts_err    (qe, dark_current, readout, channel, wavelength)           adu
                          [Sharp & Birchall extraction only]
    attrs:  t_frame_s, gain_e_per_adu, extraction_method, n_readouts_per_angle, n_angles

One (QE, DC) permutation's cube is a slice: ``ds.sel(qe=..., dark_current=...)``.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Sequence

import xarray as xr

logger = logging.getLogger(__name__)

ENGINE = "h5netcdf"
COMPLEVEL = 4


def _encoding(ds: xr.Dataset) -> dict:
    """zlib-compress the data variables, chunked so one (QE, DC) slice is one chunk."""
    encoding = {}
    for name, var in ds.data_vars.items():
        chunks = tuple(
            1 if dim in ("qe", "dark_current") else size
            for dim, size in zip(var.dims, var.shape)
        )
        encoding[name] = {"zlib": True, "complevel": COMPLEVEL, "chunksizes": chunks}
    return encoding


def write_dataset(ds: xr.Dataset, path: str) -> str:
    """Write ``ds`` to ``path`` (NetCDF4/HDF5), overwriting; returns ``path``."""
    if os.path.exists(path):
        os.remove(path)
    ds.to_netcdf(path, engine=ENGINE, encoding=_encoding(ds))
    return path


def open_sweep(path: str) -> xr.Dataset:
    """Load a sweep Dataset (partial or combined) fully into memory."""
    return xr.load_dataset(path, engine=ENGINE)


def combine_sweep(
    partial_paths: Iterable[str], out_path: str, attrs: Optional[dict] = None
) -> str:
    """
    Concatenate per-QE partial Datasets along ``qe`` and write the combined file.

    Everything without a ``qe`` dimension (wavelength grid, readout table, dark
    pedestal, attrs) must agree across the partials, and the readout axes must match
    exactly; otherwise this raises instead of padding with NaNs.
    """
    parts = [open_sweep(p) for p in partial_paths]
    if not parts:
        raise ValueError("combine_sweep: no partial Datasets given")
    combined = xr.concat(
        parts,
        dim="qe",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="no_conflicts",
    ).sortby("qe")
    if attrs:
        combined.attrs.update(attrs)
    write_dataset(combined, out_path)
    logger.info(
        "Wrote combined sweep Dataset %s (qe=%s, dims=%s)",
        out_path,
        combined["qe"].values.tolist(),
        dict(combined.sizes),
    )
    return out_path


def photon_rate(
    ds: xr.Dataset, components: Optional[Sequence[str]] = None
) -> xr.DataArray:
    """
    Detected counts converted to incident photon rate [ph/s/bin]:
    ``counts * gain / t_frame / qe``, summed over ``components`` (default: all; the
    dark pedestal is never included).

    Dims: (qe, dark_current, readout, channel, wavelength).
    """
    counts = ds["counts"]
    if components is not None:
        counts = counts.sel(component=list(components))
    rate = (
        counts.sum("component")
        * float(ds.attrs["gain_e_per_adu"])
        / float(ds.attrs["t_frame_s"])
        / ds["qe"]
    )
    rate.name = "photon_rate"
    rate.attrs = {"units": "ph/s/bin", "long_name": "incident photon rate"}
    return rate
