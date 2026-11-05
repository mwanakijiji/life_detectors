"""2D plots (and FITS equivalents) of extracted counts vs. wavelength and time (frame /
angle), per channel and for the chopped (dark output 3 - dark output 4) signal."""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.ticker import NullFormatter

from ..utils.helpers.formatting import format_astro_source_label, format_plot_title
from ..utils.types import ConfigLike

logger = logging.getLogger(__name__)

X_AXIS_CHOICES = ("frame", "time", "angle")
CHOP_PLUS, CHOP_MINUS = "CH3DK", "CH4DK"
# below this |chop| / |CH3DK|, the chopped panel is flagged as cancellation residue
CHOP_RESIDUE_RATIO = 1e-9


def _x_edges_and_label(cube: dict, x_axis: str) -> tuple[np.ndarray, str]:
    """Edges (n_readouts_total + 1,) along the readout axis, and the axis label."""
    n_total = cube["angle_deg"].size
    if x_axis == "frame":
        return np.arange(n_total + 1) - 0.5, "Frame number"
    if x_axis == "time":
        # frames back to back (no dead time between readouts or angles)
        return (
            np.arange(n_total + 1) * cube["t_frame_s"],
            "Time (s; frames back to back)",
        )
    if x_axis == "angle":
        # each angle's readouts spread evenly over its share of the full rotation
        d_angle = 360.0 / max(cube["n_angles"], 1)
        per_read = d_angle / max(cube["n_readouts_per_angle"], 1)
        starts = cube["angle_deg"] + cube["readout_index"] * per_read
        return np.append(starts, starts[-1] + per_read), "Rotation angle (deg)"
    raise ValueError(f"x_axis must be one of {X_AXIS_CHOICES}; got {x_axis!r}")


def _color_limits(data: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if symmetric:
        half = max(abs(lo), abs(hi))
        lo, hi = -half, half
    if hi <= lo:  # constant image (e.g. all zero)
        lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)


def _chop_ratio(chopped: np.ndarray, reference: np.ndarray) -> float:
    """max |chop| / mean |reference channel| (inf if the reference is all zero)."""
    ref = float(np.nanmean(np.abs(reference)))
    if ref == 0.0:
        return float("inf") if np.nanmax(np.abs(chopped)) > 0 else 0.0
    return float(np.nanmax(np.abs(chopped)) / ref)


def _panels(
    cube: dict, component: str | None
) -> tuple[str, list[tuple[str, np.ndarray, bool]], str | None]:
    """
    The images shown in the counts-vs-time plot (and written to its FITS equivalent).

    Returns (component label, [(panel title, image (n_bins, n_readouts_total) e-,
    symmetric color range?), ...], chop note or None): one panel per channel, plus the
    chopped CH3DK - CH4DK panel when both channels exist.
    """
    if component is None:
        data_adu = cube["total_adu"]
        comp_label = "Total (all components)"
    else:
        data_adu = cube["components_adu"][component]
        comp_label = format_astro_source_label(component)
    data_e = data_adu * cube["gain"]  # (n_bins, n_readouts_total, n_channels)

    channels = list(cube["channels"])
    panels = [(name, data_e[:, :, i], False) for i, name in enumerate(channels)]
    chop_note = None
    if CHOP_PLUS in channels and CHOP_MINUS in channels:
        plus = data_e[:, :, channels.index(CHOP_PLUS)]
        chopped = plus - data_e[:, :, channels.index(CHOP_MINUS)]
        panels.append((f"Chopped ({CHOP_PLUS} - {CHOP_MINUS})", chopped, True))
        ratio = _chop_ratio(chopped, plus)
        chop_note = f"max|chop| / mean|{CHOP_PLUS}| = {ratio:.2e}"
        if ratio < CHOP_RESIDUE_RATIO:
            chop_note += "\n(cancellation residue only: no real chopped signal)"

    return comp_label, panels, chop_note


def plot_counts_vs_time(
    cube: dict,
    out_path: str,
    component: str | None = None,
    x_axis: str = "frame",
    config: ConfigLike | None = None,
) -> None:
    """
    One panel per detector channel plus one for the chopped signal (CH3DK - CH4DK):
    color = extracted counts (e-, ADU x gain), y = wavelength (um, log scale, short
    wavelengths at the top), x = frame number, time or rotation angle.

    Args:
        cube: output of extracted_cube.read_extracted_cube
        out_path: .png path
        component: a component name from the cube (e.g. a source name), or None for
            the total (all components incl. dark current)
        x_axis: "frame", "time" or "angle"
        config: for the plot-title context (optional)

    Channels use a 1-99 percentile color range; the chopped panel a range symmetric
    about zero, and a text box with max|chop| / mean|CH3DK| (flagged as cancellation
    residue when below CHOP_RESIDUE_RATIO). Darker = more counts (gray_r).
    """
    comp_label, panels, chop_note = _panels(cube, component)

    x_edges, x_label = _x_edges_and_label(cube, x_axis)
    y_edges = np.append(cube["wavel_lo"], cube["wavel_hi"][-1])
    y_ticks = np.geomspace(y_edges[0], y_edges[-1], 5)

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(14, 2.8 * len(panels)), sharex=True, squeeze=False
    )
    for ax, (title, image, symmetric) in zip(axes[:, 0], panels):
        vmin, vmax = _color_limits(image, symmetric)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            image,
            cmap="gray_r",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label(r"Counts (e$^-$)")
        cbar.formatter.set_powerlimits((-2, 3))
        cbar.update_ticks()

        # geometric bins look even on a log axis; short wavelengths at the top
        ax.set_yscale("log")
        ax.set_ylim(y_edges[-1], y_edges[0])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in y_ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(r"Wavelength ($\mu$m)")
        ax.set_title(title, fontsize=12, loc="left")
        if symmetric and chop_note is not None:
            ax.text(
                0.99,
                0.04,
                chop_note,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.5"},
            )
    axes[-1, 0].set_xlabel(x_label)

    base_title = (
        f"Extracted counts vs. wavelength and {x_axis}: {comp_label}; "
        f"QE = {cube['qe']:.2f}, DC = {cube['dc']:g} e-/pix/s, "
        f"{cube['extraction_method']}"
    )
    fig.suptitle(
        format_plot_title(base_title, config) if config is not None else base_title,
        fontsize=11,
        family="monospace",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved counts-vs-%s plot (%s) to %s", x_axis, comp_label, out_path)


def counts_vs_time_file_name(
    cube_file: str, component: str | None, x_axis: str, ext: str = ".png"
) -> str:
    """extracted_cube_qe_<qe>_dc_<dc>.fits ->
    counts_vs_<x_axis>_qe_<qe>_dc_<dc>_<component or total><ext> (same directory)."""
    base = os.path.basename(cube_file)
    stem = base[: -len(".fits")] if base.endswith(".fits") else base
    stem = stem.replace("extracted_cube", f"counts_vs_{x_axis}", 1)
    return os.path.join(
        os.path.dirname(cube_file), f"{stem}_{component or 'total'}{ext}"
    )


def write_counts_vs_time_fits(
    cube: dict,
    out_path: str,
    component: str | None = None,
    x_axis: str = "frame",
) -> None:
    """
    FITS equivalent of plot_counts_vs_time: the same panels, in e- (ADU x gain).

    HDUs:
        PRIMARY      no data; header with QE, DC, GAIN, TFRAME, EXTRACT, COMPONT,
                     XAXIS, SOURCE (the cube file) and PANEL<i> = panel EXTNAMEs
        <channel>    one image per channel (e.g. CH1BR), shape (n_bins,
                     n_readouts_total) = (NAXIS2 wavelength, NAXIS1 readout), e-
        CHOPPED      CH3DK - CH4DK (if both exist), with CHOPRAT = max|chop| /
                     mean|CH3DK| in its header
        WAVELENGTH   table: lower_edge, center, upper_edge (um), one row per bin
        XEDGES       table: edge (n_readouts_total + 1) along the x axis (frame
                     number, time in s or rotation angle in deg, per XAXIS)
    """
    comp_label, panels, _ = _panels(cube, component)
    x_edges, x_label = _x_edges_and_label(cube, x_axis)
    channels = list(cube["channels"])

    primary = fits.PrimaryHDU()
    header = primary.header
    header["QE"] = (cube["qe"], "quantum efficiency")
    header["DC"] = (cube["dc"], "dark current (e-/pix/s)")
    header["GAIN"] = (cube["gain"], "e-/ADU (applied: data are in e-)")
    header["TFRAME"] = (cube["t_frame_s"], "frame integration time (s)")
    header["NANGLES"] = cube["n_angles"]
    header["NREADANG"] = (cube["n_readouts_per_angle"], "readouts per angle")
    header["EXTRACT"] = cube["extraction_method"]
    header["COMPONT"] = (component or "total", "component (total = all incl. DC)")
    header["XAXIS"] = (x_axis, x_label)
    header["BUNIT"] = "electron"
    header["SOURCE"] = os.path.basename(cube["file_name"])

    hdus: list = [primary]
    for i_panel, (title, image, _symmetric) in enumerate(panels):
        is_chop = i_panel >= len(channels)
        hdu = fits.ImageHDU(
            data=np.asarray(image, dtype=np.float64),
            name="CHOPPED" if is_chop else channels[i_panel],
        )
        hdu.header["BUNIT"] = "electron"
        hdu.header["PANEL"] = title
        if is_chop:
            plus = panels[channels.index(CHOP_PLUS)][1]
            hdu.header["CHOPRAT"] = (
                _chop_ratio(image, plus),
                f"max|chop| / mean|{CHOP_PLUS}|",
            )
        header[f"PANEL{i_panel}"] = hdu.name
        hdus.append(hdu)

    hdus.append(
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="lower_edge", format="D", unit="um",
                            array=cube["wavel_lo"]),
                fits.Column(name="center", format="D", unit="um",
                            array=cube["wavel_cen"]),
                fits.Column(name="upper_edge", format="D", unit="um",
                            array=cube["wavel_hi"]),
            ],
            name="WAVELENGTH",
        )
    )
    x_unit = {"frame": "", "time": "s", "angle": "deg"}[x_axis]
    hdus.append(
        fits.BinTableHDU.from_columns(
            [fits.Column(name="edge", format="D", unit=x_unit, array=x_edges)],
            name="XEDGES",
        )
    )
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    logger.info(
        "Wrote counts-vs-%s FITS (%s) to %s", x_axis, comp_label, out_path
    )


def counts_vs_time_products_for_cubes(
    cube_files: Sequence[str],
    x_axis: str = "frame",
    config: ConfigLike | None = None,
    write_fits: bool = True,
    plot: bool = True,
) -> List[str]:
    """
    For each extracted-spectra cube: the total; and, for the first cube only (the
    components other than dark_current do not depend on DC), every component. Each
    as a .fits (if write_fits) and/or a .png (if plot), side by side.

    Returns the written paths.
    """
    # local import keeps viz free of a core import at module load
    # pylint: disable-next=import-outside-toplevel
    from ..core.calculator.extracted_cube import read_extracted_cube

    written = []
    for i_file, cube_file in enumerate(cube_files):
        cube = read_extracted_cube(cube_file)
        components: List[str | None] = [None]
        if i_file == 0:
            components += list(cube["components_adu"])
        for component in components:
            if write_fits:
                out_path = counts_vs_time_file_name(
                    cube_file, component, x_axis, ext=".fits"
                )
                write_counts_vs_time_fits(
                    cube, out_path, component=component, x_axis=x_axis
                )
                written.append(out_path)
            if plot:
                out_path = counts_vs_time_file_name(cube_file, component, x_axis)
                plot_counts_vs_time(
                    cube, out_path, component=component, x_axis=x_axis, config=config
                )
                written.append(out_path)
    return written


def plot_counts_vs_time_for_cubes(
    cube_files: Sequence[str],
    x_axis: str = "frame",
    config: ConfigLike | None = None,
) -> List[str]:
    """The .png plots only (see counts_vs_time_products_for_cubes); returns their
    paths."""
    return counts_vs_time_products_for_cubes(
        cube_files, x_axis=x_axis, config=config, write_fits=False, plot=True
    )
