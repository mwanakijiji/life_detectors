"""Plots of spectra extracted from the detector readouts."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..utils.helpers.formatting import format_astro_source_label, format_plot_title
from ..utils.types import ConfigLike

logger = logging.getLogger(__name__)


def _wavel_pixel_transforms(
    entry: dict,
) -> Tuple[Callable, Callable] | None:
    """
    (wavelength um -> detector x pix, detector x pix -> wavelength um) from the bins'
    wavelength and x edges, interpolating linearly in log(lambda) (as the wavelength
    solution does); None if the entry has no x positions.
    """
    x_lo, x_hi = entry.get("x_lo_pix"), entry.get("x_hi_pix")
    if x_lo is None or x_hi is None:
        return None
    x_edges = np.append(x_lo.to_value("pix"), x_hi.to_value("pix")[-1])
    wavel_edges_um = np.append(
        entry["wavel_lo"].to_value("um"), entry["wavel_hi"].to_value("um")[-1]
    )
    log_wavel_edges = np.log(wavel_edges_um)

    def wavel_to_pix(wavel_um):
        # matplotlib may probe non-positive limits; keep the log finite
        log_w = np.log(np.clip(np.asarray(wavel_um, dtype=float), 1e-12, None))
        return np.interp(log_w, log_wavel_edges, x_edges)

    def pix_to_wavel(x_pix):
        x_pix = np.asarray(x_pix, dtype=float)
        return np.exp(np.interp(x_pix, x_edges, log_wavel_edges))

    return wavel_to_pix, pix_to_wavel


def plot_extracted_spectra(
    spectra_with_wavel: Dict[str, dict],
    config: ConfigLike,
    out_path: str,
    dc_rates: Sequence[float] | np.ndarray | None = None,
    angle_deg: float | None = None,
    extraction_method: str | None = None,
    i_dc: int = 0,
    i_readout: int = 0,
) -> None:
    """
    One panel per output channel: the extracted spectrum (ADU per wavelength bin, one
    readout) against wavelength from the wavelength solution, per component (thin
    lines) and summed over components (black), with the 1-sigma band of the total if
    the extraction gave errors. If the wavelength solution has the bins' detector x
    positions, a secondary (top) axis gives the pixel column along the spectrum.

    Args:
        spectra_with_wavel: output of wavel_solution.apply_wavel_solution
        config: for the plot title
        out_path: .png path
        dc_rates: dark-current values (e-/pix/s) of the first spectrum axis, for the
            label
        angle_deg: rotation angle, for the title
        extraction_method: for the title
        i_dc, i_readout: which dark-current value and readout to show
    """
    channel_names = list(spectra_with_wavel)
    n_cols = 2
    n_rows = max(int(np.ceil(len(channel_names) / n_cols)), 1)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(18, 5.5 * n_rows), squeeze=False, sharex=True
    )

    dc_values = None if dc_rates is None else np.atleast_1d(dc_rates)
    if dc_values is not None and dc_values.size > i_dc:
        dc_label = f"DC = {float(dc_values[i_dc]):g} e-/pix/s"
    else:
        dc_label = f"DC index {i_dc}"

    for ax, channel_name in zip(axes.ravel(), channel_names):
        entry = spectra_with_wavel[channel_name]
        wavel_um = entry["wavel"].to_value("um")
        total = None
        for component, spectrum in entry["spectra_adu"].items():
            values = np.asarray(spectrum[i_dc, i_readout].to_value("adu"), dtype=float)
            total = values.copy() if total is None else total + values
            label = format_astro_source_label(component)
            if np.all(values == 0):
                label += " (all zero)"
            ax.plot(wavel_um, values, drawstyle="steps-mid", lw=1.0, label=label)
        if total is not None:
            ax.plot(
                wavel_um, total, drawstyle="steps-mid", color="k", lw=2.0, label="Total"
            )
            error = entry.get("error_adu")
            if error is not None:
                sigma = np.asarray(error[i_dc, i_readout].to_value("adu"), dtype=float)
                ax.fill_between(
                    wavel_um,
                    total - sigma,
                    total + sigma,
                    step="mid",
                    color="k",
                    alpha=0.2,
                    label="Total ± 1σ",
                )
            positive = total[total > 0]
            if positive.size:
                # symlog keeps zero-valued components visible next to bright ones
                ax.set_yscale("symlog", linthresh=max(positive.min(), 1e-12))

        # secondary x axis: detector pixel column along the spectrum
        transforms = _wavel_pixel_transforms(entry)
        if transforms is not None:
            secax = ax.secondary_xaxis("top", functions=transforms)
            secax.set_xlabel("Detector x (pixel; column k spans k to k+1)", fontsize=10)
            secax.minorticks_on()
            ax.set_title(channel_name, fontsize=14, pad=28)
        else:
            ax.set_title(channel_name, fontsize=14)
        ax.set_ylabel("ADU per wavelength bin (one readout)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength (µm), from the wavelength solution")
    for ax in axes.ravel()[len(channel_names) :]:
        ax.set_visible(False)

    base_title = "Extracted spectra"
    if extraction_method:
        base_title += f" ({extraction_method})"
    if angle_deg is not None:
        base_title += f", angle {angle_deg:.2f} deg"
    base_title += f", readout {i_readout}, {dc_label}"
    fig.suptitle(
        format_plot_title(base_title, config),
        fontsize=11,
        family="monospace",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot of extracted spectra vs. wavelength to %s", out_path)
