"""Aperture / dispersed-signal plots."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from ..utils.helpers.formatting import format_astro_source_label, format_plot_title

logger = logging.getLogger(__name__)


def plot_astro_ph_sec_pixel(
    channel_name: str,
    bin_edges,
    bin_centers,
    signals_by_source: Mapping[str, object],
    cumulative_signal,
    y_unit,
    config,
    out_path: str,
) -> None:
    """Stairs plot of photon rate per pixel for one output channel."""
    edges = bin_edges.value if hasattr(bin_edges, "value") else bin_edges
    fig, ax = plt.subplots(figsize=(10, 5))
    for source_name, y_vals in signals_by_source.items():
        ax.stairs(
            np.asarray(y_vals, dtype=float),
            edges=edges,
            linewidth=2,
            label=format_astro_source_label(source_name),
        )
    ax.stairs(
        cumulative_signal,
        edges=edges,
        linewidth=3,
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    ax.set_xlim(4.0, 18.5)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(f"Wavelength ({bin_centers.unit})", fontsize=22)
    ax.set_ylabel(f"Photon rate incident on pixels\n({y_unit})", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        format_plot_title(
            f"Astrophysical photon rate per pixel — {channel_name}",
            config,
            font_size=2,
        ),
        loc="left",
        pad=70,
    )
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        frameon=True,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    logging.info(
        "Saved astrophysical photon rate per pixel plot for %s: %s",
        channel_name,
        out_path,
    )


def plot_integrated_flux_by_output(
    flux_by_source: Mapping[str, tuple],
    output_name: str,
    title: str,
    config,
    out_path: str,
) -> None:
    """Overplot spatially integrated flux vs wavelength for one output."""
    fig, ax = plt.subplots(figsize=(10, 12))
    flux_unit = None
    wavel_unit = None
    for source_name, (wavel, flux_integrated) in flux_by_source.items():
        ax.plot(wavel, flux_integrated, label=format_astro_source_label(source_name))
        if flux_unit is None:
            flux_unit = flux_integrated.unit
            wavel_unit = wavel.unit
    ax.set_yscale("log")
    ax.set_xlim([4, 18])
    ax.set_ylim([1e-3, 1e10])
    ax.set_xlabel(f"Wavelength ({wavel_unit})", fontsize=22)
    ax.set_ylabel(f"Flux ({flux_unit})", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        format_plot_title(f"{title} — {output_name}", config),
        loc="left",
        fontsize=2,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved plot to %s", out_path)


def plot_poster_post_aperture_flux(
    flux_by_source: Mapping[str, tuple],
    output_name: str,
    config,
    out_path: str,
) -> None:
    """Poster-style post-aperture flux panel for one output channel."""
    fig, ax = plt.subplots(figsize=(10, 5))
    flux_unit = None
    wavel_unit = None
    for source_name, (wavel, flux_integrated) in flux_by_source.items():
        ax.plot(wavel, flux_integrated, label=format_astro_source_label(source_name))
        if flux_unit is None:
            flux_unit = flux_integrated.unit
            wavel_unit = wavel.unit

    ax.set_xlim(4.0, 18.5)
    ax.set_yscale("log")
    if "dark" in output_name:
        ax.set_ylim([1e-2, 1e5])
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(f"Wavelength ({wavel_unit})", fontsize=22)
    ax.set_ylabel(f"Flux\n({flux_unit})", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        format_plot_title(
            f"Post-aperture flux (all sources) — {output_name}",
            config,
            font_size=2,
        ),
        loc="left",
        pad=70,
    )
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        frameon=True,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    logging.info("Saved poster post-aperture flux plot to %s", out_path)
