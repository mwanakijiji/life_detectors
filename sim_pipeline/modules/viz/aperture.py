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


def plot_beam_to_detector(
    spectra_by_output: Mapping[str, Mapping[str, tuple]],
    background_names: Sequence[str],
    config,
    out_path: str,
    angle_deg: float | None = None,
) -> None:
    """
    One panel per output: the sky-integrated photon rate (ph/s/um) of every
    component entering the detector stage, i.e. each astrophysical source after the
    aperture (solid) and each instrument background (dashed), plus their total
    (black). Each panel also states the band-integrated fraction of the total that
    comes from the instrument backgrounds. Components that are zero everywhere
    cannot be drawn on the log axis; they are listed in the legend as "(all zero)".

    spectra_by_output: {output_name: {component_name: (wavel, flux_ph_sec_um)}}
    """
    background_names = set(background_names)
    output_names = list(spectra_by_output)
    n_cols = 2
    n_rows = int(np.ceil(len(output_names) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 5 * n_rows), squeeze=False, sharex=True
    )

    flux_unit = wavel_unit = None
    for ax, output_name in zip(axes.flat, output_names):
        total = None
        background_total = None
        wavel_vals = None
        for comp_name, (wavel, flux) in spectra_by_output[output_name].items():
            wavel_vals = np.asarray(getattr(wavel, "value", wavel), dtype=float)
            flux_vals = np.asarray(getattr(flux, "value", flux), dtype=float)
            if flux_unit is None:
                flux_unit = getattr(flux, "unit", "")
                wavel_unit = getattr(wavel, "unit", "")
            total = flux_vals.copy() if total is None else total + flux_vals
            is_background = comp_name in background_names
            if is_background:
                background_total = (
                    flux_vals.copy()
                    if background_total is None
                    else background_total + flux_vals
                )
            label = (
                f"{comp_name} (instrument)"
                if is_background
                else format_astro_source_label(comp_name)
            )
            if not np.any(flux_vals > 0):
                # keep it in the legend so a missing curve is not mistaken for a bug
                ax.plot(
                    [],
                    [],
                    linestyle="--" if is_background else "-",
                    label=f"{label} (all zero)",
                )
                continue
            ax.plot(
                wavel_vals,
                np.where(flux_vals > 0, flux_vals, np.nan),
                linestyle="--" if is_background else "-",
                linewidth=2,
                label=label,
            )
        if total is not None and np.any(total > 0):
            ax.plot(
                wavel_vals,
                np.where(total > 0, total, np.nan),
                color="black",
                linewidth=3,
                alpha=0.5,
                label="Total",
            )
            band_total = np.trapz(total, wavel_vals)
            band_bg = (
                np.trapz(background_total, wavel_vals)
                if background_total is not None
                else 0.0
            )
            frac_text = (
                f"instrument backgrounds: {100.0 * band_bg / band_total:.3g}% of total"
                if band_total > 0
                else "total is zero"
            )
            if not background_names:
                frac_text = "no instrument backgrounds enabled"
            ax.text(
                0.02,
                0.03,
                frac_text,
                transform=ax.transAxes,
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
            ax.set_yscale("log")
        ax.set_title(output_name, fontsize=14)
        ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_ylabel(f"Photon rate\n({flux_unit})", fontsize=12)

    for ax in axes[-1, :]:
        ax.set_xlabel(f"Wavelength ({wavel_unit})", fontsize=12)
    for ax in list(axes.flat)[len(output_names) :]:
        ax.set_visible(False)

    angle_str = f" (angle {angle_deg:.2f} deg)" if angle_deg is not None else ""
    fig.suptitle(
        format_plot_title(
            "Beam entering the detector stage: post-aperture sources + instrument "
            f"backgrounds, summed over the sky{angle_str}",
            config,
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved beam-to-detector plot to %s", out_path)
