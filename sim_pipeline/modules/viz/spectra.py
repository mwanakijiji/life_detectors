"""Plots of generated source spectra and the planet population."""

from __future__ import annotations

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib.colors import LogNorm

logger = logging.getLogger(__name__)


def plot_planet_population_sample(  # pragma: no cover
    df_planet_population: pd.DataFrame,
    cols_to_plot: list[str],
    output_path: str,
    max_sample_size: int = 10000,
) -> None:
    """
    Create and save a scatter matrix plot of the planet population.

    If the population exceeds max_sample_size, a random sample is used for plotting.
    """
    if len(df_planet_population) > max_sample_size:
        df_sample = copy.deepcopy(
            df_planet_population[cols_to_plot].sample(n=max_sample_size, replace=False)
        )
    else:
        df_sample = copy.deepcopy(df_planet_population[cols_to_plot])
    title_string = (
        f"Sample of {len(df_sample)} planets "
        f"from population of {len(df_planet_population)}"
    )
    axes = pd.plotting.scatter_matrix(df_sample, figsize=(10, 8))
    fig = axes[0, 0].figure
    fig.suptitle(title_string)
    fig.savefig(output_path)
    plt.close(fig)


def plot_luminosity_and_flux(  # pragma: no cover
    wavelength_um,
    luminosity,
    flux,
    title: str,
    out_path: str,
    *,
    log_x: bool = True,
    xlim=None,
) -> None:
    """
    Twin-axis spectrum plot: luminosity (left, blue) and flux (right, red) vs
    wavelength, both on a log y scale.
    """
    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel(rf"$\lambda$ ({wavelength_um.unit})")
    ax1.set_ylabel(rf"$L_{{\lambda}}$ ({luminosity.unit})", color=color1)
    ax1.plot(wavelength_um, luminosity, color=color1)
    if log_x:
        ax1.set_xscale("log")
    if xlim is not None:
        ax1.set_xlim(*xlim)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel(rf"$F(\lambda)$ ({flux.unit})", color=color2)
    ax2.plot(wavelength_um, flux, color=color2)
    if log_x:
        ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote spectrum plot %s", out_path)


def plot_zodiacal_emission_2d(  # pragma: no cover
    wavelength_um,
    wavelengths_to_plot_2d,
    maps_mjy_sr: dict,
    lambda_rel_lon_grid: np.ndarray,
    beta_lat_grid: np.ndarray,
    lambda_rel_lon_los: float,
    beta_lat_los: float,
    out_path: str,
) -> None:
    """
    Three maps of zodiacal emission vs ecliptic latitude and longitude relative to
    the Sun, one per wavelength, with the line of sight marked.

    maps_mjy_sr is keyed by str(wavelength) for each of wavelengths_to_plot_2d.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 15))
    for i, wl in enumerate(wavelengths_to_plot_2d):
        # Find the index in wavelength_um closest to wl
        idx = np.abs(wavelength_um / u.um - wl / u.um).argmin()
        # colorscale like in Kendall+ 2005
        im = axes[i].imshow(
            maps_mjy_sr[str(wl)].value,
            origin="lower",
            extent=[
                np.min(lambda_rel_lon_grid),
                np.max(lambda_rel_lon_grid),
                np.min(beta_lat_grid),
                np.max(beta_lat_grid),
            ],
            aspect="auto",
            norm=LogNorm(),
            cmap="rainbow",
        )

        # dot to indicate line of sight
        axes[i].plot(
            lambda_rel_lon_los,
            beta_lat_los,
            marker="o",
            markersize=8,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="None",
            zorder=5,
        )

        axes[i].set_title(f"Zodiacal background\nat {wavelength_um[idx]/u.um:.1f} μm")
        axes[i].set_xlabel(r"Relative Longitude to Sun, $\lambda_{\rm rel}$ (deg)")
        axes[i].set_ylabel(r"Latitude $\beta$ (deg)")

        # Set reasonable tick intervals
        axes[i].set_xticks([90, 135, 180, 225, 270])
        axes[i].set_yticks([-90, -45, 0, 45, 90])

        cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label(str(maps_mjy_sr[str(wl)].unit))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote zodiacal emission 2D plot %s", out_path)
