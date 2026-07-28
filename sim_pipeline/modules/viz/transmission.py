"""Transmission-screen plots."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..utils.helpers.formatting import format_astro_source_label, format_plot_title

logger = logging.getLogger(__name__)


def plot_flux_through_screens(
    wavel,
    flux_by_screen: Mapping[str, object],
    pre_screen_flux,
    source_name: str,
    out_path: str,
) -> None:
    """Plot post-screen integrated fluxes vs wavelength, plus pre-screen."""
    plt.clf()
    plt.figure(figsize=(12, 4))
    for screen_name, flux in flux_by_screen.items():
        plt.plot(wavel, flux, label=screen_name)
    plt.plot(wavel, pre_screen_flux, label="pre-screen")
    plt.xlim([4.0, 18.0])
    plt.legend()
    plt.title(f"Flux of {source_name} passed through transmission screens")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux (ph/s/m^2/um)")
    plt.savefig(out_path)
    logging.info(
        "Saved plot of flux of %s passed through transmission screens: %s",
        source_name,
        out_path,
    )
    plt.close()


def plot_source_transmission_triptych(
    source_img,
    transmission_img,
    source_name: str,
    screen_name: str,
    wavel_idx: int,
    source_unit: str,
    out_path: str,
) -> None:
    """Triptych: source map, transmission map, and product."""
    source_times_transmission_img = source_img * transmission_img
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    im0 = axs[0].imshow(source_img, origin="lower", cmap="gray")
    fig.suptitle(f"Source: {source_name}, idx_wavel: {wavel_idx}")
    axs[0].set_title("Source")
    axs[0].set_xlabel("x (pixel)")
    axs[0].set_ylabel("y (pixel)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label=f"{source_unit}")
    im1 = axs[1].imshow(transmission_img, origin="lower", cmap="gray")
    axs[1].set_title("Transmission")
    axs[1].set_xlabel("x (pixel)")
    axs[1].set_ylabel("y (pixel)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="transmission")
    im2 = axs[2].imshow(source_times_transmission_img, origin="lower", cmap="gray")
    axs[2].set_title(
        f"Source * Transmission ({screen_name})\n"
        f"({np.sum(source_times_transmission_img) / np.sum(source_img):.2f} transmitted; not chopped)"
    )
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved source/transmission triptych to %s", out_path)


def plot_pre_aperture_all_sources(
    flux_by_source: Mapping[str, tuple],
    config,
    out_path: str,
) -> None:
    """Overplot pre-screen incident fluxes for all included sources."""
    plt.clf()
    plt.figure(figsize=(8, 8))
    wavel_unit = None
    flux_unit = None
    for source_name, (wavel, flux) in flux_by_source.items():
        plt.plot(wavel, flux, label=format_astro_source_label(source_name))
        wavel_unit = wavel.unit
        flux_unit = flux.unit
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlim([4, 18])
    plt.ylim([1e-3, 1e10])
    plt.xlabel(f"Wavelength ({wavel_unit})")
    plt.ylabel(f"Flux ({flux_unit})")
    plt.legend()
    plt.title(
        format_plot_title("Photoelectrons, pre-aperture (no nulling yet)", config),
        loc="left",
    )
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info("Saved plot of incident flux pre-aperture to %s", out_path)
