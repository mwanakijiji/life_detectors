"""Astrophysical source plots."""

from __future__ import annotations

import logging
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib.colors import LogNorm

from ..utils.helpers.formatting import format_plot_title

logger = logging.getLogger(__name__)


def plot_incident_flux(wavel, flux, source_name, config, out_path: str, distance_pc: float) -> None:
    """Plot incident flux vs wavelength for one astrophysical source."""
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.plot(wavel, flux)
    plt.yscale("log")
    plt.xlim([4, 18])
    plt.ylim([1e-3, 1e9])
    plt.xlabel(f"Wavelength ({wavel.unit})")
    plt.ylabel(f"Flux ({flux.unit})")
    plt.title(
        format_plot_title(
            f"Incident flux from {source_name} (at Earth, rescaled for distance {distance_pc} pc)",
            config,
        )
    )
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info("Saved plot of incident flux to %s", out_path)


def plot_onsky_scene_fyi(
    scene_cubes: Mapping[str, object],
    total_cube,
    out_path: str,
) -> None:
    """Plot per-source and total on-sky scene collapsed over wavelength."""
    scene_names = list(scene_cubes.keys())
    cubes = list(scene_cubes.values())
    n_cols = len(cubes) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)

    for i, (source_cube, name) in enumerate(zip(cubes, scene_names)):
        ax = axes[0, i]
        summed = np.sum(source_cube, axis=0)
        im = ax.imshow(
            summed.value,
            origin="lower",
            norm=LogNorm(),
            aspect="equal",
            interpolation="none",
        )
        ax.set_title(f"{name}")
        try:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            logger.warning("No colorbar for %s", name)

    ax = axes[0, -1]
    total_summed = np.sum(total_cube, axis=0)
    total_data = np.asarray(total_summed.value, dtype=float)
    vmin, vmax = ZScaleInterval().get_limits(total_data)
    im = ax.imshow(
        total_data,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="none",
    )
    ax.set_title("Total (z-scale)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.suptitle("FYI: On-Sky Scene for Each Source (Sum Across Wavelength)", y=1.02)
    plt.subplots_adjust(top=0.85)
    plt.savefig(out_path, bbox_inches="tight")
    logger.info("FYI scene plot written to %s", out_path)
    plt.close(fig)
