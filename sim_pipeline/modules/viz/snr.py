"""S/N visualization helpers."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from ..utils.helpers.formatting import format_plot_title

logger = logging.getLogger(__name__)


def plot_snr_vs_wavelength(
    snr_lambda,
    wavel_bin_edges,
    base_title: str,
    config,
    out_path: str,
) -> None:
    """Stairs plot of SNR vs wavelength for one DC/QE slot."""
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    plt.clf()
    plt.stairs(snr_lambda, edges=wavel_bin_edges)
    plt.xlim([4, 18.5])
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("SNR")
    plt.title(format_plot_title(base_title, config), fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info("Saved plot of SNR vs wavelength to %s", out_path)
    plt.close(fig)
