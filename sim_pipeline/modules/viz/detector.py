"""Detector geometry plots."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, ZScaleInterval

from ..utils.helpers.formatting import format_plot_title
from ..utils.types import ConfigLike

logger = logging.getLogger(__name__)


def plot_detector_footprint(
    footprint_sum: np.ndarray, config: ConfigLike, out_path: str
) -> None:
    """Plot the wavelength-summed detector spectral footprint."""
    plt.clf()
    plt.title(format_plot_title("Detector spectral footprint (True)", config))
    norm = ImageNormalize(footprint_sum, interval=ZScaleInterval())
    plt.imshow(footprint_sum, origin="lower", cmap="gray", norm=norm)
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(out_path)
    logging.info(
        "Saved plot of detector footprint containing all wavelength bins to %s",
        out_path,
    )
