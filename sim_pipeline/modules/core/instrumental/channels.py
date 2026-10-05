from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import Detector

# short channel names used in FITS files (EXTNAMEs and CHANi header values)
FITS_CHANNEL_NAMES = {
    "output_1_bright": "CH1BR",
    "output_2_bright": "CH2BR",
    "output_3_dark": "CH3DK",
    "output_4_dark": "CH4DK",
}


@dataclass
class OutputChannel:
    """
    Holds signals as detected in each output channel
    name: name of the output channel
    snr: signal-to-noise ratio of the signal in the channel
    """

    name: str
    detector: Optional[Detector] = None
    astroph_signal: dict = field(
        default_factory=dict
    )  # astrophysical signals for this channel
    readout_frames: list = field(
        default_factory=list
    )  # DetectorFrames for each readout at this angle (written to FITS)
    extracted_spectra_adu: dict = field(
        default_factory=dict
    )  # per-component ADU, shape (n_readouts, n_bins), box-summed from the FITS readouts
    snr: Optional[float] = None
    spec_R: float | None = None  # spectral R
    angle_deg: float | None = None  # rotation angle of transmission screen
    bin_edges: np.ndarray | None = None  # wavelength bins
    bin_centers: np.ndarray | None = None  # wavelength bins
    bin_widths: np.ndarray | None = None  # wavelength bins
