from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.table import QTable

from .detector import Detector


@dataclass
class OutputChannel:
    '''
    Holds signals as detected in each output channel
    name: name of the output channel
    snr: signal-to-noise ratio of the signal in the channel
    '''
    name: str
    detector: Optional[Detector] = None
    instrum_noise: dict = field(default_factory=dict)   # instrumental terms for this channel
    astroph_signal: dict = field(default_factory=dict)   # astrophysical signals for this channel
    snr: Optional[float] = None
    spec_R: float | None = None # spectral R
    angle_deg: float | None = None # rotation angle of transmission screen
    bin_edges: np.ndarray | None = None # wavelength bins
    bin_centers: np.ndarray | None = None # wavelength bins
    bin_widths: np.ndarray | None = None # wavelength bins
    tables_by_dark_current: dict[float, QTable] = field(default_factory=dict) # stores all the data relevant to S/N calculations downstream (one entry for each value of DC)
