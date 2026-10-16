"""Shared pytest fixtures (sim_pipeline/ is put on
sys.path via pyproject's pythonpath).
"""

import numpy as np
import pytest
from astropy import units as u

from modules.core.instrumental.channels import OutputChannel
from modules.core.instrumental.detector import Detector
from modules.core.instrumental.dispersion import DispersionLaw

CHANNEL_NAMES = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]


@pytest.fixture
def make_output_channels():
    """Factory for the four OutputChannels with real detectors and footprints."""

    def _make(n_bins: int = 3, gain: float = 2.0):
        config = {
            "detector": {
                "size": "400",
                "pitch_pix": "25",
                "pix_per_wavel_bin": "2.2",
                "pix_spectral_width": "3",
                "quantum_efficiency": "0.8",
                "gain": str(gain),
            }
        }
        edges = np.linspace(5.0, 5.0 + 0.5 * n_bins, n_bins + 1) * u.um
        channels = {}
        for name in CHANNEL_NAMES:
            det = Detector(config=config, num_wavel_bins=n_bins)
            law = DispersionLaw(config)
            det.set_footprint(law.make_footprint(det.side_length_pix, n_bins))
            channels[name] = OutputChannel(
                name=name,
                detector=det,
                bin_edges=edges,
                bin_centers=0.5 * (edges[:-1] + edges[1:]),
                bin_widths=np.diff(edges),
            )
        return channels

    return _make
