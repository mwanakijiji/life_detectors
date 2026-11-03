"""Shared pytest fixtures (sim_pipeline/ is put on
sys.path via pyproject's pythonpath).
"""

import logging
import os

import numpy as np
import pytest
from astropy import units as u

from modules.core.instrumental.channels import OutputChannel
from modules.core.instrumental.detector import Detector
from modules.core.instrumental.dispersion import DispersionLaw
from modules.utils.loader import LOG_DIR_ENV_VAR

CHANNEL_NAMES = ["output_1_bright", "output_2_bright", "output_3_dark", "output_4_dark"]


def pytest_addoption(parser):
    """--run-smoke opts in to the (slower) end-to-end smoke tests."""
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="run tests marked 'smoke' (whole-pipeline smoke-test runs)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip 'smoke' tests unless --run-smoke is given."""
    if config.getoption("--run-smoke"):
        return
    skip_smoke = pytest.mark.skip(reason="smoke test: run with --run-smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)


@pytest.fixture(scope="session", autouse=True)
def isolated_log_dir(tmp_path_factory):
    """Send default-location log files to a temp dir, never sim_pipeline/logs/."""
    log_dir = tmp_path_factory.mktemp("logs")
    previous = os.environ.get(LOG_DIR_ENV_VAR)
    os.environ[LOG_DIR_ENV_VAR] = str(log_dir)
    yield log_dir
    if previous is None:
        os.environ.pop(LOG_DIR_ENV_VAR, None)
    else:
        os.environ[LOG_DIR_ENV_VAR] = previous
    # close file handlers that ensure_logging() attached during the session
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and str(
            handler.baseFilename
        ).startswith(str(log_dir)):
            root.removeHandler(handler)
            handler.close()


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
            det.set_footprint(law.make_footprint(det.n_pix_y, det.n_pix_x, n_bins))
            channels[name] = OutputChannel(
                name=name,
                detector=det,
                bin_edges=edges,
                bin_centers=0.5 * (edges[:-1] + edges[1:]),
                bin_widths=np.diff(edges),
            )
        return channels

    return _make
