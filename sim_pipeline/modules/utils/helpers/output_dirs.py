"""
Per-run output directory tree.

Every run writes into a fresh tree, so files from previous runs are never mixed in::

    <[dirs] save_s2n_data_unique_dir>/run_<YYYYMMDDHHMMSS>/   <- run root (new per run)
        [<per-planet folder>/]                            <- planet-population runs
            fyi_plots/                 FYI / diagnostic plots (only if plotting)
            intermediate/              intermediate FITS, HDF5 and text files
            products_counts_vs_time/   extracted-spectra cubes and the 2D
                                       counts vs. wavelength and time plots
            products_snr/              S/N cubes and S/N plots

Writers call ``output_dir(config, kind)``; the run root is
``config["dirs"]["save_s2n_data_unique_dir"]`` (set to a fresh ``make_run_root`` by
batch_process), and the subdirectory is created on first use.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

from ..types import ConfigLike

logger = logging.getLogger(__name__)

RUN_DIR_PREFIX = "run_"
FYI = "fyi"
INTERMEDIATE = "intermediate"
COUNTS_TIME = "counts_time"
SNR = "snr"
OUTPUT_SUBDIRS = {
    FYI: "fyi_plots",
    INTERMEDIATE: "intermediate",
    COUNTS_TIME: "products_counts_vs_time",
    SNR: "products_snr",
}


def run_stamp(now: Optional[datetime] = None) -> str:
    """Date-time stamp for a run directory, e.g. '20261105143012'."""
    return (now or datetime.now()).strftime("%Y%m%d%H%M%S")


def make_run_root(base_dir: str, now: Optional[datetime] = None) -> str:
    """
    Create and return a new, empty run root ``<base_dir>/run_<YYYYMMDDHHMMSS>/``.

    If that directory already exists (two runs started in the same second), a suffix
    ``_2``, ``_3``, ... is added, so an existing tree is never reused.
    """
    base = os.path.abspath(os.path.expanduser(str(base_dir)))
    stem = os.path.join(base, f"{RUN_DIR_PREFIX}{run_stamp(now)}")
    candidate, n = stem, 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{stem}_{n}"
    os.makedirs(candidate)
    logger.info("Created new run output directory: %s", candidate)
    return candidate + os.sep


def _root_from_config(config: ConfigLike) -> str:
    root = str(config["dirs"]["save_s2n_data_unique_dir"])
    return root if root.endswith(("/", os.sep)) else root + os.sep


def output_dir(config: ConfigLike, kind: str) -> str:
    """
    Path (with trailing separator) of the ``kind`` subdirectory of this run's output
    root: one of 'fyi', 'intermediate', 'counts_time', 'snr'. Created if needed.
    """
    if kind not in OUTPUT_SUBDIRS:
        raise ValueError(f"kind must be one of {sorted(OUTPUT_SUBDIRS)}; got {kind!r}")
    path = os.path.join(_root_from_config(config), OUTPUT_SUBDIRS[kind])
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def resolve_in_run_dir(config: ConfigLike, file_name: str, kind: str) -> str:
    """
    A bare file name (no directory part) is placed in this run's ``kind``
    subdirectory; a path with a directory is returned unchanged (an explicit
    location chosen in the config).
    """
    file_name = str(file_name).strip()
    if os.path.dirname(file_name):
        return file_name
    return output_dir(config, kind) + file_name


def previous_outputs_for_qe(root: str, qe: float) -> List[str]:
    """Files under the run root's output subdirectories already written for this QE."""
    tag = f"qe_{qe:04.2f}"
    found = []
    for subdir in OUTPUT_SUBDIRS.values():
        path = os.path.join(root, subdir)
        if not os.path.isdir(path):
            continue
        found += [
            os.path.join(path, name) for name in sorted(os.listdir(path)) if tag in name
        ]
    return found


def assert_no_previous_outputs(config: ConfigLike, qe: float) -> None:
    """
    Raise FileExistsError if this run's output tree already holds files for this QE
    (i.e. it would mix with, or overwrite, another run's outputs).
    """
    root = _root_from_config(config)
    found = previous_outputs_for_qe(root, qe)
    if found:
        listing = "\n  ".join(found[:10]) + ("\n  ..." if len(found) > 10 else "")
        raise FileExistsError(
            f"Output directory {root} already has {len(found)} file(s) for QE "
            f"{qe:04.2f} from a previous run; refusing to mix runs:\n  {listing}"
        )
