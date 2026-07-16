"""Load S/N cubes written by sim_pipeline save_s2n_cube()."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable


def ensure_sim_pipeline_on_path(repo_root: Path | None = None) -> Path:
    """Add sim_pipeline to sys.path when running scripts from plotting_scripts/."""
    root = (repo_root or Path.cwd()).resolve()
    sim_pipeline = root / "sim_pipeline"
    if not sim_pipeline.is_dir():
        sim_pipeline = root.parent / "sim_pipeline"
    if str(sim_pipeline) not in sys.path:
        sys.path.insert(0, str(sim_pipeline))
    return sim_pipeline


def get_read_s2n_cube_fn(*, reload_calculator: bool = True) -> Callable[[str | Path], Any]:
    """Return read_s2n_cube_hdf5, optionally reloading calculator for dev sessions."""
    ensure_sim_pipeline_on_path()
    import modules.core.calculator as calculator

    if reload_calculator:
        importlib.reload(calculator)

    return getattr(calculator, "read_s2n_cube_hdf5", calculator.load_s2n_cube)


def load_s2n_cube(path: str | Path, *, reload_calculator: bool = True) -> Any:
    """Load an S/N cube from HDF5 (or .pkl via load_s2n_cube fallback)."""
    read_s2n_cube_hdf5 = get_read_s2n_cube_fn(reload_calculator=reload_calculator)
    return read_s2n_cube_hdf5(str(path))


def format_cube_plot_title(cube: Any, base_title: str) -> str:
    """Build a multi-line plot title from cube metadata (like sim_pipeline plots)."""
    ensure_sim_pipeline_on_path()
    from modules.utils.helpers import _join_two_column_text

    body = _join_two_column_text(cube.title_context, cube.sources_context)
    if not body:
        return base_title
    underline = "=" * len(base_title)
    return f"{base_title}\n{underline}\n{body}"


def print_cube_statistics(cube: Any) -> None:
    """Print shape and coordinate summaries for a loaded cube."""
    print("-------- VITAL STATISTICS --------")
    print("snr_cube shape (wavelength, DC, QE):", cube.snr.shape)
    print("wavelength (um):", cube.wavelength.min(), "-", cube.wavelength.max())
    print("dark_current (e/pix/s):", cube.dark_current)
    print("qe:", cube.qe)
    print("----------------------------------")
