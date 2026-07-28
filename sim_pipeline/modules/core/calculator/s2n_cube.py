"""S/N cube dataclass and persistence helpers."""

import configparser
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Union

import h5py
import numpy as np

from ...utils.helpers.formatting import format_plot_title

logger = logging.getLogger(__name__)


@dataclass
class S2NCube:
    """S/N on a (wavelength, dark current, QE) grid plus plot metadata."""

    snr: np.ndarray
    wavelength: np.ndarray
    wavel_bin_width: np.ndarray
    wavel_bin_edges: np.ndarray
    dark_current: np.ndarray
    qe: np.ndarray
    snr_tot: np.ndarray
    base_titles: np.ndarray
    title_context: str
    sources_context: str
    read_dir: str
    n_angles: int
    n_int_per_angle: int
    t_int_frame: float
    n_int_total: float
    config: Dict[str, Dict[str, str]] = field(default_factory=dict)


def _config_to_dict(config) -> Dict[str, Dict[str, str]]:
    if isinstance(config, configparser.ConfigParser):
        return {section: dict(config[section]) for section in config.sections()}
    return {section: dict(values) for section, values in config.items() if isinstance(values, dict)}


def save_s2n_cube(
    cube: S2NCube,
    output_path: Union[str, Path],
    *,
    file_format: Literal["hdf5", "pickle", "both"] = "hdf5",
) -> List[str]:
    """
    Save an S2NCube to disk.

    HDF5 layout:
      snr (n_wavel, n_dc, n_qe), snr_tot (n_dc, n_qe), coordinate arrays,
      base_titles (n_dc, n_qe), and string metadata for plot titles.
    """
    output_path = Path(output_path)
    saved_paths: List[str] = []

    if file_format in {"pickle", "both"}:
        pickle_path = output_path if output_path.suffix == ".pkl" else output_path.with_suffix(".pkl")
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as handle:
            pickle.dump(cube, handle, protocol=pickle.HIGHEST_PROTOCOL)
        saved_paths.append(str(pickle_path))
        logger.info("Saved S/N cube pickle to %s", pickle_path)

    if file_format in {"hdf5", "both"}:
        hdf5_path = output_path if output_path.suffix in {".h5", ".hdf5"} else output_path.with_suffix(".hdf5")
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("snr", data=cube.snr, compression="gzip")
            handle.create_dataset("snr_tot", data=cube.snr_tot, compression="gzip")
            handle.create_dataset("wavelength", data=cube.wavelength)
            handle.create_dataset("wavel_bin_width", data=cube.wavel_bin_width)
            handle.create_dataset("wavel_bin_edges", data=cube.wavel_bin_edges)
            handle.create_dataset("dark_current", data=cube.dark_current)
            handle.create_dataset("qe", data=cube.qe)

            base_titles_flat = np.array(
                [str(title) for title in cube.base_titles.ravel()],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            handle.create_dataset("base_titles", data=base_titles_flat.reshape(cube.base_titles.shape))

            meta = handle.create_group("meta")
            meta.attrs["title_context"] = cube.title_context
            meta.attrs["sources_context"] = cube.sources_context
            meta.attrs["read_dir"] = cube.read_dir
            meta.attrs["n_angles"] = cube.n_angles
            meta.attrs["n_int_per_angle"] = cube.n_int_per_angle
            meta.attrs["t_int_frame"] = cube.t_int_frame
            meta.attrs["n_int_total"] = cube.n_int_total
            meta.attrs["axis_order"] = "wavelength, dark_current, qe"
            meta.attrs["config_json"] = json.dumps(cube.config)

            formatted_titles = np.empty(cube.base_titles.shape, dtype=object)
            for i_dc in range(cube.dark_current.size):
                for i_qe in range(cube.qe.size):
                    formatted_titles[i_dc, i_qe] = format_plot_title(
                        str(cube.base_titles[i_dc, i_qe]),
                        cube.config,
                    )
            formatted_flat = np.array(
                [str(title) for title in formatted_titles.ravel()],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            meta.create_dataset("formatted_plot_titles", data=formatted_flat.reshape(formatted_titles.shape))

        saved_paths.append(str(hdf5_path))
        logger.info("Saved S/N cube HDF5 to %s", hdf5_path)

    return saved_paths


def read_s2n_cube_hdf5(path: Union[str, Path]) -> S2NCube:
    """
    Read an S/N cube HDF5 file written by ``save_s2n_cube``.

    The primary data array ``cube.snr`` has shape ``(n_wavelength, n_dc, n_qe)``
    with coordinate vectors ``cube.wavelength``, ``cube.dark_current``, and ``cube.qe``.
  """
    path = Path(path)
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError(f"Expected an HDF5 path (.h5 / .hdf5), got: {path}")

    with h5py.File(path, "r") as handle:
        meta = handle["meta"]
        config = json.loads(meta.attrs["config_json"])
        return S2NCube(
            snr=np.array(handle["snr"]),
            wavelength=np.array(handle["wavelength"]),
            wavel_bin_width=np.array(handle["wavel_bin_width"]),
            wavel_bin_edges=np.array(handle["wavel_bin_edges"]),
            dark_current=np.array(handle["dark_current"]),
            qe=np.array(handle["qe"]),
            snr_tot=np.array(handle["snr_tot"]),
            base_titles=np.array(handle["base_titles"]).astype(str),
            title_context=str(meta.attrs["title_context"]),
            sources_context=str(meta.attrs["sources_context"]),
            read_dir=str(meta.attrs["read_dir"]),
            n_angles=int(meta.attrs["n_angles"]),
            n_int_per_angle=int(meta.attrs["n_int_per_angle"]),
            t_int_frame=float(meta.attrs["t_int_frame"]),
            n_int_total=float(meta.attrs["n_int_total"]),
            config=config,
        )


def load_s2n_cube(path: Union[str, Path]) -> S2NCube:
    """Load an S2NCube from pickle or HDF5."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            return pickle.load(handle)
    return read_s2n_cube_hdf5(path)
