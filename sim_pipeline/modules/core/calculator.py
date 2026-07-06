"""
Main noise calculator for the modules package.

This module provides the primary interface for calculating total noise
and signal-to-noise ratios for infrared detector observations.
"""

import glob
import json
import logging
import os
import re
from pathlib import Path

import astropy.units as u
import configparser
import h5py
import ipdb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy.io import fits
from astropy.table import QTable
from dataclasses import dataclass, field
from ipaddress import ip_network
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .astrophysical import AstrophysicalSources
from .instrumental import Detector, InstrumentDepTerms
from ..data.units import UnitConverter
from ..utils.helpers import (
    build_astrophysical_sources_to_use_title,
    ensure_plot_title_context,
    format_plot_title,
)
from ..utils.validator import validate_config

logger = logging.getLogger(__name__)

_DC_QE_GROUP_RE = re.compile(r"^dc_(?P<dc>.+)_qe_(?P<qe>[0-9.]+)$")


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


def _parse_dc_qe_group(dc_qe_str: str) -> Tuple[float, float]:
    match = _DC_QE_GROUP_RE.match(dc_qe_str)
    if match is None:
        raise ValueError(f"Unrecognized HDF5 group name: {dc_qe_str!r}")
    return float(match.group("dc")), float(match.group("qe"))


def read_hdf5_slots(read_dir: str) -> Dict[str, dict]:
    """Read angle_*.hdf5 files and aggregate tables by dc/qe group."""
    hdf5_files = sorted(glob.glob(os.path.join(read_dir, "angle_*.hdf5")))
    by_dc_qe: Dict[str, dict] = {}

    for hdf5_file in hdf5_files:
        angle = float(Path(hdf5_file).stem.removeprefix("angle_"))

        with h5py.File(hdf5_file, "r") as f:
            for dc_qe_str in f.keys():
                if dc_qe_str.startswith("__"):
                    continue

                chopped = QTable.read(hdf5_file, path=f"{dc_qe_str}/chopped")
                out3 = QTable.read(hdf5_file, path=f"{dc_qe_str}/output_3_dark")
                S_p = chopped[
                    "chopped_astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"
                ]
                S_p_3 = out3[
                    "astro_exoplanet_model_10pc_flux_adu_sec_for_wavel_bin_and_integration_tot"
                ]

                slot = by_dc_qe.setdefault(
                    dc_qe_str,
                    {
                        "wavel": chopped["wavel_bin_center"].value,
                        "wavel_bin_edges": chopped.meta["wavel_bin_edges"],
                        "S_p": {},
                        "S_p_3": {},
                        "chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot": {},
                        "chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot": {},
                    },
                )
                slot["S_p"][angle] = S_p
                slot["S_p_3"][angle] = S_p_3
                slot["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"][angle] = (
                    chopped["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"]
                )
                slot["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"][angle] = (
                    chopped["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"]
                )
                slot["wavel_bin_width"] = chopped["wavel_bin_width"]

                sym_tags = ("star", "exozodiacal", "zodiacal")
                slot.setdefault("sources_sym", {})
                for source_name in sym_tags:
                    col = f"astro_{source_name}_flux_adu_sec_for_wavel_bin_and_integration_tot"
                    if col not in out3.colnames:
                        continue
                    slot["sources_sym"].setdefault(source_name, {"Ssym_dark_3": {}})
                    slot["sources_sym"][source_name]["Ssym_dark_3"][angle] = chopped[
                        f"output_3_dark_astro_{source_name}_flux_adu_sec_for_wavel_bin_and_integration_tot"
                    ]

    return by_dc_qe


def _compute_snr_lambda_for_slot(slot: dict, config) -> Tuple[np.ndarray, float]:
    """Compute per-bin and total SNR for one dc/qe slot (Dannert+ 2022 Eqn. 19-20)."""
    gain = float(config["detector"]["gain"]) * u.electron / u.adu
    angles = sorted(slot["S_p"].keys())
    ref_angle = angles[0]

    cols_S_p_elec = [np.asarray(slot["S_p"][a] * gain) for a in angles] * u.electron
    cols_S_p_3_elec = [np.asarray(slot["S_p_3"][a] * gain) for a in angles] * u.electron
    S_p_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_elec), 2), axis=1)
    S_p_3_sqd_arr_mean_elec = np.mean(np.power(np.column_stack(cols_S_p_3_elec), 2), axis=1)

    sources_sym = slot.get("sources_sym", {})
    S_sym_noise_var_3_elec = None
    logging.info(
        "Astrophysical sources considered to be symmetric: %s",
        list(sources_sym.keys()),
    )
    for source_name, source_dict in sources_sym.items():
        cols_sym_noise_var_3_elec = []
        for a in angles:
            sym_noise_var_this_source_this_angle_dark_3_elec = (
                source_dict["Ssym_dark_3"][a] * gain
            )
            cols_sym_noise_var_3_elec.append(
                np.sqrt(sym_noise_var_this_source_this_angle_dark_3_elec.value) * u.electron
            )
        sym_noise_var_mean_3_elec = np.mean(
            np.column_stack(cols_sym_noise_var_3_elec).value * u.electron, axis=1
        )
        if S_sym_noise_var_3_elec is None:
            S_sym_noise_var_3_elec = sym_noise_var_mean_3_elec
        else:
            S_sym_noise_var_3_elec = S_sym_noise_var_3_elec + sym_noise_var_mean_3_elec

    if S_sym_noise_var_3_elec is None:
        S_sym_noise_var_3_elec = np.zeros(len(slot["wavel_bin_width"])) * u.electron
    elif S_sym_noise_var_3_elec.unit != u.electron:
        logger.error("Unit inconsistency in symmetric astrophysical noise sources!")
        raise ValueError("Symmetric astrophysical noise sources have inconsistent units")

    snr_lambda_array = []
    for wavel_bin_num in range(len(slot["wavel_bin_width"])):
        S_p_rms_phi = np.sqrt(S_p_sqd_arr_mean_elec[wavel_bin_num])
        S_p_3_rms_phi = np.sqrt(S_p_3_sqd_arr_mean_elec[wavel_bin_num])

        S_dark_noise_var = (
            np.power(
                slot["chopped_instrum_dark_current_rms_for_wavel_bin_and_integration_adu_tot"][
                    ref_angle
                ][wavel_bin_num]
                * gain,
                2,
            ).value
            * u.electron
        )
        S_read_noise_var = (
            np.power(
                slot["chopped_instrum_read_noise_rms_for_wavel_bin_and_integration_adu_tot"][
                    ref_angle
                ][wavel_bin_num]
                * gain,
                2,
            ).value
            * u.electron
        )

        S_sym_3_var_this = S_sym_noise_var_3_elec[wavel_bin_num]
        astro_noise_term = 2 * (S_sym_3_var_this + S_p_3_rms_phi)
        instrum_noise_term = 2 * (S_dark_noise_var + S_read_noise_var)
        denominator_ = np.sqrt(astro_noise_term + instrum_noise_term).value * u.electron
        snr_lambda_array.append((S_p_rms_phi / denominator_).value)

    snr_lambda_array = np.asarray(snr_lambda_array)
    snr_tot = float(np.sqrt(np.sum(np.power(snr_lambda_array, 2))))
    return snr_lambda_array, snr_tot


def _build_base_title(
    *,
    dc_qe_str: str,
    snr_tot: float,
    n_angles_cfg: int,
    n_int_per_angle: int,
    n_int_total: float,
) -> str:
    return (
        f"SNR for DC {dc_qe_str}  |  SNR_tot = {snr_tot:.4g}  |  "
        f"N_angles = {n_angles_cfg}  |  N_int_per_angle = {n_int_per_angle}  |  "
        f"N_int tot = {n_int_total} sec"
    )


def build_s2n_cube_from_hdf5(read_dir: str, config) -> S2NCube:
    """
    Read pipeline HDF5 files and assemble S/N on a (wavelength, DC, QE) cube.

    Expects ``angle_*.hdf5`` files written by ``record_info_at_angle_and_qe``,
    with groups named ``dc_{dc}_qe_{qe}``.
    """
    ensure_plot_title_context(config)
    by_dc_qe = read_hdf5_slots(read_dir)
    if not by_dc_qe:
        raise FileNotFoundError(f"No angle_*.hdf5 files found in {read_dir}")

    t_int_frame = float(config["observation"]["t_int_frame"])
    n_angles_cfg = int(float(config["observation"]["N_angles"]))
    n_int_per_angle = int(float(config["observation"]["N_int_per_angle"]))
    n_int_total = n_angles_cfg * n_int_per_angle * t_int_frame
    title_context = ensure_plot_title_context(config)
    sources_context = build_astrophysical_sources_to_use_title(config)

    dc_values = []
    qe_values = []
    parsed = []
    for dc_qe_str in by_dc_qe:
        dc_val, qe_val = _parse_dc_qe_group(dc_qe_str)
        dc_values.append(dc_val)
        qe_values.append(qe_val)
        parsed.append((dc_qe_str, dc_val, qe_val))

    dark_current = np.array(sorted(set(dc_values)))
    qe = np.array(sorted(set(qe_values)))
    n_wavel = len(next(iter(by_dc_qe.values()))["wavel"])
    snr_cube = np.full((n_wavel, len(dark_current), len(qe)), np.nan)
    snr_tot = np.full((len(dark_current), len(qe)), np.nan)
    base_titles = np.empty((len(dark_current), len(qe)), dtype=object)

    ref_slot = next(iter(by_dc_qe.values()))
    wavelength = np.asarray(ref_slot["wavel"])
    wavel_bin_width = np.asarray(ref_slot["wavel_bin_width"].value)
    wavel_bin_edges = np.asarray(ref_slot["wavel_bin_edges"].value)

    dc_index = {val: idx for idx, val in enumerate(dark_current)}
    qe_index = {val: idx for idx, val in enumerate(qe)}

    for dc_qe_str, dc_val, qe_val in parsed:
        snr_lambda, snr_total = _compute_snr_lambda_for_slot(by_dc_qe[dc_qe_str], config)
        i_dc = dc_index[dc_val]
        i_qe = qe_index[qe_val]
        snr_cube[:, i_dc, i_qe] = snr_lambda
        snr_tot[i_dc, i_qe] = snr_total
        base_titles[i_dc, i_qe] = _build_base_title(
            dc_qe_str=dc_qe_str,
            snr_tot=snr_total,
            n_angles_cfg=n_angles_cfg,
            n_int_per_angle=n_int_per_angle,
            n_int_total=n_int_total,
        )
        logging.info("SNR_tot for %s: %s", dc_qe_str, snr_total)

    return S2NCube(
        snr=snr_cube,
        wavelength=wavelength,
        wavel_bin_width=wavel_bin_width,
        wavel_bin_edges=wavel_bin_edges,
        dark_current=dark_current,
        qe=qe,
        snr_tot=snr_tot,
        base_titles=base_titles,
        title_context=title_context,
        sources_context=sources_context,
        read_dir=str(read_dir),
        n_angles=n_angles_cfg,
        n_int_per_angle=n_int_per_angle,
        t_int_frame=t_int_frame,
        n_int_total=n_int_total,
        config=_config_to_dict(config),
    )


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


def load_s2n_cube(path: Union[str, Path]) -> S2NCube:
    """Load an S2NCube from pickle or HDF5."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            return pickle.load(handle)

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

'''
def calculate_astrophysical_noise_adu(total_astro_adu: float) -> np.ndarray:
    """
    Calculate astrophysical noise in electrons per pixel.
    
    Args:
        total_astro_adu: the total astrophysical flux contribution to the readout in ADU
        
    Returns:
        Noise in electrons per pixel
    """
    
    # photon noise: sqrt(N)
    noise_adu = np.sqrt(total_astro_adu)
    
    return noise_adu
'''


def calculate_s2n_post_rotation(read_dir, config, *, save_cube_path: Optional[str] = None):
    """
    Calculate the S/N of the chopped dark outputs and optionally save an S/N cube.

    Args:
        read_dir: dir containing the HDF5 files
        config: configuration dictionary
        save_cube_path: optional path stem or full path for saved cube (.hdf5 / .pkl)

    Returns:
        S2NCube with axes (wavelength, dark_current, qe)
    """
    cube = build_s2n_cube_from_hdf5(read_dir, config)

    for i_dc, dc_val in enumerate(cube.dark_current):
        for i_qe, qe_val in enumerate(cube.qe):
            dc_qe_str = f"dc_{dc_val:g}_qe_{qe_val:.2f}"
            snr_lambda_array = cube.snr[:, i_dc, i_qe]
            snr_tot = cube.snr_tot[i_dc, i_qe]
            print(f"SNR_tot for DC {dc_qe_str}: {snr_tot}")

            if True:  # pragma: no cover
                fig = plt.figure(figsize=(8, 8), constrained_layout=True)
                plt.clf()
                plt.stairs(snr_lambda_array, edges=cube.wavel_bin_edges)
                plt.xlim([4, 18.5])
                plt.yscale("log")
                plt.grid(True)
                plt.xlabel("Wavelength (um)")
                plt.ylabel("SNR")
                base_title = str(cube.base_titles[i_dc, i_qe])
                plt.title(format_plot_title(base_title, config), fontsize=8, loc="left")
                file_name_plot = (
                    str(config["dirs"]["save_s2n_data_unique_dir"])
                    + f"SNR_vs_wavelength_{dc_qe_str}"
                    + f"_Nang_{cube.n_angles}_Nintpa_{cube.n_int_per_angle}_Ninttot_{cube.n_int_total}.png"
                )
                plt.tight_layout()
                plt.savefig(file_name_plot)
                logging.info("Saved plot of SNR vs wavelength for %s to %s", dc_qe_str, file_name_plot)

    if save_cube_path is not None:
        save_s2n_cube(cube, save_cube_path, file_format="both")

    return cube


class NoiseCalculator:
    """
    Puts astrophysical and instrumental sources together.
    """
    
    def __init__(self, config: Dict, sources_all, sources_to_include: list):
        """
        Initialize the noise calculator.
        
        Args:
            config: Configuration dictionary containing all parameters
            sources_all: the object including the various fluxes and noise contributions, from astro and instrumental sources
            sources_to_include: list of sources to include in the S/N calculation
            
        Raises:
            ValueError: If configuration is invalid
        """

        self.config = config

        # the object that is the 'origin' of the various noise contributions for the calculations to follow
        self.sources_all = sources_all
        self.sources_to_include = sources_to_include

    def s2n_e(self, file_name_fits_unique, plot: bool = False):
        '''
        Find S/N using photoelectrons

        Ref.: s_to_n_logic_life_detectors.pdf
        '''

        ## everything in units of photoelectrons
        
        #wavel_abcissa = self.noise_origin.prop_dict['wavel']
        logging.info("Calculating S/N ...")

        ## ## TO DO: MAKE ALL WAVELS TO BE ON AN EXPLICIT COMMON BASIS ACCORDING TO THE BINNING
        wavel_abcissa = self.sources_all.sources_astroph['star']['wavel']

        # a constant pixel spacing for each wavelength bin is assumed (but wavelength spacing within each bin is not constant)

        # map wavelengths-bins
        R = float(self.config["detector"]["spec_res"]) # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation lambda_i = lambda_{0} * (1 + 1/R)**i
        lambda_min, lambda_max = float(self.config["wavelength_range"]["min"]) * u.um, float(self.config["wavelength_range"]["max"])  * u.um
        # number of bins that fit fully in [lmin, lmax]
        n_bins = int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / R)))

        # geometric bin edges and centers
        bin_edges = lambda_min * (1.0 + 1.0 / R) ** np.arange(n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = bin_edges[1:]-bin_edges[:-1] # removed units for plotting

        # instantiate detector objects for each of the channels
        detector_bright_1 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_bright_2 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_dark_3 = Detector(config=self.config, num_wavel_bins=n_bins)
        detector_dark_4 = Detector(config=self.config, num_wavel_bins=n_bins)

        # get the boolean illumination footprint (cube where each slice is the footprint for one wavelength bin)
        ## ## NOTE THIS IS KIND OF REDUNDANT RIGHT NOW, SINCE THE NUMBER OF PIXELS PER WAVELENGTH BIN IS CONSTANT AS CALCULATED BELOW; MIGHT CHANGE THIS LATER IF THE DISPERSION IS NOT CONSTANT
        footprint_spec_cube_bright_1 = detector_bright_1.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_bright_1.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_bright_2 = detector_bright_2.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_bright_2.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_dark_3 = detector_dark_3.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_dark_3.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT
        footprint_spec_cube_dark_4 = detector_dark_4.footprint_spectral(file_name_plot=str(self.config['dirs']['save_s2n_data_unique_dir']) + 'footprint_bool_dark_4.png', plot=True) ## ## TO DO: MAKE THIS FUNCTION INHERIT THE SAVE DIR MORE CLEANLY, RAHTER THAN PASSING IT

        # integration time for 1 frame
        #t_int = float(self.config["observation"]["t_int_obs_total"]) * u.second
        n_int = int(self.config["observation"]["N_angles"]) # number of frames
        logging.info(f'Number of frames: {n_int:d}')

        # the number of pixels for each wavelength bin
        # (in practice the number of pixels is the same for all wavelength bins, but this can be updated later if the dispersion is not constant)
        #n_pix_array_reshaped = np.tile( np.sum(footprint_spec_cube[0,:,:]) * np.ones(len(wavel_bin_centers)), (len(D_tot), 1) ) * u.pix # shape (N_dark_current, N_wavel)
        n_pix_array = u.Quantity([]) * u.pix
        for wavel_bin_num in range(0, n_bins):
            val = np.sum(footprint_spec_cube_bright_1[wavel_bin_num, :, :]) * u.pix
            n_pix_array = u.Quantity(np.append(n_pix_array, val))

        # retrieve the addl systematics vector
        addl_systematics_vector_bright_1 = detector_bright_1.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_bright_2 = detector_bright_2.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_dark_3 = detector_dark_3.convert_2d_systematics_to_1d_vector()
        addl_systematics_vector_dark_4 = detector_dark_4.convert_2d_systematics_to_1d_vector()

        #ipdb.set_trace()
        
        # Now the calculation will broadcast to (N_dark_current, N_wavel)
        # return S/N; and the variable values that are either the dark current or read noise (whichever has length >1)
        #ipdb.set_trace()
        #s2n_dark_3 = self.s2n_val(wavel_bin_centers=bin_centers, del_lambda_array=bin_widths, n_pix_array=n_pix_array, addl_systematics_vector=addl_systematics_vector_dark_3)


    
        # write the S/N data to a FITS file, with the config data in the header
        file_name_fits_temp = self.config['saving']['save_s2n_data_temp'] # this is just a file that is repeatedly written over in the case of a batch job, but it handy if I want to check the last written thing
        hdu = fits.PrimaryHDU()  # s2n will be packed into this later

        # add the sweeped parameters to the header (these are what are used to make the axes of the big 4D cubes)
        hdu.header['N_INT'] = n_int
        hdu.header['QE'] = self.config["detector"]["quantum_efficiency"]

        # Add config data to header
        relevant_sections = ['telescope', 'target', 'nulling', 'detector', 'observation', 'wavelength_range']
        for section_name, _ in self.config.items():

            if section_name not in relevant_sections:
                continue

            # Add a blank line before each section (except the first)
            if len(hdu.header) > 0:
                hdu.header.add_blank('')
            
            # Add section header as a comment
            #hdu.header.add_comment(f"===== {section_name.upper()} =====")
            
            # Add each key-value pair in the section
            for key, value in self.config[section_name].items():
                # Create a hierarchical keyword using HIERARCH for long names
                # FITS keywords are limited to 8 characters, but HIERARCH allows arbitrary length
                hierarch_key = f"HIERARCH {section_name}.{key}"

                value_str = str(value)
                if len(value_str) > 68:
                    logger.warning(
                        f"Skipping FITS header entry (value too long): {hierarch_key}={value_str!r}"
                    )
                    continue
                # Validate keyword/value length before writing 
                fits.Card(hierarch_key, value)
                hdu.header[hierarch_key] = value
            

        # stuff for plots and FITS file too
        # parse dark current values (can be comma-separated list)
        dark_current_str = self.config['detector']['dark_current']
        if ',' in dark_current_str:
            #dark_current_values = [float(x.strip()) for x in dark_current_str.split(',')] * u.electron / (u.pix * u.second)
            #dark_current_display = ', '.join([f"{val:.2f}" for val in dark_current_values.value])
            parts = [float(x.strip()) for x in dark_current_str.split(',')]
            dark_current_values = np.arange(parts[0], parts[1], parts[2]) * u.electron / (u.pix * u.second)
            dark_current_display = ', '.join([f"{val:.2f}" for val in dark_current_values.value])
        else:
            dark_current_values = float(dark_current_str) * u.electron / (u.pix * u.second)
            dark_current_display = f"{float(dark_current_str):.2f}"
        # do the same for read noise
        read_noise_str = self.config['detector']['read_noise']
        if ',' in read_noise_str:
            read_noise_values = [float(x.strip()) for x in read_noise_str.split(',')] * u.electron
            read_noise_display = ', '.join([f"{val:.2f}" for val in read_noise_values.value])
        else:
            read_noise_display = f"{float(read_noise_str):.2f}"
            read_noise_values = float(read_noise_str) * u.electron

        # TODO: wire per-channel s2n_val calls for bright/dark outputs
        n_s2n_rows = int(np.size(dark_current_values)) if np.size(dark_current_values) > 1 else 1
        s2n = np.ones((n_s2n_rows, n_bins))

        # pack stuff into FITS file and save
        # add two slices denoting the wavelength bin centers, wavelength bin widths, and the dark current values
        s2n_wavel_bin_centers = np.zeros(s2n.shape)
        s2n_wavel_bin_widths = np.zeros(s2n.shape)
        s2n_dc = np.zeros(s2n.shape)
        # s2n_wavel is function of x only, s2n_dc is function of y only
        s2n_wavel_bin_centers[:,:] = np.tile(bin_centers.value, (s2n.shape[0], 1))
        s2n_wavel_bin_widths[:,:] = np.tile(bin_widths.value, (s2n.shape[0], 1))
        s2n_dc[:,:] = np.tile(dark_current_values.value.reshape(-1, 1), (1, s2n.shape[1]))
        s2n_complete = np.stack((s2n, s2n_wavel_bin_centers, s2n_wavel_bin_widths, s2n_dc), axis=0)
        hdu.data = s2n_complete
        hdu.writeto(file_name_fits_temp, overwrite=True)
        logger.info(f"Wrote S/N, wavelength, and dark current data to overwriteable file {file_name_fits_temp}")
        hdu.writeto(file_name_fits_unique, overwrite=True)
        logger.info(f"Wrote S/N, wavelength, and dark current data to unique file {file_name_fits_unique}")

        # pragma: no cover
        # Prepare two left-aligned columns for figure metadata
        total_integration_time = float(self.config['observation']['t_int_frame']) * float(self.config['observation']['N_angles']) * u.second
        instrumental_lines = [
            "INSTRUMENTAL:",
            "\n",
            f"collecting area = {float(self.config['telescope']['collecting_area']):.2f} m²",
            f"telescope throughput = {float(self.config['telescope']['eta_t']):.2f}",
            f"stellar nulling = {bool(self.config['nulling']['null'])}",
            f"nulling transmission = {float(self.config['nulling']['nulling_factor']):.2e}",
            f"quantum efficiency = {float(self.config['detector']['quantum_efficiency']):.2f}",
            f"dark current = {dark_current_display} e-/pix/sec",
            f"read noise = {read_noise_display} e- rms",
            f"gain = {float(self.config['detector']['gain']):.2f} e-/ADU",
            f"pix per wavel bin = {float(self.config['detector']['pix_per_wavel_bin']):.2f}",
            f"integration time, total for obs. = {total_integration_time} sec",
            f"integration time per readout = {float(self.config['observation']['t_int_frame']):.2f} sec",
            f"number of readouts = {int(n_int)}"
        ]
        astrophysical_lines = [
            "ASTROPHYSICAL:",
            "\n",
            f"stellar temperature = {float(self.config['target']['T_star']):.2f} K",
            f"stellar radius = {float(self.config['target']['rad_star']):.2f} solar radii",
            f"distance = {float(self.config['target']['distance']):.2f} pc",
            f"planet temperature = {float(self.config['target']['pl_temp']):.2f} K",
            f"planet radius = {float(self.config['target']['rad_planet']):.2f} Earth radii",
            f"planet albedo = {float(self.config['target']['A_albedo']):.2f}",
            fr"galactic $\lambda_{{\rm rel}}$ = {float(self.config['target']['lambda_rel_lon_los']):.2f} deg, $\beta$ = {float(self.config['target']['beta_lat_los']):.2f} deg",
        ]


        # pragma: no cover
        ############################
        # 2D plot of S/N vs wavelength and dark current
        if plot: # pragma: no cover
            plt.close()

            param_name = "Dark current" if np.size(dark_current_values) > 1 else "Read noise"
            param_units_string = 'e_pix-1_sec-1' if np.size(dark_current_values) > 1 else "e rms"
            param_values = dark_current_values if np.asarray(dark_current_values).size > 1 else read_noise_values
            #param_values = np.asarray(param_values, dtype=float)  
            N = len(param_values)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Heatmap of S/N with contours overlaid
            # make edges of the y-axis
            y_c = param_values.value
            y_edges = np.concatenate((
                [y_c[0] - 0.5*(y_c[1] - y_c[0])],
                0.5*(y_c[:-1] + y_c[1:]),
                [y_c[-1] + 0.5*(y_c[-1] - y_c[-2])]
            ))
            # Align imshow with physical axes using wavelength edges and parameter range
            im = ax.pcolormesh(
                bin_edges.value, y_edges, s2n,
                cmap='viridis', shading='flat'
                )
            levels_2d = self.config['plotting']['s2n_levels_2d']
            levels_2d = [float(x.strip()) for x in levels_2d.split(',')]
            contour = ax.contour(
                bin_centers.value, param_values.value, s2n,
                levels=levels_2d,
                colors=['white', 'white'],
                linewidths=2,
                linestyles=['dashed', 'solid']
            )
            '''
            ax.clabel(contour, inline=True, fmt='%g', fontsize=9)
            fig.colorbar(im, ax=ax, label='S/N')
            ax.set_xlabel(f"Wavelength ({bin_centers.unit})")
            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_title(format_plot_title("S/N", self.config))
            plt.tight_layout()
            plt.show()
            im = ax.imshow(
                s2n,
                extent=[bin_edges[0].value, bin_edges[-1].value,
                        np.min(param_values.value), np.max(param_values.value)],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            contour = ax.contour(
                bin_centers.value, param_values.value, s2n,
                levels=[1, 5],
                colors=['white', 'white'],
                linewidths=2,
                linestyles=['dashed', 'solid']
            )
            '''
            ax.clabel(contour, inline=True, fmt='%g', fontsize=9)
            fig.colorbar(im, ax=ax, label='S/N')
            ax.set_xlabel(f"Wavelength ({bin_centers.unit})")
            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_title(format_plot_title("S/N", self.config))
            plt.tight_layout()
            #plt.show()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"2d_s2n_vs_wavelength_and_dark_current.png"
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")

            '''
            tick_idx = np.linspace(0, N - 1, num=min(N, 10), dtype=int)
            ax.set_yticks(tick_idx)
            labels = [f"{param_values[i].value:g}" for i in tick_idx]
            ax.set_yticklabels(labels)

            ax.set_ylabel(param_name + " (" + param_units_string + ")")
            ax.set_xlabel("Wavelength bin (" + str(wavel_abcissa.unit) + ")")
            fig.colorbar(contour, ax=ax, label="S/N")
            # Keep a concise axes title and add two figure-level columns
            ax.set_title(format_plot_title("S/N", self.config))
            fig.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            #plt.tight_layout()
            plt.subplots_adjust(top=0.6,right=0.8)
            plt.show()
            '''

            #plt.imshow(s2n,aspect='auto', origin='lower')
            #plt.imshow(s2n, origin='lower')
            #plt.colorbar()

            # Set y-axis ticks to match D_rate_reshaped[:,0]
            #plt.yticks(range(len(D_rate_reshaped[:,0])), D_rate_reshaped[:,0])

            # Set bottom x-axis: wavelength
            plt.clf()
            # Set x-ticks at intervals of every three values along the x axis, and rotate the labels 30 degrees
            interval = 3
            # Set x-ticks at intervals of every two values along the x axis
            x_tick_indices = np.arange(0, len(wavel_abcissa), interval)

            ax_bottom = plt.gca()
            ax_bottom.set_xticks(x_tick_indices)
            ax_bottom.set_xticklabels(np.round(wavel_abcissa[x_tick_indices], 2), rotation=30)
            ax_bottom.set_xlabel('Wavelength (um)')

            # Add contour lines for S/N = 1 (dashed) and S/N = 5 (solid)
            X, Y = np.meshgrid(np.arange(s2n.shape[1]), np.arange(s2n.shape[0]))
            contour = ax_bottom.contour(
                X, Y, s2n, 
                levels=[1, 5], 
                colors=['white', 'white'],  
                linewidths=2, 
                linestyles=['dashed', 'solid']
            )
            ax_bottom.clabel(contour, inline=True, fmt='%.1f', fontsize=9)
            plt.ylabel(param_name + " (" + param_units_string + ")")
            # Keep a concise axes title and add two figure-level columns
            ax_bottom.set_title(format_plot_title("S/N", self.config))
            fig2 = plt.gcf()
            fig2.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig2.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            plt.tight_layout()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"2d_s2n_vs_wavelength_and_dark_current.png"
            #plt.show()
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")
            plt.close()

            

            '''
            for plot_num in range(0,len(s2n[:,0])):
                # draw a histogram-like plot of S/N using step plot that respects bin widths
                # Create x-coordinates that include both bin edges for proper step plotting
                x_step = np.repeat(wavel_bin_edges_lower, 2)
                x_step[1::2] = wavel_bin_edges_upper  # Replace every other element with upper edges
                y_step = np.repeat(s2n[plot_num,:], 2)  # Repeat y-values for step effect
                plt.plot(x_step, y_step, label=f'Dark current {plot_num}', linewidth=2)
                
                #plt.bar(wavel_bin_edges_lower.value, s2n[plot_num,:], width=bin_widths, align='edge', edgecolor='black', linewidth=0)
                #plt.plot(wavel_abcissa, s2n[plot_num,:], label=str(int(plot_num)))
            '''
            plt.figure(figsize=(10, 8))
            for plot_num in range(0,len(s2n[:,0])):
                plt.scatter(bin_centers, s2n[plot_num,:], alpha=0.5)
            # Add vertical lines at bin edges for reference
            #for line in wavel_bin_edges_lower:
            #     plt.axvline(x=line, color='gray', linestyle='--', alpha=0.5)
            cmap = cm.viridis  # or any other colormap
            norm = mcolors.Normalize(vmin=np.min(param_values.value), vmax=np.max(param_values.value))

            for plot_num in range(0,len(s2n[:,0])):
                color = cmap(norm(param_values[plot_num].value))
                plt.scatter(bin_centers, s2n[plot_num,:], label= param_name + ": " + f"{param_values[plot_num].value:g}" + " (" + param_units_string + ")", color=color)
            plt.axhline(y=1, color='gray', linestyle='--')
            plt.axhline(y=5, color='gray', linestyle='-')
            # Annotate S/N = 1 and S/N = 5 on the plot
            plt.annotate('S/N = 1', xy=(6, 1), xytext=(-10, 5), textcoords='offset points',
                        ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
            plt.annotate('S/N = 5', xy=(6, 5), xytext=(-10, 5), textcoords='offset points',
                        ha='right', va='bottom', color='gray', fontsize=10, fontweight='bold')
            #plt.yscale('log')
            plt.xlim([4, 18])
            plt.ylabel('S/N per wavelength bin')
            plt.xlabel('Wavelength (um)')

            ax = plt.gca()
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(param_name + " (" + param_units_string + ")")
            # Keep a concise axes title and add two figure-level columns
            plt.title(format_plot_title("S/N", self.config))
            fig3 = plt.gcf()
            fig3.text(0.02, 0.98, "\n".join(instrumental_lines), ha='left', va='top')
            fig3.text(0.52, 0.98, "\n".join(astrophysical_lines), ha='left', va='top')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            file_name_plot = str(self.config['dirs']['save_s2n_data_unique_dir']) + f"1d_s2n_vs_wavelength_and_dark_current_per_wavelength_bin.png"
            plt.subplots_adjust(top=0.6,right=0.8)
            #plt.show()
            plt.savefig(file_name_plot)
            logger.info(f"Wrote plot {file_name_plot}")


        return s2n