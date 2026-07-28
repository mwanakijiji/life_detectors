"""HDF5 slot reading and SNR cube assembly from pipeline outputs."""

import glob
import logging
import os
from typing import Dict, Optional, Tuple

import astropy.units as u
import h5py
import numpy as np
from astropy.table import QTable

from ...utils.helpers.formatting import (
    build_astrophysical_sources_to_use_title,
    ensure_plot_title_context,
)
from ...utils.helpers.keys import (
    DC_GROUP_DECIMALS,
    QE_GROUP_DECIMALS,
    canonical_angle_deg,
    canonical_dc_rate,
    canonical_qe,
    hdf5_path_matches_qe,
    parse_angle_from_hdf5_path,
    parse_dc_qe_group,
    resolve_float_key,
)
from ...utils.loader import config_getboolean
from ...viz.snr import plot_snr_vs_wavelength
from .s2n_cube import S2NCube, _config_to_dict, save_s2n_cube

logger = logging.getLogger(__name__)


def _parse_dc_qe_group(dc_qe_str: str) -> Tuple[float, float]:
    return parse_dc_qe_group(dc_qe_str)


def read_hdf5_slots(read_dir: str, *, qe: Optional[float] = None) -> Dict[str, dict]:
    """Read angle_*.hdf5 files and aggregate tables by dc/qe group.

    When ``qe`` is set, only files named ``angle_*_qe_{qe}.hdf5`` are read
    (legacy ``angle_*.hdf5`` names without a QE suffix are still accepted, but
    only groups matching ``qe`` are loaded).
    """
    hdf5_files = sorted(glob.glob(os.path.join(read_dir, "angle_*.hdf5")))
    if qe is not None:
        qe = canonical_qe(qe)
        hdf5_files = [f for f in hdf5_files if hdf5_path_matches_qe(f, qe)]
    by_dc_qe: Dict[str, dict] = {}

    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            for dc_qe_str in f.keys():
                if dc_qe_str.startswith("__"):
                    continue
                if qe is not None:
                    _, group_qe = _parse_dc_qe_group(dc_qe_str)
                    if canonical_qe(group_qe) != qe:
                        continue

                chopped = QTable.read(hdf5_file, path=f"{dc_qe_str}/chopped")
                out3 = QTable.read(hdf5_file, path=f"{dc_qe_str}/output_3_dark")
                meta_angle = chopped.meta.get("angle_deg")
                if meta_angle is not None:
                    angle = canonical_angle_deg(meta_angle)
                else:
                    angle = parse_angle_from_hdf5_path(hdf5_file)
                S_p = chopped["chopped_astro_exoplanet_model_10pc_adu"]
                S_p_3 = out3["astro_exoplanet_model_10pc_adu"]

                slot = by_dc_qe.setdefault(
                    dc_qe_str,
                    {
                        "wavel": chopped["center"].value,
                        "wavel_bin_edges": chopped.meta["wavel_bin_edges"],
                        "S_p": {},
                        "S_p_3": {},
                        "chopped_instrum_dc_rms_adu": {},
                        "chopped_instrum_rn_rms_adu": {},
                    },
                )
                slot["S_p"][angle] = S_p
                slot["S_p_3"][angle] = S_p_3
                slot["chopped_instrum_rn_rms_adu"][angle] = chopped["chopped_instrum_rn_rms_adu"]
                slot["chopped_instrum_dc_rms_adu"][angle] = chopped["chopped_instrum_dc_rms_adu"]
                slot["wavel_bin_width"] = chopped["width"]

                sym_tags = ("star", "exozodiacal", "zodiacal")
                slot.setdefault("sources_sym", {})
                for source_name in sym_tags:
                    col = f"astro_{source_name}_adu"
                    if col not in out3.colnames:
                        continue
                    slot["sources_sym"].setdefault(source_name, {"Ssym_dark_3": {}})
                    slot["sources_sym"][source_name]["Ssym_dark_3"][angle] = chopped[
                        f"output_3_dark_{col}"
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
                slot["chopped_instrum_dc_rms_adu"][ref_angle][wavel_bin_num] * gain,
                2,
            ).value
            * u.electron
        )
        S_read_noise_var = (
            np.power(
                slot["chopped_instrum_rn_rms_adu"][ref_angle][wavel_bin_num] * gain,
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
    qe = canonical_qe(float(config["detector"]["quantum_efficiency"]))
    by_dc_qe = read_hdf5_slots(read_dir, qe=qe)
    if not by_dc_qe:
        raise FileNotFoundError(
            f"No angle_*.hdf5 files for QE {qe:04.2f} found in {read_dir}"
        )

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

    dark_current = np.array(sorted({canonical_dc_rate(v) for v in dc_values}))
    qe = np.array(sorted({canonical_qe(v) for v in qe_values}))
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
        i_dc = dc_index[resolve_float_key(dc_index, dc_val, label="dark_current", decimals=DC_GROUP_DECIMALS)]
        i_qe = qe_index[resolve_float_key(qe_index, qe_val, label="qe", decimals=QE_GROUP_DECIMALS)]
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


def calculate_s2n_post_rotation(read_dir, config, *, save_cube_path_stem: Optional[str] = None):
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

    plot = config_getboolean(config, "tasks", "print_fyi_plots", default=False)

    for i_dc, dc_val in enumerate(cube.dark_current):
        for i_qe, qe_val in enumerate(cube.qe):
            dc_qe_str = f"dc_{dc_val:06.3f}_qe_{qe_val:04.2f}"
            snr_lambda_array = cube.snr[:, i_dc, i_qe]
            snr_tot = cube.snr_tot[i_dc, i_qe]
            print(f"SNR_tot for DC {dc_qe_str}: {snr_tot}")

            if plot:  # pragma: no cover
                file_name_plot = (
                    str(config["dirs"]["save_s2n_data_unique_dir"])
                    + f"SNR_vs_wavelength_{dc_qe_str}"
                    + f"_Nang_{cube.n_angles}_Nintpa_{cube.n_int_per_angle}_Ninttot_{cube.n_int_total}.png"
                )
                plot_snr_vs_wavelength(
                    snr_lambda_array,
                    cube.wavel_bin_edges,
                    str(cube.base_titles[i_dc, i_qe]),
                    config,
                    file_name_plot,
                )

    if save_cube_path_stem is not None:
        qe_val = float(config["detector"]["quantum_efficiency"])
        file_name_s2n_cube = f"{save_cube_path_stem}/qe_{qe_val:04.2f}_s2n_cube.hdf5"
        save_s2n_cube(cube, output_path=file_name_s2n_cube, file_format="both")

    return cube
