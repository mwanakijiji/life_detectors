"""HDF5 recording helpers."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

from .keys import (
    canonical_angle_deg,
    canonical_dc_rate,
    canonical_qe,
    format_angle_qe_hdf5_name,
)

logger = logging.getLogger(__name__)

def record_info_at_angle_and_qe(
    *,
    angle_deg: float,
    qe: float,
    output_channels: dict,
    post_chop_tables_by_dark_current: dict,
    save_dir: str,
    plot: bool = True,
) -> dict:
    """
    Save one HDF5 file for this rotation angle and QE

    File structure::

        angle_{angle}.hdf5
          dc_{dc}/
            output_1_bright   QTable, all columns from tables_by_dark_current
            output_2_bright
            output_3_dark
            output_4_dark
            chopped           QTable, all columns from post_chop_tables_by_dark_current
    """

    file_name_hdf5 = f"{save_dir}{format_angle_qe_hdf5_name(angle_deg, qe)}"
    angle_deg = canonical_angle_deg(angle_deg)
    qe = canonical_qe(qe)
    hdf5_path = Path(file_name_hdf5)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if hdf5_path.exists():
        hdf5_path.unlink()
    hdf5_paths = []
    first_dataset = True

    for dc_rate in post_chop_tables_by_dark_current:
        dc_rate_key = canonical_dc_rate(dc_rate)
        dc_group = f"dc_{dc_rate_key:06.3f}"
        qe_group = f"qe_{qe:04.2f}"
        dc_qe_str = f"{dc_group}_{qe_group}"

        # write out the tables for each output channel
        for ch_name, ch in output_channels.items():
            out_tbl = ch.tables_by_dark_current[dc_rate_key].copy()
            out_tbl.meta['angle_deg'] = float(angle_deg)
            out_tbl.meta['dark_current_e_pix_s'] = float(dc_rate_key)
            out_tbl.meta['qe'] = float(qe)
            hdf5_path = dc_qe_str + f"/{ch_name}"
            if first_dataset:
                out_tbl.write(
                    file_name_hdf5,
                    path=hdf5_path,
                    serialize_meta=True,
                    overwrite=True,
                )
                first_dataset = False
            else:
                out_tbl.write(
                    file_name_hdf5,
                    path=hdf5_path,
                    serialize_meta=True,
                    append=True,
                )
            hdf5_paths.append(hdf5_path)
            logger.info(f"Wrote {file_name_hdf5}:{hdf5_path}")

        # now include the chopped signal
        chopped_tbl = post_chop_tables_by_dark_current[dc_rate_key].copy()
        chopped_tbl.meta['angle_deg'] = float(angle_deg)
        chopped_tbl.meta['dark_current_e_pix_s'] = float(dc_rate_key)
        hdf5_path = dc_qe_str + "/chopped"
        chopped_tbl.write(
            file_name_hdf5,
            path=hdf5_path,
            serialize_meta=True,
            append=True,
        )
        hdf5_paths.append(hdf5_path)
        logger.info(f"Wrote {file_name_hdf5}:{hdf5_path}")

    snapshot = {
        'angle_deg': angle_deg,
        'hdf5_file': file_name_hdf5,
        'hdf5_paths': hdf5_paths,
        'dark_currents': list(post_chop_tables_by_dark_current.keys()),
    }

    logger.info(f"Recorded angle {angle_deg} to {file_name_hdf5}")

    if plot and post_chop_tables_by_dark_current: # pragma: no cover
        dc_rate = next(iter(post_chop_tables_by_dark_current))
        chopped_tbl = post_chop_tables_by_dark_current[dc_rate]
        wavel_center = chopped_tbl['center'].value
        wavel_width = chopped_tbl['width'].value
        wavel_edges = np.empty(len(wavel_center) + 1)
        wavel_edges[1:-1] = 0.5 * (wavel_center[:-1] + wavel_center[1:])
        wavel_edges[0] = wavel_center[0] - 0.5 * wavel_width[0]
        wavel_edges[-1] = wavel_center[-1] + 0.5 * wavel_width[-1]

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for col_name in chopped_tbl.colnames:
            if not col_name.startswith('chopped_'):
                continue
            y_vals = chopped_tbl[col_name].value
            ax.stairs(y_vals, edges=wavel_edges, label=col_name)

        ax.set_yscale('log')
        ax.set_xlim(4.0, 18.0)
        ax.set_xlabel('Wavelength (um)')
        ax.set_ylabel('Signal (ADU)')
        ax.set_title(f'Chopped signals at angle {angle_deg:06.2f}, dc={dc_rate:06.3f} e/pix/s, QE {qe:04.2f}')
        ax.legend(fontsize=8, loc='best')
        file_name_plot = f"{save_dir}chopped_dark_output_signals_at_angle_{angle_deg:06.2f}_qe_{qe:04.2f}.png"
        fig.savefig(file_name_plot)
        plt.close(fig)
        logger.info(f"Saved plot of chopped signals at angle {angle_deg:06.2f}, QE {qe:04.2f}: {file_name_plot}")

    return
