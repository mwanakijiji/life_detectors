"""Combined astrophysical + instrumental table plots."""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import astropy.units as u

logger = logging.getLogger(__name__)


def plot_debug_final_table(
    qt,
    bin_edges,
    output_name: str,
    dc_rate: float,
    sources_to_include: Sequence[str],
    out_path: str,
) -> None:
    """Stairs plot of instrumental noise and astro ADU columns from a final table."""
    wavel_bin_center = qt["center"]
    edges = bin_edges.value if hasattr(bin_edges, "value") else bin_edges
    fig, ax = plt.subplots(figsize=(10, 5))
    debug_cols = ["instrum_dc_rms_adu", "instrum_rn_rms_adu"]
    y_unit = u.adu
    for col_name in debug_cols:
        y_col = qt[col_name]
        y_vals = y_col.value if hasattr(y_col, "value") else y_col
        ax.stairs(y_vals, edges=edges, label=col_name)
        if hasattr(y_col, "unit"):
            y_unit = y_col.unit
    for source_name in sources_to_include:
        col_name = f"astro_{source_name}_adu"
        if col_name in qt.colnames:
            y_col = qt[col_name]
            y_vals = y_col.value if hasattr(y_col, "value") else y_col
            ax.stairs(y_vals, edges=edges, label=col_name)
            if hasattr(y_col, "unit"):
                y_unit = y_col.unit
    ax.set_xlim(4.0, 18.5)
    ax.set_title(f"Debug final_table: {output_name}, dc={dc_rate:.3f} e/pix/s")
    ax.set_xlabel(f"Wavelength bin center ({wavel_bin_center.unit})")
    ax.set_ylabel(f"Flux ({y_unit})")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info(
        "Saved plot of binned fluxes from output %s at dark current %.3f e/pix/s to %s",
        output_name,
        dc_rate,
        out_path,
    )
