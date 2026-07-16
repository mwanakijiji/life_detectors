# Displays S/N degradation as function of DC 

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from load_s2n_cube import format_cube_plot_title, load_s2n_cube, print_cube_statistics

# Path to S/N cubes HDF5 written by save_s2n_cube() in calculator.py
#s2n_hdf5_path = '/Users/eckhartspalding/Documents/git.repos/life_detectors/hdf5_testing/temp_s2n_sweep_planet_index_0000000_Nuniverse_1_Nstar_1_dist_10_Rp_1_Rs_1_Ts_5778_L_1.0_z_3_eclip_lon_135_eclip_lat_45_Stype_G/dc_5_qe_0.90_s2n_cube.hdf5'
s2n_hdf5_path = '/Users/eckhartspalding/Downloads/large_sweep_test/qe_0.05_s2n_cube.hdf5'

cube = load_s2n_cube(s2n_hdf5_path)
print_cube_statistics(cube)

# Primary array: shape (wavelength, DC, QE)
snr_cube = cube.snr
wavelength = cube.wavelength
dark_current = cube.dark_current
qe_list = cube.qe

# plot the maximum DC for a given S/N = N at a given wavelength bin
# (wavelength, DC, QE)

s2n_min = 3.
qe_idx = 3
qe_choice = qe_list[qe_idx]
snr_2d_at_qe = snr_cube[:, :, qe_idx]          # shape (n_wavel, n_dc)
good = snr_2d_at_qe > s2n_min
# broadcast DC across wavelength rows
dc_grid = dark_current[np.newaxis, :]
# at each wavelength: max DC where S/N is good; else nan
max_dc_1d_at_qe_good_s2n = np.nanmax(
    np.where(good, dc_grid, np.nan),
    axis=1,
)


plt.clf()
plt.figure(figsize=(10, 5))
plt.stairs(
    max_dc_1d_at_qe_good_s2n,
    edges=cube.wavel_bin_edges
    )
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.axvline(x=11.2, color='red', linestyle='--')
plt.xlim(4, 18.5)
#plt.ylim(0, 0.2)
plt.xlabel("Wavelength (um)")
plt.ylabel("DC (e/pix/s)")
plt.title(
    format_cube_plot_title(
        cube,
        f"Max DC for S/N > {s2n_min:g} (QE = {qe_choice:.1f})",
    ),
    loc="left",
    pad=20,
    fontsize=10,
)
#plt.legend()
#plt.show()
plt.savefig(
    f"/Users/eckhartspalding/Downloads/junk_max_dc_vs_wavelength_{qe_choice:.2f}.pdf",
    bbox_inches="tight",
    pad_inches=0.5,
)