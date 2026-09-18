# Displays S/N degradation as function of DC 

import sys
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import ipdb


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from load_s2n_cube import load_s2n_cube, merge_s2n_cubes, print_cube_statistics

# Path to S/N cubes HDF5 written by save_s2n_cube() in calculator.py.
# Each file is one QE slice; load them in a loop and merge along QE.
s2n_hdf5_dir = Path(
    "/Users/eckhartspalding/Documents/git.repos/life_detectors/data/20260730_s2n_cube_sweep/"
)
s2n_hdf5_files = sorted(s2n_hdf5_dir.glob("*.hdf5"))
if not s2n_hdf5_files:
    raise FileNotFoundError(f"No HDF5 cubes found in {s2n_hdf5_dir}")

cubes = []
for s2n_hdf5_path in s2n_hdf5_files:
    print(f"Loading {s2n_hdf5_path.name}")
    cubes.append(load_s2n_cube(s2n_hdf5_path))
cube = merge_s2n_cubes(cubes)
print_cube_statistics(cube)

# Primary array: shape (wavelength, DC, QE)
snr_cube = cube.snr
wavelength = cube.wavelength
dark_current = cube.dark_current
qe = cube.qe

# S/N vs wavelength for one QE, varying DC
qe_choice = 0.4
qe_idx = int(np.argmin(np.abs(qe - qe_choice)))
qe_choice = float(qe[qe_idx])
colors = rcParams["axes.prop_cycle"].by_key()["color"]
color_idx = -1
current_color = colors[0]

plt.figure(figsize=(10, 5))
for i_dc, dc_val in enumerate(dark_current):
    
    # advance color only on 0.05 DC steps; reuse for values in between
    if abs(dc_val / 0.1 - round(dc_val / 0.1)) < 1e-8 and dc_val < 0.501:
        label_this = f"{dc_val:.2f} e/pix/s"
        color_idx += 1
        current_color = colors[color_idx % len(colors)]
        plt.stairs(
            snr_cube[:, i_dc, qe_idx],
            edges=cube.wavel_bin_edges,
            label=label_this,
            linewidth=4,
            alpha=1,
            color=current_color,
        )
    '''
    elif dc_val % 0.01 == 0:
        plt.stairs(
            snr_cube[:, i_dc, qe_idx],
            edges=cube.wavel_bin_edges,
            linewidth=1,
            alpha=0.3,
            color=current_color,
        )
    '''
plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.xlabel("Wavelength (um)", fontsize=18)
plt.ylabel("S/N", fontsize=18)
plt.title('S/N for different DCs', pad=20, fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(4, 18.5)
plt.ylim(0, 11)
plt.xlabel("Wavelength (um)")
plt.ylabel("S/N")
plt.suptitle(cube.base_titles[0, qe_idx])
plt.title('S/N for different DCs', pad=20)
ax = plt.gca()
ax.text(
    0.98,
    0.98,
    f"QE = {qe_choice:.1f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=18,
)
plt.legend(fontsize=16)
#plt.show()
#plt.savefig('/Users/eckhartspalding/Downloads/junk_s2n_vs_dc.pdf')
file_name = f'/Users/eckhartspalding/Downloads/junk_s2n_vs_dc_qe_{qe_choice:.2f}.png'
plt.savefig(file_name, bbox_inches='tight', dpi=300)
print(f"Saved figure to {file_name}")