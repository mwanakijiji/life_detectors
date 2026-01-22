import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import sys
import time
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

file_name_cube = '/Users/eckhartspalding/Documents/git.repos/life_detectors/parameter_sweep/junk/s2n_sweep_n00026020.fits'

# read in the FITS file
with fits.open(file_name_cube) as hdul:
    data_cube = hdul[0].data
    header = hdul[0].header
    # [0]: S/N values
    # [1]: wavelength bin centers
    # [2]: wavelength bin widths
    # [3]: dark current values

# print the FITS header, one line per keyword
for keyword in hdul[0].header.keys():
    print(f"{keyword}: {hdul[0].header[keyword]}")

plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'axes.labelsize': 22, 'axes.titlesize': 22})

#max_int_time = 5184000 # total integration time of the longest observation in the cube, in seconds
n_int_this = 100

# indices of choices of DC values to plot in comparison with zero DC
n_dc_1 = 60
n_dc_2 = 400
n_dc_3 = -1

# the actual DC value
dc_1 = np.median(data_cube[3,n_dc_1,:])
dc_2 = np.median(data_cube[3,n_dc_2,:])
dc_3 = np.median(data_cube[3,n_dc_3,:])

plt.figure(figsize=(8, 6))
plt.step(data_cube[1,0,:], data_cube[0,0,:], where='mid', color='black', linewidth=3, label='No dark current')
plt.step(data_cube[1,n_dc_1,:], data_cube[0,n_dc_1,:], where='mid', color='red', linewidth=3, label='DC '+str(np.round(dc_1, 1)) + ' e-/pix/s')
plt.step(data_cube[1,n_dc_2,:], data_cube[0,n_dc_2,:], where='mid', color='blue', linewidth=3, label='DC '+str(np.round(dc_2, 1)) + ' e-/pix/s')
plt.step(data_cube[1,n_dc_3,:], data_cube[0,n_dc_3,:], where='mid', color='green', linewidth=3, label='DC '+str(np.round(dc_3, 1)) + ' e-/pix/s')
plt.xlim([4.0, 18.])
plt.ylim([0, 5])
plt.ylabel('S/N')
#plt.yscale('log')
plt.xlabel('Wavelength (um)')
plt.legend()
file_name_s2n_plot = '/Users/eckhartspalding/Downloads/junk2_s2n_w_diff_dc.png'
plt.savefig(file_name_s2n_plot)
print(f"Saved S/N plot to {file_name_s2n_plot}")

# plot acceptable DC: for S/N of P, how much DC can we have?
s2n_acceptable = [1,2,2.5,3,3.5,4,4.5]

# loop down the rows of S/N (i.e., for descending dark current)
# as soon as a given column rises from below the value of s2n_acceptable to above it, 
# fill the corresponding element in an array (which has the length of the wavelength bins, i.e., x) with that DC value

plt.figure(figsize=(13, 8))

for s2n_acceptable_this in s2n_acceptable:
    # for each S/N level, find the acceptable DC

    resids_from_acceptable_s2n = np.subtract(data_cube[0,:,:],s2n_acceptable_this)

    # in each column, find the location of the minimum value
    # Find the index (row, i.e., DC value) for each column (wavelength) where the S/N residual from the target is minimal across the DC axis
    arg_min_resid_in_each_column = np.argmin(np.abs(resids_from_acceptable_s2n), axis=0)  # shape: (number of wavelength bins,)

    # find the DC corresponding to each argument
    acceptable_dc_at_each_wavelength = np.array([
        data_cube[3, dc_idx, col_idx]
        for col_idx, dc_idx in enumerate(arg_min_resid_in_each_column)
    ])

    #plt.title('N_INT = '+str(n_int_this))
    plt.step(data_cube[1,0,:], acceptable_dc_at_each_wavelength, where='mid', color='green', linewidth=2*float(s2n_acceptable_this), label='For S/N = '+str(s2n_acceptable_this), alpha=2 * float(s2n_acceptable_this) / 10)


# Plot thin dashed lines at each octave (10^0, 10^1, 10^2, etc.)
octaves = [10**i for i in range(-2, 2)]  # from 0.01 to 100 (within the ylim used)
for oct in octaves:
    plt.axhline(oct, color='k', linestyle='--', linewidth=1, alpha=0.5)

plt.xlim([4.0, 18.])
plt.ylim([1e-2, 70])
plt.ylabel('Max dark current (e-/pix/s)')
plt.yscale('log')
plt.xlabel('Wavelength (um)')
plt.legend()
plt.tight_layout()
file_name_acceptable_dc_plot = '/Users/eckhartspalding/Downloads/junk3_acceptable_dc_plot.png'
plt.savefig(file_name_acceptable_dc_plot)
print(f"Saved acceptable DC plot to {file_name_acceptable_dc_plot}")