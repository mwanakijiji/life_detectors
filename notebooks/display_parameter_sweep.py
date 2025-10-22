import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import sys
import time
import ipdb

dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/parameter_sweep/20251022_spectral_width_2_smallest/'
output_dir = '/Users/eckhartspalding/Downloads/'

# read in all the FITS files in the directory, sort them by filename, and put the data into a cube
fits_files = sorted([f for f in os.listdir(dir_sample_data) if f.lower().endswith('.fits')])

# List to hold the data arrays
data_list = []
n_int_array = []

for fname in fits_files:
    fpath = os.path.join(dir_sample_data, fname)
    with fits.open(fpath) as hdul:
        # take S/N data (wavelength and dark current is in the other slices)
        data = hdul[0].data[0,:,:]
        data_list.append(data)

        # get number of integrations by parsing the filename
        n_int = int(fname.split('_')[-1].split('.')[0].split('n')[1])
        n_int_array.append(n_int)


'''
# make a copy that sets S/N<5 to nan
data_cube_nan = np.where(data_cube < 5, np.nan, data_cube)

# Save the 3D data_cubes to FITS
output_fits_path = os.path.join(output_dir, 'data_cube.fits')
hdu = fits.PrimaryHDU(data_cube)
hdulist = fits.HDUList([hdu])
hdulist.writeto(output_fits_path, overwrite=True)
print(f"Saved FITS data cube to: {output_fits_path}")

output_fits_path = os.path.join(output_dir, 'data_cube_nan.fits')
hdu = fits.PrimaryHDU(data_cube_nan)
hdulist = fits.HDUList([hdu])
hdulist.writeto(output_fits_path, overwrite=True)
print(f"Saved nan FITS data cube to: {output_fits_path}")
'''

# plot 1: For a given integration time, what DC is necessary for S/N>5 at lambda > 6 um? 8 um?

# read in FITS data cube
#with fits.open(os.path.join(dir_sample_data, fits_files[0])) as hdul:

def dc_from_s2n_and_lambda(s2n_sample_slice, s2n_cube, n_int_array, n_int_desired, s2n_threshold: float = 5, wavel_min: float = 6):
    '''
    INPUTS:
    s2n_sample_slice: single cube with one slice of S/N values (as written out by pipeline) for a single integration time, with addl slices to indicate wavelengths and dark currents
        [0]: S/N values
        [1]: wavelength bin centers
        [2]: wavelength bin widths
        [3]: dark current values
    s2n_cube: 3D array of S/N values for all integration times (note this cube does not have slices for wavelengths and dark currents)
        [0 axis]: corresponds to n_int
    n_int_array: 1D array of numbers of integrations corresponding to slices of the s2n_cube
    n_int_desired: int, number of integrations we want to know about
    s2n_threshold: float, threshold for S/N
    wavel_min: minimum wavelength for which we need S/N of 5 

    RETURNS:
    dc_max (float): maximum dark current that can be used to achieve S/N>5 at lambda > wavel_min
    s2n_desired_int (2D array): S/N FITS cube for checking
    '''

    # Check if array lengths matche the corresponding axes in the s2n_cube
    if len(n_int_array) != s2n_cube.shape[0]:
        raise ValueError(f"Length of n_int_array ({len(n_int_array)}) does not match the number of integrations axis of s2n_cube ({s2n_cube.shape[0]})")
    # extract the wavelengths from the s2n_sample_slice
    wavel_array = s2n_sample_slice[1,0,:]
    print(f"wavel_array: {wavel_array}")
    print(len(wavel_array))

    # find the slice that corresponds to the integration time (to be precise, the number of integrations)
    n_int_slice = np.argmin(np.abs(np.array(n_int_array) - n_int_desired))
    #print(f"n_int_slice: {n_int_slice}")
    #print(f"n_int_array: {n_int_array}")

    # get the S/N values for the chosen integration time
    # (i.e., grab just one slice from the cube)
    s2n_desired_int = np.zeros((s2n_sample_slice.shape))
    s2n_desired_int[0,:,:] = s2n_cube[n_int_slice,:,:]
    # past on the slices indicating wavelengths and dark current
    s2n_desired_int[1,:,:] = s2n_sample_slice[1,:,:]
    s2n_desired_int[2,:,:] = s2n_sample_slice[2,:,:]
    s2n_desired_int[3,:,:] = s2n_sample_slice[3,:,:]

    print('s2n_cube.shape: ', s2n_cube.shape)
    print('s2n_desired_int.shape: ', s2n_desired_int.shape)
    print('s2n_sample_slice[1].shape: ', s2n_sample_slice[1].shape)
    print('s2n_sample_slice[2].shape: ', s2n_sample_slice[2].shape)
    print('s2n_sample_slice[3].shape: ', s2n_sample_slice[3].shape)
    
    # for what region of this slice is S/N > 5 at min wavelength and up?

    # set region of s2n plot at less than desired wavelength to be nans
    mask = s2n_sample_slice[1,:,:] < wavel_min
    #s2n_sample_slice[0,:,:][mask] = np.nan

    print('shape,', s2n_desired_int.shape)

    # loop over the rows of the S/N cube corresponding to DC, working our way from low DC to high
    # until there is a point when a region within the ROI of the S/N space is no longer has S/N > 5
    for idx_dc in range(s2n_desired_int.shape[1]):

        for i in range(0,4):
            s2n_desired_int[i,:,:][mask] = np.nan

        print(s2n_desired_int.shape[1])
        print('checking idx_dc: ', idx_dc)

        # find the minimum S/N value in the ROI
        idx_dc_max = None
        s2n_min = np.nanmin(s2n_desired_int[0,idx_dc,:])

        # plot the ROI
        '''
        plt.clf()
        dummy = np.copy(s2n_desired_int)
        dummy[0,idx_dc,:] = 20
        plt.imshow(dummy[0,:,:], origin='lower')
        plt.show()
        '''

        print('s2n_min: ', s2n_min)
        if s2n_min > s2n_threshold:
            idx_dc_max = idx_dc

            roi = s2n_desired_int[0,idx_dc,:]

            # sanity check: take median and min of the DC and make sure they're the same


            dc_max = s2n_desired_int[3,idx_dc_max,:]

            print('median: ', np.nanmedian(dc_max))
            print('min: ', np.nanmin(dc_max))
            ipdb.set_trace()
            if np.round(np.nanmedian(roi), 4) != np.round(np.nanmin(roi), 4):
                print('! ---- median and mean of the DC in the ROI are not the same ---- !')


        else:
            # if the min S/N < 5, break out of the loop; this effectively keeps the last row of of the DC array
            break



    # find the maximum S/N value in the desired region
    return dc_max, s2n_desired_int





# read in the data
s2n_cube_file_name = '/Users/eckhartspalding/Downloads/data_cube.fits'
with fits.open(os.path.join(dir_sample_data, fits_files[0])) as hdul:
    s2n_sample_slice = hdul[0].data
with fits.open(s2n_cube_file_name) as hdul:
    s2n_cube = hdul[0].data


# for given S/N and wavelength range, what max DC do I need?
test_dc_max, test_s2n_desired_int = dc_from_s2n_and_lambda(s2n_sample_slice=s2n_sample_slice, 
                            s2n_cube=s2n_cube, 
                            n_int_array=n_int_array, 
                            n_int_desired=25920, 
                            s2n_threshold=2,
                            wavel_min=8.)

print('test_dc_max: ', test_dc_max)

# FYI
output_fits_path = os.path.join(output_dir, 'test_s2n_desired_int.fits')
hdu = fits.PrimaryHDU(test_s2n_desired_int)
hdulist = fits.HDUList([hdu])
hdulist.writeto(output_fits_path, overwrite=True)
print(f"Saved FITS file of test_s2n_desired_int to: {output_fits_path}")


ipdb.set_trace()