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

# QUESTION 1: For a given integration time, what DC is necessary for S/N>5 at lambda > 6 um? 8 um?
def dc_from_n_int_s2n_lambda(s2n_sample_slice, s2n_cube, n_int_array, n_int_desired, s2n_threshold: float = 5, wavel_min: float = 6):
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
    n_int_this = n_int_array[n_int_slice]
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
    s2n_desired_int_copy = np.copy(s2n_desired_int) # make copy, since we don't want nans in the final file written out
    for idx_dc in range(s2n_desired_int.shape[1]):

        for i in range(0,4):
            s2n_desired_int_copy[i,:,:][mask] = np.nan

        print(s2n_desired_int_copy.shape[1])
        print('checking idx_dc: ', idx_dc)

        # find the minimum S/N value in the ROI
        idx_dc_max = None
        s2n_min = np.nanmin(s2n_desired_int_copy[0,idx_dc,:])

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

            roi = s2n_desired_int_copy[0,idx_dc,:]

            # sanity check: take median and min of the DC and make sure they're the same


            dc_max = s2n_desired_int_copy[3,idx_dc_max,:]

            print('median: ', np.nanmedian(dc_max))
            print('min: ', np.nanmin(dc_max))
            if np.round(np.nanmedian(roi), 4) != np.round(np.nanmin(roi), 4):
                print('! ---- median and min of the DC in the ROI are not the same ---- !')


        else:
            # if the min S/N < N, break out of the loop; this effectively keeps the last row of of the DC array
            break

    # find the maximum S/N value in the desired region
    return dc_max, s2n_desired_int, n_int_this


# QUESTION 2: For a given DC, what integration time is necessary for S/N>5 at lambda > 6 um? 8 um?
def n_int_from_dc_s2n_lambda(s2n_sample_slice, s2n_cube, n_int_array, dc_desired: float, s2n_threshold: float = 5, wavel_min: float = 6):
    '''
    INPUTS:
    s2n_sample_slice: single cube with one slice of S/N values (as written out by pipeline) for a single integration time, with addl slices to indicate wavelengths and dark currents
        [0]: S/N values
        [1]: wavelength bin centers
        [2]: wavelength bin widths
        [3]: dark current values
    s2n_cube: 3D array of S/N values for all integration times (note this cube does not have slices for wavelengths and dark currents)
        [all slices]: S/N values     
        [height axis]: corresponds to integration times
    n_int_array: 1D array of numbers of integrations corresponding to slices of the s2n_cube
    dc_desired: float, dark current we are interested in
    s2n_threshold: float, threshold for S/N
    wavel_min: minimum wavelength for which we need S/N of 5 

    RETURNS:
    cube_s2n_nint_wavel: 3D array of acceptable S/N values, number of integrations, and wavelength bin centers
    n_int_this: int, number of integrations that achieves S/N>N in any wavelength bin at lambda > wavel_min
    t_prime_t_ratio: float, ratio of integration times with and without systematics
    s2n_prime_s2n_ratio: float, ratio of (S/N)'/(S/N) for the same integration time, where (S/N)' is with systematics and (S/N) is without
    '''


    # Check if array lengths match the corresponding axes in the s2n_cube
    if len(n_int_array) != s2n_cube.shape[0]:
        raise ValueError(f"Length of n_int_array ({len(n_int_array)}) does not match the number of integrations axis of s2n_cube ({s2n_cube.shape[0]})")
    # extract the DCs from the s2n_sample_slice
    dc_array = s2n_sample_slice[3,:,0]

    # find the slice that corresponds to the DC
    dc_slice_idx = np.argmin(np.abs(np.array(dc_array) - dc_desired))
    s2n_cube_dc_slice = s2n_cube[:,dc_slice_idx,:] # S/N array with dims (y, x)=(DC, wavelength)

    # make arrays denoting n_int and wavelength
    n_int_slice = np.tile(n_int_array, (s2n_cube_dc_slice.shape[1], 1)).T
    wavel_slice = np.tile(s2n_sample_slice[1,0,:], (s2n_cube_dc_slice.shape[0], 1))

    # pack everything into a single cube with slices
    # [0]: S/N values
    # [1]: number of integrations
    # [2]: wavelength bin centers
    cube_s2n_nint_wavel = np.concatenate((s2n_cube_dc_slice[None, ...], n_int_slice[None, ...], wavel_slice[None, ...]), axis=0)

    # now mask out the regions below the S/N threshold AND the cutoff wavelength
    mask_s2n = cube_s2n_nint_wavel[0,:,:] < s2n_threshold
    mask_wavel = cube_s2n_nint_wavel[2,:,:] < wavel_min
    mask_added = mask_wavel.astype(int) + mask_s2n.astype(int)
    mask_combined = (mask_added == 0).astype(bool)
    cube_s2n_nint_wavel[0,:,:][~mask_combined] = np.nan

    # loop through the rows of the slice of n_int and find the first row with non-nan values, and keep the index n_int_idx
    # (note that this does not necessarily indicate that ALL of the S/N bins are >5; it just means that at least one is) 
    for n_int_idx in range(0,cube_s2n_nint_wavel.shape[1]):
        if np.any(~np.isnan(cube_s2n_nint_wavel[0,n_int_idx,:])):
            break

    n_int_this = n_int_array[n_int_idx]

    # the S/N for the spectrum corresponding to this DC and integration time
    s2n_baseline = s2n_cube[n_int_idx,dc_slice_idx,:]

    #########################################################
    ## find ratio t_prime/t, where t_prime is with systematics and t is without
    # first get the slice of S/N corresponding to DC=0, with dims (n_int, wavelength)
    # recall s2n_cube dims are (n_int, DC, wavelength)
    s2n_dc_zero = s2n_cube[:,0,:]
    # now find the index of this slice that shows S/N as function of wavelength that is most similar to the S/N for the given non-zero DC and integration time?
    loss_array = np.sum(np.power(s2n_dc_zero - s2n_baseline, 2), axis=1)
    n_int_similar_zero_idx = np.argmin(loss_array) # at what n_int is the loss minimized?
    n_int_similar_zero_dc = n_int_array[n_int_similar_zero_idx]
    s2n_similar_zero_dc = s2n_cube[n_int_similar_zero_idx,0,:]
    t_prime_t_ratio = n_int_this / n_int_similar_zero_dc

    plt.clf()
    plt.plot(wavel_slice[0,:], s2n_similar_zero_dc, label='DC=0')
    plt.plot(wavel_slice[0,:], s2n_baseline, label='DC='+str(dc_desired))
    # Annotate the plot with the t_prime/t ratio
    plt.annotate(f't\'/t = {t_prime_t_ratio:.3f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 fontsize=11, 
                 color='black', 
                 verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))
    plt.xlabel('Wavelength (um)')
    plt.ylabel('S/N')
    plt.title('S/N as function of wavelength for DC=0 and DC='+str(dc_desired))
    plt.legend()
    file_name_plot = os.path.join('/Users/eckhartspalding/Downloads/junk_s2n_similar_zero_dc.png')
    plt.savefig(file_name_plot)
    print(f"Saved plot of S/N as function of wavelength for DC=0 and DC={dc_desired} to {file_name_plot}")

    #########################################################
    ## find (S/N)'/(S/N) for the same integration time, where (S/N)' is with systematics and (S/N) is without
    s2n_same_int_no_dc = s2n_cube[n_int_idx,0,:]
    s2n_prime_s2n_ratio = s2n_baseline / s2n_same_int_no_dc

    # return the cube (with the S/N values, n_int, and wavelength); and the number of integrations n_int 
    return cube_s2n_nint_wavel, n_int_this, t_prime_t_ratio, s2n_prime_s2n_ratio



def main():

    # read in the data
    s2n_cube_file_name = '/Users/eckhartspalding/Downloads/data_cube.fits'
    with fits.open(os.path.join(dir_sample_data, fits_files[0])) as hdul:
        s2n_sample_slice = hdul[0].data
        #header = hdul[0].header.copy()
    with fits.open(s2n_cube_file_name) as hdul:
        s2n_cube = hdul[0].data


    # for given S/N and wavelength range, what max DC do I need?
    '''
    test_dc_max, test_s2n_desired_int, n_int_this = dc_from_n_int_s2n_lambda(s2n_sample_slice=s2n_sample_slice, 
                                s2n_cube=s2n_cube, 
                                n_int_array=n_int_array, 
                                n_int_desired=25920, 
                                s2n_threshold=2,
                                wavel_min=8.)

    print('test_dc_max: ', test_dc_max)
    '''
    
    test = n_int_from_dc_s2n_lambda(s2n_sample_slice, s2n_cube, n_int_array, dc_desired=25, s2n_threshold=15, wavel_min=5)

    # FYI
    output_fits_path = os.path.join(output_dir, 'test_s2n_desired_int.fits')

    header = fits.Header()
    header['N_INT'] = n_int_this

    hdu = fits.PrimaryHDU(test_s2n_desired_int, header=header)
    hdulist = fits.HDUList([hdu])
    # Update the header with N_INT value

    hdulist.writeto(output_fits_path, overwrite=True)
    print(f"Saved FITS file of test_s2n_desired_int to: {output_fits_path}")

    ipdb.set_trace()

if __name__ == '__main__':
    main()