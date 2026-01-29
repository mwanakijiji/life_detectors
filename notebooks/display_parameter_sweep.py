import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import sys
import time
import ipdb
import xarray as xr
import pickle
import plotting_3d


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
def n_int_from_dc_s2n_lambda(s2n_sample_slice, s2n_cube, n_int_array, dc_desired: float, s2n_threshold: float = 5, wavel_min: float = 6, plot: bool = True):
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
    plot: bool, whether to make FYI plots

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

    if plot:

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

        plt.clf()
        fig, ax1 = plt.subplots()
        
        # Plot the ratio on the left y-axis
        ax1.plot(wavel_slice[0,:], s2n_prime_s2n_ratio, 'b-', label='(S/N)\'/(S/N)')
        ax1.set_xlabel('Wavelength (um)')
        ax1.set_ylabel('(S/N)\'/(S/N)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create a second y-axis for the S/N values
        ax2 = ax1.twinx()
        ax2.plot(wavel_slice[0,:], s2n_baseline, 'r-', label='S/N with DC='+str(dc_desired))
        ax2.plot(wavel_slice[0,:], s2n_same_int_no_dc, 'r--', label='S/N with DC=0')
        ax2.set_ylabel('S/N', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('S/N ratio and individual S/N values for same integration time')
        file_name_plot = os.path.join('/Users/eckhartspalding/Downloads/junk_s2n_prime_s2n_ratio.png')
        plt.savefig(file_name_plot)
        print(f"Saved plot of (S/N)\'/(S/N) as function of wavelength for DC=0 and DC={dc_desired} to {file_name_plot}")

    # return the cube (with the S/N values, n_int, and wavelength); and the number of integrations n_int 
    return cube_s2n_nint_wavel, n_int_this, t_prime_t_ratio, s2n_prime_s2n_ratio



def main():

    st_type = 'M'
    if st_type == 'A':
        dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/param_sweeps/stellar_type_A/temp_s2n_sweep_planet_index_0000003_Nuniverse_83_Nstar_144_dist_15.0376_Rp_0.72647_Rs_1.81_Ts_7500_L_15.0_Stype_A/'
    elif st_type == 'F':
        dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/param_sweeps/stellar_type_F/temp_s2n_sweep_planet_index_0000006_Nuniverse_496_Nstar_157_dist_10.929_Rp_0.90768_Rs_1.18_Ts_6000_L_3.0_Stype_F'
    elif st_type == 'G':
        dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/param_sweeps/stellar_type_G/temp_s2n_sweep_planet_index_0000001_Nuniverse_460_Nstar_797_dist_12.8345_Rp_1.42795_Rs_0.909_Ts_5490_L_1.0_Stype_G/'
    elif st_type == 'K':
        dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/param_sweeps/stellar_type_K/temp_s2n_sweep_planet_index_0000008_Nuniverse_199_Nstar_864_dist_12.783_Rp_1.03336_Rs_0.72_Ts_4700_L_0.4_Stype_K/'
    elif st_type == 'M':
        dir_sample_data = '/Users/eckhartspalding/Documents/git.repos/life_detectors/param_sweeps/stellar_type_M/temp_s2n_sweep_planet_index_0000000_Nuniverse_245_Nstar_273_dist_12.912_Rp_1.31511_Rs_0.46_Ts_3650_L_0.05_Stype_M/'
    output_dir = '/Users/eckhartspalding/Downloads/'
    plot_save_string = 'type_'+st_type+'_star_'
    iso_line = 1.0

    # for informative titles on plots
    source_string = os.path.basename(os.path.normpath(dir_sample_data))
    # read in all the FITS files in the directory, sort them by filename, and put the data into a cube
    fits_files = sorted([f for f in os.listdir(dir_sample_data) if f.lower().endswith('.fits')])

    # List to hold the data arrays
    data_list = []
    n_int_list = []
    qe_list = []

    # read in the data and sweeped parameters from the FITS files
    for fname in fits_files:
        fpath = os.path.join(dir_sample_data, fname)
        with fits.open(fpath) as hdul:
            print('Reading in data from file: ', fpath)
            # hdul[0].data has shape (4, n_dc, n_wavel); the 4 slices are [0]: S/N values, [1]: wavelength bin centers, [2]: wavelength bin widths, [3]: dark current values
            data = hdul[0].data[0, :, :]   
            data_list.append(data)

            # get sweeped parameters from FITS header
            n_int = int(hdul[0].header['N_INT'])
            qe = float(hdul[0].header['QE'])

            n_int_list.append(n_int)
            qe_list.append(qe)

            # for the final cube axes
            dc_array = hdul[0].data[3, :, 0] 
            wavel_array = hdul[0].data[1, 0, :]


    # build sorted unique coordinate arrays and index maps
    n_int_vals = np.array(sorted(set(n_int_list)))
    qe_vals    = np.array(sorted(set(qe_list)))
    Nn = len(n_int_vals)
    Nq = len(qe_vals)
    
    # maps: value -> index
    n_int_index = {v: i for i, v in enumerate(n_int_vals)}
    qe_index    = {v: j for j, v in enumerate(qe_vals)}
    n_dc, n_wavel = data_list[0].shape
    cube = np.zeros((Nn, Nq, n_dc, n_wavel), dtype=float)


    for data, n_int, qe in zip(data_list, n_int_list, qe_list):
        i = n_int_index[n_int]
        j = qe_index[qe]
        cube[i, j, :, :] = data

    s2n = xr.DataArray(
        cube,
        dims=("n_int", "qe", "dc", "wavel"),
        coords={
            "n_int": n_int_vals,
            "qe": qe_vals,
            "dc": dc_array,
            "wavel": wavel_array,
        },
        name="s2n"
    )

    # pickle the s2n xarray
    output_pickle_path = os.path.join(output_dir, "s2n_cube.pkl")
    with open(output_pickle_path, "wb") as f:
        pickle.dump(s2n, f)

    print(f"Pickled S/N cube to: {output_pickle_path}")

    # simple plots, mostly FYI
    # load the s2n xarray from the pickle file
    with open(output_pickle_path, "rb") as f:
        s2n = pickle.load(f)
    # example plots
    # s2n.sel(n_int=25920, dc=5.0, qe=0.6, method="nearest").plot(x="wavel") # 1D
    # s2n.sel(n_int=25920, dc=5.0, method="nearest").plot(x="wavel", y="qe") # 2D
    qe_choice = 0.8
    sl = s2n.sel(n_int=25920, qe=qe_choice, method="nearest")
    
    # Create the plot using xarray's plot method
    # This creates a figure and axes automatically
    plot_handle = sl.plot(x="wavel", y="dc")
    
    # Get the current axes (the one created by xarray's plot)
    ax = plt.gca()  # Get current axes - this is the one from xarray's plot
    
    # Overplot a white contour at S/N=iso on the same axes
    X, Y = np.meshgrid(sl.wavel.values, sl.dc.values)
    CS = ax.contour(X, Y, sl.values, levels=[iso_line], colors='white', linewidths=2)
    #ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Dark current (e-/s/pix)')
    ax.set_title(f'QE = {qe_choice:.2f}')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    #plt.show()
    file_name_plot = os.path.join(output_dir, plot_save_string + '2d_heat_map_dc_vs_wavel'+f'_iso_{iso_line:.1f}.png') 
    plt.suptitle('Data from dir: ' + source_string, fontsize=6)
    plt.savefig(file_name_plot)
    print(f"Saved 2D heat map of S/N as function of wavelength and dark current to {file_name_plot}")

    # 3D plotting
    # pick one integration time
    da = s2n.sel(n_int=25920, method="nearest")  # dims: (qe, dc, wavel)
    # Ensure the axis order is exactly (qe, dc, wavel)
    da_plot = da.transpose("qe", "dc", "wavel")
    # orient such that:
    # - wavel increases to the right (x-axis)
    # - DC increases going up (y-axis)
    # - QE increases going away from the viewer (z-axis)
    #da_plot = da.transpose("dc", "wavel", "qe")
    # Reverse the wavel axis direction (highest to lowest)
    #da_plot = da_plot.reindex(wavel=list(reversed(da_plot.wavel)))
    # If there are NaNs, marching cubes will choke; fill or mask
    da_filled = da_plot.fillna(-np.inf)  # makes NaNs safely "below" any finite iso value
    # Add zoom feature to camera: "zoom" scales the field of view (default=1)
    zoom = 1.5
    #for factor in np.arange(0,1,0.1):
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-zoom*0.9, y=zoom*1.25, z=zoom*0.5)
    )
    axis_ranges = {
        "x": [float(da_filled.qe.min()), float(da_filled.qe.max())],
        "y": [float(da_filled.dc.min()), float(da_filled.dc.max())],
        "z": [float(da_filled.wavel.min()), float(da_filled.wavel.max())],
    }

    # 3d projection
    _ = plotting_3d.plot_s2n_3d_qe_dc_wavel(
        da_filled,
        iso=iso_line,
        camera=camera,
        task='save',
        axis_ranges=axis_ranges,
        title = 'Data from dir: ' + source_string,
        file_name=os.path.join(output_dir, plot_save_string + f's2n_3d_qe_dc_wavel_iso_{iso_line:.1f}.png')
    )

    '''
    # 2d
    _ = plotting_3d.plot_s2n_3d_qe_dc_wavel(
        da_filled,
        iso=5.0,
        view="overhead",
        projection_type="orthographic",
        task="show",
    )
    '''


    '''
    # 2d heat map projections
    plt.clf()
    _ = da_filled.plot(x="wavel", y="dc", col="qe")
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Dark current (e-/s/pix)')
    plt.suptitle('Data from dir: ' + dir_sample_data)
    plt.show()
    '''

    # Make a 2D plot of QE vs wavelength, averaging S/N over DC
    plt.clf()
    # Compute mean along 'dc' axis
    da_qe_wavel = da_filled.mean(dim='dc')
    # da_qe_wavel is now dims: qe x wavel
    qe_vals = da_qe_wavel.qe.values
    wavel_vals = da_qe_wavel.wavel.values
    s2n_mean = da_qe_wavel.values  # shape (Nqe, Nwavel)
    # Use pcolormesh for a 2D image
    plt.pcolormesh(wavel_vals, qe_vals, s2n_mean, shading='auto')
    # Add white contour for s2n=5
    plt.colorbar(label='Mean S/N (across DC)')
    CS = plt.contour(wavel_vals, qe_vals, s2n_mean, levels=[iso_line], colors='white', linewidths=1.5)
    plt.clabel(CS, inline=1, fontsize=10, fmt={iso_line: f'S/N = {iso_line}'}, colors='white')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('QE')
    # Truncate directory string if too long for display
    # Print only the string of the last directory in dir_display
    plt.suptitle('Data from dir: ' + source_string, fontsize=6)
    plt.title(f'Mean S/N, across DC vals {np.min(da_filled.dc.values):.1f} to {np.max(da_filled.dc.values):.1f}')
    file_name_plot = os.path.join(output_dir, plot_save_string + 's2n_qe_vs_wavel_mean_across_dc'+f'_iso_{iso_line:.1f}.png')
    plt.savefig(file_name_plot)
    #plt.show()
    print(f"Saved 2D plot of mean S/N (across DC) as function of QE and wavelength to {file_name_plot}")

    # too many subplots
    '''
    plt.clf()
    _ = da_filled.plot(x="wavel", y="qe", col="dc")
    plt.tight_layout()
    plt.show()
    '''


if __name__ == '__main__':
    main()