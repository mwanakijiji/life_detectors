import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import sys
import time
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


file_name_cube = '/Users/eckhartspalding/Documents/git.repos/life_detectors/parameter_sweep/20251105_footprint_small_R_50/s2n_sweep_n00026020.fits'

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


def read_detector_catalog(file_name):
    
    catalog = pd.read_csv(file_name, skipinitialspace=True, delimiter='|', skiprows=1)
    catalog = catalog.rename(columns=lambda x: x.strip()) #strip some of the whitespace

    catalog = catalog.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    catalog['cutoff_blue_um'] = pd.to_numeric(catalog['cutoff_blue_um'])
    catalog['cutoff_red_um'] = pd.to_numeric(catalog['cutoff_red_um'])
    catalog['dark_current_upper (e/pix/sec)'] = pd.to_numeric(catalog['dark_current_upper (e/pix/sec)'])
    catalog['dark_current_lower (e/pix/sec)'] = pd.to_numeric(catalog['dark_current_lower (e/pix/sec)'])
    catalog['dark_current_max (e/pix/sec)'] = catalog[[
        'dark_current_upper (e/pix/sec)',
        'dark_current_lower (e/pix/sec)'
    ]].max(axis=1)
    catalog['string_name'] = catalog['string_name'].str.strip()
    catalog['string_tech'] = catalog['string_tech'].str.strip()

    return catalog


catalog = read_detector_catalog('/Users/eckhartspalding/Documents/git.repos/life_detectors/notebooks/data/detector_catalog.csv')


## 2D PLOT 2: LINEAR-COLOR PLOT WITH LOG-DC AXIS
# ------------------------------------------------------------------------------------------------
# Create heat map of data_cube[0,:,:] with custom tick marks and box widths
fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, height_ratios=[1, 3])
plt.subplots_adjust(hspace=0.01)  # tighten vertical spacing further

# Get the data for the heat map
heatmap_data = data_cube[0,:,:]  # S/N values
wavelength_centers = data_cube[1,:,:]  # Wavelength bin centers
wavelength_widths = data_cube[2,:,:]  # Wavelength bin widths
dark_current_values = data_cube[3,:,:]  # Dark current values

# Calculate the extent for imshow to account for variable bin widths
# We need to create a regular grid that represents the variable-width bins
n_rows, n_cols = heatmap_data.shape

# Create arrays for the edges of each bin
x_edges = np.zeros((n_rows, n_cols + 1))
y_edges = np.zeros((n_rows + 1, n_cols))

# Calculate x-edges (wavelength) for each row - centers at wavelength_centers with widths wavelength_widths
X_edges = np.zeros((n_rows + 1, n_cols + 1))
for i in range(n_rows):
    for j in range(n_cols):
        X_edges[i, j] = wavelength_centers[i, j] - wavelength_widths[i, j] / 2
    X_edges[i, -1] = wavelength_centers[i, -1] + wavelength_widths[i, -1] / 2
# Copy last row of x-edges to bottom edge
X_edges[-1, :] = X_edges[-2, :]

# Calculate y-edges (dark current) - dark current is constant across wavelengths in each row
Y_edges = np.zeros((n_rows + 1, n_cols + 1))
for i in range(n_rows):
    # Use median dark current for this row (should be constant, but use median to be safe)
    dc_val = np.median(dark_current_values[i, :])
    Y_edges[i, :] = dc_val - 0.5
Y_edges[-1, :] = np.median(dark_current_values[-1, :]) + 0.5

# Create the extent for contours and other uses
extent = [X_edges.min(), X_edges.max(), Y_edges.min(), Y_edges.max()]

# Create the heat map using pcolormesh for variable-width bins
# Option to use logarithmic colorscale
use_log_scale = False  # Set to False for linear scale

if use_log_scale:
    # For log scale, we need to handle zero/negative values
    heatmap_data_log = np.where(heatmap_data > 0, np.log10(heatmap_data), np.nan)
    im_middle = ax.pcolormesh(X_edges, Y_edges, heatmap_data_log, cmap='Blues', shading='flat')
    #im_bottom = ax_bottom.pcolormesh(X_edges, Y_edges, heatmap_data_log, cmap='viridis', shading='flat')
else:
    im_middle = ax.pcolormesh(X_edges, Y_edges, heatmap_data, cmap='Blues', shading='flat')
    #im_bottom = ax_bottom.pcolormesh(X_edges, Y_edges, heatmap_data, cmap='viridis', shading='flat')


# Add contour for S/N = 3 - use coordinate arrays for variable-width bins
# Create coordinate arrays: X uses wavelength_centers, Y uses dark current values (constant per row)
X_coords = wavelength_centers  # (n_rows, n_cols) - x-coordinates at bin centers
Y_coords = np.zeros_like(heatmap_data)
for i in range(n_rows):
    Y_coords[i, :] = np.median(dark_current_values[i, :])  # y-coordinate (dark current) for each row

contour_middle = ax.contour(X_coords, Y_coords, heatmap_data, levels=[1,2,3,4,5], colors=['white'], linewidths=2, )
ax.clabel(contour_middle, fmt='%d', colors='white', fontsize=20, inline=False)
#contour_bottom = ax_bottom.contour(heatmap_data, levels=[3,5], colors=['white'], linewidths=2, extent=extent, origin='lower')
#ax_bottom.clabel(contour_bottom, fmt='%d', colors='white', fontsize=20)

# Set up the top subplot
ax_top.set_ylim(100, 102)
ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis labels and ticks on ax_top
ax_top.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
for spine in ax_top.spines.values():
    spine.set_visible(False)  # Remove subplot borders

# Set up the bottom subplot
#ax_bottom.set_ylim(0, 0.125)
#ax_bottom.set_ylabel('DC')
#ax_bottom.tick_params(axis='x', which='both', bottom=False, labelbottom=True)  # Hide x-axis labels and ticks on ax_bottom

# the rows where the dark current so high it is off the chart
catalog_top = catalog[catalog['dark_current_max (e/pix/sec)'] > 70].reset_index(drop=True)
catalog_top = catalog_top.sort_values(by='dark_current_max (e/pix/sec)', ascending=False)
# plot horizontal lines, like in the code above, but make the lines dashed and remove all axes and tick marks

# Set y-limits to the min/max of catalog_top dark_current_max values
if len(catalog_top) > 0:
    y_min_top = float(catalog_top['dark_current_max (e/pix/sec)'].min())
    y_max_top = float(catalog_top['dark_current_max (e/pix/sec)'].max())
    ax_top.set_ylim(y_min_top, y_max_top)
    ax_top.set_yscale('log')

# Draw dashed horizontal lines on the top subplot without axes
if len(catalog_top) > 0:
    for i, row in catalog_top.iloc[::-1].iterrows():
        y_val = row['dark_current_max (e/pix/sec)']
        x_start = row['cutoff_blue_um']
        x_end = row['cutoff_red_um']
        ax_top.hlines(y=y_val, xmin=x_start, xmax=x_end, color='k', linestyle='--', linewidth=2, alpha=1.)

        # Add annotation similar to lower plots
        string_name = row['string_name']
        x_mid = (x_start + x_end) / 2
        x_mid_clipped = np.clip(x_mid, 4.0, 18.0)

        rotation_angle_top = 30 if x_mid_clipped < 7 else 0
        x_offset_top = np.random.uniform(0, 2) if rotation_angle_top == 30 else 0

        ax_top.annotate(string_name,
                        xy=(x_mid_clipped, y_val),
                        xytext=(x_mid_clipped + x_offset_top, y_val + 0.05),
                        ha='center', va='bottom',
                        fontsize=15, color='k', alpha=1.,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1.),
                        clip_on=False,
                        rotation=rotation_angle_top)



# Show y-axis on the top subplot; keep x-axis hidden
ax_top.set_axis_on()
for spine in ax_top.spines.values():
    spine.set_visible(False)
ax_top.spines['left'].set_visible(True)
ax_top.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_top.set_ylabel('DC')
ax_top.set_ylim(70, 1e7)

    
# Add horizontal lines from catalog data
# Check if catalog exists, if not we'll need to load it
try:
    catalog
    print("Catalog found, adding horizontal lines...")
    
    # Set up colormap for horizontal lines (use plasma/inferno for contrast with viridis)
    line_cmap = plt.cm.get_cmap('inferno')  # or 'inferno', 'hot', 'copper'
    # Normalize dark current values for colormap
    norm = Normalize(vmin=dark_current_values.min(), vmax=dark_current_values.max())
    
    # Plot horizontal lines for each catalog entry
    for i, row in catalog.iterrows():
        y_val = row['dark_current_max (e/pix/sec)']
        x_start = row['cutoff_blue_um']
        x_end = row['cutoff_red_um']
        string_name = row['string_name']
        
        # Plot lines for y-values in range normally, and out-of-range above the plot, dashed.
        if dark_current_values.min() <= y_val <= dark_current_values.max():
            # Get random color from colormap
            random_val = np.random.rand()
            line_color = 'orange' #line_cmap(random_val)
            
            # Middle subplot
            '''
            ax.axhline(y=y_val, xmin=(x_start - X_edges.min()) / (X_edges.max() - X_edges.min()), 
                      xmax=(x_end - X_edges.min()) / (X_edges.max() - X_edges.min()), 
                      color=line_color, linewidth=8, alpha=0.8)
            '''
            print(string_name)
            print(x_start, x_end)
            ax.hlines(y=y_val, xmin=x_start, xmax=x_end, 
                    color=line_color, linewidth=8, alpha=0.5)
            
            # Bottom subplot
            #ax_bottom.axhline(y=y_val, xmin=(x_start - x_edges.min()) / (x_edges.max() - x_edges.min()), 
            #          xmax=(x_end - x_edges.min()) / (x_edges.max() - x_edges.min()), 
            #          color=np.random.rand(3,), linewidth=8, alpha=0.5)
            
            # Add annotation with string_name - both subplots
            x_mid = (x_start + x_end) / 2
            # If x_mid < 5, set x_plot to 5
            x_plot = x_mid if x_mid >= 5 else 5
            
            rotation_angle = 30 if x_plot < 7 else 0
            x_offset = np.random.uniform(0, 2) if rotation_angle == 30 else 0
            
            # Middle subplot annotation
            ax.annotate(string_name, 
                       xy=(x_plot, y_val), 
                       xytext=(x_plot + x_offset, y_val),
                       ha='center', va='bottom',
                       fontsize=15, color='k', alpha=1.,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1.),
                       rotation=rotation_angle)
            
            # Bottom subplot annotation
            '''
            ax_bottom.annotate(string_name, 
                       xy=(x_plot, y_val), 
                       xytext=(x_plot + x_offset, y_val),
                       ha='center', va='bottom',
                       fontsize=15, color='k', alpha=1.,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=1.),
                       rotation=rotation_angle)
            '''
            
            # Store the annotation object if you want to manipulate it later
            # You can access it via ann if needed
    
    print(f"Added {len(catalog)} horizontal lines from catalog")
    
except NameError:
    print("Catalog not found. Please load the catalog data first.")
    print("Expected columns: 'cutoff_blue_um', 'cutoff_red_um', 'dark_current_upper (e/pix/sec)'")

# Set custom tick marks
# First, let's see what wavelength values we have
print("Wavelength range:", wavelength_centers[0, :].min(), "to", wavelength_centers[0, :].max())
print("Sample wavelengths:", wavelength_centers[0, ::10])  # Every 10th value

# For x-axis: show ticks at multiples of 5 microns (create them if they don't exist in data)
x_min, x_max = wavelength_centers[0, :].min(), wavelength_centers[0, :].max()
x_ticks = np.arange(5, x_max + 1, 5)  # Create ticks at 5, 10, 15, etc.
x_labels = [f'{w:.0f}' for w in x_ticks]
#ax_bottom.set_xticks(x_ticks)
#ax_bottom.set_xticklabels(x_labels)


# For y-axis: only show ticks at multiples of 10 electrons
y_ticks = []
y_labels = []
for dc in dark_current_values[:, 0]:
    if dc % 1 == 0:  # Check if dark current is a multiple of 10
        y_ticks.append(dc)
        y_labels.append(f'{dc:.0f}')
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

ax.set_xlim(4, 18)
ax.set_ylim(0.01, 70)
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Wavelength (um)')
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)

ax.set_ylabel('Dark Current (e/pix/s)')
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)

ax_top.xaxis.label.set_size(15)
ax_top.yaxis.label.set_size(15)

for a in [ax_top, ax]:
    a.tick_params(axis='both', which='both', labelsize=15)
#ax.set_title('S/N Heat Map')



# Add colorbar along the bottom

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="4%", pad=0.8)
cbar = plt.colorbar(im_middle, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
if use_log_scale:
    cbar.set_label('log₁₀(S/N)', fontsize=15)
else:
    cbar.set_label('S/N', fontsize=15)

plt.tight_layout()
#plt.show()

file_name_plot = '/Users/eckhartspalding/Downloads/junk.png'
plt.savefig(file_name_plot)
print('Using data from file: ' + file_name_cube)
print(f"Saved plot to {file_name_plot}")