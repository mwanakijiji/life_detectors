# helper functions for plotting 3D S/N cubes

import numpy as np
import xarray as xr
from skimage.measure import marching_cubes
import plotly.graph_objects as go
import plotly.io as pio
import ipdb

def _angles_to_camera_eye(azimuth_deg, elevation_deg, distance=2.0):
    """
    Convert rotation angles to Plotly camera eye position.
    
    Parameters:
    -----------
    azimuth_deg : float
        Rotation around vertical axis (0 = +x direction, 90 = +z direction, in degrees)
    elevation_deg : float
        Angle above/below horizontal plane (0 = horizontal, 90 = straight up, in degrees)
    distance : float
        Distance from center to camera eye
    
    Returns:
    --------
    dict : Camera eye position as {'x': x, 'y': y, 'z': z}
    """
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    
    # Convert spherical to Cartesian coordinates
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.sin(elevation_rad)
    z = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    
    return dict(x=x, y=y, z=z)

def plot_s2n_3d_qe_dc_wavel(da_pass, iso=5.0, camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    ), task='show', filename=None):
    """
    Plot 3D isosurface of S/N data.
    
    Parameters:
    -----------
    da_pass : xarray.DataArray
        DataArray with dimensions (qe, dc, wavel)
    iso : float
        Isosurface level (default: 5.0)
    camera : dict
        Dictionary containing camera parameters for viewing orientation
    task : str
        'show' or 'save'
    filename : str, optional
        Filename for saving (default: auto-generated based on iso value)
    """

    # uniform-grid step
    qe_u = np.linspace(float(da_pass.qe.min()), float(da_pass.qe.max()), da_pass.sizes["qe"])
    dc_u = np.linspace(float(da_pass.dc.min()), float(da_pass.dc.max()), da_pass.sizes["dc"])
    wv_u = np.linspace(float(da_pass.wavel.min()), float(da_pass.wavel.max()), da_pass.sizes["wavel"])

    # Convert input data AND coordinates to native byte order to avoid issues with interpolation
    # xarray's interp uses the coordinate arrays internally, so they must also be native byte order
    def to_native_byteorder(arr):
        """Convert array to native byte order if needed."""
        if arr.dtype.byteorder in ('>', '='):
            return np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder('='))
        return arr
    
    # Convert data array
    data_native = to_native_byteorder(da_pass.values)
    
    # Convert coordinate arrays
    coords_native = {}
    for dim in da_pass.dims:
        coord_vals = da_pass[dim].values
        coords_native[dim] = to_native_byteorder(coord_vals)
    
    # Recreate DataArray with native byte order data and coordinates
    da_pass_native = xr.DataArray(
        data_native,
        dims=da_pass.dims,
        coords=coords_native,
        attrs=da_pass.attrs
    )

    # Perform interpolation
    da_u = da_pass_native.interp(qe=qe_u, dc=dc_u, wavel=wv_u).fillna(-np.inf)
    
    # Convert interpolated result to native byte order as well
    # (interp might preserve byte order from input, so we ensure it's native)
    da_u = da_u.copy(data=np.ascontiguousarray(
        da_u.values,
        dtype=da_u.values.dtype.newbyteorder('=')
    ))

    vol = da_u.values  # or da_filled.values if you skipped interpolation
    
    # Convert to native byte order (fixes "Big-endian buffer not supported" error)
    # This ensures the array is in the system's native byte order for marching_cubes
    # (Even though we converted da_u, we ensure vol is definitely native before marching_cubes)
    vol = np.ascontiguousarray(vol, dtype=vol.dtype.newbyteorder('='))

    # marching_cubes returns vertices in index coordinates (i,j,k)
    verts, faces, normals, values = marching_cubes(vol, level=iso)

    # Map index coordinates -> physical coordinates using your xarray coords
    # Ensure coordinate arrays are also in native byte order
    qe = np.ascontiguousarray(da_u.qe.values, dtype=da_u.qe.values.dtype.newbyteorder('='))
    dc = np.ascontiguousarray(da_u.dc.values, dtype=da_u.dc.values.dtype.newbyteorder('='))
    wv = np.ascontiguousarray(da_u.wavel.values, dtype=da_u.wavel.values.dtype.newbyteorder('='))

    # verts columns correspond to axis order in vol: (qe, dc, wavel) => (i, j, k)
    x_qe = np.interp(verts[:, 0], np.arange(len(qe)), qe)
    y_dc = np.interp(verts[:, 1], np.arange(len(dc)), dc)
    z_wv = np.interp(verts[:, 2], np.arange(len(wv)), wv)

    ipdb.set_trace()

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_qe, y=y_dc, z=z_wv,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.6,
            name=f"s2n = {iso} isosurface",
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title="qe",
            yaxis_title="dc",
            zaxis_title="wavel",
            camera=camera,
            aspectmode='cube',  # Change to 'cube' if you want equal visual scaling
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Isosurface: s2n = {iso}"
    )

    if task == 'show':
        fig.show()
    elif task == 'save':
        # Use provided filename or generate one
        if filename is None:
            file_name = f"s2n_3d_qe_dc_wavel_iso_{iso}.png"
        else:
            file_name = filename
            # Ensure .png extension if not provided
            if not file_name.endswith('.png'):
                file_name += '.png'
        
        # Plotly requires kaleido package for image export: pip install kaleido
        try:
            fig.write_image(file_name, width=1200, height=900, scale=2)
            print(f"Saved plot to {file_name}")
        except Exception as e:
            print(f"Error saving image: {e}")
            print("Note: Plotly image export requires 'kaleido' package.")
            print("Install it with: pip install kaleido")
            # Fallback: save as HTML
            html_name = file_name.replace('.png', '.html')
            fig.write_html(html_name)
            print(f"Saved as HTML instead: {html_name}")