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
    ), task='show', axis_ranges: dict = None, view: str = None, projection_type: str = None, title: str = None, file_name: str = None):
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
    verts2, faces2, normals2, values2 = marching_cubes(vol, level=iso+1)

    # Map index coordinates -> physical coordinates using your xarray coords
    # Ensure coordinate arrays are also in native byte order
    qe = np.ascontiguousarray(da_u.qe.values, dtype=da_u.qe.values.dtype.newbyteorder('='))
    dc = np.ascontiguousarray(da_u.dc.values, dtype=da_u.dc.values.dtype.newbyteorder('='))
    wv = np.ascontiguousarray(da_u.wavel.values, dtype=da_u.wavel.values.dtype.newbyteorder('='))

    # verts columns correspond to axis order in vol: (qe, dc, wavel) => (i, j, k)
    x_qe = np.interp(verts[:, 0], np.arange(len(qe)), qe)
    y_dc = np.interp(verts[:, 1], np.arange(len(dc)), dc)
    z_wv = np.interp(verts[:, 2], np.arange(len(wv)), wv)

    x_qe2 = np.interp(verts2[:, 0], np.arange(len(qe)), qe)
    y_dc2 = np.interp(verts2[:, 1], np.arange(len(dc)), dc)
    z_wv2 = np.interp(verts2[:, 2], np.arange(len(wv)), wv)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_qe, y=y_dc, z=z_wv,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.6,
            facecolor=['red'] * len(faces),  # Color for iso surface
            name=f"s2n = {iso}",
            showlegend=True,
        )
    ])

    fig.add_trace(go.Mesh3d(
        x=x_qe2, y=y_dc2, z=z_wv2,
        i=faces2[:, 0], j=faces2[:, 1], k=faces2[:, 2],
        opacity=0.1,
        facecolor=['blue'] * len(faces2),  # Color for iso+1 surface
        name=f"s2n = {iso+1}",
        showlegend=True,
    ))
    
    if view == "overhead_dc_wavel":
        # Adjust camera so the x axis (qe) is "coming out of the screen", projecting onto (dc, wavel) plane
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.5, y=0, z=0),  # eye along +x (qe), so projection is onto (dc, wavel)
        )

    camera_scene = dict(camera)
    if projection_type:
        camera_scene["projection"] = dict(type=projection_type)

    scene = dict(
        xaxis_title="QE",
        yaxis_title="Dark current (e-/s/pix)",
        zaxis_title="Wavelength (um)",
        camera=camera_scene,
        aspectmode='cube',  # Change to 'cube' if you want equal visual scaling
    )
    if axis_ranges:
        if "x" in axis_ranges:
            scene["xaxis_range"] = axis_ranges["x"]
        if "y" in axis_ranges:
            scene["yaxis_range"] = axis_ranges["y"]
        if "z" in axis_ranges:
            scene["zaxis_range"] = axis_ranges["z"]

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=150, t=30, b=0),  # Add right margin for legend
        #title=f"Isosurface: s2n = {iso}",
        showlegend=True,  # Show legend to indicate iso values
        legend=dict(
            x=0.7,  # Position legend to the right
            y=0.7,
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
        ),
        title={'text': title, 'font': {'size': 6}},
    )

    if task == 'show':
        fig.show()
    elif task == 'save':
        # Use plotly's write_image method (requires kaleido)
        fig.write_image(file_name)
        print(f"Saved plot to {file_name}")