"""Interferometric transmission screens and passing the sky scene through them."""

import logging
from typing import Sequence

import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.output_dirs import FYI, INTERMEDIATE, output_dir
from ...utils.helpers.wavelengths import incident_cube_wavelengths
from ...utils.types import ConfigLike
from ...viz.transmission import (
    plot_flux_through_screens,
    plot_pre_aperture_all_sources,
    plot_source_transmission_triptych,
)

# slice order of the transmission-response cube returned by
# generate_instrument_transmission, with the quantity each slice holds
TRANSMISSION_CUBE_SLICES = (
    ("output_1_bright", "On-sky transmission of bright output 1", ""),
    ("output_2_bright", "On-sky transmission of bright output 2", ""),
    ("output_3_dark", "On-sky transmission of dark output 3", ""),
    ("output_4_dark", "On-sky transmission of dark output 4", ""),
    ("yy", "Sky y coordinate of each pixel (not rotated)", "arcsec"),
    ("xx", "Sky x coordinate of each pixel (not rotated)", "arcsec"),
)

# value of wavel_m that selects screens on the incident-flux wavelength grid
POLYCHROMATIC = "polychromatic"

# max elements of the complex field work buffer in generate_instrument_transmission
# (2e7 complex128 = 320 MB); the wavelengths are processed in chunks of this size
_FIELD_CHUNK_ELEMENTS = 20_000_000


def format_transmission_fits_name(angle_deg: float) -> str:
    """``transmission_instrument_response_angle_{angle:06.2f}.fits``"""
    return f"transmission_instrument_response_angle_{float(angle_deg):06.2f}.fits"


def write_transmission_responses_fits(
    file_name: str,
    transmission_cube: np.ndarray,
    header_cards: dict | None = None,
    wavel_um: np.ndarray | None = None,
) -> str:
    """
    Write the transmission responses for one angle to a single FITS file.

    transmission_cube: shape (6, n_pix, n_pix) (monochromatic) or (6, n_wavel, n_pix,
    n_pix) (polychromatic), slices ordered as TRANSMISSION_CUBE_SLICES (outputs 1-4,
    then y and x sky coordinates).

    HDUs: empty PRIMARY (header_cards, e.g. ANGLE, WAVEL_M), then OUTPUT_1_BRIGHT,
    OUTPUT_2_BRIGHT, OUTPUT_3_DARK, OUTPUT_4_DARK (each (n_pix, n_pix), or (n_wavel,
    n_pix, n_pix) if polychromatic), YY_ARCSEC, XX_ARCSEC (always (n_pix, n_pix)); each
    image HDU has QUANTITY (what it holds) and BUNIT in its header. If wavel_um is
    given (polychromatic), also a WAVELENGTH table (column wavel, um), one row per
    screen slice. Returns file_name.
    """
    if transmission_cube.shape[0] != len(TRANSMISSION_CUBE_SLICES):
        raise ValueError(
            f"Expected {len(TRANSMISSION_CUBE_SLICES)} slices in the transmission "
            f"cube, got shape {transmission_cube.shape}"
        )
    polychromatic = transmission_cube.ndim == 4
    primary = fits.PrimaryHDU()
    for key, val in (header_cards or {}).items():
        primary.header[key] = val
    hdus = [primary]
    for i, (name, quantity, unit) in enumerate(TRANSMISSION_CUBE_SLICES):
        extname = f"{name.upper()}_ARCSEC" if unit == "arcsec" else name.upper()
        data = np.asarray(transmission_cube[i], dtype=float)
        if polychromatic and unit == "arcsec":
            data = data[0]  # coordinates do not depend on wavelength
        hdu = fits.ImageHDU(data, name=extname)
        hdu.header["QUANTITY"] = (quantity, "Quantity in this HDU")
        hdu.header["BUNIT"] = (unit, "Unit ('' = dimensionless)")
        hdus.append(hdu)
    if wavel_um is not None:
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="wavel",
                        format="D",
                        unit="um",
                        array=np.asarray(wavel_um, dtype=float),
                    )
                ],
                name="WAVELENGTH",
            )
        )
    fits.HDUList(hdus).writeto(file_name, overwrite=True)
    return file_name


@pipeline_stage(
    cluster="instrument transmission (per angle)",
    cluster_color="#e41a1c",
)
def generate_instrument_transmission(
    config: ConfigLike,
    wavel_m: float | str = 11e-6,
    override_stellar_mask: bool = False,
    normalize: bool = True,
    plot: bool = False,
    angle_deg: float = 0,
) -> np.ndarray:
    # instrument transmission respose over the sky (R_theta_vec, Dannert 2025 Eqn. B12,
    # ignoring polarization for now)

    """
    INPUTS:
    # wavel_m (float or str): Wavelength in meters (e.g., 11e-6; the default), or
    # "polychromatic" for one screen per wavelength of the incident-flux cubes
    # (utils.helpers.wavelengths.incident_cube_wavelengths), so that the screens
    # multiply the (n_wavel, n_pix, n_pix) scene cubes slice by slice.
    # normalize (bool): If True, normalize transmission to unity maximum. If False, max
    # transmission is N for N identical apertures.
    # plot (bool): If True, write the responses for this angle to
    # transmission_instrument_response_angle_<angle>.fits (one HDU per output plus the
    # y and x sky coordinates, and a WAVELENGTH table if polychromatic) and
    # differential_dark_angle_<angle>.fits, in the run's intermediate/ directory.
    # angle_deg (float): Rotation angle of the transmission screens in degrees (done
    # exactly, by rotating the aperture positions; see below).

    OUTPUT:
    # Returns:
    #   transmission_instrument_response: np.ndarray of shape (6, Ny, Nx), or
    #   (6, n_wavel, Ny, Nx) if wavel_m == "polychromatic"
    #     [0]: transmission, bright output 1
    #     [1]: transmission, bright output 2
    #     [2]: transmission, dark output 3
    #     [3]: transmission, dark output 4
    #     [4]: y-coordinates on sky [arcsec] (same for every wavelength)
    #     [5]: x-coordinates on sky [arcsec] (same for every wavelength)
    """

    # wavelength(s) of the screens: always computed along a wavelength axis, which is
    # squeezed out at the end in the monochromatic case
    if isinstance(wavel_m, str):
        if wavel_m != POLYCHROMATIC:
            raise ValueError(
                f"wavel_m must be a wavelength in m or {POLYCHROMATIC!r}; got "
                f"{wavel_m!r}"
            )
        polychromatic = True
        wavel_um = incident_cube_wavelengths(config)
        wavel_m_arr = wavel_um.to_value(u.m)
        logging.info(
            "Polychromatic transmission screens: %d wavelengths, %.3g-%.3g um",
            wavel_m_arr.size,
            wavel_um[0].value,
            wavel_um[-1].value,
        )
    else:
        polychromatic = False
        wavel_um = None
        wavel_m_arr = np.array([float(wavel_m)])
    n_wavel = wavel_m_arr.size

    # read in array parameters from config file
    aperture_array_definition_file_name = config["telescope"][
        "aperture_array_config_file_name"
    ]
    with open(aperture_array_definition_file_name, "r", encoding="utf-8") as file:
        aperture_array_definition = yaml.safe_load(file)

    # construct vectors
    A_vec = []  # amplitudes
    pos_vec_m = []  # [y_m, x_m] per aperture
    for aperture in aperture_array_definition["apertures"]:
        A_vec.append(aperture["amplitude"])
        pos_vec_m.append([aperture["y_m"], aperture["x_m"]])
    A_vec = np.array(A_vec, dtype=float)
    pos_vec_m = np.array(pos_vec_m, dtype=float)

    n_pix = int(config["onsky_scene"]["n_pix"])
    pix_size_mas = float(config["onsky_scene"]["pix_size_mas"])  # milliarcseconds
    pix_size_arcsec = pix_size_mas / 1000.0  # arcsec
    axis_arcsec = (np.arange(n_pix) - (n_pix // 2)) * pix_size_arcsec
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    # θ_y varies along rows (axis 0) and θ_x along columns (axis 1) of the sky grid,
    # both over the same 1-D axis [rad], shape (n_pix,)
    theta_axis_rad = axis_arcsec * arcsec_to_rad

    # Calculate total number of baselines (unique pairs of apertures)
    N_apertures = len(aperture_array_definition["apertures"])
    logging.info("Number of apertures: %s", N_apertures)
    # For N apertures, number of unique baselines = N*(N-1)/2
    N_baselines = N_apertures * (N_apertures - 1) // 2
    logging.info("Total number of baselines: %s", N_baselines)

    # rotate the array instead of the screens (exact; no interpolation, no zero-filled
    # corners): rotating the screen T(θ) by α gives T(R(-α) θ), and
    # b · R(-α) θ = (R(α) b) · θ, so it is the screen of the array with every aperture
    # position (hence every baseline) rotated by α. The sense of rotation matches the
    # previous ndimage.rotate(..., axes=(1, 2)) on the (y, x) pixel grid.
    alpha_rad = np.deg2rad(float(angle_deg))
    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    # (y, x) -> (y cos α - x sin α, y sin α + x cos α)
    rot_yx = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pos_rot_m = pos_vec_m @ rot_yx.T

    # normalization: the response peaks at (sum_j A_j)^2 (all arms in phase)
    max_response = float(np.sum(A_vec)) ** 2 if normalize else 1.0

    # 2π/λ along the wavelength axis, shape (n_wavel,)
    k_wave = 2.0 * np.pi / wavel_m_arr

    # The phase of aperture j at sky position θ, (2π/λ) x_j · θ = k (y_j θ_y + x_j θ_x),
    # is a sum of a row term and a column term, so exp(i ...) factors into a 1-D
    # factor along y times a 1-D factor along x. These are shared by every output:
    #   phase_y[λ, iy, j] = exp(i k_λ y_j θ_y[iy]),  shape (n_wavel, n_pix, N_ap)
    #   phase_x[λ, j, ix] = exp(i k_λ x_j θ_x[ix]),  shape (n_wavel, N_ap, n_pix)
    phase_y = np.exp(
        1j
        * k_wave[:, None, None]
        * theta_axis_rad[None, :, None]
        * pos_rot_m[None, None, :, 0]
    )
    phase_x = np.exp(
        1j
        * k_wave[:, None, None]
        * pos_rot_m[None, :, 1, None]
        * theta_axis_rad[None, None, :]
    )
    # wavelengths per chunk, so the complex field work buffer stays ~ a few 100 MB
    n_wavel_chunk = max(1, _FIELD_CHUNK_ELEMENTS // (n_pix * n_pix))

    def R_m(phase_vector_rad: np.ndarray, output_name: str, out: np.ndarray) -> None:
        """
        On-sky response of output m, written into out, shape (n_wavel, Ny, Nx)
        (Dannert 2025 Eqn. B12; Lay 2004 Eqn. 3; see docs/transmission_response_math.pdf).

        The field at output m from a point source at sky position θ is
            E_m(θ, λ) = sum_j A_j exp(i [φ_mj + (2π/λ) x_j · θ])
        (A_j: amplitude of arm j; φ_mj: phase of arm j in output m, from the YAML
        phase_vector_deg; x_j = (y_j, x_j): position of aperture j, rotated by
        angle_deg). The response is the intensity, which is real:
            R_m = |E_m|^2 = sum_j sum_k A_j A_k cos(Δφ_jk + (2π/λ) b_jk · θ),
        with Δφ_jk = φ_mk - φ_mj and baseline b_jk = x_k - x_j, i.e.
            R_m = sum_j A_j^2 + 2 sum_{j<k} A_j A_k cos(Δφ_jk + (2π/λ) b_jk · θ).

        It is computed as |E_m|^2 directly. With c_mj = A_j exp(i φ_mj) and the 1-D
        factors above,
            E_m[λ, iy, ix] = sum_j (c_mj phase_y[λ, iy, j]) phase_x[λ, j, ix],
        a (n_pix x N_ap) @ (N_ap x n_pix) matrix product per wavelength, so no
        per-pixel trig is needed (and taking |E|^2 avoids the cancellation of the
        cosine sum near a null). Divided by (sum_j A_j)^2 if normalize, so the maximum
        is 1.
        """
        c_m = A_vec * np.exp(1j * phase_vector_rad)  # (N_ap,)
        weighted_y = phase_y * c_m  # (n_wavel, n_pix, N_ap)
        for i0 in range(0, n_wavel, n_wavel_chunk):
            i1 = min(i0 + n_wavel_chunk, n_wavel)
            field = np.matmul(weighted_y[i0:i1], phase_x[i0:i1])  # (chunk, Ny, Nx)
            # |E|^2 = Re^2 + Im^2
            out_chunk = out[i0:i1]
            np.multiply(field.real, field.real, out=out_chunk)
            out_chunk += field.imag * field.imag
            del field

        # conceptual point here! this response to photons is real, not complex! See Lay
        # Eqn. (3): it's the rr*
        if normalize:
            out /= max_response
            logging.info(
                "Normalized transmission instrument response to unity, based on way "
                "bookkeeping is done downstream"
            )

        # a small override mask is put over the star for now to avoid geometrical
        # leakage ## ## TODO: remove this once the geometry is properly implemented
        # Apply only to dark outputs; bright ports should keep their geometric response.
        if override_stellar_mask and output_name in (
            "output_3_dark",
            "output_4_dark",
        ):
            nulling_factor = float(config["nulling"]["nulling_factor"])
            logging.info(
                "Star is manually being nulled to %s on %s", nulling_factor, output_name
            )
            # mask a central circular region over the star
            mask_radius_pix = 4
            cy, cx = n_pix // 2, n_pix // 2
            y_idx, x_idx = np.ogrid[:n_pix, :n_pix]
            circular_mask = (y_idx - cy) ** 2 + (x_idx - cx) ** 2 <= mask_radius_pix**2
            out[..., circular_mask] = nulling_factor

    # cube of all outputs, slices ordered as TRANSMISSION_CUBE_SLICES:
    # 0 = output_1_bright, 1 = output_2_bright, 2 = output_3_dark, 3 = output_4_dark,
    # 4 = y, 5 = x; shape (6, n_wavel, n_pix, n_pix), the wavelength axis dropped below
    # if monochromatic. Each output's response is written straight into its slice.
    transmission_instrument_response_cube = np.empty(
        (len(TRANSMISSION_CUBE_SLICES), n_wavel, n_pix, n_pix)
    )
    output_slice_index = {
        slice_name: t for t, (slice_name, _, _) in enumerate(TRANSMISSION_CUBE_SLICES[:4])
    }
    filled_outputs = set()
    for output in aperture_array_definition["outputs"]:
        if output["name"] not in output_slice_index:
            logging.warning(
                "Output %r in %s is not one of %s; its response is not computed",
                output["name"],
                aperture_array_definition_file_name,
                list(output_slice_index),
            )
            continue
        R_m(
            phase_vector_rad=np.deg2rad(output["phase_vector_deg"]),
            output_name=output["name"],
            out=transmission_instrument_response_cube[output_slice_index[output["name"]]],
        )
        filled_outputs.add(output["name"])
    missing_outputs = set(output_slice_index) - filled_outputs
    if missing_outputs:
        raise KeyError(
            f"Aperture array definition {aperture_array_definition_file_name} has no "
            f"output(s) {sorted(missing_outputs)}"
        )
    # sky coordinates straight from the grid (not rotated), same at every wavelength
    transmission_instrument_response_cube[4] = axis_arcsec[:, None]  # y vals
    transmission_instrument_response_cube[5] = axis_arcsec[None, :]  # x vals

    # # check: double Bracewell should look like HOSTS
    # if plot:
    #     plt.clf()
    #
    #     ipdb.set_trace()
    #
    #     arcsec_to_rad = np.pi / (180.0 * 3600.0)
    #
    #     y_sky_asec = transmission_instrument_response[1, :, :]  # y at each pixel
    #     x_sky_asec = transmission_instrument_response[2, :, :]  # x at each pixel
    #     x_sky_rad = x_sky_asec * arcsec_to_rad
    #     y_sky_rad = y_sky_asec * arcsec_to_rad
    #     #y_sky_asec = y_sky_rad * 206265
    #     #x_sky_asec = x_sky_rad * 206265
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #
    #     extent = [
    #         x_sky_asec.min(), x_sky_asec.max(),   # left, right
    #         y_sky_asec.min(), y_sky_asec.max(),   # bottom, top
    #     ]
    #
    #     im0 = axes[0].imshow(
    #         transmission_instrument_response[0, :, :],
    #         origin="lower",
    #         extent=extent,
    #         aspect="equal",
    #     )
    #     #axes[0].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
    #     #axes[0].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
    #     axes[0].set_xlabel("x [arcsec]")
    #     axes[0].set_ylabel("y [arcsec]")
    #     axes[0].set_title("Net on-sky transmission")
    #     plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    #
    #     # make what the HOSTS screen should look like
    #     hosts_trans = np.power( np.sin(np.pi * x_sky_rad * 14.4 / 11e-6), 2 )
    #
    #     im1 = axes[1].imshow(
    #         hosts_trans,
    #         origin="lower",
    #         extent=extent,
    #         aspect="equal",
    #     )
    #     #axes[1].set_xlim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
    #     #axes[1].set_ylim(-0.5, 0.5) # zoom in on central 1x1 arcsec**2
    #     axes[1].set_xlabel("x [arcsec]")
    #     axes[1].set_ylabel("y [arcsec]")
    #     axes[1].set_title("HOSTS transmission (l/B=0.158asec)")
    #     plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    #     plt.tight_layout()
    #     ipdb.set_trace()
    #     file_name_plot = str(config['dirs']['save_s2n_data_unique_dir']) +
    # f"transmission_instrument_response.png"
    #     plt.savefig(file_name_plot)
    #     logging.info(f"Saved plot of transmission instrument response to
    # {file_name_plot}")
    #     plt.close(fig)
    #
    #     save_dir = str(config['dirs']['save_s2n_data_unique_dir'])
    #     transmission_cube_3d = np.stack(
    #         (transmission_instrument_response[0, :, :], y_sky_asec, x_sky_asec),
    #         axis=0,
    #     )
    #     hosts_cube_3d = np.stack((hosts_trans, y_sky_asec, x_sky_asec), axis=0)
    #     fits.writeto(
    #         save_dir + "transmission_instrument_response_cube.fits",
    #         transmission_cube_3d,
    #         overwrite=True,
    #     )
    #     fits.writeto(
    #         save_dir + "hosts_transmission_cube.fits",
    #         hosts_cube_3d,
    #         overwrite=True,
    #     )
    #     logging.info(
    #         f"Saved 3D FITS cubes:
    # {save_dir}transmission_instrument_response_cube.fits, "
    #         f"{save_dir}hosts_transmission_cube.fits"
    #     )
    #

    if not polychromatic:
        transmission_instrument_response_cube = transmission_instrument_response_cube[
            :, 0
        ]

    if plot:
        # save all responses for this angle to one FITS file (one HDU per output,
        # plus the y and x sky coordinates), in the run's intermediate/ directory
        save_dir = output_dir(config, INTERMEDIATE)
        header_cards = {
            "ANGLE": (float(angle_deg), "Screen rotation angle [deg]"),
            "WAVEL_M": (
                POLYCHROMATIC if polychromatic else float(wavel_m_arr[0]),
                "Wavelength of the responses [m]",
            ),
            "NWAVEL": (n_wavel, "Number of wavelengths (screen slices)"),
            "NORMALIZ": (bool(normalize), "Normalized to unit maximum transmission"),
            "STARMASK": (
                bool(override_stellar_mask),
                "Manual stellar mask on the dark outputs",
            ),
        }
        if override_stellar_mask:
            header_cards["NULLFAC"] = (
                float(config["nulling"]["nulling_factor"]),
                "Transmission inside the manual stellar mask",
            )
        file_name_fits = write_transmission_responses_fits(
            save_dir + format_transmission_fits_name(angle_deg),
            transmission_instrument_response_cube,
            header_cards=header_cards,
            wavel_um=None if wavel_um is None else wavel_um.value,
        )
        logging.info(
            "Saved transmission instrument responses for angle %.2f deg (HDUs %s) "
            "to %s",
            angle_deg,
            ", ".join(name for name, _, _ in TRANSMISSION_CUBE_SLICES),
            file_name_fits,
        )
        # save the differential dark (one file per angle, so angles do not overwrite
        # each other); (n_pix, n_pix), or (n_wavel, n_pix, n_pix) if polychromatic
        differential_dark = (
            transmission_instrument_response_cube[2]
            - transmission_instrument_response_cube[3]
        )
        file_name_diff = save_dir + f"differential_dark_angle_{float(angle_deg):06.2f}.fits"
        fits.writeto(file_name_diff, differential_dark, overwrite=True)
        logging.info("Saved differential dark to %s", file_name_diff)

    return transmission_instrument_response_cube


def sky_summed_post_screen_cube(
    source_cube: u.Quantity | np.ndarray, screen: np.ndarray
) -> u.Quantity | np.ndarray:
    """
    Sum over the sky of source_cube x screen, without forming the product cube.

    source_cube: (n_wavel, n_pix, n_pix) (Quantity or array); screen: one output's
    transmission, (n_wavel, n_pix, n_pix) (polychromatic) or (n_pix, n_pix)
    (monochromatic). Returns shape (n_wavel, 1, 1) with source_cube's unit: a cube
    whose whole sky is one pixel, so np.sum(..., axis=(1, 2)) of it equals that of
    the full product.
    """
    unit = getattr(source_cube, "unit", None)
    values = np.asarray(getattr(source_cube, "value", source_cube))
    n_wavel = values.shape[0]
    flat_source = values.reshape(n_wavel, -1)
    if screen.ndim == 3:
        # per-wavelength dot product over the sky (batched BLAS)
        summed = np.matmul(
            flat_source[:, None, :], screen.reshape(n_wavel, -1, 1)
        )[:, 0, 0]
    else:
        summed = flat_source @ screen.ravel()
    summed = summed[:, None, None]
    return summed * unit if unit is not None else summed


@pipeline_stage(
    depends_on=("arrange_onsky_scene", "generate_instrument_transmission"),
)
def pass_through_transmission_screens(
    config: ConfigLike,
    sources_astroph: dict,
    sources_to_include: Sequence[str],
    fyi_angle: float,
    source_dict_pre_screen: dict,
    transmission_screens: np.ndarray,
    plot: bool = False,
    keep_sky_cubes: bool = True,
) -> np.ndarray:
    """
    Pass each astrophysical source through the transmission screens, and update
    prop_dict with the propagated terms
    photons/sec/m^2 -> photons/sec/m^2

    keep_sky_cubes (bool): if True, "flux_cube_post_screen_ph_sec_m2_um" holds the
        full post-screen cube (n_wavel, n_pix, n_pix) of each source and output. If
        False (and not plot), it holds the cube already summed over the sky, with
        shape (n_wavel, 1, 1): everything downstream (aperture, backgrounds,
        dispersion) only scales or sky-sums these cubes, so the results are the same
        (to rounding), without building 4 x n_sources full-size cubes per angle.
        Forced True if plot (the FYI plots show the post-screen images).

    INPUTS:
        fyi_angle (float): angle of the transmission screen (for plotting strings only)
        source_cube_no_screen (dict of Quantities): on-sky scene before transmission
            screen; for each key (astro source), value (Quantity array) has shape
            (n_wavel, n_pix, n_pix)
        transmission_screens (np.ndarray): output of generate_instrument_transmission,
            shape (6, n_pix, n_pix) (monochromatic: the same screen at every
            wavelength) or (6, n_wavel, n_pix, n_pix) (polychromatic: one screen per
            wavelength of the scene cubes)
        plot (bool): whether to plot the scene (FYI plots go to fyi_plots/)

    Returns the four transmission-screen slices, shape (4, n_pix, n_pix) or (4,
    n_wavel, n_pix, n_pix).
    """

    transmission_screen_order = [
        "output_1_bright",
        "output_2_bright",
        "output_3_dark",
        "output_4_dark",
    ]  # ## TODO: insert check to ensure always consistent
    # just keep the transmission slices for now
    # ## TODO: include the yy and xx slices as a check somehow
    transmission_screens = transmission_screens[0:4]
    polychromatic = transmission_screens.ndim == 4

    # polychromatic screens multiply the scene cubes slice by slice, so the wavelength
    # grids must match (both come from incident_cube_wavelengths)
    if polychromatic:
        for source_name, source_val in source_dict_pre_screen.items():
            if source_val.shape != transmission_screens.shape[1:]:
                raise ValueError(
                    f"Scene cube of {source_name} has shape {source_val.shape}, but the "
                    "polychromatic transmission screens have shape "
                    f"{transmission_screens.shape[1:]} (n_wavel, n_pix, n_pix): the "
                    "wavelength grids or sky grids differ"
                )

    # plots need the post-screen images
    keep_sky_cubes = keep_sky_cubes or plot

    # put all the post-screen fluxes (for each source and from each channel) into a
    # single dict
    source_dict_post_screen = {}
    for source_name, source_val in source_dict_pre_screen.items():
        source_dict_post_screen[source_name] = {}

        for transmission_screen_name in transmission_screen_order:
            screen = transmission_screens[
                transmission_screen_order.index(transmission_screen_name)
            ]
            if keep_sky_cubes:
                post_screen = source_val * screen
            else:
                post_screen = sky_summed_post_screen_cube(source_val, screen)
            source_dict_post_screen[source_name][transmission_screen_name] = (
                post_screen
            )
            # collapse the sources into a single 3D array (wavel, x, y), for plotting
            # source_dict_post_screen[source_name][transmission_screen_name +
            # '_collapsed'] =
            # np.sum(source_dict_post_screen[source_name][transmission_screen_name],
            # axis=(1,2))
    # there should be a cube for each output (4 cubes total)

    # collapse the sources into a single 3D array (wavel, x, y), for plotting
    # there should be a cube for each output (4 cubes total)
    # for transmission_screen_name in transmission_screen_order:

    # source_cube_post_screen = np.stack([source_dict_post_screen[source_name] for
    # source_name in source_dict_post_screen.keys()], axis=0)
    # source_collapsed_cube_post_screen_sum = np.sum(source_cube_post_screen, axis=0)

    # integrate over 2D sky to get total flux from each source
    # update the sources
    for source_name, source_val in source_dict_post_screen.items():
        # source_val has 4 different screens, so integrate them separately
        sources_astroph[source_name][
            "flux_integrated_post_screen_ph_sec_m2_um"
        ] = {}  # will contain flux corresponding to each screen
        sources_astroph[source_name][
            "flux_cube_post_screen_ph_sec_m2_um"
        ] = {}  # will contain flux cube corresponding to each screen
        # test_flux_1 = 0 # to check flux conservation
        # test_flux_2 = 0 # to check flux conservation
        for transmission_screen_name in transmission_screen_order:
            source_val_integrated = np.sum(
                source_val[transmission_screen_name], axis=(1, 2)
            )
            sources_astroph[source_name]["flux_integrated_post_screen_ph_sec_m2_um"][
                transmission_screen_name
            ] = source_val_integrated
            sources_astroph[source_name]["flux_cube_post_screen_ph_sec_m2_um"][
                transmission_screen_name
            ] = source_val[transmission_screen_name]
            logging.info(
                "Flux of %s passed through transmission screen %s",
                source_name,
                transmission_screen_name,
            )

            # to check flux conservation, add up all the light transmitted through each
            # screen
            # test_flux_1 += source_val_integrated
            # test_flux_2 += np.sum(source_val[transmission_screen_name], axis=(1,2))
        # if total flux after transmission is same as the input
        # if np.logical_or(
        #     np.round(test_flux_1, 1) !=
        # np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1),
        #     np.round(test_flux_2, 1) !=
        # np.round(np.sum(source_dict_pre_screen[source_name], axis=(1,2)), 1)
        # ):
        #     logging.error(f'Flux conservation check failed for {source_name} at angle
        # {fyi_angle:06.2f}')
        #     ipdb.set_trace()
        #
        if plot:  # pragma: no cover
            fyi_dir = output_dir(config, FYI)
            file_name_plot = (
                fyi_dir
                + f"flux_of_{source_name}_passed_through_transmission_screens"
                + f"_angle_{fyi_angle:06.2f}.png"
            )
            plot_flux_through_screens(
                sources_astroph[source_name]["wavel"],
                sources_astroph[source_name][
                    "flux_integrated_post_screen_ph_sec_m2_um"
                ],
                sources_astroph[source_name]["pre_screen_astro_flux_ph_sec_m2_um"],
                source_name,
                file_name_plot,
            )

            idx = 15  # wavelength slice index (for plotting only)

            for transmission_screen_name in transmission_screen_order:
                source_img = source_dict_pre_screen[source_name][idx, :, :].value
                source_units = source_val[transmission_screen_name][
                    idx, :, :
                ].unit.to_string()
                transmission_img = transmission_screens[
                    transmission_screen_order.index(transmission_screen_name)
                ]
                if polychromatic:
                    transmission_img = transmission_img[idx]
                file_name_plot = (
                    fyi_dir
                    + f"source_transmission_map_triptych_{source_name}"
                    + f"_angle_{fyi_angle:06.2f}_output_{transmission_screen_name}.png"
                )
                plot_source_transmission_triptych(
                    source_img,
                    transmission_img,
                    source_name,
                    transmission_screen_name,
                    idx,
                    source_units,
                    file_name_plot,
                )

    if plot:  # pragma: no cover
        flux_by_source = {
            source_name: (
                sources_astroph[source_name]["wavel"],
                sources_astroph[source_name]["pre_screen_astro_flux_ph_sec_m2_um"],
            )
            for source_name in sources_to_include
        }
        file_name_plot = (
            output_dir(config, FYI) + "photoelectrons_all_sources_pre_aperture.png"
        )
        plot_pre_aperture_all_sources(flux_by_source, config, file_name_plot)

    return transmission_screens
