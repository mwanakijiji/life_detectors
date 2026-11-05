"""Optical dispersion: map wavelength bins onto a pixel grid.

Independent of detector QE/noise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits

from ...pipeline_registry import pipeline_stage
from ...utils.types import ConfigLike
from ...viz.aperture import plot_astro_ph_sec_pixel
from ...viz.detector import plot_detector_footprint
from .detector import Detector, detector_shape_from_config
from .effects import _config_get

logger = logging.getLogger(__name__)

# lower-left corner (y, x) of the first wavelength-bin footprint, if not configured
DEFAULT_TRACE_START_PIXEL = (100.0, 300.0)


def wavel_bin_edges_from_config(config: ConfigLike) -> u.Quantity:
    """
    Geometric wavelength-bin edges (um) from [detector] spec_res and
    [wavelength_range] min/max: lambda_i = lambda_min * (1 + 1/R)**i, keeping only
    the bins that fit fully in [min, max]. Shape (n_bins + 1,).
    """
    spec_res = float(config["detector"]["spec_res"])  # lambda / del_lambda
    lambda_min = float(config["wavelength_range"]["min"]) * u.um
    lambda_max = float(config["wavelength_range"]["max"]) * u.um
    n_bins = int(
        np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / spec_res))
    )
    return lambda_min * (1.0 + 1.0 / spec_res) ** np.arange(n_bins + 1)


class DispersionLaw:
    """Maps spectral bins to detector-pixel footprints.

    Physically this is a spectrograph / illumination law. It needs a pixel-grid
    geometry to realize the map, but it does not own QE, gain, or noise.
    """

    def __init__(
        self,
        config: ConfigLike,
        *,
        starting_pixel: Sequence[float] | None = None,
    ) -> None:
        self.config = config
        self.pix_per_wavel_bin = float(config["detector"]["pix_per_wavel_bin"])
        self.pix_spectral_width = int(config["detector"]["pix_spectral_width"])
        # lower-left corner of the first wavelength-bin footprint (y, x):
        # argument, else [detector] trace_start_pixel, else the default
        if starting_pixel is None:
            configured = _config_get(config, "detector", "trace_start_pixel")
            starting_pixel = (
                [float(v) for v in str(configured).split(",")]
                if configured is not None
                else DEFAULT_TRACE_START_PIXEL
            )
        self.starting_pixel = np.asarray(starting_pixel, dtype=float)
        if self.starting_pixel.shape != (2,):
            raise ValueError(
                f"trace_start_pixel must be 'y, x'; got {self.starting_pixel.tolist()}"
            )

    def _check_trace_fits(
        self, n_pix_y: int, n_pix_x: int, num_wavel_bins: int
    ) -> None:
        """Raise if the trace would fall (partly) off the detector."""
        y0, x0 = self.starting_pixel
        y1 = int(y0) + self.pix_spectral_width
        # columns touched: ceil(x0) - 1 (start edge) .. floor(x_end) (end edge)
        x_end = x0 + num_wavel_bins * self.pix_per_wavel_bin
        first_col = int(np.ceil(x0)) - 1
        last_col = int(np.floor(x_end))
        if y0 < 0 or y1 > n_pix_y or first_col < 0 or last_col >= n_pix_x:
            raise ValueError(
                f"Spectral trace does not fit on the {n_pix_y} x {n_pix_x} detector: "
                f"rows {int(y0)}..{y1 - 1}, columns {first_col}..{last_col} "
                f"({num_wavel_bins} bins x {self.pix_per_wavel_bin} pix/bin from "
                f"x = {x0}). Adjust [detector] trace_start_pixel, size_y/size_x, "
                "spec_res or the wavelength range."
            )

    def make_footprint(
        self, n_pix_y: int, n_pix_x: int, num_wavel_bins: int
    ) -> np.ndarray:
        """Return the wavelength-bin footprint cube on the detector.

        Shape is (n_bins, n_pix_y, n_pix_x). Values are 1 for fully covered pixels
        and fractional coverage on edges.
        """
        self._check_trace_fits(n_pix_y, n_pix_x, num_wavel_bins)

        footprint_cube = np.full((num_wavel_bins, n_pix_y, n_pix_x), 0.0, dtype=float)

        for wavel_bin_num in range(num_wavel_bins):
            # assumes horizontal spectra
            footprint_this = np.full((n_pix_y, n_pix_x), 0.0, dtype=float)
            starting_pixel_this = self.starting_pixel + np.array(
                [0.0, wavel_bin_num * self.pix_per_wavel_bin]
            )

            pixel_ceil_start_x = int(np.ceil(starting_pixel_this[1]))
            pixel_frac_start_x = pixel_ceil_start_x - starting_pixel_this[1]
            pixel_floor_end_x = int(
                np.floor(starting_pixel_this[1] + self.pix_per_wavel_bin)
            )
            pixel_frac_end_x = (
                starting_pixel_this[1] + self.pix_per_wavel_bin
            ) - pixel_floor_end_x

            y0 = int(starting_pixel_this[0])
            y1 = int(starting_pixel_this[0] + self.pix_spectral_width)
            footprint_this[y0:y1, int(pixel_ceil_start_x) : int(pixel_floor_end_x)] = (
                1.0
            )
            footprint_this[y0:y1, int(pixel_ceil_start_x) - 1] = pixel_frac_start_x
            footprint_this[y0:y1, int(pixel_floor_end_x)] = pixel_frac_end_x

            footprint_cube[wavel_bin_num, :, :] = footprint_this
            logging.info(
                "Wavelength bin %s dispersion footprint is %s pixels",
                wavel_bin_num,
                footprint_this.sum(),
            )

        footprint_sum = np.sum(footprint_cube, axis=0)
        logging.info("Total dispersion footprint is %s pixels", footprint_sum.sum())

        return footprint_cube

    def wavel_at_x(self, x_pix: np.ndarray, bin_edges: u.Quantity) -> np.ndarray:
        """
        Wavelength (um) at continuous detector x coordinate(s) along the trace; NaN
        off the trace.

        Pixel k spans x in [k, k + 1) (as in ``make_footprint``), and bin i spans
        [x0 + i * pix_per_wavel_bin, x0 + (i + 1) * pix_per_wavel_bin). Within a bin,
        the wavelength is interpolated linearly in log(lambda), consistent with the
        geometric bin edges.
        """
        edges_um = np.asarray(bin_edges.to_value(u.um), dtype=float)
        x_edges = self.starting_pixel[1] + self.pix_per_wavel_bin * np.arange(
            edges_um.size
        )
        x_pix = np.asarray(x_pix, dtype=float)
        log_wavel = np.interp(
            x_pix, x_edges, np.log(edges_um), left=np.nan, right=np.nan
        )
        return np.exp(log_wavel)

    def write_wavel_solution(
        self,
        config: ConfigLike,
        bin_edges: u.Quantity | None = None,
        file_name: str | None = None,
    ) -> str | None:
        """
        Write the wavelength solution of this dispersion law to FITS.

        Args:
            config: pipeline config; gives the detector size, and (unless given)
                the bin edges and [extraction] file_name_wavel_solution_write
            bin_edges: wavelength-bin edges (n_bins + 1,); default: from
                [detector] spec_res and [wavelength_range] (same as the channels)
            file_name: output path; default: [extraction]
                file_name_wavel_solution_write

        HDUs:
            PRIMARY    header only: trace geometry, spectral resolution, ranges
            BINS       table, one row per wavelength bin: BIN, WAVEL_LO_UM,
                       WAVEL_CEN_UM, WAVEL_HI_UM, WIDTH_UM, X_START_PIX, X_END_PIX
                       (continuous x; pixel k spans [k, k + 1)), Y_START_PIX,
                       Y_END_PIX (rows, end-exclusive), NPIX (illuminated pixels)
            WAVEL_COL  (n_pix_x,) wavelength (um) at the centre of each detector
                       column; NaN off the trace
            WAVEL_MAP  (n_pix_y, n_pix_x) wavelength (um) of each illuminated pixel
                       (column-centre value on the trace rows); NaN elsewhere

        Returns the path written, or None if no file name is configured.
        """
        if file_name is None:
            file_name = _config_get(
                config, "extraction", "file_name_wavel_solution_write"
            )
        if not file_name or not str(file_name).strip():
            logger.info(
                "No [extraction] file_name_wavel_solution_write; "
                "wavelength solution not written"
            )
            return None
        file_name = str(file_name).strip()

        if bin_edges is None:
            bin_edges = wavel_bin_edges_from_config(config)
        edges_um = np.asarray(bin_edges.to_value(u.um), dtype=float)
        n_bins = edges_um.size - 1
        n_pix_y, n_pix_x = detector_shape_from_config(config)
        self._check_trace_fits(n_pix_y, n_pix_x, n_bins)

        y_start = int(self.starting_pixel[0])
        y_end = y_start + self.pix_spectral_width
        x_start = self.starting_pixel[1] + self.pix_per_wavel_bin * np.arange(n_bins)
        x_end = x_start + self.pix_per_wavel_bin
        centers_um = np.sqrt(edges_um[:-1] * edges_um[1:])
        npix_per_bin = self.pix_per_wavel_bin * self.pix_spectral_width

        bins_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="BIN", format="J", array=np.arange(n_bins)),
                fits.Column(
                    name="WAVEL_LO_UM", format="D", unit="um", array=edges_um[:-1]
                ),
                fits.Column(
                    name="WAVEL_CEN_UM", format="D", unit="um", array=centers_um
                ),
                fits.Column(
                    name="WAVEL_HI_UM", format="D", unit="um", array=edges_um[1:]
                ),
                fits.Column(
                    name="WIDTH_UM",
                    format="D",
                    unit="um",
                    array=edges_um[1:] - edges_um[:-1],
                ),
                fits.Column(name="X_START_PIX", format="D", unit="pix", array=x_start),
                fits.Column(name="X_END_PIX", format="D", unit="pix", array=x_end),
                fits.Column(
                    name="Y_START_PIX",
                    format="J",
                    unit="pix",
                    array=np.full(n_bins, y_start),
                ),
                fits.Column(
                    name="Y_END_PIX",
                    format="J",
                    unit="pix",
                    array=np.full(n_bins, y_end),
                ),
                fits.Column(
                    name="NPIX",
                    format="D",
                    unit="pix",
                    array=np.full(n_bins, npix_per_bin),
                ),
            ],
            name="BINS",
        )
        bins_hdu.header["COMMENT"] = (
            "Continuous x: pixel k spans [k, k+1); Y_END_PIX is end-exclusive"
        )

        wavel_col = self.wavel_at_x(np.arange(n_pix_x) + 0.5, bin_edges)
        col_hdu = fits.ImageHDU(data=wavel_col, name="WAVEL_COL")
        col_hdu.header["BUNIT"] = "um"
        col_hdu.header["QUANTITY"] = "Wavelength at each detector column centre"

        wavel_map = np.full((n_pix_y, n_pix_x), np.nan)
        wavel_map[y_start:y_end, :] = wavel_col
        map_hdu = fits.ImageHDU(data=wavel_map, name="WAVEL_MAP")
        map_hdu.header["BUNIT"] = "um"
        map_hdu.header["QUANTITY"] = "Wavelength of each illuminated pixel"

        primary = fits.PrimaryHDU()
        header = primary.header
        header["SPECRES"] = (
            float(config["detector"]["spec_res"]),
            "Spectral resolution lambda/dlambda",
        )
        header["NBINS"] = (n_bins, "Number of wavelength bins")
        header["LAMMIN"] = (edges_um[0], "[um] Lower edge of the first bin")
        header["LAMMAX"] = (edges_um[-1], "[um] Upper edge of the last bin")
        header["PIXPBIN"] = (self.pix_per_wavel_bin, "Pixels per wavelength bin (x)")
        header["TRWIDTH"] = (self.pix_spectral_width, "Trace width in rows (y)")
        header["TRACEY0"] = (self.starting_pixel[0], "Trace start row (y)")
        header["TRACEX0"] = (self.starting_pixel[1], "Trace start x (continuous)")
        header["NPIXY"] = (n_pix_y, "Detector rows")
        header["NPIXX"] = (n_pix_x, "Detector columns")
        header["COMMENT"] = "Wavelength interpolated linearly in log(lambda) along x"

        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary, bins_hdu, col_hdu, map_hdu]).writeto(
            file_name, overwrite=True
        )
        logger.info("Wrote wavelength solution (%d bins) to %s", n_bins, file_name)
        return file_name

    @staticmethod
    def n_pix_per_bin(footprint_cube: np.ndarray) -> u.Quantity:
        """Pixels (or fractional pixels) illuminated in each wavelength bin."""
        footprint_pixel_count = np.sum(footprint_cube, axis=(1, 2))
        if hasattr(footprint_pixel_count, "unit"):
            return footprint_pixel_count.to(u.pix)
        return footprint_pixel_count * u.pix

    @staticmethod
    def footprint_image(footprint: np.ndarray) -> np.ndarray:
        """Footprint summed over wavelength bins, as a 2D (n_pix_y, n_pix_x) image."""
        return np.sum(footprint, axis=0)

    @staticmethod
    def weighted_sum_per_bin(footprint: np.ndarray, image_2d: np.ndarray) -> np.ndarray:
        """Sum of footprint * image_2d within each wavelength bin."""
        return np.tensordot(footprint, image_2d, axes=([1, 2], [0, 1]))


@pipeline_stage(depends_on=("pass_from_aperture_to_detector",))
def disperse_astro_signals(
    config: ConfigLike,
    prop_dict: dict,
    output_channels: dict,
    source_names: Sequence[str],
    sources_to_include: Sequence[str],
    background_source_names: Sequence[str] = (),
    plot: bool = False,
) -> None:
    """
    Disperse post-aperture flux onto each channel detector via DispersionLaw,
    then apply detector QE.
    """
    # ## TODO: allow different dispersion laws for each output channel
    dispersion_law = DispersionLaw(config)
    # write the wavelength soln to file
    dispersion_law.write_wavel_solution(config)

    source_names = list(source_names) + list(background_source_names)

    for output_channel in output_channels.values():
        if output_channel.detector is None:
            output_channel.detector = Detector(
                config=config, num_wavel_bins=len(output_channel.bin_centers)
            )
        logging.info("Dispersing signals on detector for %s", output_channel.name)

        # return footprint in shape (n_bins, n_pix_y, n_pix_x)
        footprint_cube = dispersion_law.make_footprint(
            output_channel.detector.n_pix_y,
            output_channel.detector.n_pix_x,
            output_channel.detector.num_wavel_bins,
        )
        output_channel.detector.set_footprint(footprint_cube)

        if plot:  # pragma: no cover
            plot_detector_footprint(
                DispersionLaw.footprint_image(footprint_cube),
                config,
                str(config["dirs"]["save_s2n_data_unique_dir"])
                + f"footprint_bool_{output_channel.name}.png",
            )

        n_int_this_angle = int(config["observation"]["N_int_per_angle"])
        logging.info("Number of frames at this angle position: %d", n_int_this_angle)

        n_pix_per_wavel_bin = DispersionLaw.n_pix_per_bin(footprint_cube)

        for source_name in source_names:
            if source_name not in prop_dict:
                continue
            source_prop = prop_dict[source_name]
            flux_cube = source_prop["flux_cube_post_screen_post_aperture_ph_sec_um"][
                output_channel.name
            ]

            flux_astro_1d_ph_sec_um = np.sum(flux_cube, axis=(1, 2))
            flux_unit = (
                flux_astro_1d_ph_sec_um.unit
                if hasattr(flux_astro_1d_ph_sec_um, "unit")
                else u.ph / (u.um * u.s)
            )
            wavel_bins = output_channel.bin_centers
            wavel_pts = source_prop["wavel"]
            flux_astro_1d_interpolated_ph_sec_um = (
                np.interp(
                    x=(
                        wavel_bins.value if hasattr(wavel_bins, "value") else wavel_bins
                    ),
                    xp=(wavel_pts.value if hasattr(wavel_pts, "value") else wavel_pts),
                    fp=(
                        flux_astro_1d_ph_sec_um.value
                        if hasattr(flux_astro_1d_ph_sec_um, "value")
                        else flux_astro_1d_ph_sec_um
                    ),
                )
                * flux_unit
            )
            flux_astro_1d_interpolated_ph_sec_wavel_bin = (
                flux_astro_1d_interpolated_ph_sec_um * output_channel.bin_widths
            )
            flux_astro_1d_interpolated_ph_sec_pixel = (
                flux_astro_1d_interpolated_ph_sec_wavel_bin / n_pix_per_wavel_bin
            )

            # detector QE
            det = output_channel.detector
            flux_astro_1d_interpolated_ph_sec_pixel = det.apply_qe(
                flux_astro_1d_interpolated_ph_sec_pixel
            )
            flux_astro_1d_interpolated_ph_sec_wavel_bin = det.apply_qe(
                flux_astro_1d_interpolated_ph_sec_wavel_bin
            )
            flux_astro_1d_interpolated_ph_sec_um = det.apply_qe(
                flux_astro_1d_interpolated_ph_sec_um
            )

            per_um = flux_astro_1d_interpolated_ph_sec_um.decompose()
            per_bin = flux_astro_1d_interpolated_ph_sec_wavel_bin.decompose()
            per_pixel = flux_astro_1d_interpolated_ph_sec_pixel.decompose()
            output_channel.astroph_signal[source_name] = {
                "wavel": output_channel.bin_centers,
                "flux_astro_1d_interpolated_ph_sec_um": per_um,
                "flux_astro_1d_interpolated_ph_sec_wavel_bin": per_bin,
                "flux_astro_1d_interpolated_ph_sec_pixel": per_pixel,
                "n_pix_per_wavel_bin": n_pix_per_wavel_bin.decompose(),
            }
    import ipdb
    ipdb.set_trace()

    if plot:  # pragma: no cover

        # save plot and csv of photon counts per pixel
        for output_channel in output_channels.values():
            if not output_channel.astroph_signal:
                continue

            wavel_um = np.asarray(
                (
                    output_channel.bin_centers.value
                    if hasattr(output_channel.bin_centers, "value")
                    else output_channel.bin_centers
                ),
                dtype=float,
            )
            plot_source_names = list(sources_to_include) + list(background_source_names)
            ref_source = next(
                (
                    name
                    for name in plot_source_names
                    if name in output_channel.astroph_signal
                ),
                None,
            )
            n_pix_vals = None
            if ref_source is not None:
                n_pix_qty = output_channel.astroph_signal[ref_source][
                    "n_pix_per_wavel_bin"
                ]
                n_pix_vals = np.asarray(
                    n_pix_qty.value if hasattr(n_pix_qty, "value") else n_pix_qty,
                    dtype=float,
                )

            df_signals = pd.DataFrame()
            signals_by_source = {}
            cumulative_signal = None
            y_unit = u.ph / (u.s * u.pix)
            source_name = None
            for source_name in plot_source_names:
                if source_name not in output_channel.astroph_signal:
                    continue
                sig = output_channel.astroph_signal[source_name]
                y_col = sig["flux_astro_1d_interpolated_ph_sec_pixel"]
                y_vals = np.asarray(
                    y_col.value if hasattr(y_col, "value") else y_col,
                    dtype=float,
                )
                y_unit = y_col.unit if hasattr(y_col, "unit") else u.ph / (u.s * u.pix)
                if cumulative_signal is None:
                    cumulative_signal = np.zeros_like(y_vals, dtype=float)
                signals_by_source[source_name] = y_vals
                cumulative_signal = np.add(cumulative_signal, y_vals)

                df_signals["wavel_um"] = wavel_um
                df_signals[f"{source_name}_ph_sec_pixel"] = y_vals
                df_signals["n_pix_per_wavel_bin"] = n_pix_vals

            file_name_csv = (
                str(config["dirs"]["save_s2n_data_unique_dir"])
                + f"astro_ph_sec_pixel_{output_channel.name}.csv"
            )
            df_signals["cumulative_ph_sec_pixel"] = cumulative_signal
            df_signals["cumulative_signal_ph_pix_10min"] = 600.0 * cumulative_signal
            df_signals.to_csv(file_name_csv, index=False)
            logging.info(
                "Saved astrophysical photon rate per pixel table for %s and %s to %s",
                output_channel.name,
                source_name,
                file_name_csv,
            )

            file_name_plot = (
                str(config["dirs"]["save_s2n_data_unique_dir"])
                + f"astro_ph_sec_pixel_{output_channel.name}.pdf"
            )
            plot_astro_ph_sec_pixel(
                output_channel.name,
                output_channel.bin_edges,
                output_channel.bin_centers,
                signals_by_source,
                cumulative_signal,
                y_unit,
                config,
                file_name_plot,
            )
        ipdb.set_trace()
