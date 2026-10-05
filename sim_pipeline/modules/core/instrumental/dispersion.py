"""Optical dispersion: map wavelength bins onto a pixel grid.

Independent of detector QE/noise.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import astropy.units as u
import numpy as np
import pandas as pd

from ...pipeline_registry import pipeline_stage
from ...viz.aperture import plot_astro_ph_sec_pixel
from ...viz.detector import plot_detector_footprint

logger = logging.getLogger(__name__)


class DispersionLaw:
    """Maps spectral bins to detector-pixel footprints.

    Physically this is a spectrograph / illumination law. It needs a pixel-grid
    geometry to realize the map, but it does not own QE, gain, or noise.
    """

    def __init__(
        self,
        config: Dict,
        *,
        starting_pixel: Sequence[float] | None = None,
    ):
        self.config = config
        self.pix_per_wavel_bin = float(config["detector"]["pix_per_wavel_bin"])
        self.pix_spectral_width = int(config["detector"]["pix_spectral_width"])
        # lower-left corner of the first wavelength-bin footprint (y, x)
        self.starting_pixel = np.asarray(
            starting_pixel if starting_pixel is not None else (100.0, 300.0),
            dtype=float,
        )

    def make_footprint(self, side_length_pix: int, num_wavel_bins: int) -> np.ndarray:
        """Return the wavelength-bin footprint cube on the detector.

        Shape is (n_bins, n_pix, n_pix). Values are 1 for fully covered pixels and
        fractional coverage on edges.
        """

        footprint_cube = np.full(
            (num_wavel_bins, side_length_pix, side_length_pix), 0.0, dtype=float
        )

        for wavel_bin_num in range(num_wavel_bins):
            # assumes horizontal spectra
            footprint_this = np.full(
                (side_length_pix, side_length_pix), 0.0, dtype=float
            )
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

    @staticmethod
    def n_pix_per_bin(footprint_cube: np.ndarray) -> u.Quantity:
        """Pixels (or fractional pixels) illuminated in each wavelength bin."""
        footprint_pixel_count = np.sum(footprint_cube, axis=(1, 2))
        if hasattr(footprint_pixel_count, "unit"):
            return footprint_pixel_count.to(u.pix)
        return footprint_pixel_count * u.pix

    @staticmethod
    def footprint_image(footprint: np.ndarray) -> np.ndarray:
        """Footprint summed over wavelength bins, as a 2D (n_pix, n_pix) image."""
        return np.sum(footprint, axis=0)

    @staticmethod
    def weighted_sum_per_bin(footprint: np.ndarray, image_2d: np.ndarray) -> np.ndarray:
        """Sum of footprint * image_2d within each wavelength bin."""
        return np.tensordot(footprint, image_2d, axes=([1, 2], [0, 1]))


@pipeline_stage(depends_on=("pass_from_aperture_to_detector",))
def disperse_astro_signals_on_detector(
    config,
    prop_dict: dict,
    output_channels: dict,
    source_names: Sequence[str],
    sources_to_include: Sequence[str],
    background_source_names: Sequence[str] = (),
    plot: bool = False,
):
    """
    Disperse post-aperture flux onto each channel detector via DispersionLaw,
    then apply detector QE.
    """
    # imported here: detector.py imports DispersionLaw from this module
    from .detector import Detector

    ## ## TODO: allow different dispersion laws for each output channel
    dispersion_law = DispersionLaw(config)

    source_names = list(source_names) + list(background_source_names)

    for output_channel in output_channels.values():
        if output_channel.detector is None:
            output_channel.detector = Detector(
                config=config, num_wavel_bins=len(output_channel.bin_centers)
            )
        logging.info(f"Dispersing signals on detector for {output_channel.name}")

        footprint_cube = dispersion_law.make_footprint(
            output_channel.detector.side_length_pix,
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
        logging.info(
            f"Number of frames at this angle position: {n_int_this_angle:d}"
        )

        n_pix_per_wavel_bin = DispersionLaw.n_pix_per_bin(footprint_cube)

        for source_name in source_names:
            if source_name not in prop_dict:
                continue
            source_prop = prop_dict[source_name]
            flux_cube = source_prop[
                "flux_cube_post_screen_post_aperture_ph_sec_um"
            ][output_channel.name]

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
                        wavel_bins.value
                        if hasattr(wavel_bins, "value")
                        else wavel_bins
                    ),
                    xp=(
                        wavel_pts.value
                        if hasattr(wavel_pts, "value")
                        else wavel_pts
                    ),
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

            # Detector QE (intrinsic to the sensor)
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

            output_channel.astroph_signal[source_name] = {
                "wavel": output_channel.bin_centers,
                "flux_astro_1d_interpolated_ph_sec_um": flux_astro_1d_interpolated_ph_sec_um.decompose(),
                "flux_astro_1d_interpolated_ph_sec_wavel_bin": flux_astro_1d_interpolated_ph_sec_wavel_bin.decompose(),
                "flux_astro_1d_interpolated_ph_sec_pixel": flux_astro_1d_interpolated_ph_sec_pixel.decompose(),
                "n_pix_per_wavel_bin": n_pix_per_wavel_bin.decompose(),
            }

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
                y_unit = (
                    y_col.unit if hasattr(y_col, "unit") else u.ph / (u.s * u.pix)
                )
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

    return
