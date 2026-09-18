import logging

import numpy as np
import astropy.units as u
import pandas as pd

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.spectra import compute_collecting_area_m2
from ...viz.aperture import (
    plot_astro_ph_sec_pixel,
    plot_integrated_flux_by_output,
    plot_poster_post_aperture_flux,
)
from ...viz.detector import plot_detector_footprint
from .detector import Detector
from .dispersion import DispersionLaw

class TransferMixin:
    @pipeline_stage(depends_on=("pass_from_aperture_to_detector",))
    def disperse_astro_signals_on_detector(self, plot: bool = False):
        '''
        Disperse post-aperture flux onto each channel detector via DispersionLaw,
        then apply detector QE.
        '''
        ## ## TODO: allow different dispersion laws for each output channel
        dispersion_law = DispersionLaw(self.config)

        source_names = list(self.sources_astroph.keys()) + list(
            getattr(self, "background_source_names", [])
        )

        for output_channel in self.output_channels.values():
            if output_channel.detector is None:
                output_channel.detector = Detector(
                    config=self.config, num_wavel_bins=len(output_channel.bin_centers)
                )
            logging.info(f"Dispersing signals on detector for {output_channel.name}")

            footprint_cube = dispersion_law.make_footprint(
                output_channel.detector.side_length_pix,
                output_channel.detector.num_wavel_bins,
            )
            output_channel.detector.set_footprint(footprint_cube)

            if plot:  # pragma: no cover
                plot_detector_footprint(
                    np.sum(footprint_cube, axis=0),
                    self.config,
                    str(self.config["dirs"]["save_s2n_data_unique_dir"])
                    + f"footprint_bool_{output_channel.name}.png",
                )

            n_int_this_angle = int(self.config["observation"]["N_int_per_angle"])
            logging.info(f"Number of frames at this angle position: {n_int_this_angle:d}")

            n_pix_per_wavel_bin = DispersionLaw.n_pix_per_bin(footprint_cube)

            for source_name in source_names:
                if source_name not in self.prop_dict:
                    continue
                source_prop = self.prop_dict[source_name]
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
                        x=wavel_bins.value if hasattr(wavel_bins, "value") else wavel_bins,
                        xp=wavel_pts.value if hasattr(wavel_pts, "value") else wavel_pts,
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
            for output_channel in self.output_channels.values():
                if not output_channel.astroph_signal:
                    continue

                wavel_um = np.asarray(
                    output_channel.bin_centers.value
                    if hasattr(output_channel.bin_centers, "value")
                    else output_channel.bin_centers,
                    dtype=float,
                )
                plot_source_names = list(self.sources_to_include) + list(
                    getattr(self, "background_source_names", [])
                )
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
                    str(self.config["dirs"]["save_s2n_data_unique_dir"])
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
                    str(self.config["dirs"]["save_s2n_data_unique_dir"])
                    + f"astro_ph_sec_pixel_{output_channel.name}.pdf"
                )
                plot_astro_ph_sec_pixel(
                    output_channel.name,
                    output_channel.bin_edges,
                    output_channel.bin_centers,
                    signals_by_source,
                    cumulative_signal,
                    y_unit,
                    self.config,
                    file_name_plot,
                )

        return



    @pipeline_stage(depends_on=("pass_through_transmission_screens",))
    def pass_through_aperture(self, plot: bool = False):
        # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
        # photons/sec/m^2 -> photons/sec

        transmission_screen_order = ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark']
        collecting_area = compute_collecting_area_m2(self.config) * u.m**2

        # telescope throughput
        eta_t = float(self.config['telescope']['eta_t'])

        # apply throughput and collecting area to the flux
        for source_name, source_val in self.sources_astroph.items():

            post_aperture_flux_by_output = {
                output_name: eta_t * np.multiply(collecting_area, source_val['flux_cube_post_screen_ph_sec_m2_um'][output_name])
                for output_name in transmission_screen_order
            }

            # note telescope throughput is incorporated at the stage of passing through the aperture
            ## ## TODO; make a separate module for throughput, and add other terms (telescope background, etc.)
            self.prop_dict[source_name] = {
                'wavel': source_val['wavel'],
                'flux_cube_post_screen_pre_aperture_ph_sec_m2_um': source_val['flux_cube_post_screen_ph_sec_m2_um'],
                'flux_cube_post_screen_post_aperture_ph_sec_um': post_aperture_flux_by_output, # includes chop signal if enabled
            }


        # overplot all the sources
        if plot: # pragma: no cover
            save_dir = str(self.config['dirs']['save_s2n_data_unique_dir'])

            def _flux_by_source_for_cube(cube_key):
                flux_by_source = {}
                for source_name, source_val in self.prop_dict.items():
                    if source_name not in self.sources_to_include:
                        continue
                    flux_integrated = np.sum(
                        source_val[cube_key][output_name], axis=(1, 2)
                    )
                    flux_by_source[source_name] = (source_val['wavel'], flux_integrated)
                return flux_by_source

            for output_name in transmission_screen_order:
                plot_integrated_flux_by_output(
                    _flux_by_source_for_cube('flux_cube_post_screen_pre_aperture_ph_sec_m2_um'),
                    output_name,
                    'Post-screen, pre-aperture flux (all sources)',
                    self.config,
                    save_dir + f'flux_all_sources_post_screen_pre_aperture_{output_name}.png',
                )
                plot_integrated_flux_by_output(
                    _flux_by_source_for_cube('flux_cube_post_screen_post_aperture_ph_sec_um'),
                    output_name,
                    'Post-screen, post-aperture flux (all sources)',
                    self.config,
                    save_dir + f'flux_all_sources_post_screen_post_aperture_{output_name}.png',
                )
                plot_poster_post_aperture_flux(
                    _flux_by_source_for_cube('flux_cube_post_screen_post_aperture_ph_sec_um'),
                    output_name,
                    self.config,
                    save_dir + f"poster_post_aperture_flux_{output_name}.pdf",
                )

        logging.info(f'Passed astrophysical flux through telescope aperture...')

        return

    @pipeline_stage(depends_on=("pass_through_aperture",))
    def pass_from_aperture_to_detector(self, plot: bool = False):
        """
        Inject enabled ``InstrumentBackground`` spectra into the post-aperture
        beam (still in photon units). Detector QE is applied later in
        ``disperse_astro_signals_on_detector``.
        """
        if not self.prop_dict:
            logging.warning(
                "pass_from_aperture_to_detector: prop_dict is empty; nothing to do"
            )
            return

        ref_name = next(iter(self.prop_dict))
        ref = self.prop_dict[ref_name]
        wavel = ref["wavel"]
        output_names = list(self.output_channels.keys())

        # Add each configured background as its own prop_dict source (spatially uniform)
        for bg in getattr(self, "instrument_backgrounds", []):
            spectrum = bg.flux_ph_sec_um(wavel, self.config)
            post_aperture_by_output = {}
            for output_name in output_names:
                ref_cube = ref["flux_cube_post_screen_post_aperture_ph_sec_um"][
                    output_name
                ]
                n_y, n_x = ref_cube.shape[1], ref_cube.shape[2]
                n_spatial = n_y * n_x
                # Conserve total ph/s/um when later summing over the sky plane
                per_pixel = spectrum / n_spatial
                spatial = np.ones((n_y, n_x))
                post_aperture_by_output[output_name] = (
                    per_pixel[:, np.newaxis, np.newaxis] * spatial
                )
            self.prop_dict[bg.name] = {
                "wavel": wavel,
                "flux_cube_post_screen_post_aperture_ph_sec_um": post_aperture_by_output,
            }
            logging.info(
                "Added instrument background %r into post-aperture beam", bg.name
            )

        if plot:
            pass

        return
