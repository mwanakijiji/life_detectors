"""Telescope aperture and the post-aperture beam (instrument optical backgrounds)."""

import logging
from typing import List, Sequence

import numpy as np
import astropy.units as u

from ...pipeline_registry import pipeline_stage
from ...utils.helpers.spectra import compute_collecting_area_m2
from ...viz.aperture import (
    plot_integrated_flux_by_output,
    plot_poster_post_aperture_flux,
)
from .effects import _config_get


def _enabled_background_names(config) -> List[str]:
    raw = _config_get(config, "instrument_backgrounds", "enabled", "") or ""
    return [name.strip() for name in str(raw).split(",") if name.strip()]


class InstrumentBackground:
    """Optical backgrounds added after the aperture, before the detector."""

    name: str = "instrument_background"

    def flux_ph_sec_um(self, wavel: u.Quantity, config) -> u.Quantity:
        """Return a 1D spectrum (ph / s / um) to inject into the post-aperture beam."""
        raise NotImplementedError


class TelescopeThermalBackground(InstrumentBackground):
    name = "telescope_thermal"

    def flux_ph_sec_um(self, wavel: u.Quantity, config) -> u.Quantity:
        # Placeholder: zero thermal background until a physical model is filled in.
        return np.zeros(np.shape(wavel)) * u.ph / (u.s * u.um)


BACKGROUND_REGISTRY: dict[str, type[InstrumentBackground]] = {
    "telescope_thermal": TelescopeThermalBackground,
}


@pipeline_stage(depends_on=("pass_through_transmission_screens",))
def pass_through_aperture(
    config,
    sources_astroph: dict,
    sources_to_include: Sequence[str],
    prop_dict: dict,
    plot: bool = False,
):
    # pass each astrophysical source through the telescope aperture, and update prop_dict with the propagated terms
    # photons/sec/m^2 -> photons/sec

    transmission_screen_order = [
        "output_1_bright",
        "output_2_bright",
        "output_3_dark",
        "output_4_dark",
    ]
    collecting_area = compute_collecting_area_m2(config) * u.m**2

    # telescope throughput
    eta_t = float(config["telescope"]["eta_t"])

    # apply throughput and collecting area to the flux
    for source_name, source_val in sources_astroph.items():

        post_aperture_flux_by_output = {
            output_name: eta_t
            * np.multiply(
                collecting_area,
                source_val["flux_cube_post_screen_ph_sec_m2_um"][output_name],
            )
            for output_name in transmission_screen_order
        }

        # note telescope throughput is incorporated at the stage of passing through the aperture
        ## ## TODO; make a separate module for throughput, and add other terms (telescope background, etc.)
        prop_dict[source_name] = {
            "wavel": source_val["wavel"],
            "flux_cube_post_screen_pre_aperture_ph_sec_m2_um": source_val[
                "flux_cube_post_screen_ph_sec_m2_um"
            ],
            "flux_cube_post_screen_post_aperture_ph_sec_um": post_aperture_flux_by_output,  # includes chop signal if enabled
        }

    # overplot all the sources
    if plot:  # pragma: no cover
        save_dir = str(config["dirs"]["save_s2n_data_unique_dir"])

        def _flux_by_source_for_cube(cube_key):
            flux_by_source = {}
            for source_name, source_val in prop_dict.items():
                if source_name not in sources_to_include:
                    continue
                flux_integrated = np.sum(
                    source_val[cube_key][output_name], axis=(1, 2)
                )
                flux_by_source[source_name] = (source_val["wavel"], flux_integrated)
            return flux_by_source

        for output_name in transmission_screen_order:
            plot_integrated_flux_by_output(
                _flux_by_source_for_cube(
                    "flux_cube_post_screen_pre_aperture_ph_sec_m2_um"
                ),
                output_name,
                "Post-screen, pre-aperture flux (all sources)",
                config,
                save_dir
                + f"flux_all_sources_post_screen_pre_aperture_{output_name}.png",
            )
            plot_integrated_flux_by_output(
                _flux_by_source_for_cube(
                    "flux_cube_post_screen_post_aperture_ph_sec_um"
                ),
                output_name,
                "Post-screen, post-aperture flux (all sources)",
                config,
                save_dir
                + f"flux_all_sources_post_screen_post_aperture_{output_name}.png",
            )
            plot_poster_post_aperture_flux(
                _flux_by_source_for_cube(
                    "flux_cube_post_screen_post_aperture_ph_sec_um"
                ),
                output_name,
                config,
                save_dir + f"poster_post_aperture_flux_{output_name}.pdf",
            )

    logging.info(f"Passed astrophysical flux through telescope aperture...")

    return


@pipeline_stage(depends_on=("pass_through_aperture",))
def pass_from_aperture_to_detector(
    config,
    prop_dict: dict,
    output_channels: dict,
    instrument_backgrounds: Sequence = (),
    plot: bool = False,
):
    """
    Inject enabled ``InstrumentBackground`` spectra into the post-aperture
    beam (still in photon units). Detector QE is applied later in
    ``disperse_astro_signals_on_detector``.
    """
    if not prop_dict:
        logging.warning(
            "pass_from_aperture_to_detector: prop_dict is empty; nothing to do"
        )
        return

    ref_name = next(iter(prop_dict))
    ref = prop_dict[ref_name]
    wavel = ref["wavel"]
    output_names = list(output_channels.keys())

    # Add each configured background as its own prop_dict source (spatially uniform)
    for bg in instrument_backgrounds:
        spectrum = bg.flux_ph_sec_um(wavel, config)
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
        prop_dict[bg.name] = {
            "wavel": wavel,
            "flux_cube_post_screen_post_aperture_ph_sec_um": post_aperture_by_output,
        }
        logging.info(
            "Added instrument background %r into post-aperture beam", bg.name
        )

    if plot:
        pass

    return
