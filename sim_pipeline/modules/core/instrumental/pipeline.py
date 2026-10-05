from typing import Dict, List

import logging

import numpy as np
import astropy.units as u

from . import aperture, dispersion, effects, tables, transmission
from .aperture import BACKGROUND_REGISTRY, InstrumentBackground, _enabled_background_names
from .channels import OutputChannel
from .effects import DetectorEffect, build_detector_effects

logger = logging.getLogger(__name__)


class InstrumentDepTerms:
    # Provides the effects of the instrument (including astro flux passed through the telescope aperture).
    # Holds the pipeline state; each stage method delegates to a module-level function.

    def __init__(self, config: Dict, sources_astroph: dict, sources_to_include: list):
        """
        Args:
            config: Configuration dictionary
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'pre_screen_astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
            sources_to_include: List of sources to actuallyinclude in the S/N calculation (and plots of incident fluxes)
        """

        self.config = config
        self.sources_astroph = sources_astroph  # all sources of astrophysical flux, as are incident on the instrument
        self.sources_to_include = sources_to_include

        # initialize dict to carry propagated astrophysical terms (i.e., intensity levels on the detector, after instrument effects)
        self.prop_dict = {}

        # Optional detector systematics, applied in order to each readout's 2D frames
        self.detector_effects: List[DetectorEffect] = build_detector_effects(config)

        # Instrument optical backgrounds (post-aperture → detector)
        self.instrument_backgrounds: List[InstrumentBackground] = []
        self.background_source_names: List[str] = []
        for bg_name in _enabled_background_names(config):
            if bg_name not in BACKGROUND_REGISTRY:
                raise ValueError(
                    f"Instrument background {bg_name!r} not found in registry"
                )
            bg = BACKGROUND_REGISTRY[bg_name]()
            self.instrument_backgrounds.append(bg)
            self.background_source_names.append(bg.name)

        # initialize output channels
        self.output_channels = {
            name: OutputChannel(name=name)
            for name in [
                "output_1_bright",
                "output_2_bright",
                "output_3_dark",
                "output_4_dark",
            ]
        }

        # for each output channel, set the detection wavelength bins (same for all channels for now)
        R = float(
            self.config["detector"]["spec_res"]
        )  # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation lambda_i = lambda_{0} * (1 + 1/R)**i
        lambda_min, lambda_max = (
            float(self.config["wavelength_range"]["min"]) * u.um,
            float(self.config["wavelength_range"]["max"]) * u.um,
        )
        # number of bins that fit fully in [lmin, lmax]
        n_bins = int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / R)))

        # geometric bin edges and centers
        bin_edges = lambda_min * (1.0 + 1.0 / R) ** np.arange(n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = bin_edges[1:] - bin_edges[:-1]  # removed units for plotting
        for output_channel in self.output_channels.values():
            output_channel.spec_R = R
            output_channel.bin_edges = bin_edges
            output_channel.bin_centers = bin_centers
            output_channel.bin_widths = bin_widths

    def generate_instrument_transmission(self, *args, **kwargs):
        return transmission.generate_instrument_transmission(
            self.config, *args, **kwargs
        )

    def pass_through_transmission_screens(self, *args, **kwargs):
        return transmission.pass_through_transmission_screens(
            self.config, self.sources_astroph, self.sources_to_include, *args, **kwargs
        )

    def pass_through_aperture(self, plot: bool = False):
        aperture.pass_through_aperture(
            self.config, self.sources_astroph, self.sources_to_include, self.prop_dict, plot=plot
        )

    def pass_from_aperture_to_detector(self, plot: bool = False):
        aperture.pass_from_aperture_to_detector(
            self.config, self.prop_dict, self.output_channels, self.instrument_backgrounds, plot=plot
        )

    def disperse_astro_signals_on_detector(self, plot: bool = False):
        dispersion.disperse_astro_signals_on_detector(
            self.config,
            self.prop_dict,
            self.output_channels,
            source_names=list(self.sources_astroph.keys()),
            sources_to_include=self.sources_to_include,
            background_source_names=self.background_source_names,
            plot=plot,
        )

    def apply_detector_effects(self):
        effects.apply_detector_effects(
            self.output_channels,
            self.detector_effects,
            t_frame=float(self.config["observation"]["t_int_frame"]) * u.second,
            n_readouts=int(self.config["observation"]["N_int_per_angle"]),
        )

    def combine_astro_and_instrum_signals(self, plot: bool = False):
        tables.combine_astro_and_instrum_signals(
            self.config, self.output_channels, self.sources_to_include, plot=plot
        )

    def chop_signal(self, plot: bool = False):
        self.post_chop_tables_by_dark_current = tables.chop_signal(
            self.output_channels, plot=plot
        )
