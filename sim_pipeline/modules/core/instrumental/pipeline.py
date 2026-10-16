"""InstrumentDepTerms: instrument-side state and the per-angle pipeline stages."""

import logging
from typing import Dict, List

import astropy.units as u
import numpy as np

from . import aperture, dispersion, effects, readouts, transmission
from .aperture import (
    BACKGROUND_REGISTRY,
    InstrumentBackground,
    _enabled_background_names,
)
from .channels import OutputChannel
from .effects import DetectorEffect, _config_get, build_detector_effects

logger = logging.getLogger(__name__)


class InstrumentDepTerms:
    # Provides the effects of the instrument (including astro flux passed through the
    # telescope aperture).
    # Holds the pipeline state; each stage method delegates to a module-level function.

    """Holds the instrument-side pipeline state (config, sources, output channels,
    detector effects, backgrounds); each stage method delegates to a module-level
    function so the stages can be tested and registered on their own.
    """

    def __init__(self, config: Dict, sources_astroph: dict, sources_to_include: list):
        """
        Args:
            config: Configuration dictionary
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>,
                'pre_screen_astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
            sources_to_include: List of sources to actuallyinclude in the S/N
                calculation (and plots of incident fluxes)
        """

        self.config = config
        # all sources of astrophysical flux, as are incident on the instrument
        self.sources_astroph = sources_astroph
        self.sources_to_include = sources_to_include

        # initialize dict to carry propagated astrophysical terms (i.e., intensity
        # levels on the detector, after instrument effects)
        self.prop_dict = {}

        # set by the readout/extraction stages at each angle
        self.readouts_file_name: str | None = None
        self.extraction_method: str | None = None
        self.extracted_spectra_adu: dict = {}
        self.extracted_errors_adu: dict = {}

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

        # for each output channel, set the detection wavelength bins (same for all
        # channels for now)
        R = float(
            self.config["detector"]["spec_res"]
        )  # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation
        # lambda_i = lambda_{0} * (1 + 1/R)**i
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
        """Transmission screens for each output; see
        transmission.generate_instrument_transmission.
        """
        return transmission.generate_instrument_transmission(
            self.config, *args, **kwargs
        )

    def pass_through_transmission_screens(self, *args, **kwargs):
        """Apply the transmission screens to the sky scene; see
        transmission.pass_through_transmission_screens.
        """
        return transmission.pass_through_transmission_screens(
            self.config, self.sources_astroph, self.sources_to_include, *args, **kwargs
        )

    def pass_through_aperture(self, plot: bool = False):
        """Stage: pass the screened sources through the telescope aperture."""
        aperture.pass_through_aperture(
            self.config,
            self.sources_astroph,
            self.sources_to_include,
            self.prop_dict,
            plot=plot,
        )

    def pass_from_aperture_to_detector(self, plot: bool = False):
        """Stage: add instrument backgrounds to the post-aperture beam."""
        aperture.pass_from_aperture_to_detector(
            self.config,
            self.prop_dict,
            self.output_channels,
            self.instrument_backgrounds,
            plot=plot,
        )

    def disperse_astro_signals_on_detector(self, plot: bool = False):
        """Stage: disperse the post-aperture signals onto each channel's detector."""
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
        """Stage: expose each readout and apply the enabled detector effects."""
        effects.apply_detector_effects(
            self.output_channels,
            self.detector_effects,
            t_frame=float(self.config["observation"]["t_int_frame"]) * u.second,
            n_readouts=int(self.config["observation"]["N_int_per_angle"]),
        )

    def write_detector_readouts(
        self, file_name: str, header_cards: dict | None = None, cutout: bool = True
    ):
        """Stage: write the readout frames to FITS (cutout
        by default); returns the path.
        """
        written = readouts.write_detector_readouts(
            self.output_channels,
            file_name,
            t_frame=float(self.config["observation"]["t_int_frame"]) * u.second,
            header_cards=header_cards,
            cutout=cutout,
        )
        # extraction reads the cutout file; a full-frame write is only an example
        if cutout:
            self.readouts_file_name = written
        return written

    def extract_spectra_from_readouts(self):
        """Stage: extract spectra from the readout FITS with the configured method."""
        method = str(
            _config_get(self.config, "extraction", "method", "sharp_birchall")
        ).strip()
        ref_detector = next(iter(self.output_channels.values())).detector
        self.extracted_spectra_adu = readouts.extract_spectra_from_readouts(
            self.output_channels,
            self.readouts_file_name,
            method=method,
            t_frame=float(self.config["observation"]["t_int_frame"]) * u.second,
            dc_rates=ref_detector.dark_current_rate_e_pix_sec.to_value(
                u.electron / (u.pix * u.second)
            ),
        )
        self.extraction_method = method
        self.extracted_errors_adu = {
            name: {
                "error": ch.extracted_error_adu,
                "vark": ch.extracted_vark_adu2,
            }
            for name, ch in self.output_channels.items()
            if ch.extracted_error_adu is not None
        }
