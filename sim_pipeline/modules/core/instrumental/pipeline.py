from typing import Dict, List

import logging

import numpy as np
import astropy.units as u

from .channels import OutputChannel
from .tables import TablesMixin
from .transfer import TransferMixin
from .transmission import TransmissionMixin


logger = logging.getLogger(__name__)


def _config_get(config, section: str, key: str, default=None):
    """Read a key from dict-like or ConfigParser configs."""
    if isinstance(config, dict):
        section_data = config.get(section, {})
        if isinstance(section_data, dict):
            return section_data.get(key, default)
        return default
    if hasattr(config, "has_section") and config.has_section(section):
        if config.has_option(section, key):
            return config.get(section, key)
    return default


def _enabled_effect_names(config) -> List[str]:
    raw = _config_get(config, "detector_systematics", "enabled", "") or ""
    return [name.strip() for name in str(raw).split(",") if name.strip()]


class DetectorEffect:
    """Base class for optional detector systematics applied per readout."""

    name: str = "detector_effect"

    def __init__(self, config):
        self.config = config

    def apply(self, channel, readout_index: int = 0) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class PersistenceEffect(DetectorEffect):
    name = "persistence"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class GainVariabilityEffect(DetectorEffect):
    name = "gain_variability"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class OneOverFNoiseEffect(DetectorEffect):
    name = "one_over_f"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class TransferFunctionEffect(DetectorEffect):
    name = "transfer_function"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class AgingPixelsEffect(DetectorEffect):
    name = "aging_pixels"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class CosmicRaysEffect(DetectorEffect):
    name = "cosmic_rays"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class HotPixelsEffect(DetectorEffect):
    name = "hot_pixels"

    def apply(self, channel, readout_index: int = 0) -> None:
        pass


class ReadNoiseEffect(DetectorEffect):
    """Parse and attach scalar/array read noise from ``[detector] read_noise``."""

    name = "readnoise"

    def __init__(self, config):
        super().__init__(config)
        read_noise_str = _config_get(config, "detector", "read_noise", "0")
        self.read_noise_e_rms = (
            np.fromstring(str(read_noise_str), sep=",") * u.electron / u.pix
        )
        logger.info("Read noise is %s rms", self.read_noise_e_rms)

    def register(self, instrument) -> None:
        instrument.sources_instrum["read_noise_e_pix-1"] = self.read_noise_e_rms
        for channel in instrument.output_channels.values():
            self.apply(channel)

    def apply(self, channel, readout_index: int = 0) -> None:
        channel.instrum_noise["read_noise_e_pix-1"] = self.read_noise_e_rms

    def reset(self) -> None:
        pass


class DarkCurrentEffect(DetectorEffect):
    """Parse and attach dark-current sweep from ``[detector] dark_current``."""

    name = "dark_current"

    def __init__(self, config):
        super().__init__(config)
        dark_current_str = str(_config_get(config, "detector", "dark_current", "0"))
        if "," in dark_current_str:
            parts = [float(x.strip()) for x in dark_current_str.split(",")]
            # Historical tables.py behavior: any comma-separated list used as
            # np.arange(start, stop, step) when it has three parts.
            if len(parts) == 3:
                dark_current_rate = np.arange(parts[0], parts[1], parts[2])
            else:
                dark_current_rate = np.asarray(parts, dtype=float)
        else:
            dark_current_rate = np.fromstring(dark_current_str, sep=",")

        self.dark_current_rate_e_pix_sec = (
            dark_current_rate * u.electron / (u.pix * u.second)
        )
        t_frame = float(_config_get(config, "observation", "t_int_frame", "0")) * u.second
        self.dark_current_e_pix = self.dark_current_rate_e_pix_sec * t_frame
        logger.info(
            "Dark current array is %s e-/pix/sec",
            self.dark_current_rate_e_pix_sec,
        )

    def register(self, instrument) -> None:
        instrument.sources_instrum["dark_current_e_pix-1_sec-1"] = (
            self.dark_current_rate_e_pix_sec
        )
        instrument.sources_instrum["dark_current_e_pix-1"] = self.dark_current_e_pix
        for channel in instrument.output_channels.values():
            self.apply(channel)

    def apply(self, channel, readout_index: int = 0) -> None:
        channel.instrum_noise["dark_current_e_pix-1_sec-1"] = (
            self.dark_current_rate_e_pix_sec
        )

    def reset(self) -> None:
        pass


EFFECT_REGISTRY: dict[str, type[DetectorEffect]] = {
    "persistence": PersistenceEffect,
    "gain_variability": GainVariabilityEffect,
    "one_over_f": OneOverFNoiseEffect,
    "transfer_function": TransferFunctionEffect,
    "aging_pixels": AgingPixelsEffect,
    "cosmic_rays": CosmicRaysEffect,
    "hot_pixels": HotPixelsEffect,
    "readnoise": ReadNoiseEffect,
    "dark_current": DarkCurrentEffect,
}


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


class InstrumentDepTerms(TablesMixin, TransmissionMixin, TransferMixin):
    # Provides the effects of the instrument (including astro flux passed through the telescope aperture)

    def __init__(self, config: Dict, sources_astroph: dict, sources_to_include: list):
        '''
        Args:
            config: Configuration dictionary
            sources: Dictionary of sources of flux; {'wavel': <Quantity um>, 'pre_screen_astro_flux_ph_sec_m2_um': <Quantity ph / (s um m2)>}
            sources_to_include: List of sources to actuallyinclude in the S/N calculation (and plots of incident fluxes)
        '''

        self.config = config
        self.sources_astroph = sources_astroph # all sources of astrophysical flux, as are incident on the instrument
        self.sources_to_include = sources_to_include

        # initialize dict to carry intrinsic instrumental terms (independent of astrophysics)
        self.sources_instrum = {}

        # initialize dict to carry propagated astrophysical terms (i.e., intensity levels on the detector, after instrument effects)
        self.prop_dict = {}
        # assume wavelengths are the same for the star and planet
        #self.prop_dict['wavel'] = self.star_flux['wavel']

        # Intrinsic RN / DC always available for S/N tables (not only when listed in enabled)
        self.readnoise_effect = ReadNoiseEffect(config)
        self.dark_current_effect = DarkCurrentEffect(config)

        # Optional systematics chain (may include readnoise / dark_current again by reference)
        self.detector_effects: List[DetectorEffect] = []
        for effect_name in _enabled_effect_names(config):
            if effect_name == "readnoise":
                self.detector_effects.append(self.readnoise_effect)
            elif effect_name == "dark_current":
                self.detector_effects.append(self.dark_current_effect)
            elif effect_name in EFFECT_REGISTRY:
                self.detector_effects.append(EFFECT_REGISTRY[effect_name](config))
            else:
                raise ValueError(
                    f"Detector effect {effect_name!r} not found in registry"
                )

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
            for name in ['output_1_bright', 'output_2_bright', 'output_3_dark', 'output_4_dark']
        }

        # for each output channel, set the detection wavelength bins (same for all channels for now)
        R = float(self.config["detector"]["spec_res"]) # spectral resolution (lambda/del_lambda)
        # bins are spaced geometrically in wavelength space, with recurrence relation lambda_i = lambda_{0} * (1 + 1/R)**i
        lambda_min, lambda_max = float(self.config["wavelength_range"]["min"]) * u.um, float(self.config["wavelength_range"]["max"])  * u.um
        # number of bins that fit fully in [lmin, lmax]
        n_bins = int(np.floor(np.log(lambda_max / lambda_min) / np.log(1.0 + 1.0 / R)))

        # geometric bin edges and centers
        bin_edges = lambda_min * (1.0 + 1.0 / R) ** np.arange(n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        # wavelength bin widths (in wavelength units, not pixels)
        bin_widths = bin_edges[1:]-bin_edges[:-1] # removed units for plotting
        for output_channel in self.output_channels.values():
            output_channel.spec_R = R
            output_channel.bin_edges = bin_edges
            output_channel.bin_centers = bin_centers
            output_channel.bin_widths = bin_widths
