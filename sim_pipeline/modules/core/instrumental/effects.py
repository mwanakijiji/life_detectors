"""Detector effects acting on 2D readout frames, and the stage that runs them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import astropy.units as u
import numpy as np
from astropy.io import fits

from ...pipeline_registry import pipeline_stage
from ...utils.types import ConfigLike

logger = logging.getLogger(__name__)

# names that used to be effects but are now always-on Detector noise parameters
_DETECTOR_NOISE_PARAMS = ("readnoise", "dark_current")


def _config_get(
    config: ConfigLike, section: str, key: str, default: Any = None
) -> Any:
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


def _enabled_effect_names(config: ConfigLike) -> List[str]:
    raw = _config_get(config, "detector_systematics", "enabled", "") or ""
    return [name.strip() for name in str(raw).split(",") if name.strip()]


@dataclass
class DetectorFrames:
    """One readout of one channel, as plain ndarrays in e-/pix."""

    source_e: Dict[str, np.ndarray]  # per-source illumination (n_pix, n_pix)
    systematics_e: np.ndarray  # non-astrophysical charge (n_pix, n_pix)
    channel_name: str = ""  # lets stateful effects keep per-channel state

    @property
    def total_e(self) -> np.ndarray:
        """Total charge per pixel: systematics plus every source."""
        return self.systematics_e + sum(self.source_e.values())


class DetectorEffect:
    """
    Base class for optional detector systematics applied per readout.

    ``apply`` mutates the frames in place, the way the physics requires:
    additive light-independent effects add to ``systematics_e``; multiplicative
    effects scale every array in ``source_e`` and ``systematics_e``;
    light-dependent effects read ``total_e``, and may keep state across
    ``readout_index`` (cleared by ``reset``).
    """

    name: str = "detector_effect"

    def __init__(self, config: ConfigLike) -> None:
        self.config = config

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        """Add the map to the frames' systematics (maps are e-/pix per readout)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-angle state."""


class AdditiveMapEffect(DetectorEffect):
    """Light-independent charge from a static FITS map (e-/pix per readout)."""

    file_key: str = ""

    def __init__(self, config: ConfigLike) -> None:
        super().__init__(config)
        file_path = _config_get(config, "detector_systematics", self.file_key)
        if not file_path:
            raise ValueError(
                f"Detector effect {self.name!r} needs [detector_systematics] "
                f"{self.file_key}"
            )
        self.map_e = np.asarray(fits.getdata(file_path), dtype=float)
        logger.info("Loaded %s map from %s", self.name, file_path)

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        if self.map_e.shape != frames.systematics_e.shape:
            raise ValueError(
                f"{self.name} map shape {self.map_e.shape} does not match detector "
                f"frame shape {frames.systematics_e.shape}"
            )
        frames.systematics_e += self.map_e


class HotPixelsEffect(AdditiveMapEffect):
    """Hot pixels: static additive map from
    [detector_systematics] hot_pixels_2d_file.
    """

    name = "hot_pixels"
    file_key = "hot_pixels_2d_file"


class CosmicRaysEffect(AdditiveMapEffect):
    """Cosmic rays: additive map from [detector_systematics] cosmic_rays_2d_file."""

    name = "cosmic_rays"
    file_key = "cosmic_rays_2d_file"


class ReadPatternEffect(AdditiveMapEffect):
    """Read pattern: additive map from [detector_systematics] read_noise_2d_file."""

    name = "read_pattern"
    file_key = "read_noise_2d_file"


class BiasEffect(AdditiveMapEffect):
    """Bias: additive map from [detector_systematics] dc_2d_file."""

    name = "bias"
    file_key = "dc_2d_file"


class PersistenceEffect(DetectorEffect):
    """Light-dependent: will read ``frames.total_e`` and
    carry charge to later readouts.
    """

    name = "persistence"

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        pass


class GainVariabilityEffect(DetectorEffect):
    """Multiplicative: will scale every array in ``source_e`` and ``systematics_e``."""

    name = "gain_variability"

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        pass


class OneOverFNoiseEffect(DetectorEffect):
    """Additive, light-independent: will add correlated noise to ``systematics_e``."""

    name = "one_over_f"

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        pass


class TransferFunctionEffect(DetectorEffect):
    """
    Light-dependent: will redistribute ``source_e`` and ``systematics_e`` (e.g. IPC).

    Relevant for avoiding aliasing with the chop frequency and spacecraft rotation
    harmonics)
    """

    name = "transfer_function"

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        pass


class AgingPixelsEffect(DetectorEffect):
    """Multiplicative: will scale ``source_e`` and ``systematics_e`` pixel by pixel."""

    name = "aging_pixels"

    def apply(self, frames: DetectorFrames, readout_index: int) -> None:
        pass


EFFECT_REGISTRY: dict[str, type[DetectorEffect]] = {
    "persistence": PersistenceEffect,
    "gain_variability": GainVariabilityEffect,
    "one_over_f": OneOverFNoiseEffect,
    "transfer_function": TransferFunctionEffect,
    "aging_pixels": AgingPixelsEffect,
    "cosmic_rays": CosmicRaysEffect,
    "hot_pixels": HotPixelsEffect,
    "read_pattern": ReadPatternEffect,
    "bias": BiasEffect,
}


def build_detector_effects(config: ConfigLike) -> List[DetectorEffect]:
    """Instantiate effects in the order listed in ``[detector_systematics] enabled``."""
    effects: List[DetectorEffect] = []
    for effect_name in _enabled_effect_names(config):
        if effect_name in _DETECTOR_NOISE_PARAMS:
            logger.warning(
                "Ignoring %r in [detector_systematics] enabled: read noise and dark "
                "current are always-on [detector] parameters",
                effect_name,
            )
            continue
        if effect_name not in EFFECT_REGISTRY:
            raise ValueError(f"Detector effect {effect_name!r} not found in registry")
        effects.append(EFFECT_REGISTRY[effect_name](config))
    return effects


@pipeline_stage(depends_on=("disperse_astro_signals",))
def expose_plus_detector_effects(
    output_channels: dict,
    effects: List[DetectorEffect],
    t_frame: u.Quantity,
    n_readouts: int,
) -> None:
    """
    For each channel and readout: expose the detector and run every effect on the
    frames. Stores the frames on ``channel.readout_frames`` (for
    ``write_detector_readouts``).
    """
    for output_channel in output_channels.values():
        det = output_channel.detector
        output_channel.readout_frames = []
        for readout_index in range(n_readouts):
            frames = det.expose(output_channel.astroph_signal, t_frame)
            frames.channel_name = output_channel.name
            for effect in effects:
                effect.apply(frames, readout_index)
            output_channel.readout_frames.append(frames)
        logger.info(
            "Applied %d detector effect(s) over %d readout(s) for %s",
            len(effects),
            n_readouts,
            output_channel.name,
        )
