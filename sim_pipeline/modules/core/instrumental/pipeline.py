from typing import Dict

import numpy as np
import astropy.units as u

from .channels import OutputChannel
from .tables import TablesMixin
from .transfer import TransferMixin
from .transmission import TransmissionMixin


class DetectorEffect:
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class PersistenceEffect:
    name = "persistence"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class GainVariabilityEffect:
    name = "gain_variability"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class OneOverFNoiseEffect:
    name = "one_over_f"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class TransferFunctionEffect:
    name = "transfer_function"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class AgingPixelsEffect:
    name = "aging_pixels"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class CosmicRaysEffect:
    name = "cosmic_rays"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class HotPixelsEffect:
    name = "hot_pixels"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')

class ReadNoiseEffect:
    name = "readnoise"
    def __init__(self, config):
        print('TBD')
    def apply(self, channel, readout_index) -> None:
        # mutate channel in place
        print('TBD')
    def reset(self) -> None:
        print('TBD')


EFFECT_REGISTRY: dict[str, type[DetectorEffect]] = {
    "persistence": PersistenceEffect,
    "gain_variability": GainVariabilityEffect,
    "one_over_f": OneOverFNoiseEffect,
    "transfer_function": TransferFunctionEffect,
    "aging_pixels": AgingPixelsEffect,
    "cosmic_rays": CosmicRaysEffect,
    "hot_pixels": HotPixelsEffect,
    "readnoise": ReadNoiseEffect,
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

        # initialize list of detector effects
        self.detector_effects = []
        for effect_name in config["detector_systematics"]["enabled"].split(","):
            if effect_name.strip() in EFFECT_REGISTRY:

                # instantiate the detector effect
                effect_inst = EFFECT_REGISTRY[effect_name.strip()](config)

                # string the detector effects together
                self.detector_effects.append(effect_inst)

            else:
                raise ValueError(f"Detector effect {effect_name.strip()} not found in registry")

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
