from typing import Protocol
import logging

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

from ..data.spectra import SpectralData, load_spectrum_from_file, create_blackbody_spectrum

logger = logging.getLogger(__name__)


class PlanetSpectrumProvider(Protocol):
    """Top-level source of a planet's spectrum: population-derived, or a single read-in file."""
    def get_spectrum(self, system_params=None) -> SpectralData: ...


# --- inner: how does a population row become a spectrum (this is what grows later) ---
class PopulationSpectrumGenerator(Protocol):
    def generate(self, system_params) -> SpectralData: ...

class BlackbodyFromParamsGenerator:
    def __init__(self, wavelength_range): self.wavelength_range = wavelength_range
    def generate(self, system_params):
        return create_blackbody_spectrum(temperature=system_params["Tp"], wavelength_range=self.wavelength_range)

class PSGSpectrumGenerator:   # later — same interface, drops in without touching PopulationSpectrumProvider
    def generate(self, system_params):
        return load_spectrum_from_file(system_params["abs_file_name_psg_spectrum"])


class PopulationSpectrumProvider:
    """Derives a planet's spectrum from its population row, via a swappable generator."""
    def __init__(self, generator: PopulationSpectrumGenerator):
        self.generator = generator
    def get_spectrum(self, system_params) -> SpectralData:
        return self.generator.generate(system_params)

class SingleModelFileProvider:
    """Load a 10 pc reference model spectrum and return intrinsic luminosity.

    File conversion matches the former ``exoplanet_model_10pc`` branch in
    ``astrophysical.py`` (F_nu -> F_lambda -> photon flux). The 10 pc
    reference is converted to intrinsic units once:

        flux_intrinsic = flux_at_10pc * 4π (10 pc)²

    Distance scaling is applied later by ``_calculate_flux_from_spectrum``.
    """

    def __init__(self, path: str):
        self.path = path

    def get_spectrum(self, system_params=None) -> SpectralData:
        df = pd.read_csv(self.path, delim_whitespace=True, names=['wavelength', 'flux', 'err_flux'])
        logger.info(f"Loaded model exoplanet spectrum: {self.path}")

        wavel = df['wavelength'].values * u.micron
        flux_nu_10pc = df['flux'].values * u.erg / (u.second * u.Hz * u.m**2)

        # convert to F_lambda
        flux_lambda_10pc = flux_nu_10pc * (const.c / wavel**2)
        flux_lambda_10pc = flux_lambda_10pc.to(u.W / (u.m**2 * u.micron))

        # convert to photon flux at 10 pc
        flux_photons_10pc = flux_lambda_10pc * (wavel / (const.h * const.c)) * u.ph
        flux_photons_10pc = flux_photons_10pc.to(u.ph / (u.micron * u.s * u.m**2))

        # undo the 10 pc reference so downstream distance scaling can be uniform
        d_ref = 10.0 * u.pc
        flux_intrinsic = (flux_photons_10pc * 4.0 * np.pi * d_ref**2).to(u.ph / (u.micron * u.s))

        return SpectralData(
            wavelength=wavel.to_value(u.um),
            flux=flux_intrinsic.value,
            wavelength_unit="um",
            flux_unit=str(flux_intrinsic.unit),
            source_name="exoplanet_model_10pc",
            metadata={"filepath": str(self.path), "reference_distance_pc": 10.0},
        )
