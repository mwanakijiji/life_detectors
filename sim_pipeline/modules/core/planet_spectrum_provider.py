from typing import Protocol
from ..data.spectra import SpectralData


class PlanetSpectrumProvider(Protocol):
    def get_spectrum(self, system_params) -> SpectralData: ...

class PopulationSpectrumProvider:
    def get_spectrum(self, system_params) -> SpectralData:
        return load_spectrum_from_file(system_params["abs_file_name_psg_spectrum"])

class BlackbodySpectrumProvider:
    def __init__(self, wavelength_range):
        self.wavelength_range = wavelength_range
    def get_spectrum(self, system_params) -> SpectralData:
        return create_blackbody_spectrum(
            temperature=system_params["Tp"],
            wavelength_range=self.wavelength_range,
        )