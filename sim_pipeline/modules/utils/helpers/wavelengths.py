"""The wavelength grid shared by the astrophysical scene cubes and the polychromatic
transmission screens, so that the two can be multiplied slice by slice."""

import numpy as np
from astropy import units as u

from ..types import ConfigLike

# number of wavelength points in the incident-flux (scene) cubes
N_INCIDENT_WAVEL = 100


def incident_cube_wavelengths(config: ConfigLike) -> u.Quantity:
    """
    Wavelengths (um) of the incident-flux cubes: N_INCIDENT_WAVEL points, linearly
    spaced from [wavelength_range] min to max (inclusive). Shape (N_INCIDENT_WAVEL,).
    """
    return (
        np.linspace(
            float(config["wavelength_range"]["min"]),
            float(config["wavelength_range"]["max"]),
            N_INCIDENT_WAVEL,
        )
        * u.um
    )
