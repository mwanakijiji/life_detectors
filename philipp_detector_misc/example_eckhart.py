import astropy.units as u
# from docs.tutorials.advanced_example import phase
from matplotlib import pyplot as plt

from phringe.core.instrument import Instrument
from phringe.core.observation import Observation
from phringe.core.scene import Scene
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star
from phringe.lib.array_configuration import XArrayConfiguration
from phringe.lib.beam_combiner import DoubleBracewell
from phringe.main import PHRINGE

# Create PHRINGE object
phringe = PHRINGE(grid_size=200)

# Create observation
obs = Observation(
    solar_ecliptic_latitude=0 * u.deg,  # alternatively: '0 deg' or 0
    total_integration_time=10 * u.day,  # alternatively: '1 d' or 86400
    detector_integration_time=0.1 * u.day,  # alternatively: '600 s' or 600
    modulation_period=10 * u.day,  # alternatively: '1 d' or 86400
    nulling_baseline=20 * u.m
)
phringe.set(obs)

# Create instrument
inst = Instrument(
    array_configuration_matrix=XArrayConfiguration.acm,
    complex_amplitude_transfer_matrix=DoubleBracewell.catm,
    kernels=DoubleBracewell.kernels,
    wavelength_bands_boundaries=[],
    aperture_diameter=3 * u.m,
    nulling_baseline_min=8 * u.m,
    nulling_baseline_max=100 * u.m,
    throughput=0.05,
    quantum_efficiency=0.7,
    spectral_resolving_power=20,
    wavelength_min=4 * u.um,
    wavelength_max=18.5 * u.um,
)
phringe.set(inst)

# Create scene
scene = Scene()
phringe.set(scene)

sun_twin = Star(
    name='Sun-Twin',
    distance=10 * u.pc,
    mass=1 * u.Msun,
    radius=1 * u.Rsun,
    temperature=5778 * u.K,
    right_ascension=10 * u.hourangle,  # Uses units of degrees, not time
    declination=45 * u.deg,
)
earth_twin = Planet(
    name='Earth-Twin',
    has_orbital_motion=False,  # Whether the planet is propagated in time along its orbit
    mass=1 * u.Mearth,
    radius=1 * u.Rearth,
    temperature=254 * u.K,
    semi_major_axis=1 * u.au,
    eccentricity=0,
    inclination=0 * u.deg,
    raan=0 * u.deg,
    argument_of_periapsis=135 * u.deg,
    true_anomaly=0 * u.deg,
    input_spectrum=None,
    host_star_distance=10 * u.pc,  # Is only required if no star is added explicitly to the scene
    host_star_mass=1 * u.Msun,  # Is only required if no star is added explicitly to the scene
)
exozodi = Exozodi(
    level=3.0,  # 3 times the local zodiacal dust level
    host_star_luminosity=1 * u.Lsun,  # Is only required if no star is added explicitly to the scene
    host_star_distance=10 * u.pc,  # Is only required if no star is added explicitly to the scene
)
local_zodi = LocalZodi(
    host_star_right_ascension=10 * u.hourangle,  # Is only required if no star is added explicitly to the scene
    host_star_declination=45 * u.deg,  # Is only required if no star is added explicitly to the scene
)

scene.add_source(sun_twin)
scene.add_source(earth_twin)
scene.add_source(exozodi)
scene.add_source(local_zodi)

# Plot sky brightness distribution of scene objects in ph/s/m^3
wavelength_index = 20  # random wavelength index
corresponding_wavelength = phringe.get_wavelength_bin_centers()[
    wavelength_index]  # FYI this is the wavelength corresponding to the index

sbd_sun = sun_twin._sky_brightness_distribution  # has shape n_wavelength x n_pixels x n_pixels
plt.imshow(sbd_sun.cpu().numpy()[wavelength_index], cmap='magma')
plt.title(f'Sun-Twin')
plt.colorbar()
plt.show()

sbd_earth = earth_twin._sky_brightness_distribution
plt.imshow(sbd_earth.cpu().numpy()[wavelength_index], cmap='magma')
plt.title(f'Earth-Twin')
plt.colorbar()
plt.show()

sbd_exozodi = exozodi._sky_brightness_distribution
plt.imshow(sbd_exozodi.cpu().numpy()[wavelength_index], cmap='magma')
plt.title(f'Exozodi')
plt.colorbar()
plt.show()

sbd_local_zodi = local_zodi._sky_brightness_distribution
plt.imshow(sbd_local_zodi.cpu().numpy()[wavelength_index], cmap='magma')
plt.title(f'Local Zodi')
plt.colorbar()
plt.show()
