import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as const
import ipdb

file_name = '/Users/eckhartspalding/Documents/git.repos/life_detectors/data/example_exoplanet_spectrum_Proof_of_Concept_R100_SN20.txt'

# 'The first column is wavelength (in um), second is flux (in erg/s/Hz/m2), and the third is the error on the flux.' --Z.B.
df = pd.read_csv(file_name, delim_whitespace=True, names=['wavelength', 'flux', 'err_flux'])

wavel = df['wavelength'].values * u.micron
flux_nu = df['flux'].values * u.erg / (u.second * u.Hz * u.m**2)
err_flux_nu = df['err_flux'].values * u.erg / (u.second * u.Hz * u.m**2)

# convert to F_lambda
flux_lambda = flux_nu * (const.c / wavel**2)
flux_lambda = flux_lambda.to(u.W / (u.m**2 * u.micron))

# convert to photon flux
flux_photons = flux_lambda * (wavel / (const.h * const.c)) * u.ph
flux_photons = flux_photons.to(u.ph / (u.micron * u.s * u.m**2))

plt.plot(wavel, flux_photons)
plt.xlabel('Wavelength (' + str(wavel.unit) + ')')
plt.ylabel('Flux (' + str(flux_photons.unit) + ')')
plt.show()


