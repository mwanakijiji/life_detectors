import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
from scipy.integrate import quad
import ipdb

# surface brightness profile is
# Ref.: Eqn. 1 in Kennedy 2015 ApJSS 216:23
#
# S_disk = Sigma_m * BB(lambda, T(r))
#
# where 
# 
# Sigma_m = z * Sigma_m_0 * (r/r0)^(-alpha), where z is number of zodis and Sigma_m_0 normalizes the surface brightness
# T(r) = 278.3K * Ls^(0.25) * r^-(0.5)


def T_temp(Ls, r):
    # Ls: luminosity of star (units L_sol)
    # r: radius in disk (units AU)

    T = 278.3*u.K * (Ls**0.25) * (r**-0.5)

    return T


def Sigma_m(r, r0, alpha, Ls, z, Sigma_m_0):
    # r: radius in disk (units AU)
    # r0: reference radius (units AU)
    # alpha: power law index
    # z: number of zodis
    # Sigma_m_0: normalization factor
    #   Kennedy: 'is to be set at some r0 (in AU) such that the surface density is in units of zodis z (see Section 2.2.3).'

    Sigma_m = z * Sigma_m_0 * (r/r0)**(-alpha)

    return Sigma_m


# spectral surface brightness profile I(lambda, r)
# Eqn. 16 in Dannert+ 2022 A&A 664:A22 
# Slight variation in notation: Eqn. 1 in Kennedy+ 2015 ApJSS 216:23
# N.b. Kennedy uses 'S' instead of 'I' 
def I_disk_lambda_r(r, r0, alpha, Ls, z, Sigma_m_0, wavel_array):

    bb = BlackBody(temperature=T_temp(Ls=Ls, r=r),  scale=1.0*u.W/(u.m**2*u.micron*u.sr))

    return Sigma_m(r=r, r0=r0, alpha=alpha, Ls=Ls, z=z, Sigma_m_0=Sigma_m_0) * bb(wavel_array)


# surface brightness as function of wavelength I(lambda): I_disk_lambda_r integrated over dA = r dr dtheta
def I_disk_lambda(r_array, r0, alpha, Ls, z, Sigma_m_0, wavel_array):
    
    # Integrate over r * I_disk_lambda_r()
    integrand = np.array( [r * I_disk_lambda_r(r, r0, alpha, Ls, z, Sigma_m_0, wavel_array) for r in r_array] )
    
    # integrate over r in r_array using the trapezoidal rule (logarithmic spacing), to leave dimensions (1, lambda)
    I_lambda = 2 * np.pi * np.trapezoid(integrand, x=r_array, axis=0)

    return np.array(I_lambda)


# scale emission for distance from Earth (effectively doing I_disk_lambda, except that integral is over d_Omega = r dr dtheta / D**2)
def I_disk_lambda_Earth(I_disk_lambda_array, D):
    return I_disk_lambda_array * (1 / 206265.)**2


########################################################

# set up some basic params
r_array = np.arange(0.1, 10, 0.1)
r0 = 1
alpha = 0.5
z = 3
Sigma_m_0 = 1
Ls = 1

T_array = T_temp(Ls=Ls, r=r_array)
wavel_array = np.arange(2., 20, 0.1) * u.um


radiance_disk_lambda = I_disk_lambda(r_array=r_array, r0=r0, alpha=alpha, Ls=Ls, z=z, Sigma_m_0=Sigma_m_0, wavel_array=wavel_array)
radiance_disk_lambda_Earth = I_disk_lambda_Earth(radiance_disk_lambda, D=10)

# tack on units
radiance_disk_lambda_Earth = radiance_disk_lambda_Earth * u.W / (u.m**2 * u.um)

ipdb.set_trace()



# plot temperature profile
plt.clf()
plt.plot(r_array, T_temp(Ls=Ls, r=r_array))
plt.title(f'T(r)')
plt.xlabel('r (AU)')
plt.ylabel('T (K)')
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(x=1, color='k', linestyle='--')
plt.show()


# Sigma_m profile
plt.clf()
plt.plot(r_array, Sigma_m(r_array, r0, alpha, Ls, z, Sigma_m_0))
plt.title(r'$\Sigma_{m}$')
plt.xlabel('r (AU)')
plt.ylabel(r'$\Sigma_{m}$')
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(x=1, color='k', linestyle='--')
plt.show()


# Sigma_m profile
plt.clf()
plt.plot(wavel_array, radiance_disk_lambda_Earth)
plt.title('Exozodiacal flux on Earth')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Flux on Earth (W m-2 um-1)')
plt.show()

