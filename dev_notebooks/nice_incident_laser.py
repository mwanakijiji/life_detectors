import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

# total max emitted power by laser
P_e_tot = 1 * u.W # W

# numberical aperture NA
NA = 0.3

# wavelength
wavel_lambda = 3.91e-6 * u.m # m

# %%
# detector received power limit: 2 mW/mm2 (WinCamD-IR-BB)
P_area_detector_limit = (2e-3) * u.W / u.mm**2 # 2 mW/mm2

# eye limit at 4 um: 0.1 W/cm2; conservative but rough estimate
P_area_eye_limit = 1e-1 * u.W/u.cm**2 # W/m2

# horizontal linear displacement
d_horiz = np.arange(0, 2, 0.01) * u.m # m

# incident power per unit area 
rad_beam = d_horiz * np.tan( np.arcsin(NA) )
P_area_incident = P_e_tot / (np.pi * np.power(rad_beam, 2) )

# beam radius 
#print('Beam radius (cm): ', rad_beam*100)
plt.clf()
plt.plot(d_horiz.to(u.cm), rad_beam.to(u.cm))
plt.xlabel('Horizontal Displacement (cm)')
plt.ylabel('Beam Radius (cm)')
plt.show()

# the amount by which the laser is stopped down by for OD5
P_area_incident_OD5 = P_area_incident * 1e-5

plt.plot(d_horiz.to(u.m), P_area_incident.to(u.W/u.m**2), label='Incident Power w/o OD')
plt.plot(d_horiz.to(u.m), P_area_incident_OD5.to(u.W/u.m**2), label='Incident Power, OD5')
# Convert limits to same units as P_area_incident (W/m²) for correct comparison
P_area_detector_limit_Wm2 = P_area_detector_limit.to(P_area_incident.unit)
P_area_eye_limit_Wm2 = P_area_eye_limit.to(P_area_incident.unit)
plt.axhline(y=P_area_detector_limit_Wm2.value, color='b', linestyle='--', label='Detector limit (2 mW/mm²)')
plt.axhline(y=P_area_eye_limit_Wm2.value, color='r', linestyle='--', label='Eye limit > 10 sec (0.1 W/cm²)')
plt.legend()
plt.xlabel('Horizontal Displacement (m)')
plt.ylabel('Incident Power per Unit Area (W/m2)')
plt.show()


