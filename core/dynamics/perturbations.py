"""Non-gravitational and environmental accelerations (SI units)."""

import numpy as np

from core.constants import AU, DAY, EARTH_OMEGA, RADIUS

P_SUN = 4.56e-6  # N/m^2, solar radiation pressure at 1 au (absorbing surface)

# Exponential atmosphere, Vallado "Fundamentals of Astrodynamics and Applications", Table 8-4:
# base altitude h0 [km], density rho0 [kg/m^3], scale height H [km]
_ATM = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682], [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714], [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382], [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523], [180, 5.464e-10, 29.740],
    [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546], [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515], [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05], [1000, 3.019e-15, 268.00],
])


def air_density(h_m):
    """Atmospheric density [kg/m^3] at geodetic-ish altitude h (spherical Earth)."""
    h = h_m / 1e3
    if h >= 1500.0:
        return 0.0
    i = max(0, np.searchsorted(_ATM[:, 0], h, side="right") - 1)
    h0, rho0, H = _ATM[i]
    return rho0 * np.exp(-(h - h0) / H)


def drag(r, v, cd_area_over_m):
    """Atmospheric drag for a geocentric state; atmosphere co-rotates with the Earth (z = pole)."""
    rho = air_density(np.linalg.norm(r) - RADIUS["earth"])
    if rho == 0.0:
        return np.zeros(3)
    v_rel = v - np.cross([0.0, 0.0, EARTH_OMEGA], r)
    return -0.5 * rho * cd_area_over_m * np.linalg.norm(v_rel) * v_rel


def cylindrical_shadow(r_sc, r_sun, r_occ, R_occ):
    """1 if sunlit, 0 if inside the occulting body's cylindrical shadow (positions in one frame)."""
    s = (r_sun - r_occ) / np.linalg.norm(r_sun - r_occ)
    p = r_sc - r_occ
    along = p @ s
    if along > 0.0:
        return 1.0
    return 0.0 if np.linalg.norm(p - along * s) < R_occ else 1.0


def srp(r_sc_minus_sun, cr_area_over_m, nu=1.0):
    """Cannonball solar radiation pressure; pushes away from the Sun, falls off as 1/r^2."""
    d = np.linalg.norm(r_sc_minus_sun)
    return nu * P_SUN * cr_area_over_m * (AU / d) ** 2 * r_sc_minus_sun / d


A2_TO_SI = AU / DAY**2  # au/d^2 -> m/s^2


def yarkovsky(r_helio, v_helio, A2_au_d2):
    """Transverse Yarkovsky model a = A2 (1 au / r)^2 t_hat (Farnocchia/Chesley A2 formulation).

    t_hat lies in the orbit plane, perpendicular to r, along the direction of motion.
    """
    r = np.linalg.norm(r_helio)
    h = np.cross(r_helio, v_helio)
    t_hat = np.cross(h, r_helio)
    t_hat /= np.linalg.norm(t_hat)
    return A2_au_d2 * A2_TO_SI * (AU / r) ** 2 * t_hat
