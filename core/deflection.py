"""Deflection physics: kinetic impact, gravity tractor, Earth-encounter B-plane.

Kinetic impact (Cheng et al. 2023, Nature 616:457, eq. 1):
    M dv = m U + m (beta - 1) (E . U) E      (E = net ejecta-momentum direction)
For ejecta thrown back along the incoming direction (E = -U_hat, head-on) this is dv = beta m U / M.

Gravity tractor (Lu & Love 2005, Nature 438:177): hovering at distance d with thrusters canted
outward by asin(R/d) + phi so the plume misses the surface,
    T cos(asin(R/d) + phi) = G M m / d^2,       asteroid acceleration a = G m / d^2
Enhanced gravity tractor (Mazanek et al. 2015): m = spacecraft + collected boulder.

B-plane at an Earth encounter: plane normal to the Earth-relative velocity s; a first-order position
change dr maps to the miss vector change (I - s s^T) dr. Axes: zeta = -(Earth's heliocentric velocity
projected on the plane) (timing axis, Opik/Valsecchi), xi = s x zeta.
"""

import numpy as np
from scipy.optimize import brentq

from core.constants import BENNU, G, GM, RADIUS

M_BENNU = BENNU["mass"]


def ki_delta_v(m, U, beta, E_hat=None, M=M_BENNU):
    """Asteroid velocity change [m/s] from an impactor of mass m with relative velocity U (vector)."""
    U = np.asarray(U, float)
    if E_hat is None:
        E_hat = -U / np.linalg.norm(U)
    return (m * U + m * (beta - 1.0) * (E_hat @ U) * E_hat) / M


def dart_period_change_minutes(dv_T=-2.70e-3, v_orb=0.1776, period_h=11.92):
    """Orbit-period change of Dimorphos from an along-track dv (circular orbit: dP/P = 3 dv / v).
    Defaults: measured dv_T (Cheng et al. 2023) and the lecture-1 values of v and P."""
    return 3.0 * dv_T / v_orb * period_h * 60.0


def gt_required_thrust(m_tractor, d, R=BENNU["radius"], phi_deg=20.0, M=M_BENNU):
    return G * M * m_tractor / d**2 / np.cos(np.arcsin(R / d) + np.radians(phi_deg))


def gt_accel(m_tractor, d):
    return G * m_tractor / d**2


def gt_min_hover_distance(T_max, m_tractor, R=BENNU["radius"], phi_deg=20.0, M=M_BENNU, d_cap=50e3):
    """Smallest hover distance the thrust can sustain (None if even d_cap is too close)."""
    f = lambda d: gt_required_thrust(m_tractor, d, R, phi_deg, M) - T_max
    lo = R / np.sin(np.radians(89.0) - np.radians(phi_deg))  # cant angle < 89 deg
    if f(d_cap) > 0:
        return None
    if f(lo) <= 0:
        return lo
    return brentq(f, lo, d_cap)


def earth_capture_radius(v_inf):
    """Gravitational focusing: an object with v_inf hits Earth if its B-plane miss < b_E."""
    return RADIUS["earth"] * np.sqrt(1.0 + 2.0 * GM["earth"] / (RADIUS["earth"] * v_inf**2))


class BPlane:
    def __init__(self, v_rel, v_earth_helio):
        self.s = v_rel / np.linalg.norm(v_rel)
        ve_perp = v_earth_helio - (v_earth_helio @ self.s) * self.s
        self.zeta = -ve_perp / np.linalg.norm(ve_perp)
        self.xi = np.cross(self.s, self.zeta)
        self.basis = np.vstack([self.xi, self.zeta])  # 2x3: (xi, zeta) components

    def project(self, dr):
        return self.basis @ dr

    def timing_shift(self, dr):
        return -(dr @ self.s)
