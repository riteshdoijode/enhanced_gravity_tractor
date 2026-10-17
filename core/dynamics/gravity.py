"""Gravity models (lecture L1, eqs. 1-2).

Lecture form, in an inertial frame:   r'' = sum_i G m_i (r_i - r) / |r_i - r|^3 + f_ext / m.
Written relative to a central body c (which is itself accelerated by the other bodies), the
"indirect term" -mu_k s_k/|s_k|^3 appears:

    r'' = -mu_c r/|r|^3 + sum_{k != c} mu_k [ (s_k - r)/|s_k - r|^3 - s_k/|s_k|^3 ],   s_k = r_k - r_c

All functions are vectorized over leading dimensions of r (shape (..., 3)).
"""

import numpy as np

from core.constants import EARTH_J2, GM, RADIUS

C_LIGHT = 299_792_458.0  # m/s

# Gravitating bodies available in the ephemeris (planet systems as barycenters)
PLANETARY_BODIES = ["sun", "mercury", "venus", "earth", "moon", "mars_bary", "jupiter_bary",
                    "saturn_bary", "uranus_bary", "neptune_bary"]
BIG_ASTEROIDS = ["ceres", "vesta", "pallas", "hygiea"]
ALL_BODIES = PLANETARY_BODIES + BIG_ASTEROIDS


def _norm(x):
    return np.linalg.norm(x, axis=-1, keepdims=True)


def point_mass(r, mu):
    """Two-body acceleration -mu r / |r|^3."""
    return -mu * r / _norm(r) ** 3


def third_body(r, s, mu):
    """Perturbation from body k at s (relative to the central body) on a particle at r."""
    d = s - r
    return mu * (d / _norm(d) ** 3 - s / _norm(s) ** 3)


def j2(r, mu=GM["earth"], R=RADIUS["earth"], J2=EARTH_J2):
    """Oblateness (J2) acceleration; r in a frame whose z axis is the body's spin pole."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = np.sum(r * r, axis=-1)
    rn = np.sqrt(r2)
    k = -1.5 * J2 * mu * R**2 / rn**5
    zz = 5.0 * z * z / r2
    return np.stack([k * x * (1 - zz), k * y * (1 - zz), k * z * (3 - zz)], axis=-1)


def gravity_gradient(r, mu):
    """d(-mu r/|r|^3)/dr = mu/|r|^3 (3 r_hat r_hat^T - I)  (3x3; used by the STM and hover linearization)."""
    rn = np.linalg.norm(r)
    u = r / rn
    return mu / rn**3 * (3.0 * np.outer(u, u) - np.eye(3))


def schwarzschild(r, v, mu=GM["sun"]):
    """First post-Newtonian (GR) correction from the Sun, PPN beta = gamma = 1; r, v heliocentric."""
    rn = np.linalg.norm(r)
    return mu / (C_LIGHT**2 * rn**3) * ((4.0 * mu / rn - v @ v) * r + 4.0 * (r @ v) * v)


class NBody:
    """Gravity of ephemeris bodies acting on a particle, expressed relative to a central body.

    central: 'ssb' (inertial, lecture eq. 2 verbatim) or a body name; for a central body that is
    not in the ephemeris (Bennu) pass its barycentric position function as `center_pos` and its GM.
    """

    def __init__(self, eph, central="ssb", bodies=PLANETARY_BODIES, center_pos=None, center_mu=None,
                 earth_j2=False):
        self.eph = eph
        self.central = central
        self.mu_c = 0.0 if central == "ssb" else (center_mu if center_mu is not None else GM[central])
        self.perturbers = [(b, GM[b]) for b in bodies if b != central]
        self._center_pos = center_pos or (lambda t: np.zeros(3) if central == "ssb" else eph.pos(central, t))
        self.earth_j2 = earth_j2 and central == "earth"

    def body_positions(self, t):
        """Barycentric positions of the perturbers at time t, shape (n, 3)."""
        return np.array([self.eph.pos(b, t) for b, _ in self.perturbers])

    def accel(self, t, r, return_parts=False):
        """Acceleration of a particle at r (relative to the central body) at time t."""
        rc = self._center_pos(t)
        a_central = point_mass(r, self.mu_c) if self.mu_c else np.zeros(3)
        a_pert = np.zeros(3)
        for (b, mu), rb in zip(self.perturbers, self.body_positions(t)):
            if self.central == "ssb":
                d = rb - r
                a_pert += mu * d / np.linalg.norm(d) ** 3
            else:
                a_pert += third_body(r, rb - rc, mu)
        a_j2 = j2(r) if self.earth_j2 else np.zeros(3)
        if return_parts:
            return {"central": a_central, "third_body": a_pert, "j2": a_j2}
        return a_central + a_pert + a_j2

    def gradient(self, t, r):
        """Gravity gradient d(accel)/dr (3x3), point masses only (J2 gradient omitted)."""
        rc = self._center_pos(t)
        G = gravity_gradient(r, self.mu_c) if self.mu_c else np.zeros((3, 3))
        for (b, mu), rb in zip(self.perturbers, self.body_positions(t)):
            G += gravity_gradient(r - (rb - rc), mu)
        return G
