"""Motion near Bennu in its body-fixed (rotating) frame, used for descent and boulder capture.

    rho'' = -2 w x rho' - w x (w x rho) + grad U(rho) + a_ext,        w = constant spin vector

Without external forces the Jacobi integral  C = 1/2 |rho'|^2 - 1/2 |w x rho|^2 - U(rho)  is conserved.
"""

import numpy as np

from core.constants import BENNU

OMEGA_BODY = np.array([0.0, 0.0, BENNU["omega"]])  # Bennu spins about its body z axis


def rhs(t, x, gravity, a_ext=None, w=OMEGA_BODY):
    """x = [rho (3), rho' (3)] in the body frame; gravity(rho) -> acceleration (body frame)."""
    rho, v = x[:3], x[3:]
    a = -2.0 * np.cross(w, v) - np.cross(w, np.cross(w, rho)) + gravity(rho)
    if a_ext is not None:
        a = a + a_ext(t, x)
    return np.concatenate([v, a])


def jacobi(x, potential, w=OMEGA_BODY):
    rho, v = x[:3], x[3:]
    return 0.5 * v @ v - 0.5 * np.sum(np.cross(w, rho) ** 2) - potential(rho)


def effective_gravity(rho, gravity, w=OMEGA_BODY):
    """Gravity + centrifugal acceleration felt by a particle at rest on the rotating body."""
    return gravity(rho) - np.cross(w, np.cross(w, rho))
