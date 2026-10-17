"""The Sun's motion around the Solar System barycenter (SSB), computed from the planets.

The planets pull on the Sun as much as it pulls on them, so the Sun orbits the common center of mass:
    sum_i m_i r_i = 0   (barycentric frame)   =>   r_sun = -(1 / m_sun) sum_planets m_i r_i
Jupiter alone moves the Sun ~1.07 solar radii (period 11.86 yr); with Saturn (29.5 yr), Uranus and
Neptune the Sun's center wanders up to ~2.2 solar radii from the barycenter.

Two independent computations, both checked against JPL's DE441 Sun:
  1. the center-of-mass relation above with the JPL planet positions (instantaneous);
  2. our own N-body integration of the Sun + 9 planets (Mercury ... Neptune, Pluto system; the Earth
     and the Moon separately), Newtonian point masses, started from JPL states and integrated with
     DOP853 (compiled right-hand side, GIL released).
"""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from core.constants import GM

BODIES = ["sun", "mercury", "venus", "earth", "moon", "mars_bary", "jupiter_bary", "saturn_bary",
          "uranus_bary", "neptune_bary", "pluto_bary"]
PLANET_LABELS = {"mercury": "Mercury", "venus": "Venus", "earth": "Earth", "moon": "Moon", "mars_bary": "Mars",
                 "jupiter_bary": "Jupiter", "saturn_bary": "Saturn", "uranus_bary": "Uranus",
                 "neptune_bary": "Neptune", "pluto_bary": "Pluto"}
MU = np.array([GM[b] for b in BODIES])


def planet_positions(eph, t):
    """SSB positions (n_bodies, ..., 3) from the JPL ephemeris."""
    return np.array([eph.pos(b, t) for b in BODIES])


def sun_from_planets(eph, t, bodies=None):
    """Center-of-mass estimate of the Sun's barycentric position from the planets' positions; also returns
    each planet's contribution -(m_i / m_sun) r_i."""
    bodies = bodies or BODIES[1:]
    contrib = {b: -GM[b] / GM["sun"] * eph.pos(b, t) for b in bodies}
    return sum(contrib.values()), contrib


@njit(cache=True, nogil=True)
def _accel(x, mu):
    n = mu.shape[0]
    out = np.zeros(6 * n)
    for i in range(n):
        out[3 * i:3 * i + 3] = x[3 * n + 3 * i:3 * n + 3 * i + 3]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * j] - x[3 * i]
            dy = x[3 * j + 1] - x[3 * i + 1]
            dz = x[3 * j + 2] - x[3 * i + 2]
            r2 = dx * dx + dy * dy + dz * dz
            inv3 = 1.0 / (r2 * np.sqrt(r2))
            ai, aj = mu[j] * inv3, mu[i] * inv3
            b = 3 * n + 3 * i
            c = 3 * n + 3 * j
            out[b] += ai * dx
            out[b + 1] += ai * dy
            out[b + 2] += ai * dz
            out[c] -= aj * dx
            out[c + 1] -= aj * dy
            out[c + 2] -= aj * dz
    return out


def integrate(eph, t0, t1, n_out=4000, rtol=1e-12):
    """N-body Sun + planets from JPL states at t0. Returns times, positions (n_out, n_bodies, 3), velocities."""
    n = len(BODIES)
    r0 = np.array([eph.pos(b, t0) for b in BODIES])
    v0 = np.array([eph.vel(b, t0) for b in BODIES])
    x0 = np.concatenate([r0.ravel(), v0.ravel()])
    t_eval = np.linspace(t0, t1, n_out)
    sol = solve_ivp(lambda t, x: _accel(x, MU), (t0, t1), x0, method="DOP853", rtol=rtol, atol=1e-3,
                    t_eval=t_eval)
    X = sol.y.T
    return t_eval, X[:, :3 * n].reshape(-1, n, 3), X[:, 3 * n:].reshape(-1, n, 3)


def invariants(R, V):
    """Barycenter position and total energy of the integrated system (conservation checks)."""
    m = MU / MU.sum()
    com = np.einsum("b,tbi->ti", m, R)
    ke = 0.5 * np.einsum("b,tbi,tbi->t", MU, V, V)
    pe = np.zeros(len(R))
    for i in range(len(MU)):
        for j in range(i + 1, len(MU)):
            pe -= MU[i] * MU[j] / np.linalg.norm(R[:, j] - R[:, i], axis=1)
    return com, ke + pe


def plot_data(eph, n=4000, with_nbody=True):
    """Sun-around-barycenter series for the app: ecliptic xy and distances in solar radii."""
    from core.constants import RADIUS
    from core.timeutil import t_to_year

    obl = np.radians(23.4392911)
    eq_to_ecl = np.array([[1, 0, 0], [0, np.cos(obl), np.sin(obl)], [0, -np.sin(obl), np.cos(obl)]])
    lo, hi = eph.span("sun")
    hi = min(hi, eph.span("pluto_bary")[1])
    ts = np.linspace(lo + 86400.0, hi - 86400.0, n)
    rs = RADIUS["sun"]
    jpl = eph.pos("sun", ts)
    out = {"years": t_to_year(ts), "xy_jpl": (jpl @ eq_to_ecl.T)[:, :2] / rs,
           "dist_jpl": np.linalg.norm(jpl, axis=1) / rs}
    _, c = sun_from_planets(eph, ts)
    out["dist_js"] = np.linalg.norm(c["jupiter_bary"] + c["saturn_bary"], axis=1) / rs
    out["dist_j"] = np.linalg.norm(c["jupiter_bary"], axis=1) / rs
    out["xy_nbody"] = None
    if with_nbody:
        T, R, _ = integrate(eph, ts[0], ts[-1], n_out=n)
        out["xy_nbody"] = (R[:, 0] @ eq_to_ecl.T)[:, :2] / rs
        out["nbody_max_err_km"] = float(np.max(np.linalg.norm(R[:, 0] - eph.pos("sun", T), axis=1)) / 1e3)
    return out
