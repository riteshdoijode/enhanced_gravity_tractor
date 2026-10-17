"""Leaving the Earth parking orbit, for any thrust level.

High thrust: patched-conic escape burn at perigee,
    dv_dep = sqrt(v_inf^2 + 2 mu / r_p) - sqrt(mu / r_p)
Low thrust (tangential spiral, Gauss variational equation for near-circular orbits):
    da/dt = 2 a^(3/2) (T/m) / sqrt(mu)   =>   sqrt(mu / a(t)) = v_c0 - (T/m) t
The orbit-averaged spiral is used while thrust is < 1 % of local gravity; the remaining (few)
revolutions to escape are integrated numerically with tangential thrust, then the craft coasts
to the hand-over radius where heliocentric guidance takes over; hand-over requires C3 >= 0 (a craft
still bound to Earth would fall back, which the Sun-centred guidance cannot model). Geocentric
two-body: the Sun's tide near the sphere-of-influence edge is left to the full N-body live run.
"""

import math

import numpy as np
from numba import njit

from core.constants import GM, RADIUS

MU_E = GM["earth"]
R_SOI = 9.25e8  # m, Earth's sphere of influence
R_HANDOFF = 0.25 * R_SOI


def circular_speed(r):
    return np.sqrt(MU_E / r)


def patched_conic_dv(v_inf, r_park):
    return np.sqrt(v_inf**2 + 2 * MU_E / r_park) - circular_speed(r_park)


def park_period(r_park):
    return 2 * np.pi * np.sqrt(r_park**3 / MU_E)


def spiral_escape(a_thrust, r_park, inclination_deg=28.5, r_handoff=R_HANDOFF, max_time=60 * 365.25 * 86400.0):
    """Tangential-thrust escape from a circular parking orbit.

    Returns time to hand-over, thrust-on delta-v, geocentric hand-over state, and a display path.
    """
    r_sw = min(np.sqrt(0.01 * MU_E / a_thrust), 0.15 * R_SOI)  # thrust reaches 1 % of gravity
    t1, path = 0.0, []
    r_start = r_park
    i = np.radians(inclination_deg)
    plane_x, plane_y = np.array([1.0, 0, 0]), np.array([0, np.cos(i), np.sin(i)])
    if r_sw > r_park:  # orbit-averaged spiral, closed form
        t1 = (circular_speed(r_park) - circular_speed(r_sw)) / a_thrust
        if t1 > max_time:
            return {"feasible": False, "t": t1, "dv": a_thrust * t1}
        ts = np.linspace(0, t1, 400)
        a_t = MU_E / (circular_speed(r_park) - a_thrust * ts) ** 2
        # phase angle = integral of mean motion n = sqrt(mu/a^3) = (v_c0 - a_T t)^3 / mu
        vc = circular_speed(r_park) - a_thrust * ts
        theta = (circular_speed(r_park) ** 4 - vc**4) / (4 * MU_E * a_thrust)
        path = [a_t[:, None] * (np.cos(theta)[:, None] * plane_x + np.sin(theta)[:, None] * plane_y)]
        r_start = r_sw
        th0 = theta[-1]
    else:
        th0 = 0.0
    r = r_start * (np.cos(th0) * plane_x + np.sin(th0) * plane_y)
    v = circular_speed(r_start) * (-np.sin(th0) * plane_x + np.cos(th0) * plane_y)
    pts = np.empty((4000, 3))
    out = np.zeros(4)  # t, dv, escaped, number of path points
    r, v = _escape_tail(r, v, a_thrust, r_handoff, max_time - t1, pts, out)
    t, dv_num, escaped, n_pts = out[0], out[1], bool(out[2]), int(out[3])
    if t1 + t > max_time:
        return {"feasible": False, "t": t1 + t, "dv": a_thrust * t1 + dv_num}
    pts = pts[:n_pts]
    path.append(np.array(pts))
    return {"feasible": True, "escaped": escaped, "t": t1 + t, "dv": a_thrust * t1 + dv_num, "r_geo": r, "v_geo": v,
            "path_geo": np.vstack(path), "t_thrust": t1 + dv_num / a_thrust}


@njit(cache=True, nogil=True)
def _escape_tail(r, v, a_thrust, r_handoff, t_max, pts, out):
    """RK4 tangential-thrust escape (compiled): thrust until C3 >= 0, then coast to r_handoff."""
    r = r.copy()
    v = v.copy()
    t, dv, thrusting, escaped = 0.0, 0.0, True, False
    pts[0] = r
    n = 1
    k_r = np.empty((4, 3))
    k_v = np.empty((4, 3))
    while not (escaped and math.sqrt(r @ r) >= r_handoff):
        rn = math.sqrt(r @ r)
        h = 0.01 * math.sqrt(rn**3 / MU_E)
        if thrusting:
            h = min(h, 0.02 * math.sqrt(v @ v) / a_thrust)
        for s in range(4):
            c = 0.0 if s == 0 else (0.5 if s < 3 else 1.0)
            rr = r + c * h * k_r[s - 1] if s > 0 else r
            vv = v + c * h * k_v[s - 1] if s > 0 else v
            k_r[s] = vv
            k_v[s] = -MU_E * rr / math.sqrt(rr @ rr) ** 3
            if thrusting:
                k_v[s] += a_thrust * vv / math.sqrt(vv @ vv)
        r = r + h / 6 * (k_r[0] + 2 * k_r[1] + 2 * k_r[2] + k_r[3])
        v = v + h / 6 * (k_v[0] + 2 * k_v[1] + 2 * k_v[2] + k_v[3])
        t += h
        if thrusting:
            dv += a_thrust * h
            if 0.5 * (v @ v) - MU_E / math.sqrt(r @ r) >= 0.0:  # C3 >= 0: escaped
                thrusting, escaped = False, True
        if n < pts.shape[0]:
            pts[n] = r
            n += 1
        if t > t_max:
            break
    out[0], out[1], out[2], out[3] = t, dv, 1.0 if escaped else 0.0, n
    return r, v
