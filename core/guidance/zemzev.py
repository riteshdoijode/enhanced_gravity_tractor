"""Generalized zero-effort-miss / zero-effort-velocity (ZEM/ZEV) feedback guidance.
Guo, Hawkins & Wie, J. Guidance Control Dyn. 36(3):810 (2013).

With t_go = t_f - t and (r_c, v_c) the state coasted (Keplerian, no thrust) to t_f:
    ZEM = r_target(t_f) - r_c,   ZEV = v_target(t_f) - v_c
    a_cmd = 6 ZEM / t_go^2 - 2 ZEV / t_go           (energy-optimal in uniform gravity)
saturated at the available acceleration T/m. Because it re-plans from the current state every step,
throttle changes and thrust outages (solar-storm safe mode) are absorbed automatically.

Minimum arrival time for a given thrust: the earliest t_f for which the saturated closed loop still
reaches the target (bisection after a coarse scan) -- more thrust -> earlier arrival.
"""

import numpy as np

from core.guidance import kepler
from core.guidance.kepler import MU_SUN


def command(r, v, rT, vT, t_go, mu=MU_SUN):
    rc, vc = kepler.propagate(r, v, t_go, mu=mu)
    tg = np.asarray(t_go, float)[..., None]
    return 6.0 * (rT - rc) / tg**2 - 2.0 * (vT - vc) / tg


def saturate(a, a_max):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a * np.minimum(1.0, np.asarray(a_max, float)[..., None] / np.maximum(n, 1e-30)) if np.ndim(a_max) \
        else a * np.minimum(1.0, a_max / np.maximum(n, 1e-30))


def _accel_grav(r, mu):
    return -mu * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3


def simulate(r0, v0, t0, tf, target, a_max, n_steps=200, mu=MU_SUN, record=False, outage=None):
    """Closed-loop two-body + thrust simulation for one or many candidate arrival times tf (vectorized).

    target(t) -> (r, v) of the rendezvous point (vectorized over t). outage(t) -> True while thrust is off.
    Returns dict with terminal position/velocity miss, delta-v used, saturation fraction, [trajectory].
    """
    tf = np.atleast_1d(np.asarray(tf, float))
    K = len(tf)
    r = np.broadcast_to(np.asarray(r0, float), (K, 3)).copy()
    v = np.broadcast_to(np.asarray(v0, float), (K, 3)).copy()
    rT, vT = target(tf)
    h = (tf - t0) / n_steps
    dv = np.zeros(K)
    sat = np.zeros(K)
    traj = [r.copy()] if record else None
    acc = [] if record else None
    for i in range(n_steps):
        t = t0 + i * h
        t_go = tf - t
        a = command(r, v, rT, vT, t_go, mu)
        a_norm = np.linalg.norm(a, axis=1)
        sat += a_norm > a_max * (1 + 1e-9)
        a = saturate(a, a_max)
        if outage is not None:
            a = a * (~np.array([outage(ti) for ti in np.atleast_1d(t)]))[:, None] if np.ndim(t) else a * (not outage(t))
        dv += np.linalg.norm(a, axis=1) * h
        hh = h[:, None]

        def f(rr, vv):
            return vv, _accel_grav(rr, mu) + a

        k1r, k1v = f(r, v)
        k2r, k2v = f(r + hh / 2 * k1r, v + hh / 2 * k1v)
        k3r, k3v = f(r + hh / 2 * k2r, v + hh / 2 * k2v)
        k4r, k4v = f(r + hh * k3r, v + hh * k3v)
        r = r + hh / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if record:
            traj.append(r.copy())
            acc.append(a.copy())
    out = {"tf": tf, "miss_r": np.linalg.norm(r - rT, axis=1), "miss_v": np.linalg.norm(v - vT, axis=1),
           "dv": dv, "saturated_fraction": sat / n_steps, "r_final": r, "v_final": v}
    if record:
        out["trajectory"] = np.stack(traj, axis=1)  # (K, n_steps + 1, 3)
        out["accel"] = np.stack(acc, axis=1)
        out["times"] = t0 + h[:, None] * np.arange(n_steps + 1)[None, :]
    return out


def min_time_rendezvous(r0, v0, t0, target, a_max, t_lo, t_hi, tol_r=1.0e6, tol_v=20.0, n_scan=40,
                        n_steps=160, refine=7, mu=MU_SUN):
    """Earliest feasible arrival time in [t0 + t_lo, t0 + t_hi] and the corresponding closed-loop run.

    Feasible = terminal miss < tol_r [m] and relative speed < tol_v [m/s] (handover to proximity ops).
    Returns None if nothing in the window is feasible.
    """
    def feasible(res):
        return (res["miss_r"] < tol_r) & (res["miss_v"] < tol_v)

    cands = t0 + np.geomspace(t_lo, t_hi, n_scan)
    res = simulate(r0, v0, t0, cands, target, a_max, n_steps, mu)
    ok = feasible(res)
    if not ok.any():
        return None
    j = int(np.argmax(ok))
    lo = t0 + t_lo * 0.98 if j == 0 else cands[j - 1]
    hi = cands[j]
    for _ in range(refine):  # bisection between the last infeasible and first feasible candidate
        mid = 0.5 * (lo + hi)
        if feasible(simulate(r0, v0, t0, [mid], target, a_max, n_steps, mu))[0]:
            hi = mid
        else:
            lo = mid
    best = simulate(r0, v0, t0, [hi], target, a_max, n_steps, mu, record=True)
    return {k: (val[0] if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == 1 else val)
            for k, val in best.items()}
