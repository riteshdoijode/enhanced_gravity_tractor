"""Closed-loop, thrust-limited heliocentric rendezvous (depart - coast - brake), for any thrust level.

Plan from the current state (re-run after throttle changes, outages, and periodically):
    choose the arrival time t_f; braking at constant deceleration a_b = eta * a_max from relative
    speed V takes tau = V / a_b and covers V tau / 2, so the coast is aimed at the brake-start point
        r_aim = r_T(t_b) - (V tau / 2) v_rel_hat,   t_b = t_f - tau
    (V and v_rel_hat from the Lambert arc to r_aim, iterated). Feasible if the departure burn
    |v_lambert - v| / a_max fits in 90 % of the time before t_b. "Minimum time" = earliest feasible t_f.
Command:
    t < t_b : velocity-to-be-gained steering (Battin) toward the aim point -> thrust until v_g = 0, coast
    t >= t_b: ZEM/ZEV to the target at t_f. From the brake-start geometry (ZEM = -V tau / 2, ZEV = -V) the
              energy-optimal profile a = 6 ZEM/t^2 - 2 ZEV/t is a *constant* deceleration V / tau = a_b,
              so it stays unsaturated with margin while correcting errors (Kepler-based ZEM).
"""

import numpy as np

from core.constants import DAY, YEAR
from core.guidance import kepler, zemzev
from core.guidance.kepler import MU_SUN, ecliptic_pole_icrf

POLE = ecliptic_pole_icrf()
ETA_BRAKE = 0.85


class RendezvousGuidance:
    def __init__(self, target, a_max, deadline, n_rev=2):
        """target(t) -> (r, v) heliocentric (vectorized). deadline = latest acceptable arrival time."""
        self.target, self.a_max, self.deadline, self.n_rev = target, float(a_max), deadline, n_rev
        self.t_f = self.t_b = np.nan
        self.r_aim = None
        self.braking = False

    def _aim(self, t, r, v, tf, iterations=3):
        """Vectorized over candidate arrival times tf: returns t_b, r_aim, departure dv, feasibility."""
        tf = np.atleast_1d(tf)
        a_b = ETA_BRAKE * self.a_max
        rT, vT = self.target(tf)
        tb, raim = tf.copy(), rT.copy()
        for _ in range(iterations):
            V1, V2 = kepler.lambert_multirev(np.broadcast_to(r, raim.shape), raim, np.maximum(tb - t, 60.0),
                                             n_max=self.n_rev, prograde_axis=POLE)
            rTb, vTb = self.target(tb)
            dv1 = np.linalg.norm(V1 - v, axis=2)
            j = np.argmin(np.where(np.isfinite(dv1), dv1 + np.linalg.norm(vTb[None] - V2, axis=2), np.inf), axis=0)
            cols = np.arange(len(tf))
            v1, v2 = V1[j, cols], V2[j, cols]
            vrel = v2 - vTb
            V = np.linalg.norm(vrel, axis=1)
            tau = V / a_b
            tb = tf - tau
            raim = self.target(tb)[0] - (0.5 * V * tau)[:, None] * vrel / np.maximum(V, 1e-9)[:, None]
        dv_dep = np.linalg.norm(v1 - v, axis=1)
        feasible = np.isfinite(dv_dep) & (tb > t) & (dv_dep / self.a_max <= 0.9 * (tb - t))
        return tb, raim, dv_dep, V, feasible

    def plan(self, t, r, v, t_min=None):
        """Earliest feasible arrival from (t, r, v); returns t_f or NaN."""
        t_max = self.deadline - t
        if t_max <= DAY:
            self.t_f = np.nan
            return np.nan
        lo = max(DAY, (t_min - t) if t_min is not None else DAY)
        tofs = np.geomspace(lo, max(t_max, lo * 1.01), 220)
        tb, raim, dv, V, ok = self._aim(t, r, v, t + tofs)
        if not ok.any():
            self.t_f = np.nan
            return np.nan
        k = int(np.argmax(ok))
        a, b = (t + tofs[k - 1], t + tofs[k]) if k > 0 else (t + 0.5 * tofs[0], t + tofs[0])
        for _ in range(12):  # refine the earliest feasible arrival
            mid = 0.5 * (a + b)
            if self._aim(t, r, v, mid)[4][0]:
                b = mid
            else:
                a = mid
        tb, raim, _, _, _ = self._aim(t, r, v, b)
        self.t_f, self.t_b, self.r_aim = b, float(tb[0]), raim[0]
        self.braking = False
        return self.t_f

    def set_arrival(self, t, r, v, t_f):
        """Fly to a given arrival time (e.g. the mission planner's), computing the brake-start geometry."""
        tb, raim, _, _, _ = self._aim(t, r, v, t_f)
        self.t_f, self.t_b, self.r_aim = float(t_f), float(tb[0]), raim[0]
        self.braking = False
        return self.t_f

    def step_limit(self, t, r, v, h_max):
        """Largest sensible step: resolve burns (5 % of remaining burn) and the approach."""
        if not np.isfinite(self.t_f):
            return h_max
        rT, vT = self.target(np.array([t]))
        d, vr = np.linalg.norm(r - rT[0]), np.linalg.norm(v - vT[0])
        h = min(h_max, max(30.0, 0.02 * d / max(vr, 1e-3)), max(self.t_f - t, 30.0))
        if self.braking:
            h = min(h, max(30.0, 0.02 * (self.t_f - t)))
        return h

    def command(self, t, r, v, h):
        """Commanded acceleration (heliocentric frame) for the next step of length h."""
        if not np.isfinite(self.t_f):
            return np.zeros(3)
        if t < self.t_b and not self.braking:
            V1, _ = kepler.lambert_multirev(r, self.r_aim, max(self.t_b - t, 60.0), n_max=self.n_rev,
                                            prograde_axis=POLE)
            V1 = V1[:, 0] if V1.ndim == 3 else V1
            dv = np.linalg.norm(V1 - v, axis=1)
            if not np.isfinite(dv).any():
                return np.zeros(3)
            vg = V1[int(np.nanargmin(dv))] - v
            n = np.linalg.norm(vg)
            return vg / n * self.a_max if n > self.a_max * h else vg / max(h, 1.0)
        self.braking = True
        t_go = max(self.t_f - t, h)
        rT, vT = self.target(np.array([self.t_f]))
        a = zemzev.command(r[None], v[None], rT, vT, np.array([t_go]))[0]
        return zemzev.saturate(a, self.a_max)
