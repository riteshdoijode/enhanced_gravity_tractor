"""Gravity-tractor tow, B-plane trim guidance and the nonlinear verification (Step 7, keyhole scenario).

Deflection state. Bennu's deviation from its (scenario) orbit dx = [dr, dv] is driven by impulses (the
impacts) and by the continuous pull of the tractor, and is propagated with the state-transition matrix:
    dx(t2) = Phi(t2, t1) dx(t1) + int_t1^t2 Phi(t2, tau) [0; a(tau)] dtau
evaluated with the midpoint rule on steps of at most one day. The tractor's pull on Bennu is its gravity,
    a(tau) = G m_T rho / |rho|^3        (rho: spacecraft relative to Bennu, Lu & Love 2005)

B-plane sensitivity. A push dv at time t moves the 2135 flyby point by  db = B(t) dv,
    B(t) = [xi; zeta] Phi_rv(t_ca, t)     (2 x 3)
Trim guidance (closed outer loop, re-evaluated at least weekly on the latest deflection estimate), all in
shifts relative to the scenario point b0:
  * "aim" (default: place the flyby at the DART aim point b_aim): tow along the steepest descent of the
    error, e = B(t)^T (b_aim - b) / |...|, for the time that minimizes the error along e,
    tau = (err . B e) / (a |B e|^2), then stop: the tractor parks 5 km out where its pull is ~1 %.
  * "maximize": push along the current miss direction w = b / |b|, e(t) = B(t)^T w / |B(t)^T w|.

Nonlinear verification. The scenario orbit is linear in a tiny offset before t_pre (60 days before the 2135
flyby) and nonlinear after it (core/scenario.py). Bennu is propagated with the full force model twice from
just before the first push to t_pre, with and without the recorded impulses and tow; their difference is
added to the scenario state at t_pre and that orbit is flown through the 2135 flyby to 2185 (stopping at
Earth's surface if it hits). Re-integrating the undeflected orbit from 2026 instead would not work: a
millimetre of integration difference before 2060 moves the 2135 point by kilometres.
"""

from concurrent.futures import Future

import numpy as np

from core.constants import DAY, G, YEAR
from core.dynamics.nominal import symplectic_inverse


class Deflection:
    """Bennu's deviation from the nominal orbit (barycentric axes), propagated with the STM.

    Stored as the epoch-referred deviation z = Phi(t, t0)^-1 dx(t), which is constant while Bennu coasts
    and changes only when it is pushed:  dz = Phi(tau, t0)^-1 [0; dv].  Then dx(t) = Phi(t, t0) z,
    vectorized over any array of times; the history of z answers queries in the past exactly."""

    def __init__(self, nominal, t0):
        self.nom = nominal
        self.t = float(t0)
        self.z = np.zeros(6)
        self._ht = np.full(1024, np.inf)  # checkpoint times and z values (arrays grown by doubling)
        self._hz = np.zeros((1024, 6))
        self._ht[0] = self.t
        self._n = 1
        self.active = False
        self.tow_dv = 0.0  # integral of |a| dt (display)
        self.tow_log = []  # (t_mid, a, h) samples (for the nonlinear check)
        self.impulses = []  # (t, dv)

    def _inv0(self, t):
        return symplectic_inverse(self.nom.Phi0(t))

    def Phi(self, t2, t1):
        return self.nom.Phi0(t2) @ self._inv0(t1)

    def _push(self, tau, dv):
        self.z = self.z + self._inv0(tau)[:, 3:] @ np.asarray(dv, float)
        self.active = True

    def _checkpoint(self):
        n = self._n
        if self.t > self._ht[n - 1]:
            if n == len(self._ht):
                self._ht = np.concatenate([self._ht, np.full(n, np.inf)])
                self._hz = np.vstack([self._hz, np.zeros((n, 6))])
            self._ht[n] = self.t
            self._hz[n] = self.z
            self._n = n + 1
        else:
            self._hz[n - 1] = self.z

    @property
    def dx(self):
        return self.nom.Phi0(self.t) @ self.z

    def states(self, t):
        """dx at times t (array), using the push history up to each time."""
        t = np.atleast_1d(np.asarray(t, float))
        if not self.active:
            return np.zeros((len(t), 6))
        n = self._n
        k = np.clip(np.searchsorted(self._ht[:n], t, side="right") - 1, 0, n - 1)
        Z = self._hz[k]
        Z[t >= self.t] = self.z
        return np.einsum("nij,nj->ni", self.nom.Phi0(t), Z)

    def state_at(self, t):
        return self.states(t)[0]

    def advance(self, t1, accel=None):
        """Coast (accel None) or tow with accel(t) -> a [m/s^2] from the current time to t1."""
        t1 = float(t1)
        if accel is None:
            self.t = max(self.t, t1)
            return
        while self.t < t1 - 1e-6:
            h = min(t1 - self.t, DAY)
            tm = self.t + h / 2
            a = np.asarray(accel(tm), float)
            if np.any(a):
                self._push(tm, a * h)
                self.tow_dv += float(np.linalg.norm(a)) * h
                self.tow_log.append((tm, a.copy(), h))
            self.t += h
            self._checkpoint()

    def impulse(self, t_imp, dv):
        """Velocity change at t_imp (exact for any time, also slightly in the past)."""
        self.impulses.append((float(t_imp), np.asarray(dv, float).copy()))
        self._push(t_imp, dv)
        self.t = max(self.t, float(t_imp))
        self._checkpoint()


def pull_on_bennu(m_tractor, rho):
    """Gravitational acceleration of Bennu toward the spacecraft at rho (relative position)."""
    r = np.linalg.norm(rho)
    return G * m_tractor * rho / r**3


class TrimGuidance:
    """Outer-loop B-plane guidance for the tow (see module docstring). Shifts are relative to b0."""

    def __init__(self, planner, mode, aim_shift=None, period=7 * DAY):
        self.pl = planner
        self.mode = mode  # "aim" or "maximize"
        self.period = period
        self.b_aim = None if aim_shift is None else np.asarray(aim_shift, float)
        self.last = -np.inf
        self.next = -np.inf  # time of the next re-plan
        self.e = None
        self.tow_on = True
        self.dv_needed = np.nan
        self.history = []  # (t, b_now, b_pred)

    def B(self, t):
        return self.pl.bplane.basis @ self.pl.Phi_rv_to_encounter(np.atleast_1d(t))[0]

    def b_now(self, defl, t):
        """Encounter shift if nothing more is done (coast from the current deflection)."""
        dx_ca = defl.Phi(self.pl.t_ca, t) @ defl.state_at(t)
        return self.pl.bplane.project(dx_ca[:3])

    def direction(self, t, b):
        """Tow direction at time t for the current law."""
        Bt = self.B(t)
        if self.mode == "aim":
            if self.e is None:
                return None
            return self.e
        w = b / np.linalg.norm(b) if np.linalg.norm(b) > 1.0 else np.linalg.svd(Bt)[0][:, 0]
        g = Bt.T @ w
        return g / np.linalg.norm(g)

    def update(self, defl, t, a_mag, t_end):
        """Weekly re-plan: returns (tow_on, unit direction)."""
        b = self.b_now(defl, t)
        if self.mode == "aim":
            err = self.b_aim - b
            Bt = self.B(t)
            g = Bt.T @ err  # steepest descent of |b_aim - b|^2: never wastes the tow on weak directions
            self.e = g / max(np.linalg.norm(g), 1e-30)
            Be = Bt @ self.e
            tau = max(0.0, err @ Be) / max(a_mag * (Be @ Be), 1e-30)  # tow time minimizing the error along e
            self.dv_needed = a_mag * tau
            self.tow_on = tau > 0.5 * DAY and t < t_end  # less than half a day of tow left: stop
            # deadbeat: tow exactly the time needed (re-checked at least weekly), then re-evaluate
            self.next = t + (min(self.period, max(tau, 0.25 * DAY)) if self.tow_on else self.period)
        else:
            self.tow_on = t < t_end
            self.e = self.direction(t, b)
            self.next = t + self.period
        self.last = t
        self.history.append((t, b, self.predict(defl, t, a_mag, t_end)))
        return self.tow_on, self.e

    def predict(self, defl, t, a_mag, t_end, n=200):
        """Predicted final encounter shift if the law is followed to t_end."""
        b = self.b_now(defl, t)
        if t >= t_end or a_mag <= 0:
            return b
        if self.mode == "aim":
            Bt = self.B(t)
            dv = np.linalg.pinv(Bt) @ (self.b_aim - b)
            reach = a_mag * (t_end - t)
            if np.linalg.norm(dv) <= reach:
                return self.b_aim.copy()
            g = Bt.T @ (self.b_aim - b)
            return b + Bt @ (g / np.linalg.norm(g) * reach)
        ts = np.linspace(t, t_end, n)
        Bs = np.einsum("ij,njk->nik", self.pl.bplane.basis, self.pl.Phi_rv_to_encounter(ts))
        w = b / np.linalg.norm(b) if np.linalg.norm(b) > 1.0 else np.linalg.svd(Bs[0])[0][:, 0]
        g = np.einsum("nij,i->nj", Bs, w)
        e = g / np.linalg.norm(g, axis=1, keepdims=True)
        push = np.trapezoid(np.einsum("nij,nj->ni", Bs, e), ts, axis=0) * a_mag
        return b + push


def cw_along_track(a_T, n, t):
    """Clohessy-Wiltshire along-track displacement under a constant along-track acceleration a_T in a
    circular orbit of mean motion n (lecture 3):  y = (4 a_T / n^2)(1 - cos nt) - (3/2) a_T t^2."""
    return 4 * a_T / n**2 * (1 - np.cos(n * t)) - 1.5 * a_T * t**2


def tow_years_left(t, t_end):
    return max(0.0, (t_end - t) / YEAR)


def _tow_blocks(defl):
    """Tow history merged into blocks of at most 7 days (same velocity change per block)."""
    starts, ends, accs = [], [], []
    for tm, a, h in defl.tow_log:
        a0, a1 = tm - h / 2, tm + h / 2
        if starts and a0 - ends[-1] < 1.0 and a1 - starts[-1] <= 7 * DAY:
            accs[-1] = accs[-1] + a * h
            ends[-1] = a1
        else:
            starts.append(a0), ends.append(a1), accs.append(a * h)
    starts, ends = np.array(starts), np.array(ends)
    accs = np.array(accs) / (ends - starts)[:, None] if len(starts) else np.zeros((0, 3))
    return starts, ends, accs


def _segments(nom, defl, t_pre):
    """The run plan of deflected_state_at_pre: start time and state, segment edges, impulses, tow blocks (None if
    nothing has pushed Bennu)."""
    t_pushes = [t for t, _ in defl.impulses] + [tm - h / 2 for tm, _, h in defl.tow_log]
    if not t_pushes:
        return None
    t0 = min(t_pushes) - DAY
    starts, ends, accs = _tow_blocks(defl)
    kick_t = np.array([float(ti) for ti, _ in defl.impulses])
    kick_dv = np.array([np.asarray(dv, float) for _, dv in defl.impulses]).reshape(-1, 3)
    edges = np.unique(np.concatenate([[t0, t_pre], kick_t, starts, ends]))
    edges = edges[(edges >= t0) & (edges <= t_pre)]
    edges = edges[np.concatenate([[True], np.diff(edges) > 1e-3])]  # merge edges closer than 1 ms
    return {"x0": nom.state(t0), "edges": edges, "kick_t": kick_t, "kick_dv": kick_dv, "starts": starts,
            "ends": ends, "accs": accs}


def fly_segments(eph, seg, rtol=1e-12, pushes=True):
    """One run of deflected_state_at_pre through the segment edges, with (or without) the pushes; the state at
    the last edge. A module-level function so that a core/fullmodel.py worker can run it."""
    from core.dynamics.asteroid import BennuDynamics

    edges, kick_t, starts, ends, accs = seg["edges"], seg["kick_t"], seg["starts"], seg["ends"], seg["accs"]
    acc_now = {"a": np.zeros(3)}
    dyn = BennuDynamics(eph, extra_accel=(lambda t, r, v: acc_now["a"]) if pushes else None)
    x = seg["x0"]
    applied = np.zeros(len(kick_t), bool)
    for a, b in zip(edges[:-1], edges[1:]):
        if pushes:
            due = ~applied & (np.abs(kick_t - a) < 1e-2)
            if due.any():  # every impulse is applied exactly once, at the segment edge it falls on
                x = x + np.concatenate([np.zeros(3), seg["kick_dv"][due].sum(0)])
                applied |= due
            k = int(np.searchsorted(starts, a + 1e-3, side="right")) - 1
            acc_now["a"] = accs[k] if k >= 0 and a < ends[k] - 1e-3 else np.zeros(3)
        x = dyn.propagate(a, x, [b], rtol=rtol, atol=1e-6)[0][-1]
    if pushes and not applied.all():
        raise RuntimeError("an impulse fell outside the propagated segments")
    return x


def deflected_states_at_pre(eph, nom, defls, t_pre, rtol=1e-12, cache=None, pool=None):
    """Nonlinear difference the pushes make at t_pre, added to the scenario state there, for each deflection.

    The two propagations (with and without the pushes) stop and restart at the same times (every impulse and
    every tow-block edge). Their adaptive steps still differ a little, and the 2060 flyby amplifies the
    resulting centimetres into a noise floor of ~150 km at the 2135 flyby (0.04 % of the Moon's distance; a
    fixed common grid brings it below 1 km but is 10x slower). The run without pushes depends only on the edges:
    it is shared between deflections with the same edges and kept in `cache`. On a core/fullmodel.py `pool`
    all runs go at once, one per worker."""
    from core.fullmodel import Inline

    pool = pool or Inline(eph)
    x_pre = nom.state(t_pre)
    segs = [_segments(nom, d, t_pre) for d in defls]
    runs, without = [], {}
    for s in segs:
        if s is None:
            runs.append(None)
            continue
        key = (tuple(np.round(s["edges"], 3)), rtol)
        runs.append((key, pool.submit(fly_segments, s, rtol, True)))
        if key not in without:
            hit = None if cache is None else cache.get(key)
            without[key] = hit if hit is not None else pool.submit(fly_segments, s, rtol, False)
    out = []
    for run in runs:
        if run is None:
            out.append((x_pre, np.zeros(6)))
            continue
        key, f = run
        if isinstance(without[key], Future):
            without[key] = without[key].result()
            if cache is not None:
                cache[key] = without[key]
        delta = f.result() - without[key]
        out.append((x_pre + delta, delta))
    return out


def deflected_state_at_pre(eph, nom, defl, t_pre, rtol=1e-12, cache=None, pool=None):
    return deflected_states_at_pre(eph, nom, [defl], t_pre, rtol=rtol, cache=cache, pool=pool)[0]


def flyby_and_tail(eph, x_pre, t_pre, t_end, t_ca, basis, to_end, rtol=1e-12):
    """The deflected orbit flown from t_pre (stopping at Earth's surface): its 2135 flyby (time, distance,
    B-plane point) and, with to_end, the impact time, the approaches inside 0.1 au and a daily track."""
    from scipy.optimize import minimize_scalar

    from core.constants import AU
    from core.scenario import find_close_approaches, integrate_tail

    tr, t_imp, _ = integrate_tail(eph, x_pre, None, t_pre, t_end, rtol=rtol)
    state = tr.state
    t_stop = t_imp if np.isfinite(t_imp) else tr.t_end
    f = lambda t: float(np.linalg.norm(state(t)[:3] - eph.pos("earth", t)))
    res = minimize_scalar(f, bounds=(t_ca - 5 * DAY, min(t_ca + 5 * DAY, t_stop - 1.0)), method="bounded",
                          options={"xatol": 1.0})
    r_rel = state(res.x)[:3] - eph.pos("earth", res.x)
    out = {"b_nonlinear": basis @ r_rel, "q": float(res.fun), "t_flyby": float(res.x), "t_impact": t_imp}
    if to_end:
        out["approaches"] = find_close_approaches(state, eph, t_pre, t_stop - DAY, max_dist=0.1 * AU, step=0.25 * DAY)
        tt = np.arange(t_pre, t_stop, DAY)
        out["track_t"], out["track_x"] = tt, state(tt)
    return out


def verify_nonlinear_many(eph, nom, defls, planner, rtol=1e-12, to_end=True, t_end=None, cache=None, pool=None):
    """verify_nonlinear for several deflections; on a core/fullmodel.py `pool` their runs go in parallel."""
    import time

    from core.constants import LUNAR_DISTANCE
    from core.fullmodel import Inline

    wall = time.perf_counter()
    pool = pool or Inline(eph)
    sc = planner.sc
    t_pre, t_ca = sc.t_pre, planner.t_ca
    if t_end is None:
        t_end = (min(eph.span("earth")[1], nom.span[1] + 3 * YEAR) - 2 * DAY) if to_end else t_ca + 20 * DAY
    pres = deflected_states_at_pre(eph, nom, defls, t_pre, rtol=rtol, cache=cache, pool=pool)
    tails = [pool.submit(flyby_and_tail, x_pre, t_pre, t_end, t_ca, planner.bplane.basis, to_end, rtol)
             for x_pre, _ in pres]
    outs = []
    for defl, (_, delta), f in zip(defls, pres, tails):
        out = f.result()
        b_nl = out["b_nonlinear"]
        dx_lin = defl.Phi(t_ca, defl.t) @ defl.state_at(defl.t)
        b_lin = planner.b0 + planner.bplane.project(dx_lin[:3])
        diff = float(np.linalg.norm(b_nl - b_lin))
        shift_lin = float(np.linalg.norm(b_lin - planner.b0))
        out.update({"b_linear": b_lin, "q_ld": out["q"] / LUNAR_DISTANCE, "difference_km": diff / 1e3,
                    "relative": diff / max(shift_lin, 1.0), "delta_pre": delta})
        if to_end:
            out["approaches"] = [a for a in out["approaches"] if a["t"] > t_ca + 30 * DAY]
            out["min_after"] = min([a["dist"] for a in out["approaches"]], default=np.inf)
        out["seconds"] = time.perf_counter() - wall
        outs.append(out)
    return outs


def verify_nonlinear(eph, nom, defl, planner, rtol=1e-12, to_end=True, t_end=None, cache=None, pool=None):
    """Full nonlinear check of the deflection (see module docstring). Returns the 2135 flyby (linear and
    nonlinear miss vectors, distance), whether and when Bennu hits Earth afterwards, all approaches inside
    0.1 au, and a daily track of the deflected orbit after t_pre (for display). With a core/fullmodel.py
    `pool` the runs with and without the pushes go in parallel (~2x faster)."""
    return verify_nonlinear_many(eph, nom, [defl], planner, rtol=rtol, to_end=to_end, t_end=t_end, cache=cache,
                                 pool=pool)[0]
