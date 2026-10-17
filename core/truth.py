"""Bennu's true orbit in the live run, integrated with the full force model as the mission goes on.

Ground truth. Before the first push Bennu is on the scenario orbit (core/scenario.py), which is itself a
full-force-model integration. From the first push on, Bennu is integrated with the full model (BennuDynamics:
Sun, planets, Moon, Pluto, 16 large asteroids, relativity, solar radiation pressure, Yarkovsky, Earth J2) with
every impulse it actually received (each DART impact: the momentum delivered on the real shape with the true
beta) and the gravity tractor's actual pull. Nothing about Bennu is replayed or linearized here.

Keyhole noise. The 2060 flyby amplifies integration noise ~4,000x and the 2135 flyby passes a 3 km keyhole, so
up to t_pre (60 days before the 2135 flyby) Bennu is flown twice in one solver, with and without the pushes
(they share every adaptive step), and their difference is added to the scenario state:
    x(t) = x_scenario(t) + [x_with(t) - x_without(t)]
The integration errors of the two runs are nearly equal and cancel, so a vanishing push leaves Bennu in the
keyhole. From t_pre the deflected orbit is flown directly through the 2135 flyby to 2185, stopping at Earth's
surface if it hits.

Segments. Stops are at the pushes (impulses, 30-day tow blocks) and 30 days, 1 year and then every 5 years after
the latest push, so the result depends only on the push history, never on when it was computed or on the
display speed. Each segment is one DOP853 solve (rtol 1e-12) sampled every 6 h for queries (cubic Hermite
between samples). The tractor's pull enters as block means: the block in progress is flown with its mean so far
and re-flown when it closes.

Work runs on a core/fullmodel.py pool (segments one after another in the background, ahead of the mission
clock) or inline. The mission asks `ensure(t)` before it steps to t.
`prediction()` is the orbit flown on to 2185 assuming no further pushes (what the ground team's orbit
determination predicts); it is computed when `speculate` is set.
"""

import threading

import numpy as np

from core.constants import DAY, YEAR

GRID = (30 * DAY, YEAR)  # segment ends after the latest push: +30 d, +1 yr, then every SEGMENT_MAX
SEGMENT_MAX = 5 * YEAR
NODE_DT = 6 * 3600.0  # samples for queries
TOW_BLOCK = 30 * DAY  # the tractor's pull enters the truth as means over these blocks
FINE_DT = 600.0  # samples near Earth (the 2135 flyby, an impact)


# ----------------------------------------------------------------------------------------- worker jobs
def _pair_rhs(eph, acc):
    """Equations of motion of the two runs together, y = [x_with (6), x_without (6)]: the full force model of
    BennuDynamics for both states, with the ephemeris looked up once per stage, plus the extra acceleration on
    the run with the pushes. One solver integrates both, so they share every step: their integration errors
    are nearly the same and cancel in the difference (a vanishing push leaves a vanishing difference)."""
    from core.constants import AU
    from core.dynamics.asteroid import BennuDynamics
    from core.dynamics.gravity import j2, schwarzschild
    from core.dynamics.perturbations import srp, yarkovsky

    dyn = BennuDynamics(eph)
    mus = np.array([mu for _, mu in dyn.grav.perturbers])[:, None]
    acc = np.asarray(acc, float)
    push = bool(np.any(acc))
    near2 = (0.05 * AU) ** 2

    def rhs(t, y):
        rb = dyn.grav.body_positions(t)
        sun_p, sun_v = eph.pos("sun", t), eph.vel("sun", t)
        earth_p = eph.pos("earth", t)
        out = np.empty(12)
        for k in (0, 6):
            r, v = y[k:k + 3], y[k + 3:k + 6]
            d = rb - r
            a = (mus * d / np.linalg.norm(d, axis=1)[:, None] ** 3).sum(0)
            rs, vs = r - sun_p, v - sun_v
            if dyn.gr:
                a = a + schwarzschild(rs, vs)
            if dyn.solar_pressure:
                a = a + srp(rs, dyn.amrat)
            if dyn.a2:
                a = a + yarkovsky(rs, vs, dyn.a2)
            if dyn.earth_j2:
                re = r - earth_p
                if re @ re < near2:
                    a = a + j2(re)
            if k == 0 and push:
                a = a + acc
            out[k:k + 3], out[k + 3:k + 6] = v, a
        return out

    return rhs


def fly_truth_segment(eph, xw, x0, t_a, t_b, accel, rtol=1e-12):
    """One segment of both runs (with and without the pushes): one DOP853 solve of the paired equations from
    t_a to t_b. Returns the sample times (every NODE_DT and the end) and the states of both runs (the last ones
    exact)."""
    from scipy.integrate import solve_ivp

    y0 = np.concatenate([np.asarray(xw, float), np.asarray(x0, float)])
    sol = solve_ivp(_pair_rhs(eph, accel), (t_a, t_b), y0, method="DOP853", rtol=rtol, atol=1e-6, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    n = max(1, int(np.ceil((t_b - t_a) / NODE_DT)))
    tn = np.linspace(t_a, t_b, n + 1)
    yn = np.vstack([sol.sol(tn[:-1]).T, sol.y[:, -1]])
    return tn, yn[:, :6], yn[:, 6:]


def fly_truth_tail(eph, x_pre, t_pre, t_end, t_ca, basis, rtol=1e-12):
    """The deflected orbit from t_pre through the 2135 flyby to t_end (stopping at Earth's surface): daily
    samples, 10-minute samples within 3 days of every pass inside 0.05 au, the flyby and any impact."""
    from scipy.optimize import minimize_scalar

    from core.constants import AU, LUNAR_DISTANCE
    from core.scenario import find_close_approaches, integrate_tail

    tr, t_imp, _ = integrate_tail(eph, x_pre, None, t_pre, t_end, rtol=rtol)
    state = tr.state
    t_stop = t_imp if np.isfinite(t_imp) else tr.t_end
    f = lambda t: float(np.linalg.norm(state(t)[:3] - eph.pos("earth", t)))
    res = minimize_scalar(f, bounds=(t_ca - 5 * DAY, min(t_ca + 5 * DAY, t_stop - 1.0)), method="bounded",
                          options={"xatol": 1.0})
    r_rel = state(res.x)[:3] - eph.pos("earth", res.x)
    apps = find_close_approaches(state, eph, t_pre, t_stop - DAY, max_dist=0.1 * AU, step=0.25 * DAY)
    tt = np.append(np.arange(t_pre, t_stop, DAY), t_stop)
    fine_t = [np.arange(max(t_pre, a["t"] - 3 * DAY), min(t_stop, a["t"] + 3 * DAY), FINE_DT)
              for a in apps if a["dist"] < 0.05 * AU] + [np.arange(max(t_pre, t_stop - 3 * DAY), t_stop, FINE_DT)]
    fine_t = np.unique(np.concatenate(fine_t + [[t_stop]]))
    return {"b_nonlinear": basis @ r_rel, "q": float(res.fun), "q_ld": float(res.fun / LUNAR_DISTANCE),
            "t_flyby": float(res.x), "t_impact": t_imp, "approaches": apps, "track_t": tt, "track_x": state(tt),
            "fine_t": fine_t, "fine_x": state(fine_t), "t_stop": t_stop}


# ----------------------------------------------------------------------------------------- truth
def _hermite(t, t0, t1, x0, x1):
    """Cubic Hermite interpolation of states [r, v] (velocity = derivative of position)."""
    h = t1 - t0
    s = (t - t0) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    r = h00 * x0[:3] + h10 * h * x0[3:] + h01 * x1[:3] + h11 * h * x1[3:]
    d00, d10, d01, d11 = (6 * s**2 - 6 * s) / h, 3 * s**2 - 4 * s + 1, (-6 * s**2 + 6 * s) / h, 3 * s**2 - 2 * s
    v = d00 * x0[:3] + d10 * x0[3:] + d01 * x1[:3] + d11 * x1[3:]
    return np.concatenate([r, v])


def _same_push(p, q):
    if p[0] != q[0] or abs(p[1] - q[1]) > 1e-6:
        return False
    if p[0] == "imp":
        return np.allclose(p[2], q[2], rtol=1e-12, atol=1e-15)
    return abs(p[2] - q[2]) < 1e-6 and np.allclose(p[3], q[3], rtol=1e-9, atol=1e-18)


class BennuTruth:
    def __init__(self, eph, nom, sc, basis, t_ca, pool=None, rtol=1e-12):
        self.eph, self.nom, self.sc = eph, nom, sc
        self.basis, self.t_ca, self.rtol = np.asarray(basis, float), float(t_ca), rtol
        self.t_pre = float(sc.t_pre)
        self.t_end = min(eph.span("earth")[1], nom.span[1] + 3 * YEAR) - 2 * DAY
        self.pool = pool  # core/fullmodel.py pool; None: inline (each piece computed when asked for)
        self.pushes = []  # ("imp", t, dv) | ("tow", t0, duration, a_mean), in time order (tow blocks once closed)
        self.segs = []  # dict(a, b, tn, xw (n, 6), x0 (n, 6))
        self.t_first = None  # first push (impulse or start of the tow)
        self.tail = None  # fly_truth_tail result (with "n_pushes")
        self.speculate = False  # fly ahead to the 2135 flyby and 2185 (the prediction)
        self.target = -np.inf
        self.error = None
        self.closed = False
        self._lock = threading.RLock()
        self._gen = 0
        self._job = None  # ("seg", gen, a, b, fw, f0) | ("tail", gen, f, n_pushes)
        self._open = None  # tow block in progress: [t0, integral of a dt, time reached]
        self._old = None  # push history of the run being re-simulated
        self._cursor = 0
        self._wake = threading.Event()
        self._driver = None

    # ----------------------------------------------------------------------------- pushes
    def impulse(self, t, dv):
        with self._lock:
            self._push(("imp", float(t), np.asarray(dv, float).copy()))

    def tow(self, t0, t1, a):
        """The tractor pulled Bennu with mean acceleration a over [t0, t1] (one mission step)."""
        a = np.asarray(a, float)
        with self._lock:
            t0, t1 = float(t0), float(t1)
            while t1 > t0 + 1e-9:
                if self._open is None:
                    self._open = [t0, np.zeros(3), t0]
                    if self.t_first is None:
                        self._start(t0)
                b0 = self._open[0]
                te = min(t1, b0 + TOW_BLOCK)
                self._open[1] = self._open[1] + a * (te - t0)
                self._open[2] = te
                if te >= b0 + TOW_BLOCK - 1e-6:  # block closed: its mean is final
                    integ = self._open[1]
                    self._open = None
                    self._push(("tow", b0, TOW_BLOCK, integ / TOW_BLOCK))
                t0 = te

    def tow_end(self, t):
        """The tow is over: the block in progress closes now with what it delivered."""
        with self._lock:
            if self._open is not None:
                b0, integ, _ = self._open
                self._open = None
                dur = float(t) - b0
                if dur > 1.0:
                    self._push(("tow", b0, dur, integ / dur))
                else:
                    self._truncate(b0)

    def _start(self, t):
        self.t_first = t
        self.segs = []

    def _push(self, p):
        if self._old is not None:  # replaying a re-simulation: keep the work done for identical pushes
            if self._cursor < len(self._old) and _same_push(p, self._old[self._cursor]):
                self._cursor += 1
                self.pushes.append(p)
                if self.t_first is None:
                    self.t_first = p[1]
                return
            t_cut = p[1] if self._cursor >= len(self._old) else min(p[1], self._old[self._cursor][1])
            self._old = None
            self._truncate(t_cut)
        else:
            self._truncate(p[1])
        if self.t_first is None or p[1] < self.t_first:
            self._start(p[1])
        self.pushes.append(p)
        self._wake.set()

    def _truncate(self, t):
        """Forget what was flown after t."""
        self._gen += 1
        self._job = None
        self.tail = None
        self.error = None
        if self.t_first is not None and t <= self.t_first + 1e-6:
            self.segs = []
        else:
            self.segs = [s for s in self.segs if s["b"] <= t + 1e-6]

    def begin_replay(self):
        """A re-simulation starts over: its pushes are matched against this history, and the work for
        identical pushes is kept."""
        with self._lock:
            if self._open is not None:  # a tow block in progress was flown with a provisional mean
                self._truncate(self._open[0])
                self._open = None
            self._old, self._cursor = list(self.pushes), 0
            self.pushes = []
            if not self._old:
                self._old = None
                self.t_first = None
                self._truncate(-np.inf)

    def _check_replay(self, t):
        """The replay went past a push of the old history that did not happen: forget from there."""
        if self._old is None:
            return
        if self._cursor < len(self._old) and t > self._old[self._cursor][1] + 1e-6:
            t_cut = self._old[self._cursor][1]
            self._old = None
            if not self.pushes and self._open is None:
                self.t_first = None
                self._truncate(-np.inf)
            else:
                self._truncate(t_cut)
        elif self._cursor >= len(self._old):
            self._old = None

    # ----------------------------------------------------------------------------- integration
    def _segment_after(self, a):
        """End of the segment that starts at a, and its constant extra acceleration."""
        starts = [p[1] for p in self.pushes if p[1] <= a + 1e-6] + ([self._open[0]] if self._open is not None
                                                                     and self._open[0] <= a + 1e-6 else [])
        tp = max(starts) if starts else self.t_first
        grid = [tp + g for g in GRID] + [tp + GRID[-1] + SEGMENT_MAX * (np.floor((a - tp - GRID[-1]) / SEGMENT_MAX
                                                                                + 1e-9) + 1)]
        b = min([g for g in grid if g > a + 1e-6] + [self.t_pre])
        acc = np.zeros(3)
        blocks = [(p[1], p[1] + p[2], p[3]) for p in self.pushes if p[0] == "tow"]
        if self._open is not None:
            el = self._open[2] - self._open[0]
            prov = self._open[1] / el if el > 0 else (blocks[-1][2] if blocks else np.zeros(3))
            blocks.append((self._open[0], self._open[0] + TOW_BLOCK, prov))
        for t0, t1, am in blocks:
            if t0 <= a + 1e-6 < t1 - 1e-6:
                acc, b = am, min(b, t1)
            elif a + 1e-6 < t0 < b:
                b = t0
        for p in self.pushes:
            if p[0] == "imp" and a + 1e-6 < p[1] < b:
                b = p[1]
        return b, acc

    def horizon(self):
        """Bennu's true state is known up to this time."""
        if self.t_first is None:
            return np.inf
        if self.segs and self.segs[-1]["b"] >= self.t_pre - 1e-6:
            return self.tail["t_stop"] if self.tail is not None else self.t_pre
        return self.segs[-1]["b"] if self.segs else self.t_first

    def ready(self, t):
        return t <= self.horizon() + 1e-6

    def ensure(self, t):
        """Ask for Bennu's state up to t; True once it is known (inline: computed now)."""
        with self._lock:
            self._check_replay(t)
            self.target = max(self.target, float(t))
        if self._inline():
            self.pump(block=True, goal=self.target)
        else:
            self._kick()
        return self.ready(t)

    def want_prediction(self, on=True):
        """Fly ahead to 2185 in the background whenever there is nothing else to do (inline: only on request)."""
        self.speculate = bool(on)
        if on and not self._inline():
            self._kick()

    def towing(self):
        """A tow block is in progress (its pushes are not final, so no prediction can be made for it)."""
        return self._open is not None

    def request_prediction(self):
        """The prediction is needed now: inline it is computed here; otherwise it is under way (None until done).
        None while a tow block is in progress."""
        if self._open is not None:
            return None
        self.speculate = True
        if self._inline():
            self.pump(block=True)
        else:
            self._kick()
        return self.prediction()

    def _inline(self):
        return self.pool is None or getattr(self.pool, "inline", False)

    def _kick(self):
        if self._driver is None:
            self._driver = threading.Thread(target=self._drive, name="bennu-truth", daemon=True)
            self._driver.start()
        self._wake.set()

    def _drive(self):
        while not self.closed:
            self._wake.wait(0.25)
            self._wake.clear()
            try:
                self.pump()
            except Exception as exc:  # report, never kill the simulation
                self.error = f"Bennu's orbit integration failed: {exc}"

    def close(self):
        self.closed = True
        self._wake.set()

    def pump(self, block=False, goal=None):
        """Collect finished work and submit the next piece toward the goal (default: the target, or 2185 when
        speculating)."""
        while not self.closed:
            with self._lock:
                self._collect()
                if self.t_first is None or self.error is not None:
                    return
                if self._job is None:
                    self._next_job(goal)
                job = self._job
                if job is None:
                    return
            if not block:
                return
            job[4 if job[0] == "seg" else 2].exception()  # waits

    def _next_job(self, goal=None):
        if goal is None:
            goal = self.t_end if self.speculate else self.target
        a = self.segs[-1]["b"] if self.segs else self.t_first
        pool = self.pool
        if pool is None:
            from core.fullmodel import Inline

            pool = self.pool = Inline(self.eph)
        if a < min(goal, self.t_pre) - 1e-6:
            b, acc = self._segment_after(a)
            if self.segs:
                xw, x0 = self.segs[-1]["xw"][-1], self.segs[-1]["x0"][-1]
            else:
                xw = x0 = self.nom.state(self.t_first)
            kick = sum((p[2] for p in self.pushes if p[0] == "imp" and abs(p[1] - a) < 1e-6), np.zeros(3))
            xw = xw + np.concatenate([np.zeros(3), kick])
            f = pool.submit(fly_truth_segment, xw, x0, a, b, acc, self.rtol)
            self._job = ("seg", self._gen, a, b, f)
            f.add_done_callback(lambda _f: self._wake.set())
        elif goal > self.t_pre and self.tail is None and a >= self.t_pre - 1e-6 and self._open is None:
            x_pre = self.nom.state(self.t_pre) + self.delta(self.t_pre)
            f = pool.submit(fly_truth_tail, x_pre, self.t_pre, self.t_end, self.t_ca, self.basis, self.rtol)
            self._job = ("tail", self._gen, f, len(self.pushes))
            f.add_done_callback(lambda _f: self._wake.set())

    def _collect(self):
        job = self._job
        if job is None:
            return
        if not job[4 if job[0] == "seg" else 2].done():
            return
        self._job = None
        if job[1] != self._gen:
            return
        try:
            if job[0] == "seg":
                tn, xw, x0 = job[4].result()
                self.segs.append({"a": job[2], "b": job[3], "tn": tn, "xw": xw, "x0": x0})
            else:
                tail = job[2].result()
                tail["n_pushes"] = job[3]
                self.tail = tail
        except Exception as exc:
            self.error = f"Bennu's orbit integration failed: {exc}"

    # ----------------------------------------------------------------------------- queries
    def delta(self, t):
        """x_with - x_without at t <= t_pre (zero before the first push)."""
        if self.t_first is None or t <= self.t_first:
            return np.zeros(6)
        for s in self.segs:
            if t <= s["b"] + 1e-6:
                tn = s["tn"]
                k = int(np.clip(np.searchsorted(tn, t) - 1, 0, len(tn) - 2))
                return _hermite(t, tn[k], tn[k + 1], s["xw"][k] - s["x0"][k], s["xw"][k + 1] - s["x0"][k + 1])
        raise ValueError("Bennu's orbit is not integrated that far yet")

    def delta_samples(self):
        """All samples of x_with - x_without so far (display curves)."""
        segs = self.segs
        if not segs:
            return np.zeros(0), np.zeros((0, 6))
        tn = np.concatenate([s["tn"][:-1] for s in segs] + [segs[-1]["tn"][-1:]])
        dn = np.concatenate([(s["xw"] - s["x0"])[:-1] for s in segs] + [(segs[-1]["xw"] - segs[-1]["x0"])[-1:]])
        return tn, dn

    def state(self, t):
        """Bennu's true barycentric state at t (scalar)."""
        t = float(t)
        if self.t_first is None or t <= self.t_first:
            return self.nom.state(min(t, self.nom.span[1]))
        if t <= self.t_pre + 1e-6:
            return self.nom.state(t) + self.delta(t)
        tail = self.tail
        if tail is None:
            raise ValueError("Bennu's orbit after the 2135 approach is not integrated yet")
        t = min(t, tail["t_stop"])
        ft = tail["fine_t"]
        k = int(np.searchsorted(ft, t))
        if 0 < k < len(ft) and ft[k] - ft[k - 1] <= FINE_DT + 1.0:
            return _hermite(t, ft[k - 1], ft[k], tail["fine_x"][k - 1], tail["fine_x"][k])
        tt = tail["track_t"]
        k = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
        return _hermite(t, tt[k - 1], tt[k], tail["track_x"][k - 1], tail["track_x"][k])

    def prediction(self):
        """The full-model orbit flown on to 2185 for the pushes so far (None while it is being computed)."""
        tail = self.tail
        if tail is None or tail.get("n_pushes") != len(self.pushes) or self._open is not None:
            return None
        return tail

    def key(self):
        """Identifies the push history (a prediction belongs to one)."""
        return (len(self.pushes), self._open is not None)

    def status(self):
        """What the background integration is doing (for the UI)."""
        job = self._job
        if job is None:
            return None
        if job[0] == "seg":
            return f"integrating Bennu's orbit with the full force model ({_year(job[3]):.1f})"
        return "flying Bennu through the 2135 flyby with the full force model"


def _year(t):
    return 2000.0 + (t / DAY + 0.5) / 365.25
