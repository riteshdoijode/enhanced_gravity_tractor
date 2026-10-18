"""Live mission: the keyhole-avoidance campaign flown in the full N-body model, one spacecraft at a time.

The run starts with both spacecraft in the Earth parking orbit and Bennu on the scenario orbit (its 2135
flyby threads the keyhole of a later Earth impact, core/scenario.py). Commands launch the legs:

DART (launch_dart, any number, one at a time): parking orbit -> orbit raising (burns at perigee, geocentric
          two-body, propellant counted per burn, core/dart.py) -> injection at perigee -> hyperbolic departure
          to the hand-over radius -> ballistic heliocentric cruise with trajectory correction maneuvers (TCMs)
          -> terminal intercept guidance aimed at the recommended point of contact -> impact on the real
          shape (Cheng et al. momentum transfer with the *true*, unknown-in-advance beta) or a miss.
          After an impact, ground tracking (orbit determination) gives beta and the deflected orbit
          (the full-model prediction of the 2135 flyby re-anchors the linear one); GROUND_OD_DAYS later the
          result is known: success (the tractor can take the flyby to the goal), short or missed.
Tractor (launch_tractor, after a successful DART): escape spiral -> heliocentric cruise on the optimized
          low-thrust flight plan -> proximity operations (core/proximity/ops.py): approach, survey,
          descent, boulder capture (MuJoCo), ascent, station; tow with the trim law toward the DART aim
          point (core/tow.py).
Bennu (ground truth): integrated with the full force model from its first push on (core/truth.py), with the
impulses and the tractor pull it actually received; the clock runs through the 2135 flyby to the impact date:
"Earth is safe" or the impact, as the integration says. The team's own predictions (planner, trim guidance,
projected flyby) use the linear model anchored to the full-model prediction.
Physics never depends on the display: chunks are set by the physics (phys_chunk), there are no speed caps,
and the clock waits only when Bennu's orbit (or the orbit-determination prediction at a verdict) is still
being integrated (busy).
Navigation: Kalman filter on each craft fusing daily radiometric fixes; guidance flies on the estimate;
thrust has execution errors. Events (storms, throttle, re-plans, burns, impacts) are logged.
"""

from dataclasses import dataclass, replace

import numpy as np

from core.constants import BENNU, DAY, G, G0, GM, LUNAR_DISTANCE, RADIUS, YEAR
from core.dart import kepler_state
from core.dynamics.asteroid import bennu_attitude
from core.dynamics.gravity import NBody
from core.earth_impact import SLOWMO_ALT as _EARTH_ALT
from core.earth_impact import SLOWMO_EARTH as _SLOWMO_EARTH
from core.estimation.navfilter import NavFilter
from core.guidance import kepler
from core.guidance.departure import R_HANDOFF, R_SOI, circular_speed
from core.guidance.lowthrust import LowThrustGuidance
from core.planner import GROUND_OD_DAYS, TractorPlan
from core.timeutil import t_to_utc_str

MU_E, MU_S = GM["earth"], GM["sun"]
HELIO_BODIES = ["earth", "moon", "venus", "mars_bary", "jupiter_bary"]
GEO_BODIES = ["sun", "moon"]
SIGMA_R, SIGMA_V = 1e3, 0.01  # radiometric orbit-determination fix (1 km, 1 cm/s)
THRUST_MAG_ERR, THRUST_DIR_ERR = 0.01, np.radians(0.3)
THRUST_ERR = float(np.hypot(THRUST_MAG_ERR, THRUST_DIR_ERR))  # per-axis thrust-error fraction for the filter
KI_RCS_ACCEL = 0.05  # m/s^2, impactor terminal-guidance authority
APPROACH_OFFSET = 30e3  # m: the cruise aims at a point this far sunward of Bennu (safe hand-over)
R_BOUND = 290.0  # m, Bennu's bounding sphere (shape model)
TCM_FRACTIONS = (0.02, 0.3, 0.7, 0.95)
SIGMA_BETA_GROUND = 0.1  # beta from the deflected orbit measured by ground tracking
# Slow motion near Bennu (a display speed limit only: the physics does not depend on the display speed):
# (time to impact or since it [s], or altitude [m], max warp [s of mission time per s], what is shown)
SLOWMO_DART = ((10.0, 1.0, "DART: the last seconds before the impact"),
               (300.0, 30.0, "DART: final approach to Bennu"))
SLOWMO_RANGE = 1000e3  # m: inside this range of Bennu the close-up begins (both craft)
SLOWMO_RANGE_DART = (5.0, "DART: the last 1,000 km to Bennu")  # ~15 s of wall time at 12 km/s, then the last 10 s
SLOWMO_RANGE_TRACTOR = (600.0, "Tractor: the last 1,000 km to Bennu")  # 10 min/s until the proximity operations
SLOWMO_IMPACT = ((40.0, 2.0, "Impact: the flash and the ejecta curtain"),
                 (1200.0, 60.0, "Impact: the ejecta cone spreading"))
SLOWMO_DESCENT = ((150.0, 90.0, "Tractor: final descent to the boulder"),
                  (np.inf, 300.0, "Tractor: descent toward the boulder"))
SLOWMO_EARTH_ALT = _EARTH_ALT  # Bennu within 1,000 km of Earth's surface on an impact course: 5 s/s (~80 s)
SLOWMO_EARTH = _SLOWMO_EARTH[0]
SLOWMO_CAPTURE = 15.0  # s/s while the arm grabs and stows the boulder (~11 min of mission time)
SLOWMO_ASCENT = (900.0, 60.0)  # the first 15 minutes of the ascent with the boulder
TRIM_HOLD_DAYS = 60.0  # the trim has held (tow off, parked) this long: the boulder is released, the tractor leaves


@dataclass
class Event:
    t: float
    who: str
    text: str
    level: str = "info"  # info | warn | good | bad


def _rk4(f, t, r, v, h):
    k1 = f(t, r, v)
    k2 = f(t + h / 2, r + h / 2 * k1[0], v + h / 2 * k1[1])
    k3 = f(t + h / 2, r + h / 2 * k2[0], v + h / 2 * k2[1])
    k4 = f(t + h, r + h * k3[0], v + h * k3[1])
    return r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])


def _hermite(t, t0, x0, t1, x1):
    """Cubic Hermite interpolation of a state [r, v] between (t0, x0) and (t1, x1): the position matches r and v
    at both ends, the velocity is its derivative. Display only: the craft step on their own physical grids, which
    overshoot the mission clock by up to half a day; the views draw them where they were at the clock's time."""
    h = t1 - t0
    if h <= 0:
        return np.asarray(x1, float).copy()
    s = float(np.clip((t - t0) / h, 0.0, 1.0))
    r0, v0, r1, v1 = x0[:3], x0[3:], x1[:3], x1[3:]
    s2, s3 = s * s, s * s * s
    r = (2 * s3 - 3 * s2 + 1) * r0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * r1 + (s3 - s2) * h * v1
    v = ((6 * s2 - 6 * s) * r0 + (3 * s2 - 4 * s + 1) * h * v0 + (-6 * s2 + 6 * s) * r1 + (3 * s2 - 2 * s) * h * v1) / h
    return np.concatenate([r, v])


def _rot_by(vec, raw_axis, ang):
    """Rotate vec by the angle ang about raw_axis made perpendicular to vec (a fixed pointing error)."""
    axis = raw_axis - raw_axis @ vec / max(vec @ vec, 1e-30) * vec
    n = np.linalg.norm(axis)
    if n < 1e-12 or not np.any(vec):
        return vec
    axis /= n
    return vec * np.cos(ang) + np.cross(axis, vec) * np.sin(ang)


def _rot_small(vec, angle_sigma, rng):
    """Rotate vec by a small random angle (thrust pointing error)."""
    axis = rng.normal(size=3)
    axis -= axis @ vec / max(vec @ vec, 1e-30) * vec
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return vec
    axis /= n
    ang = rng.normal(0.0, angle_sigma)
    return vec * np.cos(ang) + np.cross(axis, vec) * np.sin(ang)


def _thread(fn, name):
    import threading

    threading.Thread(target=fn, name=name, daemon=True).start()


class LiveMission:
    def __init__(self, eph, nominal, planner, params, t_start, seed=0):
        self.eph, self.nom, self.planner, self.p = eph, nominal, planner, params
        self.sc = planner.sc
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)  # mission-level draws (true beta, ground beta, ejecta), in event order
        self.beta_true = float(max(1.0, self.rng.normal(params.beta, params.beta_sigma)))
        self.t = self.t_start = float(t_start)
        self.t_end = min(eph.span("earth")[1], self.sc.t_impact + 2 * YEAR) - 2 * DAY
        self.safe_until = -np.inf
        self.bennu_impulses = []  # (t, dv) applied to Bennu
        self.events, self.eta_history = [], []
        self.grav_helio = NBody(eph, "sun", HELIO_BODIES)
        self.grav_geo = NBody(eph, "earth", GEO_BODIES)
        self.prox_controller = "MPC"
        self.ejecta = None
        self.t_impact = np.nan  # last DART impact
        from core.dynamics.asteroid import reset_spin

        reset_spin()  # Bennu's rotation is nominal until this mission's first impact
        self.spin_change = None  # (t, omega) after the latest impact
        self.beta_measured = None  # by the tractor's navigation filter (if it watched an impact)
        self.beta_ground = None  # from ground tracking of the deflected orbit
        self._prox_env = None
        self._station_logged = False
        from core.tow import Deflection
        from core.truth import BennuTruth

        self.defl = Deflection(nominal, self.t)  # the team's linear model of Bennu's deviation (guidance, outlook)
        self.truth = BennuTruth(eph, nominal, self.sc, planner.bplane.basis, planner.t_ca)  # Bennu: full physics
        self.display_from = -np.inf  # a re-simulation computes the past quickly: no ejecta before this time
        self.b_offset = np.zeros(2)  # nonlinear correction of the linear 2135 prediction (orbit determination)
        self.trim = None  # B-plane trim guidance, created when the tractor first reaches its station
        self.tow_start = self.tow_end = np.nan
        self.tow_complete = False
        self.trim_off_since = np.nan  # when the trim switched the tow off (the hold that ends in the release)
        self.final_check = None  # full-model prediction for the pushes so far (dict)
        self._check_key = None  # the pushes it covers
        self._wait = None  # (why, kind, t) the clock waits for: Bennu's orbit to t, or the 2135 prediction
        self.closed = False
        self._od_pending = []  # (t_known, impactor) DART results waiting for orbit determination
        self.dart_results = []
        self.flyby_logged = self.end_logged = False
        self.curve_version = 0  # bumps when the verified Bennu track changes (the UI redraws the distance plot)
        self.tractor_plan = TractorPlan()
        self.tractor = TractorFlight(self)
        self.impactors = []
        self.log(self.t, "mission", f"Mission clock starts (Bennu's true β = {self.beta_true:.2f}, unknown to the team). "
                                    f"Without a deflection Bennu hits Earth on {t_to_utc_str(self.sc.t_impact)[:10]}.")

    # ------------------------------------------------------------------ environment
    @property
    def fullmodel(self):
        return self.truth.pool

    @fullmodel.setter
    def fullmodel(self, pool):
        self.truth.pool = pool

    @property
    def impactor(self):
        return self.impactors[-1] if self.impactors else None

    @property
    def plan(self):  # backward-compatible view used by the proximity operations
        return self

    def log(self, t, who, text, level="info"):
        self.events.append(Event(t, who, text, level))

    def sun(self, t):
        return self.eph.state("sun", t)

    def bennu_helio(self, t):
        """Bennu's true heliocentric state (core/truth.py: the full force model with every push so far). Times
        not integrated yet (only display queries ask for them) fall back to the team's prediction."""
        t = np.atleast_1d(np.asarray(t, float))
        tr = self.truth
        if tr.t_first is None:
            return self.nom.state(np.minimum(t, self.nom.span[1])) - self.eph.state("sun", t)
        out = np.empty((len(t), 6))
        h = tr.horizon()
        for i, ti in enumerate(t):
            out[i] = tr.state(ti) if ti <= h + 1e-6 else self.bennu_predict(ti)[0] + self.eph.state("sun", ti)
        return out - self.eph.state("sun", t)

    def bennu_predict(self, t):
        """The team's prediction of Bennu's heliocentric state (guidance targets, plans): the scenario orbit plus
        the linear deviation of the pushes so far; after t_pre the full-model prediction when there is one."""
        t = np.atleast_1d(np.asarray(t, float))
        x = self.nom.state(np.minimum(t, self.nom.span[1])) - self.eph.state("sun", t)
        if self.defl.active:
            x = x + self.defl.states(np.minimum(t, self.sc.t_pre))
            fc = self.final_check
            after = t > self.sc.t_pre
            if after.any() and fc is not None and "track_t" in fc:
                tt, xx = fc["track_t"], fc["track_x"]
                for i in np.where(after)[0]:
                    ti = min(t[i], tt[-1])
                    x[i] = np.array([np.interp(ti, tt, xx[:, k]) for k in range(6)]) - self.eph.state("sun", t[i])
        return x

    def bennu_bary(self, t):
        return self.bennu_helio(t)[0, :3] + self.sun(t)[:3]

    def apply_impact(self, ki, t_imp, U, hit):
        """Kinetic impact on the shape model: momentum transfer with ejecta along the surface normal,
        spin change, ejecta cloud (core/impact.py)."""
        from core.dynamics.asteroid import set_spin_change, spin_rate
        from core.impact import EjectaCloud, momentum_transfer, spin_change
        from core.proximity.ops import ProxEnvironment

        R = bennu_attitude(t_imp)
        r_b, n_b = hit["point"], hit["normal"]
        E = R @ n_b if self.p.ejecta_model == "normal" else -U / np.linalg.norm(U)
        dv, p_ej = momentum_transfer(ki.mass, U, self.beta_true, E)
        self.bennu_impulses.append((t_imp, dv))
        self.defl.impulse(t_imp, dv)
        self.truth.impulse(t_imp, dv)  # the ground truth: Bennu flown on with this impulse (full force model)
        spin = spin_change(r_b, R.T @ (BENNU["mass"] * dv))
        w_old = spin_rate(t_imp)
        set_spin_change(t_imp, w_old + spin["d_omega"][2])  # rate change; the small pole tilt is reported only
        self.spin_change = (float(t_imp), float(w_old + spin["d_omega"][2]))
        dP = 2 * np.pi / (w_old + spin["d_omega"][2]) - 2 * np.pi / w_old
        lat = np.degrees(np.arcsin(r_b[2] / np.linalg.norm(r_b)))
        lon = np.degrees(np.arctan2(r_b[1], r_b[0]))
        incidence = np.degrees(np.arccos(np.clip(-(U @ (R @ n_b)) / np.linalg.norm(U), -1, 1)))
        env = ProxEnvironment.shared()
        seed = int(self.rng.integers(1 << 30))
        self.ejecta = EjectaCloud(t_imp, r_b, n_b, np.linalg.norm(p_ej), env.grav, env.ray, seed=seed,
                                  n=800 if t_imp > self.display_from - 2 * DAY else 40)  # long past: summary only
        energy = 0.5 * ki.mass * (U @ U)
        miss_aim = (float(np.linalg.norm(r_b - ki.contact_b)) if ki.contact_b is not None else np.nan)
        ki.impact = {"t": t_imp, "U": U, "dv_bennu": dv, "beta": self.beta_true, "E_hat": E, "point_b": r_b,
                     "normal_b": n_b, "lat": lat, "lon": lon, "incidence_deg": incidence, "p_ejecta": p_ej,
                     "period_change_s": dP, "pole_shift_deg": spin["pole_shift_deg"], "aim_error_m": miss_aim,
                     "specific_energy": energy / BENNU["mass"], "ejecta_mass": self.ejecta.total_mass}
        self.t_impact = t_imp
        db = self.planner.B([t_imp])[0] @ dv
        self.log(t_imp, ki.name, f"IMPACT at {np.linalg.norm(U) / 1e3:.2f} km/s, {lat:+.0f}° lat {lon:+.0f}° lon"
                                 + (f" ({miss_aim:.0f} m from the aim point)" if np.isfinite(miss_aim) else "")
                                 + f": Bennu Δv {np.linalg.norm(dv) * 1e3:.3f} mm/s, 2135 flyby moved "
                                   f"{np.linalg.norm(db) / 1e3:,.0f} km (linear estimate)", "good")
        self.log(t_imp, ki.name, f"Bennu's day {'shortened' if dP < 0 else 'lengthened'} by {abs(dP):,.1f} s; "
                                 f"ejecta ≈ {self.ejecta.total_mass:.1e} kg (model). Ground tracking starts.")
        self._od_pending.append((t_imp + GROUND_OD_DAYS * DAY, ki))
        self.truth.want_prediction()  # orbit determination: the deflected orbit flown on to the 2135 flyby

    def safe_mode(self, t):
        return t < self.safe_until

    # ------------------------------------------------------------------ commands
    def launch_dart(self, plan):
        if self.impactor is not None and self.impactor.phase not in ("impacted", "missed"):
            self.log(self.t, "mission", "A DART is already on its way: one at a time", "warn")
            return False
        if not plan.feasible or plan.raising is None:
            self.log(self.t, "mission", "DART plan not feasible", "bad")
            return False
        if plan.raising.t_start < self.t - 60.0:
            self.log(self.t, "mission", f"DART plan is too late: its orbit raising was due on "
                                        f"{t_to_utc_str(plan.raising.t_start)[:10]}; re-plan from today", "bad")
            return False
        ki = ImpactorFlight(self, plan, len(self.impactors) + 1)
        self.impactors.append(ki)
        rp = plan.raising
        self.log(self.t, ki.name, f"Committed: {plan.mass:,.0f} kg, orbit raising from {t_to_utc_str(rp.t_start)[:10]} "
                                  f"({len(rp.burns) - 1} burns), departure {t_to_utc_str(plan.t_launch)[:10]}, impact "
                                  f"{t_to_utc_str(plan.t_arrive)[:10]} at {plan.impact_speed / 1e3:.1f} km/s", "good")
        return True

    def launch_tractor(self, plan):
        if self.tractor.phase != "waiting" or np.isfinite(self.tractor.t_launch):
            self.log(self.t, "mission", "The gravity tractor is already launched", "warn")
            return False
        if not plan.feasible:
            self.log(self.t, "mission", "Tractor plan not feasible", "bad")
            return False
        self.tractor_plan = plan
        # the tractor's settings as they are at the launch (a change before the launch re-plans the tractor
        # instead of re-flying the mission: the runner hands the parameters over with the launch)
        self.tractor.mass, self.tractor.thrust = self.p.tractor_mass_kg, self.p.tractor_thrust_n
        self.tractor.set_plan(plan, max(plan.t_launch, self.t))
        arrive = plan.t_arrive_flight if np.isfinite(getattr(plan, "t_arrive_flight", np.nan)) else plan.t_arrive
        self.log(self.t, "tractor", f"Launch ordered for {t_to_utc_str(self.tractor.t_launch)[:10]}: "
                                    f"arrival about {t_to_utc_str(arrive)[:10]}"
                                    + (" (optimized flight plan)" if np.isfinite(getattr(plan, "t_arrive_flight", np.nan))
                                       else " (quick estimate)"), "good")
        return True

    def solar_storm(self, days):
        self.safe_until = max(self.safe_until, self.t) + days * DAY
        self.log(self.t, "mission", f"☀ Solar storm: safe mode, thrusters off for {days:.0f} days", "warn")
        if self.tractor.phase == "cruise":
            self.tractor.after_outage = self.safe_until

    def set_prox_controller(self, kind):
        self.prox_controller = kind
        pr = self.tractor.prox
        if pr is not None and pr.controller_kind != kind:
            pr.controller_kind = kind
            self.log(self.t, "tractor", f"Proximity controller → {kind}")

    def set_tractor_thrust(self, thrust_n):
        if abs(thrust_n - self.tractor.thrust) > 1e-9:
            if self.tractor.phase != "waiting":
                self.log(self.t, "tractor", f"Throttle {self.tractor.thrust:g} N → {thrust_n:g} N")
            self.tractor.set_thrust(thrust_n)

    def finish(self):
        """Run on to the end (the real-time runner goes as fast as the physics allows); the full-model prediction
        is asked for now."""
        self.log(self.t, "mission", "Running on to the end: Bennu's orbit through the 2135 flyby to 2185")
        if self.truth.t_first is not None:
            self.truth.request_prediction()
            self._update_prediction()

    # ------------------------------------------------------------------ waiting for the physics
    def settle(self):
        """Process what is due at this instant without moving the clock: a DART verdict whose orbit determination
        falls due now. advance() does this only while the clock moves, so a re-simulation that stops at a verdict
        date (the run had paused there for a decision) would otherwise leave the verdict pending. True when nothing
        due now is left; False while it waits for the full-model prediction (call again)."""
        if self.busy():
            return False
        self._milestones()
        return not self.busy() and not any(t <= self.t + 1e-6 for t, _ in self._od_pending)

    def busy(self):
        """What the clock is waiting for (Bennu's orbit still being integrated), or None."""
        w = self._wait
        if w is None:
            return None
        why, kind, t = w
        done = self.truth.ready(t) if kind == "truth" else self.truth.prediction() is not None
        if done or self.truth.error:
            self._wait = None
            return None
        return why

    def background(self):
        """The full-model integration running in the background (the clock does not wait for it), or None."""
        st = self.truth.status()
        return {"what": st, "needed_by": np.nan} if st else None

    def _push_key(self):
        return self.truth.key()

    def _update_prediction(self):
        """A new full-model prediction for the pushes so far: anchor the linear model to it and log it."""
        pred = self.truth.prediction()
        key = self._push_key()
        if pred is None:
            if self.final_check is not None and self._check_key != key:
                self.final_check = None
            return
        if self._check_key == key and self.final_check is not None:
            return
        dx_lin = self.defl.Phi(self.planner.t_ca, self.defl.t) @ self.defl.state_at(self.defl.t)
        b_lin = self.planner.b0 + self.planner.bplane.project(dx_lin[:3])
        fc = dict(pred)
        diff = float(np.linalg.norm(fc["b_nonlinear"] - b_lin))
        shift_lin = float(np.linalg.norm(b_lin - self.planner.b0))
        fc.update(b_linear=b_lin, difference_km=diff / 1e3, relative=diff / max(shift_lin, 1.0))
        fc["approaches"] = [a for a in fc["approaches"] if a["t"] > self.planner.t_ca + 30 * DAY]
        fc["min_after"] = min([a["dist"] for a in fc["approaches"]], default=np.inf)
        self.b_offset = fc["b_nonlinear"] - b_lin  # the team's linear model, re-anchored
        self.final_check, self._check_key = fc, key
        self.curve_version += 1
        self._log_verification(fc)

    def close(self, keep_truth=False):
        """The run has ended: stop the ejecta computation (and Bennu's integration unless a re-simulation takes
        it over). Returns True if Bennu's integration was still busy."""
        self.closed = True
        if self.ejecta is not None:
            self.ejecta.stop()
        busy = self.truth.status() is not None
        if not keep_truth:
            self.truth.close()
        return busy

    def adopt_truth(self, truth):
        """A re-simulation: take over the previous run's Bennu integration (identical pushes keep their work)."""
        truth.begin_replay()
        truth.speculate = False
        self.truth = truth

    def _log_verification(self, fc):
        q, ld = fc["q"], fc["q_ld"]
        goal = self.p.goal_ld
        lvl = "good" if ld >= goal else "bad"
        txt = (f"Full-model prediction (no further pushes): 2135 flyby at {q / 1e3:,.0f} km = {ld:.2f} LD "
               f"({'beyond' if ld >= goal else 'inside'} the {goal:g} LD goal; linear prediction off by "
               f"{fc['difference_km']:,.0f} km). ")
        if np.isfinite(fc["t_impact"]):
            txt += f"Bennu still hits Earth on {t_to_utc_str(fc['t_impact'])[:10]}."
            lvl = "bad"
        else:
            m = fc.get("min_after", np.inf)
            txt += "No Earth impact through 2184" + (f" (closest later approach {m / LUNAR_DISTANCE:.1f} LD)."
                                                    if np.isfinite(m) else ".")
            close = [a for a in fc.get("approaches", []) if a["dist"] < goal * LUNAR_DISTANCE]
            if close:
                txt += f" Warning: {len(close)} later approach(es) inside {goal:g} LD."
                lvl = "warn" if lvl == "good" else lvl
        self.log(self.t, "mission", txt, lvl)

    # ------------------------------------------------------------------ stepping
    def phys_chunk(self):
        """Longest step between interleavings of the craft, the tow and the events: set by the physics, never
        by the display speed (the proximity phases couple the tractor's pull every 10 minutes)."""
        pr = self.tractor.prox
        return 600.0 if pr is not None and pr.phase not in ("station", "parked", "departed") else DAY

    def slow_motion(self):
        """(max warp, what is shown) while something worth watching happens at Bennu -- the DART's final
        approach and impact, the tractor's descent, boulder capture and lift-off -- else None. A display hint
        for the real-time runner (which the user can switch off); the physics never uses it."""
        ki = self.impactor
        if ki is not None and ki.phase == "terminal":
            t_go, warp, what = SLOWMO_DART[0]
            if ki.t_go <= t_go:
                return warp, what
            rng_, _ = self.craft_range(ki)
            if rng_ <= SLOWMO_RANGE:
                return SLOWMO_RANGE_DART
            for t_go, warp, what in SLOWMO_DART[1:]:
                if ki.t_go <= t_go:
                    return warp, what
        tr = self.tractor
        if tr.phase == "cruise" and tr.x is not None and np.isfinite(tr.t_launch):
            rng_, closing = self.craft_range(tr)
            if rng_ <= SLOWMO_RANGE and closing > 0:
                return SLOWMO_RANGE_TRACTOR
        t_hit = self.earth_impact_time()
        if np.isfinite(t_hit) and t_hit - 3 * DAY < self.t < t_hit and self.earth_altitude()[0] <= SLOWMO_EARTH_ALT:
            return SLOWMO_EARTH  # Bennu's last 1,000 km into Earth (the 2182 impact page, app/scenes/earth_impact.py)
        if np.isfinite(self.t_impact):
            since = self.t - self.t_impact
            for t_s, warp, what in SLOWMO_IMPACT:
                if -60.0 <= since < t_s:  # (the DART's last step may end a moment past the mission clock)
                    return warp, what
        pr = self.tractor.prox
        if pr is not None:
            if pr.phase == "descent":
                alt = pr.hist["alt"][-1] if pr.hist["alt"] else np.inf
                for a, warp, what in SLOWMO_DESCENT:
                    if alt <= a:
                        return warp, what
            if pr.phase == "capture":
                stage = getattr(pr.capture, "stage", "")
                return SLOWMO_CAPTURE, f"Tractor: capturing the boulder ({stage})"
            if pr.phase == "ascent" and pr.t - pr.phase_t0 < SLOWMO_ASCENT[0]:
                return SLOWMO_ASCENT[1], "Tractor: lifting off with the boulder"
        return None

    def advance(self, t_target, max_chunk=None, budget=None, stop=None):
        """Advance all craft to t_target in physics chunks (events interleave at their own times). Returns early
        while Bennu's orbit is still being integrated past the next step (busy), at a DART verdict (a decision
        point), after `budget` seconds of computing (the real-time caller stays responsive), or when stop()
        says so after a chunk (the caller's display changes, e.g. slow motion begins)."""
        import time

        wall = time.perf_counter()
        max_chunk = self.phys_chunk() if max_chunk is None else max_chunk
        while self.t < t_target - 1e-6:
            if self.busy():
                return
            n_results = len(self.dart_results)
            self._milestones()  # what waited for the physics happens now, before the clock moves on
            if len(self.dart_results) > n_results or self.busy():
                return  # a DART verdict just came in (decision point), or it needs the prediction first
            t_next = min(t_target, self.t + min(max_chunk, self.phys_chunk()), self.t_end)
            for tb in (self.safe_until, self.planner.t_ca, self.sc.t_impact, self.sc.t_pre,
                       *(t for t, _ in self._od_pending), *self._craft_events()):
                if np.isfinite(tb) and self.t < tb < t_next:
                    t_next = tb
            if not self.truth.ensure(t_next + DAY):  # (the craft step on their own grids, up to half a day on)
                self._wait = ("integrating Bennu's orbit with the full force model", "truth", t_next + DAY)
                return
            for ki in self.impactors:  # first: an impact stops the whole mission at its instant
                ki.advance_to(t_next)
                if ki.t < t_next - 1e-6:
                    t_next = ki.t
            self.tractor.advance_to(t_next)
            self._tow(self.t, t_next)
            self.t = t_next
            self._milestones()
            if self.on_station() and not self._station_logged:
                self._station_logged = True
                self.log(self.t, "mission", f"Towing until {t_to_utc_str(self.tow_end)[:10]} (trim to the DART aim point)",
                         "good")
            if self.complete() or (budget is not None and time.perf_counter() - wall > budget):
                return
            if stop is not None and stop():
                return

    def craft_range(self, craft):
        """(range to Bennu, closing speed) of a heliocentric craft as displayed at the mission clock."""
        x = craft.state_at(self.t)
        if x is None:
            return np.inf, 0.0
        xb = self.bennu_helio(self.t)[0]
        rel, vrel = x[:3] - xb[:3], x[3:] - xb[3:]
        rng_ = float(np.linalg.norm(rel))
        return rng_, float(-(rel @ vrel) / max(rng_, 1.0))

    def _range_crossing(self, craft):
        """Predicted time the craft comes inside SLOWMO_RANGE (straight-line estimate, landing half a second inside
        so the chunk that ends there is already in the close-up), or nan."""
        rng_, closing = self.craft_range(craft)
        if not np.isfinite(rng_) or rng_ <= SLOWMO_RANGE or closing <= 0:
            return np.nan
        return self.t + (rng_ - SLOWMO_RANGE) / closing + 0.5

    def earth_altitude(self):
        """(Bennu's altitude above Earth's surface, its descent rate) at the mission clock [m, m/s]."""
        x = self.bennu_helio(self.t)[0] + self.sun(self.t) - self.eph.state("earth", self.t)
        r = float(np.linalg.norm(x[:3]))
        return r - RADIUS["earth"], float(-(x[:3] @ x[3:]) / r)

    def _earth_crossing(self):
        """Bennu on its way into Earth (the verified orbit hits within 3 days): the predicted time it comes within
        SLOWMO_EARTH_ALT of the surface (straight-line estimate, half a second inside), or nan."""
        t_hit = self.earth_impact_time()
        if not (np.isfinite(t_hit) and self.t < t_hit < self.t + 3 * DAY):
            return np.nan
        alt, rate = self.earth_altitude()
        if alt <= SLOWMO_EARTH_ALT or rate <= 0:
            return np.nan
        return min(self.t + (alt - SLOWMO_EARTH_ALT) / rate + 0.5, t_hit - 1.0)

    def _craft_events(self):
        """Times at which a craft changes what it is doing (chunk boundaries): a chunk ends exactly where a
        close-up begins, so a skip stops there too."""
        out = [self._earth_crossing()]  # (and where Bennu's last 1,000 km to Earth begin)
        ki = self.impactor
        if ki is not None and ki.phase not in ("impacted", "missed"):
            if ki.burn_k < len(ki.burn_times):
                out.append(ki.burn_times[ki.burn_k] + 1.0)
            if np.isfinite(ki.t_handoff):
                out.append(ki.t_handoff + 1.0)
            out.append(ki.ip.t_arrive - 2 * DAY + 1.0)
            if ki.phase == "terminal" and np.isfinite(ki.t_go):  # where the slow motion begins
                out.append(ki.t + ki.t_go - SLOWMO_DART[-1][0])
                out.append(self._range_crossing(ki))
        tr = self.tractor
        if np.isfinite(tr.t_launch) and tr.phase == "waiting":
            out.append(tr.t_launch + 1.0)
        if self.tractor.prox is None and np.isfinite(tr.t_launch):
            eta = tr.eta()
            if np.isfinite(eta):
                out.append(eta - 1.0)
            if tr.phase == "cruise":
                out.append(self._range_crossing(tr))
        return out

    def _milestones(self):
        t = self.t
        self._update_prediction()
        for item in list(self._od_pending):
            t_known, ki = item
            if t >= t_known - 1e-6:
                if (not self.truth.towing() and self.truth.prediction() is None
                        and self.truth.request_prediction() is None):
                    # the orbit determination needs the full-model prediction of the deflected orbit (during a
                    # tow the team uses its linear model: the full-model prediction follows when the tow ends)
                    self._wait = ("orbit determination: flying the deflected Bennu to the 2135 flyby (full force "
                                  "model)", "prediction", t)
                    return
                self._update_prediction()
                self._od_pending.remove(item)
                self._dart_verdict(ki)
        for ki in self.impactors:
            if ki.phase == "missed" and not ki.verdict_logged:
                ki.verdict_logged = True
                self._dart_verdict(ki)
        if not self.flyby_logged and t >= self.planner.t_ca:
            self.flyby_logged = True
            fc = self.final_check if self.defl.active else None
            q = fc["q"] if fc is not None and "q" in fc else self.sc.q_ca
            self.log(t, "Bennu", f"2135 flyby: Bennu passes {q / 1e3:,.0f} km from Earth ({q / LUNAR_DISTANCE:.2f} LD)"
                                 + (" — through the keyhole" if not self.defl.active else ""),
                     "good" if q >= self.p.goal_ld * LUNAR_DISTANCE else "bad")
        t_hit = self.earth_impact_time()
        if not self.end_logged and np.isfinite(t_hit) and t >= t_hit:
            self.end_logged = True
            from core.scenario import impact_site

            self.log(t_hit, "Bennu", f"IMPACT: Bennu hits Earth on {t_to_utc_str(t_hit)} UTC", "bad")
        elif not self.end_logged and not np.isfinite(t_hit) and t >= self.sc.t_impact + DAY and self.final_check:
            self.end_logged = True
            self.log(t, "Bennu", f"{t_to_utc_str(self.sc.t_impact)[:10]} passes: no impact — Earth is safe", "good")

    def earth_impact_time(self):
        """When Bennu hits Earth on the current (verified) orbit; nan if it does not."""
        if not self.defl.active:
            return self.sc.t_impact
        fc = self.final_check
        if fc is not None and "t_impact" in fc:
            return fc["t_impact"]
        return np.nan

    def _dart_verdict(self, ki):
        """Orbit determination done: the DART's real effect is known."""
        p = self.p
        b_after = self.b_now_abs()
        if ki.phase == "missed":
            res = self.planner.dart_result(p, b_after, self.t, tractor_launched=self.tractor.phase != "waiting")
            res.update(result="missed", number=ki.number, t=self.t, b_after=b_after)
            self.dart_results.append(res)
            self.log(self.t, ki.name, f"DART #{ki.number} MISSED Bennu (closest {ki.closest / 1e3:,.1f} km): "
                                      f"another DART is needed for {res['remaining'] / 1e3:,.0f} km", "bad")
            return
        self.beta_ground = (float(max(1.0, self.beta_true + self.rng.normal(0, SIGMA_BETA_GROUND))), SIGMA_BETA_GROUND)
        res = self.planner.dart_result(replace(p, beta=self.beta_ground[0], beta_sigma=SIGMA_BETA_GROUND), b_after,
                                       self.t, tractor_launched=self.tractor.phase != "waiting")
        res.update(result="success" if res["success"] else "short", number=ki.number, t=self.t, b_after=b_after,
                   beta=self.beta_ground)
        self.dart_results.append(res)
        q_ld = res["q"] / LUNAR_DISTANCE
        pct = 100 * res["fraction"]
        if res["success"]:
            extra = ("the goal is already met" if res["need_goal"] <= 0 else
                     f"{res['need_goal'] / 1e3:,.0f} km still needed for the goal")
            self.log(self.t, ki.name, f"DART #{ki.number} SUCCESS: delivered {pct:.0f} % of the {res['target'] / 1e3:,.0f} km "
                                      f"target (β = {self.beta_ground[0]:.2f} ± {SIGMA_BETA_GROUND:.1f} from tracking); 2135 "
                                      f"flyby now at {q_ld:.2f} LD, {extra}. The gravity tractor may launch to steer it to "
                                      "the aim point.", "good")
        else:
            self.log(self.t, ki.name, f"DART #{ki.number} FELL SHORT: delivered {pct:.0f} % of its "
                                      f"{res['target'] / 1e3:,.0f} km target (needs {100 * (1 - p.dart_tolerance):.0f} %; "
                                      f"β = {self.beta_ground[0]:.2f}). 2135 flyby at {q_ld:.2f} LD: another DART is needed "
                                      f"for the remaining {res['remaining'] / 1e3:,.0f} km (see the recommendations)", "bad")

    def b_now_abs(self):
        """2135 miss vector of the orbit as it is now (pushes so far, no future tow), with the nonlinear offset."""
        dx = self.defl.Phi(self.planner.t_ca, self.t) @ self.defl.state_at(self.t) if self.defl.active else np.zeros(6)
        return self.planner.b0 + self.planner.bplane.project(dx[:3]) + self.b_offset

    def dart_ok(self):
        return any(r["result"] == "success" for r in self.dart_results)

    # ------------------------------------------------------------------ tow (Step 7)
    def _tow(self, t0, t1):
        """Bennu is pulled toward the tractor while it is near; the trim guidance moves the station."""
        from core.tow import TrimGuidance, pull_on_bennu

        pr = self.tractor.prox
        if pr is None or self.tow_complete:  # after the tow the tractor's pull is no longer counted (it leaves)
            self.defl.advance(t1)
            self.truth.want_prediction(self.defl.active)
            return
        rho, mass = pr.x[:3].copy(), pr.mass
        near = np.linalg.norm(rho) < 50e3
        self.defl.advance(t1, (lambda tau: pull_on_bennu(mass, rho)) if near else None)
        if near:
            self.truth.tow(t0, t1, pull_on_bennu(mass, rho))  # the ground truth feels the same pull
        self.truth.want_prediction(False)  # no prediction while the tow goes on (it changes every block)
        if pr.phase not in ("station", "parked") or self.tow_complete:
            return
        if self.trim is None:  # first time on station: the tow begins
            self.trim = TrimGuidance(self.planner, "aim", aim_shift=self._aim_shift())
            self.tow_start = pr.t_station if np.isfinite(pr.t_station) else t1
            self.tow_end = min(self.tow_start + self.p.tow_years * YEAR, self.planner.t_ca - 0.5 * YEAR)
        a_mag = G * pr.mass / pr.d_station**2
        if t1 >= self.tow_end:
            why = (f"the {self.p.tow_years:g}-year tow time is up" if np.isfinite(self.p.tow_years)
                   else "half a year before the 2135 flyby, the latest a tow can run")
            self.end_tow(t1, why)
            return
        if t1 >= self.trim.next:
            was_on = self.trim.tow_on
            on, e = self.trim.update(self.defl, t1, a_mag, self.tow_end)
            if was_on != on or on:  # the station moves only while towing (parked: stays parked, see core/tow.py)
                pr.retarget(pr.e_station if e is None else e, tow_on=on)
            if was_on != on:
                self.trim_off_since = t1 if not on else np.nan
                self.log(t1, "tractor", "Trim reached: parking 5 km out (tow off)" if not on else
                         "Trim error grew: back on station (tow on)", "good" if not on else "warn")
        if (not self.trim.tow_on and np.isfinite(self.trim_off_since)
                and t1 - self.trim_off_since >= TRIM_HOLD_DAYS * DAY):
            self.end_tow(t1, f"the trim has held for {TRIM_HOLD_DAYS:.0f} days")

    def end_tow(self, t=None, why="released on command"):
        """End of the tow: the boulder is released and the tractor leaves Bennu (proximity phases release ->
        depart -> departed, core/proximity/ops.py); Bennu is flown on with the full force model and its prediction
        is requested. Called when the tow time is up, when the trim has held TRIM_HOLD_DAYS, and by the runner's
        'release' command. False if there is no tractor on station / parked to release."""
        t = self.t if t is None else float(t)
        pr = self.tractor.prox
        if pr is None or self.tow_complete or pr.phase not in ("station", "parked", "transfer"):
            self.log(t, "mission", "Nothing to release: the tractor is not holding at Bennu", "warn")
            return False
        self.tow_complete = True
        if not np.isfinite(self.tow_end) or t < self.tow_end:
            self.tow_end = t
        self.truth.tow_end(t)
        self.truth.want_prediction()
        pr.release()
        t_log = max(t, pr.t_release) if np.isfinite(pr.t_release) else t  # (the proximity clock may run ahead)
        if self.trim is not None:
            b = self.trim.b_now(self.defl, t)
            years = (t - self.tow_start) / YEAR if np.isfinite(self.tow_start) else 0.0
            flyby = np.linalg.norm(self.planner.b0 + b + self.b_offset) / LUNAR_DISTANCE
            self.log(t_log, "mission", f"Tow complete ({why}): boulder released, the tractor leaves Bennu. "
                                   f"{self.defl.tow_dv * 1e3:,.4f} mm/s delivered over {years:.1f} yr; 2135 flyby now "
                                   f"{flyby:.3f} LD (linear model); Bennu is now flown on with the full force model", "good")
        else:
            self.log(t_log, "mission", f"Tow ended before it began ({why}): boulder released, the tractor leaves Bennu",
                     "warn")
        return True

    def _aim_shift(self):
        """Aim of the trim tow (a shift from b0 in the linear model): the DART aim distance, on the line the
        tow can move the flyby along from where the impacts left it."""
        b_now = self.b_now_abs()
        aim = self.planner.aim_on_line(self.p, b_now)
        if np.linalg.norm(aim) < np.linalg.norm(b_now):  # overshoot: hold it (no pull-back, see the planner)
            aim = b_now
        return aim - self.planner.b0 - self.b_offset

    def tow_direction_now(self, t):
        """Where the tow should start: the trim law's direction for the deflection achieved so far."""
        from core.tow import TrimGuidance

        tg = TrimGuidance(self.planner, "aim", aim_shift=self._aim_shift())
        G_ = self.tractor.prox
        on, e = tg.update(self.defl, t, G * G_.mass / G_.d_station**2, self.planner.t_ca - 0.5 * YEAR)
        return e

    def coarse_ok(self):
        """Nothing needs day-by-day interleaving (no tractor at Bennu, or its tow is over): long chunks are fine."""
        pr = self.tractor.prox
        return pr is None or self.tow_complete

    def complete(self):
        t_hit = self.earth_impact_time()
        return self.t >= self.t_end - 1.0 or (np.isfinite(t_hit) and self.t >= t_hit and self.end_logged)

    def cruise_complete(self):
        tr_done = self.tractor.phase == "proximity" or not np.isfinite(self.tractor.t_launch)
        ki_done = self.impactor is None or self.impactor.phase in ("impacted", "missed")
        return tr_done and ki_done

    def on_station(self):
        tr = self.tractor
        return tr.prox is not None and tr.prox.phase == "station"

    def next_event_time(self):
        """Next time worth stopping at (skip-to-next-event)."""
        cands = [self.t_end, self.sc.t_pre - 10 * DAY, self.planner.t_ca - DAY, self.sc.t_impact - DAY]
        cands += [t for t, _ in self._od_pending]
        ki = self.impactor
        if ki is not None and ki.phase not in ("impacted", "missed"):
            cands += [b.t - 3600.0 for b in ki.ip.raising.burns] + [ki.ip.t_arrive - 2.5 * DAY]
        tr = self.tractor
        if np.isfinite(tr.t_launch):
            cands += [tr.t_launch - DAY, tr.eta() - DAY]
        if np.isfinite(self.tow_end):
            cands.append(self.tow_end)
        fut = [c for c in cands if np.isfinite(c) and c > self.t + 60.0]
        return min(fut) if fut else self.t_end

    def outlook(self):
        """Projected 2135 flyby using what has actually happened so far."""
        tp = self.tractor_plan
        pl, p = self.planner, self.p
        tow_start, gt_accel = np.nan, 0.0
        tr = self.tractor
        if np.isfinite(tr.t_launch):
            eta = tr.eta()
            tow_start = eta + p.capture_days * DAY if np.isfinite(eta) else np.nan
            gt_accel = tp.gt_accel
            pr = tr.prox
            if pr is not None:
                from core.deflection import gt_accel as _gt

                tow_start = pr.t_station if np.isfinite(pr.t_station) else max(self.t, eta) + 3 * DAY
                if pr.phase in ("ascent", "transfer", "station") or pr.m_boulder == 0.0:
                    gt_accel = _gt(pr.mass, pr.d_station)
        pending = [(k.ip.t_arrive, k.ip.dv_bennu) for k in self.impactors if k.phase not in ("impacted", "missed")]
        q = replace(p, beta=self.beta_ground[0], beta_sigma=SIGMA_BETA_GROUND) if self.beta_ground else p
        out = pl.outcome(q, list(self.bennu_impulses), tow_start, gt_accel, pending=pending, b_offset=self.b_offset)
        if self.trim is not None:
            pr = tr.prox
            a_mag = G * pr.mass / pr.d_station**2
            b_now = self.trim.b_now(self.defl, self.t)
            pred = b_now if self.tow_complete else self.trim.predict(self.defl, self.t, a_mag, self.tow_end)
            out["b_final"] = pl.b0 + pred + self.b_offset
            out["db_gt"] = out["b_final"] - out["b_ki"]
            out["q_final"] = float(np.linalg.norm(out["b_final"]))
            out["b_now"] = pl.b0 + b_now + self.b_offset
            out["trim_mode"] = self.trim.mode
            out["tow_dv"] = self.defl.tow_dv
            out["tow_end"] = self.tow_end
            out["trim_history"] = [(t, pl.b0 + b + self.b_offset, pl.b0 + qq + self.b_offset)
                                   for t, b, qq in self.trim.history[-400:]]
        else:
            out["b_now"] = self.b_now_abs()
        if self.final_check is not None:
            out["final_check"] = {k: v for k, v in self.final_check.items() if k not in ("track_t", "track_x")}
            if "q" in self.final_check and self._push_key() == self._check_key:
                out["q_final"] = self.final_check["q"]
                out["b_final"] = self.final_check["b_nonlinear"]
        out["success"] = bool(out["q_final"] >= out["goal"])
        out["dart_results"] = list(self.dart_results)
        out["t_earth_impact"] = self.earth_impact_time()
        return out


class TractorFlight:
    def __init__(self, m):
        self.m = m
        self.rng = np.random.default_rng([m.seed, 2000])  # this craft's own noise stream
        self.thrust_rot = (np.zeros(3), 0.0)  # pointing error of the current thrust command (axis, angle)
        self.tp = m.tractor_plan
        self.t_launch = np.inf
        self.mass = m.p.tractor_mass_kg
        self.thrust = m.p.tractor_thrust_n
        self.phase = "waiting"
        self.name = "tractor"
        self.t = m.t
        self.r_park = RADIUS["earth"] + m.p.parking_alt_km * 1e3
        self.a_sp, self.theta = self.r_park, 0.0
        incl = np.radians(28.5)
        self.ex, self.ey = np.array([1.0, 0, 0]), np.array([0, np.cos(incl), np.sin(incl)])
        self.x = None  # geocentric (escape) or heliocentric (cruise) state
        self._prev = None  # (t, x) before the last integration step, same frame as x (display interpolation)
        self.nav = None
        self.guid = None
        self.prop_used = self.dv_used = 0.0
        self.after_outage = None
        self.replan_pending = False
        self.next_update = self.next_meas = -np.inf
        self.fail_streak = 0
        self.thrust_err = (1.0, None)
        self.trail = []
        self.arrival_offset = None
        self.t_arrived = np.nan
        self.prox = None
        self.last_measurement = None

    def set_plan(self, tp, t_launch):
        self.tp = tp
        self.t_launch = float(t_launch)

    @property
    def a_max(self):
        return self.thrust / self.mass

    def set_thrust(self, thrust_n):
        self.thrust = thrust_n
        if self.guid is not None:
            self.guid.a_max = self.a_max
            self.replan_pending = True

    def eta(self):
        if self.phase == "proximity":
            return self.t_arrived
        if self.guid is not None and np.isfinite(self.guid.t_f):
            return self.guid.t_f
        t_flight = getattr(self.tp, "t_arrive_flight", np.nan)  # the optimized flight plan (until the cruise starts)
        return t_flight if np.isfinite(t_flight) else self.tp.t_arrive

    def state_at(self, t=None):
        """The craft's state (escape: geocentric; cruise: heliocentric) at mission time t, interpolated between
        its last two integration steps (the steps may end past the mission clock). None: the latest state."""
        if t is None or self.x is None or self._prev is None or t >= self.t:
            return self.x
        t0, x0 = self._prev
        if t <= t0:
            return x0
        return _hermite(t, t0, x0, self.t, self.x)

    def geo_position(self, t=None):
        t = self.t if t is None else min(t, self.t)
        if self.phase == "waiting":  # circling in the parking orbit (display)
            th = circular_speed(self.r_park) / self.r_park * t
            return self.r_park * (np.cos(th) * self.ex + np.sin(th) * self.ey)
        if self.phase == "spiral":  # (the spiral steps end on the chunk: no overshoot to interpolate)
            return self.a_sp * (np.cos(self.theta) * self.ex + np.sin(self.theta) * self.ey)
        if self.phase == "escape":
            return self.state_at(t)[:3].copy()
        return None

    def position_bary(self, t=None):
        """Barycentric position for display, at mission time t (default: the craft's own time)."""
        e = self.m.eph
        t = self.t if t is None else min(t, self.t)
        g = self.geo_position(t)
        if g is not None:
            return e.pos("earth", t) + g
        if self.phase == "proximity":
            return self.m.bennu_bary(t) + self.prox.x[:3]
        return self.state_at(t)[:3] + self.m.sun(t)[:3]

    def _thrust_on(self, t):
        return not self.m.safe_mode(t)

    def _consume(self, accel_mag, h):
        self.dv_used += accel_mag * h
        self.prop_used += self.thrust * (accel_mag / max(self.a_max, 1e-30)) * h / (self.m.p.tractor_isp_s * G0)

    def advance_to(self, t1):
        while self.t < t1 - 1e-6:
            if self.phase == "waiting":
                if t1 < self.t_launch:
                    self.t = t1
                    return
                self.t = self.t_launch
                self.phase = "spiral" if np.sqrt(0.01 * MU_E / self.a_max) > self.r_park else "escape"
                if self.phase == "escape":
                    self._start_cartesian(self.r_park)
                self.m.log(self.t, "tractor", f"Launch: leaving the {self.m.p.parking_alt_km:.0f} km parking orbit")
            elif self.phase == "spiral":
                self._spiral(t1)
            elif self.phase == "escape":
                self._escape(t1)
            elif self.phase == "cruise":
                self._cruise(t1)
            else:  # proximity operations (Step 5)
                self.prox.advance(t1)
                self.t = self.prox.t
            if self.phase != "waiting" and (len(self.trail) == 0 or self.t - self.trail[-1][0] > 2 * DAY):
                self.trail.append((self.t, self.position_bary()))

    # ---------------------------------------------------------------- escape
    def _spiral(self, t1):
        """Orbit-averaged tangential-thrust spiral: sqrt(mu/a) decreases at rate T/m."""
        a_T = self.a_max
        r_sw = min(np.sqrt(0.01 * MU_E / a_T), 0.15 * R_SOI)
        h = t1 - self.t
        if not self._thrust_on(self.t):
            h = min(h, self.m.safe_until - self.t)
            self.theta += np.sqrt(MU_E / self.a_sp**3) * h
            self.t += h
            return
        vc0 = np.sqrt(MU_E / self.a_sp)
        t_sw = max(0.0, (vc0 - np.sqrt(MU_E / r_sw)) / a_T)
        if np.isfinite(self.m.safe_until) and self.m.safe_until > self.t:
            h = min(h, self.m.safe_until - self.t)
        h = min(h, t_sw) if t_sw > 0 else 0.0
        vc1 = vc0 - a_T * h
        self.theta += (vc0**4 - vc1**4) / (4 * MU_E * a_T)
        self.a_sp = MU_E / vc1**2
        self._consume(a_T, h)
        self.t += h
        if self.a_sp >= r_sw * (1 - 1e-9) or t_sw <= 0:
            self._start_cartesian(self.a_sp)
            self.phase = "escape"
            self.m.log(self.t, "tractor", f"Spiral reached {self.a_sp / 1e3:,.0f} km: final escape revolutions")

    def _start_cartesian(self, radius):
        c, s = np.cos(self.theta), np.sin(self.theta)
        r = radius * (c * self.ex + s * self.ey)
        v = circular_speed(radius) * (-s * self.ex + c * self.ey)
        self.x = np.concatenate([r, v])
        self._prev = None

    def _escape(self, t1):
        m = self.m
        while self.t < t1 - 1e-6:
            r, v = self.x[:3], self.x[3:]
            rn = np.linalg.norm(r)
            on = self._thrust_on(self.t)
            energy = 0.5 * v @ v - MU_E / rn
            thrusting = on and energy < 0
            h = 0.01 * np.sqrt(rn**3 / MU_E)  # natural steps (they may end past t1: never cut by the display)
            if thrusting:
                h = min(h, 0.02 * np.linalg.norm(v) / self.a_max)
            if np.isfinite(m.safe_until) and self.t < m.safe_until:
                h = min(h, m.safe_until - self.t)
            a_th = self.a_max * v / np.linalg.norm(v) if thrusting else np.zeros(3)
            r, v = _rk4(lambda t, rr, vv: (vv, m.grav_geo.accel(t, rr) + a_th), self.t, r, v, h)
            if thrusting:
                self._consume(self.a_max, h)
            self._prev = (self.t, self.x)
            self.x = np.concatenate([r, v])
            self.t += h
            if energy >= 0 and np.linalg.norm(r) >= R_HANDOFF:
                self._handover()
                return

    def _handover(self):
        m = self.m
        xh = self.x + m.eph.state("earth", self.t) - m.sun(self.t)
        self.x = xh
        self._prev = None  # a new frame: nothing to interpolate with yet
        self.phase = "cruise"
        self.nav = NavFilter(xh + np.concatenate([self.rng.normal(0, SIGMA_R, 3), self.rng.normal(0, SIGMA_V, 3)]),
                             SIGMA_R, SIGMA_V)
        self.guid = LowThrustGuidance(self._approach_point, self.a_max)
        m.log(self.t, "tractor", "Escaped Earth: heliocentric cruise begins")
        fp = getattr(self.tp, "flight_plan", None)
        prof = (fp or {}).get("profile") if fp and fp.get("feasible") else None
        if prof is not None and prof.get("dv") is not None and prof["t_f"] - self.t > 30 * DAY \
                and abs(fp["params"].tractor_thrust_n / fp["params"].tractor_mass_kg - self.a_max) < 1e-9:
            # warm start from the ground flight plan the user saw at the launch: refined here from the actual
            # hand-over state toward the same arrival (a re-plan from scratch only if that fails)
            self.guid.adopt_profile(prof["t0"], prof["t_f"], prof["dt"], prof["dv"])
            if self.guid.update(self.t, self.nav.x[:6]):
                self.replan_pending, self.fail_streak, self.next_update = False, 0, self.t
                late = (self.t - fp["t_handover"]) / DAY
                m.log(self.t, "tractor", f"Flight plan: the ground profile, arrival {t_to_utc_str(self.guid.t_f)[:10]}"
                                         + (f" (hand-over {late:+.0f} d from the plan)" if abs(late) >= 1 else ""), "good")
                m.eta_history.append((self.t, self.guid.t_f))
                return
            m.log(self.t, "tractor", "The ground flight plan no longer fits from here: re-planning", "warn")
        self._replan("Flight plan")

    # ---------------------------------------------------------------- cruise
    def _approach_point(self, tt):
        """Cruise target: Bennu's state shifted APPROACH_OFFSET toward the Sun (a sunlit, safe hand-over)."""
        x = self.m.bennu_predict(tt).copy()
        u = -x[:, :3] / np.linalg.norm(x[:, :3], axis=1, keepdims=True)
        x[:, :3] += APPROACH_OFFSET * u
        return x

    def _replan(self, why, keep_schedule=False):
        """Ground re-plan from the current estimate. keep_schedule: an anomaly (storm) may only delay the
        arrival (operations keep the existing date if still feasible); a throttle-up may bring it earlier."""
        m = self.m
        old = self.guid.t_f
        deadline = m.planner.t_ca - YEAR
        t_lo = max(2 * DAY, old - self.t) if (keep_schedule and np.isfinite(old)) else 2 * DAY
        tf = self.guid.plan(self.t, self.nav.x[:6], t_lo=t_lo, t_hi=min(8 * YEAR, max(t_lo + 30 * DAY, deadline - self.t)))
        self.replan_pending = False
        self.fail_streak = 0
        self.next_update = self.t
        if np.isfinite(tf):
            delta = "" if not np.isfinite(old) else f" ({(tf - old) / DAY:+.0f} d)"
            m.log(self.t, "tractor", f"{why}: arrival {t_to_utc_str(tf)[:10]}{delta}",
                  "good" if (not np.isfinite(old) or tf <= old) else "warn")
        else:
            m.log(self.t, "tractor", f"{why}: no feasible arrival before the deadline", "bad")
        m.eta_history.append((self.t, tf))

    def _cruise(self, t1):
        m = self.m
        g = self.guid
        while self.t < t1 - 1e-6:
            if self.replan_pending or (self.after_outage is not None and self.t >= self.after_outage):
                storm = self.after_outage is not None and self.t >= self.after_outage
                if storm:
                    self.after_outage = None
                self._replan("Re-plan after the storm" if storm else "Re-plan for new thrust", keep_schedule=storm)
            if not np.isfinite(g.t_f):
                self.t = t1
                return
            # arrival check (true relative state)
            xb = m.bennu_helio(self.t)[0]
            d = np.linalg.norm(self.x[:3] - xb[:3])
            if self.t >= g.t_f - 600 or (d < 100e3 and np.linalg.norm(self.x[3:] - xb[3:]) < 2.0):
                from core.proximity.ops import ProximityOps

                self.phase = "proximity"
                self.t_arrived = self.t
                self.arrival_offset = self.x[:3] - xb[:3]
                m.log(self.t, "tractor", f"Arrived at Bennu: {d / 1e3:.1f} km, "
                                         f"{np.linalg.norm(self.x[3:] - xb[3:]):.2f} m/s relative", "good")
                m.eta_history.append((self.t, self.t))
                self.prox = ProximityOps(m, self, self.t, self.x[:3] - xb[:3], self.x[3:] - xb[3:])
                return
            on = self._thrust_on(self.t)
            if on and self.t >= self.next_update:  # receding-horizon re-solve on the *estimate*
                if g.update(self.t, self.nav.x[:6]):
                    self.fail_streak = 0
                else:
                    self.fail_streak += 1
                    if self.fail_streak >= 3:
                        self._replan("Ground re-plan")
                self.next_update = self.t + min(2 * DAY, max(600.0, 0.05 * (g.t_f - self.t)))
                self.thrust_err = (1.0 + self.rng.normal(0, THRUST_MAG_ERR), self.rng)
                self.thrust_rot = (self.rng.normal(size=3), self.rng.normal(0.0, THRUST_DIR_ERR))
            a_cmd = g.accel(self.t) if on else np.zeros(3)
            h = min(DAY / 2, max(60.0, g.dt / 4), max(60.0, g.t_f - self.t))  # natural steps (may pass t1)
            for tb in (self.next_update, self.next_meas, m.safe_until):
                if np.isfinite(tb) and tb > self.t:
                    h = min(h, tb - self.t)
            h = max(h, 1.0)
            a_true = _rot_by(a_cmd * self.thrust_err[0], *self.thrust_rot) if on else a_cmd
            r, v = _rk4(lambda t, rr, vv: (vv, m.grav_helio.accel(t, rr) + a_true), self.t, self.x[:3], self.x[3:], h)
            b = self.nav.bias.copy()
            re, ve = _rk4(lambda t, rr, vv: (vv, m.grav_helio.accel(t, rr) + a_cmd + b), self.t, self.nav.x[:3],
                          self.nav.x[3:6], h)
            self.nav.x[:6] = np.concatenate([re, ve])
            self.nav.propagate_bias(h)
            self.nav.predict_cov(h, re, MU_S, THRUST_ERR * max(np.linalg.norm(a_cmd), 1e-9))
            self._prev = (self.t, self.x)
            self.x = np.concatenate([r, v])
            self._consume(np.linalg.norm(a_cmd), h)
            self.t += h
            if self.t >= self.next_meas:
                y = self.x + np.concatenate([self.rng.normal(0, SIGMA_R, 3), self.rng.normal(0, SIGMA_V, 3)])
                self.nav.update(y, SIGMA_R, SIGMA_V)
                self.last_measurement = (self.t, y)
                self.next_meas = self.t + DAY
            if self.t - (m.eta_history[-1][0] if m.eta_history else -np.inf) > 30 * DAY:
                m.eta_history.append((self.t, g.t_f))


class ImpactorFlight:
    """One DART: parking orbit -> orbit raising -> injection -> departure -> cruise -> terminal -> impact/miss."""

    def __init__(self, m, ip, number):
        self.m, self.ip, self.number = m, ip, number
        self.rng = np.random.default_rng([m.seed, 1000 + number])  # this craft's own noise stream
        self.name = f"DART #{number}"
        self.mass = ip.mass
        self.rp = ip.raising
        self.phase = "parking"
        self.t = m.t
        self.x = None  # heliocentric state (cruise, terminal)
        self._prev = None  # (t, x) before the last integration step (display interpolation)
        self.nav = None
        self.dv_used = self.prop_used = 0.0
        self.burn_k = 0
        self.v_p = self.rp.v_circ  # perigee speed of the current geocentric orbit
        self.t_peri = self.rp.burns[0].t  # a perigee passage of the current orbit
        self.burn_times = [b.t for b in self.rp.burns]
        self.t_handoff = np.nan
        self.tcm_times = [ip.t_launch + f * (ip.t_arrive - ip.t_launch) for f in TCM_FRACTIONS]
        self.trail = []
        self.impact = None
        self.closest = np.inf
        self.next_meas = -np.inf
        self.last_measurement = None
        self.verdict_logged = False
        self.t_go = np.inf  # time to impact (terminal phase: range / closing speed)
        c = ip.contact or {}
        self.contact_b = np.asarray(c["point_b"], float) if "point_b" in c else None

    # ---------------------------------------------------------------- geometry
    def geo_state(self, t=None):
        t = self.t if t is None else t
        return kepler_state(self.rp.r_park, self.v_p, self.rp.P_hat, self.rp.Q_hat, t - self.t_peri)

    def orbit(self):
        """Current geocentric conic (display): perigee radius and speed, plane."""
        return {"r_p": self.rp.r_park, "v_p": self.v_p, "P": self.rp.P_hat, "Q": self.rp.Q_hat}

    def state_at(self, t=None):
        """Heliocentric state (cruise, terminal) at mission time t, interpolated between the last two integration
        steps (the 12-hour cruise steps end past the mission clock). None: the latest state."""
        if t is None or self.x is None or self._prev is None or t >= self.t:
            return self.x
        t0, x0 = self._prev
        if t <= t0:
            return x0
        return _hermite(t, t0, x0, self.t, self.x)

    def position_bary(self, t=None):
        """Barycentric position for display, at mission time t (default: the craft's own time)."""
        t = self.t if t is None else min(t, self.t)
        if self.phase in ("parking", "raising", "departing"):
            return self.m.eph.pos("earth", t) + self.geo_state(t)[0]
        if self.phase == "impacted":
            return self.m.bennu_bary(t)
        return self.state_at(t)[:3] + self.m.sun(t)[:3]

    def burns_done(self):
        return self.burn_k

    # ---------------------------------------------------------------- stepping
    def advance_to(self, t1):
        m = self.m
        while self.t < t1 - 1e-6:
            if self.phase in ("parking", "raising"):
                t_burn = self.burn_times[self.burn_k]
                if t1 < t_burn:
                    self.t = t1
                    return
                self.t = t_burn
                if m.safe_mode(self.t):  # thrusters off: the burn slips one revolution of the current orbit
                    P = 2 * np.pi * np.sqrt((MU_E / (2 * MU_E / self.rp.r_park - self.v_p**2)) ** 3 / MU_E)
                    slip = P * np.ceil((m.safe_until - self.t) / P)
                    self.burn_times = [tb + slip if k >= self.burn_k else tb for k, tb in enumerate(self.burn_times)]
                    m.log(self.t, self.name, f"Safe mode: burn {self.burn_k + 1} slips {slip / DAY:.1f} days", "warn")
                    continue
                self._burn()
            elif self.phase == "departing":
                if t1 < self.t_handoff:
                    self.t = t1
                    return
                self.t = self.t_handoff
                self._start_cruise()
            elif self.phase in ("cruise", "terminal"):
                if self._fly(t1) == "impact":  # stop at the impact instant (the mission clock stops there too)
                    self.trail.append((self.t, self.position_bary()))
                    return
            else:
                self.t = t1
            if self.phase in ("raising", "departing", "cruise", "terminal") and (
                    not self.trail or self.t - self.trail[-1][0] > (0.05 * DAY if self.phase != "cruise" else DAY)):
                self.trail.append((self.t, self.position_bary()))

    def _burn(self):
        m, b = self.m, self.rp.burns[self.burn_k]
        dv_err = 1.0 + self.rng.normal(0, THRUST_MAG_ERR)
        self.v_p = self.v_p + b.dv * dv_err if b.kind == "raise" else b.v_perigee  # injection: closed loop
        self.t_peri = self.t
        self.dv_used += b.dv
        self.prop_used += b.propellant_kg
        self.burn_k += 1
        n_raise = len(self.rp.burns) - 1
        if b.kind == "raise":
            self.phase = "raising"
            from core.dart import apogee_radius

            m.log(self.t, self.name, f"Orbit-raising burn {self.burn_k}/{n_raise}: +{b.dv:,.0f} m/s, perigee speed "
                                     f"{self.v_p / 1e3:.2f} km/s, apogee {apogee_radius(self.rp.r_park, self.v_p) / 1e3:,.0f} km, "
                                     f"fuel so far {self.prop_used:,.0f} kg")
        else:
            self.phase = "departing"
            e = self.rp.r_park * self.v_p**2 / MU_E - 1.0
            a = self.rp.r_park / (1.0 - e)
            H = np.arccosh((R_HANDOFF / -a + 1.0) / e)
            self.t_handoff = self.t + (e * np.sinh(H) - H) / np.sqrt(MU_E / (-a) ** 3)
            m.log(self.t, self.name, f"Injection burn at perigee: +{b.dv / 1e3:.2f} km/s to {self.v_p / 1e3:.2f} km/s "
                                     f"(v∞ {self.rp.v_inf / 1e3:.2f} km/s); total fuel {self.prop_used:,.0f} kg", "good")

    def _start_cruise(self):
        """Hand-over to the heliocentric model at the edge of Earth's neighbourhood."""
        m = self.m
        r, v = self.geo_state()
        e = m.eph.state("earth", self.t) - m.sun(self.t)
        self.x = np.concatenate([e[:3] + r, e[3:] + v])
        self._prev = None
        self.nav = NavFilter(self.x + np.concatenate([self.rng.normal(0, SIGMA_R, 3), self.rng.normal(0, SIGMA_V, 3)]),
                             SIGMA_R, SIGMA_V)
        self.phase = "cruise"
        self.tcm_times = [t for t in self.tcm_times if t > self.t]
        m.log(self.t, self.name, f"Left Earth's neighbourhood ({R_HANDOFF / 1e3:,.0f} km): cruise to Bennu")

    def _tcm(self):
        m, ip = self.m, self.ip
        xb = m.bennu_predict(ip.t_arrive)[0]  # the team's ephemeris of Bennu at the arrival
        v1, _ = kepler.lambert(self.nav.x[:3], xb[:3], ip.t_arrive - self.t, prograde_axis=kepler.ecliptic_pole_icrf())
        if not np.all(np.isfinite(v1)):
            return
        dv = v1 - self.nav.x[3:6]
        dv_true = _rot_small(dv * (1 + self.rng.normal(0, THRUST_MAG_ERR)), THRUST_DIR_ERR, self.rng)
        self.x[3:] += dv_true
        self.nav.x[3:6] += dv
        self.nav.P[3:6, 3:6] += (THRUST_ERR * np.linalg.norm(dv)) ** 2 * np.eye(3)  # burn execution uncertainty
        self.dv_used += np.linalg.norm(dv)
        m.log(self.t, self.name, f"Trajectory correction: {np.linalg.norm(dv):.2f} m/s")

    def _aim_offset(self, t_hit):
        """Bennu-relative position of the point of contact at the predicted impact time (rotates with Bennu)."""
        if self.contact_b is None:
            return np.zeros(3)
        return bennu_attitude(t_hit) @ self.contact_b

    def _fly(self, t1):
        m = self.m
        while self.t < t1 - 1e-6:
            if self.tcm_times and self.t >= self.tcm_times[0] and not m.safe_mode(self.t):
                self.tcm_times.pop(0)
                self._tcm()
            xb = m.bennu_helio(self.t)[0]
            rel, vrel = self.x[:3] - xb[:3], self.x[3:] - xb[3:]
            rng_, vr = np.linalg.norm(rel), np.linalg.norm(vrel)
            self._check_terminal(rng_, vr)
            a = np.zeros(3)
            h = DAY / 2  # natural steps (they may end past t1: the step sequence never depends on the display)
            if self.tcm_times and self.tcm_times[0] > self.t:  # stop exactly at the next correction burn
                h = min(h, self.tcm_times[0] - self.t)
            elif self.tcm_times and m.safe_mode(self.t):  # burn deferred by safe mode: wait for its end
                h = min(h, max(1.0, m.safe_until - self.t))
            if self.phase == "terminal":
                # optical relative navigation (noise grows with range) + zero-effort-miss intercept law aimed at
                # the point of contact (rotated to the predicted impact time)
                sig = max(10.0, 1e-4 * rng_)  # optical navigation noise: a fixed random signal of mission time
                rel_est = rel + sig * self._nav_noise(self.t)
                tgo = max(rng_ / max(vr, 1.0), 1.0)
                zem = -(rel_est + vrel * tgo) + self._aim_offset(self.t + tgo)
                a = 3.0 * zem / tgo**2
                n = np.linalg.norm(a)
                if n > KI_RCS_ACCEL:
                    a *= KI_RCS_ACCEL / n
                h = min(h, max(0.05, 0.02 * rng_ / max(vr, 1.0)))
            if self.next_meas > self.t:
                h = min(h, self.next_meas - self.t)
            h = max(h, 0.05)
            r0 = self.x[:3].copy()
            r, v = _rk4(lambda t, rr, vv: (vv, m.grav_helio.accel(t, rr) + a), self.t, self.x[:3], self.x[3:], h)
            re, ve = _rk4(lambda t, rr, vv: (vv, m.grav_helio.accel(t, rr) + a), self.t, self.nav.x[:3],
                          self.nav.x[3:6], h)
            self.nav.x[:6] = np.concatenate([re, ve])
            self.nav.propagate_bias(h)
            self.nav.predict_cov(h, re, MU_S, THRUST_ERR * max(np.linalg.norm(a), 1e-9))
            self._prev = (self.t, self.x)
            self.x = np.concatenate([r, v])
            self.dv_used += np.linalg.norm(a) * h
            # impact detection: closest approach of the straight segment relative to Bennu
            xb1 = m.bennu_helio(self.t + h)[0]
            p0, p1 = r0 - xb[:3], r - xb1[:3]
            seg = p1 - p0
            s = np.clip(-(p0 @ seg) / max(seg @ seg, 1e-30), 0.0, 1.0)
            d_min = np.linalg.norm(p0 + s * seg)
            self.closest = min(self.closest, d_min)
            hit = None
            if d_min <= R_BOUND + 5.0:  # inside Bennu's bounding sphere: ray-cast the real shape (Step 6)
                from core.impact import locate_impact
                from core.proximity.ops import ProxEnvironment

                Rb = bennu_attitude(self.t)
                hit = locate_impact(ProxEnvironment.shared().ray, Rb.T @ p0, Rb.T @ p1)
            if hit is not None:
                t_imp = self.t + hit["fraction"] * h
                m.apply_impact(self, t_imp, self.x[3:] - xb1[3:], hit)
                self.phase = "impacted"
                self.t, self.t_go = t_imp, 0.0
                return "impact"
            if self.phase == "terminal" and (p1 @ seg) > 0 and rng_ < 1e7:  # moving away: missed
                self.phase = "missed"
                m.log(self.t, self.name, f"Missed Bennu by {self.closest / 1e3:,.1f} km", "bad")
                self.t = t1
                return None
            self.t += h
            if self.phase == "terminal":
                self.t_go = max(0.0, float(np.linalg.norm(p1)) / max(vr, 1.0))
            if self.t >= self.next_meas:  # daily radiometric fix
                y = self.x + np.concatenate([self.rng.normal(0, SIGMA_R, 3), self.rng.normal(0, SIGMA_V, 3)])
                self.nav.update(y, SIGMA_R, SIGMA_V)
                self.last_measurement = (self.t, y)
                self.next_meas = self.t + DAY
        if self.phase == "cruise":  # at the end of a chunk too, so the mission slows down for the approach
            xb = m.bennu_helio(self.t)[0]
            self._check_terminal(np.linalg.norm(self.x[:3] - xb[:3]), np.linalg.norm(self.x[3:] - xb[3:]))
        return None

    def _nav_noise(self, t):
        """Unit-variance noise as a function of mission time (independent draws every 50 ms, interpolated), so
        the same mission gives the same noise whatever steps the integration took."""
        k = int(np.floor(t * 20.0))
        w = t * 20.0 - k
        n0, n1 = (np.random.default_rng([self.m.seed, 1000 + self.number, kk]).normal(size=3) for kk in (k, k + 1))
        return ((1 - w) * n0 + w * n1) / np.sqrt((1 - w) ** 2 + w**2)

    def _check_terminal(self, rng_, vr):
        if self.phase == "cruise" and self.ip.t_arrive - self.t < 2 * DAY:
            self.phase = "terminal"
            self.t_go = rng_ / max(vr, 1.0)
            self.m.log(self.t, self.name, f"Terminal guidance: {rng_ / 1e3:,.0f} km out, closing at {vr / 1e3:.2f} km/s"
                       + (" — aiming at the recommended point of contact" if self.contact_b is not None else ""))


def _common(m, craft):
    """The craft as the views draw it: everything evaluated at the mission clock m.t (the craft's own integration
    step may end up to half a day later; drawing it there while Earth and Bennu are drawn at m.t made it freeze
    and jump at slow warps)."""
    pos = craft.position_bary(m.t)
    recent = [p for t, p in craft.trail if m.t - 3 * YEAR < t <= m.t]  # the last three years (decades at Bennu clutter)
    s = {"phase": craft.phase, "pos": pos, "dv_used": craft.dv_used, "prop_used": craft.prop_used,
         "trail": np.array(recent + [pos]) if recent else None,
         "last_measurement": craft.last_measurement, "name": craft.name}
    if craft.nav is not None and craft.phase not in ("proximity", "impacted", "missed"):
        s["nav_err"] = float(np.linalg.norm(craft.nav.x[:3] - craft.x[:3]))
        s["nees"] = craft.nav.nees(craft.x)
        s["nav_sigma"] = craft.nav.sigma()[0]
    if craft.x is not None and craft.phase in ("cruise", "terminal"):
        xb = m.bennu_helio(m.t)[0]
        x = craft.state_at(m.t)
        s["range"] = float(np.linalg.norm(x[:3] - xb[:3]))
        s["rel_speed"] = float(np.linalg.norm(x[3:] - xb[3:]))
        s["vel_helio"] = x[3:].copy()
        s["rel_bennu"] = x[:3] - xb[:3]
    return s


def _tractor_snapshot(m, craft):
    s = _common(m, craft)
    s["launched"] = bool(np.isfinite(craft.t_launch))
    s["t_launch"] = craft.t_launch
    s["eta"] = craft.eta()
    s["thrust"] = craft.thrust
    s["thrusting"] = (craft.phase in ("spiral", "escape", "cruise") and not m.safe_mode(m.t))
    if craft.prox is not None:
        s["prox"] = craft.prox.snapshot()
        s["thrusting"] = bool(np.any(craft.prox.u)) and not m.safe_mode(m.t)
    g = craft.geo_position(m.t)
    if g is not None:
        s["geo"] = g
    gd = craft.guid
    if gd is not None and gd.dv is not None and craft.phase == "cruise" and gd.t_f > m.t:
        # remaining planned path: the current thrust profile resampled onto 16 segments, Kepler-propagated
        from core.guidance.lowthrust import _forward
        n, _ = _forward(craft.nav.x[None, :6], gd._resampled(m.t, 16), np.array([(gd.t_f - m.t) / 16]), MU_S)
        ts = m.t + (gd.t_f - m.t) * np.arange(17) / 16
        s["plan_ahead"] = n[0, :, :3] + m.eph.pos("sun", ts)
    return s


def _impactor_snapshot(m, craft):
    from core.dart import apogee_radius

    s = _common(m, craft)
    ip = craft.ip
    s.update(number=craft.number, mass=craft.mass, t_arrive=ip.t_arrive, t_launch=ip.t_launch,
             impact=craft.impact, closest=craft.closest, burns_done=craft.burn_k,
             burns_total=len(craft.rp.burns), next_burn=(craft.burn_times[craft.burn_k]
                                                          if craft.burn_k < len(craft.burn_times) else np.nan),
             t_raise_start=craft.rp.t_start, prop_plan=craft.rp.propellant_kg, dv_plan=craft.rp.dv_total,
             v_dep=craft.rp.v_dep, v_inf=craft.rp.v_inf, contact=ip.contact, impact_speed_plan=ip.impact_speed,
             db_plan=None if ip.db is None else np.asarray(ip.db, float).copy())  # the 2135 shift the team expects
    if craft.phase in ("parking", "raising", "departing"):
        r, v = craft.geo_state(min(m.t, craft.t))
        s["geo"], s["geo_vel"] = r, v
        s["orbit"] = craft.orbit()
        s["v_perigee"] = craft.v_p
        s["r_apogee"] = apogee_radius(craft.rp.r_park, craft.v_p)
        s["speed_geo"] = float(np.linalg.norm(v))
    return s


def snapshot(m):
    """Plain-data view of the mission for the UI (built on the simulation thread)."""
    out = m.outlook()
    s = {
        "t": m.t, "t_start": m.t_start, "safe_until": m.safe_until,
        "tractor": _tractor_snapshot(m, m.tractor),
        "impactors": [_impactor_snapshot(m, k) for k in m.impactors],
        "bennu_pos": m.bennu_bary(m.t),
        "events": list(m.events[-60:]), "eta_history": list(m.eta_history),
        "outlook": out, "beta_true": m.beta_true if m.bennu_impulses else None,
        "beta_measured": m.beta_measured, "beta_ground": m.beta_ground,
        "prox_controller": m.prox_controller, "on_station": m.on_station(), "bennu_R": bennu_attitude(m.t),
        "ejecta": _ejecta_snapshot(m), "busy": m.busy(), "background": m.background(), "dart_ok": m.dart_ok(),
        "spin_change": m.spin_change,
        "curve_version": m.curve_version, "complete": m.complete(), "t_end": m.t_end,
        "tractor_launched": bool(np.isfinite(m.tractor.t_launch)),
        "dart_in_flight": bool(m.impactor is not None and m.impactor.phase not in ("impacted", "missed")),
        "tow_complete": bool(m.tow_complete), "tow_end": m.tow_end,
    }
    s["impactor"] = s["impactors"][-1] if s["impactors"] else None
    return s


def verified_curve(m, step=DAY):
    """Earth distance of the deflected Bennu from the full-model integration (core/truth.py): its integrated
    segments up to t_pre and the orbit flown on from there."""
    fc = m.final_check
    t_pre = m.sc.t_pre
    t_hi = fc["t_impact"] if np.isfinite(fc.get("t_impact", np.nan)) else fc["track_t"][-1]
    t1 = np.arange(m.t_start, t_pre, step)
    tn, dn = m.truth.delta_samples()
    x1 = m.nom.state(t1)
    if len(tn):
        x1 = x1 + np.column_stack([np.interp(t1, tn, dn[:, k], left=0.0) for k in range(6)])
    tt, xx = fc["track_t"], fc["track_x"]
    sel = tt <= t_hi
    t_all = np.concatenate([t1, tt[sel]])
    pos = np.vstack([x1[:, :3], xx[sel, :3]])
    return t_all, np.linalg.norm(pos - m.eph.pos("earth", t_all), axis=1)


def _ejecta_snapshot(m):
    if m.ejecta is None or m.t < m.ejecta.t0 or m.t > m.ejecta.t0 + m.ejecta.duration + DAY:
        return None
    got = m.ejecta.at(m.t)
    if got is None:
        return {"summary": None}
    pos, code = got
    ki = next((k for k in reversed(m.impactors) if k.impact is not None), None)
    return {"pos": pos, "code": code, "summary": m.ejecta.summary(), "t0": m.ejecta.t0,
            "site_b": ki.impact["point_b"] if ki else None}
