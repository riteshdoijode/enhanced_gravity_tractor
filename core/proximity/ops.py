"""Proximity operations of the gravity tractor at Bennu (Step 5).

After the cruise hand-over (< 100 km), in the Bennu-centered frame with ICRF axes:
    approach  -> ZEM/ZEV guidance to a hold point (Guo, Hawkins & Wie 2013), optical navigation
    standoff  -> only if a DART is still on its way: hold 10 km from Bennu, off the impactor's path, until
                 the impact has happened; the navigation filter measures beta from the jump in Bennu's
                 velocity (in the keyhole mission the tractor launches after the impact, so it goes
                 straight to the survey)
    survey    -> hover 780 m above the capture site, co-rotating with Bennu (body frame)
    descent   -> down the site's vertical inside a glide-slope cone (LQR or MPC, body frame)
    capture   -> MuJoCo arm grasps and lifts the boulder (core/proximity/capture.py)
    ascent    -> climb back up with the boulder (body frame)
    transfer  -> radial climb + arc to the gravity-tractor station (inertial frame)
    station   -> hover at distance d along the tow direction with the plume-cone thrust constraint; the
                 tow itself is Step 7
    release   -> end of the tow: the boulder is let go (it stays at the release point), the craft holds there
                 with its own mass while the arm stows
    depart    -> radial climb to DEPART_RADIUS, then a small departure burn
    departed  -> coasting away from Bennu; the proximity operations are over
Truth: RK4 with polyhedron gravity (rotating with Bennu), solar tide, solar pressure with Bennu's shadow
and thrust execution errors. Navigation: EKF on LIDAR range + camera bearing (core/proximity/ekf.py).
Controllers act on the *estimate*. The stand-off hold always uses LQR (no constraint can become active
10 km out, and it may last months); elsewhere the user picks LQR or MPC. Attitude is assumed to follow its command here (full 6-DOF attitude
runs in the capture, where the arm's reaction torques matter).
"""

import numpy as np

from core.constants import AU, BENNU, G0, GM
from core.control.prox import LQR, MPC, relative_model, weights
from core.dynamics.asteroid import bennu_attitude, omega_body
from core.dynamics.perturbations import P_SUN, cylindrical_shadow
from core.proximity import sensors
from core.proximity.capture import Capture, site_frame
from core.proximity.ekf import ProxEKF

MU_S = GM["sun"]
SC_AREA, SC_CR_TRUE, SC_CR_MODEL = 120.0, 1.3, 1.2  # m^2, solar-pressure coefficients (truth / filter)
STANDOFF = 10e3  # m
D_PARK = 5e3  # m: tow off (pull 1 % of the station's)
SURVEY_ALT = 780.0  # m above the capture site
GLIDESLOPE = np.radians(60.0)
THRUST_MAG_ERR, THRUST_DIR_ERR = 0.01, np.radians(0.3)
LIFT_MARGIN = 1.3  # required thrust / weight of spacecraft + boulder at the site
WAIT_FOR_THRUST = 14 * 86400.0
PHASE_DT = {  # control period [s] per phase (truth integrates 3 RK4 steps per period)
    "approach": 300.0, "standoff": 1800.0, "survey": 60.0, "descent": 20.0, "ascent": 20.0,
    "transfer": 30.0, "station": 60.0, "parked": 600.0, "release": 60.0, "depart": 30.0, "departed": 3600.0,
}
BANDWIDTH = {"parked": 5e-4, "standoff": 2e-4, "survey": 3e-3, "descent": 6e-3, "ascent": 6e-3, "transfer": 4e-3,
             "station": 3e-3, "release": 3e-3, "depart": 4e-3}
HOLD_DETAIL_S = 86400.0  # s on station flown with the closed-loop controller before the averaged tow model
RELEASE_HOLD_S = 1800.0  # s the craft holds at the release point after letting the boulder go
DEPART_RADIUS = 50e3  # m: the departure climbs to here (where the pull on Bennu no longer counts), then burns
DEPART_DV = 0.5  # m/s radial departure burn: the craft leaves Bennu's neighbourhood (escape speed there ~1.4 cm/s)


def _min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4, 60 * s - 180 * s**2 + 120 * s**3


def _rot_by(vec, raw_axis, ang):
    """Rotate vec by the angle ang about raw_axis made perpendicular to vec (the command's pointing error)."""
    axis = raw_axis - raw_axis @ vec / max(vec @ vec, 1e-30) * vec
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return vec
    axis /= n
    return vec * np.cos(ang) + np.cross(axis, vec) * np.sin(ang)


def _rot_small(vec, angle_sigma, rng):
    axis = rng.normal(size=3)
    axis -= axis @ vec / max(vec @ vec, 1e-30) * vec
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return vec
    axis /= n
    a = rng.normal(0.0, angle_sigma)
    return vec * np.cos(a) + np.cross(axis, vec) * np.sin(a)


class ProxEnvironment:
    """Shared models (built once per process): gravity, shape ray caster, capture-site frame."""

    _shared = None

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self):
        from core.dynamics.fastgrav import BennuGravity

        self.grav = BennuGravity()
        self.ray = sensors.ShapeRaycaster()
        self.site_origin, self.site_R, self.site_g = site_frame()
        self.site_up = self.site_R[:, 2]


class ProximityOps:
    def __init__(self, mission, tractor, t0, rho0, v0):
        self.m, self.tr = mission, tractor
        if getattr(mission, "_prox_env", None) is None:
            mission._prox_env = ProxEnvironment.shared()
        self.env = env = mission._prox_env
        self.rng = mission.rng
        self.t = float(t0)
        self.x = np.concatenate([rho0, v0]).astype(float)
        self.m_sc = tractor.mass
        self.m_boulder = mission.p.boulder_mass_kg
        self.has_boulder = False
        self.thrust_err = (1.0, None)
        x_est = self.x + np.concatenate([self.rng.normal(0, 1e3, 3), self.rng.normal(0, 0.01, 3)])
        self.ekf = ProxEKF(x_est, sigma_p=1e3, sigma_v=0.01, beta0=mission.p.beta, sigma_beta=mission.p.beta_sigma)
        self.controller_kind = getattr(mission, "prox_controller", "MPC")
        self.ctrl = None
        self.ctrl_key = None
        self.u = np.zeros(3)  # current commanded acceleration (inertial)
        self.trail = []  # (t, rho) relative positions
        self.hist = {k: [] for k in ("t", "alt", "nav_err", "nav_sigma", "thrust", "a_max", "phase", "nis")}
        self.capture = Capture(self.m_sc, self.m_boulder, tractor.thrust) if self.m_boulder > 0 else None
        self.capture_result = None
        self.sc_shape = self.capture.craft_shape(False) if self.capture is not None else []
        self.boulder_mesh = self.capture.boulder_mesh_data() if self.capture is not None else None
        self.boulder_rest = None
        if self.capture is not None:  # boulder at rest on the surface (site frame) until captured
            self.boulder_rest = [g for g in self.capture.poses() if g[0] == "boulder"][0]
        self.boulder_free = None  # the boulder after its release: (geom, inertial position, rotation), fixed
        self.t_release = self.t_departed = np.nan
        self.t_station = np.nan
        self.impulses_seen = len(mission.bennu_impulses)
        self.beta_est = None
        self.beta_check_t = np.nan
        self.lidar_last = None
        self.waiting_since = None
        self.next_ctrl = self.t
        self.mpc_fallbacks = 0
        tp = mission.tractor_plan
        tow = tp.tow_direction
        self.e_station = (tow / np.linalg.norm(tow)) if tow is not None else self._sun_dir()
        self.d_station = tp.hover_distance_m if np.isfinite(tp.hover_distance_m) \
            else 2 * BENNU["radius"]
        self.site_body = env.site_origin
        self.survey_body = env.site_origin + SURVEY_ALT * env.site_up
        self.phase = None
        self.phase_done = {}
        self._was_safe = mission.safe_mode(t0)
        self._start_approach(first=True)

    # ------------------------------------------------------------------ environment model
    @property
    def mass(self):
        return self.m_sc + (self.m_boulder if self.has_boulder else 0.0)

    @property
    def a_max(self):
        return self.tr.thrust / self.mass

    def _sun_dir(self):
        s = self.m.sun(self.t)[:3] - (self.m.bennu_helio(self.t)[0, :3] + self.m.sun(self.t)[:3])
        return s / np.linalg.norm(s)

    def _env_cache(self, t):
        """Bennu's heliocentric position (tide, solar pressure), refreshed every 10 minutes."""
        if getattr(self, "_env_t", None) is None or abs(t - self._env_t) > 600.0:
            self._rB = self.m.bennu_helio(t)[0, :3]
            self._env_t = t
        return self._rB

    def _accel_env(self, t, p, cr):
        """Gravity + solar tide + solar pressure (shadowed by Bennu)."""
        rB = self._env_cache(t)
        a = self.env.grav.accel(t, p)
        r = rB + p
        a = a - MU_S * (r / np.linalg.norm(r) ** 3 - rB / np.linalg.norm(rB) ** 3)
        lit = cylindrical_shadow(p, -rB, np.zeros(3), BENNU["radius"])
        d = np.linalg.norm(r)
        a = a + lit * P_SUN * cr * SC_AREA / self.mass * (AU / d) ** 2 * r / d
        return a

    def _accel_env_many(self, ts, P, cr):
        """_accel_env for a batch of (time, position) pairs; gravity evaluated on the worker threads."""
        rB = self._env_cache(ts[0])  # Bennu's heliocentric position barely moves over one horizon
        a = self.env.grav.accel_many(ts, P)
        r = rB + P
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        a -= MU_S * (r / rn**3 - rB / np.linalg.norm(rB) ** 3)
        lit = np.array([cylindrical_shadow(pk, -rB, np.zeros(3), BENNU["radius"]) for pk in P])
        a += (lit * P_SUN * cr * SC_AREA / self.mass)[:, None] * (AU / rn) ** 2 * r / rn
        return a

    # ------------------------------------------------------------------ frames / references
    def _to_body(self, t, p, v):
        R = bennu_attitude(t)
        w_i = R @ omega_body(t)
        return R.T @ p, R.T @ (v - np.cross(w_i, p))

    def _from_body(self, t, pb, vb):
        R = bennu_attitude(t)
        p = R @ pb
        return p, R @ vb + np.cross(R @ omega_body(t), p)

    def altitude(self):
        """Distance to the surface below (LIDAR-like truth, ray toward Bennu's center)."""
        R = bennu_attitude(self.t)
        pb = R.T @ self.x[:3]
        r = np.linalg.norm(pb)
        if r > sensors.LIDAR_MAX:
            return r - BENNU["radius"]
        d, _ = self.env.ray.cast(pb, -pb / r)
        return d if d is not None else r - BENNU["radius"]

    # ------------------------------------------------------------------ phases
    def _set_phase(self, phase, text=None, level="info"):
        if self.phase is not None:
            self.phase_done[self.phase] = True
        self.phase = phase
        self.phase_t0 = self.t
        self.ctrl, self.ctrl_key = None, None
        self.next_ctrl = self.t
        if text:
            self.m.log(self.t, "tractor", text, level)

    def _impact_pending(self):
        ki = self.m.impactor
        return ki is not None and ki.phase not in ("impacted", "missed")

    def _start_approach(self, first=False):
        if self._impact_pending():
            ki = self.m.impactor
            U = ki.ip.U if ki.ip is not None and ki.ip.U is not None else np.array([1.0, 0, 0])
            u = U / np.linalg.norm(U)
            s = self._sun_dir()
            e = s - (s @ u) * u  # sunlit side, perpendicular to the impactor's path (ejecta go back along it)
            e /= np.linalg.norm(e)
            self.hold = ("inertial", STANDOFF * e)
            self.after_approach = "standoff"
            why = f"approach to a {STANDOFF / 1e3:.0f} km stand-off point (impact pending)"
        elif self.has_boulder or self.phase_done.get("capture") or self.m_boulder == 0.0 and self.phase_done.get("survey"):
            self.hold = ("inertial", self.d_station * self.e_station)
            self.after_approach = "station"
            why = "approach to the tractor station"
        else:
            self.hold = ("body", self.survey_body)
            self.after_approach = "survey"
            why = "approach to the survey point 780 m above the capture site"
        dist = np.linalg.norm(self.x[:3] - self._hold_inertial(self.t)[0])
        rng_km = np.linalg.norm(self.x[:3]) / 1e3
        self._set_phase("approach", f"Proximity operations {'begin' if first else 'continue'} at {rng_km:,.1f} km: {why}")
        self.t_go_f = self.t + max(4 * 3600.0, 1.6 * dist / 0.8)

    def _after_safe_mode(self):
        """Thrusters back after a storm: the craft has drifted (it was co-rotating or hovering on thrust).
        Anything that is not a simple hold restarts from a fresh approach (not the departure: it goes on)."""
        if self.phase in ("release", "depart", "departed"):
            return
        p = self.ekf.x[:3]
        R = bennu_attitude(self.t)
        refs = {"standoff": getattr(self, "hold", ("inertial", p))[1], "survey": R @ self.survey_body,
                "station": self.d_station * self.e_station, "parked": D_PARK * self.e_station}
        off = np.linalg.norm(p - refs[self.phase]) if self.phase in refs else np.inf
        if off > 300.0:
            self.m.log(self.t, "tractor", f"Safe mode over: {off / 1e3:,.1f} km from where the {self.phase} phase "
                                          "needs to be; re-approaching" if np.isfinite(off) else
                       f"Safe mode over during {self.phase}: aborting it and re-approaching", "warn")
            self._start_approach()

    def _hold_inertial(self, t):
        kind, p = self.hold
        if kind == "inertial":
            return p.copy(), np.zeros(3)
        return self._from_body(t, p, np.zeros(3))

    # ------------------------------------------------------------------ public
    def advance(self, t1):
        while self.t < t1 - 1e-6:
            if self.phase == "capture":  # the arm sequence is short and is not interrupted
                self._capture_step(t1)
                continue
            safe = self.m.safe_mode(self.t)
            if self._was_safe and not safe:
                self._after_safe_mode()
            self._was_safe = safe
            if self.phase == "departed":
                self._departed_fast(t1)
                continue
            if self._averaged_hold():
                self._hold_fast(t1)
                continue
            if self.phase in ("transfer", "depart") and getattr(self, "_tow_averaged", False):
                self._transfer_fast(t1)
                continue
            if getattr(self, "_fast_was", False):  # back to the detailed loop: fresh filter at the hold point
                self._fast_was = False
                self.ekf = ProxEKF(self.x + np.concatenate([self.rng.normal(0, 0.2, 3), self.rng.normal(0, 1e-4, 3)]),
                                   sigma_p=0.2, sigma_v=1e-4, beta0=self.ekf.x[9], sigma_beta=np.sqrt(self.ekf.P[9, 9]))
                self.ctrl = None
                self.next_ctrl = self.t
            if self.t >= self.next_ctrl - 1e-6:
                self._control()
                if self.phase == "capture":
                    continue
                waiting = self.phase == "survey" and self.waiting_since is not None
                self.next_ctrl = self.t + (300.0 if waiting else PHASE_DT[self.phase])
            h_end = self.next_ctrl  # whole control periods (they may end past t1: never cut by the display)
            if self.m.safe_until > self.t:
                h_end = min(h_end, self.m.safe_until)
            self._propagate(h_end)
            self._check_impact()
            if self.phase is None:
                break

    # ------------------------------------------------------------------ dynamics + navigation
    def _propagate(self, t_end):
        n_sub = 3
        h_total = t_end - self.t
        if h_total <= 0:
            return
        h = h_total / n_sub
        on = not self.m.safe_mode(self.t)
        u_cmd = self.u if on else np.zeros(3)
        rot = self.thrust_err[1]
        u_true = _rot_by(u_cmd * self.thrust_err[0], *rot) if on and np.any(u_cmd) and rot is not None else u_cmd
        for _ in range(n_sub):
            t = self.t
            f = lambda tt, p, v: (v, self._accel_env(tt, p, SC_CR_TRUE) + u_true)
            p, v = self.x[:3], self.x[3:]
            k1 = f(t, p, v)
            k2 = f(t + h / 2, p + h / 2 * k1[0], v + h / 2 * k1[1])
            k3 = f(t + h / 2, p + h / 2 * k2[0], v + h / 2 * k2[1])
            k4 = f(t + h, p + h * k3[0], v + h * k3[1])
            self.x = np.concatenate([p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                                     v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])])
            pe = self.ekf.x[:3]
            _, Gm = self.env.grav.accel_gradient(t, pe)
            self.ekf.predict(h, lambda pp, vv: self._accel_env(t, pp, SC_CR_MODEL) + u_cmd, Gm,
                             THRUST_MAG_ERR * np.linalg.norm(u_cmd) + 2e-8)
            self.t += h
        self.tr.dv_used += np.linalg.norm(u_cmd) * h_total
        self.tr.prop_used += self.mass * np.linalg.norm(u_cmd) * h_total / (self.m.p.tractor_isp_s * G0)
        self._measure()
        if not self.trail or self.t - self.trail[-1][0] >= max(60.0, PHASE_DT[self.phase]):
            self.trail.append((self.t, self.x[:3].copy()))
            if len(self.trail) > 3000:
                self.trail = self.trail[-3000:]

    def _measure(self):
        """LIDAR (or optical range) and camera bearing to Bennu's center / the capture site."""
        R = bennu_attitude(self.t)
        near_site = self.phase in ("survey", "descent", "ascent") and \
            np.linalg.norm(self.ekf.x[:3] - R @ self.site_body) < 3e3
        landmark = R @ self.site_body if near_site else np.zeros(3)
        p_true, p_est = self.x[:3], self.ekf.x[:3]
        # camera bearing
        d = landmark - p_true
        u = d / np.linalg.norm(d)
        E = sensors.tangent_basis(u)
        u_meas = u + E @ self.rng.normal(0, sensors.SIGMA_BEARING, 2)
        u_meas /= np.linalg.norm(u_meas)
        nis = self.ekf.update_bearing(u_meas, landmark, sensors.SIGMA_BEARING)
        # range along the boresight toward the landmark (pointed with the estimate)
        b = landmark - p_est
        b /= np.linalg.norm(b)
        rng_true = np.linalg.norm(landmark - p_true)
        if rng_true < sensors.LIDAR_MAX:
            hit, _ = self.env.ray.cast(R.T @ p_true, R.T @ b)
            pred, n_b = self.env.ray.cast(R.T @ p_est, R.T @ b)
            grazing = pred is not None and -(n_b @ (R.T @ b)) < np.cos(np.radians(70.0))
            if hit is not None and pred is not None and not grazing:
                y = hit + self.rng.normal(0, sensors.SIGMA_LIDAR)
                nis += self.ekf.update_range(y, pred, R @ n_b, b, sensors.SIGMA_LIDAR + 0.05)
                self.lidar_last = y
        else:  # optical range from Bennu's apparent size
            sig = sensors.SIGMA_RANGE_OPTICAL * rng_true
            y = rng_true + self.rng.normal(0, sig)
            pred = np.linalg.norm(landmark - p_est)
            nis += self.ekf.update_range(y, pred, -b, b, sig)
            self.lidar_last = None
        H = self.hist
        H["t"].append(self.t)
        H["alt"].append(self.altitude() if np.linalg.norm(self.x[:3]) < 20e3 else np.linalg.norm(self.x[:3]))
        H["nav_err"].append(float(np.linalg.norm(self.ekf.x[:3] - self.x[:3])))
        H["nav_sigma"].append(self.ekf.sigma_pos())
        H["thrust"].append(float(np.linalg.norm(self.u) * self.mass))
        H["a_max"].append(self.tr.thrust)
        H["phase"].append(self.phase)
        H["nis"].append(nis)
        if len(H["t"]) > 20000:
            for k in H:
                H[k] = H[k][-20000:]

    def _check_impact(self):
        imps = self.m.bennu_impulses
        if len(imps) > self.impulses_seen:
            t_imp, dv = imps[-1]
            self.impulses_seen = len(imps)
            late = max(0.0, self.t - t_imp)  # the impact may fall inside the step just flown
            self.x[3:] -= dv  # Bennu jumps, the spacecraft does not
            self.x[:3] -= dv * late
            ki = self.m.impactor
            if ki is not None and ki.impact is not None:  # dv = c + beta k (Cheng eq. 1, known geometry)
                U, E = ki.impact["U"], ki.impact["E_hat"]
                k = ki.mass * (E @ U) * E / BENNU["mass"]
                c = ki.mass * (U - (E @ U) * E) / BENNU["mass"]
                self.ekf.impact_jump(k, late, c)
                self.beta_check_t = self.t + 2 * 86400.0
            d = np.linalg.norm(self.x[:3])
            self.m.log(self.t, "tractor", f"Watched the impact from {d / 1e3:,.1f} km: Bennu moved "
                                          f"{np.linalg.norm(dv) * 1e3:.1f} mm/s relative to the spacecraft")
        if np.isfinite(self.beta_check_t) and self.t >= self.beta_check_t:
            self.beta_check_t = np.nan
            b, s = float(self.ekf.x[9]), float(np.sqrt(self.ekf.P[9, 9]))
            self.beta_est = (b, s)
            self.m.beta_measured = (b, s)
            self.m.log(self.t, "tractor", f"Navigation filter measured β = {b:.2f} ± {s:.2f} from the "
                                          "spacecraft's motion relative to Bennu", "good")

    # ------------------------------------------------------------------ guidance & control
    def _controller(self, key, build):
        kind = "LQR" if self.phase == "standoff" else self.controller_kind
        if self.ctrl_key != key or self.ctrl is None or self.ctrl.name != kind:
            self.ctrl = build()
            self.ctrl_key = key
        return self.ctrl

    def _make(self, A, B, phase, N=30, gamma=None, keepout=False, glideslope=None):
        Q, R = weights(BANDWIDTH[phase])
        if self.controller_kind == "MPC" and phase != "standoff":  # 10 km out no constraint is ever active
            return MPC(A, B, Q, R, N, gamma, keepout=keepout, glideslope=glideslope, omega_n=BANDWIDTH[phase])
        return LQR(A, B, Q, R)

    def _control(self):
        if self.m.safe_mode(self.t):
            self.u = np.zeros(3)
            return
        # execution error of this command (magnitude and pointing), drawn once per control tick
        self.thrust_err = (1.0 + self.rng.normal(0, THRUST_MAG_ERR), (self.rng.normal(size=3),
                                                                      self.rng.normal(0.0, THRUST_DIR_ERR)))
        getattr(self, f"_ctl_{self.phase}")()
        if isinstance(self.ctrl, MPC):
            self.mpc_fallbacks = self.ctrl.fallbacks

    # ---- approach (ZEM/ZEV)
    def _ctl_approach(self):
        t_go = self.t_go_f - self.t
        if t_go < 2 * PHASE_DT["approach"]:
            if self.after_approach == "standoff":
                self._set_phase("standoff", f"Holding {STANDOFF / 1e3:.0f} km from Bennu until the impact")
            elif self.after_approach == "station":
                self.t_station = self.t
                self._set_phase("station", f"On station {self.d_station:,.0f} m from Bennu's center: the gravity "
                                           f"tractor is towing ({self.mass / 1e3:,.1f} t)", "good")
            else:
                self._set_phase("survey", "Surveying the capture site (Nightingale crater, 56°N 43°E) from 780 m")
            self._control_phase_now()
            return
        rT, vT = self._hold_inertial(self.t_go_f)
        p, v = self.ekf.x[:3], self.ekf.x[3:6]
        g = self.env.grav.accel(self.t, p)
        zem = rT - (p + v * t_go + 0.5 * g * t_go**2)
        zev = vT - (v + g * t_go)
        a = 6.0 * zem / t_go**2 - 2.0 * zev / t_go - self.ekf.bias
        n = np.linalg.norm(a)
        self.u = a * min(1.0, self.a_max / max(n, 1e-30))
        if t_go < 3 * 3600:
            self.next_ctrl = self.t + 60.0

    def _control_phase_now(self):
        self.next_ctrl = self.t
        getattr(self, f"_ctl_{self.phase}")()

    # ---- inertial hold / tracking
    def _track_inertial(self, phase, ref, gamma=None, keepout=True):
        """ref(t) -> (p, v, a) inertial; feedforward + LQR/MPC on the estimate."""
        dt = PHASE_DT[phase]
        p0, _, _ = ref(self.t)
        d = np.linalg.norm(p0)
        e = p0 / d
        G = BENNU["gm"] / d**3 * (3 * np.outer(e, e) - np.eye(3))
        key = (phase, round(d, -1) if phase not in ("transfer", "depart") else round(self.t / 600.0))
        ctrl = self._controller(key, lambda: self._make(*relative_model(G, dt), phase, gamma=gamma, keepout=keepout))
        N = getattr(ctrl, "N", 1)
        ts = self.t + dt * np.arange(N)
        refs = [ref(tk) for tk in ts]
        P = np.array([pk for pk, _, _ in refs])
        uff = np.array([a for _, _, a in refs]) - self._accel_env_many(ts, P, SC_CR_MODEL) - self.ekf.bias
        axes = np.array([pk / np.linalg.norm(pk) for pk, _, _ in refs])
        dx = np.concatenate([self.ekf.x[:3] - refs[0][0], self.ekf.x[3:6] - refs[0][1]])
        r_ko = self.env.grav.R_bound + 60.0
        ko = np.array([r_ko - ax @ pk for ax, (pk, _, _) in zip(axes, refs)]) if keepout else None
        self.u, _ = ctrl.command(dx, uff, axes, self.a_max, gamma=gamma, keepout=ko)

    def _ctl_standoff(self):
        if not self._impact_pending() and self.t - self.phase_t0 > 3 * 86400.0 and not np.isfinite(self.beta_check_t):
            self._start_approach()
            return
        p_hold = self.hold[1]
        self._track_inertial("standoff", lambda t: (p_hold, np.zeros(3), np.zeros(3)), keepout=False)

    def _ctl_station(self):
        p_st = self.d_station * self.e_station
        gamma = np.arcsin(min(0.99, self.env.grav.R_bound / self.d_station)) + np.radians(self.m.p.plume_half_angle_deg)
        self._track_inertial("station", lambda t: (p_st, np.zeros(3), np.zeros(3)), gamma=gamma)

    def _ctl_parked(self):
        p_park = D_PARK * self.e_station
        self._track_inertial("parked", lambda t: (p_park, np.zeros(3), np.zeros(3)), keepout=False)

    def retarget(self, e, tow_on=True):
        """Trim guidance moved the station (new tow direction) or switched the tow on/off."""
        e = np.asarray(e, float) / np.linalg.norm(e)
        want = "station" if tow_on else "parked"
        d = self.d_station if tow_on else D_PARK
        turn = np.degrees(np.arccos(np.clip(e @ self.e_station, -1, 1)))
        self.e_station = e
        if self.phase not in ("station", "parked", "transfer"):
            return
        if self.phase == want and turn < 3.0:
            return  # small turn: the station-keeping controller follows the moving reference
        self._start_transfer(d, e, then=want)

    def _averaged_hold(self):
        """On station (or parked) for more than a day: the multi-year tow uses averaged station-keeping. The
        model follows the mission time only, never the display speed; the first day on station is flown with
        the closed-loop controller and every transfer between hold points is flown in full."""
        return self.phase in ("station", "parked") and (getattr(self, "_tow_averaged", False)
                                                         or self.t - self.phase_t0 >= HOLD_DETAIL_S)

    def _transfer_fast(self, t1):
        """Averaged transfer between hold points once the tow runs on the averaged model: the craft follows its
        reference path (the closed-loop tracking of Step 5 holds it there); thrust = the reference acceleration
        minus the environment, propellant integrated. In safe mode the transfer waits."""
        while self.t < t1 - 1e-6 and self.phase in ("transfer", "depart"):
            if self.t - self.phase_t0 >= self.transfer_T:
                self._ctl_transfer()  # arrival: station, parked or departed
                return
            h = 300.0
            if self.m.safe_mode(self.t):
                self.phase_t0 += h  # the path waits for the thrusters
                self.u = np.zeros(3)
            else:
                p, v, a_ref = self._transfer_ref(self.t)
                self.x = np.concatenate([p, v])
                self.u = a_ref - self._accel_env(self.t, p, SC_CR_TRUE)
                a = np.linalg.norm(self.u)
                self.tr.dv_used += a * h
                self.tr.prop_used += self.mass * a * h / (self.m.p.tractor_isp_s * G0)
            self.t += h
            self.ekf.x[:6] = self.x
            if not self.trail or self.t - self.trail[-1][0] >= 600.0:
                self.trail.append((self.t, self.x[:3].copy()))
                self.trail = self.trail[-3000:]
        self._fast_was = True

    def _hold_fast(self, t1):
        """Averaged station-keeping for the tow: the craft sits on its hold point (the closed-loop hover keeps
        it within centimetres, Step 5); thrust = the hover thrust, propellant integrated."""
        d = self.d_station if self.phase == "station" else D_PARK
        while self.t < t1 - 1e-6:
            h = 86400.0  # whole days on the hold point (independent of how the mission is chunked)
            p = d * self.e_station
            self.x = np.concatenate([p, np.zeros(3)])
            self.u = -self._accel_env(self.t, p, SC_CR_TRUE) if not self.m.safe_mode(self.t) else np.zeros(3)
            a = np.linalg.norm(self.u)
            self.tr.dv_used += a * h
            self.tr.prop_used += self.mass * a * h / (self.m.p.tractor_isp_s * G0)
            self.t += h
            self.ekf.x[:6] = self.x
            if not self.trail or self.t - self.trail[-1][0] >= 86400.0:
                self.trail.append((self.t, self.x[:3].copy()))
                self.trail = self.trail[-3000:]
            H = self.hist
            for k, v in (("t", self.t), ("alt", d - BENNU["radius"]), ("nav_err", np.nan), ("nav_sigma", np.nan),
                         ("thrust", a * self.mass), ("a_max", self.tr.thrust), ("phase", self.phase), ("nis", np.nan)):
                H[k].append(v)
        self._fast_was = True
        self._tow_averaged = True  # from now on the tow (holds and transfers) runs on the averaged model

    def _ctl_transfer(self):
        T = self.transfer_T
        s_now = (self.t - self.phase_t0) / T
        if s_now >= 1.0:
            d, e, then = self.tr_target
            if then == "station":
                if not np.isfinite(self.t_station):
                    self.t_station = self.t
                self._set_phase("station", f"On station {self.d_station:,.0f} m from Bennu's center along the tow "
                                           f"direction: the gravity tractor is towing ({self.mass / 1e3:,.1f} t)", "good")
            elif then == "departed":
                self._departed()
                return
            else:
                self._set_phase("parked", f"Parked {D_PARK / 1e3:.0f} km from Bennu (tow off)")
            getattr(self, f"_ctl_{self.phase}")()
            return
        self._track_inertial(self.phase, self._transfer_ref)

    _ctl_depart = _ctl_transfer

    # ---- end of the tow: release the boulder, depart
    def release(self):
        """End of the tow (core/mission.py end_tow): the boulder is let go where the craft is (it stays there as a
        marker: at 5 km from Bennu it drifts for months), the mass properties revert to the bare spacecraft and the
        craft holds the point for RELEASE_HOLD_S while the arm stows, then departs. True if it could release."""
        if self.phase not in ("station", "parked", "transfer"):
            return False
        had = self.has_boulder
        if had:
            shape = {g[0]: g for g in self.sc_shape}
            n, ty, sz, rgba, p_b, R_b = shape["boulder"]  # bus frame -> Bennu-centered inertial
            Rsc = self.attitude()
            self.boulder_free = ((n, ty, sz, rgba), self.x[:3] + Rsc @ p_b, Rsc @ R_b)
            self.has_boulder = False
            self.sc_shape = self.capture.craft_shape(False)
            self.m_boulder = 0.0
        self.t_release = self.t
        self.hold = ("inertial", self.x[:3].copy())
        self._fast_was = True  # the release hold is flown with the closed loop: fresh filter at the hold point
        self._set_phase("release", (f"Boulder released {np.linalg.norm(self.x[:3]) / 1e3:.1f} km from Bennu: the "
                                    f"spacecraft is {self.m_sc / 1e3:,.1f} t again" if had else
                                    "End of the tow: the spacecraft prepares to leave Bennu"), "good")
        return True

    def _ctl_release(self):
        if self.t - self.phase_t0 >= RELEASE_HOLD_S:
            self._tow_averaged = True  # the days-long climb follows its reference path (as the averaged transfers do)
            self._start_transfer(DEPART_RADIUS, self.e_station, then="departed", phase="depart")
            return
        p_hold = self.hold[1]
        self._track_inertial("release", lambda t: (p_hold, np.zeros(3), np.zeros(3)), keepout=False)

    def _departed(self):
        """At DEPART_RADIUS: a small radial burn, then the craft coasts away (no more station-keeping)."""
        u = self.x[:3] / np.linalg.norm(self.x[:3])
        self.x[3:] = DEPART_DV * u  # the burn leaves it receding radially (the transfer ended at rest)
        self.ekf.x[:6] = self.x
        self.tr.dv_used += DEPART_DV
        self.tr.prop_used += self.mass * DEPART_DV / (self.m.p.tractor_isp_s * G0)
        self.u = np.zeros(3)
        self.t_departed = self.t
        self._set_phase("departed", f"Departure burn {DEPART_DV:g} m/s at {np.linalg.norm(self.x[:3]) / 1e3:.0f} km: "
                                    "the gravity tractor leaves Bennu", "good")

    def _ctl_departed(self):
        self.u = np.zeros(3)

    def _departed_fast(self, t1):
        """Coasting away after the departure burn: a straight line relative to Bennu (its gravity at > 50 km with
        0.5 m/s is negligible), no thrust; daily samples for the plots."""
        while self.t < t1 - 1e-6:
            h = min(86400.0, t1 - self.t)
            self.x[:3] += self.x[3:] * h
            self.t += h
            self.ekf.x[:6] = self.x
            if not self.trail or self.t - self.trail[-1][0] >= 86400.0:
                self.trail.append((self.t, self.x[:3].copy()))
                self.trail = self.trail[-3000:]
            H = self.hist
            for k, v in (("t", self.t), ("alt", np.linalg.norm(self.x[:3]) - BENNU["radius"]), ("nav_err", np.nan),
                         ("nav_sigma", np.nan), ("thrust", 0.0), ("a_max", self.tr.thrust), ("phase", self.phase),
                         ("nis", np.nan)):
                H[k].append(v)

    def _transfer_ref(self, t):
        """Radial climb to the station radius while rotating the direction toward the tow axis."""
        T = self.transfer_T
        s = (t - self.phase_t0) / T
        f, fd, fdd = _min_jerk(s)
        r0, r1 = self.tr_r0, self.tr_target[0]
        r = r0 + (r1 - r0) * f
        rd = (r1 - r0) * fd / T
        rdd = (r1 - r0) * fdd / T**2
        th = self.tr_theta * f
        thd = self.tr_theta * fd / T
        thdd = self.tr_theta * fdd / T**2
        u0, w = self.tr_u0, self.tr_w  # unit start direction, unit perpendicular in the rotation plane
        e = np.cos(th) * u0 + np.sin(th) * w
        et = -np.sin(th) * u0 + np.cos(th) * w
        p = r * e
        v = rd * e + r * thd * et
        a = (rdd - r * thd**2) * e + (2 * rd * thd + r * thdd) * et
        if s >= 1.0:
            return self.tr_target[0] * self.tr_target[1], np.zeros(3), np.zeros(3)
        return p, v, a

    # ---- body-frame hover / descent / ascent
    def _track_body(self, phase, ref, glide=True, dt=None):
        """ref(t) -> (p_b, v_b, a_b) in Bennu's body frame; command returned inertial."""
        dt = dt or PHASE_DT[phase]
        pb0, _, _ = ref(self.t)
        G = self.env.grav.gradient_body(pb0)
        key = (phase, dt, round(4 * np.log(max(float(np.linalg.norm(pb0 - self.site_body)), 1.0))))
        W = omega_body(self.t)
        key = key + (round(float(W[2]) * 1e9),)
        ctrl = self._controller(key, lambda: self._make(*relative_model(G, dt, omega=W), phase,
                                                        glideslope=(self.env.site_up, GLIDESLOPE) if glide else None))
        N = getattr(ctrl, "N", 1)
        ts = self.t + dt * np.arange(N)
        refs = [ref(tk) for tk in ts]
        Rs = np.array([bennu_attitude(tk) for tk in ts])
        Pb = np.array([pk for pk, _, _ in refs])
        Vb = np.array([vk for _, vk, _ in refs])
        Ab = np.array([ak for _, _, ak in refs])
        env_i = self._accel_env_many(ts, np.einsum("nij,nj->ni", Rs, Pb), SC_CR_MODEL) + self.ekf.bias
        env_b = np.einsum("nji,nj->ni", Rs, env_i)
        fict = -2 * np.cross(W, Vb) - np.cross(W, np.cross(W, Pb))
        uff = Ab - env_b - fict
        pb, vb = self._to_body(self.t, self.ekf.x[:3], self.ekf.x[3:6])
        dx = np.concatenate([pb - refs[0][0], vb - refs[0][1]])
        axes = np.tile(self.env.site_up, (N, 1))
        gs = np.array([pk - self.site_body for pk, _, _ in refs]) if glide else None
        u_b, _ = ctrl.command(dx, uff, axes, self.a_max, gamma=None, glideslope=gs)
        self.u = bennu_attitude(self.t) @ u_b

    def _lift_thrust_needed(self):
        g = np.linalg.norm(self.env.site_g)
        return LIFT_MARGIN * (self.m_sc + self.m_boulder) * g

    def _ctl_survey(self):
        sv = self.survey_body
        self._track_body("survey", lambda t: (sv, np.zeros(3), np.zeros(3)), glide=False,
                         dt=300.0 if self.waiting_since is not None else None)
        if self.t - self.phase_t0 < 6 * 3600.0:
            return
        if self.capture is None:
            self.m.log(self.t, "tractor", "No boulder in the plan: classic gravity tractor")
            self._start_transfer()
            return
        need = self._lift_thrust_needed()
        if self.tr.thrust >= need and self.waiting_since is not None:
            self.m.log(self.t, "tractor", f"Thrust {self.tr.thrust:g} N is enough to lift the boulder: descending", "good")
            self.waiting_since = None
            self.ctrl = None
        if self.tr.thrust < need:
            if self.waiting_since is None:
                self.waiting_since = self.t
                self.m.log(self.t, "tractor", f"Capture on hold: lifting {self.m_boulder / 1e3:.0f} t off the surface "
                                              f"needs ≥ {need:.2f} N (now {self.tr.thrust:g} N). Raise the thrust.", "warn")
            if self.t - self.waiting_since > WAIT_FOR_THRUST:
                self.m.log(self.t, "tractor", "No boulder: continuing as a classic gravity tractor", "warn")
                self.m_boulder = 0.0
                self._start_transfer()
            return
        self._start_descent()

    def _start_descent(self):
        cap = self.capture
        cap.thrust_max = self.tr.thrust
        self.descent_target = self.site_body + self.env.site_R @ cap.bus_ref0  # bus center, body frame
        pb, _ = self._to_body(self.t, self.ekf.x[:3], self.ekf.x[3:6])
        self.desc_from = pb
        h = np.linalg.norm(pb - self.descent_target)
        T_v = 1.875 * h / 0.15  # peak descent speed 15 cm/s
        T_a = np.sqrt(5.77 * h / (0.4 * self.a_max))
        self.desc_T = max(T_v, T_a, 1800.0)
        self._set_phase("descent", f"Descent to the boulder: {h:,.0f} m in {self.desc_T / 3600:.1f} h, "
                                   f"{self.controller_kind}, {np.degrees(GLIDESLOPE):.0f}° glide-slope cone")

    def _descent_ref(self, t):
        T = self.desc_T
        s = (t - self.phase_t0) / T
        f, fd, fdd = _min_jerk(s)
        dp = self.descent_target - self.desc_from
        if s >= 1.0:
            return self.descent_target, np.zeros(3), np.zeros(3)
        return self.desc_from + dp * f, dp * fd / T, dp * fdd / T**2

    def _ctl_descent(self):
        if self.t - self.phase_t0 >= self.desc_T + 300.0:
            pb, vb = self._to_body(self.t, self.x[:3], self.x[3:])
            err = np.linalg.norm(pb - self.descent_target)
            if err < 1.0 and np.linalg.norm(vb) < 0.02 or self.t - self.phase_t0 > self.desc_T + 3600:
                self._start_capture()
                return
        self._track_body("descent", self._descent_ref)

    def _start_capture(self):
        cap = self.capture
        R = self.env.site_R
        cap.omega = R.T @ omega_body(self.t)  # the spin may have changed at the impact
        pb, vb = self._to_body(self.t, self.x[:3], self.x[3:])
        loc_p = R.T @ (pb - self.site_body)
        loc_v = R.T @ vb
        d = cap.data
        d.qpos[0:3] = loc_p
        d.qvel[0:3] = loc_v
        cap.bus_ref = cap.bus_ref0.copy()
        import mujoco

        mujoco.mj_forward(cap.model, d)
        self.cap_stage = cap.stage
        self.u = np.zeros(3)
        self._set_phase("capture", f"Capture: hovering {np.linalg.norm(loc_p - cap.bus_ref0):.2f} m from the grasp "
                                   f"point, arm deploying")

    def _capture_step(self, t1):
        cap = self.capture
        dt = 5.0  # fixed steps of the capture simulation (independent of how the mission is chunked)
        cap.advance(dt)
        self.t += dt
        if cap.stage != self.cap_stage:
            text = {"reach": "Capture: palm reaching the boulder", "grip": "Capture: claws closing, microspines engaging",
                    "lift": f"Capture: spines anchored, lifting off with the {self.m_boulder / 1e3:.0f} t boulder",
                    "stow": "Capture: stowing the boulder under the bus", "done": "Capture: settling"}.get(cap.stage)
            if text:
                self.m.log(self.t, "tractor", text)
            self.cap_stage = cap.stage
        # spacecraft (bus) position for display / truth bookkeeping
        bus_p, bus_v = cap.bus_state()
        pb = self.site_body + self.env.site_R @ bus_p
        vb = self.env.site_R @ bus_v
        self.x = np.concatenate(self._from_body(self.t, pb, vb))
        if cap.stage == "finished":
            self._finish_capture()

    def _finish_capture(self):
        cap = self.capture
        res = cap.result
        self.capture_result = res
        mp = res["mass"]
        R = self.env.site_R
        # merged spacecraft: center of mass of bus + arm + boulder
        jac_v = cap.momentum() / mp["mass"]
        pb = self.site_body + R @ mp["com"]
        vb = R @ jac_v
        self.x = np.concatenate(self._from_body(self.t, pb, vb))
        if res["success"]:
            self.has_boulder = True
            self.sc_shape = cap.craft_shape(True)
            self.boulder_rest = None
            self.m.log(self.t, "tractor", f"Boulder captured: lifted {res['lifted_m']:.1f} m, grip slip "
                                          f"{res['grip_slip_m'] * 100:.0f} cm. Spacecraft + boulder = "
                                          f"{self.mass / 1e3:,.1f} t", "good")
        else:
            self.m_boulder = 0.0
            self.m.log(self.t, "tractor", "Capture failed: continuing without the boulder", "bad")
        x_est = self.x + np.concatenate([self.rng.normal(0, 0.3, 3), self.rng.normal(0, 0.003, 3)])
        old = self.ekf
        self.ekf = ProxEKF(x_est, sigma_p=0.3, sigma_v=0.003, beta0=old.x[9], sigma_beta=np.sqrt(old.P[9, 9]))
        self.asc_from = pb
        self.asc_to = self.site_body + 200.0 * self.env.site_up
        self.asc_T = max(1.875 * 190.0 / 0.15, np.sqrt(5.77 * 190.0 / (0.3 * max(self.a_max - np.linalg.norm(self.env.site_g), 1e-7))))
        self._set_phase("ascent", f"Ascent to 200 m with {self.mass / 1e3:,.1f} t")

    def _ascent_ref(self, t):
        T = self.asc_T
        s = (t - self.phase_t0) / T
        f, fd, fdd = _min_jerk(s)
        dp = self.asc_to - self.asc_from
        if s >= 1.0:
            return self.asc_to, np.zeros(3), np.zeros(3)
        return self.asc_from + dp * f, dp * fd / T, dp * fdd / T**2

    def _ctl_ascent(self):
        if self.t - self.phase_t0 >= self.asc_T + 120.0:
            self._start_transfer()
            return
        self._track_body("ascent", self._ascent_ref)

    def _start_transfer(self, d=None, e=None, then="station", phase="transfer"):
        if e is None and then == "station" and self.m.trim is None and len(self.m.bennu_impulses) and hasattr(self.m, "tow_direction_now"):
            e_trim = self.m.tow_direction_now(self.t)  # first station: already along the trim law
            if e_trim is not None:
                self.e_station = e_trim
        d = self.d_station if d is None else d
        e = self.e_station if e is None else e
        self.tr_target = (d, e, then)
        p = self.ekf.x[:3]
        self.tr_r0 = float(np.linalg.norm(p))
        self.tr_u0 = p / self.tr_r0
        c = np.clip(self.tr_u0 @ e, -1.0, 1.0)
        self.tr_theta = float(np.arccos(c))
        w = e - c * self.tr_u0
        nw = np.linalg.norm(w)
        self.tr_w = w / nw if nw > 1e-9 else np.cross(self.tr_u0, [0, 0, 1.0]) / np.linalg.norm(np.cross(self.tr_u0, [0, 0, 1.0]))
        path = abs(d - self.tr_r0) + self.tr_theta * max(d, self.tr_r0)
        self.transfer_T = max(1.875 * path / 0.2, 3600.0)
        # the start velocity is the co-rotating one; the first minutes of tracking absorb it
        if phase == "depart":
            self._set_phase("depart", f"Departure: radial climb from {self.tr_r0 / 1e3:.1f} km to {d / 1e3:.0f} km, "
                                      f"{self.transfer_T / 86400:.1f} days")
            return
        where = "the tractor station" if then == "station" else f"the {D_PARK / 1e3:.0f} km parking point"
        self._set_phase("transfer", f"Transfer to {where}: {np.degrees(self.tr_theta):.0f}° around Bennu, "
                                    f"{self.transfer_T / 3600:.1f} h")

    # ------------------------------------------------------------------ status
    def snapshot(self):
        R = bennu_attitude(self.t)
        s = {
            "phase": self.phase, "rho": self.x[:3].copy(), "v_rel": self.x[3:].copy(), "R_bennu": R,
            "thrust_vec": self.u * self.mass, "thrust_max": self.tr.thrust, "mass": self.mass,
            "has_boulder": self.has_boulder, "controller": self.controller_kind,
            "nav_err": float(np.linalg.norm(self.ekf.x[:3] - self.x[:3])), "nav_sigma": self.ekf.sigma_pos(),
            "range": float(np.linalg.norm(self.x[:3])), "altitude": self.hist["alt"][-1] if self.hist["alt"] else np.nan,
            "lidar": self.lidar_last, "site_body": self.site_body, "beta_est": self.beta_est,
            "trail": np.array([p for _, p in self.trail[-600:]]) if len(self.trail) > 1 else None,
            "hist": {k: np.asarray(v[-400:]) for k, v in self.hist.items() if k != "phase"},
            "station": self.d_station * self.e_station, "t_station": self.t_station,
            "mpc_fallbacks": self.mpc_fallbacks, "t_release": self.t_release, "t_departed": self.t_departed,
            "boulder_free": None if self.boulder_free is None else self.boulder_free[1].copy(),
        }
        Rs, o = self.env.site_R, self.site_body
        to_i = lambda p, Rm: (R @ (o + Rs @ p), R @ Rs @ Rm)  # site frame -> Bennu-centered ICRF
        if self.phase == "capture":  # the arm's own close-range sensing guides the grasp; the EKF is paused
            s["capture_stage"] = self.capture.stage
            s["nav_err"] = s["nav_sigma"] = np.nan
            s["lidar"] = None
            s["geoms"] = [(n, ty, sz, rgba, *to_i(p, Rm)) for n, ty, sz, rgba, p, Rm in self.capture.poses()]
        else:
            Rsc = self.attitude()
            s["geoms"] = [(n, ty, sz, rgba, self.x[:3] + Rsc @ p, Rsc @ Rm) for n, ty, sz, rgba, p, Rm in self.sc_shape]
            if self.boulder_rest is not None:
                n, ty, sz, rgba, p, Rm = self.boulder_rest
                s["geoms"].append((n, ty, sz, rgba, *to_i(p, Rm)))
            if self.boulder_free is not None:  # the released boulder, left where it was let go
                (n, ty, sz, rgba), p, Rm = self.boulder_free
                s["geoms"].append((n, ty, sz, rgba, p, Rm))
        s["boulder_mesh"] = self.boulder_mesh
        s["boresight_target"] = R @ self.site_body if self.phase in ("survey", "descent", "ascent") else np.zeros(3)
        s["site"] = R @ self.site_body
        return s

    def attitude(self):
        """Displayed bus attitude: arm side (-z) toward the surface: the site's vertical near the site,
        Bennu's center elsewhere (attitude control assumed to track this outside the capture)."""
        R = bennu_attitude(self.t)
        if self.phase in ("survey", "descent", "ascent"):
            return R @ self.env.site_R
        z = self.x[:3] / max(np.linalg.norm(self.x[:3]), 1e-9)
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= max(np.linalg.norm(x), 1e-9)
        return np.column_stack([x, np.cross(z, x), z])
