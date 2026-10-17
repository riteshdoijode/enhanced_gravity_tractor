"""Keyhole scenario (hypothetical): Bennu's 2135 flyby threads the keyhole of a later Earth impact.

Bennu's orbit is known so well that, by 2135, only its position along one line of the encounter's
B-plane is uncertain: the 2060 flyby stretches any small early difference about 4,000 times along the
timing axis zeta (the sensitivity of the 2135 B-plane point to a push in 2026 is rank one: 1e7 km per mm/s
along that line, a few km per mm/s across it). A *keyhole* is a narrow window on that line from which the
post-flyby orbit returns to hit Earth years later (JPL's most likely Bennu impact date is 2182-09-24).
The scenario assumes Bennu goes through it: it picks the point c on the line of variations whose
post-flyby orbit hits Earth (a virtual impactor), and builds its trajectory:

    before t_pre (60 days before the 2135 flyby):  x(t) = x_nom(t) + Phi(t, t0) c v1
        (linear in a change c v1 of the 2026 state of ~1e-5 m/s, exactly the model the planner uses)
    after t_pre: the full nonlinear force model from x(t_pre), with the STM, through the keyhole flyby
        to the Earth impact (integration stopped at Earth's surface).

Nonlinear checks of a deflection never re-integrate the undeflected orbit from 2026: a millimetre of
integration difference before 2060 moves the 2135 point by kilometres, more than the keyhole is wide.
They add the nonlinear *difference* the pushes make, at t_pre, to this stored state (core/tow.py).

B-plane convention: the modified target plane at the closest approach (normal to the Earth-relative
velocity, axes xi and zeta as in core/deflection.py); the miss vector is the closest-approach position,
so its length is the flyby distance itself (gravitational focusing is inside the trajectory and the STM).
"""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from core.constants import AU, DATA_DIR, DAY, GM, LUNAR_DISTANCE, RADIUS
from core.deflection import BPlane
from core.dynamics.nominal import J6, NominalBennu

SCENARIO_FILE = DATA_DIR / "scenario_keyhole.json"
VI_FILE = DATA_DIR / "bennu_vi.npz"
KEYHOLE_DATE = "2135-09-25 07:49"
PRE_DAYS = 60.0  # the scenario trajectory is nonlinear from this many days before the 2135 flyby


def _inv(P):
    return -J6 @ np.swapaxes(P, -1, -2) @ J6


def find_close_approaches(state_fn, eph, t0, t1, max_dist=None, step=0.25 * DAY):
    """Local minima of the Earth distance of a trajectory state_fn(t) -> (..., 6) barycentric, refined.
    Returns a list of dicts {t, dist, v_rel} sorted in time."""
    t = np.arange(t0, t1, step)
    d = np.linalg.norm(state_fn(t)[:, :3] - eph.pos("earth", t), axis=1)
    idx = np.where((d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]))[0] + 1
    out = []
    for k in idx:
        if max_dist is not None and d[k] > 1.5 * max_dist:
            continue
        f = lambda tt: float(np.linalg.norm(state_fn(tt)[:3] - eph.pos("earth", tt)))
        res = minimize_scalar(f, bounds=(t[k] - step, t[k] + step), method="bounded", options={"xatol": 1.0})
        if max_dist is not None and res.fun > max_dist:
            continue
        x = state_fn(res.x)
        out.append({"t": float(res.x), "dist": float(res.fun),
                    "v_rel": float(np.linalg.norm(x[3:] - eph.vel("earth", res.x)))})
    return out


def mtp(eph, t, x):
    """BPlane at time t (closest approach) of the barycentric state x, and the 2-D miss vector."""
    r_rel, v_rel = x[:3] - eph.pos("earth", t), x[3:] - eph.vel("earth", t)
    bp = BPlane(v_rel, eph.vel("earth", t) - eph.vel("sun", t))
    return bp, bp.project(r_rel)


def b_from_hyperbola(r_rel, v_rel, v_earth_helio):
    """Asymptotic B-plane vector (2-D, xi/zeta) of the geocentric hyperbola through (r_rel, v_rel): used for
    an Earth-impacting approach without integrating into the Earth. Returns (b, |B|, v_inf)."""
    mu = GM["earth"]
    h = np.cross(r_rel, v_rel)
    v2 = v_rel @ v_rel
    v_inf = np.sqrt(v2 - 2 * mu / np.linalg.norm(r_rel))
    e_vec = np.cross(v_rel, h) / mu - r_rel / np.linalg.norm(r_rel)
    e = np.linalg.norm(e_vec)
    P = e_vec / e
    W = h / np.linalg.norm(h)
    Q = np.cross(W, P)
    k = np.sqrt(e**2 - 1)
    s = (P + k * Q) / e  # incoming asymptote
    B = np.linalg.norm(h) / v_inf * (k * P - Q) / e
    bp = BPlane(s, v_earth_helio)
    return bp.project(B), float(np.linalg.norm(B)), float(v_inf)


def earth_capture_b(v_inf):
    return RADIUS["earth"] * np.sqrt(1.0 + 2.0 * GM["earth"] / (RADIUS["earth"] * v_inf**2))


@dataclass
class KeyholeScenario:
    t0: float
    t_pre: float
    t_ca: float  # 2135 keyhole flyby of the scenario trajectory
    b0: np.ndarray  # its miss vector (xi, zeta) [m]
    basis: np.ndarray  # 2 x 3 B-plane axes
    v_rel: np.ndarray  # Earth-relative velocity at the flyby
    lov: np.ndarray  # 2-D direction in which early pushes move the flyby point
    c_lov: float  # offset of the keyhole from the real nominal along the line of variations [m]
    t_impact: float
    impact_latlon: tuple
    impact_v: float
    keyhole_width: float  # [m] along the line of variations
    approaches: list = field(default_factory=list)  # [{t, dist, v_rel}] of the scenario trajectory
    nominal: NominalBennu = None
    meta: dict = field(default_factory=dict)

    @property
    def q_ca(self):
        return float(np.linalg.norm(self.b0))

    @property
    def bplane(self):
        bp = BPlane.__new__(BPlane)
        bp.s = self.v_rel / np.linalg.norm(self.v_rel)
        bp.xi, bp.zeta = self.basis
        bp.basis = self.basis
        return bp

    @classmethod
    def load(cls, path=SCENARIO_FILE, vi_path=VI_FILE):
        if not path.exists() or not vi_path.exists():
            raise FileNotFoundError(f"{path.name} missing; run `uv run python -m scripts.make_scenario`")
        d = json.loads(path.read_text())
        return cls(t0=d["t0"], t_pre=d["t_pre"], t_ca=d["t_ca"], b0=np.array(d["b0"]), basis=np.array(d["basis"]),
                   v_rel=np.array(d["v_rel"]), lov=np.array(d["lov"]), c_lov=d["c_lov"], t_impact=d["t_impact"],
                   impact_latlon=tuple(d["impact_latlon"]), impact_v=d["impact_v"],
                   keyhole_width=d["keyhole_width"], approaches=d["approaches"], nominal=NominalBennu(path=vi_path),
                   meta=d.get("meta", {}))

    def goal_radius(self, goal_ld):
        return goal_ld * LUNAR_DISTANCE


# ----------------------------------------------------------------------------- construction
def lov_vector(nom, t_ca, basis):
    """Unit B-plane direction of the line of variations and the 2026 state change v1 per metre along it."""
    t0 = nom.t[0]
    M = basis @ (nom.Phi0(t_ca) @ _inv(nom.Phi0(t0)))[:3, :]  # 2 x 6
    U, S, Vt = np.linalg.svd(M)
    u = U[:, 0] * np.sign(U[1, 0])  # +zeta
    return u, Vt[0] / S[0] * np.sign(U[1, 0])


class Track:
    """Piecewise dense solution: state (and, where integrated, the STM) at any time inside its span."""

    def __init__(self):
        self.segs = []  # (t0, t1, dense solution, has_stm)
        self.t_impact = np.nan

    def add(self, sol, has_stm):
        self.segs.append((float(sol.t[0]), float(sol.t[-1]), sol.sol, has_stm))

    @property
    def t_end(self):
        return self.segs[-1][1]

    def _seg(self, t):
        for t0, t1, f, stm in self.segs:
            if t <= t1:
                return f, stm
        return self.segs[-1][2], self.segs[-1][3]

    def state(self, t):
        if np.ndim(t):
            return np.array([self.state(ti) for ti in np.asarray(t, float)])
        f, _ = self._seg(t)
        return f(t)[:6]

    def full(self, t):
        f, stm = self._seg(t)
        y = f(t)
        return y[:6], (y[6:].reshape(6, 6) if stm else None)


NEAR_EARTH = 2e8  # m: inside this the integration uses short steps (a grazing pass lasts minutes)


def integrate_tail(eph, x_pre, Phi_pre, t_pre, t_end, rtol=1e-12, atol=1e-6, stm_until=None):
    """Nonlinear propagation from t_pre to t_end, stopped at Earth's surface. The STM (chained to Phi_pre) is
    integrated from t_pre to stm_until (None: no STM); after it only the state. Passes inside NEAR_EARTH
    are flown with 20 s steps so a grazing impact is not stepped over. Returns (Track, impact time, dyn)."""
    from core.dynamics.asteroid import BennuDynamics

    dyn = BennuDynamics(eph)
    tr = Track()
    t, x = float(t_pre), np.asarray(x_pre, float)
    if stm_until is not None and stm_until > t:
        y0 = np.concatenate([x, Phi_pre.ravel()])
        sol = solve_ivp(dyn.rhs_stm, (t, stm_until), y0, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        tr.add(sol, True)
        t, x = float(sol.t[-1]), sol.y[:6, -1]

    def dist(tt, y):
        return np.linalg.norm(y[:3] - eph.pos("earth", tt))

    def enter(tt, y):
        return dist(tt, y) - NEAR_EARTH

    enter.terminal, enter.direction = True, -1

    def leave(tt, y):
        return dist(tt, y) - NEAR_EARTH

    leave.terminal, leave.direction = True, 1

    def hit(tt, y):
        return dist(tt, y) - RADIUS["earth"]

    hit.terminal, hit.direction = True, -1
    inside = dist(t, x) < NEAR_EARTH
    while t < t_end - 1.0:
        if inside:
            sol = solve_ivp(dyn.rhs, (t, t_end), x, method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                            events=(hit, leave), max_step=20.0)
        else:
            sol = solve_ivp(dyn.rhs, (t, t_end), x, method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                            events=enter)
        if not sol.success:
            raise RuntimeError(sol.message)
        tr.add(sol, False)
        t, x = float(sol.t[-1]), sol.y[:, -1]
        if inside and len(sol.t_events[0]):
            tr.t_impact = float(sol.t_events[0][0])
            break
        crossed = len(sol.t_events[1]) if inside else len(sol.t_events[0])
        if not crossed:
            break  # reached t_end
        inside = not inside  # switch mode at the crossing (never re-test the distance exactly on the boundary)
    return tr, tr.t_impact, dyn


def build_vi_file(eph, nom, c, v1, t_pre, path=VI_FILE, rtol=1e-12):
    """Scenario trajectory + STM: linear offset on the real nominal before t_pre, nonlinear after it."""
    t0, t_end = nom.t[0], nom.t[-1]
    dx0 = c * v1
    keep = nom.t < t_pre
    tt = nom.t[keep]
    Phi = nom.Phi0(tt)
    Phidot = nom._Phi(tt, 1).reshape(-1, 6, 6)
    x = nom.state(tt) + Phi @ dx0
    xdot = nom._x(tt, 1) + Phidot @ dx0
    x_pre = nom.state(t_pre) + nom.Phi0(t_pre) @ dx0
    t_stm = t_pre + PRE_DAYS * DAY + 30 * DAY  # the STM is carried through the keyhole flyby only
    tr, t_imp, dyn = integrate_tail(eph, x_pre, nom.Phi0(t_pre), t_pre, t_end, rtol=rtol, stm_until=t_stm)
    t_stop = t_imp - 60.0 if np.isfinite(t_imp) else tr.t_end
    # nodes: daily, hourly within 10 days of each close approach, every minute in the last hour
    nodes = np.arange(t_pre, t_stop, DAY)
    for a in find_close_approaches(tr.state, eph, t_pre, t_stop, max_dist=0.1 * AU):
        nodes = np.concatenate([nodes, np.arange(a["t"] - 10 * DAY, min(a["t"] + 10 * DAY, t_stop), 3600.0)])
    if np.isfinite(t_imp):
        nodes = np.concatenate([nodes, np.arange(t_stop - 10 * DAY, t_stop, 3600.0),
                                np.arange(t_stop - 3600.0, t_stop, 60.0), [t_stop]])
    nodes = np.unique(np.round(np.concatenate([nodes[(nodes >= t_pre) & (nodes <= t_stop)], [t_stm]]), 3))
    xs, Ps, Pds = np.empty((len(nodes), 6)), np.empty((len(nodes), 6, 6)), np.zeros((len(nodes), 6, 6))
    P_last = None
    for k, t in enumerate(nodes):
        xs[k], P = tr.full(t)
        if P is not None and t <= t_stm:
            Ps[k] = P_last = P
            Pds[k] = np.vstack([P[3:6], dyn.grav.gradient(t, xs[k, :3]) @ P[0:3]])
        else:  # beyond the keyhole flyby the STM is frozen (every push happens before it)
            Ps[k] = P_last
    xds = np.array([dyn.rhs(t, xi) for t, xi in zip(nodes, xs)])
    np.savez(path, t=np.concatenate([tt, nodes]), x=np.vstack([x, xs]), xdot=np.vstack([xdot, xds]),
             Phi=np.vstack([Phi, Ps]).reshape(-1, 36), Phidot=np.vstack([Phidot, Pds]).reshape(-1, 36),
             a2=nom.a2, amrat=nom.amrat, t_pre=t_pre, c=c, t_impact=t_imp)
    return t_imp, tr


def impact_site(eph, nominal, t_imp):
    """Geographic latitude / longitude of the impact point (Earth rotation from GMST; display only)."""
    from core.timeutil import t_to_jd

    r = nominal.state(t_imp)[:3] - eph.pos("earth", t_imp)
    v = nominal.state(t_imp)[3:] - eph.vel("earth", t_imp)
    d = t_to_jd(t_imp) - 2451545.0
    gmst = np.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    lon = np.degrees(np.arctan2(r[1], r[0]) - gmst)
    lat = np.degrees(np.arcsin(r[2] / np.linalg.norm(r)))
    return float(lat), float((lon + 180.0) % 360.0 - 180.0), float(np.linalg.norm(v))
