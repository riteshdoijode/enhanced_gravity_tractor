"""Mission planner for the keyhole scenario: push Bennu's 2135 flyby beyond the Moon's distance.

Scenario (core/scenario.py): Bennu's 2135 flyby goes through the keyhole of a later Earth impact. Goal:
move that flyby to at least goal_ld lunar distances (1 LD = 384,400 km) so Bennu misses the keyhole.

Deflection in the 2135 B-plane (modified target plane, miss vector b, |b| = flyby distance). Pushes map
linearly through Bennu's state-transition matrix:
    b = b0 + sum_i B(t_i) dv_i + integral B(t) a_tow(t) dt,     B(t) = [xi; zeta] Phi_rv(t_ca, t)
B(t) is rank one in practice (early pushes move the flyby only along the line of variations d), so the
shift lam that brings b to the goal circle |b| = b_T is the positive root of |b + lam d| = b_T.

DART (kinetic impactor, Cheng et al. 2023 momentum transfer) and its launch opportunities. Bennu is cheapest
to reach near its Earth close passes; every pass within 0.3 au (passes closer than 300 days merged) is an
opportunity. For each, a porkchop of Lambert arcs from the parking orbit (launch 1.5 yr before to 0.3 yr after
the pass, flight 30-700 days, departure speed v_dep = sqrt(v_inf^2 + 2 mu / r_p) <= the orbit-raising cut-off
v_max, impact >= 2 years before the flyby) gives the shift per kilogram of each arc along the push direction,
    s = d . B(t_arr) (beta / M) U      (head-on: ejecta back along the incoming direction)
and the recommended arc is the one needing the least mass for the remaining shift. Mass recommendations:
nominal beta, and 90 % confidence (beta - 1.28 sigma). The point of contact on the real shape and the orbit-
raising burns are planned for the chosen arc (core/dart.py).

Window capability (the distance plot's markers): what one DART as set (mass, speed limit, beta, sizing) does in
every window, as the plan for that window would fly it (its arc, aimed at that arc's point of contact), classed
by the verdict's own success test: reaches the target / clears the goal only / falls short / no arc.

Gravity tractor (Lu & Love 2005, enhanced with a boulder, Mazanek et al. 2015): launched only after a
successful impact. Earth escape spiral (any thrust) and 0-3 revolution Lambert arcs as finite burns give
the arrival; after the capture it tows with the steered law (Step 7) toward the DART aim point. Its reach
(largest shift it can still add along d) decides whether a DART result is good enough.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import math
import time

import numpy as np

from core.assets import bennu_props
from core.constants import BENNU, DAY, G0, GM, LUNAR_DISTANCE, RADIUS, YEAR
from core.dart import departure_speed, raising_schedule
from core.deflection import M_BENNU, gt_accel, gt_min_hover_distance, gt_required_thrust, ki_delta_v
from core.dynamics.nominal import J6
from core.guidance.departure import park_period, spiral_escape
from core.guidance.kepler import ecliptic_pole_icrf, lambert_batch, lambert_multirev, propagate
from core.timeutil import iso_to_t

POLE = ecliptic_pole_icrf()
TODAY = "2026-10-15"
PASS_MAX_AU = 0.3
PASS_MERGE = 300 * DAY
GROUND_OD_DAYS = 30.0  # orbit determination after an impact before its result is known
PORKCHOP_STEP = 4 * DAY  # launch-date grid of the DART porkchops
# slowest impact the DART planner accepts: the momentum enhancement beta is measured (DART, ~6.1 km/s) and modelled
# (impact simulations and the ejecta scaling laws behind them) for impacts of several km/s; a sub-km/s arc's beta
# would be an extrapolation (and it needs several times the mass)
MIN_IMPACT_SPEED = 3e3
Z90 = 1.2816


@dataclass
class MissionParams:
    goal_ld: float = 1.0  # required 2135 flyby distance [lunar distances]
    dart_target_km: float = 0.0  # DART aim: shift of the 2135 flyby [km]; 0 = flyby at 1.25 x goal
    earliest_launch: str = TODAY
    parking_alt_km: float = 400.0
    tractor_mass_kg: float = 10_000.0
    tractor_thrust_n: float = 20.0
    tractor_isp_s: float = 3000.0
    tractor_objective: str = "fastest"  # or "efficient"
    boulder_mass_kg: float = 20_000.0
    hover_factor: float = 2.0  # hover distance / Bennu mean radius
    plume_half_angle_deg: float = 20.0
    capture_days: float = 90.0
    tow_years: float = 10.0
    opportunity: int = -1  # index into the opportunity list (-1: the recommended one)
    use_recommended_mass: bool = True
    dart_sizing: str = "nominal"  # "nominal" (planned beta) or "p90" (90 % confidence over beta)
    impactor_mass_kg: float = 580.0  # used when use_recommended_mass is off
    dart_max_speed_kms: float = 12.5  # orbit-raising cut-off: perigee speed at departure
    dart_tolerance: float = 0.1  # a DART succeeds if it delivers at least (1 - tolerance) of its target shift
    refined_mass_kg: float = 0.0  # full-model recommended mass for the selected opportunity (core/refine.py), 0 = none
    impactor_thrust_n: float = 50_000.0
    impactor_isp_s: float = 320.0
    beta: float = 3.6
    beta_sigma: float = 0.7
    ejecta_model: str = "normal"  # live impact: ejecta momentum along the surface normal, or "head-on"
    unlimited_fuel: bool = True


TRACTOR_FIELDS = frozenset({"tractor_mass_kg", "tractor_thrust_n", "tractor_isp_s", "tractor_objective",
                            "boulder_mass_kg", "hover_factor", "plume_half_angle_deg", "capture_days", "tow_years"})


def physical_params(p):
    """The parameters that change the physics: the full-model mass travels in the DART plan, and with the
    recommended mass on, the mass slider only displays it."""
    if p is None:
        return None
    p = replace(p, refined_mass_kg=0.0)
    return replace(p, impactor_mass_kg=0.0) if p.use_recommended_mass else p


def dart_params(p):
    """The physical parameters a DART plan depends on: all but the gravity tractor's."""
    q = physical_params(p)
    return None if q is None else {k: v for k, v in asdict(q).items() if k not in TRACTOR_FIELDS}


@dataclass
class Check:
    level: str  # "ok" | "warn" | "fail"
    text: str


@dataclass
class Opportunity:
    index: int
    t_pass: float
    pass_dist: float
    feasible: bool = False
    t_launch: float = np.nan  # departure (injection burn)
    t_arrive: float = np.nan
    t_raise_start: float = np.nan
    impact_speed: float = np.nan
    v_inf: float = np.nan
    v_dep: float = np.nan
    gain_per_kg: float = np.nan  # shift along d per kg of impactor [m/kg]
    mass_nominal: float = np.nan  # for the remaining shift at the planned beta
    mass_p90: float = np.nan
    trade: list = field(default_factory=list)  # [(v_max, impact speed, mass nominal)]
    remaining: float = np.nan  # shift still needed along d [m]
    note: str = ""
    capability: dict = None  # what the DART as set does in this window (Planner.window_capability)


@dataclass
class TractorPlan:
    feasible: bool = False
    t_launch: float = np.nan
    t_handover: float = np.nan
    t_arrive: float = np.nan
    t_arrive_flight: float = np.nan  # optimized low-thrust flight plan (if computed)
    escape_days: float = np.nan
    cruise_days: float = np.nan
    dv_escape: float = np.nan
    dv_depart: float = np.nan
    dv_arrive: float = np.nan
    dv_total: float = np.nan
    propellant_kg: float = np.nan
    accel: float = np.nan
    revolutions: int = 0
    hover_distance_m: float = np.nan
    hover_thrust_n: float = np.nan
    hover_ok: bool = False
    gt_accel: float = 0.0
    tow_start: float = np.nan
    tow_end: float = np.nan
    gt_dv_bennu: float = 0.0
    reach: float = 0.0  # largest shift the tow can add along d [m]
    tow_direction: np.ndarray = field(default=None, repr=False)
    path: np.ndarray = field(default=None, repr=False)  # barycentric display path (N, 3)
    path_t: np.ndarray = field(default=None, repr=False)
    flight_plan: dict = field(default=None, repr=False)  # optimized low-thrust profile (tractor_flight_plan), if any
    note: str = ""


@dataclass
class ImpactorPlan:
    feasible: bool = False
    number: int = 1
    opportunity: int = -1
    t_launch: float = np.nan  # departure (injection burn at perigee)
    t_arrive: float = np.nan
    tof_days: float = np.nan
    c3: float = np.nan
    v_inf: float = np.nan
    v_dep: float = np.nan
    dv_depart: float = np.nan
    impact_speed: float = np.nan
    mass: float = np.nan
    mass_linear: float = np.nan  # recommended mass from the linear (STM) estimate
    mass_source: str = "linear"  # "linear" or "full model" (refined with nonlinear propagations)
    mass_nominal: float = np.nan
    mass_p90: float = np.nan
    U: np.ndarray = field(default=None, repr=False)
    dv_bennu: np.ndarray = field(default=None, repr=False)
    db: np.ndarray = field(default=None, repr=False)  # planned B-plane shift (head-on, planned beta)
    raising: object = field(default=None, repr=False)  # core.dart.RaisingPlan
    propellant_kg: float = np.nan
    contact: dict = field(default=None, repr=False)  # recommended point of contact (body frame)
    porkchop: dict = field(default=None, repr=False)
    path: np.ndarray = field(default=None, repr=False)
    path_t: np.ndarray = field(default=None, repr=False)
    r_depart: np.ndarray = field(default=None, repr=False)
    v_depart: np.ndarray = field(default=None, repr=False)
    v_inf_vec: np.ndarray = field(default=None, repr=False)


@dataclass
class PlanResult:
    params: MissionParams
    opportunities: list
    impactor: ImpactorPlan
    tractor: TractorPlan
    t_encounter: float
    b0: np.ndarray
    b_aim: np.ndarray
    b_goal: float
    direction: np.ndarray
    required_shift: float
    target_shift: float
    db_ki: np.ndarray
    db_gt: np.ndarray
    db_total: np.ndarray
    b_final: np.ndarray
    lead_time_years: float
    success: bool
    p_success: float
    beta_samples: np.ndarray = field(repr=False)
    b_samples: np.ndarray = field(repr=False)
    checks: list = field(default_factory=list)
    compute_ms: float = 0.0
    capability: dict = None  # summary of Planner.window_capability for the DART as set (per window: opp.capability)

    @property
    def q_final(self):
        return float(np.linalg.norm(self.b_final))


@lru_cache(maxsize=256)
def _escape_cached(a_key, r_park):
    return spiral_escape(a_key, r_park)


def shift_to_circle(b, d, radius):
    """Smallest lam >= 0 with |b + lam d| >= radius (d unit)."""
    if np.linalg.norm(b) >= radius:
        return 0.0
    bd = float(b @ d)
    return -bd + float(np.sqrt(bd**2 + radius**2 - b @ b))


def _p_reach(s1, k1, need, beta, sigma):
    """Chance that a shift s1 + (x - 1) k1 (linear in beta, x = max(1, beta'), beta' ~ N(beta, sigma)) is >= need."""
    if k1 == 0:
        return float(s1 >= need)
    x = 1.0 + (need - s1) / k1  # the beta at which the shift equals need
    if k1 > 0 and x <= 1.0:
        return 1.0
    if k1 < 0 and x < 1.0:
        return 0.0
    if sigma <= 0:
        return float(beta >= x if k1 > 0 else beta <= x)
    cdf = 0.5 * (1.0 + math.erf((x - beta) / (sigma * math.sqrt(2.0))))
    return 1.0 - cdf if k1 > 0 else cdf


class Planner:
    def __init__(self, eph, nominal, scenario):
        self.eph, self.nom, self.sc = eph, nominal, scenario
        self.t_ca = scenario.t_ca
        self.bplane = scenario.bplane
        self.b0 = np.asarray(scenario.b0, float)
        self.v_rel = scenario.v_rel
        self.v_inf = float(np.linalg.norm(scenario.v_rel))
        self.Phi_ca = nominal.Phi0(self.t_ca)
        props = bennu_props()
        self.R_bound = 0.5 * max(props["dimensions_along_principal_axes_m"])  # plume must clear this
        self.lov = np.asarray(scenario.lov, float) / np.linalg.norm(scenario.lov)
        self.passes = self._merge_passes(scenario.approaches)
        self._ray = None
        self._opp_cache = {}
        self._contact_geom = {}  # contact candidates per arc and Bennu attitude (core/dart.py)

    # ------------------------------------------------------------------ geometry
    def _merge_passes(self, approaches):
        close = [a for a in approaches if a["dist"] <= PASS_MAX_AU * 1.495978707e11 and a["t"] < self.t_ca - 2 * YEAR]
        merged = []
        for a in close:
            if merged and a["t"] - merged[-1][-1]["t"] < PASS_MERGE:
                merged[-1].append(a)
            else:
                merged.append([a])
        return [min(g, key=lambda a: a["dist"]) for g in merged]

    def goal_radius(self, p):
        return p.goal_ld * LUNAR_DISTANCE

    def push_direction(self, p, b=None):
        """Direction along the line of variations that reaches the goal circle with the smaller shift."""
        b = self.b0 if b is None else b
        R = self.goal_radius(p)
        up, down = shift_to_circle(b, self.lov, R), shift_to_circle(b, -self.lov, R)
        return (self.lov, up) if up <= down else (-self.lov, down)

    def target_shift(self, p):
        """DART aim shift from b0 (user value, or the shift that puts the flyby at 1.25 x the goal)."""
        d, required = self.push_direction(p)
        if p.dart_target_km > 0:
            return max(p.dart_target_km * 1e3, required)
        return shift_to_circle(self.b0, d, 1.25 * self.goal_radius(p))

    def aim_point(self, p):
        d, _ = self.push_direction(p)
        return self.b0 + self.target_shift(p) * d

    def aim_on_line(self, p, b):
        """The aim as the tractor can reach it from b: on the line b + lam d (pushes only move the flyby along the
        line of variations) at the aim distance |aim_point|; the root with the smaller |lam|. (Outcome and the live
        trim hold an overshoot instead of pulling it back.)"""
        d, _ = self.push_direction(p)
        b = np.asarray(b, float)
        R = float(np.linalg.norm(self.aim_point(p)))
        bd = float(b @ d)
        disc = bd**2 - float(b @ b) + R**2
        if disc < 0:  # the line never reaches that distance: its closest point
            return b - bd * d
        roots = (-bd + np.sqrt(disc), -bd - np.sqrt(disc))
        lam = min(roots, key=abs)
        return b + lam * d

    def B(self, t):
        """B-plane sensitivity db = B dv for pushes at times t: (N, 2, 3)."""
        return np.einsum("ij,njk->nik", self.bplane.basis, self.Phi_rv_to_encounter(t))

    def Phi_rv_to_encounter(self, t):
        """Phi_rv(t_ca, t) for an array of times t, shape (N, 3, 3)."""
        P = self.nom.Phi0(np.atleast_1d(t))
        inv = -J6 @ np.transpose(P, (0, 2, 1)) @ J6
        return (self.Phi_ca @ inv)[:, :3, 3:]

    def bennu_helio(self, t):
        t = np.atleast_1d(t)
        x, s = self.nom.state(t), self.eph.state("sun", t)
        return x[:, :3] - s[:, :3], x[:, 3:] - s[:, 3:]

    def earth_helio(self, t):
        t = np.atleast_1d(t)
        e, s = self.eph.state("earth", t), self.eph.state("sun", t)
        return e[:, :3] - s[:, :3], e[:, 3:] - s[:, 3:]

    def _arc_path(self, r0, v0, t0, tof, n=160):
        """Barycentric display samples (times, positions) of a heliocentric Kepler arc."""
        tt = t0 + np.linspace(0, tof, n)
        r, _ = propagate(np.broadcast_to(r0, (n, 3)), np.broadcast_to(v0, (n, 3)), tt - t0)
        return tt, r + self.eph.pos("sun", tt)

    def ray(self):
        if self._ray is None:
            from core.proximity.sensors import ShapeRaycaster

            self._ray = ShapeRaycaster()
        return self._ray

    def contact_point(self, t_arrive, U, beta, g):
        """Recommended point of contact for an arc arriving at t_arrive with relative velocity U (core/dart.py).
        The ray-cast candidates are kept per arc and Bennu attitude (a spin change in a live run gives new ones),
        so another beta or push direction only re-scores them."""
        from core.dart import best_contact_point, contact_candidates
        from core.dynamics.asteroid import bennu_attitude

        R = bennu_attitude(t_arrive)
        U = np.asarray(U, float)
        key = (round(float(t_arrive)), U.round(6).tobytes(), R.round(12).tobytes())
        cand = self._contact_geom.get(key)
        if cand is None:
            if len(self._contact_geom) > 2000:
                self._contact_geom.clear()
            cand = self._contact_geom[key] = contact_candidates(self.ray(), R, U)
        return best_contact_point(self.ray(), R, U, beta, 1.0, g, cand=cand)

    def _aim(self, t_arrive, U, beta, g):
        """(contact, efficiency, ejecta direction E_hat) as a plan flies it: the recommended point when it is found
        and gives more than 0.2x a head-on hit, else head-on (efficiency 1, E_hat None)."""
        try:
            c = self.contact_point(t_arrive, U, beta, g)
        except Exception as exc:  # the plan stays usable without the recommendation
            return {"error": str(exc)}, 1.0, None
        if c and np.isfinite(c.get("efficiency", np.nan)) and c["efficiency"] > 0.2:
            from core.dynamics.asteroid import bennu_attitude

            return c, float(c["efficiency"]), bennu_attitude(t_arrive) @ np.asarray(c["normal_b"], float)
        return c, 1.0, None

    # ------------------------------------------------------------------ DART opportunities
    def _porkchop(self, p, t_lo, t_hi, t_after):
        """Lambert arcs launched in [t_lo, t_hi], not before t_after: departure speed, impact velocity, B-plane shift
        per kg (beta = 1). The launch grid is anchored at t_lo (a later t_after only drops its first dates: an
        earliest launch that does not bind leaves the plan as it was)."""
        r_park = RADIUS["earth"] + p.parking_alt_km * 1e3
        launch = np.arange(t_lo, t_hi, PORKCHOP_STEP)
        launch = launch[launch >= t_after - 1.0]
        tofs = np.arange(30.0, 701.0, 5.0) * DAY
        if len(launch) == 0:
            return None
        L, T = np.meshgrid(launch, tofs, indexing="ij")
        rE, vE = self.earth_helio(L.ravel())
        arrive = (L + T).ravel()
        rB, vB = self.bennu_helio(arrive)
        v1, v2 = lambert_batch(rE, rB, T.ravel(), prograde_axis=POLE)
        vinf = np.linalg.norm(v1 - vE, axis=1)
        v_dep = departure_speed(vinf, r_park)
        U = v2 - vB
        speed = np.linalg.norm(U, axis=1)
        cosang = np.sum(rE * rB, 1) / (np.linalg.norm(rE, axis=1) * np.linalg.norm(rB, axis=1))
        valid = np.isfinite(v_dep) & (cosang < np.cos(np.radians(5))) & (arrive < self.t_ca - 2 * YEAR)
        # B-plane shift per kg (2-D), head-on momentum transfer, evaluated on whole arrival days
        s_vec = np.full((len(arrive), 2), np.nan)
        if valid.any():
            days = np.round(arrive[valid] / DAY) * DAY
            ud, inv = np.unique(days, return_inverse=True)
            B = self.B(ud)[inv]
            s_vec[valid] = np.einsum("nij,nj->ni", B, U[valid]) / M_BENNU  # per kg and per unit beta
        return {"launch": launch, "tof": tofs, "L": L.ravel(), "T": T.ravel(), "arrive": arrive, "v_dep": v_dep,
                "vinf": vinf, "U": U, "speed": speed, "valid": valid, "s_vec": s_vec, "rE": rE, "vE": vE, "v1": v1,
                "shape": L.shape}

    def _pass_window(self, a):
        return a["t"] - 1.5 * YEAR, a["t"] + 0.3 * YEAR

    def opportunities(self, p, b_now=None, t_after=None, number=1):
        """One recommendation per close pass after t_after for the shift still needed toward the aim point."""
        b_now = self.b0 if b_now is None else np.asarray(b_now, float)
        t_after = iso_to_t(p.earliest_launch) if t_after is None else float(t_after)
        d, _ = self.push_direction(p)
        remaining = max(0.0, float((self.aim_point(p) - b_now) @ d))
        v_max = p.dart_max_speed_kms * 1e3
        beta_lo = max(1.0, p.beta - Z90 * p.beta_sigma)
        out = []
        for k, a in enumerate(self.passes):
            t_lo, t_hi = self._pass_window(a)
            if t_hi < t_after + 40 * DAY:
                continue
            opp = Opportunity(index=k, t_pass=a["t"], pass_dist=a["dist"], remaining=remaining)
            t_first = max(t_lo, t_after + 40 * DAY)  # the first launch: leave time for the orbit raising
            key = (k, p.parking_alt_km, int(np.ceil((t_first - t_lo) / PORKCHOP_STEP - 1e-9)))  # (grid dates dropped)
            pc = self._opp_cache.get(key)
            if pc is None:
                pc = self._porkchop(p, t_lo, t_hi, t_first)
                if len(self._opp_cache) > 400:
                    self._opp_cache.clear()
                self._opp_cache[key] = pc
            if pc is None:
                continue
            s = np.where(pc["valid"], p.beta * (pc["s_vec"] @ d), np.nan)
            fast = pc["speed"] >= MIN_IMPACT_SPEED  # (slower impacts: outside the regime where beta is measured)
            ok = np.isfinite(s) & (s > 0) & (pc["v_dep"] <= v_max) & fast
            for vm in (11.5, 12.5, 14.0, 16.0):
                okv = np.isfinite(s) & (s > 0) & (pc["v_dep"] <= vm * 1e3) & fast
                if okv.any():
                    j = int(np.nanargmax(np.where(okv, s, -np.inf)))
                    opp.trade.append((vm, float(pc["speed"][j]), remaining / s[j] if remaining > 0 else 0.0))
            if not ok.any():
                slow = np.isfinite(s) & (s > 0) & (pc["v_dep"] <= v_max)
                opp.note = (f"only slow arcs (impact below {MIN_IMPACT_SPEED / 1e3:g} km/s) within the departure-speed limit"
                            if slow.any() else "no arc within the departure-speed limit")
                out.append(opp)
                continue
            j = int(np.nanargmax(np.where(ok, s, -np.inf)))
            opp.feasible = True
            opp.t_launch, opp.t_arrive = float(pc["L"][j]), float(pc["arrive"][j])
            opp.impact_speed, opp.v_inf, opp.v_dep = float(pc["speed"][j]), float(pc["vinf"][j]), float(pc["v_dep"][j])
            opp.gain_per_kg = float(s[j])
            opp.mass_nominal = remaining / s[j]
            opp.mass_p90 = remaining / (s[j] * beta_lo / p.beta)
            opp._arc, opp._pc = j, pc
            out.append(opp)
        return out

    def recommended(self, opps):
        """The recommended opportunity: the earliest one whose nominal mass is within 3x of the cheapest."""
        feas = [o for o in opps if o.feasible]
        if not feas:
            return None
        cheapest = min(o.mass_nominal for o in feas)
        return next(o for o in feas if o.mass_nominal <= 3.0 * cheapest + 1.0)

    def window_capability(self, p, opps, mass, b_now=None, contact=True):
        """What one DART of `mass`, with p's speed limit, beta, beta uncertainty and sizing, delivers in each window
        of `opps` (from opportunities(), porkchops attached) as the plan for that window flies it: the window's arc
        (the largest head-on shift within the speed limit) aimed at that arc's point of contact (contact=False:
        head-on). Linear (STM) estimates from the 2135 point b_now (the deflection so far; default b0). The shift is
        taken at the planned beta, or at the 90 %-confidence beta (beta - 1.28 sigma) when the sizing is "p90".
        Sets opp.capability:
            status        "target": reaches the DART target (the verdict's own test, dart_success);
                          "goal": the 2135 flyby clears the goal distance but falls short of the target;
                          "short": falls short of the goal; "none": no arc within the speed limit
            shift, fraction (of the target, earlier DARTs included), q (predicted 2135 flyby distance)
            mass_target   mass this window needs for the aim point (the recommended-mass sizing);
            mass_success  ... to count as a success; mass_goal: ... to clear the goal
            p_target      chance over beta ~ N(beta, sigma) of reaching the target (aim point chosen for beta)
            efficiency    point of contact vs a head-on hit; db_per_kg: 2-D B-plane shift per kg
        Returns a summary of the DART as set and the thresholds."""
        b_now = self.b0 if b_now is None else np.asarray(b_now, float)
        d, _ = self.push_direction(p)
        R = self.goal_radius(p)
        target = self.target_shift(p)
        achieved = float((b_now - self.b0) @ d)
        need = max(0.0, (1.0 - p.dart_tolerance) * target - achieved)  # this DART's part of a success
        remaining = max(0.0, float((self.aim_point(p) - b_now) @ d))
        beta_lo = max(1.0, p.beta - Z90 * p.beta_sigma)
        beta_eval = beta_lo if p.dart_sizing == "p90" else p.beta
        for o in opps:
            if not o.feasible:
                o.capability = {"status": "none"}
                continue
            if getattr(o, "_pc", None) is None:  # porkchop stripped: nothing to evaluate
                continue
            U = o._pc["U"][o._arc]
            B = self.B([o.t_arrive])[0]
            eff, E_hat = 1.0, None
            if contact:
                _, eff, E_hat = self._aim(o.t_arrive, U, p.beta, B.T @ d)
            db1 = B @ ki_delta_v(1.0, U, beta_eval, E_hat=E_hat)
            spk = float(db1 @ d)
            b_after = b_now + mass * db1
            q = float(np.linalg.norm(b_after))
            status = "target" if self.dart_success(p, b_after) else ("goal" if q >= R else "short")
            s1 = float(d @ B @ ki_delta_v(mass, U, 1.0, E_hat=E_hat))  # this DART's shift, linear in beta
            k1 = float(d @ B @ ki_delta_v(mass, U, 2.0, E_hat=E_hat)) - s1
            n1 = float(np.linalg.norm(db1))
            if np.linalg.norm(b_now) >= R:
                m_goal = 0.0
            else:
                m_goal = shift_to_circle(b_now, db1 / n1, R) / n1 if n1 > 0 else np.inf
            o.capability = {"status": status, "shift": mass * spk, "fraction": (achieved + mass * spk) / max(target, 1.0),
                            "q": q, "efficiency": eff, "db_per_kg": db1,
                            "mass_target": remaining / spk if spk > 0 else np.inf,
                            "mass_success": need / spk if spk > 0 else np.inf, "mass_goal": m_goal,
                            "p_target": _p_reach(s1, k1, need, p.beta, p.beta_sigma)}
        return {"mass": float(mass), "v_max": p.dart_max_speed_kms, "beta": p.beta, "beta_sigma": p.beta_sigma,
                "beta_eval": beta_eval, "sizing": p.dart_sizing, "goal": R, "goal_ld": p.goal_ld, "target": target,
                "tolerance": p.dart_tolerance, "achieved": achieved, "need": need, "remaining": remaining,
                "q_now": float(np.linalg.norm(b_now)), "contact": contact}

    def plan_dart(self, p, opp, b_now=None, number=1, contact=True):
        """Full plan for one opportunity: arc, mass, orbit raising, point of contact, planned shift."""
        ip = ImpactorPlan(number=number, opportunity=opp.index)
        ip.mass_nominal, ip.mass_p90 = opp.mass_nominal, opp.mass_p90
        if not opp.feasible:
            if opp.note and opp.note.startswith("only slow arcs"):
                return ip, Check("fail", f"DART: near the {self._year(opp.t_pass)} pass {opp.note}")
            return ip, Check("fail", f"DART: no arc near the {self._year(opp.t_pass)} pass within "
                                     f"{p.dart_max_speed_kms:g} km/s departure speed")
        t_after = iso_to_t(p.earliest_launch)
        pc, j = opp._pc, opp._arc
        d, _ = self.push_direction(p)
        r_park = RADIUS["earth"] + p.parking_alt_km * 1e3
        ip.feasible = True
        ip.t_launch, ip.t_arrive = opp.t_launch, opp.t_arrive
        ip.tof_days = float(pc["T"][j] / DAY)
        ip.v_inf, ip.v_dep = opp.v_inf, opp.v_dep
        ip.c3 = opp.v_inf**2
        ip.dv_depart = ip.v_dep - np.sqrt(GM["earth"] / r_park)
        ip.U = pc["U"][j].copy()
        ip.impact_speed = opp.impact_speed
        B_imp = self.B([ip.t_arrive])[0]
        # point of contact (candidates cached per arc): ejecta along its crater normal change the push direction;
        # the mass is sized with the shift that point gives (efficiency vs a head-on hit)
        eff, E_hat = 1.0, None
        if contact:
            ip.contact, eff, E_hat = self._aim(ip.t_arrive, ip.U, p.beta, B_imp.T @ d)
        beta_lo = max(1.0, p.beta - Z90 * p.beta_sigma)
        ip.mass_linear = opp.mass_nominal / eff
        ip.mass_nominal = p.refined_mass_kg if p.refined_mass_kg > 0 else ip.mass_linear
        ip.mass_source = "full model" if p.refined_mass_kg > 0 else "linear"
        ip.mass_p90 = ip.mass_nominal * p.beta / beta_lo
        if p.use_recommended_mass:
            ip.mass = float(np.clip(ip.mass_p90 if p.dart_sizing == "p90" else ip.mass_nominal, 20.0, 5e7))
        else:
            ip.mass = p.impactor_mass_kg
        ip.dv_bennu = ki_delta_v(ip.mass, ip.U, p.beta, E_hat=E_hat)
        ip.db = B_imp @ ip.dv_bennu
        ip.r_depart, ip.v_depart = pc["rE"][j].copy(), pc["v1"][j].copy()
        ip.v_inf_vec = (pc["v1"][j] - pc["vE"][j]).copy()
        ip.raising = raising_schedule(ip.mass, ip.v_inf_vec, r_park, p.impactor_thrust_n, p.impactor_isp_s,
                                      ip.t_launch, unlimited=p.unlimited_fuel)
        ip.propellant_kg = ip.raising.propellant_kg
        ip.path_t, ip.path = self._arc_path(ip.r_depart, ip.v_depart, ip.t_launch, pc["T"][j])
        n_l, n_t = pc["shape"]
        ip.porkchop = {"launch": pc["launch"], "tof": pc["tof"], "v_dep": pc["v_dep"].reshape(n_l, n_t),
                       "speed": pc["speed"].reshape(n_l, n_t),
                       "shift": np.where(pc["valid"] & (pc["v_dep"] <= p.dart_max_speed_kms * 1e3)
                                         & (pc["speed"] >= MIN_IMPACT_SPEED),
                                         np.nan_to_num(pc["s_vec"] @ d) * p.beta * ip.mass, 0.0).clip(0).reshape(n_l, n_t)}
        if ip.raising.t_start < t_after:
            return ip, Check("warn", "DART: the orbit raising would have to start before today")
        return ip, None

    @staticmethod
    def _year(t):
        from core.timeutil import t_to_utc_str

        return t_to_utc_str(t)[:4]

    # ------------------------------------------------------------------ tractor
    def plan_tractor(self, p, t_launch, b_now=None):
        """Tractor launched at t_launch: arrival (escape spiral + Lambert estimate), hover, tow window, reach."""
        tp = TractorPlan()
        b_now = self.b0 if b_now is None else np.asarray(b_now, float)
        tp.t_launch = float(t_launch)
        r_park = RADIUS["earth"] + p.parking_alt_km * 1e3
        tp.accel = a = p.tractor_thrust_n / p.tractor_mass_kg
        esc = _escape_cached(float(a), float(r_park))
        if not esc["feasible"]:
            return tp, Check("fail", f"Tractor thrust too low: escaping Earth alone takes >{esc['t'] / YEAR:.0f} years")
        tp.escape_days, tp.dv_escape = esc["t"] / DAY, esc["dv"]
        tp.t_handover = th = tp.t_launch + esc["t"]
        rE, vE = self.earth_helio(th)
        r_h, v_h = rE[0] + esc["r_geo"], vE[0] + esc["v_geo"]
        tof = np.geomspace(5 * DAY, 12 * YEAR, 600)
        tof = tof[th + tof < self.t_ca - YEAR]
        if len(tof) == 0:
            return tp, Check("fail", "Tractor leaves Earth too late to reach Bennu before the 2135 flyby")
        rT, vT = self.bennu_helio(th + tof)
        V1, V2 = lambert_multirev(np.broadcast_to(r_h, rT.shape), rT, tof, n_max=3, prograde_axis=POLE)
        dv1 = np.linalg.norm(V1 - v_h, axis=2)
        dv2 = np.linalg.norm(vT[None] - V2, axis=2)
        total = np.where(np.isfinite(dv1 + dv2), dv1 + dv2, np.inf)
        j = np.argmin(total, axis=0)
        cols = np.arange(len(tof))
        dv1, dv2, v1 = dv1[j, cols], dv2[j, cols], V1[j, cols]
        cosang = np.sum(r_h * rT, 1) / (np.linalg.norm(r_h) * np.linalg.norm(rT, axis=1))
        burn = (dv1 + dv2) / a
        ok = np.isfinite(dv1) & ((cosang < np.cos(np.radians(5))) | (j > 0)) & (burn <= 0.8 * tof)
        if not ok.any():
            return tp, Check("fail", f"Tractor ({p.tractor_thrust_n:g} N) cannot rendezvous before 2134 - more thrust needed")
        if p.tractor_objective == "efficient":
            eff = ok & (tof <= 6 * YEAR)
            k = int(np.argmin(np.where(eff if eff.any() else ok, dv1 + dv2, np.inf)))
        else:
            k = int(np.argmax(ok))
        tp.feasible = True
        tp.revolutions = int((j[k] + 1) // 2)
        tp.cruise_days = tof[k] / DAY
        tp.t_arrive = th + tof[k]
        tp.dv_depart, tp.dv_arrive = float(dv1[k]), float(dv2[k])
        tp.dv_total = tp.dv_escape + tp.dv_depart + tp.dv_arrive
        c = p.tractor_isp_s * G0
        tp.propellant_kg = (p.tractor_mass_kg * tp.dv_total / c if p.unlimited_fuel
                            else p.tractor_mass_kg * (np.exp(tp.dv_total / c) - 1.0))
        geo = esc["path_geo"][:: max(1, len(esc["path_geo"]) // 60)]
        t_esc = np.linspace(tp.t_launch, th, len(geo))  # display only: samples spread over the spiral
        t_arc, arc = self._arc_path(r_h, v1[k], th, tof[k])
        tp.path_t = np.concatenate([t_esc, t_arc])
        tp.path = np.vstack([geo + self.eph.pos("earth", t_esc), arc])
        self._gravity_tractor(p, tp, b_now)
        return tp, None

    def plan_tractor_launch(self, p, t, b_now, flight_plan=None):
        """The tractor launched at t: quick plan, optimized flight plan, tow window and reach (what the live run
        flies after a successful DART). The optimized profile travels in tp.flight_plan: the live run flies it from
        the hand-over (TractorFlight warm start), so the launch flies the plan the user saw."""
        tp, msg = self.plan_tractor(p, t, b_now)
        fp = flight_plan if flight_plan is not None else (self.tractor_flight_plan(p, t) if tp.feasible else None)
        if fp is not None and fp.get("feasible"):
            tp.t_arrive_flight = fp["t_arrive"]
            tp.flight_plan = fp
            self._gravity_tractor(p, tp, b_now, t_arrive=fp["t_arrive"])
        return tp, msg

    def tractor_launch_window(self, p, t_from, b_now=None, step=30 * DAY):
        """Launch dates for the tractor from t_from: which still reach Bennu in time for a tow that closes the gap
        to the aim point (the blue launch-window band of the distance plot, and "launch window open until").

        Each launch date goes through plan_tractor (escape + Lambert estimate, hover sizing, steered tow until
        t_ca - 0.5 yr): a date is in the window when the plan is feasible and the tow's reach along the aim
        direction covers the shift still needed from b_now (any reach, when the aim is already met). The sweep is
        yearly, then monthly (`step`) across the year in which the window closes (the reach collapses after the
        2060 flyby: a tow that starts later moves the 2135 flyby thousands of km instead of millions).
        Returns {"t": launch dates, "t_arrive", "reach", "ok", "t_last": last launch in the window (nan if none),
                 "gap": the shift still needed [m]}."""
        b_now = self.b0 if b_now is None else np.asarray(b_now, float)
        w = self.aim_dir(p, b_now)
        aim = self.aim_on_line(p, b_now)
        gap = float((aim - b_now) @ w) if np.linalg.norm(aim) >= np.linalg.norm(b_now) else 0.0  # overshoot: held
        gap = max(0.0, gap)
        t_end = self.t_ca - YEAR

        def point(t):
            tp, _ = self.plan_tractor(p, t, b_now)
            ok = bool(tp.feasible and (tp.reach >= gap if gap > 0 else True))
            return (float(t), tp.t_arrive if tp.feasible else np.nan, tp.reach if tp.feasible else 0.0, ok)

        rows = [point(t) for t in np.arange(t_from, t_end, YEAR)]
        if rows and rows[-1][0] < t_end - step:
            rows.append(point(t_end - 1.0))
        ok_idx = [i for i, r in enumerate(rows) if r[3]]
        if ok_idx:  # refine the year after the last launch in the window (where it closes)
            i = ok_idx[-1]
            t_a = rows[i][0]
            t_b = rows[i + 1][0] if i + 1 < len(rows) else t_end
            rows += [point(t) for t in np.arange(t_a + step, t_b, step)]
        rows.sort(key=lambda r: r[0])
        out = {k: np.array([r[j] for r in rows]) for j, k in enumerate(("t", "t_arrive", "reach", "ok"))}
        out["ok"] = out["ok"].astype(bool)
        out["t_last"] = float(out["t"][out["ok"]].max()) if out["ok"].any() else np.nan
        out["gap"] = gap
        return out

    def _gravity_tractor(self, p, tp, b_now, t_arrive=None):
        m_T = p.tractor_mass_kg + p.boulder_mass_kg
        d = max(p.hover_factor * BENNU["radius"], self.R_bound + 30.0)
        T_req = gt_required_thrust(m_T, d, R=self.R_bound, phi_deg=p.plume_half_angle_deg)
        tp.hover_ok = T_req <= p.tractor_thrust_n
        if not tp.hover_ok:
            d_min = gt_min_hover_distance(p.tractor_thrust_n, m_T, R=self.R_bound, phi_deg=p.plume_half_angle_deg)
            d = d_min if d_min is not None else np.inf
            T_req = gt_required_thrust(m_T, d, R=self.R_bound, phi_deg=p.plume_half_angle_deg) if np.isfinite(d) else np.inf
        tp.hover_distance_m, tp.hover_thrust_n = d, T_req
        tp.gt_accel = gt_accel(m_T, d) if np.isfinite(d) else 0.0
        arrive = tp.t_arrive if t_arrive is None else t_arrive
        tp.tow_start = arrive + p.capture_days * DAY
        tp.tow_end = min(tp.tow_start + p.tow_years * YEAR, self.t_ca - 0.5 * YEAR)
        tp.gt_dv_bennu = tp.gt_accel * max(0.0, tp.tow_end - tp.tow_start)
        tp.reach, tp.tow_direction = self.tow_reach(tp.gt_accel, tp.tow_start, tp.tow_end, self.aim_dir(p, b_now))

    def aim_dir(self, p, b_now):
        """Unit direction from the current point toward the aim point (falls back to the push direction)."""
        g = self.aim_point(p) - np.asarray(b_now, float)
        n = np.linalg.norm(g)
        return g / n if n > 1.0 else self.push_direction(p)[0]

    def tow_reach(self, a, t0, t1, w):
        """Largest shift along w the steered tow (acceleration a) adds over [t0, t1], and its first direction."""
        if not (np.isfinite(t0) and a > 0 and t1 > t0):
            return 0.0, None
        n = int(min(600, max(20, (t1 - t0) / (7 * DAY))))
        ts = np.linspace(t0, t1, n)
        B = self.B(ts)
        g = np.einsum("nij,i->nj", B, w)
        e = g / np.linalg.norm(g, axis=1, keepdims=True)
        db = a * np.trapezoid(np.einsum("nij,nj->ni", B, e), ts, axis=0)
        return float(db @ w), e[0]

    def tractor_flight_plan(self, p, t_launch):
        """Optimized low-thrust flight plan (what the live run flies) for a launch at t_launch."""
        from core.guidance.lowthrust import LowThrustGuidance

        a = p.tractor_thrust_n / p.tractor_mass_kg
        esc = _escape_cached(float(a), float(RADIUS["earth"] + p.parking_alt_km * 1e3))
        if not esc["feasible"]:
            return {"feasible": False}
        th = t_launch + esc["t"]
        rE, vE = self.earth_helio(th)
        x = np.concatenate([rE[0] + esc["r_geo"], vE[0] + esc["v_geo"]])
        g = LowThrustGuidance(lambda t: self.nom.state(np.atleast_1d(t)) - self.eph.state("sun", np.atleast_1d(t)), a)
        tf = g.plan(th, x, t_lo=2 * DAY, t_hi=min(8 * YEAR, max(30 * DAY, self.t_ca - YEAR - th)))
        if not np.isfinite(tf):
            return {"feasible": False}
        dv = esc["dv"] + float(np.linalg.norm(g.dv, axis=1).sum())
        return {"feasible": True, "t_launch": t_launch, "t_arrive": tf, "dv": dv, "params": p, "t_handover": th,
                "profile": g.profile()}  # the profile: the live run's warm start (TractorFlight._handover)

    def tractor_eta_curve(self, p, t_launch, thrusts):
        """Arrival time vs thrust for a launch at t_launch."""
        out = []
        for T in thrusts:
            q = MissionParams(**{**asdict(p), "tractor_thrust_n": float(T)})
            tp, _ = self.plan_tractor(q, t_launch)
            out.append((T, tp.t_arrive if tp.feasible else np.nan, tp.dv_total if tp.feasible else np.nan))
        return np.array(out)

    # ------------------------------------------------------------------ outcome
    def dart_success(self, p, b_after):
        """The DART verdict's test: the DARTs so far delivered at least (1 - dart_tolerance) of the target shift
        (2135 point b_after, measured from the undeflected flyby along the push direction)."""
        d, _ = self.push_direction(p)
        achieved = float((np.asarray(b_after, float) - self.b0) @ d)
        return achieved >= (1.0 - p.dart_tolerance) * self.target_shift(p)

    def dart_result(self, p, b_after, t_now, tractor_launched=False, q_full=None):
        """Verdict on the DARTs once their effect is known (ground tracking): success if they delivered at least
        (1 - tolerance) of the target shift (the tractor then steers the flyby to the aim point); otherwise the
        shift still missing toward the aim point is for another DART.
        q_full: the full-model 2135 flyby distance. A push so large that the 2135 encounter no longer happens
        projects onto the B-plane anywhere (even behind b0 along the push direction); a full-model flyby at or
        beyond the aim distance is then an overshoot, counted as a success (the tow holds it, no pull-back)."""
        R = self.goal_radius(p)
        d, _ = self.push_direction(p)
        b_after = np.asarray(b_after, float)
        target = self.target_shift(p)
        achieved = float((b_after - self.b0) @ d)
        need_goal = shift_to_circle(b_after, d, R)
        q = float(np.linalg.norm(b_after))
        remaining = max(0.0, float((self.aim_point(p) - b_after) @ d))
        ok = self.dart_success(p, b_after)
        overshoot = bool(not ok and q_full is not None and q_full >= np.linalg.norm(self.aim_point(p)))
        if overshoot:  # measured by the full-model distance: the shift along d that puts the flyby there
            ok, q, remaining = True, float(q_full), 0.0
            achieved = shift_to_circle(self.b0, d, q)
            need_goal = 0.0 if q >= R else need_goal
        reach = 0.0
        if not tractor_launched:
            tp, _ = self.plan_tractor(p, t_now, b_after)
            reach = tp.reach if tp.feasible else 0.0
        return {"success": bool(ok), "achieved": achieved, "target": target, "fraction": achieved / max(target, 1.0),
                "need_goal": need_goal, "tractor_reach": reach, "remaining": remaining,
                "q": q, "goal": R, "overshoot": overshoot}

    def outcome(self, p, impacts, tow_start, gt_accel, pending=(), n_beta=400, seed=0, b_offset=None):
        """Projected 2135 point from impacts [(t, dv)] that happened (beta known), planned impacts still on their
        way (pending, beta uncertain) and a steered tow from tow_start toward the aim point.
        b_offset: nonlinear correction of the linear prediction (ground orbit determination after an impact)."""
        R = self.goal_radius(p)
        d, required = self.push_direction(p)
        b_aim = self.aim_point(p)

        def shift(lst):
            db = np.zeros(2)
            for t_i, dv in lst:
                if dv is not None and np.isfinite(t_i):
                    db = db + self.B([t_i])[0] @ np.asarray(dv, float)
            return db

        db_known, db_pend = shift(impacts), shift(pending)
        off = np.zeros(2) if b_offset is None else np.asarray(b_offset, float)
        b_ki = self.b0 + off + db_known + db_pend
        if len(impacts) or len(pending):
            b_aim = self.aim_on_line(p, b_ki)
            if np.linalg.norm(b_aim) < np.linalg.norm(b_ki):
                # an overshoot is held, not pulled back: the tow's sensitivity is linearized about the scenario
                # orbit, and a large pull-back through the 2060 flyby is too nonlinear to steer safely
                b_aim = b_ki.copy()
        tow_end = min(tow_start + p.tow_years * YEAR, self.t_ca - 0.5 * YEAR) if np.isfinite(tow_start) else np.nan
        g = b_aim - b_ki
        w = g / np.linalg.norm(g) if np.linalg.norm(g) > 1.0 else d
        reach, e0 = self.tow_reach(gt_accel, tow_start, tow_end, w)
        gap = float((b_aim - b_ki) @ w)
        db_gt = min(max(gap, 0.0), reach) * w
        rng = np.random.default_rng(seed)
        if len(pending):
            betas = np.maximum(1.0, rng.normal(p.beta, p.beta_sigma, n_beta))
        else:
            betas = np.full(n_beta, p.beta)
        bk = (self.b0 + off + db_known)[None, :] + (betas / p.beta)[:, None] * db_pend[None, :]
        gaps = np.clip((b_aim[None, :] - bk) @ w, 0.0, reach)
        b_samples = bk + gaps[:, None] * w[None, :]
        b_final = b_ki + db_gt
        return {"db_ki": db_known + db_pend, "db_gt": db_gt, "db_total": db_known + db_pend + db_gt + off,
                "b_final": b_final, "b_ki": b_ki, "b0": self.b0, "b_aim": b_aim, "required": required, "goal": R,
                "direction": d, "betas": betas, "b_samples": b_samples, "tow_start": tow_start, "tow_end": tow_end,
                "tow_direction": e0, "tow_reach": reach, "q_final": float(np.linalg.norm(b_final)),
                "offset": off, "success": bool(np.linalg.norm(b_final) >= R),
                "p_success": float(np.mean(np.linalg.norm(b_samples, axis=1) >= R))}

    # ------------------------------------------------------------------ planning mode
    def plan(self, p: MissionParams, tractor_arrival=None, n_beta=400, seed=0, contact=True):
        """Opportunities, the selected DART, and a tractor launched once the impact is confirmed."""
        t_start = time.perf_counter()
        checks = []
        opps = self.opportunities(p)
        sel = None
        if 0 <= p.opportunity:
            sel = next((o for o in opps if o.index == p.opportunity), None)
        if sel is None:
            sel = self.recommended(opps)
        if sel is None:
            ip = ImpactorPlan()
            checks.append(Check("fail", "No DART opportunity within the departure-speed limit"))
        else:
            ip, msg = self.plan_dart(p, sel, contact=contact)
            if msg:
                checks.append(msg)
        tp, msg = TractorPlan(), None
        b_dart = self.b0 + (ip.db if ip.feasible else np.zeros(2))
        if ip.feasible:
            tp, msg = self.plan_tractor(p, ip.t_arrive + GROUND_OD_DAYS * DAY, b_dart)
            if tp.feasible and tractor_arrival is not None and np.isfinite(tractor_arrival):
                tp.t_arrive_flight = float(tractor_arrival)
                self._gravity_tractor(p, tp, b_dart, t_arrive=tp.t_arrive_flight)
            if msg:
                checks.append(msg)
        out = self.outcome(p, [], tp.tow_start if tp.feasible else np.nan, tp.gt_accel if tp.feasible else 0.0,
                           pending=[(ip.t_arrive, ip.dv_bennu)] if ip.feasible else [], n_beta=n_beta, seed=seed)
        R = out["goal"]
        q = out["q_final"]
        if ip.feasible:
            checks.append(Check("ok", f"DART #{ip.number}: {ip.mass:,.0f} kg at {ip.impact_speed / 1e3:.1f} km/s on "
                                      f"{self._date(ip.t_arrive)} (orbit raising {len(ip.raising.burns) - 1} burns, "
                                      f"{ip.propellant_kg:,.0f} kg propellant)"))
            if ip.raising.long_injection:
                checks.append(Check("warn", f"Injection burn longer than 8 minutes at {p.impactor_thrust_n / 1e3:g} kN: "
                                            "finite-burn losses not modeled"))
            c = ip.contact or {}
            if "lat" in c:
                checks.append(Check("ok", f"Point of contact {c['lat']:+.0f}° lat {c['lon']:+.0f}° lon, "
                                          f"{c['incidence_deg']:.0f}° from the vertical: {c['efficiency']:.2f}× the "
                                          "deflection of a head-on hit"))
        if tp.feasible:
            checks.append(Check("ok", f"Tractor launched after the impact arrives {self._date(tp.t_arrive)}; "
                                      f"tow can add up to {tp.reach / 1e3:,.0f} km"))
            checks.append(Check("ok" if tp.hover_ok else "warn",
                                f"Hover at {tp.hover_distance_m:.0f} m needs {tp.hover_thrust_n:.2f} N" +
                                ("" if tp.hover_ok else " (thrust-limited: hovering farther, weaker tow)")))
        checks.append(Check("ok" if q >= R else "fail", f"2135 flyby at {q / 1e3:,.0f} km = {q / LUNAR_DISTANCE:.2f} LD "
                                                        f"(goal {p.goal_ld:g} LD)"))
        lead = (self.t_ca - ip.t_arrive) / YEAR if ip.feasible else np.nan
        return PlanResult(params=p, opportunities=opps, impactor=ip, tractor=tp, t_encounter=self.t_ca, b0=self.b0,
                          b_aim=out["b_aim"], b_goal=R, direction=out["direction"], required_shift=out["required"],
                          target_shift=self.target_shift(p), db_ki=out["db_ki"], db_gt=out["db_gt"],
                          db_total=out["db_total"], b_final=out["b_final"], lead_time_years=lead,
                          success=bool(ip.feasible and q >= R), p_success=out["p_success"],
                          beta_samples=out["betas"], b_samples=out["b_samples"], checks=checks,
                          compute_ms=(time.perf_counter() - t_start) * 1e3)

    @staticmethod
    def _date(t):
        from core.timeutil import t_to_utc_str

        return t_to_utc_str(t)[:10]
