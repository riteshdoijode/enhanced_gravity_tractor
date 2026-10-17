"""DART-style impactor: Earth orbit raising before departure, and the recommended point of contact.

Orbit raising (high thrust, burns at perigee). The impactor is assembled in the circular parking orbit
(radius r_p). Each burn at perigee raises the apogee; the perigee speed after burn k is v_k and the
orbit is the ellipse with r_p and r_a = r_p v_k^2 / (2 mu / r_p - v_k^2):
    burns of at most  min(T/m * 8 min, 500 m/s)  (short burns near perigee: small gravity losses),
    apogee kept below 300,000 km (inside the Moon's orbit, so geocentric two-body motion is a fair model),
then one injection burn at perigee to the departure speed v_dep = sqrt(v_inf^2 + 2 mu / r_p): the
hyperbolic excess v_inf of the chosen Lambert arc. The user sets a maximum speed v_max: raising stops
when v_dep is reached and only arcs with v_dep <= v_max are flown. All burns are at perigee, so the total
is the single-burn value v_dep - v_c (the Oberth effect is kept). Propellant: rocket equation
m_prop = m_after (exp(dv / c) - 1), or m dv / c with the project's "unlimited fuel" convention (mass
held constant, consumption tracked). Burn k+1 follows burn k after one period of orbit k; the schedule is
counted back from the departure date.
The orbit plane contains the departure asymptote u, and the perigee sits at the asymptote's true anomaly
theta_inf = acos(-1/e) behind it, so the craft leaves Earth along u.

Point of contact (Cheng et al. 2023 eq. 1 with ejecta along the crater-scale surface normal n):
    dv = (m / M) [U + (beta - 1) (n . U) n]
The B-plane shift along the desired direction w is  w . B dv = g . dv  with g = B^T w. Parallel rays along
U are cast on the shape model inside 60 % of Bennu's radius (aiming margin), and the hit maximizing
g . dv is recommended. A flat patch facing the impactor gives the head-on value beta m (g . U) / M; a
sloped one loses part of the ejecta recoil.
"""

from dataclasses import dataclass, field

import numpy as np

from core.constants import BENNU, G0, GM

MU_E = GM["earth"]
R_APO_MAX = 300_000e3  # m
BURN_MINUTES = 8.0
DV_BURN_MAX = 500.0  # m/s


@dataclass
class Burn:
    t: float
    dv: float
    v_perigee: float  # perigee speed after the burn [m/s]
    r_apogee: float  # apogee radius after the burn [m] (inf after the injection)
    period: float  # period of the orbit after the burn [s] (inf after the injection)
    propellant_kg: float
    kind: str  # "raise" | "inject"


@dataclass
class RaisingPlan:
    burns: list = field(default_factory=list)
    r_park: float = np.nan
    v_circ: float = np.nan
    v_dep: float = np.nan
    v_inf: float = np.nan
    dv_total: float = np.nan
    propellant_kg: float = np.nan
    t_start: float = np.nan
    t_depart: float = np.nan
    P_hat: np.ndarray = None  # perigee direction (ICRF)
    Q_hat: np.ndarray = None
    h_hat: np.ndarray = None
    long_injection: bool = False  # injection burn longer than the 8-minute limit (finite-burn losses)


def apogee_radius(r_p, v_p):
    """Apogee radius of the orbit with perigee r_p and perigee speed v_p (inf if unbound)."""
    e2 = 2 * MU_E / r_p - v_p**2
    return np.inf if e2 <= 0 else r_p * v_p**2 / e2


def ellipse_period(r_p, r_a):
    a = 0.5 * (r_p + r_a)
    return 2 * np.pi * np.sqrt(a**3 / MU_E)


def departure_speed(v_inf, r_p):
    return np.sqrt(v_inf**2 + 2 * MU_E / r_p)


def burn_limit(mass, thrust):
    return min(thrust / mass * BURN_MINUTES * 60.0, DV_BURN_MAX)


def departure_plane(u):
    """Orbit plane (P perigee, Q, h) whose hyperbola leaves along unit u; lowest-inclination plane holding u."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    h = np.array([0.0, 0.0, 1.0]) - u[2] * u
    if np.linalg.norm(h) < 1e-6:
        h = np.array([1.0, 0.0, 0.0]) - u[0] * u
    return u, h / np.linalg.norm(h)


def raising_schedule(mass, v_inf_vec, r_park, thrust, isp, t_depart, unlimited=True):
    """Perigee burns from the circular parking orbit to the departure on v_inf_vec at t_depart."""
    v_inf = float(np.linalg.norm(v_inf_vec))
    v_c = np.sqrt(MU_E / r_park)
    v_dep = departure_speed(v_inf, r_park)
    v_bound = np.sqrt(2 * MU_E * R_APO_MAX / (r_park * (r_park + R_APO_MAX)))  # apogee at the cap
    step = burn_limit(mass, thrust)
    speeds = []  # perigee speed after each raising burn
    v = v_c
    while v < v_bound - 1e-6:
        v = min(v + step, v_bound)
        speeds.append(v)
    c = isp * G0
    plan = RaisingPlan(r_park=r_park, v_circ=v_c, v_dep=v_dep, v_inf=v_inf, t_depart=t_depart)
    # times: the injection is at t_depart; each earlier burn one period of the orbit it starts
    raise_t = []
    t = t_depart
    for vk in reversed(speeds):
        t -= ellipse_period(r_park, apogee_radius(r_park, vk))
        raise_t.append(t)
    raise_t = raise_t[::-1]
    dvs = np.diff(np.concatenate([[v_c], speeds, [v_dep]]))
    # propellant, counted back from the impact mass (finite fuel) or at constant mass (unlimited)
    props = np.empty(len(dvs))
    m_after = mass
    for k in range(len(dvs) - 1, -1, -1):
        props[k] = mass * dvs[k] / c if unlimited else m_after * (np.exp(dvs[k] / c) - 1.0)
        m_after += props[k]
    for k, vk in enumerate(speeds):
        ra = apogee_radius(r_park, vk)
        plan.burns.append(Burn(raise_t[k], float(dvs[k]), float(vk), float(ra), float(ellipse_period(r_park, ra)),
                               float(props[k]), "raise"))
    plan.burns.append(Burn(float(t_depart), float(dvs[-1]), float(v_dep), np.inf, np.inf, float(props[-1]), "inject"))
    plan.dv_total = float(v_dep - v_c)
    plan.propellant_kg = float(props.sum())
    plan.t_start = plan.burns[0].t
    plan.long_injection = dvs[-1] * mass / thrust > BURN_MINUTES * 60.0
    # geometry: hyperbola eccentricity e = 1 + r_p v_inf^2 / mu, asymptote at theta_inf = acos(-1/e)
    u, h = departure_plane(v_inf_vec)
    e = 1.0 + r_park * v_inf**2 / MU_E
    th = np.arccos(-1.0 / e)
    P = np.cos(th) * u - np.sin(th) * np.cross(h, u)
    plan.P_hat, plan.h_hat = P, h
    plan.Q_hat = np.cross(h, P)
    return plan


def kepler_state(r_p, v_p, P_hat, Q_hat, dt):
    """Geocentric state dt seconds after a perigee pass of the conic (r_p, v_p) in the plane (P, Q).
    Ellipse or hyperbola (universal Newton iteration on the anomaly)."""
    e = r_p * v_p**2 / MU_E - 1.0
    a = r_p / (1.0 - e) if abs(1.0 - e) > 1e-12 else np.inf
    if e < 1.0:
        n = np.sqrt(MU_E / a**3)
        M = (n * dt) % (2 * np.pi)
        E = M if e < 0.8 else np.pi
        for _ in range(50):
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= dE
            if abs(dE) < 1e-13:
                break
        cosv = (np.cos(E) - e) / (1 - e * np.cos(E))
        sinv = np.sqrt(1 - e**2) * np.sin(E) / (1 - e * np.cos(E))
    else:
        n = np.sqrt(MU_E / (-a) ** 3)
        M = n * dt
        H = np.arcsinh(M / e) if e > 0 else M
        for _ in range(80):
            dH = (e * np.sinh(H) - H - M) / (e * np.cosh(H) - 1)
            H -= dH
            if abs(dH) < 1e-13:
                break
        cosv = (e - np.cosh(H)) / (e * np.cosh(H) - 1)
        sinv = np.sqrt(e**2 - 1) * np.sinh(H) / (e * np.cosh(H) - 1)
    p = r_p * (1 + e)
    r = p / (1 + e * cosv)
    pos = r * (cosv * P_hat + sinv * Q_hat)
    vel = np.sqrt(MU_E / p) * (-sinv * P_hat + (e + cosv) * Q_hat)
    return pos, vel


def orbit_points(r_p, v_p, P_hat, Q_hat, n=200, r_max=None):
    """Points of the conic (display): the full ellipse, or the hyperbola branch out to r_max."""
    e = r_p * v_p**2 / MU_E - 1.0
    p = r_p * (1 + e)
    if e < 1.0:
        nu = np.linspace(0, 2 * np.pi, n)
    else:
        lim = np.arccos(np.clip((p / (r_max or 1e9) - 1) / e, -1, 1))
        nu = np.linspace(0, lim, n)
    r = p / (1 + e * np.cos(nu))
    return r[:, None] * (np.cos(nu)[:, None] * P_hat + np.sin(nu)[:, None] * Q_hat)


# ----------------------------------------------------------------------------- point of contact
def contact_candidates(ray, R_imp, U, n_grid=15, frac=0.6):
    """The candidate aim points: hits of parallel rays along U inside frac x Bennu's radius (body frame), with
    their crater-scale normals. Pure geometry (no beta, mass or push direction), so a planner can keep it per
    arc and score it for any beta. R_imp: body -> ICRF at impact; U: impactor velocity relative to Bennu (ICRF)."""
    from core.impact import crater_normal

    Ub = R_imp.T @ np.asarray(U, float)
    u = Ub / np.linalg.norm(Ub)
    a = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(u, a)
    rmax = frac * BENNU["radius"]
    points, normals, offsets = [], [], []
    grid = np.linspace(-rmax, rmax, n_grid) if n_grid > 1 else np.zeros(1)  # n_grid = 1: the central ray
    for x in grid:
        for y in grid:
            if x * x + y * y > rmax**2:
                continue
            p0 = x * a + y * b - 1000.0 * u
            dist, n_facet = ray.cast(p0, u, max_range=2000.0)
            if dist is None:
                continue
            point = p0 + u * dist
            n = crater_normal(ray, point)
            points.append(point)
            normals.append(np.asarray(n_facet, float) if n is None else n)
            offsets.append(float(np.hypot(x, y)))
    return {"Ub": Ub, "points": np.reshape(points, (-1, 3)), "normals": np.reshape(normals, (-1, 3)),
            "offsets": np.asarray(offsets, float)}


def contact_scores(cand, beta, mass, gb, M=BENNU["mass"]):
    """Desired B-plane shift g . dv of each candidate (gb: g in the body frame), ejecta along its normal."""
    Ub, N = cand["Ub"], cand["normals"]
    dv = (mass * Ub + mass * (beta - 1.0) * (N @ Ub)[:, None] * N) / M
    return dv @ gb


def best_contact_point(ray, R_imp, U, beta, mass, g, n_grid=15, frac=0.6, M=BENNU["mass"], cand=None):
    """Recommended aim point on the shape model. R_imp: body -> ICRF at impact; U: impactor velocity relative
    to Bennu (ICRF); g: gradient of the desired B-plane shift with respect to Bennu's velocity change; cand: the
    arc's contact_candidates if already computed. Returns a dict (point_b, normal_b, lat, lon, incidence_deg,
    efficiency vs head-on, offset_m) or None."""
    cand = contact_candidates(ray, R_imp, U, n_grid, frac) if cand is None else cand
    if len(cand["offsets"]) == 0:
        return None
    Ub, gb = cand["Ub"], R_imp.T @ np.asarray(g, float)
    u = Ub / np.linalg.norm(Ub)
    score = contact_scores(cand, beta, mass, gb, M)
    i = int(np.argmax(score))  # the first of equal scores, as a scan in grid order
    point, n = cand["points"][i], cand["normals"][i]
    r = np.linalg.norm(point)
    head_on = beta * mass * (gb @ Ub) / M
    return {"point_b": point, "normal_b": n, "score": float(score[i]), "offset_m": float(cand["offsets"][i]),
            "lat": float(np.degrees(np.arcsin(point[2] / r))), "lon": float(np.degrees(np.arctan2(point[1], point[0]))),
            "incidence_deg": float(np.degrees(np.arccos(np.clip(-(n @ u), -1, 1)))),
            "efficiency": float(score[i]) / head_on if head_on != 0 else np.nan}
