"""Bennu hits Earth (the scenario's 2182-09-24 impact without a deflection, or a live run whose final check still
predicts one): the approach track, the impact site, the numbers for the caption, and a ballistic plume.

Track: Earth-centred ICRF states from the scenario orbit (core/scenario.py: minute nodes in the last hour, ending
60 s before the surface) or a live run's full-model track; the last seconds are flown on under Earth's gravity
(point mass) to the surface.

Entry and impact numbers follow the Earth Impact Effects Program (Collins, Melosh & Marcus 2005, Meteoritics &
Planetary Science 40, 817-840; equation numbers below are theirs): exponential atmosphere (Eq. 5, H = 8 km,
rho_0 = 1 kg/m^3), drag before breakup (Eq. 8, C_D = 2), the strength-density relation (Eq. 9), the breakup altitude
(Eqs. 11-12), pancake spreading (Eqs. 15-16) to the pancake factor 7 (airburst altitude, Eq. 18), the drag after
breakup (Eq. 17, integrated numerically here), and the transient crater (Eq. 21: 1.161 for rock targets, 1.365 for
water) with the final crater (Eqs. 22 and 27). Their fireball estimate (Eq. 32) applies above ~15 km/s; Bennu enters
at ~12.7 km/s, so none is quoted. The site's surface type (ocean or land) is read from the Earth texture of the views.

The plume is a display model: particles thrown up from the site in a cone around the local vertical with speeds
of a few km/s, flown ballistically under Earth's gravity (no air); they land and stay on the turning Earth.
"""

from dataclasses import dataclass, field

import numpy as np

from core.constants import BENNU, DAY, GM, RADIUS

MU_E = GM["earth"]
R_E = RADIUS["earth"]
G_E = 9.81
H_ATM, RHO_0, C_D, PANCAKE = 8000.0, 1.0, 2.0, 7.0  # Collins et al. 2005
ATM_TOP = 100e3  # m: entry interface (the speed "at the top of the atmosphere")
RHO_WATER, RHO_ROCK = 1000.0, 2750.0
MT_TNT = 4.184e15  # J per megaton of TNT
SHOW_BEFORE = 2 * DAY  # the impact page: the last two days ...
SHOW_AFTER = 20 * 60.0  # ... and 20 minutes after the impact
SLOWMO_ALT = 1000e3  # slow motion over the last 1,000 km (5 s/s) ...
SLOWMO_EARTH = ((5.0, "Bennu: the last 1,000 km to Earth"), (60.0, "Impact: the plume (display model)"))


def gmst(t):
    """Greenwich mean sidereal angle [rad] (as app/scenes/base.py: display accuracy)."""
    from core.timeutil import t_to_jd

    return np.radians((280.46061837 + 360.98564736629 * (t_to_jd(t) - 2451545.0)) % 360.0)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _two_body(x, h, n):
    """n RK4 steps of h seconds under Earth's point-mass gravity; x (..., 6)."""
    def f(y):
        r = y[..., :3]
        a = -MU_E * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3
        return np.concatenate([y[..., 3:], a], axis=-1)

    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def entry(L0, rho_i, v0, theta):
    """Atmospheric entry of a body of diameter L0 [m], density rho_i, speed v0 at the top of the atmosphere and
    angle theta to the horizontal (Collins et al. 2005). Returns breakup altitude z_star, airburst altitude z_b
    (<= 0: the fragments reach the surface together), speed and spread diameter at the surface."""
    s = np.sin(theta)
    rho = lambda z: RHO_0 * np.exp(-z / H_ATM)
    Y = 10 ** (2.107 + 0.0624 * np.sqrt(rho_i))  # Eq. 9 (Pa)
    I_f = 4.07 * C_D * H_ATM * Y / (rho_i * L0 * v0**2 * s)  # Eq. 12
    out = {"strength_pa": Y, "I_f": I_f}
    if I_f >= 1.0:  # strong enough to reach the surface intact: drag only (Eq. 8)
        out.update(z_star=None, z_b=None, v_surface=v0 * np.exp(-3 * rho(0.0) * C_D * H_ATM / (4 * rho_i * L0 * s)),
                   L_surface=L0, airburst=False)
        return out
    z_s = -H_ATM * (np.log(Y / (RHO_0 * v0**2)) + 1.308 - 0.314 * I_f - 1.303 * np.sqrt(1 - I_f))  # Eq. 11
    v_s = v0 * np.exp(-3 * rho(z_s) * C_D * H_ATM / (4 * rho_i * L0 * s))  # Eq. 8 at the breakup
    ell = L0 * s * np.sqrt(rho_i / (C_D * rho(z_s)))  # Eq. 16
    z_b = z_s - 2 * H_ATM * np.log(1 + ell / (2 * H_ATM) * np.sqrt(PANCAKE**2 - 1))  # Eq. 18
    L = lambda z: L0 * np.sqrt(1 + (2 * H_ATM / ell) ** 2 * (np.exp((z_s - z) / (2 * H_ATM)) - 1) ** 2)  # Eq. 15
    z_end = max(z_b, 0.0)
    zz = np.linspace(z_end, z_s, 4001)
    integral = np.trapezoid(np.exp((z_s - zz) / H_ATM) * L(zz) ** 2, zz)
    v_end = v_s * np.exp(-0.75 * C_D * rho(z_s) / (rho_i * L0**3 * s) * integral)  # Eq. 17
    out.update(z_star=float(z_s), z_b=float(z_b), v_breakup=float(v_s), v_surface=float(v_end),
               L_surface=float(L(z_end)), airburst=bool(z_b > 0))
    return out


def transient_crater(L, rho_i, v, theta, water=False):
    """Transient crater (cavity) diameter [m] at the target surface, Collins et al. 2005 Eq. 21."""
    k, rho_t = (1.365, RHO_WATER) if water else (1.161, RHO_ROCK)
    return k * (rho_i / rho_t) ** (1 / 3) * L**0.78 * v**0.44 * G_E**-0.22 * np.sin(theta) ** (1 / 3)


def final_crater(D_tc):
    """Rim-to-rim diameter [m] of the final crater from the transient one (Eq. 22 simple; Eq. 27 complex, in km
    with D_c = 3.2 km, used above a 2.56 km transient crater)."""
    if D_tc <= 2560.0:
        return 1.25 * D_tc
    return 1e3 * 1.17 * (D_tc / 1e3) ** 1.13 / 3.2**0.13


@dataclass
class EarthImpact:
    """The approach and the impact, Earth-centred ICRF [m, m/s]. geo_state(t) -> (n, 6) for t <= t_hit."""

    t_hit: float
    t_track: np.ndarray
    x_track: np.ndarray
    site_lat: float = np.nan
    site_lon: float = np.nan
    numbers: dict = field(default_factory=dict)
    _plume: dict = field(default=None, repr=False)

    @classmethod
    def from_scenario(cls, eph, sc):
        """The scenario orbit without a deflection (sc: core/scenario.py KeyholeScenario)."""
        nom = sc.nominal
        t1 = min(nom.span[1], sc.t_impact)
        tt = np.unique(np.concatenate([np.arange(sc.t_impact - SHOW_BEFORE - DAY, t1 - 3600.0, 600.0),
                                       np.arange(t1 - 3600.0, t1, 10.0), [t1]]))
        x = nom.state(tt) - eph.state("earth", tt)
        return cls._finish(tt, x)

    @classmethod
    def from_track(cls, eph, t_track, x_track):
        """A live run's full-model track (barycentric states, e.g. final_check fine_t / fine_x)."""
        tt = np.asarray(t_track, float)
        return cls._finish(tt, np.asarray(x_track, float) - eph.state("earth", tt))

    @classmethod
    def _finish(cls, tt, x):
        """Fly the last state on to the surface (two-body, 0.5 s steps), and work out the impact."""
        r = np.linalg.norm(x[:, :3], axis=1)
        k = int(np.argmax(r <= R_E)) if np.any(r <= R_E) else len(tt)
        tt, x = tt[:k], x[:k]
        t, y, add_t, add_x = tt[-1], x[-1].copy(), [], []
        while np.linalg.norm(y[:3]) > R_E and len(add_t) < 20_000:
            y_new = _two_body(y, 0.5, 1)
            if np.linalg.norm(y_new[:3]) <= R_E:  # the last fraction of a step: linear to the surface
                r0, r1 = np.linalg.norm(y[:3]) - R_E, np.linalg.norm(y_new[:3]) - R_E
                f = r0 / (r0 - r1)
                y = y + f * (y_new - y)
                t += 0.5 * f
                add_t.append(t)
                add_x.append(y.copy())
                break
            y, t = y_new, t + 0.5
            if len(add_t) % 10 == 0:
                add_t.append(t)
                add_x.append(y.copy())
        if add_t:
            tt = np.concatenate([tt, add_t])
            x = np.vstack([x, add_x])
        imp = cls(t_hit=float(tt[-1]), t_track=tt, x_track=x)
        imp._site()
        return imp

    def geo_state(self, t):
        """Earth-centred ICRF state(s) of Bennu at t (clamped to the impact); linear in the 10-s samples."""
        t = np.atleast_1d(np.asarray(t, float))
        tc = np.clip(t, self.t_track[0], self.t_hit)
        return np.column_stack([np.interp(tc, self.t_track, self.x_track[:, k]) for k in range(6)])

    def altitude(self, t):
        return float(np.linalg.norm(self.geo_state(t)[0, :3]) - R_E)

    def _site(self):
        x = self.x_track[-1]
        r, v = x[:3], x[3:]
        n = r / np.linalg.norm(r)
        g = gmst(self.t_hit)
        body = rot_z(-g) @ r
        self.site_lat = float(np.degrees(np.arcsin(body[2] / np.linalg.norm(body))))
        self.site_lon = float((np.degrees(np.arctan2(body[1], body[0])) + 180.0) % 360.0 - 180.0)
        # speed and angle at the top of the atmosphere (the entry interface)
        alt = np.linalg.norm(self.x_track[:, :3], axis=1) - R_E
        k = int(np.argmax(alt <= ATM_TOP)) if np.any(alt <= ATM_TOP) else len(alt) - 1
        xe = self.x_track[k]
        ne = xe[:3] / np.linalg.norm(xe[:3])
        v0 = float(np.linalg.norm(xe[3:]))
        theta = float(np.arcsin(np.clip(-(xe[3:] @ ne) / v0, -1, 1)))  # to the local horizontal
        L0 = 2 * BENNU["radius"]
        rho_i = BENNU["mass"] / (4 / 3 * np.pi * BENNU["radius"] ** 3)
        ent = entry(L0, rho_i, v0, theta)
        self.numbers = {"t_hit": self.t_hit, "lat": self.site_lat, "lon": self.site_lon, "v_entry": v0,
                        "angle_deg": float(np.degrees(theta)), "v_surface_vacuum": float(np.linalg.norm(v)),
                        "diameter_m": L0, "density": rho_i, "mass": BENNU["mass"],
                        "energy_j": 0.5 * BENNU["mass"] * v0**2, "energy_mt": 0.5 * BENNU["mass"] * v0**2 / MT_TNT,
                        "entry": ent, "normal": n}

    def set_surface(self, water):
        """Ocean or land at the site (from the Earth texture): the crater numbers for that target."""
        nb = self.numbers
        ent, theta = nb["entry"], np.radians(nb["angle_deg"])
        v = ent["v_surface"]
        D_tc = transient_crater(nb["diameter_m"], nb["density"], v, theta, water=water)
        nb.update(water=bool(water), v_surface=v, energy_surface_mt=0.5 * nb["mass"] * v**2 / MT_TNT,
                  transient_m=float(D_tc), final_m=None if water else float(final_crater(D_tc)))
        return nb

    # ------------------------------------------------------------------ plume (display model)
    def plume(self, n=1500, seed=7, dt=2.0, t_max=SHOW_AFTER):
        """Ballistic plume from the site: positions (n_t, n, 3) Earth-centred ICRF [m] every dt seconds after the
        impact; particles that came down stay on the surface (turning with the Earth). Cached."""
        if self._plume is not None:
            return self._plume
        rng = np.random.default_rng(seed)
        nrm = self.numbers["normal"]
        a = np.cross(nrm, [0.0, 0.0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(nrm, a)
        half = np.radians(40.0)
        cos_t = rng.uniform(np.cos(half), 1.0, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        sin_t = np.sqrt(1 - cos_t**2)
        d = cos_t[:, None] * nrm + sin_t[:, None] * (np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b)
        speed = 300.0 * (5000.0 / 300.0) ** rng.uniform(0, 1, n) ** 1.6  # mostly slow, a few up to 5 km/s
        site = nrm * R_E
        omega = np.array([0.0, 0.0, 7.2921159e-5])
        v_rot = np.cross(omega, site)
        x = np.concatenate([np.repeat(site[None] * (1 + 1e-6), n, 0), d * speed[:, None] + v_rot], axis=1)
        n_t = int(t_max / dt) + 1
        out = np.empty((n_t, n, 3))
        landed = np.zeros(n, bool)
        land_body = np.zeros((n, 3))
        for i in range(n_t):
            t = i * dt
            R = rot_z(omega[2] * t)
            if i:
                x[~landed] = _two_body(x[~landed], dt, 1)
                down = ~landed & (np.linalg.norm(x[:, :3], axis=1) < R_E)
                if down.any():
                    p = x[down, :3]
                    land_body[down] = (R.T @ (R_E * p / np.linalg.norm(p, axis=1, keepdims=True)).T).T
                    landed |= down
            out[i] = x[:, :3]
            out[i, landed] = (R @ land_body[landed].T).T
        self._plume = {"dt": dt, "pos": out, "speed": speed}
        return self._plume

    def plume_at(self, t):
        """Plume particle positions at mission time t (None before the impact)."""
        since = t - self.t_hit
        if since < 0:
            return None
        pl = self.plume()
        i = min(int(since / pl["dt"]), len(pl["pos"]) - 1)
        return pl["pos"][i]

    def slow_motion(self, t):
        """(warp, what) for the display around the impact, or None."""
        if self.t_hit <= t < self.t_hit + SHOW_AFTER:
            return SLOWMO_EARTH[1]
        if t < self.t_hit and self.altitude(t) <= SLOWMO_ALT:
            return SLOWMO_EARTH[0]
        return None

    def t_slowmo(self):
        """When Bennu crosses 1,000 km altitude."""
        alt = np.linalg.norm(self.x_track[:, :3], axis=1) - R_E
        k = int(np.argmax(alt <= SLOWMO_ALT))
        return float(self.t_track[k])
