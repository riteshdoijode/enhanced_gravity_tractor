"""The stones the enhanced gravity tractor can pick up, their capture sites, and the LIDAR scan that finds them.

Stones (`population`):
  native  one boulder at each of the four OSIRIS-REx candidate sample sites, at the published coordinates
          (NASA, 2019-08-12): Nightingale 56°N 43°E (on the 5 cm OLA local model, the default capture site),
          Kingfisher 11°N 56°E, Osprey 11°N 88°E, Sandpiper 47°S 322°E.
  debris  fragments thrown out by the DART impact slower than the local escape speed, flown from the crater in
          Bennu's polyhedron gravity until they land (core/impact.py, EjectaCloud._fly_fragments). They rest
          where they land.
Sizes: volume-equivalent diameters from power-law size-frequency distributions, truncated to D_DRAW:
  native: cumulative slope -2.65, the middle of the -3.0 ... -2.3 measured at the four candidate sites
          (Burke et al. 2021, Remote Sensing 13, 1315);
  debris: differential index -3.9 (cumulative -2.9), measured for the boulders the DART impact ejected from
          Dimorphos (Jewitt et al. 2023, ApJL 952, L12).
Shape: convex rock (capture.boulder_mesh), axis ratios jittered around 1.3 : 1 : 0.75. Density: the team assumes
BOULDER_DENSITY (2,000 kg/m^3); each boulder's true density differs by DENSITY_SPREAD (1 sigma, illustrative),
so the scan gives a mass estimate +- sigma and the real mass shows only when the arm lifts it.

Scan (`Scan`): from SCAN_RANGE on the sunlit side the craft holds still while Bennu turns once under it (4.3 h).
Every control tick a north-south fan of SCAN_RAYS laser shots is ray-cast on the shape model; hits mark a
2 deg x 2 deg latitude/longitude coverage grid, and a stone is found once its cell has been hit. The fan is a
sample of the real instrument's shots: the size estimate uses the shot spacing of an OLA-class laser altimeter
firing at OLA_RATE_HZ (100 Hz beyond 7 km; Lauretta et al. 2017, Daly et al. 2017), s = sqrt(area / shots),
and each axis is measured to s / sqrt(6) (two edges, each uniform over one spacing).

Feasible: estimated diameter in D_CAPTURE (the robotic boulder capture design range, "typically 2-4 m",
Gates et al. 2015, Acta Astronautica 111, 29), local slope <= SLOPE_MAX, and thrust >= LIFT_MARGIN x the weight
of the craft plus the upper (2 sigma) mass estimate. Recommended: the feasible stone whose estimated mass is
closest to the planning mass (the "Boulder collected" setting the launch and the tow were planned for).

Site (`make_site`): origin on the surface under the stone, x east, y north, z up = -effective gravity (polyhedron
gravity + centrifugal at the current spin), the terrain a 20 m x 20 m heightfield (the 5 cm Nightingale grid
there; elsewhere ray-cast at 0.25 m on the shape model). The MuJoCo capture runs on it (core/proximity/capture.py).
"""

from dataclasses import dataclass, field

import numpy as np

from core.constants import BENNU
from core.proximity.capture import BOULDER_DENSITY, HFIELD_HALF, boulder_mesh

SITES = (("Nightingale", 56.0, 43.0), ("Kingfisher", 11.0, 56.0), ("Osprey", 11.0, 88.0), ("Sandpiper", -47.0, 322.0))
N_DEBRIS = 6
D_DRAW = (1.9, 4.4)  # m: the candidates the team looks for, around the capture range (volume-equivalent)
D_CAPTURE = (2.0, 4.0)  # m: what the arm and claws are designed for
SFD_SLOPE = {"native": 2.65, "debris": 2.9}  # cumulative N(>D) ~ D^-slope
DENSITY_SPREAD = 0.15
SLOPE_MAX = 20.0  # deg: the rock must lie still while the claws close (MuJoCo captures failed on some 23-25 deg sites)
LIFT_MARGIN = 1.3  # as core/proximity/ops.py
SCAN_RANGE = 2000.0  # m from Bennu's center
SCAN_RAYS = 120  # shots per fan (one fan per control tick)
OLA_RATE_HZ = 100.0
LAT_BINS, LON_BINS = 90, 180
BEAMS_SHOWN = 16


def draw_stones(rng, n, kind):
    """Size, shape and density of n stones: dicts with d_eq, aspect, density, seed (deterministic given rng)."""
    a = SFD_SLOPE[kind]
    lo, hi = D_DRAW
    u = rng.uniform(0.0, 1.0, n)
    d = (lo ** -a - u * (lo ** -a - hi ** -a)) ** (-1.0 / a)  # inverse CDF of the truncated power law
    out = []
    for k in range(n):
        aspect = tuple(float(x) for x in np.array([1.3, 1.0, 0.75]) * np.exp(rng.normal(0.0, 0.08, 3)))
        out.append(dict(d_eq=float(d[k]), aspect=aspect,
                        density=float(BOULDER_DENSITY * np.clip(1.0 + rng.normal(0.0, DENSITY_SPREAD), 0.6, 1.4)),
                        seed=int(rng.integers(1 << 30))))
    return out


@dataclass
class Stone:
    key: str
    kind: str  # "native" | "debris"
    name: str
    point_b: np.ndarray  # surface point under the stone (body frame, m)
    d_eq: float
    aspect: tuple
    density: float
    seed: int
    up_b: np.ndarray = None  # local vertical (body frame)
    mass: float = 0.0  # true
    dims: np.ndarray = None  # true extents (m)
    slope: float = np.nan
    g: float = np.nan
    detected: bool = False
    t_found: float = np.nan
    dims_est: np.ndarray = None
    d_est: float = np.nan
    mass_est: float = np.nan
    mass_sigma: float = np.nan
    feasible: bool = False
    why: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def lat(self):
        return float(np.degrees(np.arcsin(self.point_b[2] / np.linalg.norm(self.point_b))))

    @property
    def lon(self):
        return float(np.degrees(np.arctan2(self.point_b[1], self.point_b[0])) % 360.0)

    def label(self, index):
        kind = "DART debris" if self.kind == "debris" else self.name
        return f"#{index} {kind}"

    def to_dict(self):
        return {"key": self.key, "kind": self.kind, "name": self.name, "point_b": self.point_b, "up_b": self.up_b,
                "lat": self.lat, "lon": self.lon, "d_est": self.d_est, "dims_est": self.dims_est,
                "mass_est": self.mass_est, "mass_sigma": self.mass_sigma, "slope": self.slope, "g": self.g,
                "feasible": self.feasible, "why": self.why, "detected": self.detected, "t_found": self.t_found,
                # the real rock, for drawing it (the labels show only the estimates)
                "mass": self.mass, "seed": self.seed, "aspect": self.aspect, "density": self.density}


def _surface_point(env, u):
    """Surface point (body frame) seen along the unit direction u from Bennu's center."""
    d, _ = env.ray.cast(1000.0 * u, -u, max_range=1000.0)
    return (1000.0 - d) * u if d is not None else BENNU["radius"] * u


def _effective_gravity(env, p_b, omega_b):
    """Polyhedron gravity + centrifugal at body point p_b (body frame)."""
    up0 = p_b / np.linalg.norm(p_b)
    return env.grav.accel_body(p_b + 1.0 * up0) - np.cross(omega_b, np.cross(omega_b, p_b))


def _local_frame(g_b):
    up = -g_b / np.linalg.norm(g_b)
    east = np.cross([0.0, 0.0, 1.0], up)
    if np.linalg.norm(east) < 1e-6:
        east = np.cross([0.0, 1.0, 0.0], up)
    east /= np.linalg.norm(east)
    return np.column_stack([east, np.cross(up, east), up])


def _heights(env, origin, R, xs, top=40.0):
    """Terrain height (along up, relative to origin) on the grid xs x xs, ray-cast on the shape model."""
    Z = np.full((len(xs), len(xs)), np.nan)
    down = -R[:, 2]
    for i, y in enumerate(xs):
        for j, x in enumerate(xs):
            d, _ = env.ray.cast(origin + R @ np.array([x, y, top]), down, max_range=2 * top)
            if d is not None:
                Z[i, j] = top - d
    if np.isnan(Z).all():
        return np.zeros_like(Z)
    return np.where(np.isnan(Z), np.nanmin(Z), Z)


def _slope(xs, Z, radius=3.0):
    """Mean slope [deg] of the terrain within radius of the center (plane fit)."""
    X, Y = np.meshgrid(xs, xs)
    m = np.hypot(X, Y) <= radius
    A = np.column_stack([X[m], Y[m], np.ones(m.sum())])
    (a, b, _), *_ = np.linalg.lstsq(A, Z[m], rcond=None)
    return float(np.degrees(np.arctan(np.hypot(a, b))))


@dataclass
class Site:
    """A capture site: origin (body frame), R (local east/north/up -> body), g_local (effective gravity, local
    frame), the heightfield grid (m) and heights relative to the origin (rows: y, columns: x)."""
    name: str
    origin: np.ndarray
    R: np.ndarray
    g_local: np.ndarray
    grid: np.ndarray
    heights: np.ndarray
    terrain: str

    @property
    def up(self):
        return self.R[:, 2]

    @property
    def g(self):
        return self.R @ self.g_local  # body frame

    def slope(self):
        return _slope(self.grid, self.heights)


def nightingale_site():
    from core.proximity.capture import _hfield, site_frame

    origin, R, g_local = site_frame()
    grid, Z = _hfield()
    return Site("Nightingale", origin, R, g_local, grid, Z, "OLA 5 cm local model")


def make_site(env, stone, omega_b, step=0.25):
    """The capture site under a stone (see the module docstring)."""
    if stone.key == "native:Nightingale":
        return nightingale_site()
    p = np.asarray(stone.point_b, float)
    g_b = _effective_gravity(env, p, omega_b)
    R = _local_frame(g_b)
    xs = np.arange(-HFIELD_HALF, HFIELD_HALF + 1e-9, step)
    Z = _heights(env, p, R, xs)
    z0 = float(Z[len(xs) // 2, len(xs) // 2])
    origin = p + z0 * R[:, 2]
    return Site(stone.name, origin, R, R.T @ g_b, xs, Z - z0, "global shape model")


def population(mission, env, t):
    """Every stone on Bennu the scan can find (see the module docstring): the four native boulders and the
    DART debris that has landed. Deterministic for a given run (own random stream)."""
    from core.dynamics.asteroid import omega_body
    from core.proximity.capture import site_frame

    rng = np.random.default_rng([int(getattr(mission, "seed", 0)), 11])
    omega_b = omega_body(t)
    stones = []
    native = draw_stones(rng, len(SITES), "native")
    for (name, lat, lon), spec in zip(SITES, native):
        if name == "Nightingale":
            p = site_frame()[0]  # the flattest 3 m spot of the 5 cm model (core/proximity/capture.py)
        else:
            a, b = np.radians(lat), np.radians(lon)
            p = _surface_point(env, np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]))
        stones.append(Stone(f"native:{name}", "native", name, p, **spec))
    ej = getattr(mission, "ejecta", None)
    for k, f in enumerate((ej.fragments or []) if ej is not None else []):
        u = f["land_b"] / np.linalg.norm(f["land_b"])
        p = _surface_point(env, u)
        lat, lon = np.degrees(np.arcsin(u[2])), np.degrees(np.arctan2(u[1], u[0])) % 360.0
        spec = {key: f[key] for key in ("d_eq", "aspect", "density", "seed")}
        stones.append(Stone(f"debris:{k}:{lat:.2f}:{lon:.2f}", "debris", f"DART debris {k + 1}", p, **spec,
                            extra={"v_launch": f["v_launch"], "t_land": f["t_land"]}))
    for s in stones:
        s.mass = s.density * np.pi / 6.0 * s.d_eq ** 3
        hull = boulder_mesh(s.mass, seed=s.seed, aspect=s.aspect, density=s.density)
        s.dims = hull.extents.copy()
        g_b = _effective_gravity(env, s.point_b, omega_b)
        s.g = float(np.linalg.norm(g_b))
        R = _local_frame(g_b)
        s.up_b = R[:, 2]
        xs = np.linspace(-3.0, 3.0, 13)  # (the same plane fit as the capture site's: 3 m around the stone)
        s.slope = _slope(xs, _heights(env, s.point_b, R, xs), radius=3.0)
    return stones


def recommend(stones, planning_mass):
    ok = [s for s in stones if s.detected and s.feasible]
    if not ok:
        return None
    return min(ok, key=lambda s: (abs(s.mass_est - planning_mass), s.key))


class Scan:
    """The LIDAR scan (see the module docstring). Physics-side: called once per control tick."""

    def __init__(self, stones, t0, rng):
        self.stones = stones
        self.t0 = float(t0)
        self.period = 2 * np.pi / BENNU["omega"]  # one Bennu day
        self.scanned = 0.0
        self.hits = np.zeros((LAT_BINS, LON_BINS), np.uint16)
        self.version = 0
        self.beams = []  # the latest fan, shown: (start, end) inertial, Bennu-centered
        self.rng = rng
        self.complete = False
        self.recommended = None
        self.shots = 0

    @staticmethod
    def cell(p_b):
        r = np.linalg.norm(p_b, axis=-1)
        lat = np.degrees(np.arcsin(np.clip(p_b[..., 2] / r, -1, 1)))
        lon = np.degrees(np.arctan2(p_b[..., 1], p_b[..., 0])) % 360.0
        return (np.clip(((lat + 90.0) / 180.0 * LAT_BINS).astype(int), 0, LAT_BINS - 1),
                np.clip((lon / 360.0 * LON_BINS).astype(int), 0, LON_BINS - 1))

    def coverage(self):
        """Fraction of Bennu's surface (area-weighted) hit at least once."""
        lat_c = np.radians(-90.0 + (np.arange(LAT_BINS) + 0.5) * 180.0 / LAT_BINS)
        w = np.cos(lat_c)[:, None] * np.ones((1, LON_BINS))
        return float((w * (self.hits > 0)).sum() / w.sum())

    def sigma_axis(self):
        """1-sigma size error per axis from the instrument's shot spacing over the scanned area."""
        area = 4 * np.pi * BENNU["radius"] ** 2 * max(self.coverage(), 1e-3)
        shots = max(OLA_RATE_HZ * max(self.scanned, 1.0), 1.0)
        return float(np.sqrt(area / shots) / np.sqrt(6.0))

    def sweep(self, t, p_i, R, env, dt, thrust, m_sc):
        """One fan of shots from the craft at p_i (Bennu-centered inertial), Bennu's attitude R. Returns the stones
        found by it."""
        b0 = -p_i / np.linalg.norm(p_i)
        pole = R[:, 2]
        across = pole - (pole @ b0) * b0
        across /= np.linalg.norm(across)
        half = np.arcsin(min(1.0, 1.05 * env.grav.R_bound / np.linalg.norm(p_i)))
        th = np.linspace(-half, half, SCAN_RAYS)
        p_b = R.T @ p_i
        beams, pts = [], []
        for k, a in enumerate(th):
            d_i = np.cos(a) * b0 + np.sin(a) * across
            d_b = R.T @ d_i
            dist, _ = env.ray.cast(p_b, d_b, max_range=2 * np.linalg.norm(p_i))
            if dist is None:
                continue
            hit_b = p_b + dist * d_b
            pts.append(hit_b)
            if k % max(1, SCAN_RAYS // BEAMS_SHOWN) == 0:
                beams.append((p_i.copy(), R @ hit_b))
        self.beams = beams
        if pts:
            i, j = self.cell(np.array(pts))
            np.add.at(self.hits, (i, j), 1)
            self.version += 1
        self.scanned += dt
        self.shots += len(pts)
        found = []
        sig = self.sigma_axis()
        for s in self.stones:
            if s.detected:
                continue
            i, j = self.cell(s.point_b)
            if self.hits[i, j] > 0:
                self._measure(s, t, sig, thrust, m_sc)
                found.append(s)
        return found

    def _measure(self, s, t, sig, thrust, m_sc):
        s.detected, s.t_found = True, float(t)
        s.dims_est = np.maximum(s.dims + self.rng.normal(0.0, sig, 3), 0.3)
        v_est = s.mass / s.density * float(np.prod(s.dims_est / s.dims))
        s.mass_est = BOULDER_DENSITY * v_est
        s.d_est = float((6.0 * v_est / np.pi) ** (1.0 / 3.0))
        s.mass_sigma = s.mass_est * float(np.sqrt(np.sum((sig / s.dims_est) ** 2) + DENSITY_SPREAD ** 2))
        lo, hi = D_CAPTURE
        weight = LIFT_MARGIN * (m_sc + s.mass_est + 2 * s.mass_sigma) * s.g
        if s.d_est < lo:
            s.feasible, s.why = False, f"too small for the claws ({s.d_est:.2f} m < {lo:g} m)"
        elif s.d_est > hi:
            s.feasible, s.why = False, f"too large for the arm ({s.d_est:.2f} m > {hi:g} m)"
        elif s.slope > SLOPE_MAX:
            s.feasible, s.why = False, f"slope {s.slope:.0f}° (> {SLOPE_MAX:.0f}°): unsafe descent"
        elif thrust < weight:
            s.feasible, s.why = False, f"too heavy to lift with {thrust:g} N"
        else:
            s.feasible, s.why = True, ""

    def snapshot(self, with_map):
        out = {"scanned": self.scanned, "period": self.period, "coverage": self.coverage(), "version": self.version,
               "complete": self.complete, "recommended": self.recommended,
               "beams": [b for b in self.beams] if not self.complete else []}
        if with_map:
            out["hits"] = (self.hits > 0).astype(np.uint8)
        return out
