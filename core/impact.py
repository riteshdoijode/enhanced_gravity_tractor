"""Kinetic-impact physics on Bennu's real shape (Step 6).

Impact point: the impactor's straight path over the last integration step, expressed in Bennu's body
frame, ray-cast on the OSIRIS-REx shape model -> surface point r; outward normal n averaged over a 20 m
patch (the crater scale; single 1 m facets of a boulder-strewn surface point anywhere).

Momentum transfer (Cheng et al. 2023, Nature 616:457, eq. 1), E = net ejecta-momentum direction:
    M dv = m U + m (beta - 1) (E . U) E
Ejecta leave roughly along the local surface normal, so E = n by default ("head-on" E = -U/|U| is the
planner's assumption, since the impact point is not known in advance).

Spin change: both the impactor momentum and the ejecta recoil act at r, so about the center of mass
    dL = r x (M dv),   d(omega) = J^-1 dL     (J: inertia tensor from the shape model, body frame)
rotation period change dP = 2 pi / |omega + d omega| - 2 pi / |omega|.

Ejecta cloud (illustrative model): speeds from point-source crater scaling, cumulative mass faster than v
    M(> v) ~ v^(-3 mu),  mu ~ 0.41 for porous, sand-like targets (Housen & Holsapple 2011, Icarus 211:856)
truncated to [v_min, v_max], launched in a 45 deg cone around n. The total ejecta mass is set so that the
escaping ejecta carry the recoil momentum m (beta - 1) (E . U). Tracers are then flown in Bennu's polyhedron
gravity (batched over the worker threads) until they land or leave; this runs on a background thread while
the live simulation continues.
"""

import threading

import numpy as np

from core.assets import ASSET_DIR, bennu_props
from core.constants import BENNU
from core.deflection import ki_delta_v
from core.dynamics.asteroid import bennu_attitude

M_BENNU = BENNU["mass"]
OMEGA_B = np.array([0.0, 0.0, BENNU["omega"]])
MU_SCALING = 0.41
CONE_DEG = 45.0


def inertia_body():
    """Bennu's inertia tensor (body frame), scaled to the measured mass."""
    props = bennu_props()
    return np.array(props["inertia_kg_m2"]) * (M_BENNU / props["mass_kg"])


CRATER_SCALE = 20.0  # m: ejecta follow the surface orientation averaged over roughly the crater size


def crater_normal(ray, point_b, radius=CRATER_SCALE):
    """Area-weighted mean outward normal of the global shape model within `radius` of the impact point.
    The facets near the point come from a k-d tree of the 400k facet centers, then the full scan's own test:
    the same facets in the same order (bit-identical result), ~50x faster than scanning every facet."""
    if not hasattr(ray, "_cells"):
        from scipy.spatial import cKDTree

        mesh = ray.global_mesh
        c = np.asarray(mesh.cell_centers().points)
        ray._cells = (c, np.asarray(mesh.cell_normals),
                      np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"]), cKDTree(c))
    c, n, a, tree = ray._cells
    point_b = np.asarray(point_b, float)
    idx = np.sort(np.asarray(tree.query_ball_point(point_b, radius * (1.0 + 1e-9)), dtype=int))
    near = idx[np.sum((c[idx] - point_b) ** 2, axis=1) < radius**2]
    if len(near) == 0:
        return None
    m = (a[near, None] * n[near]).sum(0)
    return m / np.linalg.norm(m)


def locate_impact(ray, p0_b, p1_b):
    """First intersection of the segment p0 -> p1 (Bennu body frame) with the shape model.
    'facet_normal' is the local facet, 'normal' the crater-scale surface orientation."""
    d = np.asarray(p1_b, float) - np.asarray(p0_b, float)
    L = float(np.linalg.norm(d))
    if L <= 0:
        return None
    dist, n = ray.cast(np.asarray(p0_b, float), d / L, max_range=L)
    if dist is None:
        return None
    point = p0_b + d / L * dist
    nc = crater_normal(ray, point)
    return {"point": point, "normal": nc if nc is not None else np.asarray(n, float),
            "facet_normal": np.asarray(n, float), "fraction": dist / L}


def momentum_transfer(m, U, beta, E_hat):
    """Bennu's velocity change and the ejecta recoil momentum (both in the frame of U)."""
    E_hat = np.asarray(E_hat, float) / np.linalg.norm(E_hat)
    dv = ki_delta_v(m, U, beta, E_hat)
    p_ej = m * (beta - 1.0) * (E_hat @ U) * E_hat
    return dv, p_ej


def spin_change(r_b, dP_b, J=None, omega=OMEGA_B):
    """Rotation change from momentum dP_b delivered at body-frame point r_b."""
    J = inertia_body() if J is None else J
    dL = np.cross(r_b, dP_b)
    dw = np.linalg.solve(J, dL)
    w1 = omega + dw
    P0, P1 = 2 * np.pi / np.linalg.norm(omega), 2 * np.pi / np.linalg.norm(w1)
    tilt = np.degrees(np.arccos(np.clip(w1 @ omega / (np.linalg.norm(w1) * np.linalg.norm(omega)), -1, 1)))
    return {"d_omega": dw, "dL": dL, "period_change_s": P1 - P0, "pole_shift_deg": tilt}


def radius_map(ray=None, n_lat=181, n_lon=361):
    """Surface radius on a latitude/longitude grid (body frame), from the shape model; cached."""
    path = ASSET_DIR / "bennu_radius_map.npy"
    if path.exists():
        return np.load(path)
    lat = np.radians(np.linspace(-90, 90, n_lat))
    lon = np.radians(np.linspace(-180, 180, n_lon))
    R = np.empty((n_lat, n_lon))
    for i, a in enumerate(lat):
        for j, b in enumerate(lon):
            u = np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])
            d, _ = ray.cast(1000.0 * u, -u, max_range=1000.0)
            R[i, j] = 1000.0 - d if d is not None else 250.0
    np.save(path, R)
    return R


def surface_radius(R_map, X):
    """Bilinear lookup of the surface radius under body-frame points X (n, 3)."""
    n_lat, n_lon = R_map.shape
    r = np.linalg.norm(X, axis=1)
    lat = np.degrees(np.arcsin(np.clip(X[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    lon = np.degrees(np.arctan2(X[:, 1], X[:, 0]))
    fi = (lat + 90.0) / 180.0 * (n_lat - 1)
    fj = (lon + 180.0) / 360.0 * (n_lon - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, n_lat - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n_lon - 2)
    a, b = fi - i0, fj - j0
    return ((1 - a) * (1 - b) * R_map[i0, j0] + a * (1 - b) * R_map[i0 + 1, j0]
            + (1 - a) * b * R_map[i0, j0 + 1] + a * b * R_map[i0 + 1, j0 + 1])


def sample_ejecta(n, p_ej_mag, normal_i, v_esc, rng, v_min=0.05, v_max=50.0, mu=MU_SCALING, cone_deg=CONE_DEG):
    """Tracer launch velocities (relative to the surface, inertial axes) and the mass each represents."""
    k = 3.0 * mu
    u = rng.uniform(0.0, 1.0, n)
    tail = (v_max / v_min) ** (-k)
    v = v_min * (1.0 - u * (1.0 - tail)) ** (-1.0 / k)  # inverse CDF of the truncated power law
    nrm = normal_i / np.linalg.norm(normal_i)
    a = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(nrm, a)
    az = rng.uniform(0, 2 * np.pi, n)
    th = np.radians(cone_deg + rng.normal(0, 8.0, n))  # ejecta curtain ~45 deg from the normal
    dirs = (np.cos(th)[:, None] * nrm + np.sin(th)[:, None] * (np.cos(az)[:, None] * a + np.sin(az)[:, None] * b))
    vel = v[:, None] * dirs
    escaping = v > v_esc
    carried = np.sum(vel[escaping] @ nrm) if escaping.any() else 1.0
    mass_each = p_ej_mag / max(carried, 1e-9)  # escaping ejecta carry the recoil momentum
    return vel, mass_each


class EjectaCloud:
    """Ballistic ejecta tracers in Bennu's gravity, computed on a background thread after the impact.

    Positions are Bennu-centered, ICRF axes. frames: list of (time, positions (n, 3), state codes) with
    codes 0 = flying, 1 = landed, 2 = escaped (> 30 km), every 20 s for the first half hour, every minute to
    6 h, then every 15 min; `at(t)` interpolates between frames so the cloud moves smoothly at any time warp.
    t_computed: the latest computed time (the live run never runs ahead of it)."""

    def __init__(self, t_imp, r_imp_b, normal_b, p_ej_mag, grav, ray, n=800, seed=0, duration=2 * 86400.0):
        self.t0 = float(t_imp)
        self.grav = grav
        self.R_map = radius_map(ray)
        rng = np.random.default_rng(seed)
        R = bennu_attitude(t_imp)
        r_i, n_i = R @ r_imp_b, R @ normal_b
        g_surf = np.linalg.norm(grav.accel_body(r_imp_b + 0.5 * normal_b))
        r_mag = float(np.linalg.norm(r_imp_b))
        self.v_esc = float(np.sqrt(2.0 * g_surf * r_mag))  # local escape speed (spherical estimate)
        vel, self.mass_each = sample_ejecta(n, p_ej_mag, n_i, self.v_esc, rng)
        self.v_launch = np.linalg.norm(vel, axis=1)
        surf_v = np.cross(R @ OMEGA_B, r_i)  # launched from the rotating surface
        self.x = np.tile(r_i + 0.5 * n_i, (n, 1))
        self.v = vel + surf_v
        self.code = np.zeros(n, int)
        self.land_b = np.zeros((n, 3))
        self.total_mass = self.mass_each * n
        self.duration = duration
        self.frames = []
        self.t_computed = self.t0
        self.done = False
        self._stop = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="ejecta", daemon=True)
        self.thread.start()

    def _accel(self, t, X):
        R = bennu_attitude(t)
        return self.grav.accel_body_many(X @ R) @ R.T

    def _run(self):
        t = self.t0
        x, v = self.x.copy(), self.v.copy()
        code = self.code.copy()
        next_frame = t
        while t - self.t0 < self.duration and not self._stop:
            el = t - self.t0
            h = 20.0 if el < 1800 else (120.0 if el < 6 * 3600 else 600.0)
            fly = code == 0
            if fly.any():
                xf, vf = x[fly], v[fly]
                k1v = self._accel(t, xf)
                k2v = self._accel(t + h / 2, xf + h / 2 * vf)
                k3v = self._accel(t + h / 2, xf + h / 2 * (vf + h / 2 * k1v))
                k4v = self._accel(t + h, xf + h * (vf + h / 2 * k2v))
                x[fly] = xf + h * vf + h * h / 6 * (k1v + k2v + k3v)
                v[fly] = vf + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            x[code == 2] += h * v[code == 2]  # escaped: straight line
            t += h
            # landing: below the local surface radius (body frame)
            R = bennu_attitude(t)
            fly = code == 0
            Xb = x[fly] @ R
            descending = np.einsum("ij,ij->i", x[fly], v[fly]) < 0.0
            landed = descending & (np.linalg.norm(Xb, axis=1) < surface_radius(self.R_map, Xb))
            idx = np.where(fly)[0]
            code[idx[landed]] = 1
            self.land_b[idx[landed]] = Xb[landed]
            v[idx[landed]] = 0.0
            far = (code == 0) & (np.linalg.norm(x, axis=1) > 30e3)
            code[far] = 2
            if t >= next_frame:
                pos = x.copy()
                lb = code == 1
                pos[lb] = self.land_b[lb] @ R.T  # landed tracers ride on the rotating surface
                with self._lock:
                    self.frames.append((t, pos.astype(np.float32), code.copy()))
                next_frame = t + (20.0 if el < 1800 else 60.0 if el < 6 * 3600 else 900.0)
            self.t_computed = t
            if not (code == 0).any() and t - self.t0 > 6 * 3600:
                break
        self.t_computed = self.t0 + self.duration
        self.done = True

    def stop(self):
        self._stop = True

    def at(self, t):
        """Tracer positions and codes at time t (interpolated between the computed frames), or None if not yet
        available. Landed tracers ride on the rotating surface."""
        with self._lock:
            if not self.frames or t < self.t0:
                return None
            times = [f[0] for f in self.frames]
            k = int(np.searchsorted(times, t))
            f1 = self.frames[min(k, len(times) - 1)]
            f0 = self.frames[max(k - 1, 0)]
        if k == 0 or k >= len(times):  # before the first or beyond the last frame
            pos, code = f1[1].astype(float), f1[2]
        else:
            w = (t - f0[0]) / max(f1[0] - f0[0], 1e-9)
            code = f1[2].copy()
            pos = (1 - w) * f0[1].astype(float) + w * f1[1].astype(float)
            lands = (f0[2] == 0) & (f1[2] == 1)  # touches down between the frames: shown landed from then on
            code[lands & (w < 0.5)] = 0
        lb = code == 1
        if lb.any():
            pos[lb] = self.land_b[lb] @ bennu_attitude(t).T
        return pos, code

    def summary(self):
        with self._lock:
            if not self.frames:
                return None
            code = self.frames[-1][2]
        n = len(code)
        return {"escaped": float(np.mean(code == 2)), "landed": float(np.mean(code == 1)),
                "flying": float(np.mean(code == 0)), "total_mass": self.total_mass, "v_esc": self.v_esc, "n": n}
