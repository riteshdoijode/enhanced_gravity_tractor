"""Bennu: heliocentric dynamics (barycentric N-body, lecture eq. 2) and attitude.

Acceleration = planets + Moon + 4 largest asteroids (JPL Horizons) + Earth J2 during close
             approaches + solar GR + solar radiation pressure + Yarkovsky (transverse A2 model)
             [+ gravity-tractor pull + impulses, Steps 6-7].
"""

import json

import numpy as np
from scipy.integrate import solve_ivp

from core.constants import AU, BENNU, DATA_DIR, DAY
from core.dynamics.gravity import ALL_BODIES, NBody, j2, schwarzschild
from core.dynamics.perturbations import srp, yarkovsky

LIT_A2 = -4.618e-14  # au/d^2, Chesley et al. (2014) Icarus 235:5 (da/dt = -284 m/yr)
LIT_AMRAT = 2.635943157e-6  # m^2/kg, JPL SBDB orbit solution 118 (SRP area-to-mass ratio)
FIT_FILE = DATA_DIR / "bennu_nongrav_fit.json"


def _fitted():
    if FIT_FILE.exists():
        f = json.loads(FIT_FILE.read_text())
        return f["a2"], f["amrat"]
    return LIT_A2, LIT_AMRAT


# Defaults: A2 and area-to-mass ratio re-estimated within this force model (scripts/fit_bennu.py)
BENNU_A2, BENNU_AMRAT = _fitted()


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


# Bennu's spin state in this process: nominal until a kinetic impact changes the rotation rate
# (set by the live mission; reset when a new mission starts). Rotation phase W(t) stays continuous.
_SPIN = {"t": np.inf, "W": 0.0, "omega": BENNU["omega"]}


def set_spin_change(t, omega_new):
    _SPIN.update(t=float(t), W=float(BENNU["omega"] * t), omega=float(omega_new))


def reset_spin():
    _SPIN.update(t=np.inf, W=0.0, omega=BENNU["omega"])


def spin_rate(t):
    return _SPIN["omega"] if t >= _SPIN["t"] else BENNU["omega"]


def rotation_angle(t):
    if t >= _SPIN["t"]:
        return _SPIN["W"] + _SPIN["omega"] * (t - _SPIN["t"])
    return BENNU["omega"] * t


def bennu_attitude(t):
    """Body-fixed -> ICRF rotation, IAU convention R = Rz(a+90deg) Rx(90deg-d) Rz(W).

    Pole (RA, Dec) and spin rate from SBDB. The prime-meridian angle W at J2000 is not
    modeled (set to 0); only the spin rate matters for the dynamics. After an impact the rate
    changes (set_spin_change); the pole tilt it would also cause is reported, not simulated.
    """
    a, d = np.radians(BENNU["pole_ra_deg"]), np.radians(BENNU["pole_dec_deg"])
    return rot_z(a + np.pi / 2) @ rot_x(np.pi / 2 - d) @ rot_z(rotation_angle(t))


def omega_body(t):
    """Spin vector in the body frame [rad/s]."""
    return np.array([0.0, 0.0, spin_rate(t)])


def spin_vector_icrf():
    """Bennu's angular velocity vector in ICRF [rad/s]."""
    a, d = np.radians(BENNU["pole_ra_deg"]), np.radians(BENNU["pole_dec_deg"])
    return BENNU["omega"] * np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


class BennuDynamics:
    """Barycentric equations of motion for Bennu; state x = [r (3), v (3)] in m, m/s."""

    def __init__(self, eph, bodies=ALL_BODIES, gr=True, solar_pressure=True, a2=BENNU_A2,
                 amrat=BENNU_AMRAT, earth_j2=True, extra_accel=None):
        self.eph = eph
        self.grav = NBody(eph, central="ssb", bodies=bodies)
        self.gr, self.solar_pressure, self.a2, self.amrat = gr, solar_pressure, a2, amrat
        self.earth_j2 = earth_j2
        self.extra_accel = extra_accel  # callable (t, r, v) -> a, e.g. averaged gravity-tractor pull

    def accel(self, t, r, v):
        a = self.grav.accel(t, r)
        rs, vs = r - self.eph.pos("sun", t), v - self.eph.vel("sun", t)
        if self.gr:
            a = a + schwarzschild(rs, vs)
        if self.solar_pressure:
            a = a + srp(rs, self.amrat)
        if self.a2:
            a = a + yarkovsky(rs, vs, self.a2)
        if self.earth_j2:
            re = r - self.eph.pos("earth", t)
            if re @ re < (0.05 * AU) ** 2:  # only matters during close approaches
                a = a + j2(re)
        if self.extra_accel is not None:
            a = a + self.extra_accel(t, r, v)
        return a

    def rhs(self, t, x):
        return np.concatenate([x[3:6], self.accel(t, x[:3], x[3:6])])

    def rhs_stm(self, t, y):
        """State + 6x6 state-transition matrix: Phi' = [[0, I], [G(r), 0]] Phi (gravity gradient G)."""
        x, Phi = y[:6], y[6:].reshape(6, 6)
        G = self.grav.gradient(t, x[:3])
        dPhi = np.vstack([Phi[3:6], G @ Phi[0:3]])
        return np.concatenate([self.rhs(t, x), dPhi.ravel()])

    def propagate(self, t0, x0, t_eval, stm=False, rtol=1e-12, atol=1e-6):
        """Integrate from t0 through the (sorted) times t_eval with DOP853. Returns (x [N,6], Phi [N,6,6] | None)."""
        t_eval = np.atleast_1d(np.asarray(t_eval, float))
        y0 = np.concatenate([x0, np.eye(6).ravel()]) if stm else np.asarray(x0, float)
        sol = solve_ivp(self.rhs_stm if stm else self.rhs, (t0, t_eval[-1]), y0, method="DOP853",
                        t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y.T
        return (y[:, :6], y[:, 6:].reshape(-1, 6, 6)) if stm else (y, None)


def fit_nongrav(eph, t0, t1, step=30 * DAY, iterations=3, verbose=True):
    """Least-squares estimate of (A2, AMRAT) so our force model reproduces JPL's trajectory.

    JPL's Bennu solution uses 343 perturbing asteroids and a thermophysical Yarkovsky model; our
    simpler model absorbs those differences into the two non-gravitational parameters (Gauss-Newton
    with finite-difference Jacobian; observations = JPL positions every `step` from t0 to t1).
    """
    x0 = eph.state("bennu", t0)
    t_obs = np.arange(t0 + step, t1, step)
    obs = eph.pos("bennu", t_obs)

    def residual(p):
        xs, _ = BennuDynamics(eph, a2=p[0], amrat=p[1]).propagate(t0, x0, t_obs, rtol=1e-11)
        return (xs[:, :3] - obs).ravel()

    p = np.array([LIT_A2, LIT_AMRAT])
    dp_fd = np.array([1e-15, 1e-7])
    history = []
    for _ in range(iterations):
        r0 = residual(p)
        J = np.column_stack([(residual(p + d) - r0) / d.sum() for d in np.diag(dp_fd)])
        p = p + np.linalg.lstsq(J, -r0, rcond=None)[0]
        history.append(float(np.sqrt(np.mean(r0**2))))
        if verbose:
            print(f"  rms {history[-1] / 1e3:8.3f} km -> A2 {p[0]:.5e} au/d^2, AMRAT {p[1]:.5e} m^2/kg")
    rms = float(np.sqrt(np.mean(residual(p) ** 2)))
    return {"a2": float(p[0]), "amrat": float(p[1]), "rms_m": rms, "rms_history_m": history,
            "fit_window": [float(t0), float(t1)], "literature": {"a2": LIT_A2, "amrat": LIT_AMRAT}}
