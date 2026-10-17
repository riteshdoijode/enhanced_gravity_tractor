"""Bennu's nominal trajectory and state-transition matrix (STM), precomputed and cached.

The STM Phi(t, t0) = d x(t) / d x(t0) of the variational equations  Phi' = [[0, I], [G, 0]] Phi
(G = gravity gradient) linearizes the deflection problem (lecture L2/L3 linearization):

    dx(t2) = Phi(t2, t1) dx(t1),     Phi(t2, t1) = Phi(t2, t0) Phi(t1, t0)^-1

so the shift of Bennu at a future encounter caused by any small push dv at any time is one 3x3
matrix-vector product. Phi is symplectic (gravity-only variational equations are Hamiltonian), so
the inverse is exact and cheap:  Phi^-1 = -J Phi^T J.
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar

from core.constants import BENNU_CLOSE_APPROACHES, DATA_DIR, DAY, EPHEM_START, EPHEM_STOP
from core.dynamics.asteroid import BennuDynamics
from core.timeutil import iso_to_t

NOMINAL_FILE = DATA_DIR / "bennu_nominal.npz"
J6 = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


def symplectic_inverse(Phi):
    return -J6 @ Phi.T @ J6


def _node_times(t0, t1):
    nodes = np.arange(t0, t1 + 0.5 * DAY, DAY)
    for date, _, _ in BENNU_CLOSE_APPROACHES:  # hourly within +-10 d of each Earth approach
        tc = iso_to_t(date)
        nodes = np.concatenate([nodes, np.arange(tc - 10 * DAY, tc + 10 * DAY, 3600.0)])
    nodes = np.unique(np.round(nodes, 3))
    return nodes[(nodes >= t0) & (nodes <= t1)]


def compute_nominal(eph, path=NOMINAL_FILE, rtol=1e-11):
    """Propagate Bennu + STM from the start of the ephemeris to its end; ~2 min, cached to disk."""
    dyn = BennuDynamics(eph)
    t0 = iso_to_t(EPHEM_START)
    t1 = iso_to_t(EPHEM_STOP) - DAY
    nodes = _node_times(t0, t1)
    x, Phi = dyn.propagate(t0, eph.state("bennu", t0), nodes, stm=True, rtol=rtol, atol=1e-9)
    xdot = np.array([dyn.rhs(t, xi) for t, xi in zip(nodes, x)])
    Phidot = np.empty_like(Phi)
    for k, (t, xi, P) in enumerate(zip(nodes, x, Phi)):
        G = dyn.grav.gradient(t, xi[:3])
        Phidot[k] = np.vstack([P[3:6], G @ P[0:3]])
    np.savez(path, t=nodes, x=x, xdot=xdot, Phi=Phi.reshape(-1, 36), Phidot=Phidot.reshape(-1, 36),
             a2=dyn.a2, amrat=dyn.amrat)
    return path


class NominalBennu:
    def __init__(self, eph=None, path=NOMINAL_FILE):
        if not path.exists():
            if eph is None:
                raise FileNotFoundError(f"{path} missing; run `uv run python -m scripts.precompute`")
            compute_nominal(eph, path)
        d = np.load(path)
        self.t = d["t"]
        self._x = CubicHermiteSpline(self.t, d["x"], d["xdot"], axis=0)
        self._Phi = CubicHermiteSpline(self.t, d["Phi"], d["Phidot"], axis=0)
        self.a2, self.amrat = float(d["a2"]), float(d["amrat"])

    @property
    def span(self):
        return float(self.t[0]), float(self.t[-1])

    def state(self, t):
        """Nominal barycentric state [r, v] of Bennu (our propagation), shape (..., 6)."""
        return self._x(t)

    def Phi0(self, t):
        """Phi(t, t_start)."""
        return self._Phi(t).reshape(*np.shape(t), 6, 6)

    def Phi(self, t2, t1):
        """Phi(t2, t1) = Phi(t2, t0) Phi(t1, t0)^-1."""
        return self.Phi0(t2) @ symplectic_inverse(self.Phi0(t1))

    def dr_from_dv(self, t2, t1, dv):
        """Linear position change at t2 from an impulsive velocity change dv at t1."""
        return self.Phi(t2, t1)[:3, 3:] @ dv

    def close_approach(self, eph, date, window_days=2.0):
        """Time of minimum Earth distance near `date`, with the relative state at that instant."""
        tc = iso_to_t(date)

        def dist(t):
            return np.linalg.norm(self.state(t)[:3] - eph.pos("earth", t))

        res = minimize_scalar(dist, bounds=(tc - window_days * DAY, tc + window_days * DAY),
                              method="bounded", options={"xatol": 1.0})
        t = float(res.x)
        x = self.state(t)
        return t, x[:3] - eph.pos("earth", t), x[3:] - eph.vel("earth", t)

    @staticmethod
    def miss_shift(dr, v_rel):
        """First-order change of the encounter miss vector: project onto the plane normal to the
        Earth-relative velocity (modified target plane). Returns (in-plane shift, timing shift [s])."""
        s = v_rel / np.linalg.norm(v_rel)
        return dr - (dr @ s) * s, -(dr @ s) / np.linalg.norm(v_rel)
