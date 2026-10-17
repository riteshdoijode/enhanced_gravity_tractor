"""Compiled (numba) polyhedron gravity for proximity operations, plus the gravity-gradient tensor.

Same Werner & Scheeres (1997) formulas as core/dynamics/polyhedron.py (the NumPy reference), evaluated
one field point at a time in a single pass over edges and faces (~0.1 ms instead of ~3 ms), so the
closed-loop hover/descent simulation can call it every RK4 stage. With r = (vertex - x):

    a(x)      = -G sigma sum_e E_e r_e L_e + G sigma sum_f F_f r_f w_f
    da/dx     =  G sigma sum_e E_e L_e     - G sigma sum_f F_f w_f        (gravity-gradient tensor)
    trace     = -G sigma sum_f w_f          (= -4 pi G sigma inside, 0 outside)

The gradient is what the hover linearization (A matrix) and the navigation filter's Jacobian need.
Beyond FAR_FIELD the body is replaced by a point mass. The kernels release the GIL; accel_body_many
splits batches of points (MPC look-ahead, ejecta particles) across the worker threads.
"""

import numpy as np
from numba import njit

from core import parallel
from core.assets import bennu_physics_mesh
from core.constants import BENNU
from core.dynamics.asteroid import bennu_attitude
from core.dynamics.polyhedron import Polyhedron

FAR_FIELD = 3e3  # m (point-mass error there 3e-10 m/s^2, far below solar pressure)


@njit(cache=True, fastmath=True, nogil=True)
def _poly_eval(x, V, edges, edge_len, E, F, Fd, k, want_grad):
    nv = V.shape[0]
    r = np.empty((nv, 3))
    d = np.empty(nv)
    for i in range(nv):
        r0, r1, r2 = V[i, 0] - x[0], V[i, 1] - x[1], V[i, 2] - x[2]
        r[i, 0], r[i, 1], r[i, 2] = r0, r1, r2
        d[i] = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    acc = np.zeros(3)
    grad = np.zeros((3, 3))
    pot = 0.0
    for e in range(edges.shape[0]):
        i, j = edges[e, 0], edges[e, 1]
        s = d[i] + d[j]
        L = np.log((s + edge_len[e]) / (s - edge_len[e]))
        for a in range(3):
            Er = E[e, a, 0] * r[i, 0] + E[e, a, 1] * r[i, 1] + E[e, a, 2] * r[i, 2]
            acc[a] -= Er * L
            pot += 0.5 * r[i, a] * Er * L
            if want_grad:
                for b in range(3):
                    grad[a, b] += E[e, a, b] * L
    wsum = 0.0
    for f in range(F.shape[0]):
        i1, i2, i3 = F[f, 0], F[f, 1], F[f, 2]
        c0 = r[i2, 1] * r[i3, 2] - r[i2, 2] * r[i3, 1]
        c1 = r[i2, 2] * r[i3, 0] - r[i2, 0] * r[i3, 2]
        c2 = r[i2, 0] * r[i3, 1] - r[i2, 1] * r[i3, 0]
        num = r[i1, 0] * c0 + r[i1, 1] * c1 + r[i1, 2] * c2
        d23 = r[i2, 0] * r[i3, 0] + r[i2, 1] * r[i3, 1] + r[i2, 2] * r[i3, 2]
        d31 = r[i3, 0] * r[i1, 0] + r[i3, 1] * r[i1, 1] + r[i3, 2] * r[i1, 2]
        d12 = r[i1, 0] * r[i2, 0] + r[i1, 1] * r[i2, 1] + r[i1, 2] * r[i2, 2]
        den = d[i1] * d[i2] * d[i3] + d[i1] * d23 + d[i2] * d31 + d[i3] * d12
        w = 2.0 * np.arctan2(num, den)
        wsum += w
        for a in range(3):
            Fr = Fd[f, a, 0] * r[i1, 0] + Fd[f, a, 1] * r[i1, 1] + Fd[f, a, 2] * r[i1, 2]
            acc[a] += Fr * w
            pot -= 0.5 * r[i1, a] * Fr * w
            if want_grad:
                for b in range(3):
                    grad[a, b] -= Fd[f, a, b] * w
    return k * acc, k * grad, -k * wsum, k * pot


@njit(cache=True, fastmath=True, nogil=True)
def _poly_accel_batch(X, V, edges, edge_len, E, F, Fd, k, gm, far, out):
    """Accelerations at many body-fixed points (point mass beyond `far`), written into out (n, 3)."""
    for i in range(X.shape[0]):
        x = X[i]
        r = np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        if r > far:
            c = -gm / r**3
            out[i, 0], out[i, 1], out[i, 2] = c * x[0], c * x[1], c * x[2]
        else:
            out[i] = _poly_eval(x, V, edges, edge_len, E, F, Fd, k, False)[0]


class BennuGravity:
    """Bennu's gravity in its body frame (polyhedron) and in the Bennu-centered ICRF frame."""

    def __init__(self, gm=BENNU["gm"]):
        self.poly = Polyhedron(*bennu_physics_mesh(), gm=gm)
        p = self.poly
        self._args = (np.ascontiguousarray(p.V), np.ascontiguousarray(p.edges, dtype=np.int64),
                      np.ascontiguousarray(p.edge_len), np.ascontiguousarray(p.E_dyad),
                      np.ascontiguousarray(p.F, dtype=np.int64), np.ascontiguousarray(p.F_dyad), float(p.k))
        self.gm = gm
        self.R_bound = float(np.linalg.norm(p.V, axis=1).max())

    # ---- body frame
    def eval_body(self, xb, want_grad=True):
        """(acceleration (3,), gradient (3,3), laplacian, potential) at body-fixed point xb."""
        xb = np.ascontiguousarray(xb, dtype=float)
        r = np.linalg.norm(xb)
        if r > FAR_FIELD:
            a = -self.gm * xb / r**3
            G = self.gm / r**3 * (3.0 * np.outer(xb, xb) / r**2 - np.eye(3))
            return a, G, 0.0, self.gm / r
        return _poly_eval(xb, *self._args, want_grad)

    def accel_body(self, xb):
        return self.eval_body(xb, False)[0]

    def gradient_body(self, xb):
        return self.eval_body(xb)[1]

    def accel_body_many(self, X):
        """Accelerations at many body-fixed points (n, 3), split across the worker threads."""
        X = np.ascontiguousarray(X, dtype=float).reshape(-1, 3)
        out = np.empty_like(X)
        parallel.chunked(lambda a, b: _poly_accel_batch(X[a:b], *self._args, self.gm, FAR_FIELD, out[a:b]),
                         len(X), min_chunk=8)
        return out

    def accel_many(self, ts, P):
        """Inertial accelerations at positions P (n, 3) and times ts (n,) (Bennu rotates between them)."""
        R = np.array([bennu_attitude(t) for t in np.atleast_1d(ts)])
        Xb = np.einsum("nji,nj->ni", R, np.asarray(P, float).reshape(-1, 3))
        return np.einsum("nij,nj->ni", R, self.accel_body_many(Xb))

    # ---- Bennu-centered inertial (ICRF axes)
    def accel(self, t, rho):
        R = bennu_attitude(t)
        return R @ self.accel_body(R.T @ rho)

    def accel_gradient(self, t, rho):
        R = bennu_attitude(t)
        a, G, _, _ = self.eval_body(R.T @ rho)
        return R @ a, R @ G @ R.T
