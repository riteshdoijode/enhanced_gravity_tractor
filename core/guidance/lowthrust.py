"""Thrust-limited heliocentric transfers by direct transcription (Sims & Flanagan style).

The flight [t0, tf] is split into N equal segments; segment i carries one velocity impulse dv_i at its
midpoint (flown as the constant acceleration dv_i / dt), with exact Kepler arcs in between.
Unknowns: dv_0..dv_{N-1}. Constraint: final state = target state, F(dv) = x_N - x_T = 0 (6 equations).

Minimum-energy solution (sum |dv_i|^2, i.e. integral of a^2 dt) by damped Newton / minimum-norm least
squares: with J = dx_N/d(dv) (6 x 3N, a chain of segment state-transition matrices),
    dv <- J^T (J J^T)^-1 (J dv - F)          (projection onto the linearized constraint, min-norm)
Thrust limit: feasible if max_i |dv_i| / dt <= a_max. "Fastest" = earliest feasible arrival.
Everything is vectorized over K candidate arrival times (one batched Newton solve).
In flight the problem is re-solved from the current state, warm-started from the previous profile
(receding horizon), so N-body perturbations, throttle changes and outages are corrected.
"""

import numpy as np

from core.constants import AU, DAY
from core.guidance import kepler
from core.guidance.kepler import MU_SUN, ecliptic_pole_icrf

POLE = ecliptic_pole_icrf()
SCALE = np.array([1e6, 1e6, 1e6, 1.0, 1.0, 1.0])  # residual scaling: 1000 km ~ 1 m/s
EPS = np.array([10.0, 10.0, 10.0, 1e-4, 1e-4, 1e-4])


def _fd_stm(X, dt, mu=MU_SUN):
    """Finite-difference Kepler state-transition matrices for many arcs: X (n, 6), dt (n,) -> (n, 6, 6)."""
    n = len(X)
    pert = np.concatenate([X[:, None, :] + np.diag(EPS)[None], X[:, None, :] - np.diag(EPS)[None]], axis=1)
    rr, vv = kepler.propagate(pert[..., :3].reshape(-1, 3), pert[..., 3:].reshape(-1, 3), np.repeat(dt, 12), mu)
    out = np.concatenate([rr, vv], axis=1).reshape(n, 12, 6)
    return np.transpose((out[:, :6] - out[:, 6:]) / (2 * EPS)[None, :, None], (0, 2, 1))


def _forward(x0, dv, dt, mu):
    """x0 (K, 6), dv (K, N, 3), dt (K,) -> node states (K, N+1, 6), pre-impulse midpoint states (K, N, 6)."""
    K, N, _ = dv.shape
    nodes = np.empty((K, N + 1, 6))
    mids = np.empty((K, N, 6))
    nodes[:, 0] = x0
    r, v = x0[:, :3], x0[:, 3:]
    for i in range(N):
        r, v = kepler.propagate(r, v, dt / 2, mu)
        mids[:, i, :3], mids[:, i, 3:] = r, v
        v = v + dv[:, i]
        r, v = kepler.propagate(r, v, dt / 2, mu)
        nodes[:, i + 1, :3], nodes[:, i + 1, 3:] = r, v
    return nodes, mids


def lambert_guess(x0, t0, xT, tf, N, mu=MU_SUN):
    """Initial impulses from the cheapest 0-2 revolution Lambert arc: first and last segment."""
    K = len(tf)
    V1, V2 = kepler.lambert_multirev(np.broadcast_to(x0[:, :3], (K, 3)), xT[:, :3], tf - t0, n_max=2, mu=mu,
                                     prograde_axis=POLE)
    cost = np.linalg.norm(V1 - x0[None, :, 3:], axis=-1) + np.linalg.norm(xT[None, :, 3:] - V2, axis=-1)
    j = np.argmin(np.where(np.isfinite(cost), cost, np.inf), axis=0)
    cols = np.arange(K)
    dv = np.zeros((K, N, 3))
    dv[:, 0] = V1[j, cols] - x0[:, 3:]
    dv[:, -1] = xT[:, 3:] - V2[j, cols]
    return np.nan_to_num(dv)


def solve_batch(x0, t0, xT, tf, N=24, dv_init=None, max_iter=20, tol=(1e3, 0.05), mu=MU_SUN):
    """Batched minimum-energy solve. x0 (6,) or (K,6); xT (K,6); tf (K,). Returns dict of arrays."""
    tf = np.atleast_1d(np.asarray(tf, float))
    K = len(tf)
    x0 = np.broadcast_to(np.asarray(x0, float), (K, 6)).copy()
    xT = np.asarray(xT, float).reshape(K, 6)
    dt = (tf - t0) / N
    dv = lambert_guess(x0, t0, xT, tf, N, mu) if dv_init is None else np.array(dv_init, float).reshape(K, N, 3)
    nodes, mids = _forward(x0, dv, dt, mu)
    F = (nodes[:, -1] - xT) / SCALE
    fn = np.linalg.norm(F, axis=1)
    conv = np.zeros(K, bool)
    for _ in range(max_iter):
        act = ~conv & np.isfinite(fn)
        if not act.any():
            break
        idx = np.where(act)[0]
        k = len(idx)
        post = mids[idx].copy()
        post[..., 3:] += dv[idx]
        S_b = _fd_stm(post.reshape(-1, 6), np.repeat(dt[idx] / 2, N), mu).reshape(k, N, 6, 6)
        S_a = _fd_stm(nodes[idx, 1:-1].reshape(-1, 6), np.repeat(dt[idx] / 2, N - 1), mu).reshape(k, N - 1, 6, 6)
        J = np.empty((k, 6, 3 * N))
        P = np.broadcast_to(np.eye(6), (k, 6, 6)).copy()
        for i in range(N - 1, -1, -1):
            Pm = P @ S_b[:, i]
            J[:, :, 3 * i:3 * i + 3] = Pm[:, :, 3:]
            if i > 0:
                P = Pm @ S_a[:, i - 1]
        Js = J / SCALE[None, :, None]
        rhs = np.einsum("kij,kj->ki", Js, dv[idx].reshape(k, -1)) - F[idx]
        JJ = Js @ np.transpose(Js, (0, 2, 1))
        JJ += 1e-12 * np.trace(JJ, axis1=1, axis2=2)[:, None, None] * np.eye(6)  # tiny regularization
        bad_j = ~np.all(np.isfinite(JJ), axis=(1, 2)) | ~np.all(np.isfinite(rhs), axis=1)
        JJ[bad_j], rhs[bad_j] = np.eye(6), 0.0
        lam = np.linalg.solve(JJ, rhs[..., None])[..., 0]
        target = np.einsum("kji,kj->ki", Js, lam).reshape(k, N, 3)
        step = np.ones(k)
        best_dv, best_nodes, best_mids, best_F = dv[idx], nodes[idx], mids[idx], F[idx]
        pending = np.ones(k, bool)
        for _ls in range(6):  # damped Newton: halve the step where the residual would blow up
            trial = dv[idx] + step[:, None, None] * (target - dv[idx])
            n_t, m_t = _forward(x0[idx], trial, dt[idx], mu)
            F_t = (n_t[:, -1] - xT[idx]) / SCALE
            good = pending & np.all(np.isfinite(F_t), axis=1) & (np.linalg.norm(F_t, axis=1) < 1.5 * fn[idx] + 1e-9)
            best_dv[good], best_nodes[good], best_mids[good], best_F[good] = trial[good], n_t[good], m_t[good], F_t[good]
            pending &= ~good
            if not pending.any():
                break
            step[pending] *= 0.5
        dv[idx], nodes[idx], mids[idx], F[idx] = best_dv, best_nodes, best_mids, best_F
        fn[idx] = np.linalg.norm(F[idx], axis=1)
        err_r = np.linalg.norm(F[:, :3] * SCALE[:3], axis=1)
        err_v = np.linalg.norm(F[:, 3:], axis=1)
        conv = (err_r < tol[0]) & (err_v < tol[1])
    acc = np.linalg.norm(dv, axis=2) / dt[:, None]
    return {"tf": tf, "dt": dt, "dv": dv, "nodes": nodes, "converged": conv, "peak": acc.max(axis=1),
            "total_dv": np.linalg.norm(dv, axis=2).sum(axis=1),
            "min_r_sun": np.linalg.norm(nodes[..., :3], axis=2).min(axis=1)}


def _pick(sol, j):
    K = len(sol["tf"])
    return {k: (v[j:j + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (K,) else v) for k, v in sol.items()}


def fastest(x0, t0, target, a_max, t_lo, t_hi, N=20, n_scan=24, refine=6, margin=0.9, min_r_sun=0.3 * AU):
    """Earliest arrival tf in [t0 + t_lo, t0 + t_hi] whose minimum-energy profile fits the thrust.

    Multi-start: low-thrust transfers are multimodal, so every candidate arrival time is seeded from
    each 0-2 revolution Lambert branch (left/right) and the earliest feasible solution is refined.
    target(t_array) -> (K, 6) heliocentric target states. Returns (solution dict for 1 tf, tf) or (None, nan).
    """
    def ok(sol):
        return sol["converged"] & (sol["peak"] <= margin * a_max) & (sol["min_r_sun"] >= min_r_sun)

    cands = t0 + np.geomspace(t_lo, t_hi, n_scan)
    xT = target(cands)
    x0 = np.asarray(x0, float)
    V1, V2 = kepler.lambert_multirev(np.broadcast_to(x0[:3], (n_scan, 3)), xT[:, :3], cands - t0, n_max=2,
                                     prograde_axis=POLE)
    inits, tfs, xts = [], [], []
    for j in range(V1.shape[0]):
        valid = np.isfinite(V1[j, :, 0])
        dv = np.zeros((valid.sum(), N, 3))
        dv[:, 0] = V1[j, valid] - x0[3:]
        dv[:, -1] = xT[valid, 3:] - V2[j, valid]
        inits.append(dv), tfs.append(cands[valid]), xts.append(xT[valid])
    tf_all = np.concatenate(tfs)
    sol = solve_batch(x0, t0, np.concatenate(xts), tf_all, N, dv_init=np.concatenate(inits))
    good = ok(sol)
    if not good.any():
        return None, np.nan
    j = int(np.argmin(np.where(good, tf_all, np.inf)))
    best = _pick(sol, j)
    earlier = cands[cands < tf_all[j]]
    lo = earlier[-1] if len(earlier) else t0 + t_lo  # never refine below the requested earliest arrival
    hi = tf_all[j]
    if hi - lo < 60.0:
        return best, float(best["tf"][0])
    for _ in range(refine):  # bisection, warm-started from the best feasible profile
        mid = 0.5 * (lo + hi)
        s = solve_batch(x0, t0, target(np.array([mid])), np.array([mid]), N, dv_init=best["dv"])
        if ok(s)[0]:
            best, hi = s, mid
        else:
            lo = mid
    return best, float(best["tf"][0])


class LowThrustGuidance:
    """Receding-horizon flight of a minimum-energy transfer (warm-started re-solves)."""

    def __init__(self, target, a_max, N=24):
        self.target, self.a_max, self.N = target, float(a_max), N
        self.t_f = np.nan
        self.t0 = np.nan
        self.dt = np.nan
        self.dv = None

    def plan(self, t, x, t_lo=2 * DAY, t_hi=12 * 365.25 * DAY):
        sol, tf = fastest(x, t, self.target, self.a_max, t_lo, t_hi, self.N)
        if sol is None:
            self.t_f = np.nan
            return np.nan
        self._adopt(t, sol)
        return tf

    def _adopt(self, t, sol):
        self.t0, self.t_f, self.dt, self.dv = t, float(sol["tf"][0]), float(sol["dt"][0]), sol["dv"][0].copy()

    def _resampled(self, t, N):
        """Remaining profile mapped onto N new segments over [t, t_f] (warm start)."""
        new_dt = (self.t_f - t) / N
        mids = t + (np.arange(N) + 0.5) * new_dt
        k = np.clip(((mids - self.t0) / self.dt).astype(int), 0, len(self.dv) - 1)
        return (self.dv[k] / self.dt * new_dt)[None]

    def update(self, t, x):
        """Re-solve from the current state toward the same arrival time (warm start).

        Far from arrival the tolerance is loose (refined as t_go shrinks) and a peak up to 1.2 a_max is
        accepted (flown saturated). Returns True if adopted; the caller re-plans after repeated failures.
        """
        if not np.isfinite(self.t_f) or self.t_f - t < 60.0:
            return False
        t_go = self.t_f - t
        N = int(np.clip(round(t_go / max(self.dt, 1.0)), 4, self.N))
        tol = (5e3, 0.2) if t_go > 30 * DAY else (1e3, 0.05)
        sol = solve_batch(x, t, self.target(np.array([self.t_f])), np.array([self.t_f]), N,
                          dv_init=self._resampled(t, N), max_iter=12, tol=tol)
        if sol["converged"][0] and sol["peak"][0] <= 1.2 * self.a_max:
            self._adopt(t, sol)
            return True
        return False

    def accel(self, t):
        """Commanded acceleration at time t from the current profile (zero after arrival)."""
        if self.dv is None or not (self.t0 <= t < self.t_f):
            return np.zeros(3)
        k = min(int((t - self.t0) / self.dt), len(self.dv) - 1)
        a = self.dv[k] / self.dt
        n = np.linalg.norm(a)
        return a if n <= self.a_max else a / n * self.a_max
