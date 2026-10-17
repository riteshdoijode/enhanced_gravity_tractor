"""Proximity-operations controllers near Bennu: LQR (lecture 3) and constrained MPC (cvxpy).

Error dynamics about a reference trajectory x_ref(t) flown with feedforward thrust u_ff(t):
    inertial frame (hover):      d/dt [dp, dv] = [[0, I], [G, 0]] [dp, dv] + [0, I] du
    body frame (descent/ascent): d/dt [dp, dv] = [[0, I], [G - [w]x^2, -2 [w]x]] [dp, dv] + [0, I] du
G = gravity-gradient tensor at the reference (point mass: mu/d^3 (3 e e^T - I), eigenvalues
+-sqrt(2 mu/d^3) (unstable radial mode) and +-i sqrt(mu/d^3)). Discretized with zero-order hold.

LQR: du = -K dx, K from the DARE (lecture 3). Its command is then forced into the thrust set
(projection), which is all a linear controller can do with constraints.
MPC: at every control step solve
    min  sum_k dx_k^T Q dx_k + du_k^T R du_k + dx_N^T P dx_N          (P = DARE solution)
    s.t. dx_{k+1} = A dx_k + B du_k
         u_k = u_ff,k + du_k,   |u_k| <= a_max,   angle(u_k, e_k) <= gamma      (thrust cone)
         e_k^T p_k >= r_keepout         (hover: stay outside a keep-out sphere)       [soft]
         |p_k - s| cos(theta) <= n^T (p_k - s)   (descent: glide-slope cone over the site) [soft]
The thrust cone models the gravity-tractor thruster cluster: engines canted outward so their plumes miss
the surface (Lu & Love 2005), so the net force can only point within gamma of the outward axis e;
moving inward means throttling down and letting gravity act.
"""

import warnings

import cvxpy as cp
import numpy as np

from core.control.dare import c2d, dare


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def relative_model(G, dt, omega=None):
    """Discrete (A, B) of the error dynamics; omega (body spin vector) selects the rotating frame."""
    Ac = np.zeros((6, 6))
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = G
    if omega is not None:
        W = skew(omega)
        Ac[3:, :3] = G - W @ W
        Ac[3:, 3:] = -2.0 * W
    Bc = np.vstack([np.zeros((3, 3)), np.eye(3)])
    return c2d(Ac, Bc, dt)


def weights(omega_n):
    """LQR/MPC weights giving a closed-loop bandwidth ~ omega_n [rad/s] (double-integrator rule
    omega_n = (q/r)^(1/4)), critically damped."""
    Q = np.diag([1.0] * 3 + [2.0 / omega_n**2] * 3)
    R = np.eye(3) / omega_n**4
    return Q, R


def project_thrust(u, e, gamma, a_max):
    """Euclidean projection of u onto the cone {angle(u, e) <= gamma} (gamma None: no cone), then onto
    the ball |u| <= a_max."""
    if gamma is None:
        n = np.linalg.norm(u)
        return u * (a_max / n) if n > a_max else u
    s = float(u @ e)
    w = u - s * e
    t = np.linalg.norm(w)
    tg = np.tan(gamma)
    if t > s * tg:  # outside the cone
        if t * tg <= -s:  # inside the polar cone: nearest point is the apex
            u = np.zeros(3)
        else:
            alpha = (s + t * tg) / (1.0 + tg * tg)
            u = alpha * (e + tg * w / t)
    n = np.linalg.norm(u)
    return u * (a_max / n) if n > a_max else u


class LQR:
    name = "LQR"

    def __init__(self, A, B, Q, R):
        self.A, self.B = A, B
        self.P, self.K = dare(A, B, Q, R)

    def command(self, dx, u_ff, axes, a_max, gamma, **_):
        """Returns the first thrust acceleration (inertial or body, same frame as the model)."""
        u = u_ff[0] - self.K @ dx
        return project_thrust(u, axes[0], gamma, a_max), None


class MPC:
    name = "MPC"

    def __init__(self, A, B, Q, R, N, gamma, keepout=False, glideslope=None, omega_n=None):
        """glideslope: (n_hat, theta) constant cone axis/half-angle over the site (body frame).
        omega_n: closed-loop bandwidth used to scale velocities (z = [p, v / omega_n]) for conditioning."""
        self.N, self.gamma = N, gamma
        self.lqr = LQR(A, B, Q, R)
        n = A.shape[0]
        w = omega_n if omega_n is not None else (float(Q[0, 0] / R[0, 0])) ** 0.25
        self.T = np.diag([1.0] * 3 + [1.0 / w] * 3)  # z = T x (all states in metres)
        Ti = np.linalg.inv(self.T)
        Az = self.T @ A @ Ti
        Qz, Pz = Ti.T @ Q @ Ti, Ti.T @ self.lqr.P @ Ti
        c = float(np.trace(Qz)) / n  # cost normalization
        # decision variables: scaled state z, thrust in units of a_max
        self.dz = cp.Variable((N + 1, n))
        self.nu = cp.Variable((N, 3))  # du / a_max
        self.U = cp.Variable((N, 3))  # total thrust / a_max
        self.z0 = cp.Parameter(n)
        self.uff = cp.Parameter((N, 3))  # u_ff / a_max
        self.axes = cp.Parameter((N, 3))
        self.Bs = cp.Parameter((n, 3))  # T B a_max
        self.Rs = cp.Parameter((3, 3), PSD=True)  # R a_max^2 / c
        Qs, Ps = Qz / c, (Pz + Pz.T) / (2 * c)
        cons = [self.dz[0] == self.z0, self.U == self.uff + self.nu]
        cost = 0
        slack = []
        for k in range(N):
            cons.append(self.dz[k + 1] == Az @ self.dz[k] + self.Bs @ self.nu[k])
            cons.append(cp.norm(self.U[k]) <= 1.0)
            if gamma is not None:
                cons.append(cp.norm(self.U[k]) * np.cos(gamma) <= self.axes[k] @ self.U[k])
            cost += cp.quad_form(self.dz[k], Qs) + cp.quad_form(self.nu[k], self.Rs)
        cost += cp.quad_form(self.dz[N], Ps)
        self._R, self._B, self._c = R, B, c
        if keepout:
            self.ko = cp.Parameter(N)  # e_k^T dp_k >= r_ko - e_k^T p_ref,k
            s = cp.Variable(N, nonneg=True)
            slack.append(s)
            cons += [cp.sum(cp.multiply(self.axes, self.dz[1:, :3]), axis=1) + s >= self.ko]
        if glideslope is not None:
            n_hat, theta = glideslope
            self.gs = cp.Parameter((N, 3))  # p_ref,k - site
            s = cp.Variable(N, nonneg=True)
            slack.append(s)
            for k in range(N):
                rel = self.dz[k + 1, :3] + self.gs[k]
                cons.append(cp.norm(rel) * np.cos(theta) <= n_hat @ rel + s[k])
        for s in slack:
            cost += 10.0 * cp.sum(s)  # exact penalty (1 m of violation costs more than any tracking error)
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self.fallbacks = 0

    def command(self, dx, u_ff, axes, a_max, gamma=None, keepout=None, glideslope=None):
        self.z0.value = self.T @ dx
        self.uff.value = u_ff[: self.N] / a_max
        self.axes.value = axes[: self.N]
        self.Bs.value = self.T @ self._B * a_max
        self.Rs.value = self._R * a_max**2 / self._c
        if keepout is not None:
            self.ko.value = keepout[: self.N]
        if glideslope is not None:
            self.gs.value = glideslope[: self.N]
        ok = False
        for opts in ({}, {"max_iter": 400, "tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6}):
            try:
                with warnings.catch_warnings():  # "optimal_inaccurate" is accepted (then re-projected)
                    warnings.simplefilter("ignore")
                    self.prob.solve(solver=cp.CLARABEL, **opts)
                ok = self.prob.status in ("optimal", "optimal_inaccurate") and self.nu.value is not None
            except cp.SolverError:
                ok = False
            if ok:
                break
        if not ok:  # infeasible / solver trouble: fall back to the projected LQR command
            self.fallbacks += 1
            return self.lqr.command(dx, u_ff, axes, a_max, self.gamma)[0], None
        u = self.U.value[0] * a_max
        return project_thrust(u, axes[0], self.gamma, a_max), self.dz.value @ np.linalg.inv(self.T).T
