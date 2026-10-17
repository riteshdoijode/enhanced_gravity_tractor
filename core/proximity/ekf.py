"""Extended Kalman filter for proximity navigation on raw LIDAR range and camera bearing (lecture 3 EKF).

State (Bennu-centered, ICRF axes): x = [p (3), v (3), b (3), beta]
    p' = v,  v' = g_poly(t, p) + a_tide + a_srp,model + u_cmd + b,  b' = -b / tau + w,  beta' = 0
b absorbs unmodeled accelerations (thrust execution error, solar-pressure mismodeling); beta is the
kinetic impactor's momentum enhancement, made observable only by the impact itself: at impact Bennu's
velocity jumps by dv = beta (m U / M) (Cheng et al. 2023, head-on), i.e. the spacecraft's velocity relative
to Bennu jumps by -beta k, a linear map x+ = J x, P+ = J P J^T. The tractor watching from its stand-off
point therefore measures beta.
Prediction: RK4 on the mean, covariance through F = I + A h + A^2 h^2 / 2 with
    A = [[0, I, 0, 0], [Gamma, 0, I, 0], [0, 0, -I/tau, 0], [0, 0, 0, 0]],  Gamma = polyhedron gravity gradient.
Update: Joseph form with a chi-square (99.9 %) innovation gate. Measurements:
    LIDAR  h = ray-cast range from p along boresight b,  H = [-n^T/(n^T b), 0, 0, 0]
    camera y = E^T u (unit vector to a landmark l),       H = [-E^T/|l - p|, 0, 0, 0]
"""

import numpy as np

N_STATE = 10
GATE = {1: 10.83, 2: 13.82}  # chi-square 99.9 % innovation gate by measurement dimension


class ProxEKF:
    def __init__(self, x0, sigma_p=1.0, sigma_v=0.01, sigma_b=2e-8, tau=6 * 3600.0, beta0=3.6, sigma_beta=0.7):
        self.x = np.concatenate([np.asarray(x0, float)[:6], np.zeros(3), [beta0]])
        self.P = np.diag([sigma_p**2] * 3 + [sigma_v**2] * 3 + [sigma_b**2] * 3 + [sigma_beta**2])
        self.tau = tau
        self.sigma_b = sigma_b
        self.rejected = 0

    @property
    def bias(self):
        return self.x[6:9]

    def predict(self, h, accel, gravity_gradient, sigma_unmodeled):
        """accel(p, v) -> modeled acceleration (gravity + tide + srp + commanded thrust); RK4 on the mean."""
        p, v = self.x[:3], self.x[3:6]
        b = self.x[6:9]
        f = lambda pp, vv: (vv, accel(pp, vv) + b)
        k1 = f(p, v)
        k2 = f(p + h / 2 * k1[0], v + h / 2 * k1[1])
        k3 = f(p + h / 2 * k2[0], v + h / 2 * k2[1])
        k4 = f(p + h * k3[0], v + h * k3[1])
        self.x[:3] = p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.x[3:6] = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.x[6:9] = b * np.exp(-h / self.tau)
        A = np.zeros((N_STATE, N_STATE))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = gravity_gradient
        A[3:6, 6:9] = np.eye(3)
        A[6:9, 6:9] = -np.eye(3) / self.tau
        F = np.eye(N_STATE) + A * h + A @ A * (h * h / 2)
        Q = np.zeros((N_STATE, N_STATE))
        sb = max(self.sigma_b, sigma_unmodeled)
        Q[6:9, 6:9] = sb**2 * (1.0 - np.exp(-2.0 * h / self.tau)) * np.eye(3)
        Q[3:6, 3:6] = (1e-9) ** 2 * h * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    def _update(self, innov, H, R):
        S = H @ self.P @ H.T + R
        nis = float(innov @ np.linalg.solve(S, innov))  # normalized innovation squared
        if nis > GATE[len(innov)]:  # outlier (e.g. LIDAR hitting a different boulder): reject
            self.rejected += 1
            return np.nan
        L = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + L @ innov
        IL = np.eye(N_STATE) - L @ H
        self.P = IL @ self.P @ IL.T + L @ R @ L.T
        return nis

    def update_range(self, y, y_pred, normal, boresight, sigma):
        H = np.zeros((1, N_STATE))
        H[0, :3] = -normal / (normal @ boresight)
        return self._update(np.array([y - y_pred]), H, np.array([[sigma**2]]))

    def update_bearing(self, u_meas, landmark, sigma):
        d = landmark - self.x[:3]
        rng = np.linalg.norm(d)
        u_pred = d / rng
        from core.proximity.sensors import tangent_basis

        E = tangent_basis(u_pred)
        H = np.zeros((2, N_STATE))
        H[:, :3] = -E.T / rng
        return self._update(E.T @ u_meas, H, sigma**2 * np.eye(2))

    def impact_jump(self, k, elapsed=0.0, c=None):
        """Bennu's velocity jumps by dv = c + beta k at the impact (Cheng eq. 1 is affine in beta), so the
        relative velocity jumps by -(c + beta k); `elapsed` seconds after it the position has also
        drifted by -(c + beta k) elapsed."""
        J = np.eye(N_STATE)
        J[3:6, 9] = -k
        J[0:3, 9] = -k * elapsed
        self.x = J @ self.x
        if c is not None:
            self.x[3:6] -= c
            self.x[0:3] -= c * elapsed
        self.P = J @ self.P @ J.T

    def sigma_pos(self):
        return float(np.sqrt(max(np.trace(self.P[:3, :3]), 0.0)))

    def sigma_vel(self):
        return float(np.sqrt(max(np.trace(self.P[3:6, 3:6]), 0.0)))

    def nees(self, x_true):
        e = self.x[:6] - x_true[:6]
        return float(e @ np.linalg.solve(self.P[:6, :6], e))
