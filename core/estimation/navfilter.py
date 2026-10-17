"""Cruise navigation Kalman filter with dynamic model compensation (lecture L3 + orbit-determination practice).

State x = [r (3), v (3), b (3)]: position, velocity and an unknown acceleration b (thrust execution
error: magnitude ~1 %, pointing ~0.3 deg, held for days). b is a first-order Gauss-Markov process,
    b' = -b / tau + w,
because a white-noise model badly underestimates errors that persist for days (the filter would become
overconfident). Prediction uses the dynamics with the *commanded* thrust plus the estimated b; the
covariance propagates through the linearization
    F = [[I, h I, h^2/2 I], [G h, I, h I], [0, 0, e^(-h/tau) I]],  Q_bb = sigma_b^2 (1 - e^(-2h/tau)) I
Measurement (radiometric orbit-determination fix): y = C x + v, C = [I_6 0], V = diag(sigma_r^2 I, sigma_v^2 I).
Update: L = P C^T (C P C^T + V)^-1, x <- x + L (y - C x), P <- (I - L C) P (I - L C)^T + L V L^T (Joseph).
"""

import numpy as np

from core.dynamics.gravity import gravity_gradient

C = np.hstack([np.eye(6), np.zeros((6, 3))])


class NavFilter:
    def __init__(self, x0, sigma_r0=1e3, sigma_v0=0.05, sigma_b0=1e-6, tau=2 * 86400.0):
        self.x = np.concatenate([np.asarray(x0, float)[:6], np.zeros(3)])
        self.P = np.diag([sigma_r0**2] * 3 + [sigma_v0**2] * 3 + [sigma_b0**2] * 3)
        self.tau = tau

    @property
    def bias(self):
        return self.x[6:9]

    def propagate_bias(self, h):
        self.x[6:9] *= np.exp(-h / self.tau)

    def predict_cov(self, h, r, mu_c, sigma_b, white=1e-10):
        F = np.eye(9)
        F[0:3, 3:6] = h * np.eye(3)
        F[0:3, 6:9] = 0.5 * h * h * np.eye(3)
        F[3:6, 0:3] = gravity_gradient(r, mu_c) * h
        F[3:6, 6:9] = h * np.eye(3)
        F[6:9, 6:9] = np.exp(-h / self.tau) * np.eye(3)
        Q = np.zeros((9, 9))
        Q[6:9, 6:9] = sigma_b**2 * (1.0 - np.exp(-2.0 * h / self.tau)) * np.eye(3)
        Q[3:6, 3:6] = white**2 * h * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    def update(self, y, sigma_r, sigma_v):
        V = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)
        S = C @ self.P @ C.T + V
        L = self.P @ C.T @ np.linalg.inv(S)
        innov = y - C @ self.x
        self.x = self.x + L @ innov
        IL = np.eye(9) - L @ C
        self.P = IL @ self.P @ IL.T + L @ V @ L.T
        return innov

    def sigma(self):
        d = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        return float(np.linalg.norm(d[:3])), float(np.linalg.norm(d[3:6]))

    def nees(self, x_true):
        """Normalized estimation error squared of [r, v] (consistency check: mean ~ 6)."""
        e = self.x[:6] - x_true[:6]
        return float(e @ np.linalg.solve(self.P[:6, :6], e))
