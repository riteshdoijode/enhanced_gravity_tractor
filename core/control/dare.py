"""Discrete algebraic Riccati equation, LQR gain, steady-state Kalman gain, ZOH discretization.

`is_stabilizable` and `dare` are copied from the course repository (fall-2026-space-robotics,
lecture3/util/dare.py, Prof. B. Riviere) so the proximity controllers use exactly the lecture's method.
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def is_stabilizable(A, B, tol=1e-9):
    """PBH test: rank [A - lam I, B] = n for every eigenvalue |lam| >= 1."""
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1 - tol and np.linalg.matrix_rank(np.hstack([A - lam * np.eye(n), B]), tol=1e-6) < n:
            return False
    return True


def dare(A, B, Q, R):
    """Solve P = Q + A'PA - A'PB (R + B'PB)^-1 B'PA, return P and K = (R + B'PB)^-1 B'PA."""
    if not is_stabilizable(A, B):
        raise ValueError("(A, B) is not stabilizable: DARE has no stabilizing solution")
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def kalman_gain(A, C, W, V):
    """Steady-state predictor-form Kalman gain L (x+ = A x + B u + L (y - C x)) via the dual DARE."""
    _, K = dare(A.T, C.T, W, V)
    return K.T


def c2d(Ac, Bc, dt):
    """Zero-order-hold discretization: [[A, B], [0, I]] = expm([[Ac, Bc], [0, 0]] dt)."""
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = Ac, Bc
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]
