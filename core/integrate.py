"""Fixed-step integrators with zero-order-hold control (the simulator's discrete-time loop).

The lecture code uses forward Euler, x_{k+1} = x_k + dt f(x_k, u_k); for orbits its energy error grows
every revolution (shown in the validation figures), so the simulator steps with classical RK4.
"""

import numpy as np

from core.dynamics.spacecraft import IDX_Q


def euler_step(f, t, x, dt, u=None):
    return x + dt * f(t, x, u)


def rk4_step(f, t, x, dt, u=None):
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt / 2 * k1, u)
    k3 = f(t + dt / 2, x + dt / 2 * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def normalize_quaternion(x, idx=IDX_Q):
    x = x.copy()
    x[idx] /= np.linalg.norm(x[idx])
    return x


def propagate(f, t0, x0, dt, n_steps, control=None, step=rk4_step, quaternion=True, every=1):
    """Integrate x' = f(t, x, u) with u = control(t, x) held constant over each step.

    Returns times (K,), states (K, n), controls (K, m) sampled every `every` steps.
    """
    t, x = float(t0), np.asarray(x0, float).copy()
    ts, xs, us = [t], [x.copy()], []
    for k in range(n_steps):
        u = control(t, x) if control is not None else None
        x = step(f, t, x, dt, u)
        if quaternion:
            x = normalize_quaternion(x)
        t += dt
        if (k + 1) % every == 0:
            ts.append(t)
            xs.append(x.copy())
            us.append(np.zeros(6) if u is None else np.asarray(u, float))
    return np.array(ts), np.array(xs), np.array(us)
