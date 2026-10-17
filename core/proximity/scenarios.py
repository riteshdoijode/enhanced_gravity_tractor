"""Stand-alone proximity scenarios for validation and report figures (no mission timeline needed).

station_keeping: the gravity tractor acquiring its hover station after a disturbance, in the full rotating
polyhedron field with solar pressure, flown by LQR or MPC on the true state (isolates the controller) or
on the EKF estimate from LIDAR + camera (closes the loop through navigation).
"""

import numpy as np

from core.constants import AU, BENNU
from core.control.prox import LQR, MPC, project_thrust, relative_model, weights
from core.dynamics.fastgrav import BennuGravity
from core.dynamics.perturbations import P_SUN
from core.proximity import sensors
from core.proximity.ekf import ProxEKF

T_REF = 0.0  # Bennu's rotation phase is arbitrary for these tests


def station_keeping(kind="MPC", d=485.0, e=(1.0, 0.0, 0.0), dp0=(-60.0, 120.0, 0.0), dv0=(-0.05, -0.05, 0.0),
                    hours=8.0, thrust_n=2.0, mass=30_000.0, dt=60.0, bandwidth=3e-3, plume_deg=20.0,
                    navigation=False, seed=0, grav=None, ray=None):
    grav = grav or BennuGravity()
    rng = np.random.default_rng(seed)
    e = np.asarray(e, float) / np.linalg.norm(e)
    p_ref = d * e
    gamma = np.arcsin(min(0.99, grav.R_bound / d)) + np.radians(plume_deg)
    r_ko = grav.R_bound + 60.0
    a_max = thrust_n / mass
    sun = np.array([0.0, -1.0, 0.0]) * AU  # Sun direction fixed for the few hours simulated
    a_srp = P_SUN * 1.3 * 120.0 / mass * (-sun / AU)
    G = BENNU["gm"] / d**3 * (3 * np.outer(e, e) - np.eye(3))
    A, B = relative_model(G, dt)
    Q, R = weights(bandwidth)
    ctrl = MPC(A, B, Q, R, 30, gamma, keepout=True, omega_n=bandwidth) if kind == "MPC" else LQR(A, B, Q, R)
    x = np.concatenate([p_ref + np.asarray(dp0, float), np.asarray(dv0, float)])
    ekf = None
    if navigation:
        ray = ray or sensors.ShapeRaycaster()
        ekf = ProxEKF(x + np.concatenate([rng.normal(0, 1.0, 3), rng.normal(0, 1e-3, 3)]), sigma_p=1.0, sigma_v=1e-3)
    acc = lambda t, p: grav.accel(t, p) + a_srp
    out = {k: [] for k in ("t", "dp", "u", "angle", "margin", "raw_angle", "nav_err", "nav_sigma")}
    t = T_REF
    n = int(hours * 3600 / dt)
    for _ in range(n):
        xe = ekf.x[:6] if ekf is not None else x
        b_est = ekf.bias if ekf is not None else np.zeros(3)
        N = getattr(ctrl, "N", 1)
        ts = t + dt * np.arange(N)
        uff = np.array([-(grav.accel(tk, p_ref) + a_srp * 0.92) - b_est for tk in ts])  # filter's 8 % SRP error
        axes = np.tile(e, (N, 1))
        dx = np.concatenate([xe[:3] - p_ref, xe[3:6]])
        ko = np.full(N, r_ko - e @ p_ref)
        raw = uff[0] - ctrl.lqr.K @ dx if kind == "MPC" else uff[0] - ctrl.K @ dx
        u, _ = ctrl.command(dx, uff, axes, a_max, gamma=gamma, keepout=ko)
        u = project_thrust(u, e, gamma, a_max)
        out["t"].append(t)
        out["dp"].append(x[:3] - p_ref)
        out["u"].append(u * mass)
        nu = np.linalg.norm(u)
        out["angle"].append(np.degrees(np.arccos(np.clip(u @ e / nu, -1, 1))) if nu > 1e-12 else 0.0)
        out["raw_angle"].append(np.degrees(np.arccos(np.clip(raw @ e / max(np.linalg.norm(raw), 1e-30), -1, 1))))
        out["margin"].append(e @ x[:3] - r_ko)
        if ekf is not None:
            out["nav_err"].append(np.linalg.norm(ekf.x[:3] - x[:3]))
            out["nav_sigma"].append(ekf.sigma_pos())
        u_true = u * (1 + rng.normal(0, 0.01))
        h = dt / 3
        for _ in range(3):
            f = lambda tt, p, v: (v, acc(tt, p) + u_true)
            p, v = x[:3], x[3:]
            k1 = f(t, p, v)
            k2 = f(t + h / 2, p + h / 2 * k1[0], v + h / 2 * k1[1])
            k3 = f(t + h / 2, p + h / 2 * k2[0], v + h / 2 * k2[1])
            k4 = f(t + h, p + h * k3[0], v + h * k3[1])
            x = np.concatenate([p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                                v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])])
            if ekf is not None:
                _, Gm = grav.accel_gradient(t, ekf.x[:3])
                tt = t
                ekf.predict(h, lambda pp, vv: grav.accel(tt, pp) + a_srp * 0.92 + u, Gm, 0.01 * np.linalg.norm(u) + 2e-8)
            t += h
        if ekf is not None:
            from core.dynamics.asteroid import bennu_attitude

            Rb = bennu_attitude(t)
            u_true_dir = -x[:3] / np.linalg.norm(x[:3])
            E = sensors.tangent_basis(u_true_dir)
            um = u_true_dir + E @ rng.normal(0, sensors.SIGMA_BEARING, 2)
            ekf.update_bearing(um / np.linalg.norm(um), np.zeros(3), sensors.SIGMA_BEARING)
            b = -ekf.x[:3] / np.linalg.norm(ekf.x[:3])
            hit, _ = ray.cast(Rb.T @ x[:3], Rb.T @ b)
            pred, nb = ray.cast(Rb.T @ ekf.x[:3], Rb.T @ b)
            if hit is not None and pred is not None:
                ekf.update_range(hit + rng.normal(0, sensors.SIGMA_LIDAR), pred, Rb @ nb, b, sensors.SIGMA_LIDAR + 0.05)
    res = {k: np.asarray(v) for k, v in out.items()}
    err = np.linalg.norm(res["dp"], axis=1)
    res["cost"] = float(np.sum(err**2) * dt)
    res["dv"] = float(np.sum(np.linalg.norm(res["u"], axis=1)) / mass * dt)
    res["gamma_deg"] = float(np.degrees(gamma))
    res["final_err"] = float(err[-1])
    res["min_range"] = float(np.min(np.linalg.norm(res["dp"] + p_ref, axis=1)))
    above = np.where(err > 1.0)[0]
    res["settle_h"] = float(res["t"][above[-1] + 1] / 3600) if len(above) and above[-1] + 1 < len(err) else (
        0.0 if not len(above) else np.inf)
    res["fallbacks"] = getattr(ctrl, "fallbacks", 0)
    return res
