"""Rigid-body attitude (lecture L1 eqs. 3-4) with quaternions.

Quaternions are scalar-first, Hamilton convention, q = [q0, qv], rotating body -> inertial:
    v_inertial = R(q) v_body,      R(q) = (q0^2 - qv.qv) I + 2 qv qv^T + 2 q0 [qv x]
Kinematics (replaces the lecture's Theta' = B(Theta)^-1 omega, singular at theta = +-90 deg):
    q' = 1/2 q (x) [0, omega]    (omega in body axes)
Dynamics (Euler's equation, lecture eq. 4):
    J omega' = tau - omega x (J omega)
"""

import numpy as np


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def qmul(p, q):
    p0, pv, q0, qv = p[0], p[1:], q[0], q[1:]
    return np.concatenate([[p0 * q0 - pv @ qv], p0 * qv + q0 * pv + np.cross(pv, qv)])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnormalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def q_to_R(q):
    q0, qv = q[0], q[1:]
    return (q0 * q0 - qv @ qv) * np.eye(3) + 2.0 * np.outer(qv, qv) + 2.0 * q0 * skew(qv)


def R_to_q(R):
    """Rotation matrix -> unit quaternion (Shepperd's method, numerically robust)."""
    tr = np.trace(R)
    k = np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]])
    if k == 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif k == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif k == 2:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def q_dot(q, omega):
    """q' = 1/2 q (x) [0, omega]."""
    return 0.5 * qmul(q, np.concatenate([[0.0], omega]))


def omega_dot(omega, J, tau, J_inv=None):
    """Euler's equation: omega' = J^-1 (tau - omega x J omega)."""
    J_inv = np.linalg.inv(J) if J_inv is None else J_inv
    return J_inv @ (tau - np.cross(omega, J @ omega))


def gravity_gradient_torque(r_inertial, q, J, mu):
    """tau_gg = 3 mu / |r|^5 (r_b x J r_b), r = spacecraft position relative to the attracting body."""
    r_b = q_to_R(q).T @ r_inertial
    rn = np.linalg.norm(r_b)
    return 3.0 * mu / rn**5 * np.cross(r_b, J @ r_b)


def euler321_from_R(R):
    """3-2-1 (yaw psi, pitch theta, roll phi) angles of R = Rz(psi) Ry(theta) Rx(phi) (lecture 1 convention)."""
    theta = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])


def B_inv(Theta):
    """Lecture 1 Euler-angle kinematics Theta' = B(Theta)^-1 omega (same as lecture1/spacecraft_sim.py)."""
    phi, theta, _ = Theta
    return np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
