"""Step 2 validation: core physics against analytic results, invariants, and JPL's Bennu solution."""

import json

import numpy as np
import pytest

from core.assets import bennu_physics_mesh
from core.constants import AU, BENNU, DATA_DIR, DAY, EARTH_J2, GM, RADIUS
from core.dynamics import attitude as att
from core.dynamics import rotating
from core.dynamics.asteroid import FIT_FILE, LIT_A2, BennuDynamics, bennu_attitude
from core.dynamics.gravity import PLANETARY_BODIES, NBody
from core.dynamics.nominal import J6, NominalBennu
from core.dynamics.perturbations import P_SUN, air_density, cylindrical_shadow, srp
from core.dynamics.polyhedron import Polyhedron
from core.dynamics.spacecraft import SpacecraftDynamics, SpacecraftParams, initial_state
from core.ephemeris import Ephemeris
from core.integrate import euler_step, propagate, rk4_step
from core.timeutil import iso_to_t

MU = GM["earth"]
T0 = iso_to_t("2030-03-20 00:00")


@pytest.fixture(scope="module")
def eph():
    return Ephemeris()


@pytest.fixture(scope="module")
def poly():
    return Polyhedron(*bennu_physics_mesh())


@pytest.fixture(scope="module")
def nominal():
    return NominalBennu()


def circular(alt, inc_deg):
    a = RADIUS["earth"] + alt
    v, i = np.sqrt(MU / a), np.radians(inc_deg)
    return a, np.array([a, 0.0, 0.0]), np.array([0.0, v * np.cos(i), v * np.sin(i)])


def two_body(eph, **kw):
    opts = dict(perturbers=[], j2=False, use_drag=False, use_srp=False, gravity_gradient_torque=False)
    opts.update(kw)
    return SpacecraftDynamics(eph, "earth", **opts)


def energy(xs):
    return 0.5 * np.sum(xs[:, 3:6] ** 2, 1) - MU / np.linalg.norm(xs[:, :3], axis=1)


# --------------------------------------------------------------------- orbital mechanics (lecture L1)
def test_rk4_conserves_energy_but_euler_drifts(eph):
    """Lecture code uses forward Euler; on orbits its energy error grows every revolution."""
    a, r0, v0 = circular(400e3, 51.6)
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    f = two_body(eph).rhs
    _, x_rk, _ = propagate(f, T0, initial_state(r0, v0), 10.0, int(10 * T / 10.0), every=100)
    _, x_eu, _ = propagate(f, T0, initial_state(r0, v0), 1.0, int(10 * T), step=euler_step, every=1000)
    drift_rk = abs(energy(x_rk)[-1] / energy(x_rk)[0] - 1)
    drift_eu = abs(energy(x_eu)[-1] / energy(x_eu)[0] - 1)
    assert drift_rk < 1e-8
    assert drift_eu > 1e-2  # 10x smaller step and still ~10% energy error after 10 orbits


def test_kepler_period(eph):
    a, r0, v0 = circular(400e3, 51.6)
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    _, xs, _ = propagate(two_body(eph).rhs, T0, initial_state(r0, v0), T / 2000, 2000)
    assert np.linalg.norm(xs[-1, :3] - r0) < 1.0  # back to the start after one Keplerian period


def test_j2_nodal_regression(eph):
    a, r0, v0 = circular(400e3, 51.6)
    ts, xs, _ = propagate(two_body(eph, j2=True).rhs, T0, initial_state(r0, v0), 30.0, int(5 * DAY / 30), every=10)
    h = np.cross(xs[:, :3], xs[:, 3:6])
    rate = np.polyfit(ts - T0, np.unwrap(np.arctan2(h[:, 0], -h[:, 1])), 1)[0]
    n = np.sqrt(MU / a**3)
    rate_analytic = -1.5 * n * EARTH_J2 * (RADIUS["earth"] / a) ** 2 * np.cos(np.radians(51.6))
    assert abs(rate / rate_analytic - 1) < 0.01  # osculating vs mean-element theory


def test_drag_decay_rate(eph):
    a, r0, v0 = circular(400e3, 90.0)
    p = SpacecraftParams(mass=1000.0, drag_area=10.0, cd=2.2)
    f = two_body(eph, params=p, use_drag=True).rhs
    ts, xs, _ = propagate(f, T0, initial_state(r0, v0, mass=1000.0), 10.0, int(DAY / 10), every=6)
    dadt = np.polyfit(ts - T0, -MU / (2 * energy(xs)), 1)[0]
    dadt_analytic = -air_density(400e3) * p.cd * p.drag_area / p.mass * np.sqrt(MU * a)
    assert abs(dadt / dadt_analytic - 1) < 0.02


def test_indirect_term_identity(eph):
    """Lecture eq. 2 (inertial) == central-body form: a_geo(r) = a_bary(r + r_E) - a_E, exactly."""
    others = [b for b in PLANETARY_BODIES if b != "earth"]
    geo = NBody(eph, "earth", others)
    bary = NBody(eph, "ssb", PLANETARY_BODIES)
    earth_only = NBody(eph, "ssb", others)
    t = iso_to_t("2031-07-04 12:00")
    r = np.array([3.1e7, -4.2e7, 1.1e7])
    rE = eph.pos("earth", t)
    a_geo = geo.accel(t, r)
    a_ref = bary.accel(t, r + rE) - earth_only.accel(t, rE)
    assert np.linalg.norm(a_geo - a_ref) < 1e-12 * np.linalg.norm(a_geo)


def test_geocentric_vs_barycentric_integration(eph):
    """Short arc agrees; over days the barycentric version accumulates the Earth-ephemeris
    interpolation error (mu * delta / r^3), which is why the simulator integrates relative to Earth."""
    a, r0, v0 = circular(50_000e3, 30.0)
    others = [b for b in PLANETARY_BODIES if b != "earth"]
    geo = two_body(eph, perturbers=others)
    bary = NBody(eph, "ssb", PLANETARY_BODIES)

    def f_bary(t, x, u):
        return np.concatenate([x[3:6], bary.accel(t, x[:3]), np.zeros(9)])

    xb0 = initial_state(r0 + eph.pos("earth", T0), v0 + eph.vel("earth", T0))
    ts, xg, _ = propagate(geo.rhs, T0, initial_state(r0, v0), 60.0, 360, quaternion=False, every=60)
    _, xb, _ = propagate(f_bary, T0, xb0, 60.0, 360, quaternion=False, every=60)
    diff = np.linalg.norm(xb[:, :3] - eph.pos("earth", ts) - xg[:, :3], axis=1)
    assert diff.max() < 20.0  # meters after 6 hours


def test_rocket_equation(eph):
    """Constant thrust, realistic mass depletion: delta-v matches Tsiolkovsky."""
    p = SpacecraftParams(mass=1000.0, isp=320.0, unlimited_fuel=False)
    dyn = SpacecraftDynamics(eph, "sun", params=p, perturbers=[], use_srp=False, gravity_gradient_torque=False)
    r0 = eph.pos("earth", T0) - eph.pos("sun", T0) + np.array([0, 0, 0.05 * AU])
    v0 = np.array([0.0, 0.0, 0.0])
    x0 = initial_state(r0, v0, mass=1000.0)
    thrust = np.array([2000.0, 0, 0, 0, 0, 0])
    n, dt = 600, 1.0
    _, xa, _ = propagate(dyn.rhs, T0, x0, dt, n, control=lambda t, x: thrust)
    _, xc, _ = propagate(dyn.rhs, T0, x0, dt, n)  # coasting reference removes gravity
    dv = np.linalg.norm(xa[-1, 3:6] - xc[-1, 3:6])
    m_f = xa[-1, 13]
    assert abs(m_f - (1000.0 - 2000.0 * 600 / (320.0 * 9.80665))) < 1e-6
    assert abs(dv - 320.0 * 9.80665 * np.log(1000.0 / m_f)) < 1e-3
    assert abs(xa[-1, 14] - (1000.0 - m_f)) < 1e-6  # propellant used is tracked


def test_unlimited_fuel_tracks_propellant(eph):
    p = SpacecraftParams(mass=1000.0, isp=3000.0, unlimited_fuel=True)
    dyn = SpacecraftDynamics(eph, "sun", params=p, perturbers=[], use_srp=False, gravity_gradient_torque=False)
    x0 = initial_state(np.array([AU, 0, 0]), np.zeros(3), mass=1000.0)
    _, xs, _ = propagate(dyn.rhs, T0, x0, 10.0, 100, control=lambda t, x: np.array([1.0, 0, 0, 0, 0, 0]))
    assert xs[-1, 13] == 1000.0
    assert abs(xs[-1, 14] - 1000.0 / (3000.0 * 9.80665)) < 1e-9


def test_srp_and_shadow():
    a = srp(np.array([AU, 0, 0]), 1.3 * 60.0 / 10_000.0)
    assert abs(np.linalg.norm(a) - P_SUN * 1.3 * 60.0 / 10_000.0) < 1e-20 and a[0] > 0
    sun, earth = np.array([AU, 0, 0]), np.zeros(3)
    assert cylindrical_shadow(np.array([-7e6, 0, 0]), sun, earth, RADIUS["earth"]) == 0.0
    assert cylindrical_shadow(np.array([7e6, 0, 0]), sun, earth, RADIUS["earth"]) == 1.0
    assert cylindrical_shadow(np.array([-7e6, 7e6, 0]), sun, earth, RADIUS["earth"]) == 1.0


# --------------------------------------------------------------------- attitude (lecture L1 eqs 3-4)
def test_torque_free_invariants():
    J = np.diag([2.4e4, 2.1e4, 1.8e4])
    Ji = np.linalg.inv(J)

    def f(t, x, u):
        return np.concatenate([att.q_dot(x[:4], x[4:]), att.omega_dot(x[4:], J, np.zeros(3), Ji)])

    x = np.concatenate([[1.0, 0, 0, 0], [0.10, 0.02, 0.05]])
    H0, T0_ = att.q_to_R(x[:4]) @ J @ x[4:], 0.5 * x[4:] @ J @ x[4:]
    for k in range(6000):
        x = rk4_step(f, 0.05 * k, x, 0.05)
        x[:4] /= np.linalg.norm(x[:4])
    R = att.q_to_R(x[:4])
    assert np.linalg.norm(R @ J @ x[4:] - H0) / np.linalg.norm(H0) < 1e-10  # angular momentum (inertial)
    assert abs(0.5 * x[4:] @ J @ x[4:] / T0_ - 1) < 1e-12  # rotational kinetic energy
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-14 and abs(np.linalg.det(R) - 1) < 1e-14  # SO(3)


def test_quaternion_matches_lecture_euler_kinematics():
    w = np.array([0.01, -0.02, 0.015])
    q, Th = np.array([1.0, 0, 0, 0]), np.zeros(3)
    for _ in range(2000):
        q = rk4_step(lambda t, q, u: att.q_dot(q, w), 0, q, 0.05)
        q /= np.linalg.norm(q)
        Th = rk4_step(lambda t, th, u: att.B_inv(th) @ w, 0, Th, 0.05)
    phi, theta, psi = Th
    cz, sz, cy, sy, cx, sx = np.cos(psi), np.sin(psi), np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    R_euler = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
               @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
    assert np.abs(att.q_to_R(q) - R_euler).max() < 1e-10
    back = att.euler321_from_R(R_euler)
    assert np.abs(np.angle(np.exp(1j * (back - Th)))).max() < 1e-10
    assert np.abs(att.q_to_R(att.R_to_q(R_euler)) - R_euler).max() < 1e-12


# --------------------------------------------------------------------- Bennu gravity and rotating frame
def test_polyhedron_self_consistency(poly):
    k = poly.k
    assert abs(poly.laplacian([0.0, 0.0, 0.0])[0] / (-4 * np.pi * k) - 1) < 1e-12  # inside
    assert abs(poly.laplacian([2000.0, 0.0, 0.0])[0]) < 1e-12 * 4 * np.pi * k  # outside
    x = np.array([30_000.0, 32_000.0, 24_000.0])
    a_pm = -BENNU["gm"] * x / np.linalg.norm(x) ** 3
    assert np.linalg.norm(poly.acceleration(x) - a_pm) / np.linalg.norm(a_pm) < 1e-5  # far field
    x, h = np.array([300.0, -150.0, 220.0]), 0.01
    grad = np.array([(poly.potential(x + h * e)[0] - poly.potential(x - h * e)[0]) / (2 * h) for e in np.eye(3)])
    assert np.linalg.norm(grad - poly.acceleration(x)) / np.linalg.norm(grad) < 1e-7  # a = grad U


def test_rotating_frame_jacobi_and_inertial_equivalence(poly):
    g, U = poly.acceleration, (lambda r: poly.potential(r)[0])
    w = rotating.OMEGA_BODY
    r0 = np.array([0.0, 1000.0, 120.0])
    x = np.concatenate([r0, [-np.sqrt(BENNU["gm"] / 1000.0), 0, 0] - np.cross(w, r0)])
    y = np.concatenate([r0, x[3:] + np.cross(w, r0)])
    C0, dt = rotating.jacobi(x, U), 30.0

    def spin(t):
        c, s = np.cos(w[2] * t), np.sin(w[2] * t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def f_inertial(t, y, u):
        R = spin(t)
        return np.concatenate([y[3:], R @ g(R.T @ y[:3])])

    n = int(6 * 3600 / dt)
    for k in range(n):
        x = rk4_step(lambda t, x, u: rotating.rhs(t, x, g), k * dt, x, dt)
        y = rk4_step(f_inertial, k * dt, y, dt)
    assert abs(rotating.jacobi(x, U) / C0 - 1) < 1e-9
    assert np.linalg.norm(spin(n * dt).T @ y[:3] - x[:3]) < 1e-3  # same physics, two frames


def test_spacecraft_model_around_bennu(eph, poly):
    """Full wiring: Bennu-centered spacecraft dynamics + IAU spin model + polyhedron conserve Jacobi."""
    p = SpacecraftParams(mass=1.0)
    dyn = SpacecraftDynamics(eph, "bennu", params=p, perturbers=[], use_srp=False, gravity_gradient_torque=False,
                             bennu_state=lambda t: np.zeros(6), bennu_gravity=poly.acceleration)
    t0 = iso_to_t("2031-01-01")
    R0 = bennu_attitude(t0)
    rho0 = np.array([0.0, 900.0, 200.0])
    v0 = R0 @ np.array([-np.sqrt(BENNU["gm"] / 920.0), 0.0, 0.0])
    _, xs, _ = propagate(dyn.rhs, t0, initial_state(R0 @ rho0, v0, mass=1.0), 30.0, 480, every=480)
    w_body = rotating.OMEGA_BODY

    def C(t, x):
        R = bennu_attitude(t)
        rb, vb = R.T @ x[:3], R.T @ x[3:6] - np.cross(w_body, R.T @ x[:3])
        return rotating.jacobi(np.concatenate([rb, vb]), lambda r: poly.potential(r)[0])

    assert abs(C(t0 + 480 * 30.0, xs[-1]) / C(t0, xs[0]) - 1) < 1e-6


# --------------------------------------------------------------------- Bennu heliocentric orbit vs JPL
def test_nongrav_fit_consistent_with_literature():
    fit = json.loads(FIT_FILE.read_text())
    assert abs(fit["a2"] / LIT_A2 - 1) < 0.03  # Chesley et al. 2014: -4.618e-14 au/d^2
    assert fit["rms_m"] < 2e3  # 2026-2059 residual vs JPL


def test_nominal_matches_jpl_encounters(eph, nominal):
    sbdb = {c["cd"][:4]: c for c in json.loads((DATA_DIR / "sbdb_bennu.json").read_text())["ca_data"]}
    t, r, _ = nominal.close_approach(eph, "2060-09-23 00:36")
    assert abs(np.linalg.norm(r) / AU - float(sbdb["2060"]["dist"])) * AU < 5e3  # JPL 3-sigma: +-3 km
    assert abs(t - iso_to_t("2060-09-23 00:36")) < 120.0
    t, r, _ = nominal.close_approach(eph, "2135-09-25 07:49")
    c = sbdb["2135"]
    three_sigma = (float(c["dist_max"]) - float(c["dist_min"])) / 2 * AU
    assert abs(np.linalg.norm(r) - float(c["dist"]) * AU) < 1.5 * three_sigma
    assert abs(t - iso_to_t("2135-09-25 07:49")) < 2 * float(c["sigma_t"]) * 60.0


def test_stm_symplectic(nominal):
    for date in ["2030-01-01", "2060-09-23 00:36", "2135-09-25 07:49"]:
        P = nominal.Phi0(iso_to_t(date))
        assert np.abs(P.T @ J6 @ P - J6).max() < 1e-9 * np.abs(P).max() ** 2


def test_stm_linear_deflection_matches_nonlinear(eph, nominal):
    """Linearized deflection (the instant planner) vs two full nonlinear propagations, through 2060."""
    t1 = iso_to_t("2032-06-01")
    x1 = nominal.state(t1)
    dv = 1e-4 * x1[3:] / np.linalg.norm(x1[3:])  # 0.1 mm/s along-track
    tc, _, v_rel = nominal.close_approach(eph, "2060-09-23 00:36")
    dyn = BennuDynamics(eph)
    xa, _ = dyn.propagate(t1, x1, [tc])
    xb, _ = dyn.propagate(t1, x1 + np.r_[0, 0, 0, dv], [tc])
    dr_nl = xb[0, :3] - xa[0, :3]
    dr_lin = nominal.dr_from_dv(tc, t1, dv)
    assert np.linalg.norm(dr_lin - dr_nl) / np.linalg.norm(dr_nl) < 1e-3
    shift, _ = NominalBennu.miss_shift(dr_lin, v_rel)
    assert 100e3 < np.linalg.norm(shift) < 300e3  # ~186 km miss-distance change in 2060


def test_parallel_kernels_match_serial(monkeypatch):
    """Multi-threaded paths (Kepler batches, polyhedron batches, Lambert batches) give bit-identical
    results to single-threaded runs."""
    from core import parallel
    from core.dynamics.fastgrav import BennuGravity
    from core.guidance import kepler

    rng = np.random.default_rng(5)
    n = 3000
    r0 = rng.normal(size=(n, 3)) * 1.5e11
    v0 = rng.normal(size=(n, 3)) * 1e4 + np.cross(r0, [0, 0, 1.0]) / 1.5e11 * 3e4
    dt = rng.uniform(-3e7, 3e7, n)
    g = BennuGravity()
    X = rng.normal(size=(400, 3))
    X *= (rng.uniform(300, 4000, 400) / np.linalg.norm(X, axis=1))[:, None]
    results = {}
    for threads in (1, 4):
        monkeypatch.setattr(parallel, "N_THREADS", threads)
        monkeypatch.setattr(parallel, "_pool", None)
        r, v = kepler.propagate(r0, v0, dt)
        A = g.accel_body_many(X)
        w1, w2 = kepler.lambert_batch(r0[:2500], np.roll(r0[:2500], 1, axis=0), np.abs(dt[:2500]) + 86400.0)
        results[threads] = (r, v, A, w1, w2)
    for a, b in zip(results[1], results[4]):
        assert np.array_equal(a, b, equal_nan=True)
    assert np.allclose(results[1][2], np.array([g.accel_body(x) for x in X]), rtol=0, atol=1e-18)
    # compiled Kepler kernel against the vectorized NumPy reference
    rn, vn = kepler._propagate_numpy(r0, v0, dt, kepler.MU_SUN, 60, 1e-12)
    assert np.allclose(results[1][0], rn, rtol=1e-9) and np.allclose(results[1][1], vn, rtol=1e-9)
