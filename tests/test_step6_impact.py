"""Step 6 validation: impact point on the shape model, momentum transfer (Cheng et al. 2023 eq. 1), spin change,
ejecta, and the tractor measuring beta from the impact in the live mission."""

import time

import numpy as np
import pytest

from core.constants import BENNU, DAY
from core.dynamics import asteroid
from core.impact import EjectaCloud, inertia_body, locate_impact, momentum_transfer, sample_ejecta, spin_change

M = BENNU["mass"]


@pytest.fixture(scope="module")
def env():
    from core.proximity.ops import ProxEnvironment

    return ProxEnvironment.shared()


def test_momentum_transfer_cheng_eq1():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, beta = rng.uniform(500, 5e4), rng.uniform(1, 5)
        U = rng.normal(size=3) * 1e4
        E = rng.normal(size=3)
        E /= np.linalg.norm(E)
        dv, p_ej = momentum_transfer(m, U, beta, E)
        assert np.allclose(M * dv, m * U + p_ej, rtol=1e-12)  # impactor + ejecta recoil
        perp = U - (E @ U) * E
        assert np.allclose(M * dv - (M * dv @ E) * E, m * perp, rtol=1e-9)  # beta acts only along E
    U = np.array([0.0, -1.2e4, 0.0])
    dv, _ = momentum_transfer(4e4, U, 3.6, -U / np.linalg.norm(U))  # head-on: dv = beta m U / M
    assert np.allclose(dv, 3.6 * 4e4 * U / M)


def test_impact_point_is_on_the_surface(env):
    from core.dynamics.polyhedron import Polyhedron

    poly = env.grav.poly
    rng = np.random.default_rng(1)
    for _ in range(10):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        off = np.cross(u, rng.normal(size=3))
        off *= rng.uniform(0, 150) / np.linalg.norm(off)
        hit = locate_impact(env.ray, -2000 * u + off, 2000 * u + off)
        assert hit is not None
        p, n = hit["point"], hit["normal"]
        assert -(n @ u) > 0  # the ray enters through the outward-facing surface
        # 8 m outside / inside the surface according to the polyhedron (shape models differ by ~metres)
        assert not poly.contains(p + 8 * n)[0] and poly.contains(p - 8 * n)[0]
    assert locate_impact(env.ray, np.array([-2000.0, 400.0, 0]), np.array([2000.0, 400.0, 0])) is None  # misses


def test_spin_change_physics():
    J = inertia_body()
    w = asteroid.omega_body(0.0)
    r = np.array([0.0, 250.0, 0.0])
    # momentum through the center of mass: no torque
    assert np.allclose(spin_change(r, np.array([0.0, -1e9, 0.0]))["d_omega"], 0.0)
    # prograde push at the equator (along omega x r) spins Bennu up: shorter day, angular momentum balance
    dP = 1e9 * np.cross(w, r) / np.linalg.norm(np.cross(w, r))
    out = spin_change(r, dP)
    assert out["period_change_s"] < 0
    dL = np.cross(r, dP)
    assert np.allclose(J @ out["d_omega"], dL, rtol=0, atol=1e-9 * np.linalg.norm(dL))
    assert out["pole_shift_deg"] < 0.1  # torque along the spin axis barely tilts it (principal axis 0.7 deg from z)


def test_spin_change_keeps_attitude_continuous():
    t = 1e9
    R0 = asteroid.bennu_attitude(t)
    asteroid.set_spin_change(t, BENNU["omega"] * 1.01)
    try:
        assert np.allclose(asteroid.bennu_attitude(t), R0, atol=1e-12)
        assert np.isclose(asteroid.spin_rate(t + 1), BENNU["omega"] * 1.01)
        w1 = asteroid.rotation_angle(t + 3600) - asteroid.rotation_angle(t)
        assert np.isclose(w1, BENNU["omega"] * 1.01 * 3600)
    finally:
        asteroid.reset_spin()
    assert asteroid.spin_rate(t + 1) == BENNU["omega"]


def test_ejecta_scaling_and_momentum():
    rng = np.random.default_rng(3)
    n_hat = np.array([0.0, 0.0, 1.0])
    vel, mass_each = sample_ejecta(200_000, 1e9, n_hat, 0.2, rng)
    v = np.linalg.norm(vel, axis=1)
    # cumulative distribution follows the point-source power law M(>v) ~ v^(-3 mu) (mu = 0.41)
    for v1, v2 in [(0.1, 0.4), (0.2, 2.0)]:
        ratio = np.mean(v > v2) / np.mean(v > v1)
        expected = ((v2 / 0.05) ** -1.23 - (50 / 0.05) ** -1.23) / ((v1 / 0.05) ** -1.23 - (50 / 0.05) ** -1.23)
        assert abs(ratio / expected - 1) < 0.05
    esc = v > 0.2
    assert np.isclose(mass_each * np.sum(vel[esc] @ n_hat), 1e9)  # escaping ejecta carry the recoil
    ang = np.degrees(np.arccos(vel @ n_hat / v))
    assert 40 < np.median(ang) < 50


def test_ejecta_cloud_fates(env):
    hit = locate_impact(env.ray, np.array([-2000.0, 30.0, 20.0]), np.array([2000.0, 30.0, 20.0]))
    cloud = EjectaCloud(0.0, hit["point"], hit["normal"], 5e8, env.grav, env.ray, n=300, duration=6 * 3600.0)
    cloud.thread.join(120)
    assert cloud.done and len(cloud.frames) > 10
    _, code = cloud.at(6 * 3600.0)
    s = cloud.summary()
    assert s["landed"] > 0.3 and s["escaped"] > 0.0
    assert 0.1 < cloud.v_esc < 0.3  # Bennu's escape speed ~ 20 cm/s


def test_tractor_measures_beta_from_the_impact():
    """The tractor launched first waits at the stand-off point while a DART is on its way: its navigation
    filter estimates beta from Bennu's velocity jump (with the impact geometry); spin and ejecta are computed."""
    from core.mission import LiveMission
    from core.planner import MissionParams
    from tests.keyhole import setup

    _, eph, pl = setup()
    p = MissionParams(tractor_thrust_n=20.0)
    res = pl.plan(p)
    ip = res.impactor
    t_launch = ip.t_arrive - 1.6 * 365.25 * DAY
    tp, _ = pl.plan_tractor(p, t_launch)
    assert tp.t_arrive < ip.t_arrive - 30 * DAY  # arrives first
    m = LiveMission(eph, pl.nom, pl, p, t_launch, seed=6)
    m.launch_tractor(tp)
    m.launch_dart(ip)
    while m.beta_measured is None and m.t < ip.t_arrive + 10 * DAY:
        if m.busy():
            time.sleep(0.2)
            continue
        m.advance(m.t + 5 * DAY)
    assert m.beta_measured is not None
    b, s = m.beta_measured
    assert abs(b - m.beta_true) < 3 * s + 0.05
    im = m.impactor.impact
    assert asteroid.spin_rate(m.t) != BENNU["omega"]  # the spin changed at the impact
    assert np.isclose(2 * np.pi / asteroid.spin_rate(m.t) - 2 * np.pi / BENNU["omega"], im["period_change_s"])
    assert m.ejecta is not None
    asteroid.reset_spin()
