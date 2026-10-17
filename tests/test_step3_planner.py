"""Step 3 validation: orbital guidance math, deflection physics, and the mission planner's behavior."""

from dataclasses import replace

import numpy as np
import pytest

from core.constants import AU, BENNU, DAY, G, GM, RADIUS, YEAR
from core.deflection import (BPlane, dart_period_change_minutes, earth_capture_radius, gt_accel,
                             gt_required_thrust, ki_delta_v)
from core.guidance.departure import circular_speed, patched_conic_dv, spiral_escape
from core.guidance.kepler import ecliptic_pole_icrf, lambert, lambert_multirev, propagate
from core.planner import MissionParams, shift_to_circle
from core.timeutil import iso_to_t
from tests.keyhole import setup

MU_E = 398600.4418e9


@pytest.fixture(scope="module")
def planner():
    return setup()[2]


def _rk4(r, v, T, n, mu):
    x, h = np.concatenate([r, v]), T / n
    f = lambda x: np.concatenate([x[3:], -mu * x[:3] / np.linalg.norm(x[:3]) ** 3])
    for _ in range(n):
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[:3]


# --------------------------------------------------------------------- two-body tools
def test_lambert_curtis_example_5_2():
    """Curtis, Orbital Mechanics for Engineering Students, Example 5.2."""
    r1, r2 = np.array([5000.0, 10000.0, 2100.0]) * 1e3, np.array([-14600.0, 2500.0, 7000.0]) * 1e3
    v1, v2 = lambert(r1, r2, 3600.0, mu=MU_E)
    assert np.allclose(v1 / 1e3, [-5.9925, 1.9254, 3.2456], atol=1e-4)
    assert np.allclose(v2 / 1e3, [-3.3125, -4.1966, -0.38529], atol=1e-4)
    r, v = propagate(r1, v1, 3600.0, mu=MU_E)
    assert np.linalg.norm(r - r2) < 1e-3 and np.linalg.norm(v - v2) < 1e-6


@pytest.mark.parametrize("v0,T", [([0, 8.5e3, 1e3], 20000.0), ([0, 12e3, 1e3], 20000.0),
                                  ([0, 8.5e3, 1e3], -5000.0), ([0, 10.9e3, 0], 30000.0)])
def test_kepler_propagation_matches_rk4(v0, T):
    r0, v0 = np.array([7000e3, 0, 0]), np.array(v0, float)
    assert np.linalg.norm(propagate(r0, v0, T, mu=MU_E)[0] - _rk4(r0, v0, T, 40000, MU_E)) < 1e-3


def test_lambert_heliocentric_batch_closes(planner):
    """Earth -> Bennu arcs over a porkchop grid: every solution re-propagates onto Bennu."""
    tl = iso_to_t("2027-01-01") + np.arange(40) * 15 * DAY
    tof = np.arange(40, 40 + 40 * 20, 20) * DAY
    L, T = np.meshgrid(tl, tof, indexing="ij")
    rE, _ = planner.earth_helio(L.ravel())
    rB, _ = planner.bennu_helio((L + T).ravel())
    v1, _ = lambert(rE, rB, T.ravel(), prograde_axis=ecliptic_pole_icrf())
    ang = np.degrees(np.arccos(np.sum(rE * rB, 1) / np.linalg.norm(rE, axis=1) / np.linalg.norm(rB, axis=1)))
    ok = np.isfinite(v1[:, 0]) & (ang > 5)
    assert ok.mean() > 0.95
    r, _ = propagate(rE[ok], v1[ok], T.ravel()[ok])
    assert np.median(np.linalg.norm(r - rB[ok], axis=1)) < 10.0  # meters at ~1.5e11 m
    assert np.max(np.linalg.norm(r - rB[ok], axis=1) / np.linalg.norm(rB[ok], axis=1)) < 1e-7


def test_multirev_lambert_closes(planner):
    th = iso_to_t("2028-02-01")
    tof = np.geomspace(200, 12 * 365.25, 120) * DAY
    rE, _ = planner.earth_helio(th)
    rB, _ = planner.bennu_helio(th + tof)
    V1, _ = lambert_multirev(np.broadcast_to(rE[0], rB.shape), rB, tof, n_max=3, prograde_axis=ecliptic_pole_icrf())
    assert np.isfinite(V1[1:, :, 0]).sum() > 100  # multi-revolution solutions exist for long trips
    for j in range(V1.shape[0]):
        ok = np.isfinite(V1[j, :, 0])
        if ok.any():
            r, _ = propagate(np.broadcast_to(rE[0], rB.shape)[ok], V1[j, ok], tof[ok])
            assert np.max(np.linalg.norm(r - rB[ok], axis=1) / AU) < 1e-7


# --------------------------------------------------------------------- Earth departure
def test_patched_conic_and_spiral_escape():
    r_p = RADIUS["earth"] + 400e3
    vc = circular_speed(r_p)
    assert abs(patched_conic_dv(0.0, r_p) - (np.sqrt(2) - 1) * vc) < 1e-9  # C3 = 0 escape burn
    low = spiral_escape(1e-4, r_p)  # 1 N on 10 t
    assert abs(low["dv"] / vc - 1) < 0.1  # classic low-thrust result: escape costs ~ v_circular
    high = spiral_escape(1.0, r_p)  # 10 kN on 10 t: near-impulsive
    assert (np.sqrt(2) - 1) * vc < high["dv"] < 0.6 * vc and high["escaped"]
    assert spiral_escape(1e-3, r_p)["t"] < spiral_escape(1e-4, r_p)["t"] / 5  # more thrust, faster


# --------------------------------------------------------------------- deflection physics
def test_cheng_momentum_transfer():
    U = np.array([0.0, -6144.7, 0.0])
    assert np.allclose(ki_delta_v(579.4, U, 3.61), 3.61 * 579.4 * U / BENNU["mass"])  # head-on
    E = np.array([0.0, 0.0, 1.0])  # ejecta normal to U: beta does not add along U
    assert np.allclose(ki_delta_v(579.4, U, 3.61, E_hat=E), 579.4 * U / BENNU["mass"])


def test_dart_period_change():
    """dP/P = 3 dv_T / v with the measured dv_T = -2.70 mm/s (Cheng et al. 2023) -> ~ -33 min."""
    dP = dart_period_change_minutes()
    assert abs(dP - (-33.0)) < 1.0  # measured: -33.0 +- 1.0 min (3 sigma), Thomas et al. 2023


def test_lu_love_gravity_tractor_example():
    """Lu & Love (2005): 20 t at d/r = 1.5 above a 200 m, 2 g/cc asteroid, phi = 20 deg -> ~1 N;
    delta-v per year = 4.2e-3 (m / 20 t) (d / 100 m)^-2 m/s."""
    r = 100.0
    M = 4 / 3 * np.pi * r**3 * 2000.0
    assert abs(gt_required_thrust(2e4, 1.5 * r, R=r, phi_deg=20.0, M=M) - 1.0) < 0.1
    assert abs(gt_accel(2e4, 100.0) * YEAR - 4.2e-3) < 0.05e-3


def test_earth_capture_radius():
    assert abs(earth_capture_radius(6.1e3) / 1e3 - 13_300) < 200
    assert abs(earth_capture_radius(1e7) / RADIUS["earth"] - 1) < 1e-5  # fast objects: no focusing


def test_bplane_projection():
    bp = BPlane(np.array([0.0, 6e3, 0.0]), np.array([29e3, 1e3, 0.0]))
    assert np.allclose(bp.project(np.array([0.0, 1e6, 0.0])), 0.0)  # along v_rel: timing only
    assert abs(bp.timing_shift(np.array([0.0, 6e3, 0.0])) + 6e3) < 1e-9
    assert np.allclose(np.linalg.norm(bp.basis, axis=1), 1.0) and abs(bp.xi @ bp.zeta) < 1e-12


# --------------------------------------------------------------------- planner behavior
def test_more_thrust_arrives_earlier(planner):
    t_launch = iso_to_t("2031-01-01")
    arrivals = [planner.plan_tractor(MissionParams(tractor_thrust_n=T), t_launch)[0].t_arrive for T in (1, 10, 100, 1000)]
    assert all(np.isfinite(arrivals))
    assert all(a >= b for a, b in zip(arrivals, arrivals[1:]))
    assert not planner.plan_tractor(MissionParams(tractor_thrust_n=0.05), t_launch)[0].feasible  # too weak to escape


def test_impact_deflection_linear_in_mass_and_beta(planner):
    """For a fixed arc and point of contact the 2135 shift is linear in the impactor mass; the head-on gain
    per kilogram of every opportunity is linear in beta."""
    base = planner.plan(MissionParams(use_recommended_mass=False, impactor_mass_kg=100.0))
    heavy = planner.plan(MissionParams(use_recommended_mass=False, impactor_mass_kg=400.0))
    assert heavy.impactor.t_arrive == base.impactor.t_arrive  # same best arc
    assert np.allclose(heavy.impactor.db, 4.0 * base.impactor.db, rtol=1e-9)
    o1 = planner.opportunities(MissionParams(beta=2.0))
    o2 = planner.opportunities(MissionParams(beta=4.0))
    g1 = [o.gain_per_kg for o in o1 if o.feasible]
    g2 = [o.gain_per_kg for o in o2 if o.feasible]
    assert np.allclose(g2, 2.0 * np.array(g1), rtol=1e-9)


def test_required_shift_reaches_the_goal_circle(planner):
    p = MissionParams(goal_ld=1.0)
    d, lam = planner.push_direction(p)
    R = planner.goal_radius(p)
    assert np.isclose(np.linalg.norm(planner.b0 + lam * d), R, rtol=1e-12)
    assert lam < shift_to_circle(planner.b0, -d, R)  # the cheaper side
    assert np.linalg.norm(planner.b0) < R  # the keyhole flyby is inside the Moon's distance
    assert shift_to_circle(planner.b0 + 2 * lam * d, d, R) == 0.0


def test_planner_success_logic_and_speed(planner):
    """The recommended DART (sized for the target, with the tractor's trim) meets the goal; a token impactor
    does not; a warm re-plan is interactive."""
    good = planner.plan(MissionParams())
    weak = planner.plan(MissionParams(use_recommended_mass=False, impactor_mass_kg=10.0, tractor_thrust_n=0.05))
    assert good.success and good.q_final >= planner.goal_radius(good.params)
    assert not weak.success
    assert 0.0 <= weak.p_success <= good.p_success <= 1.0
    again = planner.plan(MissionParams(tractor_thrust_n=30.0))
    assert again.compute_ms < 1500.0  # interactive once the opportunity porkchops are cached


def test_dart_is_far_cheaper_before_the_2060_flyby(planner):
    """The 2060 flyby amplifies earlier pushes ~4,000x: the recommended impactor before it is hundreds of kg,
    after it hundreds of tonnes and more."""
    opps = [o for o in planner.opportunities(MissionParams()) if o.feasible]
    t60 = iso_to_t("2060-09-23")
    before = min(o.mass_nominal for o in opps if o.t_arrive < t60)
    after = min(o.mass_nominal for o in opps if o.t_arrive > t60 + 365 * DAY)
    assert before < 2_000.0 and after > 50_000.0
    rec = planner.recommended(opps)
    assert rec.t_arrive < t60
    # recommended mass x gain = the shift still needed toward the aim point
    assert np.isclose(rec.mass_nominal * rec.gain_per_kg, rec.remaining, rtol=1e-9)
