"""Keyhole scenario and the sequential DART -> gravity-tractor campaign: the virtual impactor (2135 keyhole
flyby -> Earth impact), the DART's orbit raising, its point of contact, the opportunity recommendations, and a
DART that falls short followed by a recommended second DART."""

import time

import numpy as np
import pytest

from core.constants import AU, DAY, G0, GM, LUNAR_DISTANCE, RADIUS, YEAR
from core.dart import (R_APO_MAX, apogee_radius, best_contact_point, departure_speed, kepler_state, orbit_points,
                       raising_schedule)
from core.timeutil import iso_to_t
from tests.keyhole import setup

MU_E = GM["earth"]


@pytest.fixture(scope="module")
def ks():
    return setup()


# --------------------------------------------------------------------- the scenario
def test_virtual_impactor_hits_earth_after_the_keyhole_flyby(ks):
    sc, eph, _ = ks
    vi = sc.nominal
    # the stored trajectory ends at the Earth impact
    t_end = vi.span[1]
    d_end = np.linalg.norm(vi.state(t_end)[:3] - eph.pos("earth", t_end))
    assert d_end < RADIUS["earth"] + 2000e3 and abs(t_end - sc.t_impact) < 120.0
    assert 2136 < 2000 + (sc.t_impact / DAY + 0.5) / 365.25 < 2185
    # the 2135 flyby goes between the Earth and the Moon
    assert RADIUS["earth"] < sc.q_ca < LUNAR_DISTANCE
    assert abs(sc.t_ca - iso_to_t("2135-09-25")) < 2 * DAY
    assert 0.1e3 < sc.keyhole_width < 50e3  # a keyhole is a few km wide


def test_virtual_impactor_matches_the_real_orbit_before_2060(ks):
    """The scenario differs from Bennu's real (nominal) orbit by a ~1e-5 m/s change in 2026: kilometres before the
    2060 flyby, consistent with the orbit's uncertainty; the 2060 flyby stretches it to the keyhole offset."""
    from core.dynamics.nominal import NominalBennu

    sc, eph, _ = ks
    real = NominalBennu(eph)
    for date in ("2030-01-01", "2045-01-01", "2059-01-01"):
        t = iso_to_t(date)
        assert np.linalg.norm(sc.nominal.state(t)[:3] - real.state(t)[:3]) < 20e3
    assert sc.meta["dv0_ms"] < 1e-4


def test_keyhole_is_narrow(ks):
    """Moving the 2135 flyby two keyhole widths along the line of variations removes the Earth impact."""
    from core.scenario import integrate_tail, lov_vector

    sc, eph, pl = ks
    from core.dynamics.nominal import NominalBennu

    real = NominalBennu(eph)
    t_ca, _, _ = real.close_approach(eph, "2135-09-25 07:49")
    from core.scenario import mtp

    bp, _ = mtp(eph, t_ca, real.state(t_ca))
    _, v1 = lov_vector(real, t_ca, bp.basis)
    hits = []
    for dc in (0.0, 2 * sc.keyhole_width):
        x_pre = real.state(sc.t_pre) + real.Phi0(sc.t_pre) @ ((sc.c_lov + dc) * v1)
        _, t_imp, _ = integrate_tail(eph, x_pre, None, sc.t_pre, sc.t_impact + 30 * DAY)
        hits.append(np.isfinite(t_imp))
    assert hits == [True, False]


def test_close_approaches_reproduce_sbdb(ks):
    """The scenario's Earth approaches before 2060 match JPL's (SBDB) to the orbit's few-km difference."""
    sc, _, _ = ks
    for date, dist_au in (("2054-09-30", 0.03930), ("2060-09-23", 0.00501)):
        a = min(sc.approaches, key=lambda a: abs(a["t"] - iso_to_t(date)))
        assert abs(a["t"] - iso_to_t(date)) < DAY
        assert abs(a["dist"] / AU - dist_au) < 1e-4


# --------------------------------------------------------------------- the DART
def test_orbit_raising_keeps_the_single_burn_delta_v():
    r_p = RADIUS["earth"] + 400e3
    v_inf_vec = np.array([3e3, 4e3, 1e3])
    mass, thrust, isp, t_dep = 600.0, 50e3, 320.0, 1e9
    for unlimited in (True, False):
        rp = raising_schedule(mass, v_inf_vec, r_p, thrust, isp, t_dep, unlimited=unlimited)
        v_dep = departure_speed(np.linalg.norm(v_inf_vec), r_p)
        assert np.isclose(sum(b.dv for b in rp.burns), v_dep - np.sqrt(MU_E / r_p), rtol=1e-12)  # Oberth kept
        assert all(b.r_apogee <= R_APO_MAX * 1.0001 for b in rp.burns[:-1])  # inside the Moon's orbit
        assert rp.burns[-1].t == t_dep and all(np.diff([b.t for b in rp.burns]) > 0)
        # each burn waits one period of the orbit before it
        for b0, b1 in zip(rp.burns[:-1], rp.burns[1:]):
            assert np.isclose(b1.t - b0.t, b0.period, rtol=1e-9)
        c = isp * G0
        dv = rp.dv_total
        expect = mass * dv / c if unlimited else mass * (np.exp(dv / c) - 1.0)
        assert np.isclose(rp.propellant_kg, expect, rtol=1e-9)
    # the hyperbola leaves along v_inf: far along it the velocity points along v_inf
    pos, vel = kepler_state(r_p, v_dep, rp.P_hat, rp.Q_hat, 30 * DAY)
    u = v_inf_vec / np.linalg.norm(v_inf_vec)
    assert vel @ u / np.linalg.norm(vel) > 0.999 and np.isclose(np.linalg.norm(vel), np.linalg.norm(v_inf_vec), rtol=2e-3)
    assert np.isclose(np.linalg.norm(kepler_state(r_p, rp.burns[2].v_perigee, rp.P_hat, rp.Q_hat, 0.0)[0]), r_p)
    pts = orbit_points(r_p, rp.burns[2].v_perigee, rp.P_hat, rp.Q_hat)
    assert np.isclose(np.linalg.norm(pts, axis=1).max(), apogee_radius(r_p, rp.burns[2].v_perigee), rtol=1e-3)


def test_point_of_contact_scores_at_least_a_head_on_hit():
    """A flat patch facing the impactor gives Cheng's head-on value beta m U / M; the recommended point is the
    best of the ray-cast candidates (ejecta along the crater normal)."""
    from core.proximity.sensors import ShapeRaycaster

    ray = ShapeRaycaster()
    U = np.array([6e3, 1e3, -2e3])
    g = np.array([0.2, 1.0, 0.3])  # B-plane gradient of the desired shift
    best = best_contact_point(ray, np.eye(3), U, 3.6, 500.0, g, n_grid=9)
    assert best is not None and best["efficiency"] > 0
    center = best_contact_point(ray, np.eye(3), U, 3.6, 500.0, g, n_grid=1)  # the single central ray
    assert best["score"] >= center["score"] - 1e-12
    assert 200 < np.linalg.norm(best["point_b"]) < 300 and best["offset_m"] <= 0.6 * 242.3


def test_recommended_dart_plan_is_consistent(ks):
    from core.planner import MissionParams

    _, _, pl = ks
    res = pl.plan(MissionParams())
    ip = res.impactor
    assert ip.feasible and ip.contact is not None and "lat" in ip.contact
    assert ip.raising.t_start > iso_to_t("2026-10-15")
    assert ip.v_dep <= 12.5e3 + 1e-6  # within the orbit-raising cut-off
    shift = float((res.b0 + ip.db - res.b0) @ res.direction)
    assert np.isclose(shift, res.target_shift, rtol=1e-2)  # sized for the target at the point of contact (porkchop B on whole days)
    assert res.success and res.q_final >= LUNAR_DISTANCE


# --------------------------------------------------------------------- live: a DART falls short
def test_short_dart_gets_a_recommendation_that_closes_the_gap(ks):
    from dataclasses import replace

    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams(use_recommended_mass=False)
    rec = pl.plan(MissionParams())
    p = replace(p, impactor_mass_kg=0.3 * rec.impactor.mass_nominal)
    res = pl.plan(p)
    m = LiveMission(eph, sc.nominal, pl, p, iso_to_t("2026-10-15"), seed=5)
    m.launch_dart(res.impactor)
    while not m.dart_results and m.t < res.impactor.t_arrive + 60 * DAY:
        wall = time.time()
        while m.busy() and time.time() - wall < 300:
            time.sleep(0.2)
        m.advance(m.t + 5 * DAY)
    r = m.dart_results[0]
    assert r["result"] in ("short", "missed") and r["remaining"] > 0
    assert not m.dart_ok()
    # the recommendation for DART #2 (beta now measured) closes the remaining shift toward the aim point
    q = replace(MissionParams(), beta=m.beta_ground[0], beta_sigma=0.1)
    b_now = m.b_now_abs()
    opps = pl.opportunities(q, b_now=b_now, t_after=m.t + 30 * DAY)
    nxt = pl.recommended(opps)
    assert nxt is not None and nxt.t_launch > m.t
    ip, _ = pl.plan_dart(q, nxt, b_now=b_now, number=2)
    assert np.isclose(float(ip.db @ pl.push_direction(q)[0]), r["remaining"], rtol=0.02)
    assert m.launch_dart(ip)
    assert len(m.impactors) == 2 and m.impactor.name == "DART #2"


def test_full_model_refines_the_recommended_mass(ks):
    """The linear (STM) sizing overshoots for mission-size shifts; the full-model refinement finds the mass whose
    nonlinear shift of the 2135 flyby equals the DART target."""
    from core.planner import MissionParams
    from core.refine import refine_dart_mass

    _, _, pl = ks
    r = refine_dart_mass(pl, MissionParams())
    assert r is not None and abs(r["fraction"] - 1.0) < 0.03
    m_lin, f_lin = r["evals"][0]
    assert m_lin == r["linear_mass"] and f_lin > 1.1  # the linear mass overshoots
    assert r["mass"] < m_lin
    res = pl.plan(MissionParams(refined_mass_kg=r["mass"]))
    assert res.impactor.mass_source == "full model" and np.isclose(res.impactor.mass, r["mass"])
