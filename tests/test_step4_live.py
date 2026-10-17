"""Step 4 validation: low-thrust flight planning, the live N-body mission, events and navigation."""

import time

import numpy as np
import pytest

from core.constants import AU, BENNU, DAY, GM
from core.estimation.navfilter import NavFilter
from core.guidance import lowthrust
from core.mission import LiveMission
from core.planner import MissionParams
from core.timeutil import iso_to_t
from tests.keyhole import setup

MU_S = GM["sun"]
T0 = iso_to_t("2030-01-01")


@pytest.fixture(scope="module")
def planner():
    return setup()[2]


def _mission(planner, thrust=20.0, seed=1, dart=False):
    """A live mission from 2030 with the tractor launched at once (tests of the cruise), or the recommended
    DART (dart=True)."""
    p = MissionParams(tractor_thrust_n=thrust)
    m = LiveMission(planner.eph, planner.nom, planner, p, T0 if not dart else iso_to_t("2026-10-15"), seed=seed)
    if dart:
        res = planner.plan(p)
        m.launch_dart(res.impactor)
        return m, res.impactor
    tp, _ = planner.plan_tractor(p, T0)
    m.launch_tractor(tp)
    return m, tp


def _to_cruise(m, tp):
    """Advance until the escape is over (its length depends on the thrust)."""
    while m.tractor.phase != "cruise" and m.t < tp.t_launch + 2 * 365.25 * DAY:
        m.advance(m.t + 10 * DAY)
    assert m.tractor.phase == "cruise"


def _wait_idle(m, limit=300):
    t = time.time()
    while (m.busy() or m.background()) and time.time() - t < limit:
        time.sleep(0.2)


def test_minimum_energy_transfer_respects_thrust_and_closes(planner):
    """Direct transcription: converged profile meets the target and the thrust limit."""
    t0 = iso_to_t("2027-03-01")
    rE, vE = planner.earth_helio(t0)
    x0 = np.concatenate([rE[0], vE[0]])
    target = lambda t: planner.nom.state(np.atleast_1d(t)) - planner.eph.state("sun", np.atleast_1d(t))
    a_max = 2e-3
    sol, tf = lowthrust.fastest(x0, t0, target, a_max, 20 * DAY, 4 * 365.25 * DAY)
    assert sol is not None and sol["converged"][0]
    assert sol["peak"][0] <= 0.9 * a_max
    nodes, _ = lowthrust._forward(x0[None], sol["dv"], sol["dt"], MU_S)  # re-fly the impulses
    xT = target(np.array([tf]))[0]
    assert np.linalg.norm(nodes[0, -1, :3] - xT[:3]) < 5e3 and np.linalg.norm(nodes[0, -1, 3:] - xT[3:]) < 0.5
    assert sol["min_r_sun"][0] >= 0.3 * AU


def test_full_live_mission(planner):
    """DART: orbit raising -> injection -> cruise with TCMs -> terminal guidance to the point of contact ->
    impact; ground tracking gives the result; then the tractor: escape -> cruise -> rendezvous (N-body)."""
    m, ip = _mission(planner, dart=True, seed=1)
    ki = m.impactor
    while not m.dart_results and m.t < ip.t_arrive + 60 * DAY:
        _wait_idle(m)
        m.advance(m.t + 5 * DAY)
    burns = [e for e in m.events if "Orbit-raising burn" in e.text]
    assert len(burns) == len(ip.raising.burns) - 1 and any("Injection burn" in e.text for e in m.events)
    assert ki.prop_used == pytest.approx(ip.raising.propellant_kg, rel=1e-9)
    assert ki.phase == "impacted" and ki.closest <= 290.0
    im = ki.impact
    U, E, mi = im["U"], im["E_hat"], ki.mass
    expected = (mi * U + mi * (m.beta_true - 1) * (E @ U) * E) / BENNU["mass"]
    assert np.allclose(im["dv_bennu"], expected, rtol=1e-12)  # Cheng et al. eq. 1, ejecta along the normal
    assert 200.0 < np.linalg.norm(im["point_b"]) < 290.0  # on Bennu's real surface
    assert im["aim_error_m"] < 60.0  # terminal guidance hit the recommended point of contact
    assert m.dart_results and m.dart_results[0]["result"] in ("success", "short")
    # the tractor, launched after the result
    tp, _ = planner.plan_tractor(m.p, m.t, m.b_now_abs())
    assert m.launch_tractor(tp)
    inside, n = 0, 0
    t_wall = time.time()
    while not m.cruise_complete() and m.t < tp.t_launch + 6 * 365.25 * DAY:
        m.advance(m.t + 5 * DAY)
        tr = m.tractor
        if tr.phase == "cruise":
            n += 1
            inside += np.linalg.norm(tr.nav.x[:3] - tr.x[:3]) < 3 * tr.nav.sigma()[0]
    assert time.time() - t_wall < 240  # (a stall guard; Bennu's full-model orbit is now integrated inline here too)
    assert m.tractor.phase == "proximity"
    assert np.linalg.norm(m.tractor.arrival_offset) < 100e3  # handed to proximity ops within 100 km
    assert inside / n > 0.95  # navigation filter consistent


def test_storm_keeps_or_slips_schedule(planner):
    m, tp = _mission(planner, seed=2)
    _to_cruise(m, tp)
    m.advance(m.t + 60 * DAY)
    eta0 = m.tractor.eta()
    m.solar_storm(40)
    t_storm = m.t
    m.advance(m.t + 45 * DAY)
    assert any("Solar storm" in e.text for e in m.events)
    assert any("after the storm" in e.text and e.t >= t_storm + 40 * DAY - 1 for e in m.events)
    assert m.tractor.eta() >= eta0 - 3600.0  # an outage can only keep or delay the arrival


def test_throttle_up_does_not_delay(planner):
    m, tp = _mission(planner, thrust=10.0, seed=3)
    _to_cruise(m, tp)
    eta0 = m.tractor.eta()
    m.set_tractor_thrust(40.0)
    m.advance(m.t + DAY)
    assert any("Re-plan for new thrust" in e.text for e in m.events)
    assert m.tractor.eta() <= eta0 + DAY


def test_storm_during_impactor_cruise_does_not_stall(planner):
    """Regression: a correction burn deferred by safe mode must not shrink the step to 1 s."""
    m, ip = _mission(planner, dart=True, seed=1)
    m.advance(ip.t_launch + 2 * DAY)
    m.solar_storm(40)
    t_wall = time.time()
    m.advance(ip.t_arrive + 5 * DAY)
    assert time.time() - t_wall < 60
    assert m.impactor.phase == "impacted"
    _wait_idle(m)


def test_dynamic_model_compensation_estimates_thrust_bias():
    """A constant unknown acceleration is recovered from position/velocity fixes (two-body, 60 days)."""
    rng = np.random.default_rng(0)
    r, v = np.array([AU, 0, 0]), np.array([0, 29.78e3, 0])
    bias = np.array([2e-5, -1e-5, 5e-6])
    nav = NavFilter(np.concatenate([r, v]) + np.concatenate([rng.normal(0, 1e3, 3), rng.normal(0, 0.01, 3)]),
                    1e3, 0.01, sigma_b0=3e-5, tau=30 * DAY)
    f = lambda rr, a: -MU_S * rr / np.linalg.norm(rr) ** 3 + a
    h = 3600.0
    for k in range(int(60 * DAY / h)):
        for (rr, vv, a, tgt) in [(r, v, bias, "truth"), (nav.x[:3], nav.x[3:6], nav.bias, "est")]:
            k1 = (vv, f(rr, a)); k2 = (vv + h / 2 * k1[1], f(rr + h / 2 * k1[0], a))
            k3 = (vv + h / 2 * k2[1], f(rr + h / 2 * k2[0], a)); k4 = (vv + h * k3[1], f(rr + h * k3[0], a))
            rn = rr + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            vn = vv + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if tgt == "truth":
                r, v = rn, vn
            else:
                nav.x[:6] = np.concatenate([rn, vn])
        nav.propagate_bias(h)
        nav.predict_cov(h, nav.x[:3], MU_S, 3e-5)
        if k % 24 == 23:
            nav.update(np.concatenate([r, v]) + np.concatenate([rng.normal(0, 1e3, 3), rng.normal(0, 0.01, 3)]),
                       1e3, 0.01)
    assert np.linalg.norm(nav.bias - bias) < 0.3 * np.linalg.norm(bias)
    assert np.linalg.norm(nav.x[:3] - r) < 3 * nav.sigma()[0]
