"""Step 7 validation: deflection state, gravity-tractor tow vs Clohessy-Wiltshire, B-plane trim guidance (2135
keyhole flyby), nonlinear verification; and the Sun's motion around the Solar System barycenter."""

import numpy as np
import pytest

from core.constants import DAY, G, RADIUS, YEAR
from core.timeutil import iso_to_t
from core.tow import Deflection, TrimGuidance, cw_along_track, verify_nonlinear


@pytest.fixture(scope="module")
def setup():
    from tests.keyhole import setup as keyhole

    sc, eph, pl = keyhole()
    return eph, sc.nominal, pl


def test_deflection_state_matches_stm(setup):
    _, nom, _ = setup
    t0 = iso_to_t("2029-01-01")
    d = Deflection(nom, t0)
    dv = np.array([1e-3, -2e-3, 5e-4])
    d.impulse(t0 + 10 * DAY, dv)
    t2 = t0 + 400 * DAY
    assert np.allclose(d.state_at(t2), nom.Phi(t2, t0 + 10 * DAY)[:, 3:] @ dv, rtol=1e-12, atol=1e-12)
    assert np.allclose(d.state_at(t0), 0.0)  # before the push
    a = np.array([0.0, 1e-11, 0.0])
    d2 = Deflection(nom, t0)
    d2.advance(t0 + 2 * YEAR, lambda t: a)
    ts = np.linspace(t0, t0 + 2 * YEAR, 8001)
    tm = 0.5 * (ts[1:] + ts[:-1])
    ref = sum(nom.Phi(t0 + 2 * YEAR, tt)[:, 3:] @ a for tt in tm) * (ts[1] - ts[0])
    assert np.abs(d2.dx - ref).max() < 1e-4 * np.abs(ref).max()
    T = np.array([t0 + 100 * DAY, t0 + 1.5 * YEAR])
    assert np.allclose(d2.states(T), np.array([d2.state_at(x) for x in T]))


def test_tow_drift_follows_clohessy_wiltshire(setup):
    """A constant along-track pull drifts Bennu along-track as y = (4a/n^2)(1 - cos nt) - (3/2) a t^2 (CW,
    circular orbit). Bennu's orbit is eccentric (e = 0.20), so the comparison is on the secular part."""
    _, nom, _ = setup
    t0 = iso_to_t("2030-01-01")
    a_T = 1e-11
    d = Deflection(nom, t0)
    vhat = lambda t: nom.state(t)[3:] / np.linalg.norm(nom.state(t)[3:])
    t_end = t0 + 8 * YEAR
    d.advance(t_end, lambda t: a_T * vhat(t))
    ts = t0 + np.linspace(1, 8, 15) * YEAR
    y = np.array([d.state_at(t)[:3] @ vhat(t) for t in ts])
    n = 2 * np.pi / (436.65 * DAY)
    cw = cw_along_track(a_T, n, ts - t0)
    assert np.all(y < 0)  # pushing forward makes Bennu fall behind (higher, slower orbit)
    ratio = y[-5:] / cw[-5:]
    assert np.all((ratio > 0.7) & (ratio < 1.3))


def test_trim_guidance_reaches_aim_point(setup):
    _, nom, pl = setup
    t_imp = iso_to_t("2030-06-22")
    dv = np.array([1.5e-6, -2.5e-6, 1e-6])  # a small impact
    d = Deflection(nom, t_imp - DAY)
    d.impulse(t_imp, dv)
    b_ki = pl.bplane.project((d.Phi(pl.t_ca, t_imp)[:, 3:] @ dv)[:3])
    trim = TrimGuidance(pl, "aim", aim_shift=1.5 * b_ki)  # the tow must add half the impact's shift
    a = G * 30_000 / 485.0**2
    t, t_end = t_imp + 5 * DAY, t_imp + 10 * YEAR
    while t < t_end:
        on, e = trim.update(d, t, a, t_end)
        t1 = min(trim.next, t_end)
        d.advance(t1, (lambda tau, e=e: a * e) if on else None)
        t = t1
    err = np.linalg.norm(trim.b_now(d, t) - trim.b_aim)
    # before 2060 one day of towing moves the 2135 flyby by thousands of km: the deadbeat trim (re-planned at
    # least every quarter day, stopping when less than half a day is left) lands within 0.5 % of the aim shift
    assert err < 5e-3 * np.linalg.norm(trim.b_aim)
    assert not trim.tow_on  # the tow switched itself off once the aim was reached


def test_steered_tow_beats_fixed_direction(setup):
    """Towing along e(t) = B(t)^T w grows the miss far more than a fixed inertial direction (which mostly
    cancels over Bennu's orbits)."""
    _, nom, pl = setup
    t0 = iso_to_t("2030-01-01")
    a = G * 30_000 / 485.0**2
    ts = np.linspace(t0, t0 + 10 * YEAR, 600)
    B = np.einsum("ij,njk->nik", pl.bplane.basis, pl.Phi_rv_to_encounter(ts))
    w = np.linalg.svd(np.trapezoid(B, ts, axis=0))[0][:, 0]
    best_fixed = a * np.linalg.svd(np.trapezoid(B, ts, axis=0), compute_uv=False)[0]
    steered = a * np.trapezoid(np.linalg.norm(np.einsum("nij,i->nj", B, w), axis=1), ts)
    assert steered > 5 * best_fixed


def test_linear_prediction_matches_nonlinear(setup):
    """Full nonlinear check vs the linear (STM) prediction of the 2135 flyby point. A vanishing push measures the
    check's own noise floor (adaptive steps, amplified by the 2060 flyby): well below the goal's scale. A
    mission-size shift carries a second-order part of a few percent, re-anchored in the live run."""
    eph, nom, pl = setup
    t_imp = iso_to_t("2030-06-22")
    res = {}
    for k, scale in (("floor", 1e-12), ("large", 1e-5)):
        d = Deflection(nom, t_imp - DAY)
        d.impulse(t_imp, scale * np.array([1.5, -2.5, 1.0]))
        res[k] = verify_nonlinear(eph, nom, d, pl, to_end=False)
    assert res["floor"]["difference_km"] < 400.0  # < 0.1 % of the Moon's distance
    shift = np.linalg.norm(res["large"]["b_linear"] - pl.b0)
    assert shift > 5e7 and res["large"]["relative"] < 0.15
    assert res["large"]["q"] > res["floor"]["q"] + 0.5 * shift  # pushed outward, as predicted


def test_sun_orbits_the_barycenter(setup):
    eph, _, _ = setup
    from core.dynamics import solar_system as SS

    t0, t1 = iso_to_t("2026-01-02"), iso_to_t("2139-12-30")  # the Step 7 span (the data now reach 2185)
    ts = np.linspace(t0, t1, 1500)
    jpl = eph.pos("sun", ts)
    d = np.linalg.norm(jpl, axis=1) / RADIUS["sun"]
    assert d.max() > 1.5 and d.min() < 0.3  # the Sun's center wanders ~0-2 solar radii from the barycenter
    est, contrib = SS.sun_from_planets(eph, ts)
    assert np.linalg.norm(est - jpl, axis=1).max() < 200e3  # center of mass of Sun + 9 planets (asteroids aside)
    jup = np.linalg.norm(contrib["jupiter_bary"], axis=1).max()
    assert 0.7e9 < jup < 0.8e9 and all(np.linalg.norm(c, axis=1).max() <= jup for c in contrib.values())
    T, R, V = SS.integrate(eph, t0, t1, n_out=1500)
    assert np.linalg.norm(R[:, 0] - eph.pos("sun", T), axis=1).max() < 100e3  # our N-body vs DE441, 114 yr
    com, E = SS.invariants(R, V)
    assert abs(E[-1] / E[0] - 1) < 1e-10


def test_solar_system_motion_through_the_milky_way():
    """The Solar System view's side view (the home button): the barycenter's velocity around the Galactic center
    (astropy's Galactocentric frame, v4.0), rotated into ICRF axes, points where its Galactic components (U, V, W) =
    (12.9, 245.6, 7.78) km/s point (the Galactocentric axes are tilted ~0.15 deg by the Sun's height above the
    plane), in Cygnus; the planets' orbital plane is tilted 60.2 deg to the Galactic plane."""
    from core.galaxy import solar_motion

    m = solar_motion()
    U, V, W = 12.9, 245.6, 7.78
    assert np.isclose(m["speed_kms"], np.sqrt(U**2 + V**2 + W**2), rtol=1e-9)
    assert abs(m["l_deg"] - np.degrees(np.arctan2(V, U))) < 0.2
    assert abs(m["b_deg"] - np.degrees(np.arcsin(W / m["speed_kms"]))) < 0.2
    assert m["constellation"] == "Cygnus"
    assert np.isclose(np.linalg.norm(m["forward"]), 1.0) and np.isclose(np.linalg.norm(m["up"]), 1.0)
    assert np.isclose(m["forward"] @ m["up"], np.sin(np.radians(m["b_deg"])), atol=1e-9)  # "up" is the Galactic pole
    assert abs(m["center"] @ m["up"]) < 1e-9  # the Galactic center lies in the plane
    assert np.isclose(m["center"] @ m["forward"], U / m["speed_kms"], atol=3e-3)  # U: the part toward the center
    assert abs(m["ecliptic_tilt_deg"] - 60.19) < 0.05


def test_live_mission_tows_to_the_aim_point(setup):
    """Keyhole mission: the recommended DART, ground tracking (re-anchored prediction), then the tractor:
    capture, and the trim tow places the 2135 flyby at the DART aim point; fast-forwarded (averaged
    station-keeping) through the years of towing; verified with the full force model to the end."""
    import time

    from core.constants import LUNAR_DISTANCE
    from core.mission import LiveMission
    from core.planner import MissionParams

    eph, nom, pl = setup
    p = MissionParams(tractor_thrust_n=20.0, tow_years=3.0, dart_sizing="p90")  # sized for 90 % confidence over beta
    res = pl.plan(p)
    m = LiveMission(eph, nom, pl, p, iso_to_t("2026-10-15"), seed=3)
    m.launch_dart(res.impactor)

    def wait():
        wall = time.time()
        while m.busy() and time.time() - wall < 300:
            time.sleep(0.2)

    while not m.dart_results and m.t < res.impactor.t_arrive + 60 * DAY:
        wait()
        m.advance(m.t + 5 * DAY)
    assert m.dart_results[0]["result"] == "success"
    tp, _ = pl.plan_tractor(p, m.t, m.b_now_abs())
    m.launch_tractor(tp)
    while not m.tow_complete and m.t < pl.t_ca:
        wait()
        pr = m.tractor.prox
        on_station = pr is not None and pr.phase in ("station", "parked")
        m.advance(m.t + (30 * DAY if on_station else DAY))
    m.finish()  # the full-model prediction of the deflected orbit (Bennu's truth flown on to 2185)
    o = m.outlook()
    fc = m.final_check
    assert m.tow_complete and fc is not None and "q" in fc
    assert np.linalg.norm(o["b_now"] - o["b_aim"]) < 1e-2 * np.linalg.norm(o["b_aim"] - pl.b0)  # trim reached the aim
    assert np.linalg.norm(o["b_aim"]) >= np.linalg.norm(pl.aim_point(p)) - 1e3  # the aim, or an overshoot held
    assert fc["q_ld"] >= p.goal_ld  # verified: the 2135 flyby is beyond the Moon's distance
    assert not np.isfinite(fc["t_impact"])  # and Bennu no longer hits Earth
    assert abs(fc["q"] - np.linalg.norm(o["b_aim"])) < 0.05 * LUNAR_DISTANCE
