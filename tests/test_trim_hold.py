"""Parked stays parked. Once the trim law has placed the 2135 flyby at the aim point (or holds an overshoot) the
B-plane error is numerical noise of a few metres; before the fix its steepest-descent direction g / |g| flipped sign
from week to week and the parked tractor was sent on a 175-degree transfer around Bennu every week (session log of
2026-10-17). Now the direction is kept while the tow is off and the parked tractor keeps its hold point and its
boulder: the tow ends only on the release command (or half a year before the 2135 flyby). Before 2026-10-18 the
mission released the boulder by itself after 60 days of holding, which at 1 month/s took two seconds and looked
like the boulder vanishing."""

import numpy as np
import pytest

from core.constants import DAY, G, YEAR
from core.timeutil import iso_to_t
from core.tow import Deflection, TrimGuidance


@pytest.fixture(scope="module")
def setup():
    from tests.keyhole import setup as keyhole

    sc, eph, pl = keyhole()
    return eph, sc.nominal, pl


def _tow_to_the_aim(nom, pl):
    t_imp = iso_to_t("2030-06-22")
    dv = np.array([1.5e-6, -2.5e-6, 1e-6])
    d = Deflection(nom, t_imp - DAY)
    d.impulse(t_imp, dv)
    b_ki = pl.bplane.project((d.Phi(pl.t_ca, t_imp)[:, 3:] @ dv)[:3])
    trim = TrimGuidance(pl, "aim", aim_shift=1.5 * b_ki)
    a = G * 30_000 / 485.0**2
    t, t_end = t_imp + 5 * DAY, t_imp + 10 * YEAR
    while t < t_end:
        on, e = trim.update(d, t, a, t_end)
        if not on:
            break
        t1 = min(trim.next, t_end)
        d.advance(t1, lambda tau, e=e: a * e)
        t = t1
    assert not trim.tow_on
    return d, trim, a, t, t_end


def test_parked_direction_does_not_flip(setup):
    """20 weekly re-plans after the aim is reached: the tow stays off and the direction never changes (before the
    fix it flipped by ~180 degrees between updates)."""
    _, nom, pl = setup
    d, trim, a, t, t_end = _tow_to_the_aim(nom, pl)
    e0 = trim.e.copy()
    for _ in range(20):
        d.advance(trim.next)
        t = trim.next
        on, e = trim.update(d, t, a, t_end)
        assert not on
        assert np.allclose(e, e0), "the parked tractor's direction must not change"
    # (the deadbeat tow stops with less than half a day of towing left, so tens of km of error remain: the hold
    # is decided by tow_on, not by the error floor)


def test_nothing_to_tow_keeps_the_station_direction(setup):
    """An overshoot held from the start (aim = where the flyby already is): the error is rounding noise, so the law
    gives no direction (before the fix its noise direction sent the tractor 180 degrees around Bennu to park)."""
    _, nom, pl = setup
    t_imp = iso_to_t("2030-06-22")
    d = Deflection(nom, t_imp - DAY)
    d.impulse(t_imp, np.array([1.5e-6, -2.5e-6, 1e-6]))
    t = t_imp + 30 * DAY
    trim = TrimGuidance(pl, "aim", aim_shift=TrimGuidance(pl, "aim").b_now(d, t) + 1e-9)
    on, e = trim.update(d, t, G * 30_000 / 485.0**2, t + 10 * YEAR)
    assert not on and e is None


def test_parked_tractor_keeps_its_hold_point_and_boulder(setup):
    """Mission level: a tractor parked by the trim law keeps its 5 km hold point and its boulder over 20 weekly
    updates (well past the old 60-day automatic release); the release command then leaves the boulder there."""
    from core.mission import LiveMission
    from core.planner import MissionParams
    from core.proximity.ops import D_PARK, ProximityOps

    eph, nom, pl = setup
    p = MissionParams(tractor_thrust_n=20.0)
    tp, _ = pl.plan_tractor(p, iso_to_t("2031-01-01"))
    t0 = tp.t_arrive
    m = LiveMission(eph, nom, pl, p, t0, seed=4)
    m.tractor_plan = tp
    tr = m.tractor
    m.t = tr.t = t0
    xb = m.bennu_helio(t0)[0]
    tr.phase, tr.t_arrived = "proximity", t0
    tr.prox = pr = ProximityOps(m, tr, t0, -30e3 * xb[:3] / np.linalg.norm(xb[:3]), np.zeros(3))
    # straight onto the station with the boulder, the aim already reached (aim shift 0: nothing more to tow)
    pr.has_boulder, pr.sc_shape = True, pr.capture.craft_shape(True)
    pr.phase_done["capture"] = True
    pr.x = np.concatenate([pr.d_station * pr.e_station, np.zeros(3)])
    pr.ekf.x[:6] = pr.x
    pr._set_phase("station")
    pr.t_station = pr.t
    pr._tow_averaged = True
    m.trim = TrimGuidance(pl, "aim", aim_shift=np.zeros(2))
    m.tow_start, m.tow_end = t0, t0 + 10 * YEAR
    e_station = pr.e_station.copy()
    while pr.phase != "parked" and m.t < t0 + 30 * DAY:
        m.advance(m.t + DAY)
    assert pr.phase == "parked"
    assert np.allclose(pr.e_station, e_station), "nothing to tow: the tractor parks straight out from its station"
    e_park = pr.e_station.copy()
    n_transfers = sum("Transfer" in e.text for e in m.events)
    t_parked = m.t
    while m.t < t_parked + 20 * 7 * DAY:
        m.advance(m.t + DAY)
    assert sum("Transfer" in e.text for e in m.events) == n_transfers, "a parked tractor must not be re-parked"
    assert np.allclose(pr.e_station, e_park)
    assert not m.tow_complete and pr.phase == "parked" and pr.has_boulder, "parked: the boulder stays aboard"
    assert not any("Tow complete" in e.text for e in m.events)
    assert np.linalg.norm(pr.x[:3]) == pytest.approx(D_PARK, rel=0.01)
    assert m.end_tow()  # what the ⤒ Release button does
    assert m.tow_complete and not pr.has_boulder and pr.boulder_free is not None
    assert np.linalg.norm(pr.boulder_free[1]) == pytest.approx(D_PARK, rel=0.05)  # left at the parking point
    m.close()
