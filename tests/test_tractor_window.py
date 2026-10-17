"""The gravity tractor's launch window (core/planner.py tractor_launch_window: the blue band of the distance plot and
"launch window open until" in the status strip), and the live tractor flying the ground flight plan the user saw
(TractorFlight._handover warm-starts its guidance from tp.flight_plan instead of re-planning from scratch)."""

import numpy as np
import pytest

from core.constants import DAY, YEAR
from core.timeutil import iso_to_t
from tests.keyhole import setup


@pytest.fixture(scope="module")
def ks():
    return setup()


@pytest.fixture(scope="module")
def after_dart(ks):
    """The 2135 point after the planned DART and the date its verdict is due."""
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    assert ip.feasible
    return p, pl.b0 + ip.db, ip.t_arrive + 30 * DAY


def test_window_after_a_successful_dart(ks, after_dart):
    """After the planned DART only the small remaining gap is left: a launch at the verdict reaches Bennu and closes
    it, the window stays open for decades (the tow's leverage before the 2060 flyby is large), and it is closed by
    the deadline (a tractor launched within a year of the 2135 flyby cannot arrive and tow)."""
    sc, eph, pl = ks
    p, b, t_from = after_dart
    w = pl.tractor_launch_window(p, t_from, b)
    t = w["t"]
    assert len(t) == len(w["ok"]) == len(w["reach"]) == len(w["t_arrive"])
    assert np.all(np.diff(t) > 0), "launch dates are sorted"
    assert t[0] == pytest.approx(t_from) and t[-1] <= pl.t_ca - YEAR
    assert w["ok"][0], "a launch at the verdict closes the gap"
    assert not w["ok"][-1], "a launch just before the deadline does not"
    assert 0 < w["gap"] < 5e6
    assert np.isfinite(w["t_last"]) and w["t_last"] > t_from + 20 * YEAR
    # the window is one block: every date up to t_last is in it, none after
    assert np.all(w["ok"][t <= w["t_last"]]) and not np.any(w["ok"][t > w["t_last"]])
    ok = w["ok"]
    assert np.all(w["t_arrive"][ok] > t[ok]) and np.all(w["reach"][ok] >= w["gap"])


def test_window_without_a_dart_closes_before_2060(ks, after_dart):
    """Without the DART the tractor alone has the whole shift to make: only an early tow, before the 2060 flyby,
    has the leverage (the window closes decades earlier)."""
    sc, eph, pl = ks
    p, b, t_from = after_dart
    w0 = pl.tractor_launch_window(p, t_from, pl.b0)
    w1 = pl.tractor_launch_window(p, t_from, b)
    assert w0["gap"] > 10 * w1["gap"]
    assert not np.isfinite(w0["t_last"]) or w0["t_last"] < iso_to_t("2060-01-01")
    assert not np.isfinite(w0["t_last"]) or w0["t_last"] < w1["t_last"]


def test_tractor_flies_the_ground_flight_plan(ks):
    """The plan the user saw (plan_tractor_launch: the optimized profile in tp.flight_plan) is the one that flies:
    at the hand-over the guidance adopts that profile ("Flight plan: the ground profile") and the tractor arrives
    within a few days of the planned date."""
    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    t_launch = iso_to_t("2031-01-01")
    tp, _ = pl.plan_tractor_launch(p, t_launch, pl.b0)
    assert tp.feasible and tp.flight_plan is not None and tp.flight_plan["profile"]["dv"] is not None
    m = LiveMission(eph, sc.nominal, pl, p, t_launch - DAY, seed=3)
    try:
        assert m.launch_tractor(tp)
        assert any("optimized flight plan" in e.text for e in m.events)
        tr = m.tractor
        t_end = tp.t_arrive_flight + 20 * DAY
        for _ in range(20_000):
            if m.t >= t_end or tr.phase == "proximity":
                break
            m.advance(min(m.t + 5 * DAY, t_end))
        texts = [e.text for e in m.events if e.who == "tractor"]
        assert any(s.startswith("Flight plan: the ground profile") for s in texts), texts
        assert not any("re-planning" in s for s in texts), texts
        assert tr.phase == "proximity"
        assert abs(tr.t_arrived - tp.t_arrive_flight) < 3 * DAY
    finally:
        m.close()
