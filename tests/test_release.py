"""End of the tow: the 'release' command (and the tow_years end) lets the boulder go, the tractor departs, the pull on
Bennu stops, and Bennu's full-model prediction is requested."""

import numpy as np
import pytest

from core.constants import DAY, YEAR
from core.timeutil import iso_to_t


@pytest.fixture(scope="module")
def setup():
    from tests.keyhole import setup as keyhole

    sc, eph, pl = keyhole()
    return eph, sc.nominal, pl


def _towing_mission(setup, seed=4):
    """A tractor already on station with the boulder, towing toward the aim point (no DART: the tow never ends by
    itself within the test)."""
    from core.mission import LiveMission
    from core.planner import MissionParams
    from core.proximity.ops import ProximityOps

    eph, nom, pl = setup
    p = MissionParams(tractor_thrust_n=20.0, tow_years=10.0)
    tp, _ = pl.plan_tractor(p, iso_to_t("2031-01-01"))
    t0 = tp.t_arrive
    m = LiveMission(eph, nom, pl, p, t0, seed=seed)
    m.tractor_plan = tp
    tr = m.tractor
    m.t = tr.t = t0
    xb = m.bennu_helio(t0)[0]
    tr.phase, tr.t_arrived = "proximity", t0
    tr.prox = pr = ProximityOps(m, tr, t0, -30e3 * xb[:3] / np.linalg.norm(xb[:3]), np.zeros(3))
    pr.has_boulder, pr.sc_shape = True, pr.capture.craft_shape(True)
    pr.phase_done["capture"] = True
    pr.x = np.concatenate([pr.d_station * pr.e_station, np.zeros(3)])
    pr.ekf.x[:6] = pr.x
    pr._set_phase("station")
    pr.t_station = pr.t
    pr._tow_averaged = True
    for _ in range(30):
        m.advance(m.t + DAY)
    assert m.trim is not None and m.trim.tow_on and pr.phase == "station"
    return m, pr


def test_release_command_ends_the_tow(setup):
    from core.live_runner import LiveRunner
    from core.proximity.ops import DEPART_RADIUS

    m, pr = _towing_mission(setup)
    r = LiveRunner(m, DAY)
    mass_with = pr.mass
    dv_before = m.defl.tow_dv
    pushes_before = len(m.truth.pushes)
    r.command("release", ())
    assert m.tow_complete and pr.phase == "release" and r.record[-1]["name"] == "release"
    assert not pr.has_boulder and pr.mass < mass_with and pr.boulder_free is not None
    assert any("Boulder released" in e.text for e in m.events)
    assert any("Tow complete (released on command)" in e.text for e in m.events)
    phases = []
    while m.t < m.tow_end + 12 * DAY:
        m.advance(m.t + DAY)
        if not phases or phases[-1] != pr.phase:
            phases.append(pr.phase)
    assert phases[-1] == "departed" and "depart" in phases
    assert np.linalg.norm(pr.x[:3]) > DEPART_RADIUS
    assert m.defl.tow_dv == dv_before, "the pull on Bennu stops at the release"
    assert len(m.truth.pushes) <= pushes_before + 1  # at most the block in progress, closed at the release
    assert m.truth.speculate  # the full-model prediction was asked for
    assert not m.end_tow()  # nothing left to release
    m.close()


def test_tow_years_end_releases_the_boulder(setup):
    m, pr = _towing_mission(setup, seed=5)
    m.tow_end = m.t + 3 * DAY  # the tow time runs out
    while not m.tow_complete and m.t < m.tow_end + 2 * DAY:
        m.advance(m.t + DAY)
    assert m.tow_complete and not pr.has_boulder
    assert any("year tow time is up" in e.text for e in m.events)
    m.close()


def test_release_phases_in_the_snapshot(setup):
    """The snapshot carries the released boulder and the departure for the views and the panels."""
    from core.mission import snapshot

    m, pr = _towing_mission(setup, seed=6)
    assert m.end_tow()
    m.advance(m.t + DAY)
    s = snapshot(m)["tractor"]["prox"]
    assert s["phase"] in ("release", "depart") and s["boulder_free"] is not None
    assert any(g[0] == "boulder" for g in s["geoms"])  # drawn where it was let go
    assert np.isfinite(s["t_release"])
    m.close()
