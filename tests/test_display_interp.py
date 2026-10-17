"""The displayed spacecraft follows the mission clock. The craft integrate on their own grids (the DART cruise in
12-hour steps that end past the mission clock); drawing them at the end of their step while Earth and Bennu are drawn
at the clock's time made them freeze and jump at slow warps (a sawtooth that looked like repeated course corrections).
The snapshot now interpolates the craft's state to the clock (core/mission.py: state_at, position_bary(t)); the
physics is untouched (the steps themselves are the same, see test_live_flow.py)."""

import numpy as np
import pytest

from core.constants import DAY
from tests.keyhole import setup


@pytest.fixture(scope="module")
def ks():
    return setup()


def _cruising_dart(ks, seed=5):
    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    m = LiveMission(eph, sc.nominal, pl, p, ip.raising.t_start - 10 * DAY, seed=seed)
    assert m.launch_dart(ip)
    m.advance(ip.t_arrive - 4 * DAY)  # day chunks, as at 1 day/s
    assert m.impactor.phase == "cruise"
    return m


def test_hermite_matches_the_ends_and_a_straight_line():
    from core.mission import _hermite

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    x1 = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])  # uniform motion for 10 s
    assert np.allclose(_hermite(0.0, 0.0, x0, 10.0, x1), x0)
    assert np.allclose(_hermite(10.0, 0.0, x0, 10.0, x1), x1)
    mid = _hermite(4.0, 0.0, x0, 10.0, x1)
    assert np.allclose(mid[:3], [4.0, 8.0, 12.0]) and np.allclose(mid[3:], [1.0, 2.0, 3.0])


def test_displayed_dart_moves_smoothly_at_a_slow_warp(ks):
    """Stepping the clock in 30 s frames (10 min/s on screen is 30 s of mission time per frame at 20 fps) across the
    12-hour integration steps: the displayed Bennu range decreases monotonically (no backward jump), each frame
    moves the craft by about the closing speed times 30 s (never a half-day leap), and the craft's own time is
    ahead of the clock for most frames (the overshoot the display used to show)."""
    from core.mission import _common

    m = _cruising_dart(ks)
    ki = m.impactor
    ranges, steps, ahead = [], [], 0
    prev = None
    for _ in range(2 * 24 * 120):  # 2 days in 30 s frames: four 12-hour boundaries
        m.advance(m.t + 30.0)
        s = _common(m, ki)
        ranges.append(s["range"])
        if prev is not None:
            steps.append(np.linalg.norm(s["pos"] - prev))
        prev = s["pos"]
        ahead += ki.t > m.t + 60.0
    ranges, steps = np.array(ranges), np.array(steps)
    v_rel = np.linalg.norm(ki.x[3:] - m.bennu_helio(ki.t)[0, 3:])
    assert ahead > 0.9 * len(ranges)  # the craft's step really does run ahead of the clock
    assert np.all(np.diff(ranges) < 0), "the displayed range must never jump back"
    assert np.all(np.abs(np.diff(ranges)) < 1.5 * v_rel * 30.0 + 1e3)  # no half-day leap (that would be ~1e8 m)
    assert np.all(steps < 2.0 * 30.0 * np.linalg.norm(ki.x[3:]) + 1e3)  # heliocentric speed ~30 km/s


def test_close_up_begins_at_1000_km(ks):
    """A fast run (year-long chunks) stops where the 1,000 km close-up begins: the chunk boundary at the predicted
    crossing puts the clock within a second of 1,000 km, in the 5 s/s zone."""
    from core.mission import SLOWMO_DART, SLOWMO_RANGE, SLOWMO_RANGE_DART

    m = _cruising_dart(ks, seed=7)
    zones = []
    while m.t < m.impactor.ip.t_arrive and (m.slow_motion() or (None,))[0] != SLOWMO_RANGE_DART[0]:
        z = m.slow_motion()
        zones.append(z[0] if z else None)
        m.advance(m.t + 365 * DAY, stop=lambda: (m.slow_motion() or (None,))[0] != zones[-1])
    rng_, closing = m.craft_range(m.impactor)
    assert m.slow_motion() == SLOWMO_RANGE_DART
    assert SLOWMO_RANGE - 2.0 * closing <= rng_ <= SLOWMO_RANGE  # stopped at the crossing (to a step of the clock)
    assert m.impactor.t_go > SLOWMO_DART[0][0]  # well before the last 10 s
    m.close()


def test_physics_unchanged_by_the_display_queries(ks):
    """Asking for the displayed state every 30 s (snapshots) leaves the DART's own integration steps alone: the
    same impact as a run queried once a day."""
    from core.mission import _common

    out = []
    for query in (DAY, 30.0):
        m = _cruising_dart(ks, seed=6)
        t_end = m.impactor.ip.t_arrive - 2.5 * DAY
        while m.t < t_end:
            m.advance(m.t + query)
            _common(m, m.impactor)
        m.advance(t_end + DAY)  # the same clock time for both runs
        out.append((m.impactor.t, m.impactor.x.copy()))
        m.close()
    assert out[0][0] == out[1][0] and np.array_equal(out[0][1], out[1][1])
