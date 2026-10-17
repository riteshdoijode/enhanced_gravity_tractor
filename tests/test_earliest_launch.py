"""The DART's earliest launch ("Launch no earlier than", bound to the time slider in planning mode): the plan uses
only launches from that date on (+40 days of orbit raising), so a later date in the same window gives another arc,
a date that does not bind leaves the plan as it was, and the recommendation moves on to a window still open. Every
planned arc hits at least MIN_IMPACT_SPEED (a slower impact is outside the regime where beta is known)."""

import numpy as np
import pytest

from core.constants import DAY
from core.timeutil import iso_to_t
from tests.keyhole import setup


@pytest.fixture(scope="module")
def pl():
    return setup()[2]


def _dart(pl, date, opportunity=0):
    from core.planner import MissionParams

    return pl.plan(MissionParams(earliest_launch=date, opportunity=opportunity)).impactor


def test_later_date_in_the_same_window_gives_another_arc(pl):
    a = _dart(pl, "2026-10-15")
    b = _dart(pl, "2030-04-01")  # after the 2030 window's best launch (orbit raising from 2030-03)
    assert a.feasible and b.feasible and a.opportunity == b.opportunity == 0
    assert b.raising.t_start >= iso_to_t("2030-04-01")
    assert b.t_launch >= iso_to_t("2030-04-01") + 40 * DAY - 1.0
    assert abs(b.t_launch - a.t_launch) > 30 * DAY and abs(b.t_arrive - a.t_arrive) > 30 * DAY
    assert b.mass > a.mass  # the best arc of the window was earlier


def test_a_date_that_does_not_bind_keeps_the_plan(pl):
    """The porkchop's launch grid is anchored at the window's start: an earliest launch before the best arc only
    drops grid dates nobody uses, so the plan is the same (not shifted by the grid's phase)."""
    a = _dart(pl, "2026-10-15")
    b = _dart(pl, "2029-12-01")
    assert (b.t_launch, b.t_arrive, b.raising.t_start) == (a.t_launch, a.t_arrive, a.raising.t_start)
    assert b.mass == pytest.approx(a.mass)


def test_recommended_window_follows_the_date(pl):
    a = _dart(pl, "2026-10-15", opportunity=-1)
    b = _dart(pl, "2031-06-01", opportunity=-1)  # the 2030 window is over
    assert a.feasible and b.feasible and b.opportunity > a.opportunity
    assert b.raising.t_start >= iso_to_t("2031-06-01")


@pytest.mark.parametrize("date", ["2026-10-15", "2030-04-01", "2030-09-01", "2044-01-01"])
def test_no_slow_impact_arcs(pl, date):
    """Before this minimum the planner picked 0.65-1.1 km/s arcs late in a window (and a 0.8 km/s next-DART
    recommendation in a live run): beta there would be an extrapolation."""
    from core.planner import MIN_IMPACT_SPEED, MissionParams

    opps = pl.opportunities(MissionParams(earliest_launch=date))
    assert any(o.feasible for o in opps)
    for o in opps:
        if o.feasible:
            assert o.impact_speed >= MIN_IMPACT_SPEED and o.t_launch >= iso_to_t(date) + 40 * DAY - 1.0
            assert all(speed >= MIN_IMPACT_SPEED for _, speed, _ in o.trade)
