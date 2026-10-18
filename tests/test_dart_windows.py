"""DART window markers on the distance plot (Planner.window_capability): what one DART as set (mass, max speed,
beta, sizing) does in every launch window, as the plan for that window would fly it, and its class:
reaches the target / clears the goal only / falls short / no arc."""

from dataclasses import replace

import numpy as np
import pytest

from core.constants import LUNAR_DISTANCE
from core.planner import MissionParams
from core.timeutil import iso_to_t
from tests.keyhole import setup

RANK = {"none": 0, "short": 1, "goal": 2, "target": 3}  # worse -> better
MASSES = (30.0, 100.0, 191.0, 600.0, 2e3, 2e4, 3e5, 1e7)


@pytest.fixture(scope="module")
def ks():
    return setup()


def _classes(pl, p, mass, b_now=None, contact=True):
    opps = pl.opportunities(p, b_now=b_now)
    pl.window_capability(p, opps, mass, b_now=b_now, contact=contact)
    return {o.index: o.capability["status"] for o in opps}


def _live_state(pl, p):
    """A DART that fell short (40 % of the target) and a beta measured from tracking."""
    d, _ = pl.push_direction(p)
    return replace(p, beta=2.9, beta_sigma=0.1), pl.b0 + 0.4 * pl.target_shift(p) * d


def test_heavier_dart_turns_at_least_as_many_windows_green(ks):
    """A heavier DART never makes a window's class worse, so the green set only grows; a 30 kg DART reaches the
    target nowhere and a 10,000 t one everywhere it has an arc (planned and 90 %-confidence sizing, and the next
    DART after a shortfall)."""
    _, _, pl = ks
    p0 = MissionParams()
    for p, b_now in [(p0, None), (replace(p0, dart_sizing="p90"), None), _live_state(pl, p0)]:
        prev, greens = None, []
        for m in MASSES:
            cls = _classes(pl, p, m, b_now)
            if prev is not None:
                assert all(RANK[cls[k]] >= RANK[prev[k]] for k in cls), (p.dart_sizing, m)
            greens.append(sum(c == "target" for c in cls.values()))
            prev = cls
        assert greens == sorted(greens)
        assert greens[0] == 0 and greens[-1] == sum(c != "none" for c in prev.values()) > 5


def test_lower_max_speed_can_only_turn_windows_red(ks):
    """Lowering the orbit-raising cut-off only removes arcs: each window's best head-on shift (which picks its arc)
    never grows, a window without an arc stays without one, and with head-on hits no class ever improves.
    (With each arc's point of contact the class can occasionally improve at a lower speed: the planner picks the
    arc by its head-on shift, and a slower arc can present a better face of Bennu, e.g. 2.28x vs 1.23x head-on
    for the 2030 DART at 11.25 vs 11.75 km/s.)"""
    _, _, pl = ks
    speeds = (16.5, 14.0, 12.5, 12.0, 11.75, 11.5, 11.25, 11.0)
    for m in (191.0, 2e3, 3e5):
        prev = prev_gain = None
        for v in speeds:
            p = MissionParams(dart_max_speed_kms=v)
            opps = pl.opportunities(p)
            gain = {o.index: o.gain_per_kg if o.feasible else 0.0 for o in opps}
            head_on = _classes(pl, p, m, contact=False)
            aimed = _classes(pl, p, m)
            if prev is not None:
                assert all(gain[k] <= prev_gain[k] * (1 + 1e-12) for k in gain), v
                assert all(RANK[head_on[k]] <= RANK[prev[k]] for k in head_on), (m, v)
                assert all(aimed[k] == "none" for k in aimed if prev_none[k]), (m, v)
            prev, prev_gain, prev_none = head_on, gain, {k: c == "none" for k, c in aimed.items()}
    assert any(c == "none" for c in _classes(pl, MissionParams(dart_max_speed_kms=11.0), 191.0).values())


def test_colour_thresholds_match_the_success_logic(ks):
    """Green is exactly the DART verdict's success (dart_result: at least 90 % of the target, earlier DARTs
    included); yellow is a 2135 flyby beyond the goal without that; the class flips at the masses the tooltip
    gives. The selected window's shift is the plan's own."""
    _, _, pl = ks
    p0 = MissionParams()
    t_now = iso_to_t("2031-01-01")
    for p, b_now in [(p0, None), (replace(p0, dart_sizing="p90"), None), _live_state(pl, p0)]:
        b = pl.b0 if b_now is None else b_now
        opps = pl.opportunities(p, b_now=b_now)
        for m in np.geomspace(20.0, 2e7, 60):
            pl.window_capability(p, opps, m, b_now=b_now)
            for o in opps:
                c = o.capability
                if c["status"] == "none":
                    continue
                verdict = pl.dart_result(p, b + m * c["db_per_kg"], t_now, tractor_launched=True)
                assert (c["status"] == "target") == verdict["success"]
                assert (c["status"] in ("target", "goal")) == (verdict["success"] or verdict["q"] >= verdict["goal"])
                assert np.isclose(c["q"], verdict["q"], rtol=1e-12)
        # the class flips at the masses given in the tooltip
        o = next(o for o in opps if o.feasible)
        pl.window_capability(p, [o], 1.0, b_now=b_now)
        c = o.capability
        for m_edge, below, above in [(c["mass_success"], "goal", "target"), (c["mass_goal"], "short", "goal")]:
            for f, want in [(1 - 1e-6, below), (1 + 1e-6, above)]:
                pl.window_capability(p, [o], m_edge * f, b_now=b_now)
                assert o.capability["status"] == want, (p.dart_sizing, m_edge, f)
    # the selected window: the marker's shift is the plan's (its arc, its point of contact), and with the
    # recommended mass the window reaches the target
    res = pl.plan(p0)
    ip = res.impactor
    pl.window_capability(p0, res.opportunities, ip.mass)
    sel = next(o for o in res.opportunities if o.index == ip.opportunity)
    assert np.isclose(sel.capability["shift"], float(ip.db @ res.direction), rtol=1e-9)
    assert np.isclose(sel.capability["q"], np.linalg.norm(res.b0 + ip.db), rtol=1e-9)
    assert sel.capability["status"] == "target" and sel.capability["q"] >= LUNAR_DISTANCE
    assert np.isclose(sel.capability["mass_target"], ip.mass_linear, rtol=1e-2)  # porkchop B on whole days
    assert sel.capability["p_target"] >= 0.5  # the planned beta is the median


def test_full_model_overshoot_is_a_success(ks):
    """A DART so large that the 2135 encounter no longer happens (session 2026-10-18: 9,400 kg, full-model flyby at
    691 LD) projects onto the B-plane behind b0 along the push direction. With the full-model flyby distance the
    verdict is an overshoot (a success, positive fraction, nothing remaining); without it, or below the aim
    distance, the linear verdict is unchanged."""
    _, _, pl = ks
    p = MissionParams(dart_target_km=319_000.0)
    t_now = iso_to_t("2043-01-01")
    d, _ = pl.push_direction(p)
    target, aim = pl.target_shift(p), float(np.linalg.norm(pl.aim_point(p)))
    b_far = pl.b0 - 87e9 * d  # what the 691 LD full-model flyby projected to (shift along d: -87 million km)
    old = pl.dart_result(p, b_far, t_now, tractor_launched=True)
    assert not old["success"] and old["fraction"] < 0 and not old["overshoot"]
    r = pl.dart_result(p, b_far, t_now, tractor_launched=True, q_full=691.0 * LUNAR_DISTANCE)
    assert r["success"] and r["overshoot"] and r["remaining"] == 0.0 and r["need_goal"] == 0.0
    assert r["q"] == 691.0 * LUNAR_DISTANCE and r["fraction"] > 100.0
    assert np.isclose(np.linalg.norm(pl.b0 + r["achieved"] * d), r["q"], rtol=1e-9)
    # a short DART stays short when the full model agrees it is short of the aim distance
    b_short = pl.b0 + 0.5 * target * d
    s = pl.dart_result(p, b_short, t_now, tractor_launched=True, q_full=0.999 * aim)
    assert not s["success"] and not s["overshoot"]
    assert s == {**pl.dart_result(p, b_short, t_now, tractor_launched=True), "tractor_reach": s["tractor_reach"]}
