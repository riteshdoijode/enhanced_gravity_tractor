"""Bennu hits Earth in 2182 (core/earth_impact.py): the approach track, the impact site and numbers (Collins et al.
2005 entry and crater scaling), the ballistic plume, and the display's slow-motion zones."""

import numpy as np
import pytest

from core.constants import BENNU, DAY, RADIUS
from core.earth_impact import (MT_TNT, SHOW_AFTER, SHOW_BEFORE, EarthImpact, entry, final_crater,
                               transient_crater)
from tests.keyhole import setup


@pytest.fixture(scope="module")
def imp():
    sc, eph, _ = setup()
    return EarthImpact.from_scenario(eph, sc), sc


def test_track_ends_at_the_surface(imp):
    e, sc = imp
    assert abs(e.t_hit - sc.t_impact) < 120.0  # the scenario's impact time (its track stops 60 s above)
    assert e.altitude(e.t_hit) == pytest.approx(0.0, abs=50.0)
    assert e.altitude(e.t_hit - SHOW_BEFORE) > 1e9  # two days out: about a million km
    alt = [e.altitude(t) for t in np.linspace(e.t_hit - SHOW_BEFORE, e.t_hit, 400)]
    assert np.all(np.diff(alt) < 0)  # straight in
    assert 12e3 < np.linalg.norm(e.x_track[-1, 3:]) < 13.5e3


def test_site_and_numbers(imp):
    e, sc = imp
    nb = e.numbers
    # the scenario quotes the sub-Bennu point 60 s before the impact: within a couple of degrees of the site
    assert abs(nb["lat"] - sc.impact_latlon[0]) < 2.5 and abs(nb["lon"] - sc.impact_latlon[1]) < 2.5
    assert nb["energy_j"] == pytest.approx(0.5 * BENNU["mass"] * nb["v_entry"] ** 2)
    assert 1000 < nb["energy_mt"] < 2000 and nb["energy_mt"] == pytest.approx(nb["energy_j"] / MT_TNT)
    assert 0 < nb["angle_deg"] <= 90
    ent = nb["entry"]
    assert 40e3 < ent["z_star"] < 100e3 and not ent["airburst"]  # a 0.5 km body breaks up high but lands together
    assert 0.9 * nb["v_entry"] < ent["v_surface"] < nb["v_entry"]
    water = e.set_surface(True)
    assert water["water"] and water["final_m"] is None and 3e3 < water["transient_m"] < 15e3
    rock = e.set_surface(False)
    assert rock["final_m"] > rock["transient_m"]


def test_collins_scaling():
    """Eq. 21 (rock 1.161 / water 1.365), Eqs. 22 and 27, and the entry model's limits."""
    D = transient_crater(100.0, 3000.0, 20e3, np.pi / 4)
    assert D == pytest.approx(1.161 * (3000 / 2750) ** (1 / 3) * 100**0.78 * 20e3**0.44 * 9.81**-0.22
                              * np.sin(np.pi / 4) ** (1 / 3))
    assert transient_crater(100.0, 3000.0, 20e3, np.pi / 4, water=True) > D
    assert final_crater(2000.0) == pytest.approx(2500.0)  # simple: 1.25 D_tc
    assert final_crater(10e3) == pytest.approx(1e3 * 1.17 * 10**1.13 / 3.2**0.13)  # complex (km)
    small = entry(20.0, 3000.0, 20e3, np.pi / 4)  # a Chelyabinsk-sized stone: airburst
    assert small["airburst"] and small["z_b"] > 0
    iron = entry(50.0, 8000.0, 15e3, np.pi / 2)
    assert iron["z_star"] < small["z_star"]  # stronger and denser: breaks up lower


def test_plume_is_ballistic(imp):
    e, _ = imp
    pl = e.plume()
    pos = pl["pos"]
    r = np.linalg.norm(pos, axis=2)
    R = RADIUS["earth"]
    assert np.allclose(r[0], R, rtol=1e-5)
    assert r.max() > R + 500e3  # the fast ones climb hundreds of km
    assert np.all(r >= R * (1 - 1e-6))  # nothing below the surface: landed particles stay on it
    landed = np.isclose(r[-1], R, rtol=1e-9)
    assert landed.any() and (~landed).any()  # the slow ones are back down after 20 minutes, the fast ones not yet
    assert e.plume_at(e.t_hit - 1.0) is None and np.array_equal(e.plume_at(e.t_hit + 1e6), pos[-1])


def test_slow_motion_zones(imp):
    e, _ = imp
    t_c = e.t_slowmo()
    assert 60.0 < e.t_hit - t_c < 100.0  # the last 1,000 km take about 80 s
    assert e.slow_motion(t_c - 10.0) is None
    assert e.slow_motion(t_c + 1.0)[0] == 5.0
    assert e.slow_motion(e.t_hit + 60.0)[0] == 60.0
    assert e.slow_motion(e.t_hit + SHOW_AFTER + 1.0) is None
    assert e.slow_motion(e.t_hit - 2 * DAY) is None
