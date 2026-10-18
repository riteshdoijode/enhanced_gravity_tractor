"""The tractor's LIDAR scan for boulders, the choice of stone and the capture at any site (core/proximity/boulders.py,
core/proximity/ops.py scan/choose, core/mission.py slow motion, core/live_runner.py decision point)."""

import time

import numpy as np
import pytest

from core.constants import DAY
from core.timeutil import iso_to_t


@pytest.fixture(scope="module")
def setup():
    from tests.keyhole import setup as keyhole

    return keyhole()


class _Mission:  # the two things population() reads from a mission
    def __init__(self, seed, ejecta=None):
        self.seed, self.ejecta = seed, ejecta


def _env():
    from core.proximity.ops import ProxEnvironment

    return ProxEnvironment.shared()


def _at_bennu(setup, choice="ask", seed=4, pick=None):
    """A mission with the tractor handed over 30 km from Bennu (no DART: the native stones only)."""
    from core.mission import LiveMission
    from core.planner import MissionParams
    from core.proximity.ops import ProximityOps

    _, eph, pl = setup
    p = MissionParams(tractor_thrust_n=20.0)
    tp, _ = pl.plan_tractor(p, iso_to_t("2031-01-01"))
    t0 = tp.t_arrive
    m = LiveMission(eph, pl.nom, pl, p, t0, seed=seed)
    m.boulder_choice = choice
    m._boulder_pick = pick
    m.tractor_plan = tp
    tr = m.tractor
    m.t = tr.t = t0
    xb = m.bennu_helio(t0)[0]
    rho0 = -30e3 * xb[:3] / np.linalg.norm(xb[:3])
    tr.phase, tr.t_arrived, tr.arrival_offset = "proximity", t0, rho0
    tr.prox = ProximityOps(m, tr, t0, rho0, np.array([0.2, -0.1, 0.15]))
    return m, tr.prox


def _until(m, cond, t_max, chunk=600.0):
    t_end = m.t + t_max
    while not cond() and m.t < t_end:
        m.advance(m.t + chunk, max_chunk=chunk)


def test_population_is_on_the_surface_and_deterministic():
    from core.proximity import boulders

    env = _env()
    a = boulders.population(_Mission(4), env, iso_to_t("2031-06-01"))
    b = boulders.population(_Mission(4), env, iso_to_t("2031-06-01"))
    assert [s.key for s in a] == [f"native:{n}" for n, _, _ in boulders.SITES]
    assert [s.d_eq for s in a] == [s.d_eq for s in b] and [s.seed for s in a] == [s.seed for s in b]
    for s in a:
        u = s.point_b / np.linalg.norm(s.point_b)
        d, _ = env.ray.cast(1000.0 * u, -u, max_range=1000.0)
        assert abs((1000.0 - d) - np.linalg.norm(s.point_b)) < 1.0  # on the shape model
        assert s.up_b @ u > 0.7  # the local vertical points away from Bennu
        assert boulders.D_DRAW[0] <= s.d_eq <= boulders.D_DRAW[1]
        assert s.mass == pytest.approx(s.density * np.pi / 6 * s.d_eq ** 3)
    for name, lat, lon in boulders.SITES[1:]:  # the published coordinates (planetocentric, east longitude)
        s = next(x for x in a if x.name == name)
        assert abs(s.lat - lat) < 1.0 and abs((s.lon - lon + 180) % 360 - 180) < 1.0


def test_debris_lands_and_does_not_depend_on_the_tracer_count():
    """The DART debris boulders are flown with their own random stream: a re-simulation, which computes the past
    with fewer display tracers, gets the same stones in the same places."""
    from core.impact import EjectaCloud
    from core.proximity import boulders

    env = _env()
    u = np.array([np.cos(0.2), 0.0, np.sin(0.2)])
    r_b = boulders._surface_point(env, u)
    clouds = []
    for n in (40, 120):
        ej = EjectaCloud(iso_to_t("2030-06-22"), r_b, u, 2e4, env.grav, env.ray, n=n, seed=123)
        ej.thread.join(120)
        clouds.append(ej)
    f40, f120 = (c.fragments for c in clouds)
    assert len(f40) >= 3 and len(f40) == len(f120)
    for a, b in zip(f40, f120):
        assert np.allclose(a["land_b"], b["land_b"]) and a["d_eq"] == b["d_eq"]
        assert a["v_launch"] < clouds[0].v_esc  # slower than escape: it falls back
        assert abs(np.linalg.norm(a["land_b"]) - 245.0) < 40.0  # on the surface
    stones = boulders.population(_Mission(4, clouds[0]), env, iso_to_t("2031-06-01"))
    debris = [s for s in stones if s.kind == "debris"]
    assert len(debris) == len(f40) and all(s.key.startswith("debris:") for s in debris)


def test_scan_maps_bennu_and_finds_the_stones_independent_of_the_chunking(setup):
    """One Bennu day of scanning maps most of the surface and finds every native stone; the scan runs on the
    physics' own control ticks, so the result does not depend on how the mission is stepped."""
    runs = []
    for chunk in (600.0, DAY):
        m, pr = _at_bennu(setup)
        _until(m, m.awaiting_boulder, 3 * DAY, chunk)
        assert m.awaiting_boulder() and pr.phase == "choose"
        runs.append((m, pr))
    (m1, p1), (m2, p2) = runs
    assert p1.scan.coverage() > 0.85
    assert all(s.detected for s in p1.stones)  # the four candidate sites lie within the mapped latitudes
    assert np.array_equal(p1.scan.hits, p2.scan.hits)
    assert [s.mass_est for s in p1.stones] == [s.mass_est for s in p2.stones]
    for s in p1.stones:  # the estimate is within 4 sigma of the truth (size noise + unknown density)
        assert abs(s.mass_est - s.mass) < 4 * s.mass_sigma
    assert 0.1 < p1.scan.sigma_axis() < 0.5
    rec = p1.scan.recommended
    assert rec is not None and next(s for s in p1.stones if s.key == rec).feasible
    assert any("Scan complete" in e.text for e in m1.events)


def test_slow_motion_and_the_capture_at_the_chosen_stone(setup):
    """The surface sequence at the user's chosen speeds: 5 min/s for the scan, the choice, the move over the stone,
    the hover, the descent and the ascent, 1 min/s while the arm works, then the user's speed again. The capture
    (MuJoCo) succeeds on the chosen stone's own terrain, and the tow uses its real mass."""
    from core.mission import SLOWMO_ARM, SLOWMO_SURFACE

    m, pr = _at_bennu(setup)
    zones = {}

    def step():
        z = m.slow_motion()
        zones.setdefault(pr.phase, set()).add(None if z is None else z[0])
        return m.on_station()

    _until(m, lambda: step() or m.awaiting_boulder(), 3 * DAY)
    rec = pr.scan.recommended
    stone = next(s for s in pr.stones if s.key == rec)
    assert m.select_boulder(rec) and pr.chosen == rec
    assert not m.select_boulder(rec)  # one stone per run
    wall = time.time()
    _until(m, step, 3 * DAY, chunk=120.0)  # (fine enough to see the 11 minutes of the arm)
    assert time.time() - wall < 240
    assert m.on_station() and pr.has_boulder and pr.capture_result["success"]
    assert pr.mass == pytest.approx(pr.m_sc + stone.mass, rel=1e-6)  # the real mass, not the estimate
    assert zones["scan"] == {SLOWMO_SURFACE} and zones["choose"] == {SLOWMO_SURFACE}
    assert zones["descent"] == {SLOWMO_SURFACE} and zones["ascent"] == {SLOWMO_SURFACE}
    assert zones["capture"] == {SLOWMO_ARM}
    assert zones["transfer"] == {None} and zones["station"] == {None}
    assert SLOWMO_SURFACE in zones["approach"] and None in zones["approach"]  # the long approach is not slowed
    texts = [e.text for e in m.events]
    order = ["LIDAR scan", "Scan complete", f"Stone #{pr.stones.index(stone) + 1} chosen", "Surveying the capture site",
             "Descent", "Capture: hovering", "Boulder captured", "Ascent", "On station"]
    idx = [next(i for i, s in enumerate(texts) if key in s) for key in order]
    assert idx == sorted(idx)


def test_a_choice_made_before_the_scan_ends_is_applied_then(setup):
    """What a re-simulation does: the stone picked in the first run is applied when the new run's scan ends,
    without stopping for a new choice; an unknown stone falls back to the recommended one."""
    m, pr = _at_bennu(setup, pick="native:Kingfisher")
    _until(m, lambda: pr.chosen is not None, 3 * DAY)
    assert pr.chosen == "native:Kingfisher" and not m.awaiting_boulder()
    m2, pr2 = _at_bennu(setup, pick="debris:9:0.00:0.00")
    _until(m2, lambda: pr2.chosen is not None, 3 * DAY)
    assert pr2.chosen == pr2.scan.recommended
    assert any("not among this run's stones" in e.text for e in m2.events)


def test_runner_pauses_for_the_choice_and_records_it(setup):
    """The end of the scan is a decision point: the clock stops until a stone is chosen; the choice is recorded
    (re-simulations replay it) and the clock runs again."""
    from core.live_runner import LiveRunner

    m, pr = _at_bennu(setup)
    r = LiveRunner(m, 1e7)
    r.command("slowmo", (False,))  # as fast as the physics goes (the slow motion is display only)
    state = {"paused": False, "wall": time.time()}

    def commands():
        if state["paused"] or time.time() - state["wall"] > 120:
            return [("stop", ())]
        return []

    def emit(s):
        if not s["running"] and m.awaiting_boulder():
            state["paused"] = True

    r.loop(commands, emit)
    assert state["paused"] and not r.running and pr.phase == "choose"
    r.stop_requested = False
    r.command("select_boulder", (pr.scan.recommended,))
    assert r.running and r.record[-1]["name"] == "select_boulder" and r.record[-1]["arg"] == pr.scan.recommended
    assert pr.phase == "approach" and pr.chosen == pr.scan.recommended


def test_capture_succeeds_at_every_feasible_stone():
    """The MuJoCo capture on each stone's own terrain (ray-cast heightfield, the rock lying on the slope): every
    stone within the size range and the slope limit is grasped and lifted (DART debris and native boulders)."""
    from core.dynamics.asteroid import omega_body
    from core.impact import EjectaCloud
    from core.proximity import boulders
    from core.proximity.capture import Capture

    env = _env()
    t_imp = iso_to_t("2030-06-22")
    lat, lon = np.radians(-19.0), np.radians(23.0)
    u = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    ej = EjectaCloud(t_imp, boulders._surface_point(env, u), u, 2e4, env.grav, env.ray, n=40, seed=123)
    ej.thread.join(120)
    t = t_imp + 200 * DAY
    ok = [s for s in boulders.population(_Mission(4, ej), env, t)
          if boulders.D_CAPTURE[0] <= s.d_eq <= boulders.D_CAPTURE[1] and s.slope <= boulders.SLOPE_MAX]
    assert len(ok) >= 4 and any(s.kind == "debris" for s in ok)
    for s in ok:
        site = boulders.make_site(env, s, omega_body(t))
        assert abs(site.slope() - s.slope) < 2.0  # the scan judged the same slope the capture meets
        cap = Capture(10_000.0, s.mass, 19.8, site=site, seed=s.seed, aspect=s.aspect, density=s.density)
        res = cap.run()
        assert res["success"], (s.key, res["lifted_m"], cap.retries)
        assert 2.0 < res["lifted_m"] < 10.0 and res["mass"]["mass"] == pytest.approx(10_000.0 + s.mass, rel=1e-6)
