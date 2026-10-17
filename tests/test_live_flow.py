"""Live-run flow: the physics never depends on the display (same mission, any chunking -> the same impact), the
clock runs at exactly the chosen speed except for the slow motion at the impact, a parameter change re-flies
the mission with the new physics (and the same parameters reproduce the same run), the ejecta move smoothly,
and the simulation process starts a new run cleanly even when the previous one is ended at the same moment."""

import multiprocessing as mp
import queue
import time
from dataclasses import replace

import numpy as np
import pytest

from core.constants import DAY
from tests.keyhole import setup


@pytest.fixture(scope="module")
def ks():
    return setup()


def _to_impact(m, step):
    t_end = m.impactor.ip.t_arrive + 5 * DAY
    while m.impactor.phase != "impacted" and m.t < t_end:
        m.advance(m.t + step)
    im = m.impactor.impact
    return im["t"], im["point_b"], im["dv_bennu"]


def test_physics_does_not_depend_on_the_display_speed(ks):
    """The same mission stepped in 3-hour chunks and in ~week chunks (two display speeds) hits Bennu at the same
    instant, at the same point, with the same momentum: every craft steps on its own physical grid."""
    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    out = []
    for step in (3 * 3600.0, 7 * DAY + 1234.5):
        m = LiveMission(eph, sc.nominal, pl, p, ip.raising.t_start - 10 * DAY, seed=3)
        assert m.launch_dart(ip)
        out.append(_to_impact(m, step))
        m.close()
    (t1, x1, v1), (t2, x2, v2) = out
    assert t1 == t2 and np.array_equal(x1, x2) and np.array_equal(v1, v2)


def test_clock_runs_at_the_chosen_speed(ks):
    """No automatic speed changes: through the orbit raising and the cruise the clock advances at exactly the
    requested warp (the physics keeps up there)."""
    from core.live_runner import LiveRunner
    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    factory = lambda q, t0: LiveMission(eph, sc.nominal, pl, q, t0, seed=1)
    r = LiveRunner(factory(p, ip.raising.t_start - 10 * DAY), 7 * DAY, factory=factory)
    r.command("launch_dart", (ip,))
    snaps, wall = [], time.time()

    def cmds():
        return [("end", ())] if time.time() - wall > 6.0 else []

    r.loop(cmds, lambda s: snaps.append(s) or True)
    rates = [s["warp_achieved"] for s in snaps[3:] if s["running"] and not s.get("busy")]
    assert len(rates) > 40 and not hasattr(r.mission, "max_warp")
    assert np.median(rates) == pytest.approx(7 * DAY, rel=0.01)
    assert all(not s["behind"] for s in snaps[3:])
    r.mission.close()


def test_slow_motion_at_the_impact(ks):
    """The display slows down for the DART's last minutes and the impact (if the user keeps that on): a fast
    run stops exactly where the close-up begins, then the zones follow in order and end two days later."""
    from core.mission import SLOWMO_DART, LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    m = LiveMission(eph, sc.nominal, pl, p, ip.raising.t_start - 10 * DAY, seed=2)
    assert m.launch_dart(ip)
    while m.slow_motion() is None and m.t < ip.t_arrive + DAY:  # one year per step, as at 1 year/s
        m.advance(m.t + 365 * DAY, stop=lambda: m.slow_motion() is not None)
    assert m.impactor.phase == "terminal" and m.slow_motion()[0] == SLOWMO_DART[1][1]
    assert SLOWMO_DART[1][0] - 15.0 < m.impactor.t_go <= SLOWMO_DART[1][0]  # caught at the start of the close-up
    seen = []
    while m.t < m.impactor.ip.t_arrive + 2 * DAY:
        z = m.slow_motion()
        if not seen or seen[-1] != (z[0] if z else None):
            seen.append(z[0] if z else None)
        m.advance(m.t + (z[0] * 0.05 if z else DAY))  # 20 frames per second at the slow-motion warp
    assert seen == [30.0, 1.0, 2.0, 60.0, None]
    m.close()


def test_parameter_change_reflies_the_mission(ks):
    """A parameter change during the run flies the whole mission again from its start with the new physics:
    a different beta changes Bennu's true beta and the impulse; the same parameters give the same run back."""
    from core.live_runner import LiveRunner
    from core.mission import LiveMission
    from core.planner import MissionParams

    sc, eph, pl = ks
    p = MissionParams()
    ip = pl.plan(p).impactor
    t0 = ip.raising.t_start - 10 * DAY
    factory = lambda q, t: LiveMission(eph, sc.nominal, pl, q, t, seed=4)
    r = LiveRunner(factory(p, t0), DAY, factory=factory)
    r.command("launch_dart", (ip,))
    m0 = r.mission
    _to_impact(m0, 7 * DAY)
    first = (m0.beta_true, m0.bennu_impulses[0][1].copy())
    t_back = m0.t + DAY

    def refly(q, plan):
        r.resim(q, plan, t0, t_back)
        wall = time.time()
        while r.replay is not None and time.time() - wall < 300:
            r._replay_step()
        assert r.replay is None
        m = r.mission
        assert m.t == pytest.approx(t_back, abs=DAY) and m.impactor.phase == "impacted"
        return m.beta_true, m.bennu_impulses[0][1].copy(), m

    q = replace(p, beta=2.0)
    beta2, dv2, m2 = refly(q, pl.plan(q).impactor)
    assert beta2 != first[0] and not np.allclose(dv2, first[1])
    assert m2.p.beta == 2.0 and abs(m2.impactor.mass - pl.plan(q).impactor.mass) < 1e-9
    beta3, dv3, m3 = refly(p, ip)  # back to the original parameters: the original run, bit for bit
    assert beta3 == first[0] and np.array_equal(dv3, first[1])
    m3.close()


def test_ejecta_frames_are_interpolated():
    from core.impact import EjectaCloud
    from core.proximity.ops import ProxEnvironment

    env = ProxEnvironment.shared()
    r_b = env.site_origin.copy()
    ej = EjectaCloud(0.0, r_b, r_b / np.linalg.norm(r_b), 2e6, env.grav, env.ray, n=60, seed=2, duration=3600.0)
    ej.thread.join(60)
    assert ej.done and ej.t_computed >= 3600.0
    (t0, x0, c0), (t1, x1, c1) = ej.frames[3], ej.frames[4]
    assert t1 - t0 == pytest.approx(20.0, abs=1e-6)  # 20 s frames during the first half hour
    pos, code = ej.at(0.5 * (t0 + t1))
    fly = (c0 == 0) & (c1 == 0)
    assert np.allclose(pos[fly], 0.5 * (x0[fly] + x1[fly]), atol=1e-2)


def test_sim_process_starts_a_new_run_while_ending_the_last():
    """'end' and the next 'start' arrive together (the user ends a run and launches again at once): the old run
    ends, the new one runs, and stale snapshots are told apart by their run id."""
    from core.live_runner import process_main
    from core.planner import MissionParams
    from core.timeutil import iso_to_t

    ctx = mp.get_context("spawn")
    cmd_q, snap_q, pending = ctx.Queue(), ctx.Queue(), ctx.Value("i", 0)
    proc = ctx.Process(target=process_main, args=(cmd_q, snap_q, pending), daemon=False)
    proc.start()

    def wait_for(pred, limit=120.0):
        t = time.time()
        while time.time() - t < limit:
            try:
                msg = snap_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg[0] == "snap":
                with pending.get_lock():
                    pending.value -= 1
            if msg[0] == "error":
                raise AssertionError(msg[-1])
            if pred(msg):
                return msg
        raise AssertionError("timed out")

    try:
        wait_for(lambda m: m[0] == "ready")
        t0 = iso_to_t("2030-01-01")
        cmd_q.put(("start", 1, MissionParams(), t0, 1, "MPC", DAY, []))
        wait_for(lambda m: m[0] == "snap" and m[1] == 1)
        cmd_q.put(("end",))
        cmd_q.put(("start", 2, MissionParams(), t0 + 100 * DAY, 2, "MPC", DAY, []))
        seen = []
        msg = wait_for(lambda m: seen.append(m[:2]) or (m[0] == "snap" and m[1] == 2))
        assert msg[2]["t"] >= t0 + 100 * DAY
        assert ("ended", 1) in seen
    finally:
        cmd_q.put(("quit",))
        proc.join(20)
        if proc.is_alive():
            proc.terminate()


def test_refly_at_a_verdict_reaches_it_and_keeps_the_tractor(ks):
    """The run pauses on a DART verdict for the decision; there the tractor is launched, then a setting changes and
    the mission is flown again from its start. The clock had stopped exactly on the verdict, where advance() alone
    does not process it: the re-fly must reach that verdict again (its orbit determination is computed anew when
    the DART differs) and replay the tractor launch after it. Which settings re-plan DART #1 (dart_params): not the
    tractor's. As in the app, Bennu's full-model orbit is flown in background processes (inline, a prediction is
    ready at once and the pending verdict never shows)."""
    from core.fullmodel import FullModelPool
    from core.live_runner import LiveRunner
    from core.mission import LiveMission
    from core.planner import MissionParams, dart_params

    sc, eph, pl = ks
    rec = pl.plan(MissionParams()).impactor
    p = MissionParams(use_recommended_mass=False, impactor_mass_kg=2.0 * rec.mass)  # a success at any drawn beta
    assert dart_params(replace(p, tractor_thrust_n=30.0, boulder_mass_kg=5e3, tow_years=3.0)) == dart_params(p)
    assert dart_params(replace(p, beta=3.0)) != dart_params(p)
    assert dart_params(replace(p, refined_mass_kg=123.0)) == dart_params(p)  # the plan carries the full-model mass
    ip = pl.plan(p).impactor
    t0 = ip.raising.t_start - 10 * DAY
    pool = FullModelPool(2)

    def factory(q, t):
        mission = LiveMission(eph, sc.nominal, pl, q, t, seed=4)
        mission.fullmodel = pool
        return mission

    r = None
    try:
        r = LiveRunner(factory(p, t0), DAY, factory=factory)
        r.command("launch_dart", (ip,))
        m = r.mission
        while m.impactor.phase != "impacted":
            if m.busy():
                time.sleep(0.2)
            m.advance(m.t + 7 * DAY)
        wall = time.time()
        while not m.dart_results and time.time() - wall < 600:
            if m.busy():
                time.sleep(0.2)
            m.advance(m.t + 5 * DAY)
        assert m.dart_ok()
        t_verdict = m.t  # the clock stops on the verdict (a decision point)
        r.command("launch_tractor", (None, p))
        assert np.isfinite(m.tractor.t_launch)

        q = replace(p, impactor_mass_kg=2.3 * rec.mass, tractor_thrust_n=30.0)  # a heavier DART: a new verdict
        ip2 = pl.plan(q).impactor
        assert ip2.t_arrive == ip.t_arrive  # same arc, so the verdict falls on the same instant
        r.resim(q, ip2, t0, t_verdict)
        wall = time.time()
        while r.replay is not None and time.time() - wall < 600:
            r._replay_step()
            time.sleep(0.01)
        assert r.replay is None
        m2 = r.mission
        assert m2.t == pytest.approx(t_verdict, abs=1.0) and m2.impactors[0].mass == ip2.mass
        assert m2.dart_ok() and len(m2.dart_results) == 1
        assert np.isfinite(m2.tractor.t_launch) and m2.tractor.thrust == 30.0
        assert not any("not launched in this run" in e.text for e in m2.events)
    finally:
        if r is not None:
            r.mission.close()
        pool.close()
