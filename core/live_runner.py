"""Real-time driver of a LiveMission, independent of the UI toolkit.

The display follows the physics, never the other way round. `LiveRunner.loop` advances the mission clock at
the time warp the user chose. The one exception, which the user can switch off ("slowmo"): near Bennu the
display slows down so the DART's final approach and impact and the tractor's descent, boulder capture and
lift-off are not missed (LiveMission.slow_motion; a skip stops where a close-up begins). The physics is the same
at any display speed. The physics steps in chunks set by the
physics itself. If it cannot keep up (a long proximity phase at a high warp, or Bennu's orbit still being
integrated), the clock goes only as far as the physics has computed: the snapshot says so ("behind", or
"busy" while it waits) and the UI shows "computing" / "recalculating" instead of guessing.

Commands: pause/resume, warp, launch a DART or the tractor, solar storm, proximity controller, release the boulder
(end of the tow), skip to the next event, run to the end. Launches, storms and the release are recorded with
their mission times.

Re-simulation ("resim", params, dart_plan, t_start, t_back): when a parameter changes during a run, the whole
mission is flown again from its start with the new parameters, replaying the recorded commands at their times
(a later DART or the tractor is re-planned for the state the new run reaches, and dropped if that run no longer
calls for it), as fast as the physics allows, up to t_back. Bennu's integrated orbit is kept for every push that
comes out identical. Meanwhile snapshots carry only the progress ({"resim": ...}).

`process_main` runs it in a separate process (spawned by app/livesim.py), so the physics has its own
interpreter, its own GIL and its own numba worker threads (budget()["sim_threads"] of the core budget,
core/parallel.py) while the UI process only renders. Bennu's full-model integration (core/truth.py) runs on a
pool of full-model processes (budget()["sim_fullmodel"], core/fullmodel.py). The worker starts with the app,
loads the ephemerides, Bennu's models and the compiled kernels once, then waits for "start" commands.
"""

import queue
import time
import traceback
from dataclasses import replace

import numpy as np

from core.constants import DAY
from core.mission import snapshot, verified_curve

SNAP_PERIOD = 0.05  # s between snapshots
RESIM_BUDGET = 0.25  # s of computing between progress snapshots while re-simulating
STALE_TRACTOR_PLAN_DAYS = 45.0  # a tractor plan older than this at the launch is re-planned for a launch now


class LiveRunner:
    def __init__(self, mission, warp, factory=None):
        self.mission = mission
        self.warp = warp
        self.running = True
        self.stop_requested = False
        self.skip_to = None
        self.factory = factory  # (params, t_start) -> a fresh LiveMission (re-simulation)
        self.record = []  # launches and storms as they happened
        self.replay = None
        self.slowmo = True  # slow down near Bennu (DART impact, boulder pickup): display speed only
        self._zone = None
        self._curve_sent = -1
        self._results_seen = 0
        self._achieved = 0.0
        self._behind = 0

    # ------------------------------------------------------------------ commands
    def command(self, name, args):
        m = self.mission
        if self.replay is not None and name not in ("pause", "resume", "warp", "resim", "stop", "end"):
            self.replay["after"].append((name, args))  # applied once the re-simulation has caught up
            return
        if name == "pause":
            self._set_running(False)
        elif name == "resume":
            self._set_running(True)
        elif name == "warp":
            self.warp = args[0]
        elif name == "slowmo":
            self.slowmo = bool(args[0])
        elif name == "storm":
            m.solar_storm(args[0])
            self._record("storm", args[0])
        elif name == "thrust":
            m.set_tractor_thrust(args[0])
        elif name == "controller":
            m.set_prox_controller(args[0])
            self._record("controller", args[0])
        elif name == "launch_dart":
            n = len(m.impactors) + 1
            state = self._state(m)
            if m.launch_dart(args[0]):
                self._record("launch_dart", args[0], number=n, **state)
                self.running = True
        elif name == "launch_tractor":
            if len(args) > 1 and args[1] is not None:
                m.p = args[1]
            plan = args[0]
            # The UI keeps its plan fresh (re-planned for "launch now" every 30 mission days while the button is
            # available), so the plan the user saw is the one that flies; this re-plan is only a safety net for a
            # plan that has gone stale (never silently for a plan a few days old)
            if plan is None or plan.t_launch < m.t - STALE_TRACTOR_PLAN_DAYS * DAY:
                plan, _ = m.planner.plan_tractor_launch(m.p, m.t, m.b_now_abs())
            state = self._state(m)
            if m.launch_tractor(plan):
                self._record("launch_tractor", plan, **state)
                self.running = True
        elif name == "release":  # end of the tow: the boulder is let go, the tractor leaves Bennu
            if m.end_tow():
                self._record("release", None)
                self.running = True
        elif name == "skip":
            self.skip_to = m.next_event_time()
            self.running = True
        elif name == "finish":
            m.finish()
            self.skip_to = m.t_end
            self.running = True
        elif name == "resim":
            self.resim(*args)
        elif name in ("stop", "end"):
            self.stop_requested = True

    def _set_running(self, on):
        if self.replay is not None:
            self.replay["running"] = on
        else:
            self.running = on

    @staticmethod
    def _state(m):
        return {"b_now": m.b_now_abs(), "beta": m.beta_ground, "params": m.p}

    def _record(self, name, arg, **extra):
        self.record.append(dict(t=self.mission.t, name=name, arg=arg, **extra))

    # ------------------------------------------------------------------ re-simulation
    def resim(self, params, plan, t_start, t_back=None):
        """Fly the mission again from t_start with new parameters (see the module docstring)."""
        old = self.mission
        if self.replay is not None:  # a newer change while re-simulating: start over, same goal
            t_back = self.replay["t_back"] if t_back is None else t_back
            commands, running = self.replay["commands"], self.replay["running"]
        else:
            t_back = old.t if t_back is None else t_back
            commands, running = self.record, self.running
        t_back = max(float(t_back), float(t_start))
        old.close(keep_truth=True)
        m = self.factory(params, t_start)
        m.adopt_truth(old.truth)
        m.display_from = t_back
        m.prox_controller = old.prox_controller
        later = [c for c in commands if not (c["name"] == "launch_dart" and c.get("number") == 1)]
        queue = [dict(t=float(t_start), name="launch_dart", arg=plan, number=1, first=True)] + later
        queue.sort(key=lambda c: c["t"])
        self.mission = m
        self.record = []
        self.skip_to = None
        self._results_seen = 0
        self._curve_sent = -1
        self.replay = {"queue": queue, "commands": commands, "t_back": t_back, "t0": float(t_start),
                       "running": running, "after": []}
        m.log(m.t, "mission", f"Parameters changed: the mission is flown again from {_date(t_start)} with the "
                              "new parameters", "warn")

    def _replay_step(self, budget=RESIM_BUDGET):
        r, m = self.replay, self.mission
        wall = time.perf_counter()
        while time.perf_counter() - wall < budget:
            if r["queue"] and r["queue"][0]["t"] <= m.t + 1e-6:
                if not m.settle():
                    return  # a verdict due at this instant comes first, as in the first run (e.g. the tractor
                            # launch ordered at the verdict); its orbit determination is still being computed
                self._replay_command(r["queue"].pop(0))
                continue
            t_goal = min(r["t_back"], r["queue"][0]["t"]) if r["queue"] else r["t_back"]
            if m.t >= t_goal - 1e-6 or m.complete():
                if not r["queue"] or m.complete():
                    if not m.complete() and not m.settle():
                        return  # the run had paused at a verdict: reach that verdict before handing back
                    self._end_replay()
                    return
                continue
            m.advance(t_goal, budget=max(0.01, budget - (time.perf_counter() - wall)))
            if m.busy():
                return  # Bennu's orbit is being integrated: come back on the next frame

    def _replay_command(self, c):
        m = self.mission
        name = c["name"]
        if name == "launch_dart":
            if c.get("first"):
                if not m.launch_dart(c["arg"]):
                    m.log(m.t, "mission", "With the new parameters the DART plan is not feasible", "bad")
                else:
                    self._record("launch_dart", c["arg"], number=1, **self._state(m))
                return
            if m.dart_ok() or (m.impactor is not None and m.impactor.phase not in ("impacted", "missed")):
                m.log(m.t, "mission", f"DART #{c['number']} is not launched in this run: it is no longer needed")
                return
            ip = c["arg"] if self._same_state(c, m, dart_only=True) else self._plan_next_dart(m, c["arg"])
            n, state = len(m.impactors) + 1, self._state(m)
            if ip is not None and m.launch_dart(ip):
                self._record("launch_dart", ip, number=n, **state)
        elif name == "launch_tractor":
            if not m.dart_ok():
                m.log(m.t, "mission", "The tractor is not launched in this run: no DART has succeeded yet", "warn")
                return
            plan = c["arg"] if self._same_state(c, m) else m.planner.plan_tractor_launch(m.p, m.t, m.b_now_abs())[0]
            state = self._state(m)
            if m.launch_tractor(plan):
                self._record("launch_tractor", plan, **state)
        elif name == "storm":
            m.solar_storm(c["arg"])
            self._record("storm", c["arg"])
        elif name == "controller":
            m.set_prox_controller(c["arg"])
            self._record("controller", c["arg"])
        elif name == "release":
            if m.end_tow():
                self._record("release", None)

    @staticmethod
    def _same_state(c, m, dart_only=False):
        """The new run reached a recorded launch in the same state with the same parameters: same plan. For a DART
        only the DART-side parameters count (a tractor setting does not change a DART plan)."""
        from core.planner import dart_params, physical_params

        same = (dart_params(c.get("params")) == dart_params(m.p) if dart_only
                else physical_params(c.get("params")) == physical_params(m.p))
        return (same and c.get("params") is not None and c.get("beta") == m.beta_ground and c.get("b_now") is not None
                and np.allclose(c["b_now"], m.b_now_abs(), rtol=0.0, atol=1.0))

    @staticmethod
    def _plan_next_dart(m, old_plan):
        """The next DART re-planned for this run's state: the same window if it is still open."""
        from core.timeutil import t_to_utc_str

        q = m.p
        if m.beta_ground is not None:
            q = replace(q, beta=float(m.beta_ground[0]), beta_sigma=float(max(m.beta_ground[1], 0.1)))
        b_now = m.b_now_abs()
        opps = m.planner.opportunities(q, b_now=b_now, t_after=m.t + 30 * DAY)
        sel = next((o for o in opps if o.feasible and old_plan is not None and o.index == old_plan.opportunity), None)
        sel = sel or m.planner.recommended(opps)
        if sel is None:
            m.log(m.t, "mission", "No window is left for another DART in this run", "bad")
            return None
        ip, _ = m.planner.plan_dart(replace(q, earliest_launch=t_to_utc_str(m.t)[:10]), sel, b_now=b_now,
                                    number=len(m.impactors) + 1)
        return ip if ip.feasible else None

    def _end_replay(self):
        r = self.replay
        self.replay = None
        self.running = r["running"]
        self.mission.log(self.mission.t, "mission", "Re-simulation complete: the run continues", "good")
        for name, args in r["after"]:
            self.command(name, args)

    def _progress(self):
        r, m = self.replay, self.mission
        span = max(r["t_back"] - r["t0"], 1.0)
        return {"resim": {"t": m.t, "t0": r["t0"], "t1": r["t_back"], "fraction": float(np.clip((m.t - r["t0"]) / span, 0, 1)),
                          "busy": m.busy(), "background": (m.background() or {}).get("what")},
                "t": m.t, "running": False}

    # ------------------------------------------------------------------ real time
    def loop(self, get_commands, emit):
        """get_commands() -> list of (name, args); emit(snapshot dict)."""
        last = time.perf_counter()
        while not self.stop_requested:
            for name, args in get_commands():
                self.command(name, args)
            if self.stop_requested:
                break
            now = time.perf_counter()
            dt_real = min(now - last, 0.25)
            last = now
            if self.replay is not None:
                self._replay_step()
                if self.replay is not None:
                    emit(self._progress())
                    time.sleep(0.005)
                    continue
                last = time.perf_counter()
            m = self.mission
            zone = m.slow_motion() if self.slowmo else None
            if zone is not None and self._zone is None:
                self.skip_to = None  # a close-up begins: a skip stops here so it is not missed
            self._zone = zone
            warp = min(self.warp, zone[0]) if zone is not None else self.warp
            requested = np.inf if self.skip_to is not None else warp
            key = zone[0] if zone is not None else None
            stop = (lambda: (m.slow_motion() or (None,))[0] != key) if self.slowmo else None
            if self.running and not m.complete():
                t0 = m.t
                target = t0 + (warp * dt_real if self.skip_to is None else np.inf)
                if self.skip_to is not None:
                    target = self.skip_to
                if not m.busy():
                    m.advance(target, budget=0.8 * SNAP_PERIOD if self.skip_to is None else 0.25, stop=stop)
                if self.skip_to is not None and m.t >= self.skip_to - 1.0:
                    self.skip_to = None
                self._achieved = (m.t - t0) / max(dt_real, 1e-3)
                short = (m.t < target - max(1e-3, 0.02 * warp * dt_real) and not m.complete()
                         and (m.slow_motion() if self.slowmo else None) == zone)  # (not a slow-motion stop)
                self._behind = self._behind + 1 if short else 0
            else:
                self._achieved, self._behind = 0.0, 0
            if len(m.dart_results) > self._results_seen:  # a DART verdict is a decision point: pause the clock
                self._results_seen = len(m.dart_results)
                self.running = False
                self.skip_to = None
            s = snapshot(m)
            s["running"], s["warp_achieved"] = self.running and not m.complete(), self._achieved
            s["warp_requested"] = requested
            s["warp_user"] = self.warp
            s["slowmo"] = {"warp": zone[0], "what": zone[1]} if zone is not None else None
            s["behind"] = self._behind >= 4 and self.skip_to is None and not s.get("busy")
            version = m.curve_version
            if version != self._curve_sent and m.final_check is not None and "track_t" in m.final_check:
                s["verified_curve"] = verified_curve(m)  # the full-model Earth-distance curve, once per version
            if emit(s) is not False and "verified_curve" in s:
                self._curve_sent = version
            time.sleep(max(0.0, SNAP_PERIOD - (time.perf_counter() - now)))


def _date(t):
    from core.timeutil import t_to_utc_str

    return t_to_utc_str(t)[:10]


def process_main(cmd_q, snap_q, pending):
    """Entry point of the simulation process. Messages in: ("start", run_id, params, t_start, seed, controller,
    warp, first_commands) or (command, args) during a run, ("quit",) to exit. A "start" during a run ends that
    run first. Messages out: ("ready",), ("snap", run_id, s), ("error", run_id, text), ("ended", run_id).
    `pending` (shared int) counts snapshots not yet taken by the UI."""
    fullmodel = None
    try:
        from core import parallel
        from core.ephemeris import Ephemeris
        from core.fullmodel import FullModelPool
        from core.planner import Planner
        from core.proximity.ops import ProxEnvironment
        from core.scenario import KeyholeScenario

        share = parallel.budget()
        parallel.configure(share["sim_threads"])
        parallel.set_qos()
        parallel.exit_with_parent()
        fullmodel = FullModelPool(share["sim_fullmodel"])
        sc = KeyholeScenario.load()
        eph = Ephemeris().with_bennu(sc.nominal)
        planner = Planner(eph, sc.nominal, sc)
        env = ProxEnvironment.shared()  # gravity, shape ray caster, capture site; compiles the kernels
        env.grav.accel_body(env.site_origin * 2.0)
        snap_q.put(("ready",))
    except Exception:
        snap_q.put(("error", traceback.format_exc()))
        if fullmodel is not None:
            fullmodel.close()
        return
    try:
        _serve(cmd_q, snap_q, pending, eph, planner, env, fullmodel)
    finally:
        fullmodel.close()  # never wait at exit for a propagation nobody will read


def _serve(cmd_q, snap_q, pending, eph, planner, env, fullmodel):
    from core import parallel
    from core.fullmodel import FullModelPool
    from core.mission import LiveMission

    held = []  # a "start" that arrived while a run was still going: it ends that run and begins the next
    pool = {"fm": fullmodel}

    def drain():
        out = []
        while not held:
            try:
                msg = cmd_q.get_nowait()
            except queue.Empty:
                return out
            if msg[0] == "quit":
                raise SystemExit
            if msg[0] == "start":
                held.append(msg)
                out.append(("end", ()))
                return out
            out.append((msg[0], msg[1:]))
        return out

    try:
        while True:
            msg = held.pop() if held else cmd_q.get()
            if msg[0] == "quit":
                return
            if msg[0] != "start":
                continue
            _, run_id, params, t_start, seed, controller, warp, first = msg

            def emit(s, run_id=run_id):
                with pending.get_lock():  # the UI only needs the latest state; never let a backlog build up
                    if pending.value >= 2:
                        return False
                    pending.value += 1
                snap_q.put(("snap", run_id, s))
                return True

            def factory(p, t0, seed=seed, controller=controller):
                m = LiveMission(eph, planner.nom, planner, p, t0, seed=seed)
                m.prox_controller = controller
                m._prox_env = env
                m.fullmodel = pool["fm"]
                return m

            runner, quitting = None, False
            try:
                runner = LiveRunner(factory(params, t_start), warp, factory=factory)
                for name, *args in first:
                    runner.command(name, tuple(args))
                runner.loop(drain, emit)
            except SystemExit:
                quitting = True
                return
            except Exception:
                snap_q.put(("error", run_id, traceback.format_exc()))
            finally:
                busy = runner is not None and runner.mission.close()
                if busy and not quitting:  # its propagation would delay the next run's: fresh workers
                    pool["fm"].close()
                    pool["fm"] = FullModelPool(parallel.budget()["sim_fullmodel"])
            snap_q.put(("ended", run_id))
    finally:
        pool["fm"].close()
