"""Planner work in worker processes, so the GUI thread never waits for it.

Python runs one thread at a time per process (the GIL), so planner jobs on background threads still slowed
the GUI: every porkchop, point-of-contact ray cast or low-thrust flight plan held the interpreter. Here each
kind of job has its own worker process, which keeps its caches warm:
    plan      the mission plan (opportunities, DART, tractor estimate) and what the DART as set does
              in every window (the distance plot's markers)                   ~30 ms warm, ~3 s cold
    tractor   optimized flight plan, arrival-vs-thrust curve, the Sun plot    ~1-5 s
    live      next-DART recommendation, tractor plan after a success, a DART planned from the
              slider time                                                     ~1 s
(the full-model mass refinement has its own process, core/refine.py, and the live simulation another,
core/live_runner.py). Each worker loads the scenario, ephemerides and planner once.

ComputePool.run(job, *args) blocks the calling thread until the worker answers. It is called from the
QThreadPool jobs of app/workers.py, whose threads wait without holding the GIL. Each worker runs its
Lambert / Kepler batches on its share of the core budget (core/parallel.py: budget()["planner_threads"]).
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from core.constants import DAY
from core.parallel import kill_pool

_W = {}


def _init():
    from core import parallel
    from core.ephemeris import Ephemeris
    from core.planner import Planner
    from core.scenario import KeyholeScenario

    parallel.configure(parallel.budget()["planner_threads"])
    parallel.set_qos()
    parallel.exit_with_parent()
    sc = KeyholeScenario.load()
    eph = Ephemeris().with_bennu(sc.nominal)
    _W["eph"], _W["pl"] = eph, Planner(eph, sc.nominal, sc)


def _strip(opps):
    """Opportunities travel back without their porkchops (megabytes; the GUI only needs the summary)."""
    for o in opps:
        o._pc = None
    return opps


def _spin(spin):
    from core.dynamics.asteroid import reset_spin, set_spin_change

    reset_spin()
    if spin is not None:
        set_spin_change(*spin)


def ping():
    return True


def job_plan(p, tractor_arrival):
    """The plan, and the DART as set (the plan's mass: the recommended one, or the slider's) in every window."""
    _spin(None)
    pl = _W["pl"]
    res = pl.plan(p, tractor_arrival=tractor_arrival)
    ip = res.impactor
    res.capability = pl.window_capability(p, res.opportunities, ip.mass if ip.feasible else p.impactor_mass_kg)
    _strip(res.opportunities)
    return res


def job_eta(p, t_launch, thrusts):
    return t_launch, _W["pl"].tractor_eta_curve(p, t_launch, thrusts)


def job_flight_plan(p, t_launch):
    return _W["pl"].tractor_flight_plan(p, t_launch)


def job_sun():
    from core.dynamics.solar_system import plot_data

    return plot_data(_W["eph"])


def job_recommend(p, b_now, t_now, number, spin):
    """The next DART after a shortfall: opportunities from now, the selected (or recommended) one planned, and what
    that DART does in each remaining window (p carries the measured beta; b_now is the deflection so far)."""
    from core.timeutil import t_to_utc_str

    _spin(spin)
    pl = _W["pl"]
    opps = pl.opportunities(p, b_now=b_now, t_after=t_now + 30 * DAY)
    sel = next((o for o in opps if o.index == p.opportunity), None) or pl.recommended(opps)
    ip = None
    if sel is not None:
        ip, _ = pl.plan_dart(replace(p, earliest_launch=t_to_utc_str(t_now)[:10]), sel, b_now=np.asarray(b_now),
                             number=number)
    cap = pl.window_capability(p, opps, ip.mass if ip is not None and ip.feasible else p.impactor_mass_kg,
                               b_now=np.asarray(b_now, float))
    return _strip(opps), ip, cap


def job_launch_plan(p, t_now):
    """Planning mode with the time slider past the selected window's orbit raising: that window planned for a
    launch from t_now if it is still open, else None, and the windows still ahead."""
    from core.timeutil import t_to_utc_str

    _spin(None)
    pl = _W["pl"]
    opps = pl.opportunities(p, t_after=t_now)
    sel = next((o for o in opps if o.index == p.opportunity and o.feasible), None)
    if sel is None and p.opportunity < 0:
        sel = pl.recommended(opps)
    ip = None
    if sel is not None:
        ip, _ = pl.plan_dart(replace(p, earliest_launch=t_to_utc_str(t_now)[:10]), sel)
    return _strip(opps), ip


def job_tractor(p, b_now, t_launch, with_window=True):
    """The tractor launched at t_launch (now, or the verdict date when planned ahead of it): quick plan, optimized
    flight plan (the live run's warm start), tow window and reach; with the launch window (the dates from t_launch
    on that still reach Bennu in time to close the gap: the blue band of the distance plot)."""
    pl = _W["pl"]
    tp, msg = pl.plan_tractor_launch(p, t_launch, b_now)
    window = pl.tractor_launch_window(p, t_launch, b_now) if with_window else None
    return tp, msg, window


def job_tractor_window(p, t_from, b_now):
    """Planning mode: the tractor launch window from the planned verdict date, for the DART's planned shift."""
    return _W["pl"].tractor_launch_window(p, t_from, np.asarray(b_now, float))


class ComputePool:
    """One single-worker process pool per kind of job (warm caches, no queueing behind long jobs)."""

    KINDS = ("plan", "tractor", "live")

    def __init__(self):
        ctx = mp.get_context("spawn")  # never fork a Qt/VTK process
        self.pools = {k: ProcessPoolExecutor(1, mp_context=ctx, initializer=_init) for k in self.KINDS}
        for ex in self.pools.values():  # start the workers now: they load the models while the window opens
            ex.submit(ping)

    def run(self, kind, fn, *args):
        return self.pools[kind].submit(fn, *args).result()

    def shutdown(self):
        for ex in self.pools.values():
            kill_pool(ex)
