"""Full-force-model propagations on several cores at once.

Each full-model run (scipy's DOP853 over a century of Bennu's orbit, ~15-25 s) is pure Python driving
NumPy, so it holds its process's interpreter lock: threads cannot run two of them side by side. They run in
worker processes instead. The verification of a deflection needs three runs (core/tow.py): with the pushes and
without them to 60 days before the 2135 flyby (independent: run together), then the deflected orbit through
the flyby. The full-model DART mass refinement (core/refine.py) evaluates several candidate masses together.

    pool = FullModelPool(4)
    future = pool.submit(fn, *args)       # runs fn(eph, *args) in a worker, with the worker's ephemerides

`Inline(eph)` has the same interface and runs each job immediately in the calling thread (no pool, used by
tests, scripts and the in-process fallback). Workers load the planetary ephemerides once (~1 s), run at
user-initiated priority with single-threaded kernels, and exit on their own if the process that started
them dies.
"""

import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor

from core import parallel

_W = {}


def _init():
    from core.ephemeris import Ephemeris

    parallel.configure(1)
    parallel.set_qos()
    parallel.exit_with_parent()
    _W["eph"] = Ephemeris()  # Bennu itself is the integrated body: the perturbers are all a run needs


def _call(fn, args):
    return fn(_W["eph"], *args)


def _ping():
    return True


class FullModelPool:
    def __init__(self, workers):
        self.workers = max(1, int(workers))
        self.ex = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"), initializer=_init)
        for _ in range(self.workers):  # start them all now, so they have loaded the ephemerides when needed
            self.ex.submit(_ping)

    def submit(self, fn, *args):
        return self.ex.submit(_call, fn, args)

    def close(self):
        parallel.kill_pool(self.ex)


class Inline:
    """FullModelPool stand-in: each job runs immediately in the calling thread."""

    workers = 1
    inline = True

    def __init__(self, eph):
        self.eph = eph

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(self.eph, *args))
        except Exception as exc:
            f.set_exception(exc)
        return f

    def close(self):
        pass
