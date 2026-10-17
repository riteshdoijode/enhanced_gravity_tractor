"""Multi-core execution: the core budget, per-process thread pools for compiled kernels, scheduling priority.

Core budget. BENNU_CORES (default: every core of the machine; 12 on an M3 Pro = 6 performance + 6
efficiency) is shared out between the app's processes by `budget()`: the GUI, the live simulation (its
thread pool and a small pool of full-model processes), three planner workers and the full-model mass
refinement. Each process sizes its own thread pool from its share (`configure`). `uv run python -m app.main
--cores N` sets the budget for one run.

macOS cannot reserve cores for one program (there is no CPU-affinity call). What it offers instead is a
quality-of-service class per thread: `set_qos()` marks a thread "user-initiated", so the scheduler prefers the
performance cores for it and does not throttle it when the window is in the background.

Thread pools. The heavy numerical kernels (Kepler propagation, polyhedron gravity, Lambert batches) are
numba functions compiled with nogil=True, so several Python threads run them truly in parallel. `chunked`
cuts a batch into slices and runs them on the pool; with more threads than performance cores it cuts more
slices than threads, so the fast cores take more of them and the efficiency cores do not hold everyone up.
Calls made from inside a pool worker run serially (no nested submission, no deadlock), and small batches
skip the pool because dispatch costs ~20 us.
"""

import ctypes
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_local = threading.local()
_pool = None
_lock = threading.Lock()

QOS = {"user_interactive": 0x21, "user_initiated": 0x19, "default": 0x15, "utility": 0x11}


def _sysctl_int(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip())
    except Exception:
        return 0


def _performance_cores():
    n = _sysctl_int("hw.perflevel0.physicalcpu")  # Apple Silicon: performance cores
    return n if n > 0 else max(1, (os.cpu_count() or 2) // 2)


PERF_CORES = _performance_cores()
CORES = int(os.environ.get("BENNU_CORES", 0)) or (os.cpu_count() or PERF_CORES)
N_THREADS = int(os.environ.get("BENNU_THREADS", 0)) or min(PERF_CORES, CORES)


def budget(cores=None):
    """How many threads / processes each part of the app may use, out of `cores`."""
    c = max(1, int(cores or CORES))
    return {
        "cores": c,
        "gui_threads": max(1, c // 6),  # the GUI process's own kernels (rare: planning runs in workers)
        "sim_threads": max(1, c // 3),  # live simulation: polyhedron gravity, ejecta, Kepler batches
        "sim_fullmodel": 2 if c >= 6 else 1,  # orbit determination / verification: with and without pushes
        "planner_threads": max(1, c // 6),  # each of the 3 planner workers (Lambert porkchops)
        "refine_procs": max(1, c // 2),  # full-model mass refinement: DART masses evaluated side by side
    }


def configure(threads):
    """Resize this process's thread pool (the next batch uses the new size). BENNU_THREADS, if set, wins."""
    global N_THREADS, _pool
    with _lock:
        threads = max(1, int(os.environ.get("BENNU_THREADS", 0)) or int(threads))
        if threads != N_THREADS and _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
        N_THREADS = threads


def set_qos(level="user_initiated"):
    """Ask macOS to schedule the calling thread at this quality-of-service class (no-op elsewhere)."""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        lib.pthread_set_qos_class_self_np(ctypes.c_uint(QOS[level]), ctypes.c_int(0))
        return True
    except Exception:
        return False


def exit_with_parent(period=1.0):
    """End this worker process if the process that started it dies (a killed or crashed app must not leave
    workers behind finishing a job for nobody). On macOS an orphan is re-parented to launchd."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(period)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def kill_pool(ex, grace=0.0):
    """Stop a process pool now, including a job in progress (a worker would otherwise finish it after the window
    has closed, using a core for nothing). With `grace` > 0 the jobs already queued may finish and the workers
    exit normally (cleaning up after themselves) for up to that many seconds before they are terminated."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=grace <= 0)
    deadline = time.monotonic() + grace
    for proc in procs:
        if grace > 0:
            proc.join(max(0.0, deadline - time.monotonic()))
        if proc.is_alive():
            proc.terminate()


def _worker_init():
    _local.worker = True
    set_qos()


def pool():
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=N_THREADS, thread_name_prefix="bennu", initializer=_worker_init)
        return _pool


def in_worker():
    return getattr(_local, "worker", False)


def chunked(fn, n, min_chunk=64):
    """Run fn(start, stop) over [0, n) in slices on the pool; returns the list of results in order."""
    if n < 2 * min_chunk or N_THREADS == 1 or in_worker():
        return [fn(0, n)]
    pieces = N_THREADS if N_THREADS <= PERF_CORES else 3 * N_THREADS  # mixed cores: balance by queueing
    k = max(1, min(pieces, n // min_chunk))
    edges = [round(i * n / k) for i in range(k + 1)]
    futures = [pool().submit(fn, a, b) for a, b in zip(edges[:-1], edges[1:])]
    return [f.result() for f in futures]


def map_threads(fn, items):
    """fn over items on the pool (for independent tasks whose work releases the GIL)."""
    items = list(items)
    if len(items) < 2 or N_THREADS == 1 or in_worker():
        return [fn(x) for x in items]
    return list(pool().map(fn, items))


def describe(cores=None):
    b = budget(cores)
    return (f"{b['cores']} cores ({PERF_CORES} performance + {(os.cpu_count() or 0) - PERF_CORES} efficiency on "
            f"this Mac): simulation 1 + {b['sim_threads']} threads + {b['sim_fullmodel']} full-model processes, "
            f"3 planner workers x {b['planner_threads']} threads, mass refinement {b['refine_procs']} processes")
