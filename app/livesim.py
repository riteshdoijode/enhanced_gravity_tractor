"""Live-run plumbing between the UI and the simulation.

SimProcess (default): the simulation runs in a separate *process* (core/live_runner.py) that is started
with the app and kept warm (ephemerides, Bennu models and compiled kernels loaded once). The UI process
only renders, so neither waits for the other's GIL: the physics gets a full core plus its numba worker
threads and full-model processes (its share of the core budget, core/parallel.py), and the window stays
responsive. A small QThread relays snapshots from the process to Qt signals.
LiveSimThread: the same loop on a thread inside the UI process (fallback; BENNU_SIM_PROCESS=0).

Both expose start(params, t_start, seed, controller, warp, first_commands), send(name, *args), stop(), and
the signals snap(dict), failed(str), ready().
"""

import multiprocessing as mp
import os
import queue

from PySide6 import QtCore

from core.live_runner import LiveRunner, process_main


class SimProcess(QtCore.QObject):
    snap = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        ctx = mp.get_context("spawn")  # never fork a Qt/VTK process
        self.cmd_q, self.snap_q = ctx.Queue(), ctx.Queue()
        self.pending = ctx.Value("i", 0)
        # not a daemon: it starts its own full-model worker processes. It still ends with the app: shutdown()
        # stops it, and it exits by itself if the app process dies (core/parallel.py: exit_with_parent)
        self.proc = ctx.Process(target=process_main, args=(self.cmd_q, self.snap_q, self.pending), daemon=False,
                                name="bennu-sim")
        self.proc.start()
        self.is_ready = False
        self.active = False
        self.run_id = 0  # snapshots of an earlier run (still in the queue) are ignored
        self._relay = _Relay(self.snap_q, self.pending, self)
        self._relay.message.connect(self._on_message)
        self._relay.start()

    def _on_message(self, msg):
        kind = msg[0]
        if kind == "ready":
            self.is_ready = True
            self.ready.emit()
        elif kind == "snap" and self.active and msg[1] == self.run_id:
            self.snap.emit(msg[2])
        elif kind == "error" and (len(msg) == 2 or msg[1] == self.run_id):
            self.failed.emit(msg[-1])
        elif kind == "ended" and msg[1] == self.run_id:
            self.active = False

    def start(self, params, t_start, seed, controller, warp, first=()):
        self.run_id += 1
        self.active = True
        self.cmd_q.put(("start", self.run_id, params, t_start, seed, controller, warp, list(first)))

    def send(self, name, *args):
        self.cmd_q.put((name, *args))

    def stop(self):
        if self.active:
            self.active = False
            self.cmd_q.put(("end",))

    def shutdown(self):
        self.cmd_q.put(("quit",))
        self._relay.stop_requested = True
        self._relay.wait(2000)
        self.proc.join(3)
        if self.proc.is_alive():
            self.proc.terminate()


class _Relay(QtCore.QThread):
    """Blocks on the process queue and forwards each message to the UI thread."""
    message = QtCore.Signal(object)

    def __init__(self, q, pending, parent=None):
        super().__init__(parent)
        self.q, self.pending = q, pending
        self.stop_requested = False

    def run(self):
        while not self.stop_requested:
            try:
                msg = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            if msg[0] == "snap":
                with self.pending.get_lock():
                    self.pending.value -= 1
            self.message.emit(msg)


class LiveSimThread(QtCore.QThread):
    """In-process fallback with the same interface."""
    snap = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    def __init__(self, eph, planner, parent=None):
        super().__init__(parent)
        self.eph, self.planner = eph, planner
        self.is_ready = True
        self.active = False
        self.runner = None
        self._queue = queue.Queue()

    def start(self, params, t_start, seed, controller, warp, first=()):  # noqa: D102 (QThread.start is called inside)
        from core.mission import LiveMission

        def factory(p, t0):
            m = LiveMission(self.eph, self.planner.nom, self.planner, p, t0, seed=seed)
            m.prox_controller = controller
            return m

        self.runner = LiveRunner(factory(params, t_start), warp, factory=factory)
        for name, *args in first:
            self.runner.command(name, tuple(args))
        self.active = True
        super().start()

    def run(self):
        def drain():
            out = []
            while not self._queue.empty():
                name, *args = self._queue.get()
                out.append((name, tuple(args)))
            return out
        try:
            self.runner.loop(drain, self.snap.emit)
        except Exception as exc:
            import traceback

            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
        self.runner.mission.close()
        self.active = False

    def send(self, name, *args):
        self._queue.put((name, *args))

    def stop(self):
        if self.active:
            self._queue.put(("stop",))
            self.wait(5000)

    def shutdown(self):
        self.stop()


def make_sim(eph, planner, parent=None):
    if os.environ.get("BENNU_SIM_PROCESS", "1") == "0":
        return LiveSimThread(eph, planner, parent)
    return SimProcess(parent)
