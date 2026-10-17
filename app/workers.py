"""Background computation so the window never freezes while sliders move."""

from PySide6 import QtCore


class _Signals(QtCore.QObject):
    done = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)


class Job(QtCore.QRunnable):
    """Runs fn() on the thread pool and emits done(job_id, result)."""

    def __init__(self, job_id, fn):
        super().__init__()
        self.job_id, self.fn = job_id, fn
        self.signals = _Signals()

    def run(self):
        try:
            result = self.fn()
        except Exception as exc:  # surfaced in the UI instead of killing the thread
            import traceback

            self.signals.failed.emit(self.job_id, f"{exc}\n{traceback.format_exc()}")
        else:
            self.signals.done.emit(self.job_id, result)


class Debounced(QtCore.QObject):
    """Coalesces bursts of slider events: runs `fn` once, `delay_ms` after the last request.
    Only the newest job's result is delivered (older ones are dropped)."""

    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    busy = QtCore.Signal(bool)

    def __init__(self, fn_factory, delay_ms=180, parent=None):
        super().__init__(parent)
        self.fn_factory = fn_factory
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(delay_ms)
        self.timer.timeout.connect(self._launch)
        self.latest = 0
        self.pool = QtCore.QThreadPool.globalInstance()

    def request(self):
        self.timer.start()

    def _launch(self):
        self.latest += 1
        job = Job(self.latest, self.fn_factory())
        job.signals.done.connect(self._done)
        job.signals.failed.connect(self._failed)
        self.busy.emit(True)
        self.pool.start(job)

    def _done(self, job_id, result):
        if job_id == self.latest:
            self.busy.emit(False)
            self.result.emit(result)

    def _failed(self, job_id, msg):
        if job_id == self.latest:
            self.busy.emit(False)
            self.error.emit(msg)
