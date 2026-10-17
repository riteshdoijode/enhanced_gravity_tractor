"""Session log: what was clicked and how the app responded, one JSON object per line in outs/logs/.

So that a problem seen in an interactive run can be traced afterwards, the window writes:
    click      every mouse press: the widget under the mouse (type, text, enabled or not, where it sits)
    params     every edit of a mission parameter (what changed), and whether it re-flies the live run
    buttons    every change of the launch buttons' state, with the function that changed it
    action     what a launch button press did (or why it did nothing)
    command    every command sent to the simulation process
    job        planner jobs: errors (they are otherwise only flashed in a panel)
    result     a DART verdict, a tractor plan, a next-DART recommendation, a plan
    event      the mission's own event log (the Live flight tab's list)
    state      a snapshot summary when the mission state changes, and every 5 s of a live run
    error      every Python exception (with its traceback) and Qt warning
Each line carries the wall-clock time and the simulation date. The file is flushed line by line, so it survives
a crash; one file per session (session_YYYYmmdd_HHMMSS.jsonl).
"""

import dataclasses
import datetime
import json
import os
import sys
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from core.timeutil import t_to_utc_str


def _brief(o):
    """A short, JSON-friendly form of the app's objects (plans, numpy values)."""
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 6 else f"array{o.shape}"
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        keep = ("feasible", "number", "opportunity", "t_launch", "t_arrive", "mass", "mass_source", "impact_speed",
                "v_dep", "note", "use_recommended_mass", "impactor_mass_kg", "dart_max_speed_kms", "beta",
                "beta_sigma", "dart_sizing", "tractor_thrust_n", "boulder_mass_kg", "tow_years", "hover_factor",
                "goal_ld", "dart_target_km")
        out = {"type": type(o).__name__}
        for f in dataclasses.fields(o):
            if f.name in keep:
                v = getattr(o, f.name)
                out[f.name] = _date(v) if f.name.startswith("t_") else v
        return out
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return repr(o)[:200]


def _date(t):
    try:
        return t_to_utc_str(float(t))[:16] if t is not None and np.isfinite(t) else None
    except (TypeError, ValueError):
        return None


class SessionLog(QtCore.QObject):
    def __init__(self, directory="outs/logs", parent=None):
        super().__init__(parent)
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.abspath(os.path.join(directory, f"session_{stamp}.jsonl"))
        self._f = open(self.path, "a", buffering=1, encoding="utf-8")
        self.t_sim = None  # the simulation date shown (set by the window)
        self._last_press = None
        self._hooks_installed = False

    def write(self, kind, **data):
        if self._f.closed:
            return
        rec = {"wall": datetime.datetime.now().isoformat(timespec="milliseconds"), "sim": _date(self.t_sim),
               "kind": kind, **data}
        try:
            self._f.write(json.dumps(rec, default=_brief, ensure_ascii=False) + "\n")
        except Exception as exc:  # the log must never break the app
            print(f"session log: {exc}", file=sys.stderr)

    def close(self):
        self.write("session", what="end")
        self._f.close()

    # ------------------------------------------------------------------ clicks
    def install(self, app):
        """Watch every mouse press in the application, and log uncaught exceptions and Qt warnings."""
        app.installEventFilter(self)
        if not self._hooks_installed:
            self._hooks_installed = True
            previous = sys.excepthook

            def hook(kind, value, tb):
                self.write("error", where="python", text="".join(traceback.format_exception(kind, value, tb)))
                previous(kind, value, tb)

            sys.excepthook = hook

            def qt_message(mode, context, message):
                self.write("error", where="qt", level=str(mode), text=message)
                print(message, file=sys.stderr)  # still shown in the terminal, as without the log

            QtCore.qInstallMessageHandler(qt_message)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.MouseButtonPress:
            stamp = ev.timestamp()
            if stamp != self._last_press:  # one line per press (Qt offers it to each widget up the chain)
                self._last_press = stamp
                pos = ev.globalPosition().toPoint()
                w = QtWidgets.QApplication.widgetAt(pos)
                self.write("click", button=ev.button().name, **describe(w))
        return False


def describe(w):
    """What a widget is, for the log: type, text, enabled, and its place (group box, tab, view)."""
    if w is None:
        return {"widget": None}
    out = {"widget": type(w).__name__, "enabled": w.isEnabled()}
    if w.objectName():
        out["name"] = w.objectName()
    if isinstance(w, QtWidgets.QAbstractButton):
        out["text"] = w.text()
        if w.isCheckable():
            out["checked_before"] = w.isChecked()
    elif isinstance(w, QtWidgets.QComboBox):
        out["text"] = w.currentText()
    elif isinstance(w, QtWidgets.QLabel):
        out["text"] = w.text()[:80]
    elif isinstance(w, QtWidgets.QTabBar):
        out["text"] = "tab bar"
    place = []
    p = w.parentWidget()
    while p is not None and len(place) < 4:
        if isinstance(p, QtWidgets.QGroupBox) and p.title():
            place.append(f"section '{p.title()}'")
        elif p.objectName():
            place.append(p.objectName())
        if isinstance(p, QtWidgets.QTabWidget):
            place.append(f"tab '{p.tabText(p.currentIndex())}'")
        p = p.parentWidget()
    if place:
        out["in"] = " < ".join(place)
    # a disabled widget passes its press to its parent: say so, it is the usual "nothing happens"
    if not w.isEnabled():
        out["note"] = "disabled: the press does nothing"
    return out
