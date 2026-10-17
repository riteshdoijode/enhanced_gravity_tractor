"""Bennu hits Earth on 2182-09-24 (the scenario without a deflection), on its own:

    uv run python -m app.impact_show            # plays the last two days, the impact and the plume
    uv run python -m app.impact_show --smoke    # screenshots at a few moments into outs/snapshots/, then exits

The clock runs at 2 hours/s for the approach, slows to 5 s/s for the last 1,000 km and to 1 min/s for the 20
minutes after the impact (display speeds: the track is the scenario's full-model orbit, core/earth_impact.py).
Space pauses, R restarts.
"""

import os
import sys

from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

APPROACH_WARP = 7200.0  # s/s for the approach (2 days in 24 s)


def main():
    from core.earth_impact import SHOW_AFTER, SHOW_BEFORE, EarthImpact
    from core.ephemeris import Ephemeris
    from core.scenario import KeyholeScenario
    from core.timeutil import t_to_utc_str

    from app.scenes.earth_impact import EarthImpactScene

    app = QtWidgets.QApplication(sys.argv)
    sc = KeyholeScenario.load()
    eph = Ephemeris().with_bennu(sc.nominal)
    imp = EarthImpact.from_scenario(eph, sc)
    win = QtWidgets.QMainWindow()
    plotter = QtInteractor(win, auto_update=False)
    win.setCentralWidget(plotter.interactor)
    win.resize(1400, 900)
    scene = EarthImpactScene(plotter, eph, sc)
    scene.set_impact(imp, "the scenario without a deflection")

    win.setWindowTitle(f"Bennu hits Earth — {t_to_utc_str(imp.t_hit)[:16]} UTC (no deflection)")
    st = {"t": imp.t_hit - SHOW_BEFORE, "run": True, "last": None, "shots": set()}
    smoke = "--smoke" in sys.argv

    def frame():
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        dt = 0.0 if st["last"] is None else min(0.2, max(0.0, now - st["last"]))
        st["last"] = now
        sm = imp.slow_motion(st["t"])
        if st["run"]:
            st["t"] = min(st["t"] + dt * (sm[0] if sm else APPROACH_WARP), imp.t_hit + SHOW_AFTER)
        if smoke:  # jump through the moments worth a picture
            marks = [("approach", imp.t_hit - SHOW_BEFORE), ("closing", imp.t_hit - 3 * 3600.0),
                     ("last_1000km", imp.t_slowmo() + 30.0), ("impact", imp.t_hit + 3.0),
                     ("plume", imp.t_hit + 300.0), ("end", imp.t_hit + SHOW_AFTER)]
            todo = [m for m in marks if m[0] not in st["shots"]]
            if not todo:
                app.quit()
                return
            name, st["t"] = todo[0]
            sm = imp.slow_motion(st["t"])
        scene.update(st["t"], sm)
        plotter.update()
        if smoke:
            app.processEvents()
            plotter.render()
            os.makedirs("outs/snapshots", exist_ok=True)
            win.grab().save(f"outs/snapshots/impact_{name}.png")
            st["shots"].add(name)
            if name == "impact":
                n = imp.numbers
                print(f"impact {t_to_utc_str(imp.t_hit)} at {n['lat']:.2f}, {n['lon']:.2f} "
                      f"({'ocean' if n['water'] else 'land'}): {n['v_entry'] / 1e3:.2f} km/s, {n['angle_deg']:.0f} deg, "
                      f"{n['energy_mt']:,.0f} Mt; transient {n['transient_m'] / 1e3:.1f} km", flush=True)

    def key(ev):
        if ev.key() == QtCore.Qt.Key_Space:
            st["run"] = not st["run"]
        elif ev.key() == QtCore.Qt.Key_R:
            st["t"] = imp.t_hit - SHOW_BEFORE

    win.keyPressEvent = key
    timer = QtCore.QTimer(win)
    timer.timeout.connect(frame)
    timer.start(33)
    win.show()
    rc = app.exec()
    timer.stop()
    plotter.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
