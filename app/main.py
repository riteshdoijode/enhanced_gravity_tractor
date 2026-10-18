"""Bennu Defense — Mission Control (keyhole scenario: push the 2135 flyby beyond the Moon's distance).

    uv run python -m app.main                # the simulator
    uv run python -m app.main --smoke        # automated check: plan, screenshots, exit
    uv run python -m app.main --smoke-live   # DART -> impact -> gravity tractor launch, screenshots
    uv run python -m app.main --smoke-prox   # ... through the tractor's proximity operations and the tow
    uv run python -m app.main --smoke-fail   # an undersized DART falls short -> recommendation -> DART #2
    uv run python -m app.main --smoke-windows  # DART window markers: shown / hidden, before / after a heavier DART
    uv run python -m app.main --smoke-impact   # planning: jump to the 2182 impact, play through it (impact page)
    uv run python -m app.main --cores 8      # core budget (default: all of them, 12 on an M3 Pro)

Layout: controls on the left; the Solar System in the center (always); on the right Bennu (always), the craft
view (DART or gravity tractor) and the info tabs; plots and the Earth-Moon view in the bottom tabs.

Planning mode previews the plan (the planned trajectories; no physics is run). 🚀 Launch DART starts the live
run: from then on everything shown is computed by the physics in the simulation process, at exactly the chosen
time warp. If the physics needs time (Bennu's full-force-model orbit, an orbit determination, a re-simulation),
the 3-D views make way for a "Recalculating physics" page with its progress. Changing any parameter during the
run flies the whole mission again with it (core/live_runner.py: re-simulation).

Cores: the window, the live simulation, three planner workers and the full-model refinement are separate
processes that share the core budget (core/parallel.py: budget()); the status bar shows the split at start.
"""

import os
import sys
from dataclasses import asdict, replace

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from app.livesim import make_sim
from app.panels import SPEEDS, START_DATE, ControlsPanel, LivePanel, MissionPanel, TelemetryPanel
from app.plots import BPlanePlot, DistancePlot, EtaPlot, LivePlots, PorkchopPlot, ProxPlots, SunPlot
from app.scenes import BennuScene, CraftScene, EarthImpactScene, EarthMoonScene, SolarScene
from app.scenes.sky import SKY_FADE_S
from app.scenes.solar import GALACTIC_SLOW
from app.session_log import SessionLog
from app.workers import Debounced
from core.constants import AU, DAY
from core.earth_impact import SHOW_AFTER, SHOW_BEFORE, EarthImpact
from core.ephemeris import Ephemeris
from core.planner import GROUND_OD_DAYS, Planner, dart_params, physical_params
from core.scenario import KeyholeScenario
from core.timeutil import iso_to_t, t_to_utc_str

ETA_THRUSTS = np.geomspace(0.5, 3000.0, 18)
SIDE_RENDER_EVERY = 3  # the Bennu and craft views render every 3rd frame (~7 Hz), every frame around an impact
LAUNCH_LEAD_DAYS = 10.0  # a live run starts this long before the DART's orbit raising (in the chosen window)


def _interactor(parent):
    """A PyVista Qt view rendered only when the window asks for it. pyvistaqt's defaults (a 5 Hz render timer
    per view, and on macOS a new thread for every render request, including one per actor added or removed)
    made the GUI thread render the same frame many times over; here every draw is explicit (_draw)."""
    p = QtInteractor(parent, auto_update=False)
    p.render = lambda *a, **k: None  # PyVista's internal render-after-change calls
    return p


def _draw(p):
    """Ask Qt to repaint one view if it is on screen. Its paint renders the VTK scene exactly once, and Qt merges
    repeated requests within a frame (a direct VTK render would draw twice: once now, and again when the
    finished frame schedules the widget's paint)."""
    if p.isVisible():
        p.update()


def _titled(widget, title, extra=None):
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    row = QtWidgets.QHBoxLayout()
    lab = QtWidgets.QLabel(f"<b>{title}</b>")
    lab.setStyleSheet("color: #c3c2b7; padding-left: 4px;")
    row.addWidget(lab)
    row.addStretch(1)
    for w in extra or []:
        row.addWidget(w)
    lay.addLayout(row)
    lay.addWidget(widget, 1)
    return box


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bennu Defense — Mission Control")
        self.resize(1800, 1080)
        self.log = SessionLog(parent=self)  # what was clicked and how the app responded (app/session_log.py)
        self.log.write("session", what="start", argv=sys.argv)
        self._seen_events, self._log_state, self._log_wall = set(), None, 0.0
        self.sc = KeyholeScenario.load()
        self.eph = Ephemeris().with_bennu(self.sc.nominal)
        self.planner = Planner(self.eph, self.sc.nominal, self.sc)
        lo, hi = self.eph.span("earth")
        self.t = iso_to_t(START_DATE)
        self.result = None
        self.sim = None
        self._frame = 0

        self.controls = ControlsPanel(lo, hi)
        self.controls.setMinimumWidth(390)
        # center: the Solar System
        self.p_solar = _interactor(self)
        self.solar = SolarScene(self.p_solar, self.eph)
        # right column: Bennu, the craft view, the info tabs
        self.p_bennu = _interactor(self)
        self.bennu = BennuScene(self.p_bennu, self.eph)
        self.p_craft = _interactor(self)
        self.craft = CraftScene(self.p_craft, self.eph)
        self.craft_mode = QtWidgets.QButtonGroup(self)
        btns = []
        for i, label in enumerate(["Auto", "DART", "Tractor"]):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(22)
            self.craft_mode.addButton(b, i)
            btns.append(b)
        btns[0].setChecked(True)
        reset = QtWidgets.QPushButton("⟲")
        reset.setFixedSize(26, 22)
        reset.setToolTip("Reset the chase camera")
        reset.clicked.connect(lambda: (self.craft.reset_view(), _draw(self.p_craft)))
        self.mission = MissionPanel()
        self.live_panel = LivePanel()
        self.telemetry = TelemetryPanel(self.sc)
        self.info = info = QtWidgets.QTabWidget()
        info.addTab(self.mission, "Mission plan")
        info.addTab(self.live_panel, "Live flight")
        info.addTab(self.telemetry, "Bennu now")
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for w in (self.p_bennu.interactor, self.p_craft.interactor):
            w.setMinimumHeight(190)
        right.addWidget(_titled(self.p_bennu.interactor, "Bennu"))
        right.addWidget(_titled(self.p_craft.interactor, "Spacecraft", btns + [reset]))
        right.addWidget(info)
        info.setMinimumHeight(160)
        for k, f in enumerate((3, 3, 4)):
            right.setStretchFactor(k, f)
        right.setSizes([300, 280, 360])
        right.setMinimumWidth(480)
        self._right = right

        # bottom tabs
        self.distance = DistancePlot(self.eph, self.sc)
        self.bplane = BPlanePlot()
        self.porkchop = PorkchopPlot()
        self.eta = EtaPlot()
        self.live_plots = LivePlots()
        self.prox_plots = ProxPlots()
        self.sun_plot = SunPlot()
        self.p_em = _interactor(self)
        self.earth_moon = EarthMoonScene(self.p_em, self.eph, self.sc)
        self.bottom = bottom = QtWidgets.QTabWidget()
        bottom.addTab(self.distance, "Earth–Bennu distance · DART opportunities")
        bottom.addTab(self.bplane, "B-plane 2135 (goal)")
        bottom.addTab(self.p_em.interactor, "Earth–Moon (2135 keyhole flyby)")
        bottom.addTab(self.porkchop, "DART porkchop")
        bottom.addTab(self.live_plots, "Live flight")
        bottom.addTab(self.prox_plots, "Near Bennu")
        bottom.addTab(self.eta, "Tractor: arrival vs thrust")
        bottom.addTab(self.sun_plot, "Sun around the barycenter")
        self.show_windows = QtWidgets.QCheckBox("Show DART window markers")  # next to the distance plot's tab
        self.show_windows.setChecked(True)
        self.show_windows.setToolTip("Show or hide the DART window markers, their legend and the 'selected' label\n"
                                     "(a clean distance plot; the selected window and the mission do not change)")
        self.show_windows.toggled.connect(self.distance.set_markers_visible)
        bottom.setCornerWidget(self.show_windows, QtCore.Qt.TopRightCorner)

        # center: the Solar System, or "Recalculating physics…" while the physics catches up (no stale picture)
        self.recalc = RecalcPage()
        self.center = QtWidgets.QStackedWidget()
        self.home = home = QtWidgets.QPushButton("🏠")
        home.setCheckable(True)
        home.setFixedSize(30, 22)
        home.setToolTip("Side view of the Solar System moving around the Milky Way (north Galactic pole up, the\n"
                        "Galactic center behind), with the paths the Sun and planets trace. Click again to leave it.")
        home.toggled.connect(self._solar_home)
        self.wide = wide = QtWidgets.QPushButton("🪐")
        wide.setCheckable(True)
        wide.setFixedSize(30, 22)
        wide.setToolTip("Zoom out to the whole Solar System: Jupiter, Saturn, Uranus, Neptune and Pluto's orbits\n"
                        "(out to ~49 AU). Click again for the inner planets.")
        wide.toggled.connect(self._solar_wide)
        self.center.addWidget(_titled(self.p_solar.interactor, "Solar System", [wide, home]))
        self.center.addWidget(self.recalc)
        # page 2: Bennu hits Earth (the last two days and 20 minutes after; app/scenes/earth_impact.py)
        self.p_impact = _interactor(self)
        self.earth_impact = EarthImpactScene(self.p_impact, self.eph, self.sc)
        self.center.addWidget(_titled(self.p_impact.interactor, "Bennu hits Earth"))
        self._impact_on = False
        self._impacts = {}  # cached EarthImpact per track: "scenario" or ("live", t_hit)
        self._aftermath = None  # a live run that ended with the impact: the display plays the 20 minutes after it
        self._side_views = [right.widget(0), right.widget(1)]
        self._busy_since = None
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(self.controls)
        top.addWidget(self.center)
        top.addWidget(right)
        top.setStretchFactor(1, 1)
        top.setSizes([400, 900, 500])
        main = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main.addWidget(top)
        main.addWidget(bottom)
        main.setSizes([760, 300])
        self._main_split = main
        self.setCentralWidget(main)
        self.status = self.statusBar()

        # background planning jobs, computed in worker processes (app/compute.py)
        from app import compute

        self.compute = compute.ComputePool()
        self._fp_arrival = None
        self.plan_job = Debounced(lambda: (lambda p=self._with_refined(self.controls.params()), ta=self._fp_arrival:
                                           self.compute.run("plan", compute.job_plan, p, ta)), 150, self)
        self.plan_job.result.connect(self._on_plan)
        self.plan_job.error.connect(self.mission.show_error)
        self.plan_job.busy.connect(lambda b: self.status.showMessage("planning…" if b else "plan up to date"))
        self.eta_job = Debounced(lambda: (lambda p=self.controls.params(), tl=self._tractor_launch_time():
                                          self.compute.run("tractor", compute.job_eta, p, tl, ETA_THRUSTS)), 400, self)
        self.eta_job.result.connect(self._on_eta)
        self.fp_job = Debounced(lambda: (lambda p=self.controls.params(), tl=self._tractor_launch_time():
                                         self.compute.run("tractor", compute.job_flight_plan, p, tl)), 700, self)
        self.fp_job.result.connect(self._on_flight_plan)
        self._fp_key = None
        # planning mode: the tractor's launch window (the blue band of the distance plot) for the planned DART
        self.win_job = Debounced(lambda: (lambda p=self.controls.params(), tl=self._tractor_launch_time(),
                                          b=self._planned_b_dart():
                                          self.compute.run("tractor", compute.job_tractor_window, p, tl, b)), 900, self)
        self.win_job.result.connect(self._on_tractor_window)
        # full-model refinement of the recommended DART mass, in a separate process (core/refine.py)
        self._refiner = None
        self._refined = {}  # key -> refined mass
        self._ref_future, self._ref_key = None, None
        self._ref_timer = QtCore.QTimer(self)
        self._ref_timer.setInterval(500)
        self._ref_timer.timeout.connect(self._poll_refine)
        self._ref_timer.start()
        # live: recommendation for the next DART, tractor plan after a success
        self.rec_job = Debounced(lambda: (lambda a=self._rec_args(), sp=(self._last_snap or {}).get("spin_change"):
                                          self.compute.run("live", compute.job_recommend, *a, sp)), 200, self)
        self.rec_job.result.connect(self._on_recommendation)
        self.rec_job.error.connect(self.live_panel.show_error)
        self.tr_job = Debounced(lambda: (lambda a=self._tr_args():
                                         self.compute.run("live", compute.job_tractor, *a)), 200, self)
        self.tr_job.result.connect(self._on_tractor_plan)
        self.tr_job.error.connect(self.live_panel.show_error)
        self.tr_job.error.connect(lambda msg: self.controls.set_tractor_status(
            "red", "Not available: the tractor plan failed — " + msg.strip().splitlines()[0][:120]))
        self.tr_job.busy.connect(lambda b: setattr(self, "_tr_busy", b))
        self._tr_busy = False
        self._tr_sent = False  # the launch was ordered and the snapshot does not show it yet
        self._tr_planned_t = -np.inf  # mission time the current tractor plan was made for ("launch now")
        self._tr_want_window = True  # the next tractor job also sweeps the launch window (b_now changed)
        self._tr_ahead = None  # the DART impact for which the tractor was planned ahead of the verdict
        self._tractor_window = None
        # planning mode, the slider past the selected window: plan the DART from the slider time, then launch
        self._launch_from = None
        self.launch_job = Debounced(lambda: (lambda p=self.controls.params(), t=self._launch_from:
                                             self.compute.run("live", compute.job_launch_plan, p, t)), 0, self)
        self.launch_job.result.connect(self._on_launch_plan)
        self.launch_job.error.connect(self.mission.show_error)
        for name in ("plan_job", "eta_job", "fp_job", "win_job", "rec_job", "tr_job", "launch_job"):
            getattr(self, name).error.connect(lambda msg, n=name: self.log.write("job", job=n, error=msg))
        self.next_dart = None
        self.next_tractor = None
        self._tractor_check = None
        self._next_choice = -1  # window picked for the next DART after a shortfall (-1: the recommended one)
        self._live_params = None  # the parameters the running physics uses
        self._live_dirty = False  # parameters changed during the run: re-simulate once the plan is ready
        self._run_t_start, self._run_window, self._run_plan = None, None, None
        self._launch_when_planned = False
        self._results_seen = 0
        self._last_snap = None
        self._was_busy = False

        c = self.controls
        c.on_stage = lambda stage, dart, tractor, by: self.log.write(
            "buttons", stage=stage, launch_dart=dart, launch_tractor=tractor, by=by, next_dart=self.next_dart,
            next_tractor=self.next_tractor)
        c.gravity_overlay.toggled.connect(lambda on: (self.bennu.set_coloring("gravity" if on else "regolith"),
                                                      _draw(self.p_bennu)))
        c.sun_wobble.toggled.connect(lambda on: (self.solar.set_sun_wobble(30.0 if on else 1.0), _draw(self.p_solar)))
        c.galactic_slow.toggled.connect(lambda on: (self.solar.set_galactic_slow(GALACTIC_SLOW if on else 1.0),
                                                    _draw(self.p_solar)))
        c.slider.valueChanged.connect(lambda d: self.set_time(lo + d * DAY))
        c.constellations.toggled.connect(self._apply_constellations)
        self._sky_timer = QtCore.QTimer(self)  # the background cross-fade (Milky Way <-> constellations)
        self._sky_timer.setInterval(33)
        self._sky_timer.timeout.connect(self._sky_tick)
        self._sky_fade = (0.0, 0.0, 0.0)
        # the earliest DART launch and the time slider, two-way: the slider dragged by hand (actionTriggered: user
        # actions only, not Play or Jump to…) sets the date 400 ms after the last move; a new date moves the clock
        self._slider_date = QtCore.QTimer(self)
        self._slider_date.setSingleShot(True)
        self._slider_date.setInterval(400)
        self._slider_date.timeout.connect(self._slider_to_launch_date)
        c.slider.actionTriggered.connect(lambda _: c.follow_slider.isChecked() and self._slider_date.start())
        c.launch_after.dateChanged.connect(self._launch_date_to_clock)
        self._date_from_slider = False
        c.play.toggled.connect(self._on_play)
        c.jump.activated.connect(self._on_jump)
        c.paramsChanged.connect(self._on_params)
        c.launch_dart.clicked.connect(self._on_launch_dart)
        c.launch_tractor.clicked.connect(self._on_launch_tractor)
        c.release.clicked.connect(self._on_release)
        c.storm.clicked.connect(lambda: self.sim and self.sim.send("storm", c.storm_days.value()))
        c.skip.clicked.connect(lambda: self.sim and self.sim.send("skip"))
        c.finish.clicked.connect(lambda: self.sim and self.sim.send("finish"))
        c.end_live.clicked.connect(self._end_live)
        c.speed.currentIndexChanged.connect(lambda _: self.sim and self.sim.send("warp", self._warp()))
        c.slowmo.toggled.connect(lambda on: self.sim and self.sim.send("slowmo", on))
        c.prox_ctrl.currentIndexChanged.connect(lambda _: self.sim and self.sim.send("controller", c.prox_ctrl.currentData()))
        self.craft_mode.idClicked.connect(lambda _: self._refresh_craft_kind())
        self.distance.clicked_time.connect(self.set_time)
        self.distance.opportunity_clicked.connect(self._on_opportunity_clicked)
        self.bottom.currentChanged.connect(lambda _: (self._render_em(), self.show_windows.setVisible(
            self.bottom.currentWidget() is self.distance)))
        self.info.currentChanged.connect(lambda _: self._refresh_info())
        self.engine = make_sim(self.eph, self.planner, self)
        send = self.engine.send

        def logged_send(name, *args):
            self.log.write("command", name=name, args=list(args))
            return send(name, *args)

        self.engine.send = logged_send
        self.engine.snap.connect(self._on_snap)
        self.engine.failed.connect(self.live_panel.show_error)
        self.engine.failed.connect(lambda msg: self.log.write("error", where="simulation", text=msg))
        self.engine.ready.connect(lambda: self.status.showMessage("Simulation engine ready (separate process)", 4000))
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._last_tick = None
        self._after_timer = QtCore.QTimer(self)  # the 20 minutes after an Earth impact that ended a live run
        self._after_timer.setInterval(33)
        self._after_timer.timeout.connect(self._aftermath_tick)

        self._set_jumps()
        self.set_time(self.t)
        self._on_params()
        self.sun_job = Debounced(lambda: (lambda: self.compute.run("tractor", compute.job_sun)), 0, self)
        self.sun_job.result.connect(lambda d: (self.sun_plot.set_data(d), self.sun_plot.set_time(self.t)))
        self.sun_job.request()

    # ------------------------------------------------------------------ planning
    def _tractor_launch_time(self):
        r = self.result
        if r is not None and r.tractor.feasible:
            return r.tractor.t_launch
        if r is not None and r.impactor.feasible:
            return r.impactor.t_arrive + GROUND_OD_DAYS * DAY
        return iso_to_t("2031-01-01")

    def _on_params(self):
        p = self.controls.params()
        old = getattr(self, "_log_params", None)
        if old is not None:
            changed = {k: (getattr(old, k), v) for k, v in asdict(p).items() if getattr(old, k) != v}
            self.log.write("params", changed=changed, live=self.sim is not None,
                           refly=self.sim is not None and _physical(p) != _physical(self._live_params))
        self._log_params = p
        if self.sim is not None:  # live run: the physics is flown again with the new parameters
            if _physical(self.controls.params()) == _physical(self._live_params):
                return  # (a display or storm setting: nothing physical changed)
            s = self._last_snap or {}
            if dart_params(p) == dart_params(self._live_params) and not s.get("tractor_launched") and not self._live_dirty:
                # only the gravity tractor's settings changed and it has not launched: nothing that has flown
                # depends on them, so the mission is not re-flown; the tractor is re-planned (and the Mission plan
                # preview updated) for the new settings
                self.log.write("action", what="tractor settings changed before its launch: re-plan the tractor only")
                self._live_params = p
                self._tr_want_window = True
                ki = s.get("impactor") or {}
                if s.get("dart_ok") or ki.get("phase") == "impacted":
                    self.next_tractor = None
                    self._tr_ahead = None
                    self.controls.set_stage("live", dart_ready=self.controls.launch_dart.isEnabled(), tractor_ready=False)
                    self.tr_job.request()
                    self._update_tractor_status(s)
                self.plan_job.request()
                return
            self._live_dirty = True
            self._set_recalc("Recalculating physics", "Planning the DART for the new parameters…", None)
        self.plan_job.request()

    def _resimulate(self, res):
        """Live run, parameters changed and their plan is ready: fly the whole mission again with them. When only the
        gravity tractor's settings changed, DART #1 is the one that flew (not re-planned: in the meantime the
        recommended mass may have been re-sized with the full force model)."""
        p = res.params
        keep = self._run_plan is not None and dart_params(p) == dart_params(self._live_params)
        ip = self._run_plan if keep else res.impactor
        if keep:
            self.log.write("action", what="re-fly keeps DART #1 as flown (only tractor settings changed)", plan=ip)
        if not ip.feasible:
            self.log.write("action", what="re-fly refused: no feasible DART with these parameters")
            self._set_recalc(None)
            self.status.showMessage("These parameters give no feasible DART: the run keeps flying the previous "
                                    "ones (see the Mission plan checks)", 10000)
            return
        if not keep and p.use_recommended_mass and p.refined_mass_kg <= 0:  # wait for the full-model mass, then fly
            self.log.write("action", what="re-fly waits for the full-model DART mass")
            self._set_recalc("Recalculating physics", "Sizing the DART with the full force model…", None)
            return
        self.log.write("action", what="re-fly the mission with the new parameters", plan=ip)
        self._live_dirty = False
        same_window = ip.opportunity == self._run_window
        if same_window:
            t_start, t_back = min(self._run_t_start, ip.raising.t_start - 3600.0), None
        else:  # another launch window: the run starts there, as a launch in that window would
            t_start = ip.raising.t_start - LAUNCH_LEAD_DAYS * DAY
            t_back = t_start
        self._run_t_start, self._run_window, self._run_plan = t_start, ip.opportunity, ip
        self._live_params = self.controls.params()
        self._results_seen = 0
        self.next_dart = self.next_tractor = None
        self._tr_ahead, self._tr_planned_t, self._tractor_window, self._tr_want_window = None, -np.inf, None, True
        self._tr_sent = False
        self._aftermath = None
        self.distance.set_tractor(None)
        self.live_panel.recommendation = None
        self.controls.set_stage("live", dart_ready=False, tractor_ready=False)
        self.live_plots.reset()
        self.prox_plots.reset()
        self.bplane.clear_live()
        self.distance.set_deflected(None, None)
        for scene in (self.solar, self.bennu, self.craft, self.earth_moon):
            scene.clear_live()
        self._impact_framed = 0
        self.sim.send("resim", replace(p, refined_mass_kg=0.0), ip, t_start, t_back)  # (the mass travels in ip)
        self._set_recalc("Recalculating physics", "Flying the mission again with the new parameters…", 0.0)

    # ------------------------------------------------------------------ full-model mass refinement
    def _refine_key(self, p, live=None):
        base = (p.goal_ld, p.dart_target_km, round(p.beta, 2), p.dart_max_speed_kms, p.opportunity, p.parking_alt_km)
        # planning mode: the earliest launch picks the arc (a live run's next DARTs launch from its clock instead)
        return base + (p.earliest_launch,) if live is None else base + live

    def _slider_to_launch_date(self):
        """The time slider was dragged by hand: the DART's earliest launch follows it (planning mode)."""
        if self.sim is not None or not self.controls.follow_slider.isChecked():
            return
        self._date_from_slider = True
        try:
            self.controls.set_earliest_launch(self.t)
        finally:
            self._date_from_slider = False

    def _launch_date_to_clock(self, _date=None):
        """A new earliest-launch date typed or picked: the clock moves there (unless the slider set it)."""
        if self.sim is None and not self._date_from_slider:
            self.log.write("action", what="clock moved to the earliest launch", t=self.controls.params().earliest_launch)
            self.set_time(self.controls.earliest_launch_t())

    def _with_refined(self, p, live=None):
        m = self._refined.get(self._refine_key(p, live))
        return replace(p, refined_mass_kg=m) if (m is not None and p.use_recommended_mass) else p

    def _submit_refine(self, key, args):
        if key in self._refined or key == self._ref_key:
            return
        if self._ref_future is not None and not self._ref_future.done():
            self._ref_future.cancel()  # only the newest request matters (a running one aborts, core/refine.py)
        from core.refine import refine_init, refine_job

        if self._refiner is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            ctx = mp.get_context("spawn")
            self._ref_gen = ctx.Value("i", 0)  # bumped per request: a superseded refinement stops its workers
            self._refiner = ProcessPoolExecutor(1, mp_context=ctx, initializer=refine_init, initargs=(self._ref_gen,))
        with self._ref_gen.get_lock():
            self._ref_gen.value += 1
            job_id = self._ref_gen.value
        self._ref_key = key
        self._ref_future = self._refiner.submit(refine_job, args, job_id)
        self.mission.refining = self.live_panel.refining = True

    def _poll_refine(self):
        f = self._ref_future
        if f is None or not f.done():
            return
        key, self._ref_future, self._ref_key = self._ref_key, None, None
        self.mission.refining = self.live_panel.refining = False
        try:
            r = f.result()
        except Exception as exc:  # report and keep the linear estimate
            self.log.write("job", job="refinement", error=str(exc))
            self.status.showMessage(f"Full-model refinement failed: {exc}", 8000)
            return
        if r is None:
            return
        self.log.write("result", what="full-model DART mass", mass=r["mass"], linear=r["linear_mass"], key=list(key))
        self._refined[key] = r["mass"]
        self.status.showMessage(f"Full model: recommended DART mass {r['mass']:,.0f} kg "
                                f"(linear estimate {r['linear_mass']:,.0f} kg)", 8000)
        if self.sim is None or self._live_dirty:
            self.plan_job.request()
        elif len(key) > 6 and self.next_dart is not None:  # live recommendation: re-plan with the refined mass
            self.rec_job.request()

    def _on_plan(self, res):
        self.log.write("result", what="plan", feasible=res.impactor.feasible, dart=res.impactor,
                       live=self.sim is not None, live_dirty=self._live_dirty)
        self.result = res
        if self.sim is not None and self._live_dirty:
            self._resimulate(res)
        if self._launch_when_planned and self.sim is None and self._plan_is_current():
            self._launch_when_planned = False
            QtCore.QTimer.singleShot(0, self._on_launch_dart)
        self.mission.show_plan(res)
        self.bplane.update_plan(res)
        self.porkchop.update_plan(res)
        rec = self.planner.recommended(res.opportunities)
        self.controls.set_opportunities(res.opportunities, res.params.opportunity, rec.index if rec else None)
        self.distance.set_opportunities(res.opportunities, res.impactor.opportunity if res.impactor.feasible else None,
                                        _window_summary(res.capability, res.impactor))
        self.controls.set_goal_info(res)
        if res.impactor.feasible and not self.controls.use_rec.isChecked():
            pass
        elif res.impactor.feasible:
            self.controls.imp_mass.set_value(res.impactor.mass)
        self.solar.set_plan(res)
        p = res.params
        if res.impactor.feasible and p.use_recommended_mass and p.refined_mass_kg <= 0:
            self._submit_refine(self._refine_key(p), (replace(p, refined_mass_kg=0.0), p.opportunity, None, None, (), 1,
                                                      None))
        c = res.impactor.contact if res.impactor.feasible else None
        self.bennu.set_contact(c["point_b"] if c and "point_b" in c else None)
        self._set_jumps()
        key = (round(self._tractor_launch_time() / DAY), res.params.tractor_thrust_n, res.params.tractor_mass_kg,
               res.params.tractor_objective)
        if key != self._fp_key:
            self._fp_key = key
            self._fp_arrival = None
            self.mission.set_flight_plan(None)
            self.fp_job.request()
            self.eta_job.request()
        if self.sim is None:  # planning preview: the tractor's launch window for this plan (the blue band)
            wkey = key + (tuple(np.round(self._planned_b_dart() / 1e3)), res.params.boulder_mass_kg,
                          res.params.tow_years, res.params.hover_factor, res.params.goal_ld, res.params.dart_target_km)
            if wkey != getattr(self, "_win_key", None):
                self._win_key = wkey
                self.win_job.request()
            w = self._tractor_window
            self.distance.set_tractor(w, res.tractor.path_t if res.tractor.feasible else None,
                                      res.tractor.path if res.tractor.feasible else None, self.eph)
        self._render_all()

    def _planned_b_dart(self):
        """The 2135 point after the planned DART (the tractor window in planning mode is drawn for it)."""
        r = self.result
        return self.planner.b0 + (r.impactor.db if r is not None and r.impactor.feasible else np.zeros(2))

    def _on_tractor_window(self, window):
        self.log.write("result", what="tractor launch window", t_last=window.get("t_last"), gap_km=window.get("gap", 0) / 1e3,
                       n_ok=int(np.sum(window["ok"])))
        self._tractor_window = window
        r = self.result
        if self.sim is None and r is not None:
            self.distance.set_tractor(window, r.tractor.path_t if r.tractor.feasible else None,
                                      r.tractor.path if r.tractor.feasible else None, self.eph)

    def _on_flight_plan(self, fp):
        self.mission.set_flight_plan(fp)
        new = fp["t_arrive"] if fp and fp.get("feasible") else None
        if new != self._fp_arrival and self.sim is None:
            self._fp_arrival = new
            self.plan_job.request()

    def _on_eta(self, out):
        t_launch, curve = out
        r = self.result
        if r is not None and r.tractor.feasible:
            self.eta.update_curve(curve, r.params.tractor_thrust_n, r.tractor.t_arrive)

    def _on_opportunity_clicked(self, index):
        if self.sim is not None:  # live: the window for the next DART after a shortfall
            if self.next_dart is not None or (self._last_snap and self._awaiting_next_dart(self._last_snap)):
                self._next_choice = index
                self.rec_job.request()
            return
        self.controls.select_opportunity(index)

    def _set_jumps(self):
        items = [(f"Today ({START_DATE[:10]})", iso_to_t(START_DATE))]
        r = self.result
        if r is not None and r.impactor.feasible:
            ip = r.impactor
            items += [("DART orbit raising starts", ip.raising.t_start), ("DART departure", ip.t_launch),
                      ("DART impact", ip.t_arrive)]
        if r is not None and r.tractor.feasible:
            items += [("Tractor launch", r.tractor.t_launch), ("Tractor arrives at Bennu", r.tractor.t_arrive)]
        for a in self.sc.approaches:
            if a["dist"] < 0.1 * AU:
                items.append((f"Earth approach {t_to_utc_str(a['t'])[:10]} ({a['dist'] / AU:.4f} au)", a["t"]))
        items += [("2135 keyhole flyby", self.sc.t_ca), (f"Earth impact {t_to_utc_str(self.sc.t_impact)[:10]}",
                                                         self.sc.t_impact - SHOW_BEFORE)]
        self.controls.set_jumps(items)

    def _on_jump(self, idx):
        t = self.controls.jump.itemData(idx)
        self.controls.jump.setCurrentIndex(0)
        if t is not None and np.isfinite(t):
            self.set_time(t)

    # ------------------------------------------------------------------ time and rendering
    def set_time(self, t, render=True, side=True):
        """Move every view to time t. side: also the Bennu and craft views (playback and the live run refresh
        them on every third frame); hidden tabs are skipped and refreshed when they come back on screen."""
        lo, hi = self.eph.span("earth")
        self.t = float(np.clip(t, lo, hi))
        self.solar.update(self.t)
        if side:
            self.bennu.update(self.t)
            self.craft.update(self.t)
        self.controls.set_time(self.t)
        if self.info.currentWidget() is self.telemetry:
            live_b = self._last_snap.get("bennu_pos") if self.sim is not None and self._last_snap else None
            self.telemetry.update(self.eph, self.t, live_b)
        bottom = self.bottom.currentWidget()
        if side or self.sim is None and not self.timer.isActive():  # the plot cursors repaint their whole plot:
            if bottom is self.distance:                              # ~10 Hz during playback is plenty
                self.distance.set_time(self.t)
            elif bottom is self.sun_plot:
                self.sun_plot.set_time(self.t)
        self._impact_page(self.t)
        if render:
            self._render_all(side=side)

    # ------------------------------------------------------------------ Bennu hits Earth (centre page 2)
    def _impact_track(self):
        """(EarthImpact, label) of the impact the views show, or None: in planning mode the scenario without a
        deflection; in a live run its verified orbit, when that still hits Earth (core/earth_impact.py)."""
        if self.sim is None:
            key, label, fc = "scenario", "the scenario without a deflection", None
        else:
            out = (self._last_snap or {}).get("outlook") or {}
            t_hit = out.get("t_earth_impact", np.nan)
            if t_hit is None or not np.isfinite(t_hit):
                return None
            fc = out.get("final_check") or {}
            key = "scenario" if abs(t_hit - self.sc.t_impact) < 600.0 or "fine_t" not in fc else ("live", round(t_hit))
            label = "this run: its orbit still hits Earth"
        imp = self._impacts.get(key)
        if imp is None:
            imp = self._impacts[key] = (EarthImpact.from_scenario(self.eph, self.sc) if key == "scenario" else
                                        EarthImpact.from_track(self.eph, fc["fine_t"], fc["fine_x"]))
        return imp, label

    def _impact_slowmo(self, imp, t):
        """(warp, what) of the slow motion shown on the impact page, or None."""
        if not self.controls.slowmo.isChecked():
            return None
        if self.sim is not None and self._aftermath is None:
            sm = (self._last_snap or {}).get("slowmo")
            return (sm["warp"], sm["what"]) if sm else None
        return imp.slow_motion(t)

    def _impact_page(self, t):
        """The centre shows Bennu hitting Earth while the clock is in the last two days before the impact or the
        20 minutes after it (a live run that ended with the impact: the aftermath timer's display time)."""
        got = self._impact_track()
        am = self._aftermath if self.sim is not None else None
        t_show = am["t"] if am is not None else t
        on = (got is not None and got[0].t_hit - SHOW_BEFORE <= t_show <= got[0].t_hit + SHOW_AFTER
              and not (am is not None and am["done"]))
        if on:
            imp, label = got
            self.earth_impact.set_impact(imp, label)
            self.earth_impact.update(t_show, self._impact_slowmo(imp, t_show))
            _draw(self.p_impact)
        if on != self._impact_on:
            self._impact_on = on
            self.log.write("action", what=f"Earth impact page {'on' if on else 'off'}", t=t_to_utc_str(t_show)[:19])
            if self.center.currentIndex() != 1:  # (the recalculation page keeps priority)
                self.center.setCurrentIndex(2 if on else 0)

    def _start_aftermath(self, t_hit):
        """A live run ended with the impact (the physics stops there): the impact page plays the 20 minutes after it
        (the plume is a display model), at 1 min/s with slow motion on, else at the chosen speed."""
        self._aftermath = {"t": float(t_hit), "done": False, "t_hit": float(t_hit), "last": None}
        self._after_timer.start()

    def _aftermath_tick(self):
        am = self._aftermath
        if am is None or self.sim is None:
            self._after_timer.stop()
            return
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        dt = 0.033 if am["last"] is None else max(0.0, min(0.2, now - am["last"]))
        am["last"] = now
        warp = 60.0 if self.controls.slowmo.isChecked() else self._warp()
        am["t"] = am["t"] + dt * warp
        if am["t"] > am["t_hit"] + SHOW_AFTER:
            am["done"] = True
            self._after_timer.stop()
        self._impact_page(self.t)

    def _solar_home(self, on):
        """The Solar System view from the side of its motion around the Milky Way, with the traced paths
        (core/galaxy.py, app/scenes/solar.py); off: back to the start view."""
        if on:
            from core.galaxy import solar_motion

            if self.wide.isChecked():  # (the side view takes the camera: 🪐 is released)
                self.wide.blockSignals(True)
                self.wide.setChecked(False)
                self.wide.blockSignals(False)
            m = solar_motion()
            self.solar.set_galactic(m)
            self.status.showMessage(f"Solar System side view: it moves left around the Milky Way at {m['speed_kms']:.0f} "
                                    f"km/s toward {m['constellation']} (l {m['l_deg']:.1f}°, b {m['b_deg']:+.1f}°); up "
                                    f"is the north Galactic pole; the planets' orbits are tilted "
                                    f"{m['ecliptic_tilt_deg']:.1f}° to the Galactic plane", 12000)
        else:
            self.solar.set_galactic(None)
            self.solar.reset_camera()
        _draw(self.p_solar)

    def _solar_wide(self, on):
        """🪐: the whole Solar System out to Pluto's orbit, or back to the inner planets."""
        if on and self.home.isChecked():
            self.home.setChecked(False)  # (leaves the Galactic side view first)
        self.solar.reset_camera(wide=on)
        _draw(self.p_solar)

    def _apply_constellations(self, _on=None):
        """The background of every view, the main ones included: the Milky Way, or the constellations with the
        Display toggle. The switch cross-fades over SKY_FADE_S (the image fades out as the figures fade in)."""
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        self._sky_fade = (self.solar.sky.mix or 0.0, 1.0 if self.controls.constellations.isChecked() else 0.0, now)
        self.log.write("action", what="background", to="constellations" if self._sky_fade[1] else "Milky Way")
        self._sky_timer.start()

    def _sky_tick(self):
        c0, c1, t0 = self._sky_fade
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        f = min(1.0, max(0.0, (now - t0) / SKY_FADE_S))
        c = c0 + (c1 - c0) * f * f * (3 - 2 * f)
        for scene, p in ((self.solar, self.p_solar), (self.bennu, self.p_bennu), (self.craft, self.p_craft),
                         (self.earth_moon, self.p_em), (self.earth_impact, self.p_impact)):
            scene.sky.set_mix(c)
            _draw(p)
        if f >= 1.0:
            self._sky_timer.stop()

    def _render_all(self, side=True):
        _draw(self.p_solar)
        if side:
            _draw(self.p_bennu)
            _draw(self.p_craft)
        self._render_em()

    def _render_em(self):
        if self.bottom.currentWidget() is self.p_em.interactor:
            if self.t is not None:
                self.earth_moon.update(self.t)
            _draw(self.p_em)
        elif self.t is not None:  # a tab that just came on screen catches up
            w = self.bottom.currentWidget()
            if w is self.distance:
                self.distance.set_time(self.t)
            elif w is self.sun_plot:
                self.sun_plot.set_time(self.t)

    def _set_recalc(self, title, detail="", fraction=None):
        """Blank the 3-D views and say what the physics is computing (title None: show the views again)."""
        if title is None:
            page = 2 if self._impact_on else 0
            if self.center.currentIndex() != page:
                self.center.setCurrentIndex(page)
                for w in self._side_views:
                    w.setVisible(True)
                self._render_all()
            return
        self.recalc.show_state(title, detail, fraction)
        if self.center.currentIndex() != 1:
            self.center.setCurrentIndex(1)
            for w in self._side_views:
                w.setVisible(False)

    def _refresh_info(self):
        """An info tab that comes on screen catches up at once (it is refreshed only while visible)."""
        w = self.info.currentWidget()
        if w is self.live_panel and self._last_snap is not None and self.sim is not None:
            self.live_panel.show_snapshot(self._last_snap)
        elif w is self.telemetry and self.t is not None:
            live_b = self._last_snap.get("bennu_pos") if self.sim is not None and self._last_snap else None
            self.telemetry.update(self.eph, self.t, live_b)

    def _on_play(self, on):
        self.controls.play.setText("⏸ Pause" if on else "▶ Play")
        if self.sim is not None:
            self.sim.send("resume" if on else "pause")
            return
        self._last_tick = None
        self.timer.start() if on else self.timer.stop()

    def _tick(self):
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        dt = 0.033 if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now
        self._frame += 1
        warp = self._warp()
        t_new = self.t + dt * warp
        got = self._impact_track() if self.controls.slowmo.isChecked() else None
        if got is not None:  # Bennu hits Earth: the last 1,000 km at 5 s/s, the 20 minutes after at 1 min/s
            imp = got[0]
            sm = imp.slow_motion(self.t)
            if sm is not None:
                t_new = self.t + dt * min(warp, sm[0])
            for b in (imp.t_hit - SHOW_BEFORE, imp.t_slowmo(), imp.t_hit):  # (a zone is never jumped over)
                if self.t < b < t_new:
                    t_new = b
                    break
        self.set_time(t_new, side=self._frame % SIDE_RENDER_EVERY == 0)

    def _warp(self):
        return SPEEDS[self.controls.speed.currentIndex()][1]

    def _refresh_craft_kind(self):
        mode = self.craft_mode.checkedId()
        if mode == 1:
            kind = "dart"
        elif mode == 2:
            kind = "tractor"
        else:
            s = self._last_snap if self.sim is not None else None
            ki = s.get("impactor") if s else None
            dart_active = ki is not None and ki["phase"] not in ("impacted", "missed")
            kind = "tractor" if (s and s.get("tractor_launched") and not dart_active) else "dart"
        self.craft.set_kind(kind)
        self.craft.update(self.t)
        _draw(self.p_craft)

    # ------------------------------------------------------------------ live run
    def _start_live(self, first, t_start):
        """A live run from t_start (in the chosen DART window, not the start of the timeline)."""
        self.log.write("action", what="live run starts", t_start=t_to_utc_str(t_start)[:16], first=list(first),
                       params=self.controls.params())
        self._seen_events, self._log_state = set(), None
        self.timer.stop()
        self._prox_phase = None
        self._impact_framed = 0
        self._results_seen = 0
        self.next_dart = self.next_tractor = None
        self._tr_ahead, self._tr_planned_t, self._tractor_window, self._tr_want_window = None, -np.inf, None, True
        self._tr_sent = False
        self._aftermath = None
        self.distance.set_tractor(None)
        self._next_choice = -1
        self._live_params = self.controls.params()
        self._live_dirty = False
        self._run_t_start = t_start
        launch = next((a[0] for n, *a in first if n == "launch_dart"), None)
        self._run_window = launch.opportunity if launch is not None else None
        self._run_plan = launch  # DART #1 as flown (kept by a re-fly that only changes tractor settings)
        self.live_panel.recommendation = None
        self.live_panel.initial_eta = None
        self.sim = self.engine
        self._last_snap = None
        self.live_plots.reset()
        self.prox_plots.reset()
        self.bplane.clear_live()
        self.distance.set_deflected(None, None)
        self.controls.set_stage("live", dart_ready=False, tractor_ready=False)
        self.info.setCurrentWidget(self.live_panel)
        self.controls.play.blockSignals(True)
        self.controls.play.setChecked(True)
        self.controls.play.setText("⏸ Pause")
        self.controls.play.blockSignals(False)
        self.status.showMessage(f"Live run from {t_to_utc_str(t_start)[:10]}: both spacecraft start in Earth orbit; "
                                "the DART flies its plan" +
                                ("" if self.engine.is_ready else " (simulation engine still loading…)"))
        self.set_time(t_start)
        first = [("slowmo", self.controls.slowmo.isChecked())] + list(first)
        self.engine.start(self.controls.params(), t_start, int(QtCore.QTime.currentTime().msec()),
                          self.controls.prox_ctrl.currentData(), self._warp(), first)

    def _plan_is_current(self):
        r = self.result
        return r is not None and _physical(r.params) == _physical(self.controls.params())

    def _on_launch_dart(self):
        self.log.write("action", what="launch DART pressed", live=self.sim is not None, next_dart=self.next_dart)
        if self.sim is None:
            if self._slider_date.isActive():  # the slider was just dragged: its date first (the plan follows it)
                self._slider_date.stop()
                self._slider_to_launch_date()
            if not self._plan_is_current():  # the selected window (or a parameter) is not planned yet
                self.log.write("action", what="launch waits for the plan of the selected window")
                self._launch_when_planned = True
                self.status.showMessage("Planning the selected DART window… the launch follows")
                self.plan_job.request()
                return
            r = self.result
            if r is None or not r.impactor.feasible:
                QtWidgets.QMessageBox.warning(self, "Cannot launch", "No feasible DART plan (see checks).")
                return
            ip = r.impactor
            if self.t < ip.raising.t_start - 3600.0:  # the window is ahead: the run starts shortly before it
                self._start_live([("launch_dart", ip)], max(self.t, ip.raising.t_start - LAUNCH_LEAD_DAYS * DAY))
                return
            # the time slider is past this window's orbit raising: plan the DART for a launch from there
            self._launch_from = self.t
            self.status.showMessage(f"Planning the DART for a launch from {t_to_utc_str(self.t)[:10]}…")
            self.launch_job.request()
            return
        if self.next_dart is None or not self.next_dart.feasible:
            self.log.write("action", what="next DART not ready: nothing launched")
            QtWidgets.QMessageBox.information(self, "Next DART", "The recommendation for the next DART is not ready.")
            return
        s = self._last_snap or {}
        if self.next_dart.raising.t_start < s.get("t", -np.inf):  # the clock moved past it: re-plan first
            self.status.showMessage("That opportunity has passed: re-planning the next DART…", 6000)
            self.next_dart = None
            self.rec_job.request()
            return
        self.sim.send("launch_dart", self.next_dart)
        self.next_dart = None
        self._next_choice = -1
        self.live_panel.recommendation = None
        self.controls.set_stage("live", dart_ready=False, tractor_ready=False)

    def _on_launch_plan(self, out):
        """Planning mode: the DART planned from the slider time (the selected window if still open)."""
        opps, ip = out
        t0 = self._launch_from
        if self.sim is not None or t0 is None:
            return
        if ip is not None and ip.feasible:
            self._start_live([("launch_dart", ip)], max(t0, ip.raising.t_start - LAUNCH_LEAD_DAYS * DAY))
            return
        ahead = [o for o in opps if o.feasible]
        p = self.controls.params()
        if not ahead:
            QtWidgets.QMessageBox.warning(self, "No DART window left",
                                          f"No DART launch window is left after {t_to_utc_str(t0)[:10]}.")
            return
        nxt = self.planner.recommended(ahead) if p.opportunity < 0 else ahead[0]
        names = "\n".join(f"  • {t_to_utc_str(o.t_pass)[:7]} pass: launch {t_to_utc_str(o.t_launch)[:10]}, "
                           f"~{o.mass_nominal:,.0f} kg" for o in ahead[:6])
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("That window has passed")
        box.setText(f"At {t_to_utc_str(t0)[:10]} (the time slider) the selected DART window is over.\n\n"
                    f"Windows still ahead:\n{names}\n\nLaunch in the {t_to_utc_str(nxt.t_pass)[:7]} window?")
        go = box.addButton("Launch there", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is go:
            self.controls.select_opportunity(nxt.index)  # re-plans; then the launch starts in that window
            self._launch_from = t0
            self.launch_job.request()

    def _on_launch_tractor(self):
        self.log.write("action", what="launch tractor pressed", live=self.sim is not None, plan=self.next_tractor)
        if self.sim is None or self.next_tractor is None or not self.next_tractor.feasible:
            self.log.write("action", what="tractor not launched", reason="no live run" if self.sim is None else
                           "no tractor plan yet" if self.next_tractor is None else "the tractor plan is not feasible")
            return
        tp = self.next_tractor
        self.sim.send("launch_tractor", tp, self.controls.params())
        self.distance.set_tractor(None, tp.path_t, tp.path, self.eph, live=True)  # launched: its flight, no window
        self.next_tractor = None
        self._tr_sent = True  # until the snapshot shows it launched: no re-plan, no second launch
        self.controls.set_stage("live", dart_ready=self.controls.launch_dart.isEnabled(), tractor_ready=False)
        self._update_tractor_status(self._last_snap or {})
        self.craft_mode.button(0).setChecked(True)

    def _on_release(self):
        """End of the tow on command: the tractor lets the boulder go and leaves Bennu (core/mission.py end_tow)."""
        self.log.write("action", what="release pressed", live=self.sim is not None)
        if self.sim is not None:
            self.sim.send("release")
            self.controls.set_release_ready(False)

    def _tr_args(self):
        """Arguments of the tractor plan job: the parameters (with the measured beta), the 2135 point reached so
        far, the launch date (now; before the verdict: the verdict date, so the plan is ready when it comes) and
        whether to sweep the launch window too (only when the 2135 point changed: it takes a few seconds)."""
        s = self._last_snap or {}
        p, b_now, t_now, _ = self._rec_args()
        t_launch = t_now
        ki = s.get("impactor") or {}
        if not s.get("dart_ok") and ki.get("impact") is not None and ki.get("phase") == "impacted":
            # planned ahead of the verdict: for its date, and for the 2135 point the team expects after this DART
            # (tracking has not measured its shift yet); the verdict then refreshes the window for the measured one
            t_launch = max(t_now, float(ki["impact"]["t"]) + GROUND_OD_DAYS * DAY)
            if ki.get("db_plan") is not None:
                b_now = b_now + np.asarray(ki["db_plan"], float)
        want = self._tr_want_window
        self._tr_want_window = False
        return p, b_now, t_launch, want

    def _rec_args(self):
        s = self._last_snap or {}
        p = self.controls.params()
        bg = s.get("beta_ground")
        if bg is not None:  # after an impact the team knows beta from tracking
            p = replace(p, beta=float(bg[0]), beta_sigma=float(max(bg[1], 0.1)))
        b_now = np.asarray((s.get("outlook") or {}).get("b_now", self.planner.b0), float)
        n = len(s.get("impactors") or []) + 1
        p = replace(p, opportunity=self._next_choice)  # the next DART's window (the controls hold DART #1's)
        p = self._with_refined(p, self._live_key(s))
        return p, b_now, float(s.get("t", self.t)), n

    @staticmethod
    def _live_key(s):
        n = len(s.get("impactors") or [])
        return ("live", n, round(float(s.get("t", 0.0)) / (30 * DAY)))

    def _on_recommendation(self, out):
        opps, ip, cap = out
        self.log.write("result", what="next-DART recommendation", plan=ip)
        self.next_dart = ip
        s = self._last_snap or {}
        pr, b_now, t_now, n = self._rec_args()
        if ip is not None and ip.feasible and pr.use_recommended_mass and ip.mass_source != "full model":
            prior = [(k["impact"]["t"], k["impact"]["dv_bennu"]) for k in (s.get("impactors") or []) if k.get("impact")]
            pr = replace(pr, refined_mass_kg=0.0)
            self._submit_refine(self._refine_key(pr, self._live_key(s)),
                                (pr, pr.opportunity, b_now, t_now + 30 * DAY, prior, n, s.get("spin_change")))
        self.live_panel.recommendation = (opps, ip)
        self.distance.set_opportunities(opps, ip.opportunity if ip is not None and ip.feasible else None,
                                        _window_summary(cap, ip, live=True))
        if ip is not None and ip.feasible:
            c = ip.contact or {}
            self.bennu.set_contact(c.get("point_b"))
        # (keep the tractor button as it is: this result must not disable a tractor that became ready meanwhile)
        self.controls.set_stage("live", dart_ready=ip is not None and ip.feasible, tractor_ready=self._tractor_ready(s))
        if self._last_snap is not None:
            self.live_panel.show_snapshot(self._last_snap)

    def _tractor_ready(self, s):
        """The tractor may launch now: a feasible plan, a successful DART, not launched, no re-simulation."""
        tp = self.next_tractor
        return bool(tp is not None and tp.feasible and s.get("dart_ok") and not s.get("tractor_launched")
                    and s.get("resim") is None and not self._tr_sent)

    def _on_tractor_plan(self, out):
        tp, msg, window = out
        s = self._last_snap or {}
        self.log.write("result", what="tractor plan", plan=tp, check=None if msg is None else msg.text,
                       tractor_launched=s.get("tractor_launched"), t_last=None if window is None else window.get("t_last"))
        if self._tr_sent or s.get("tractor_launched"):
            return  # (a refresh that finished after the launch: the tractor flies the plan it was launched with)
        self.next_tractor = tp
        self._tractor_check = None if msg is None else msg.text
        self._tr_planned_t = float(tp.t_launch)  # the launch date it was planned for (ahead: the verdict date)
        if window is not None:
            self._tractor_window = window
        self.distance.set_tractor(self._tractor_window, tp.path_t if tp.feasible else None,
                                  tp.path if tp.feasible else None, self.eph, live=True)
        self.controls.set_stage("live", dart_ready=self.controls.launch_dart.isEnabled(), tractor_ready=self._tractor_ready(s))
        if s.get("dart_ok"):
            self.status.showMessage(f"Gravity tractor ready to launch: arrival about {t_to_utc_str(self._tractor_arrival(tp))[:10]}"
                                    if tp.feasible else f"The gravity tractor cannot launch: {self._tractor_check or 'no plan'}",
                                    8000)
        self._update_tractor_status(s)

    @staticmethod
    def _tractor_arrival(tp):
        return tp.t_arrive_flight if np.isfinite(getattr(tp, "t_arrive_flight", np.nan)) else tp.t_arrive

    def _update_tractor_status(self, s):
        """The coloured strip above the tractor's launch button: why it can or cannot launch now."""
        colour, text = self._tractor_status(s)
        if (colour, text) != getattr(self.controls, "_tractor_status", None):
            self.controls.set_tractor_status(colour, text)
            self.log.write("hint", tractor=text, colour=colour)

    def _tractor_status(self, s):
        """(colour, text): red = not available and why; amber = waiting for the verdict; green = ready, with the
        launch window and the arrival for a launch now; grey = planning mode / launched."""
        if self.sim is None:
            return "grey", "Launches in a live run, after a DART succeeds."
        if not s:
            return "grey", "Starting the live run…"
        if s.get("resim") is not None:
            return "amber", "Flying the mission again with the new settings…"
        tr = s.get("tractor") or {}
        if self._tr_sent and not s.get("tractor_launched"):
            return "grey", "<b>Launch ordered:</b> the tractor leaves the parking orbit…"
        if s.get("tractor_launched"):
            pr = tr.get("prox")
            where = (f"at Bennu: {pr['phase']}" if pr else f"{tr.get('phase', '')}, arrival {t_to_utc_str(tr['eta'])[:10]}"
                     if np.isfinite(tr.get("eta", np.nan)) else tr.get("phase", ""))
            return "grey", f"<b>Launched</b> {t_to_utc_str(tr['t_launch'])[:10]} — {where}."
        results = (s.get("outlook") or {}).get("dart_results") or []
        paused = "" if s.get("running") else " The clock is paused: press ▶ Play."
        tp = self.next_tractor
        if s.get("dart_ok"):
            if tp is None:
                return "amber", "The DART succeeded: planning the tractor launch…"
            if not tp.feasible:
                return "red", f"<b>Not available</b> with these settings: {self._tractor_check or 'no feasible plan'}."
            w = self._tractor_window
            until = (f" Launch window open until <b>{t_to_utc_str(w['t_last'])[:10]}</b>."
                     if w is not None and np.isfinite(w.get("t_last", np.nan)) else
                     " No later launch closes the gap: launch now." if w is not None else "")
            return "green", (f"<b>READY:</b> launch now → arrival <b>{t_to_utc_str(self._tractor_arrival(tp))[:10]}</b>."
                             + until)
        ki = s.get("impactor") or {}
        if results and results[-1]["result"] != "success" and not s.get("dart_in_flight") \
                and len(results) >= len(s.get("impactors") or []):
            r = results[-1]
            return "red", (f"<b>Not available:</b> DART #{r['number']} {'missed' if r['result'] == 'missed' else 'fell short'} "
                           f"({100 * r['fraction']:.0f} % of the target; a success needs "
                           f"{100 * (1 - self.controls.params().dart_tolerance):.0f} %). Launch the next DART first.")
        if ki.get("phase") == "impacted" and ki.get("impact") is not None:
            due = ki["impact"]["t"] + GROUND_OD_DAYS * DAY
            ahead = (" The launch is planned for that date." if tp is not None and tp.feasible else "")
            return "amber", (f"<b>Waiting for the DART's verdict</b> (ground tracking), due "
                             f"<b>{t_to_utc_str(due)[:10]}</b>.{paused}{ahead}")
        if s.get("dart_in_flight"):
            due = ki.get("t_arrive", np.nan)
            return "red", ("<b>Not available</b> until the DART's verdict, 30 days after its impact"
                           + (f" ({t_to_utc_str(due + GROUND_OD_DAYS * DAY)[:10]})." if np.isfinite(due) else "."))
        return "red", "<b>Not available:</b> launch a DART first; the tractor follows a successful impact."


    @staticmethod
    def _awaiting_next_dart(s):
        res = (s.get("outlook") or {}).get("dart_results") or []
        return bool(res) and not s.get("dart_ok") and not s.get("dart_in_flight")

    TRACTOR_REFRESH_DAYS = 30.0  # the "launch now" tractor plan is re-made this often while the button is available

    def _tractor_planning(self, s, results):
        """Keep the tractor plan ahead of the user: planned at the DART impact for the verdict date (ready the instant
        the verdict says success), then re-planned for "launch now" every 30 mission days while the button is
        available, so pressing it later flies a plan for that date."""
        if s.get("tractor_launched") or self._tr_sent or self._tr_busy or s.get("resim") is not None:
            return
        ki = s.get("impactor") or {}
        t = float(s.get("t", self.t))
        if not s.get("dart_ok") and ki.get("phase") == "impacted" and ki.get("impact") is not None \
                and len(results) < len(s.get("impactors") or []):
            key = (ki.get("number"), round(float(ki["impact"]["t"])))
            if self._tr_ahead != key:  # plan ahead of the verdict, once per impact
                self._tr_ahead = key
                self._tr_want_window = True
                self.tr_job.request()
            return
        if s.get("dart_ok") and self.next_tractor is not None and t - self._tr_planned_t > self.TRACTOR_REFRESH_DAYS * DAY:
            self.tr_job.request()

    def _on_snap(self, s):
        if self.sim is None:  # late snapshot of a run that was just ended
            return
        rs = s.get("resim")
        self.log.t_sim = s.get("t")
        if rs is not None:  # the mission is being flown again with new parameters
            self._log_snap(s, [])
            what = rs.get("busy") or rs.get("background")
            self._update_tractor_status(s)
            self._set_recalc("Recalculating physics", f"Flying the mission again with the new parameters: "
                             f"{t_to_utc_str(rs['t'])[:10]} of {t_to_utc_str(rs['t1'])[:10]}"
                             + (f"\n{what}" if what else ""), rs["fraction"])
            return
        busy = s.get("busy")
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        if busy:
            self._busy_since = self._busy_since or now
            if now - self._busy_since > 0.4 and not self._live_dirty:  # (short waits would only flicker; a pending
                self._set_recalc("Recalculating physics", busy[0].upper() + busy[1:] + "…", None)  # change says more)
        elif self._busy_since is not None or self.center.currentIndex() == 1:
            self._busy_since = None
            if not self._live_dirty:
                self._set_recalc(None)
        self._last_snap = s
        if not s["running"] and self.controls.play.isChecked():  # the simulation paused itself
            self.controls.play.blockSignals(True)
            self.controls.play.setChecked(False)
            self.controls.play.setText("▶ Play")
            self.controls.play.blockSignals(False)
        if s.get("spin_change") is not None:  # keep this process's Bennu rotation in step (point-of-contact plans)
            from core.dynamics.asteroid import set_spin_change

            set_spin_change(*s["spin_change"])
        results = (s.get("outlook") or {}).get("dart_results") or []
        self._log_snap(s, results)
        if s.get("complete") and self._aftermath is None:  # the run ended with Bennu hitting Earth: its aftermath
            t_hit = (s.get("outlook") or {}).get("t_earth_impact", np.nan)
            if t_hit is not None and np.isfinite(t_hit) and s["t"] >= t_hit - 1.0:
                self.log.write("action", what="the run ended with the Earth impact: the impact page plays the aftermath")
                self._start_aftermath(t_hit)
        self._tractor_planning(s, results)
        self._update_tractor_status(s)
        pr = (s.get("tractor") or {}).get("prox")
        self.controls.set_release_ready(pr is not None and pr["phase"] in ("station", "parked")
                                        and not s.get("tow_complete"))
        if len(results) > self._results_seen:
            self._results_seen = len(results)
            last = results[-1]
            self.log.write("result", what="DART verdict", number=last.get("number"), result=last["result"],
                           fraction=last.get("fraction"), flyby_ld=last.get("q", 0.0) / 384_400e3)
            if last["result"] == "success":
                if self._tractor_ready(s):  # planned ahead of the verdict: the button is ready at once
                    self.controls.set_stage("live", dart_ready=self.controls.launch_dart.isEnabled(), tractor_ready=True)
                    self.status.showMessage("DART succeeded: the gravity tractor can launch now", 6000)
                else:
                    self.status.showMessage("DART succeeded: planning the gravity tractor launch…", 6000)
                self._tr_want_window = True  # the 2135 point is now measured by tracking: plan and window for it
                self.tr_job.request()
            else:
                self.status.showMessage("DART fell short: computing the recommendation for the next DART…", 6000)
                self._next_choice = -1  # the recommended window (click a marker on the distance plot to pick)
                self.rec_job.request()
        if s.get("verified_curve") is not None:
            t_c, d_c = s["verified_curve"]
            self.distance.set_deflected(t_c, d_c)
        self._frame += 1
        f = self._frame
        side = f % SIDE_RENDER_EVERY == 0 or self._close_up(s)
        sm = s.get("slowmo")
        banner = f"SLOW MOTION ({_short_warp(sm['warp'])}): {sm['what']}" if sm else None
        if banner != getattr(self, "_banner", None):
            self._banner = banner
            self.bennu.set_banner(banner)
        self.solar.set_live(s)
        self.craft.set_live(s)
        self.earth_moon.set_live(s)
        if side:
            self.bennu.set_live(s)
            self._refresh_craft_kind_auto(s)
        self._follow_phase(s)
        self.set_time(s["t"], render=False, side=side)
        self._render_all(side=side)
        # panels and plots: ~4 Hz when on screen, ~1 Hz when hidden (they keep their data up to date)
        state = (bool(s.get("busy")), s["running"], (s.get("tractor") or {}).get("phase"),
                 (s.get("impactor") or {}).get("phase"), len((s.get("outlook") or {}).get("dart_results") or []))
        if self.info.currentWidget() is self.live_panel and (f % 5 == 0 or state != self._was_busy):
            self.live_panel.show_snapshot(s)  # at once when something changes (pause, phase, verdict)
        self._was_busy = state
        bottom = self.bottom.currentWidget()
        for plot, kind in ((self.live_plots, "add"), (self.prox_plots, "add"), (self.bplane, "bplane")):
            if f % (5 if bottom is plot else 20) == 0:
                plot.update_live(s["outlook"]) if kind == "bplane" else plot.add_snapshot(s)

    def _log_snap(self, s, results):
        """Session log: the mission's new events, and a state summary when it changes (every 5 s otherwise)."""
        for e in s.get("events") or []:
            key = (round(e.t), e.who, e.text)
            if key not in self._seen_events:
                self._seen_events.add(key)
                self.log.write("event", at=t_to_utc_str(e.t)[:16], who=e.who, level=e.level, text=e.text)
        ki, tr = s.get("impactor") or {}, s.get("tractor") or {}
        state = (s.get("running"), bool(s.get("busy")), ki.get("phase"), tr.get("phase"), bool(s.get("dart_ok")),
                 bool(s.get("tractor_launched")), len(results), s.get("resim") is not None,
                 self.controls.launch_dart.isEnabled(), self.controls.launch_tractor.isEnabled())
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1e3
        if state != self._log_state or abs(now - self._log_wall) > 5.0:
            self._log_state, self._log_wall = state, now
            self.log.write("state", running=s.get("running"), busy=s.get("busy"), dart=ki.get("phase"),
                           tractor=tr.get("phase"), tractor_launched=s.get("tractor_launched"),
                           dart_ok=s.get("dart_ok"), dart_in_flight=s.get("dart_in_flight"), verdicts=len(results),
                           resim=(s.get("resim") or {}).get("fraction"),
                           launch_dart_enabled=self.controls.launch_dart.isEnabled(),
                           launch_tractor_enabled=self.controls.launch_tractor.isEnabled(),
                           next_dart=self.next_dart is not None, next_tractor=self.next_tractor is not None,
                           live_dirty=self._live_dirty)

    @staticmethod
    def _close_up(s):
        """The DART's last approach and the ejecta after its impact: Bennu and craft views at the full rate."""
        ki = s.get("impactor") or {}
        return (ki.get("phase") == "terminal" or (s.get("ejecta") or {}).get("pos") is not None
                or s.get("slowmo") is not None)

    def _refresh_craft_kind_auto(self, s):
        if self.craft_mode.checkedId() != 0:
            return
        ki = s.get("impactor")
        dart_active = ki is not None and ki["phase"] not in ("impacted", "missed")
        kind = "tractor" if (s.get("tractor_launched") and not dart_active) else "dart"
        if kind != self.craft.kind:
            self.craft.set_kind(kind)

    def _follow_phase(self, s):
        """Frame Bennu on the impact and when the tractor arrives."""
        ej = s.get("ejecta")
        n = len(s.get("impactors") or [])
        ki = s.get("impactor") or {}
        sm = (s.get("slowmo") or {}).get("what") or ""
        if sm.startswith("DART") and ki.get("phase") == "terminal" and self._impact_framed < n:
            # the close-up begins: look side-on at where the DART will hit (its aim point, or the side facing it)
            self._impact_framed = n
            c = (ki.get("contact") or {}).get("point_b")
            rel = ki.get("rel_bennu")
            if c is not None:
                site = np.asarray(s["bennu_R"]) @ np.asarray(c, float)
            elif rel is not None:
                site = 245.0 * np.asarray(rel, float) / np.linalg.norm(rel)
            else:
                site = None
            if site is not None:
                self.bennu.frame_impact(site)
        if ej is not None and ej.get("site_b") is not None and self._impact_framed < n:
            self._impact_framed = n
            self.bennu.frame_impact(np.asarray(s["bennu_R"]) @ np.asarray(ej["site_b"]))
            self.bottom.setCurrentWidget(self.distance)
        pr = (s.get("tractor") or {}).get("prox")
        phase = pr["phase"] if pr else None
        if phase != self._prox_phase:
            old, self._prox_phase = self._prox_phase, phase
            if old is None and phase is not None:
                self.bottom.setCurrentWidget(self.prox_plots)
                self.bennu.live = s
                self.bennu._update_live()
                self.bennu.reset_camera()
            elif phase in ("survey", "transfer", "station", "release", "depart"):
                self.bennu.reset_camera()
            elif phase == "departed":  # the tractor has left: back to Bennu itself
                self.bennu._sc_pos = None
                self.bennu.reset_camera()

    def _end_live(self):
        self.log.write("action", what="live run ended")
        if self.sim is not None:
            self.sim.stop()
            self.sim = None
        from core.dynamics.asteroid import reset_spin

        reset_spin()
        for scene in (self.solar, self.bennu, self.craft, self.earth_moon):
            scene.clear_live()
        self.bplane.clear_live()
        self.distance.set_deflected(None, None)
        self.next_dart = self.next_tractor = None
        self._live_dirty = False
        self._set_recalc(None)
        self._banner = None
        self.bennu.set_banner(None)
        self.controls.set_stage("plan")
        self._tr_ahead, self._tr_planned_t, self._tractor_window = None, -np.inf, None
        self._tr_sent = False
        self._aftermath = None
        self._win_key = None
        self.distance.set_tractor(None)
        self._update_tractor_status(None)
        self.controls.play.setChecked(False)
        self.info.setCurrentWidget(self.mission)
        self.status.showMessage("Planning mode: the views preview the plan (no physics is run). "
                                "🚀 Launch DART starts the physics simulation.")
        self._last_snap = None
        self.plan_job.request()
        self.set_time(self.t)

    def shutdown(self):
        """Stop timers, workers and the simulation, and release the VTK views while Qt is still alive (also on
        app.quit(), which skips closeEvent: VTK objects outliving their GL context can crash the exit)."""
        if getattr(self, "_shut", False):
            return
        self._shut = True
        self.log.close()
        self.timer.stop()
        self._ref_timer.stop()
        self.compute.shutdown()
        if self._refiner is not None:
            from core.parallel import kill_pool
            from core.refine import refine_close

            with self._ref_gen.get_lock():
                self._ref_gen.value += 1  # a refinement in progress stops at once
            self._refiner.submit(refine_close)
            kill_pool(self._refiner, grace=2.0)
        if self.sim is not None:
            self.sim.stop()
        self.engine.shutdown()
        for p in (self.p_solar, self.p_bennu, self.p_craft, self.p_em):
            p.close()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def _window_summary(cap, ip, live=False):
    """The window markers' summary: the DART as set (planner summary), whether it is the next DART of a live run,
    and how the plan sized its mass."""
    if cap is None:
        return None
    return {**cap, "live": live, "mass_source": ip.mass_source if ip is not None and ip.feasible else None}


def _short_warp(w):
    return "real time" if w <= 1.0 else (f"{w:g} s per second" if w < 60 else f"{w / 60:g} min per second")


def _physical(p):
    """The parameters that change the physics (with the recommended mass on, the mass slider only displays it)."""
    return physical_params(p)


class RecalcPage(QtWidgets.QWidget):
    """Shown instead of the 3-D views while the physics catches up: what it computes and how far it is."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(2)
        self.title = QtWidgets.QLabel("Recalculating physics")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 26px; font-weight: 700; color: #e8e6dc")
        self.detail = QtWidgets.QLabel("")
        self.detail.setAlignment(QtCore.Qt.AlignCenter)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("font-size: 15px; color: #c3c2b7")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(420)
        note = QtWidgets.QLabel("Nothing is replayed: every frame is computed by the physics (N-body orbits, guidance,\n"
                                "impact, Bennu's full-force-model orbit). The views come back when it has caught up.")
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setStyleSheet("font-size: 12px; color: #8f8d84")
        lay.addWidget(self.title)
        lay.addSpacing(10)
        lay.addWidget(self.detail)
        lay.addSpacing(14)
        lay.addWidget(self.bar, 0, QtCore.Qt.AlignHCenter)
        lay.addSpacing(18)
        lay.addWidget(note)
        lay.addStretch(3)
        self.setObjectName("recalc")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet("#recalc { background: #0b0d12; } QLabel { background: transparent; }")

    def show_state(self, title, detail, fraction):
        self.title.setText(title)
        self.detail.setText(detail)
        if fraction is None:
            self.bar.setRange(0, 0)  # busy indicator
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(1000 * fraction))


def _dark_palette(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, color in [(QtGui.QPalette.Window, "#1b1f24"), (QtGui.QPalette.WindowText, "#e6e6e6"),
                        (QtGui.QPalette.Base, "#14171b"), (QtGui.QPalette.Text, "#e6e6e6"),
                        (QtGui.QPalette.Button, "#2a3038"), (QtGui.QPalette.ButtonText, "#e6e6e6"),
                        (QtGui.QPalette.Highlight, "#00a3c4"), (QtGui.QPalette.ToolTipBase, "#2a3038"),
                        (QtGui.QPalette.ToolTipText, "#e6e6e6")]:
        pal.setColor(role, QtGui.QColor(color))
    app.setPalette(pal)


def _shot(win, name):
    os.makedirs("outs/snapshots", exist_ok=True)
    win.grab().save(f"outs/snapshots/{name}.png")


def _smoke_live(win, app, mode):
    """Automated live runs: 'live' (DART, impact, tractor launch), 'prox' (through the tow and the verification),
    'fail' (an undersized DART falls short, the recommendation launches DART #2)."""
    st = {"i": 0, "launched": False, "shots": set(), "t0": None, "tractor": False, "dart2": False, "tractor_wait": 0}

    def press(button):
        """Click a launch button as a user does (a disabled button ignores the click, as on screen)."""
        from PySide6.QtTest import QTest

        if not button.isEnabled():
            return False
        QTest.mouseClick(button, QtCore.Qt.LeftButton)
        return True

    def step():
        st["i"] += 1
        i = st["i"]
        s = win._last_snap
        if i == 1 and mode == "fail":
            win.controls.use_rec.setChecked(False)
        if i >= 2 and mode == "fail" and win.result is not None and not st.get("mass_set"):
            st["mass_set"] = True
            st["mass_plan"] = 0.3 * win.result.impactor.mass_nominal
            win.controls.imp_mass.set_value(st["mass_plan"])
            win.controls.paramsChanged.emit()
            return
        if mode == "fail" and not st["launched"] and win.result is not None and st.get("mass_set") \
                and abs(win.result.impactor.mass - st["mass_plan"]) > 0.02 * st["mass_plan"]:
            return  # wait for the re-plan with the undersized impactor
        if i >= 6 and not st["launched"]:
            if win.result is None or not win.result.impactor.feasible:
                return
            win.controls.speed.setCurrentIndex(6)
            st["launched"] = press(win.controls.launch_dart)
            return
        if s is None:
            return
        ki = s.get("impactor") or {}
        ph = ki.get("phase")
        if s.get("running") and not s.get("busy") and win.sim is not None and i % 4 == 0 and ph in ("parking", "cruise") \
                and not (mode == "prox" and s.get("tractor_launched")):
            win.sim.send("skip")
        for tag, cond in [("raising", ph == "raising"), ("terminal", ph == "terminal"),
                          ("impact", s.get("ejecta") is not None and ph == "impacted"),
                          ("tractor_cruise", (s.get("tractor") or {}).get("phase") == "cruise"),
                          ("tractor_prox", (s.get("tractor") or {}).get("prox") is not None),
                          ("recommendation", win.next_dart is not None)]:
            if cond and tag not in st["shots"]:
                st["shots"].add(tag)
                _shot(win, f"app_{mode}_{tag}")
        if mode in ("live", "prox") and s.get("dart_ok") and not st["tractor"]:
            st["tractor"] = press(win.controls.launch_tractor)
            st["tractor_wait"] += 0 if st["tractor"] else 1
            if st["tractor_wait"] == 30:
                print("smoke: the Launch gravity tractor button stayed disabled for 30 s after the DART succeeded",
                      flush=True)
        if mode == "fail" and win.next_dart is not None and not st["dart2"]:
            _shot(win, "app_fail_recommendation")
            st["dart2"] = press(win.controls.launch_dart)
        elif mode in ("live", "prox") and win.next_dart is not None and not s.get("dart_in_flight") \
                and not s.get("dart_ok"):
            press(win.controls.launch_dart)  # a natural shortfall (low beta): fly the recommended next DART, go on
        done = {"live": "tractor_cruise" in st["shots"],
                "prox": (s["outlook"].get("final_check") or {}).get("q") is not None and s.get("complete"),
                "fail": st["dart2"] and len(s.get("impactors") or []) >= 2 and s["impactors"][-1]["phase"] == "impacted"}[mode]
        if mode == "prox" and s.get("tractor_launched") and (s.get("tractor") or {}).get("prox") is not None:
            pr = s["tractor"]["prox"]
            tag = f"prox_{pr['phase']}"
            if tag not in st["shots"]:
                st["shots"].add(tag)
                _shot(win, f"app_prox_{pr['phase']}")
            if pr["phase"] == "station" and i % 10 == 0:
                win.sim.send("finish")
        if done or i > 2400:
            _shot(win, f"app_{mode}_end")
            win.bottom.setCurrentWidget(win.bplane)
            app.processEvents()
            _shot(win, f"app_{mode}_bplane")
            win._end_live()
            app.quit()

    timer = QtCore.QTimer(win)
    timer.timeout.connect(step)
    timer.start(1000)
    return timer


def _smoke_windows(win, app, heavier=2000.0):
    """The DART window markers: screenshots with the markers shown and hidden (hiding must not change the selection
    or the plan), then with a manual mass, the recommended one first and then `heavier` kg; prints each window's
    class and the selected window's tooltip."""
    st = {"i": 0, "stage": "wait"}
    win.bottom.setCurrentWidget(win.distance)

    def planned(mass=None):
        r = win.result
        return (r is not None and r.capability is not None and win._plan_is_current()
                and (mass is None or abs(r.capability["mass"] - mass) <= 1e-9 * mass))

    def report(tag):
        r = win.result
        s = r.capability
        print(f"[{tag}] {s['mass']:,.0f} kg, max speed {s['v_max']:.2f} km/s, β {s['beta_eval']:.2f}: " + ", ".join(
            f"{t_to_utc_str(o.t_pass)[:7]} {o.capability['status']}" for o in r.opportunities if o.capability), flush=True)
        print(f"[{tag}] tooltip of the selected window:\n" + win.distance._tips.get(r.impactor.opportunity, ""), flush=True)

    def shots(tag):
        app.processEvents()
        _shot(win, f"app_windows_{tag}")
        top = win._main_split.widget(0)  # the plot alone, taller: the bottom tabs take the whole window
        top.hide()
        app.processEvents()
        win.bottom.grab().save(f"outs/snapshots/app_windows_{tag}_plot.png")
        top.show()
        app.processEvents()

    def manual_mass(m):
        win.controls.imp_mass.set_value(m)
        win.controls.paramsChanged.emit()
        return win.controls.imp_mass.value()

    def step():
        st["i"] += 1
        if st["stage"] == "wait" and st["i"] >= 3 and planned():
            report("recommended")
            shots("shown")
            before = (win.controls.opportunity.currentData(), win.result, win.plan_job.latest)
            win.show_windows.setChecked(False)
            shots("hidden")
            d = win.distance
            hidden = not (d.opps.isVisible() or d.sel.isVisible() or d.legend_shown()
                          or "selected" in d.getPlotItem().titleLabel.text)
            same = before == (win.controls.opportunity.currentData(), win.result, win.plan_job.latest)
            print(f"hidden: markers, legend and label hidden {hidden}; selection and plan unchanged {same}", flush=True)
            win.show_windows.setChecked(True)
            win.controls.use_rec.setChecked(False)
            st["m"] = manual_mass(win.result.impactor.mass)
            st["stage"] = "before"
        elif st["stage"] == "before" and planned(st["m"]):
            report("mass_before")
            shots("mass_before")
            st["m"] = manual_mass(heavier)
            st["stage"] = "after"
        elif st["stage"] == "after" and planned(st["m"]):
            report("mass_after")
            shots("mass_after")
            app.quit()
        elif st["i"] > 200:
            print("smoke-windows: timed out", flush=True)
            app.quit()

    timer = QtCore.QTimer(win)
    timer.timeout.connect(step)
    timer.start(1000)
    return timer


def _smoke_impact(win, app):
    """Planning mode: 'Jump to… Earth impact 2182', Play at 1 day/s with slow motion on: the centre switches to the
    impact page, slows down for the last 1,000 km and the 20 minutes after, then returns to the Solar System."""
    from PySide6.QtTest import QTest

    st = {"i": 0, "shots": set(), "pages": []}
    win.controls.constellations.setChecked(True)

    def step():
        st["i"] += 1
        c = win.controls
        if st["i"] == 3:
            k = next(j for j in range(c.jump.count()) if c.jump.itemText(j).startswith("Earth impact"))
            c.jump.setCurrentIndex(k)
            c.jump.activated.emit(k)
            c.speed.setCurrentIndex(3)  # 1 day / s
            QTest.mouseClick(c.play, QtCore.Qt.LeftButton)
            return
        if st["i"] < 3:
            return
        page = win.center.currentIndex()
        if not st["pages"] or st["pages"][-1] != page:
            st["pages"].append(page)
        imp = win._impact_track()[0]
        since = win.t - imp.t_hit
        for tag, cond in [("approach", page == 2 and since < -3600), ("slowmo", page == 2 and -80 < since < 0),
                          ("plume", page == 2 and 120 < since < SHOW_AFTER), ("after", page == 0 and since > SHOW_AFTER)]:
            if cond and tag not in st["shots"]:
                st["shots"].add(tag)
                _shot(win, f"app_impact_{tag}")
        if "after" in st["shots"] or st["i"] > 240:
            print(f"smoke-impact: pages {st['pages']}, shots {sorted(st['shots'])}", flush=True)
            app.quit()

    timer = QtCore.QTimer(win)
    timer.timeout.connect(step)
    timer.start(250)
    return timer


def _core_budget(argv):
    """--cores N: the budget every process of the app sizes its threads and workers from. It travels to the
    worker processes in BENNU_CORES, so it is set before any of them starts."""
    from core import parallel

    if "--cores" in argv:
        i = argv.index("--cores")
        n = int(argv[i + 1]) if i + 1 < len(argv) else parallel.CORES
        del argv[i:i + 2]
        parallel.CORES = max(1, min(n, os.cpu_count() or n))
    os.environ["BENNU_CORES"] = str(parallel.CORES)
    parallel.configure(parallel.budget()["gui_threads"])
    parallel.set_qos("user_interactive")  # the window's thread: macOS keeps it on a performance core
    return parallel.CORES, parallel.describe()


def main():
    n_cores, cores = _core_budget(sys.argv)
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    _dark_palette(app)
    win = MainWindow()
    win.log.install(app)
    app.aboutToQuit.connect(win.shutdown)
    win.show()
    label = QtWidgets.QLabel(f"{n_cores} cores")
    label.setToolTip(cores)
    win.status.addPermanentWidget(label)
    win.status.showMessage(cores, 15000)
    print(cores, flush=True)
    print(f"Session log: {win.log.path}", flush=True)
    log_label = QtWidgets.QLabel("log")
    log_label.setToolTip(f"Session log (clicks and how the app responded):\n{win.log.path}")
    win.status.addPermanentWidget(log_label)
    keep = None
    for flag, mode in [("--smoke-live", "live"), ("--smoke-prox", "prox"), ("--smoke-fail", "fail")]:
        if flag in sys.argv:
            keep = _smoke_live(win, app, mode)
    if "--smoke-windows" in sys.argv:
        keep = _smoke_windows(win, app)
    if "--smoke-impact" in sys.argv:
        keep = _smoke_impact(win, app)
    if "--smoke" in sys.argv:  # plan, switch tabs, save screenshots, wait for the full-model mass, exit
        def step(i=[0]):
            i[0] += 1
            if i[0] == 8:
                win.set_time(win.result.impactor.raising.t_start + 3600.0 if win.result and win.result.impactor.feasible
                             else win.t)
            elif i[0] == 10:
                _shot(win, "app_plan")
                for k in range(win.bottom.count()):
                    win.bottom.setCurrentIndex(k)
                    app.processEvents()
                    win._render_em()
                    win.bottom.currentWidget().grab().save(f"outs/snapshots/app_plan_tab{k}.png")
                win.bottom.setCurrentIndex(0)
                win.home.setChecked(True)  # the Solar System's side view in the Milky Way, true speed and slowed
                app.processEvents()
                win.center.grab().save("outs/snapshots/app_solar_side.png")
                print(win.status.currentMessage())
                win.controls.galactic_slow.setChecked(True)
                app.processEvents()
                win.center.grab().save("outs/snapshots/app_solar_side_slowed.png")
                win.controls.galactic_slow.setChecked(False)
                win.home.setChecked(False)
            elif i[0] > 10:
                r = win.result
                if (r is not None and r.impactor.feasible and r.impactor.mass_source == "full model") or i[0] > 150:
                    win.info.setCurrentWidget(win.mission)
                    app.processEvents()
                    _shot(win, "app_plan_refined")
                    print("refined:", r.impactor.mass_source if r else None,
                          f"{r.impactor.mass:,.0f} kg (linear {r.impactor.mass_linear:,.0f} kg)" if r else "")
                    app.quit()
        keep = QtCore.QTimer()
        keep.timeout.connect(step)
        keep.start(1500)
    rc = app.exec()
    if keep is not None:
        keep.stop()
    win.shutdown()
    win.deleteLater()  # let Qt delete the widgets now, before Python's own module teardown
    app.processEvents()
    sys.exit(rc)


if __name__ == "__main__":
    main()
