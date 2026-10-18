"""Side panels: controls (display, time, goal, DART, tractor, live run), mission plan, live flight, Bennu now."""

from dataclasses import asdict

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.widgets import ParamSlider, section
from core.constants import AU, BENNU, DAY, LUNAR_DISTANCE, YEAR
from core.dart import departure_speed
from core.planner import MissionParams
from core.timeutil import t_to_utc_str

START_DATE = "2026-10-15 00:00"  # simulation "today"
SPEEDS = [("1 minute / s", 60.0), ("10 minutes / s", 600.0), ("1 hour / s", 3600.0), ("1 day / s", DAY),
          ("1 week / s", 7 * DAY), ("1 month / s", 30.44 * DAY), ("1 year / s", 365.25 * DAY),
          ("5 years / s", 5 * 365.25 * DAY)]
GOOD, CRITICAL, WARN = "#0ca30c", "#d03b3b", "#fab219"
R_PARK = 6378.1366e3 + 400e3


def _fmt_date(t):
    return t_to_utc_str(t)[:10] if t is not None and np.isfinite(t) else "—"


def _rows(rows):
    return ["<table>"] + [f"<tr><td>{a}&nbsp;&nbsp;</td><td align='right'>{b}</td></tr>" for a, b in rows] + ["</table>"]


def _mass(m):
    if not np.isfinite(m):
        return "—"
    return f"{m / 1e3:,.1f} t" if m >= 1e4 else f"{m:,.0f} kg"


class ControlsPanel(QtWidgets.QScrollArea):
    """Left panel. Emits paramsChanged whenever a mission parameter is edited."""

    paramsChanged = QtCore.Signal()

    def __init__(self, t_lo, t_hi):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.t_lo, self.t_hi = t_lo, t_hi
        self.on_stage = None  # called with every launch-button state change (the session log)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        self.setWidget(inner)
        d = MissionParams()

        box, tl = section("Time (UTC)")
        self.date_label = QtWidgets.QLabel()
        self.date_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        tl.addWidget(self.date_label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, int((t_hi - t_lo) / DAY))
        tl.addWidget(self.slider)
        row = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.speed = QtWidgets.QComboBox()
        for label, _ in SPEEDS:
            self.speed.addItem(label)
        self.speed.setCurrentIndex(6)
        row.addWidget(self.play)
        row.addWidget(self.speed)
        tl.addLayout(row)
        self.slowmo = QtWidgets.QCheckBox("Slow motion at Bennu (DART impact, boulder pickup)")
        self.slowmo.setChecked(True)
        self.slowmo.setToolTip("Live run: the clock slows down for the DART's last minutes and impact and for the\n"
                               "tractor's descent, boulder capture and lift-off, then returns to your speed.\n"
                               "Only the display speed changes: the physics is the same at any speed.")
        tl.addWidget(self.slowmo)
        self.jump = QtWidgets.QComboBox()
        self.jump.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.jump.setMinimumContentsLength(20)
        self.jump.addItem("Jump to…")
        tl.addWidget(self.jump)
        lay.addWidget(box)

        box, gl = section("Mission goal")
        self.goal = ParamSlider("2135 flyby at least", 1.0, 2.0, d.goal_ld, "LD", decimals=2,
                                tooltip="Lunar distances (1 LD = 384,400 km): beyond the Moon, Bennu misses the keyhole")
        self.target = ParamSlider("DART target shift", 50_000, 2_000_000, 320_000, "km", log=True,
                                  tooltip="How far the DART(s) should move the 2135 flyby; the gravity tractor then\n"
                                          "steers it to exactly that point (the aim point)")
        self.goal_info = QtWidgets.QLabel()
        self.goal_info.setWordWrap(True)
        self.goal_info.setStyleSheet("color: #c3c2b7;")
        for w in (self.goal, self.target, self.goal_info):
            gl.addWidget(w)
        lay.addWidget(box)

        box, il = section("DART (kinetic impactor)")
        self.opportunity = QtWidgets.QComboBox()
        self.opportunity.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.opportunity.setMinimumContentsLength(20)
        self.opportunity.setToolTip("Launch opportunities at Bennu's close passes to Earth (markers on the\n"
                                    "Earth–Bennu distance plot; click a marker to select it)")
        il.addWidget(self.opportunity)
        # the earliest launch: the plan (porkchop) only uses launches from this date on (+40 days of orbit raising);
        # two-way bound to the time slider (main.py: the slider dragged by hand sets it, a new date moves the clock)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(QtWidgets.QLabel("Launch no earlier than"))
        self.launch_after = QtWidgets.QDateEdit()
        self.launch_after.setDisplayFormat("yyyy-MM-dd")
        self.launch_after.setCalendarPopup(True)
        lo_date = QtCore.QDate.fromString(t_to_utc_str(t_lo)[:10], "yyyy-MM-dd")
        self.launch_after.setDateRange(lo_date, QtCore.QDate.fromString(t_to_utc_str(t_hi)[:10], "yyyy-MM-dd"))
        self.launch_after.setDate(QtCore.QDate.fromString(d.earliest_launch, "yyyy-MM-dd"))
        self.launch_after.setToolTip("The DART is planned for launches from this date on (its orbit raising starts\n"
                                     "here at the earliest): the best arc of the selected window after it, or the\n"
                                     "recommended window still open. Setting it moves the clock to that date.")
        drow.addWidget(self.launch_after, 1)
        il.addLayout(drow)
        self.follow_slider = QtWidgets.QCheckBox("follow the time slider")
        self.follow_slider.setChecked(True)
        self.follow_slider.setToolTip("Dragging the time slider by hand sets the earliest launch to the slider's date\n"
                                      "(Play and Jump to… do not change it). Off: the date stays where you set it.")
        il.addWidget(self.follow_slider)
        mrow = QtWidgets.QHBoxLayout()
        self.use_rec = QtWidgets.QCheckBox("Use recommended mass")
        self.use_rec.setChecked(True)
        self.sizing = QtWidgets.QComboBox()
        self.sizing.addItem("for the planned β", "nominal")
        self.sizing.addItem("for 90 % confidence", "p90")
        mrow.addWidget(self.use_rec)
        mrow.addWidget(self.sizing, 1)
        il.addLayout(mrow)
        self.imp_mass = ParamSlider("Mass at impact", 20, 5_000_000, 600, "kg", log=True,
                                    tooltip="Assembled in Earth orbit; used when the recommended mass is off")
        self.vmax = ParamSlider("Orbit-raising cut-off speed (max velocity)", 11.0, 16.5, d.dart_max_speed_kms, "km/s",
                                decimals=2, tooltip="The DART raises its orbit with burns at perigee and leaves Earth when\n"
                                                    "its perigee speed reaches the planned departure speed, at most this")
        self.vmax_info = QtWidgets.QLabel()
        self.vmax_info.setStyleSheet("color: #c3c2b7;")
        self.beta = ParamSlider("β (ejecta momentum boost)", 1.0, 6.0, d.beta, "", decimals=2,
                                tooltip="DART measured 3.61 (+0.19/-0.25); Bennu's value is unknown")
        self.beta_sigma = ParamSlider("β uncertainty (1σ)", 0.0, 1.5, d.beta_sigma, "", decimals=2)
        self.launch_dart = QtWidgets.QPushButton("🚀  Launch DART")
        self.launch_dart.setStyleSheet("font-weight: 700; padding: 6px;")
        for w in (self.imp_mass, self.vmax, self.vmax_info, self.beta, self.beta_sigma, self.launch_dart):
            il.addWidget(w)
        lay.addWidget(box)

        box, trl = section("Gravity tractor (enhanced, SEP)")
        self.thrust = ParamSlider("Thrust", 0.3, 3000, d.tractor_thrust_n, "N", log=True,
                                  tooltip="10 t spacecraft; unlimited fuel, consumption tracked. In flight: the throttle")
        self.objective = QtWidgets.QComboBox()
        self.objective.addItem("Fastest arrival", "fastest")
        self.objective.addItem("Most fuel-efficient (≤ 6 yr)", "efficient")
        self.boulder = ParamSlider("Boulder collected", 0, 60, d.boulder_mass_kg / 1e3, "t", decimals=0,
                                   tooltip="Enhanced gravity tractor: mass picked up from Bennu's surface")
        self.tow = ParamSlider("Tow duration", 0, 25, d.tow_years, "yr", decimals=1)
        self.hover = ParamSlider("Hover distance", 1.5, 4.0, d.hover_factor, "× radius", decimals=2)
        self.launch_tractor = QtWidgets.QPushButton("🚀  Launch gravity tractor")
        self.launch_tractor.setStyleSheet("font-weight: 700; padding: 6px;")
        self.launch_tractor.setToolTip("Available once a DART impact has succeeded (shown after ground tracking)")
        # the status strip above the button: red (not available: why), amber (waiting for the verdict),
        # green (ready: launch window and arrival) -- set_tractor_status()
        self.tractor_status = QtWidgets.QLabel()
        self.tractor_status.setWordWrap(True)
        self.tractor_status.setTextFormat(QtCore.Qt.RichText)
        self.tractor_status.setMargin(5)
        self.tractor_status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.set_tractor_status("grey", "Launches in a live run, after a DART succeeds.")
        self.release = QtWidgets.QPushButton("⤒  Release the boulder and end the tow")
        self.release.setToolTip("End of the tow: the tractor lets the boulder go where it is, climbs to 50 km and leaves\n"
                                "Bennu; Bennu is then flown on with the full force model. Available while the tractor\n"
                                "holds its station (towing or parked). The tow also ends by itself when its time is up\n"
                                "or once the trim has held for 60 days.")
        for w in (self.thrust, self.objective, self.boulder, self.tow, self.hover, self.tractor_status,
                  self.launch_tractor, self.release):
            trl.addWidget(w)
        lay.addWidget(box)

        box, ll = section("Live run")
        self.storm_days = ParamSlider("Storm safe-mode duration", 5, 180, 40, "days", decimals=0)
        self.storm = QtWidgets.QPushButton("☀  Solar storm!")
        self.storm.setToolTip("Thrusters off (safe mode) for the chosen duration; ground re-plans afterwards")
        self.skip = QtWidgets.QPushButton("⏭  Skip to the next event")
        self.finish = QtWidgets.QPushButton("⏩  Run to the end")
        self.finish.setToolTip("Full nonlinear propagation of the deflected orbit, then the clock runs through the\n"
                               "2135 flyby to the impact date")
        self.end_live = QtWidgets.QPushButton("■  End live run")
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("Near-Bennu controller"))
        self.prox_ctrl = QtWidgets.QComboBox()
        self.prox_ctrl.addItem("MPC (constraints)", "MPC")
        self.prox_ctrl.addItem("LQR", "LQR")
        crow.addWidget(self.prox_ctrl, 1)
        for w in (self.storm_days, self.storm, self.skip, self.finish, self.end_live):
            ll.addWidget(w)
        ll.addLayout(crow)
        self.live_hint = QtWidgets.QLabel("The run starts today with both spacecraft in Earth orbit. Launch a DART; "
                                          "after a successful impact the gravity tractor can launch.")
        self.live_hint.setWordWrap(True)
        self.live_hint.setStyleSheet("color: #898781;")
        ll.addWidget(self.live_hint)
        lay.addWidget(box)

        box, dl = section("Display")
        self.gravity_overlay = QtWidgets.QCheckBox("Color Bennu by surface gravity")
        self.sun_wobble = QtWidgets.QCheckBox("Exaggerate the Sun's motion around the barycenter ×30")
        self.galactic_slow = QtWidgets.QCheckBox("🏠 side view: slow the Galactic motion ×1/50 (shows the helices)")
        self.galactic_slow.setToolTip("Display only. The Solar System moves 52 AU per year around the Milky Way, so the\n"
                                      "planets' true paths are long, stretched helices; slowed 50 times, the turns show.")
        self.constellations = QtWidgets.QCheckBox("Constellation background (off: the Milky Way)")
        self.constellations.setToolTip("The background of every view, the Solar System included. Off (default): the Milky\n"
                                       "Way, NASA SVS Deep Star Maps 2020 (Gaia DR2, Hipparcos, Tycho-2). On: the catalogue\n"
                                       "stars to magnitude 8 with the 88 IAU constellation figures and names. Switching\n"
                                       "cross-fades over 3 s. Both are the real sky at its J2000 positions.")
        self.unlimited = QtWidgets.QCheckBox("Unlimited fuel (mass constant, consumption tracked)")
        self.unlimited.setChecked(True)
        for w in (self.gravity_overlay, self.sun_wobble, self.galactic_slow, self.constellations, self.unlimited):
            dl.addWidget(w)
        lay.addWidget(box)
        lay.addStretch(1)

        for w in (self.goal, self.target, self.thrust, self.boulder, self.tow, self.hover, self.imp_mass, self.vmax,
                  self.beta, self.beta_sigma):
            w.valueChanged.connect(lambda _=None: self.paramsChanged.emit())
        for c in (self.objective, self.sizing):
            c.currentIndexChanged.connect(lambda _=None: self.paramsChanged.emit())
        self.opportunity.activated.connect(lambda _=None: self.paramsChanged.emit())
        self.launch_after.dateChanged.connect(lambda _=None: self.paramsChanged.emit())
        for c in (self.unlimited, self.use_rec):
            c.toggled.connect(lambda _=None: (self._mass_enabled(), self.paramsChanged.emit()))
        self.vmax.valueChanged.connect(lambda _=None: self._vmax_text())
        self._mass_enabled()
        self._vmax_text()
        self.set_stage("plan")

    def _mass_enabled(self):
        self.imp_mass.setEnabled(not self.use_rec.isChecked())
        self.sizing.setEnabled(self.use_rec.isChecked())

    def _vmax_text(self):
        v = self.vmax.value() * 1e3
        v_esc = departure_speed(0.0, R_PARK)
        vinf = np.sqrt(max(v**2 - v_esc**2, 0.0))
        self.vmax_info.setText(f"→ leaves Earth with v∞ up to {vinf / 1e3:.2f} km/s (C3 {vinf**2 / 1e6:.1f} km²/s²); "
                               f"Δv from the parking orbit {(v - np.sqrt(3.986004418e14 / R_PARK)) / 1e3:.2f} km/s")
        self.vmax_info.setWordWrap(True)

    STATUS_STYLE = {"red": ("#3a1c1c", "#f0a0a0", "#7a3030"), "amber": ("#3a3012", "#fad27a", "#7a6520"),
                    "green": ("#143a1e", "#9fe3b0", "#2a7a3e"), "grey": ("#22262c", "#c3c2b7", "#383835")}

    def set_tractor_status(self, colour, text):
        """The coloured strip above the tractor's launch button (red / amber / green / grey) with why it can or
        cannot launch now."""
        bg, fg, edge = self.STATUS_STYLE[colour]
        self.tractor_status.setStyleSheet(f"background: {bg}; color: {fg}; border: 1px solid {edge}; border-radius: 4px;")
        self.tractor_status.setText(text)
        self._tractor_status = (colour, text)

    def set_stage(self, stage, dart_ready=True, tractor_ready=False):
        """plan: planning mode; live: a live run (DART / tractor launch buttons per the mission state)."""
        if self.on_stage is not None:  # the session log (app/session_log.py), with the caller
            import sys

            self.on_stage(stage, dart_ready, tractor_ready, sys._getframe(1).f_code.co_name)
        live = stage == "live"
        for w in (self.storm, self.storm_days, self.skip, self.finish, self.end_live, self.release):
            w.setVisible(live)
        self.launch_dart.setEnabled(dart_ready)
        self.launch_dart.setText("🚀  Launch DART" if not live else "🚀  Launch the next DART")
        self.launch_tractor.setEnabled(tractor_ready)
        if not live:
            self.release.setEnabled(False)
        self.slider.setEnabled(not live)
        self.jump.setEnabled(not live)
        self.launch_after.setEnabled(not live)  # (a live run's next DARTs launch from the mission clock)
        self.follow_slider.setEnabled(not live)

    def earliest_launch_t(self):
        from core.timeutil import iso_to_t

        return iso_to_t(self.launch_after.date().toString("yyyy-MM-dd"))

    def set_earliest_launch(self, t):
        """The earliest launch from a time (the slider's): the date edit shows it and the plan follows it."""
        q = QtCore.QDate.fromString(t_to_utc_str(t)[:10], "yyyy-MM-dd")
        if q != self.launch_after.date():
            self.launch_after.setDate(q)  # (dateChanged: paramsChanged, the plan is redone)

    def set_release_ready(self, on):
        """The release button follows the mission state (set on every snapshot): the tractor holds at Bennu and
        the tow is not over."""
        if self.release.isEnabled() != bool(on):
            self.release.setEnabled(bool(on))

    def lock_dart(self, locked):
        for w in (self.opportunity, self.use_rec, self.sizing, self.imp_mass, self.vmax, self.beta, self.beta_sigma):
            w.setEnabled(not locked)
        if not locked:
            self._mass_enabled()

    def lock_tractor(self, locked):
        for w in (self.objective, self.boulder, self.tow, self.hover):
            w.setEnabled(not locked)

    def set_opportunities(self, opps, selected_index, recommended_index):
        self.opportunity.blockSignals(True)
        self.opportunity.clear()
        self.opportunity.addItem("Recommended" + (f" ({self._opp_label(next(o for o in opps if o.index == recommended_index))})"
                                                 if recommended_index is not None else ""), -1)
        for o in opps:
            self.opportunity.addItem(self._opp_label(o), o.index)
        k = self.opportunity.findData(selected_index)
        self.opportunity.setCurrentIndex(max(0, k))
        self.opportunity.blockSignals(False)

    @staticmethod
    def _opp_label(o):
        if not o.feasible:
            return f"{t_to_utc_str(o.t_pass)[:7]} pass — no arc"
        return f"{t_to_utc_str(o.t_pass)[:7]} pass · impact {t_to_utc_str(o.t_arrive)[:7]} · ~{_mass(o.mass_nominal)} head-on"

    def select_opportunity(self, index):
        k = self.opportunity.findData(index)
        if k >= 0:
            self.opportunity.setCurrentIndex(k)
            self.paramsChanged.emit()

    def params(self) -> MissionParams:
        return MissionParams(
            goal_ld=self.goal.value(), dart_target_km=self.target.value(), tractor_thrust_n=self.thrust.value(),
            tractor_objective=self.objective.currentData(), boulder_mass_kg=self.boulder.value() * 1e3,
            tow_years=self.tow.value(), hover_factor=self.hover.value(),
            opportunity=self.opportunity.currentData() if self.opportunity.count() else -1,
            use_recommended_mass=self.use_rec.isChecked(), dart_sizing=self.sizing.currentData(),
            impactor_mass_kg=self.imp_mass.value(), dart_max_speed_kms=self.vmax.value(), beta=self.beta.value(),
            beta_sigma=self.beta_sigma.value(), unlimited_fuel=self.unlimited.isChecked(),
            earliest_launch=self.launch_after.date().toString("yyyy-MM-dd"))

    def set_jumps(self, items):
        """items: [(label, t)]"""
        self.jump.blockSignals(True)
        self.jump.clear()
        self.jump.addItem("Jump to…")
        for label, t in items:
            self.jump.addItem(label, t)
        self.jump.blockSignals(False)

    def set_goal_info(self, res):
        if res is None:
            return
        self.goal_info.setText(f"Keyhole flyby now {np.linalg.norm(res.b0) / 1e3:,.0f} km "
                               f"({np.linalg.norm(res.b0) / LUNAR_DISTANCE:.2f} LD). Needs a shift of "
                               f"{res.required_shift / 1e3:,.0f} km; the DART target "
                               f"{res.target_shift / 1e3:,.0f} km puts it at "
                               f"{np.linalg.norm(res.b_aim) / LUNAR_DISTANCE:.2f} LD.")

    def set_time(self, t):
        self.date_label.setText(t_to_utc_str(t))
        self.slider.blockSignals(True)
        self.slider.setValue(int((t - self.t_lo) / DAY))
        self.slider.blockSignals(False)


def dart_plan_html(ip, title="Kinetic impactor (DART)", refining=False):
    h = [f"<hr><b>{title}</b>"]
    if not ip.feasible:
        h.append(f"<div style='color:{CRITICAL}'>✗ no trajectory for this opportunity (see checks)</div>")
        return h
    rp = ip.raising
    rows = [("Orbit raising starts", _fmt_date(rp.t_start)), ("Departure (injection)", _fmt_date(ip.t_launch)),
            ("Impact", _fmt_date(ip.t_arrive)), ("Flight time", f"{ip.tof_days:,.0f} d"),
            ("Impact speed", f"<b>{ip.impact_speed / 1e3:.2f} km/s</b>"),
            ("Mass at impact", f"<b>{_mass(ip.mass)}</b>"),
            ("Recommended mass", f"{_mass(ip.mass_nominal)} (planned β) · {_mass(ip.mass_p90)} (90 %)"),
            ("… sized with", "the full force model" + (f" (linear estimate {_mass(ip.mass_linear)})"
                                                      if np.isfinite(ip.mass_linear) else "")
             if ip.mass_source == "full model" else
             "the linear estimate" + (" — <i>refining with the full force model…</i>" if refining else "")),
            ("Departure speed / v∞", f"{ip.v_dep / 1e3:.2f} / {ip.v_inf / 1e3:.2f} km/s (C3 {ip.c3 / 1e6:.1f})"),
            ("Bennu Δv from impact", f"{np.linalg.norm(ip.dv_bennu) * 1e3:.3f} mm/s"),
            ("2135 flyby shift", f"<b>{np.linalg.norm(ip.db) / 1e3:,.0f} km</b>"),
            ("Propellant (all burns)", f"{ip.propellant_kg:,.0f} kg for Δv {rp.dv_total / 1e3:.2f} km/s")]
    c = ip.contact or {}
    if "lat" in c:
        rows.append(("Point of contact", f"{c['lat']:+.1f}° lat, {c['lon']:+.1f}° lon, {c['offset_m']:.0f} m off-center"))
        rows.append(("… incidence / effect", f"{c['incidence_deg']:.0f}° from the vertical · {c['efficiency']:.2f}× head-on"))
    h += _rows(rows)
    h.append("<div style='color:#c3c2b7; margin-top:4px'>Orbit raising (burns at perigee):</div><table>")
    h.append("<tr><td>#</td><td>date</td><td align='right'>Δv</td><td align='right'>perigee v</td>"
             "<td align='right'>apogee</td><td align='right'>fuel</td></tr>")
    for k, b in enumerate(rp.burns, 1):
        apo = "escape" if not np.isfinite(b.r_apogee) else f"{b.r_apogee / 1e3:,.0f} km"
        h.append(f"<tr><td>{k}{'*' if b.kind == 'inject' else ''}</td><td>{t_to_utc_str(b.t)[:16]}</td>"
                 f"<td align='right'>{b.dv:,.0f} m/s</td><td align='right'>{b.v_perigee / 1e3:.2f} km/s</td>"
                 f"<td align='right'>{apo}</td><td align='right'>{b.propellant_kg:,.0f} kg</td></tr>")
    h.append("</table><div style='color:#898781; font-size:11px'>* injection toward Bennu</div>")
    return h


def opportunities_html(opps, selected=None, title="DART opportunities (recommended mass)"):
    h = [f"<hr><b>{title}</b><table>",
         "<tr><td>pass</td><td>impact</td><td align='right'>speed</td><td align='right'>mass</td>"
         "<td align='right'>90 %</td></tr>"]
    for o in opps:
        mark = "▶ " if selected is not None and o.index == selected else ""
        if not o.feasible:
            h.append(f"<tr><td>{mark}{t_to_utc_str(o.t_pass)[:7]}</td><td colspan=4 style='color:#898781'>"
                     f"{o.note or 'no arc'}</td></tr>")
            continue
        col = GOOD if o.mass_nominal < 2e3 else (WARN if o.mass_nominal < 1e5 else CRITICAL)
        h.append(f"<tr><td>{mark}{t_to_utc_str(o.t_pass)[:7]}</td><td>{t_to_utc_str(o.t_arrive)[:10]}</td>"
                 f"<td align='right'>{o.impact_speed / 1e3:.1f} km/s</td>"
                 f"<td align='right' style='color:{col}'>{_mass(o.mass_nominal)}</td>"
                 f"<td align='right'>{_mass(o.mass_p90)}</td></tr>")
    h.append("</table><div style='color:#898781; font-size:11px'>Head-on estimate at the planned β for the shift "
             "still needed; the selected plan refines it with the point of contact.</div>")
    return h


class MissionPanel(QtWidgets.QScrollArea):
    """Info tab: the planner's answer, refreshed live."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QtWidgets.QLabel("Planning…")
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setWordWrap(True)
        self.label.setAlignment(QtCore.Qt.AlignTop)
        self.label.setMargin(8)
        self.setWidget(self.label)
        self.flight_plan = None
        self._last_res = None
        self.refining = False

    def show_error(self, msg):
        self.label.setText(f"<b style='color:{CRITICAL}'>Planner error</b><pre>{msg[:1500]}</pre>")

    def set_flight_plan(self, fp):
        self.flight_plan = fp
        if self._last_res is not None:
            self.show_plan(self._last_res)

    def show_plan(self, res, title=None):
        self._last_res = res
        tp, ip, p = res.tractor, res.impactor, res.params
        q = res.q_final
        ok = res.success
        color, icon, word = (GOOD, "✓", "EARTH IS SAFE") if ok else (CRITICAL, "✗", "GOAL NOT MET")
        h = [f"<div style='font-size:17px; font-weight:700; color:{color}'>{icon} {word}</div>",
             f"<div>Planned 2135 flyby: <b>{q / 1e3:,.0f} km = {q / LUNAR_DISTANCE:.2f} LD</b> (goal {p.goal_ld:g} LD)<br>"
             f"Chance of reaching the goal over β uncertainty: <b>{100 * res.p_success:.0f} %</b></div>",
             f"<div style='color:#c3c2b7'>Without a deflection: keyhole flyby at {np.linalg.norm(res.b0) / 1e3:,.0f} km, "
             f"then an Earth impact.</div>"]
        if title:
            h.insert(0, f"<div style='color:{WARN}; font-weight:700'>{title}</div>")
        h.append("<hr><b>Deflection of the 2135 flyby (B-plane)</b>")
        h += _rows([("Needed for the goal", f"{res.required_shift / 1e3:,.0f} km"),
                    ("DART target (aim point)", f"{res.target_shift / 1e3:,.0f} km → {np.linalg.norm(res.b_aim) / LUNAR_DISTANCE:.2f} LD"),
                    ("DART (planned β)", f"{np.linalg.norm(res.db_ki) / 1e3:,.0f} km"),
                    ("Gravity-tractor trim", f"{np.linalg.norm(res.db_gt) / 1e3:,.0f} km"),
                    ("Lead time (impact → flyby)", f"{res.lead_time_years:.1f} yr" if np.isfinite(res.lead_time_years) else "—")])
        h += dart_plan_html(ip, f"DART #{ip.number}" if ip.feasible else "DART", refining=self.refining)
        h.append("<hr><b>Gravity tractor (launched after the DART succeeds)</b>")
        if tp.feasible:
            fp = self.flight_plan
            fp_text = ("<i>optimizing…</i>" if fp is None else f"<span style='color:{CRITICAL}'>none feasible</span>"
                       if not fp["feasible"] else f"<b>{_fmt_date(fp['t_arrive'])}</b> ({fp['dv'] / 1e3:,.1f} km/s)")
            h += _rows([("Launch (after ground tracking)", _fmt_date(tp.t_launch)),
                        ("Escape spiral", f"{tp.escape_days:,.0f} d"),
                        ("Cruise", f"{tp.cruise_days:,.0f} d ({tp.revolutions} rev)"),
                        ("Arrives (quick estimate)", _fmt_date(tp.t_arrive)),
                        ("Arrives (optimized flight plan)", fp_text),
                        ("Propellant", f"{tp.propellant_kg:,.0f} kg"),
                        ("Hover distance / thrust", f"{tp.hover_distance_m:,.0f} m / {tp.hover_thrust_n:.2f} N"),
                        ("Tow", f"{_fmt_date(tp.tow_start)} → {_fmt_date(tp.tow_end)}"),
                        ("Bennu Δv from the tow", f"{tp.gt_dv_bennu * 1e3:,.2f} mm/s"),
                        ("Tow reach (2135 shift)", f"{tp.reach / 1e3:,.0f} km")])
        else:
            h.append(f"<div style='color:{CRITICAL}'>✗ cannot reach Bennu (see checks)</div>")
        h += opportunities_html(res.opportunities, ip.opportunity if ip.feasible else None)
        h.append("<hr><b>Checks</b><br>")
        sym = {"ok": ("✓", GOOD), "warn": ("⚠", WARN), "fail": ("✗", CRITICAL)}
        for c in res.checks:
            s, col = sym[c.level]
            h.append(f"<span style='color:{col}'>{s}</span> {c.text}<br>")
        h.append(f"<div style='color:#898781; font-size:11px'>planned in {res.compute_ms:.0f} ms (linear STM estimate; "
                 "in the live run Bennu is integrated with the full force model)</div>")
        self.label.setText("".join(h))


class TelemetryPanel(QtWidgets.QWidget):
    def __init__(self, scenario):
        super().__init__()
        self.sc = scenario
        lay = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Bennu now (scenario orbit)")
        self.form = QtWidgets.QFormLayout(box)
        self.values = {}
        names = {"date": "Date (UTC)", "d_earth_km": "Bennu – Earth", "d_earth_au": "", "d_earth_ld": "",
                 "v_rel": "Speed rel. Earth", "d_sun": "Bennu – Sun", "v_helio": "Heliocentric speed",
                 "next_ca": "Next Earth approach", "next_ca_dist": "", "next_ca_in": "", "impact": "Earth impact"}
        for key, name in names.items():
            lab = QtWidgets.QLabel("—")
            self.values[key] = lab
            self.form.addRow(name, lab)
        lay.addWidget(box)
        facts = QtWidgets.QGroupBox("101955 Bennu (JPL SBDB) and the scenario")
        fl = QtWidgets.QFormLayout(facts)
        lat, lon = scenario.impact_latlon
        for k, v in [("GM", f"{BENNU['gm']:.4f} m³/s²"), ("Mass", f"{BENNU['mass']:.3e} kg"),
                     ("Diameter", f"{BENNU['diameter']:.1f} m"), ("Rotation", f"{BENNU['rot_period_h']:.4f} h"),
                     ("2135 keyhole flyby", f"{t_to_utc_str(scenario.t_ca)[:16]} · {scenario.q_ca / 1e3:,.0f} km "
                                            f"({scenario.q_ca / LUNAR_DISTANCE:.2f} LD)"),
                     ("Keyhole width", f"{scenario.keyhole_width / 1e3:.1f} km along the line of variations"),
                     ("Impact (if nothing is done)", f"{t_to_utc_str(scenario.t_impact)[:16]} UTC, "
                                                     f"{scenario.impact_v / 1e3:.1f} km/s, {lat:+.0f}° {lon:+.0f}°")]:
            lab = QtWidgets.QLabel(v)
            lab.setWordWrap(True)
            fl.addRow(k, lab)
        lay.addWidget(facts)
        lay.addStretch(1)

    def update(self, eph, t, bennu_pos=None):
        lo, hi = eph.span("bennu")
        tb = float(np.clip(t, lo, hi))
        if bennu_pos is not None:
            r = np.asarray(bennu_pos) - eph.pos("earth", t)
        else:
            r = eph.rel_pos("bennu", "earth", tb)
        v = eph.rel_vel("bennu", "earth", tb)
        d = np.linalg.norm(r)
        self.values["date"].setText(t_to_utc_str(t))
        self.values["d_earth_km"].setText(f"{d / 1e3:,.0f} km")
        self.values["d_earth_au"].setText(f"{d / AU:.5f} au")
        self.values["d_earth_ld"].setText(f"{d / LUNAR_DISTANCE:.2f} lunar distances")
        self.values["v_rel"].setText(f"{np.linalg.norm(v) / 1e3:.2f} km/s")
        self.values["d_sun"].setText(f"{np.linalg.norm(eph.rel_pos('bennu', 'sun', tb)) / AU:.4f} au")
        self.values["v_helio"].setText(f"{np.linalg.norm(eph.rel_vel('bennu', 'sun', tb)) / 1e3:.2f} km/s")
        upcoming = [a for a in self.sc.approaches if a["t"] > t and a["dist"] < 0.3 * AU]
        if upcoming:
            a = upcoming[0]
            self.values["next_ca"].setText(f"{t_to_utc_str(a['t'])} UTC")
            self.values["next_ca_dist"].setText(f"{a['dist'] / AU:.5f} au ({a['dist'] / LUNAR_DISTANCE:.2f} LD)")
            self.values["next_ca_in"].setText(f"in {(a['t'] - t) / YEAR:.2f} years")
        else:
            for k in ("next_ca", "next_ca_dist", "next_ca_in"):
                self.values[k].setText("—")
        dt = self.sc.t_impact - t
        self.values["impact"].setText(f"in {dt / YEAR:,.2f} years (if not deflected)" if dt > 0 else "happened")


PHASE_TEXT = {"waiting": "in the parking orbit", "parking": "in the parking orbit", "raising": "raising its orbit",
              "departing": "leaving Earth (hyperbola)", "spiral": "spiraling out of Earth orbit",
              "escape": "final escape revolutions", "cruise": "heliocentric cruise", "proximity": "at Bennu",
              "terminal": "terminal guidance", "impacted": "impacted Bennu", "missed": "missed Bennu"}
PROX_TEXT = {"approach": "approaching Bennu", "standoff": "stand-off: waiting for the impact",
             "survey": "surveying the capture site", "descent": "descending to the boulder",
             "capture": "capturing the boulder", "ascent": "ascending with the boulder",
             "transfer": "moving to the tractor station", "station": "on station (towing)",
             "parked": "parked 5 km out (tow off)", "release": "releasing the boulder (end of the tow)",
             "depart": "departing: climbing away from Bennu", "departed": "left Bennu (tow over)"}
LEVEL_COLOR = {"info": "#c3c2b7", "warn": WARN, "good": GOOD, "bad": CRITICAL}


def _warp_text(w):
    if w >= 365.25 * DAY:
        return f"{w / (365.25 * DAY):,.1f} yr/s"
    if w >= DAY:
        return f"{w / DAY:,.1f} days/s"
    if w >= 3600:
        return f"{w / 3600:,.1f} h/s"
    if w >= 60:
        return f"{w / 60:,.1f} min/s"
    return "real time" if abs(w - 1.0) < 1e-9 else f"{w:,.1f} s/s"


def _days(dt):
    return f"{dt / DAY:,.0f} d" if abs(dt) >= DAY else f"{dt / 3600:,.1f} h"


class LivePanel(QtWidgets.QScrollArea):
    """Info tab during a live run: outlook, the DARTs (orbit raising and fuel), the tractor, events."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QtWidgets.QLabel("Press 🚀 Launch DART to start the live run.")
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setWordWrap(True)
        self.label.setAlignment(QtCore.Qt.AlignTop)
        self.label.setMargin(8)
        self.setWidget(self.label)
        self.initial_eta = None
        self.recommendation = None  # (opportunities, next DART plan) after a shortfall
        self.refining = False

    @staticmethod
    def _prox_rows(pr):
        rng = pr["range"]
        rows = [("Controller", pr["controller"] + (f" · {pr['mpc_fallbacks']} fallbacks" if pr["mpc_fallbacks"] else "")),
                ("Distance to Bennu's center", f"{rng / 1e3:,.2f} km" if rng > 5e3 else f"{rng:,.1f} m")]
        if np.isfinite(pr["altitude"]) and rng < 20e3:
            rows.append(("Altitude above the surface", f"{pr['altitude']:,.1f} m"))
        rows.append(("Speed relative to Bennu", f"{np.linalg.norm(pr['v_rel']) * 100:,.1f} cm/s"))
        rows.append(("Thrust now / max", f"{np.linalg.norm(pr['thrust_vec']):.2f} / {pr['thrust_max']:g} N"))
        rows.append(("Mass", f"{pr['mass'] / 1e3:,.1f} t" + (" (with boulder)" if pr["has_boulder"] else "")))
        if pr.get("capture_stage"):
            rows.append(("Capture", pr["capture_stage"]))
        if pr.get("lidar") is not None:
            rows.append(("LIDAR range", f"{pr['lidar']:,.2f} m"))
        if np.isfinite(pr["nav_err"]):
            rows.append(("Nav error / 3σ", f"{pr['nav_err']:,.2f} / {3 * pr['nav_sigma']:,.2f} m"))
        else:
            rows.append(("Navigation", "arm cameras (filter paused)"))
        return rows

    def show_error(self, msg):
        self.label.setText(f"<b style='color:{CRITICAL}'>Simulation error</b><pre>{msg[:1500]}</pre>")

    @staticmethod
    def _dart_rows(ki, t):
        rows = [("Phase", PHASE_TEXT.get(ki["phase"], ki["phase"])), ("Mass", _mass(ki["mass"]))]
        n_raise = ki["burns_total"] - 1
        if ki["phase"] == "parking":
            rows.append(("Orbit raising starts", f"{_fmt_date(ki['t_raise_start'])} (in {_days(ki['t_raise_start'] - t)})"))
        if ki["phase"] in ("parking", "raising", "departing"):
            rows.append(("Burns done", f"{min(ki['burns_done'], n_raise)}/{n_raise} raising"
                                       + (" + injection" if ki["burns_done"] > n_raise else "")))
            rows.append(("Speed / perigee speed", f"{ki['speed_geo'] / 1e3:.2f} / {ki['v_perigee'] / 1e3:.2f} km/s "
                                                  f"(departs at {ki['v_dep'] / 1e3:.2f})"))
            apo = ki["r_apogee"]
            rows.append(("Apogee", "escape (hyperbola)" if not np.isfinite(apo) else f"{apo / 1e3:,.0f} km"))
            if np.isfinite(ki["next_burn"]):
                rows.append(("Next burn", f"{t_to_utc_str(ki['next_burn'])[:16]} (in {_days(ki['next_burn'] - t)})"))
        rows.append(("<b>Fuel spent</b>", f"<b>{ki['prop_used']:,.0f} kg</b> of {ki['prop_plan']:,.0f} kg planned"))
        rows.append(("Δv used", f"{ki['dv_used'] / 1e3:,.3f} km/s"))
        if ki["impact"] is not None:
            im = ki["impact"]
            rows += [("Impact", t_to_utc_str(im["t"])[:16]), ("Impact speed", f"{np.linalg.norm(im['U']) / 1e3:.2f} km/s"),
                     ("Bennu Δv", f"<b>{np.linalg.norm(im['dv_bennu']) * 1e3:.3f} mm/s</b>"),
                     ("Impact site", f"{im['lat']:+.1f}° lat, {im['lon']:+.1f}° lon"
                                     + (f" ({im['aim_error_m']:.0f} m from the aim)" if np.isfinite(im.get('aim_error_m', np.nan)) else "")),
                     ("Angle from the vertical", f"{im['incidence_deg']:.0f}°"),
                     ("Bennu's day", f"{im['period_change_s']:+,.1f} s"), ("Ejecta (model)", f"{im['ejecta_mass']:.1e} kg")]
        elif ki["phase"] in ("cruise", "terminal"):
            rows += [("Impact in", _days(ki["t_arrive"] - t)), ("Range", f"{ki['range'] / 1e3:,.0f} km"),
                     ("Closing speed", f"{ki['rel_speed'] / 1e3:.2f} km/s")]
        elif ki["phase"] == "missed":
            rows.append(("Closest approach", f"{ki['closest'] / 1e3:,.1f} km"))
        return rows

    def show_snapshot(self, s):
        t = s["t"]
        req = s.get("warp_requested", np.nan)
        h = [f"<div style='font-size:15px; font-weight:700'>{t_to_utc_str(t)} UTC</div>",
             f"<div style='color:#c3c2b7'>{'running' if s['running'] else 'paused'} · speed "
             f"{_warp_text(s['warp_achieved'])}"
             + (" (skipping ahead as fast as the physics computes)" if not np.isfinite(req) and s["running"] else "")
             + "</div>"]
        sm = s.get("slowmo")
        if sm:
            h.append(f"<div style='color:#7fd4ff; font-weight:700'>🎬 Slow motion — {sm['what']}: "
                     f"{_warp_text(sm['warp'])} (your {_warp_text(s.get('warp_user', np.nan))} resumes after)</div>")
        if s.get("behind") and np.isfinite(req):
            h.append(f"<div style='color:{WARN}; font-weight:700'>⏳ The physics computes slower than the chosen "
                     f"{_warp_text(req)}: the clock shows only what has been computed</div>")
        if s.get("busy"):
            h.append(f"<div style='color:{WARN}; font-weight:700'>⏳ Waiting for the physics: {s['busy']}…</div>")
        elif s.get("background"):
            bg = s["background"]
            h.append(f"<div style='color:#c3c2b7'>⚙ In the background: {bg['what']}</div>")
        if t < s["safe_until"]:
            h.append(f"<div style='color:{WARN}; font-weight:700'>☀ SOLAR STORM — safe mode, thrusters off for "
                     f"{_days(s['safe_until'] - t)}</div>")
        o = s["outlook"]
        q, goal = o["q_final"], o["goal"]
        col, icon = (GOOD, "✓") if o["success"] else (CRITICAL, "✗")
        fc = o.get("final_check")
        verified = fc is not None and "q" in fc
        h.append(f"<hr><div style='font-size:15px; font-weight:700; color:{col}'>{icon} 2135 flyby "
                 f"{'(full-model physics)' if verified else '(projected, linear)'}: {q / 1e3:,.0f} km = "
                 f"{q / LUNAR_DISTANCE:.2f} LD</div>"
                 f"<div>goal {goal / LUNAR_DISTANCE:g} LD · aim {np.linalg.norm(o['b_aim']) / LUNAR_DISTANCE:.2f} LD · "
                 f"now (pushes so far) {np.linalg.norm(o['b_now']) / LUNAR_DISTANCE:.2f} LD</div>")
        t_hit = o.get("t_earth_impact", np.nan)
        if np.isfinite(t_hit):
            h.append(f"<div style='color:{CRITICAL}'>Earth impact on {_fmt_date(t_hit)} unless deflected</div>")
        elif verified:
            h.append(f"<div style='color:{GOOD}'>Full-model physics: no Earth impact through 2184</div>")
        if o.get("trim_mode") is not None:
            rows = [("Tow Δv delivered", f"{o['tow_dv'] * 1e3:.4f} mm/s"), ("Tow ends", _fmt_date(o["tow_end"])),
                    ("Aim-point error (predicted)", f"{np.linalg.norm(o['b_final'] - o['b_aim']) / 1e3:,.1f} km")]
            h += _rows(rows)
        if fc is not None and "q" in fc:
            h.append(f"<div style='color:#c3c2b7'>Full-model check: linear prediction off by "
                     f"{fc['difference_km']:,.0f} km ({100 * fc['relative']:.1f} %)</div>")
        for r in o.get("dart_results", []):
            if r["result"] == "success":
                h.append(f"<div style='color:{GOOD}'>DART #{r['number']}: SUCCESS — {100 * r['fraction']:.0f} % of the "
                         f"target; flyby at {r['q'] / LUNAR_DISTANCE:.2f} LD</div>")
            else:
                h.append(f"<div style='color:{CRITICAL}'>DART #{r['number']}: {r['result'].upper()} — "
                         f"{100 * r['fraction']:.0f} % of the target; another DART needed for "
                         f"{r['remaining'] / 1e3:,.0f} km</div>")
        if s.get("beta_ground") is not None:
            b, sb = s["beta_ground"]
            h.append(f"<div>β from ground tracking: <b>{b:.2f} ± {sb:.1f}</b></div>")
        if s.get("beta_measured") is not None:
            b, sb = s["beta_measured"]
            h.append(f"<div>β measured by the tractor: <b>{b:.2f} ± {sb:.2f}</b></div>")
        if self.recommendation is not None and not s.get("dart_in_flight") and not s.get("dart_ok"):
            opps, ip = self.recommendation
            h.append(f"<hr><div style='color:{WARN}; font-weight:700'>Recommended next DART</div>")
            if ip is not None and ip.feasible:
                c = ip.contact or {}
                h += _rows([("Opportunity", f"{_fmt_date(ip.t_launch)} → impact {_fmt_date(ip.t_arrive)}"),
                            ("Velocity at impact", f"<b>{ip.impact_speed / 1e3:.2f} km/s</b> (cut-off "
                                                   f"{ip.v_dep / 1e3:.2f} km/s at departure)"),
                            ("Mass", f"<b>{_mass(ip.mass)}</b> ({_mass(ip.mass_p90)} for 90 %) · "
                                     + ("full model" if ip.mass_source == "full model" else
                                        "linear estimate" + (", refining…" if self.refining else ""))),
                            ("Point of contact", f"{c['lat']:+.1f}° lat, {c['lon']:+.1f}° lon ({c['efficiency']:.2f}× head-on)"
                             if "lat" in c else "center")])
                h.append("<div style='color:#c3c2b7'>Press 🚀 Launch the next DART (the sliders hold these values).</div>")
            h += opportunities_html(opps, ip.opportunity if ip is not None else None, "Remaining opportunities")
        for ki in s.get("impactors") or []:
            h += [f"<hr><b>{ki['name']}</b>"] + _rows(self._dart_rows(ki, t))
        tr = s["tractor"]
        if tr is not None:
            if self.initial_eta is None and np.isfinite(tr["eta"]) and tr["phase"] == "cruise":
                self.initial_eta = tr["eta"]
            state = ("SAFE MODE" if t < s["safe_until"] else "THRUSTING" if tr["thrusting"] else "coasting")
            pr = tr.get("prox")
            if not tr.get("launched"):
                rows = [("Phase", "in the parking orbit — launches after a successful DART")]
            else:
                rows = [("Phase", PROX_TEXT.get(pr["phase"], pr["phase"]) if pr else PHASE_TEXT.get(tr["phase"], tr["phase"])),
                        ("Engine", f"{tr['thrust']:g} N · {state}"),
                        ("Launch", _fmt_date(tr["t_launch"])),
                        ("Arrival" if pr else "Arrival (flight plan)",
                         _fmt_date(tr["eta"]) if np.isfinite(tr["eta"]) else "<b>none feasible</b>")]
                if pr is not None:
                    rows += self._prox_rows(pr)
                if self.initial_eta is not None and np.isfinite(tr["eta"]) and pr is None:
                    rows.append(("Change since first plan", f"{(tr['eta'] - self.initial_eta) / DAY:+,.0f} days"))
                if "range" in tr and pr is None:
                    rows.append(("Distance to Bennu", f"{tr['range'] / 1e3:,.0f} km"))
                    rows.append(("Speed relative to Bennu", f"{tr['rel_speed']:,.1f} m/s"))
                rows += [("Δv used", f"{tr['dv_used'] / 1e3:,.2f} km/s"), ("<b>Fuel spent</b>", f"<b>{tr['prop_used']:,.0f} kg</b>")]
                if "nav_err" in tr:
                    rows.append(("Navigation error / 3σ", f"{tr['nav_err'] / 1e3:.2f} / {3 * tr['nav_sigma'] / 1e3:.2f} km"))
            h += ["<hr><b>Gravity tractor</b>"] + _rows(rows)
        h.append("<hr><b>Event log</b><br>")
        for e in reversed(s["events"][-18:]):
            h.append(f"<span style='color:#898781'>{t_to_utc_str(e.t)[:10]}</span> "
                     f"<span style='color:{LEVEL_COLOR[e.level]}'>[{e.who}] {e.text}</span><br>")
        self.label.setText("".join(h))
