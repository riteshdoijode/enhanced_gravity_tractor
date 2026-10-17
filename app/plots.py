"""In-app pyqtgraph plots (dark theme).

Dark-mode steps of the validated palette (adjacent CVD dE >= 8.4, all >= 3:1 on the app surface);
text in ink tokens, hairline solid grid, one y axis per plot, a legend whenever >= 2 series.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from core.constants import AU, DAY, LUNAR_DISTANCE, RADIUS, YEAR
from core.timeutil import t_to_utc_str, t_to_year

SURFACE = "#14171b"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
S1, S2, S3, S4 = "#3987e5", "#d95926", "#199e70", "#c98500"  # blue, orange, aqua, yellow (dark steps)
CRITICAL, GOOD, WARNING = "#d03b3b", "#0ca30c", "#fab219"  # status palette (critical / good / warning)
BENNU_COLOR = "#ffb347"  # Bennu's orange, as in the Solar System view
TRACTOR_COLOR = S1  # the gravity tractor's blue: its launch window and planned path, here and in the Solar System view
# DART window markers: what the DART as set does in each window (core/planner.py: window_capability). Each class
# has its own symbol as well as its status colour, so the legend never relies on colour alone.
WINDOW_STATUS = [("target", GOOD, "o", "●", "reaches the target with this DART"),
                 ("goal", WARNING, "d", "◆", "clears the goal ({goal}) but not the target"),
                 ("short", CRITICAL, "t", "▼", "falls short of the goal"),
                 ("none", CRITICAL, "x", "✕", "no arc within the max speed")]
LINEAR_NOTE = "Linear (STM) planner estimates: the full-model refinement can differ by tens of % for big shifts"
BLUE_RAMP_DARK = ["#14171b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
                  "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]  # surface -> light: magnitude up


def _style(plot: pg.PlotWidget, xlabel, ylabel, title=None):
    plot.setBackground(SURFACE)
    item = plot.getPlotItem()
    for name in ("left", "bottom"):
        ax = item.getAxis(name)
        ax.setPen(pg.mkPen(AXIS))
        ax.setTextPen(pg.mkPen(INK2))
    item.setLabel("bottom", xlabel, color=INK2)
    item.setLabel("left", ylabel, color=INK2)
    item.showGrid(x=True, y=True, alpha=0.25)
    if title:
        item.setTitle(title, color=INK, size="10pt")


def minmax_decimate(x, y, n_bins=2000):
    """Keep each bin's minimum and maximum (in time order): ~2 n_bins points that draw the same curve, close
    approaches included, for a fraction of the painting cost of tens of thousands of points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) <= 2 * n_bins:
        return x, y
    edges = np.linspace(0, len(x), n_bins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted((i, j)) if i != j else (i,):
            xs.append(x[k])
            ys.append(y[k])
    return np.array(xs), np.array(ys)


def cache_static(view):
    """Draw every item of a plot that does not move into a cached pixmap: a moving time cursor then repaints by
    copying pixmaps instead of re-stroking tens of thousands of antialiased segments (redone only on zoom)."""
    for item in view.getPlotItem().items + [view.getPlotItem().getAxis(n) for n in ("left", "bottom")]:
        if isinstance(item, pg.InfiniteLine):
            continue
        targets = [item]
        if isinstance(item, pg.PlotDataItem):
            targets = [item.curve, item.scatter]
        for t in targets:
            t.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)


def _legend(plot):
    leg = plot.addLegend(offset=(10, 10), labelTextColor=INK2)
    leg.setBrush(pg.mkBrush(20, 23, 27, 200))
    return leg


def _fmt_mass(m):
    if not np.isfinite(m):
        return "—"
    return f"{m / 1e3:,.1f} t" if m >= 1e4 else f"{m:,.0f} kg"


def _window_tip(o, s):
    """Tooltip of one window marker: what the DART as set (summary s) delivers there."""
    c = o.capability
    head = f"{t_to_utc_str(o.t_pass)[:7]} pass ({o.pass_dist / AU:.4f} au)"
    if c["status"] == "none":
        return (f"{head}\nNo arc within the max speed ({s['v_max']:.2f} km/s): raise the orbit-raising cut-off "
                f"for this window\n{LINEAR_NOTE}")
    live = s.get("live")
    beta = (f"β {s['beta_eval']:.2f} (90 % confidence of {s['beta']:.2f} ± {s['beta_sigma']:.2f})"
            if s["sizing"] == "p90" else f"β {s['beta']:.2f}" + (" (measured)" if live else " (planned)"))
    ld = LUNAR_DISTANCE
    shift = (f"Adds {c['shift'] / 1e3:,.0f} km: the DARTs reach {100 * c['fraction']:.0f} % of the "
             f"{s['target'] / 1e3:,.0f} km target" if live else
             f"Shift {c['shift'] / 1e3:,.0f} km = {100 * c['fraction']:.0f} % of the {s['target'] / 1e3:,.0f} km target")
    verdict = {"target": "● Reaches the target", "goal": "◆ Clears the goal but falls short of the target",
               "short": "▼ Falls short of the goal"}[c["status"]]
    return "\n".join([
        f"{head}: launch {t_to_utc_str(o.t_launch)[:10]}, impact {t_to_utc_str(o.t_arrive)[:10]} at "
        f"{o.impact_speed / 1e3:.1f} km/s",
        f"{'Next DART' if live else 'This DART'}: {_fmt_mass(s['mass'])} · {beta} · point of contact "
        f"{c['efficiency']:.2f}× head-on",
        f"{shift} → 2135 flyby {c['q'] / ld:.2f} LD (goal {s['goal'] / ld:.2f} LD)",
        f"{verdict} (a success needs ≥ {100 * (1 - s['tolerance']):.0f} %); chance over β ± {s['beta_sigma']:.2f}: "
        f"{100 * c['p_target']:.0f} %",
        f"Mass needed: {_fmt_mass(c['mass_target'])} for the target · {_fmt_mass(c['mass_success'])} for a success · "
        f"{_fmt_mass(c['mass_goal'])} for {s['goal'] / ld:.2f} LD",
        LINEAR_NOTE])


class DistancePlot(pg.PlotWidget):
    """Earth-Bennu distance on the scenario orbit (2026 to the Earth impact), the DART launch windows at Bennu's
    close passes and, after a live run's verification, the deflected orbit. Each window marker says what the DART
    as set (mass, max speed, beta, sizing) does there: reaches the target / clears the goal only / falls short /
    no arc (legend, tooltip with the shift, the 2135 flyby and the mass needed). Click a marker to select that
    window, elsewhere to jump in time. The markers, their legend and the "selected" label can be hidden."""

    clicked_time = QtCore.Signal(float)
    opportunity_clicked = QtCore.Signal(int)

    def __init__(self, eph, scenario):
        super().__init__()
        _style(self, "Year", "Earth–Bennu distance [au]", "Bennu's distance from Earth — scenario: keyhole in 2135, "
                                                         "impact in " + t_to_utc_str(scenario.t_impact)[:4])
        self.setLogMode(y=True)
        self.leg = self.addLegend(offset=(10, -2), labelTextColor=INK2, labelTextSize="8pt")
        self.leg.setBrush(pg.mkBrush(20, 23, 27, 200))
        lo, hi = eph.span("bennu")
        t = np.concatenate([np.linspace(lo, hi - 30 * DAY, 60_000), np.linspace(hi - 30 * DAY, hi, 4000)])
        d = np.linalg.norm(eph.rel_pos("bennu", "earth", t), axis=1) / AU
        self.plot(*minmax_decimate(t_to_year(t), np.maximum(d, 1e-5)), pen=pg.mkPen(BENNU_COLOR, width=1.5),
                  name="Bennu, scenario orbit (no deflection)")
        self.getPlotItem().setClipToView(True)
        self.defl = self.plot([], [], pen=pg.mkPen(GOOD, width=1.8), name="Deflected orbit (full-model physics)")
        # the gravity tractor: its launch window (translucent blue band over the launch dates that still reach Bennu
        # in time) and the planned flight's Earth distance, rising from Earth to meet Bennu's curve at the intercept
        self.tr_band = pg.LinearRegionItem(values=(0, 0), movable=False, brush=pg.mkBrush(57, 135, 229, 45),
                                           pen=pg.mkPen(TRACTOR_COLOR, width=1, style=QtCore.Qt.DotLine))
        self.tr_band.setZValue(-10)
        self.tr_band.setVisible(False)
        self.addItem(self.tr_band)
        self.tr_curve = self.plot([], [], pen=pg.mkPen(TRACTOR_COLOR, width=2.0), name=self.TRACTOR_NAME)
        self.leg.setColumnCount(3)  # one row along the bottom, below the window markers
        self.addLine(y=np.log10(LUNAR_DISTANCE / AU), pen=pg.mkPen(MUTED, width=1))
        txt = pg.TextItem("Moon's distance (goal)", color=INK2, anchor=(0, 1))
        txt.setPos(t_to_year(lo), np.log10(LUNAR_DISTANCE / AU))
        self.addItem(txt)
        self.addLine(y=np.log10(RADIUS["earth"] / AU), pen=pg.mkPen(CRITICAL, width=1, style=QtCore.Qt.DashLine))
        for tt, label in [(scenario.t_ca, "keyhole 2135"), (scenario.t_impact, "impact " + t_to_utc_str(scenario.t_impact)[:4])]:
            dd = np.linalg.norm(eph.rel_pos("bennu", "earth", min(tt, hi))) / AU
            self.plot([t_to_year(tt)], [max(dd, 1e-5)], pen=None, symbol="star", symbolBrush=CRITICAL,
                      symbolPen=pg.mkPen(SURFACE, width=1), symbolSize=14)
            lab = pg.TextItem(label, color=INK, anchor=(1.1, 0.5))
            lab.setPos(t_to_year(tt), np.log10(max(dd, 1e-5)))
            self.addItem(lab)
        self._tips = {}
        self.opps = pg.ScatterPlotItem(size=12, pen=pg.mkPen(SURFACE, width=1.5), hoverable=True,
                                       tip=lambda x, y, data: self._tips.get(data, ""))
        self.opps.sigClicked.connect(self._on_opp)
        self.addItem(self.opps)
        self.sel = pg.ScatterPlotItem(size=22, pen=pg.mkPen(INK, width=2), brush=pg.mkBrush(0, 0, 0, 0))
        self.addItem(self.sel)
        # the window markers' legend and the selected window: two lines under the title, outside the data area, so
        # they never cover a marker or the curve
        self._title = self.getPlotItem().titleLabel.text
        self._key_lines = []
        self._markers_on = True
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#00e5ff", width=1.5))
        self.addItem(self.cursor)
        self.scene().sigMouseClicked.connect(self._on_click)
        self._opp_clicked = False
        self._items = []
        cache_static(self)

    def set_opportunities(self, opps, selected, summary=None):
        """Window markers coloured by what the DART as set does there (opp.capability; summary: the DART and the
        thresholds, plus "live" for the next DART after a shortfall and the plan's "mass_source")."""
        style = {k: (col, sym) for k, col, sym, _, _ in WINDOW_STATUS}
        spots = []
        self._items = opps
        for o in opps:
            c = o.capability
            if summary is None or c is None:  # not evaluated (yet)
                col, sym = MUTED, "o" if o.feasible else "x"
                tip = f"{t_to_utc_str(o.t_pass)[:10]} pass ({o.pass_dist / AU:.3f} au)\n" + (o.note or "evaluating…")
            else:
                col, sym = style[c["status"]]
                tip = _window_tip(o, summary)
            spots.append({"pos": (t_to_year(o.t_pass), np.log10(o.pass_dist / AU)), "data": o.index,
                          "brush": pg.mkBrush(col), "symbol": sym})
            self._tips[o.index] = tip
        self.opps.setData(spots)
        sel = next((o for o in opps if o.index == selected), None)
        if sel is not None:
            self.sel.setData([t_to_year(sel.t_pass)], [np.log10(sel.pass_dist / AU)])
        else:
            self.sel.setData([], [])
        self._set_key(summary, None if sel is None else
                      f"selected: {t_to_utc_str(sel.t_pass)[:7]} — " + self._selected_text(sel, summary))

    @staticmethod
    def _selected_text(o, s):
        c = o.capability
        if not o.feasible:
            return "no arc within the max speed"
        if s is None or c is None:
            return f"~{_fmt_mass(o.mass_nominal)} head-on at {o.impact_speed / 1e3:.1f} km/s"
        word = {"target": "reaches the target", "goal": "clears the goal, short of the target",
                "short": "short of the goal"}[c["status"]]
        text = (f"{_fmt_mass(s['mass'])} at {o.impact_speed / 1e3:.1f} km/s → 2135 flyby {c['q'] / LUNAR_DISTANCE:.2f} LD, "
                f"{word} (linear)")
        if s.get("mass_source") == "full model":
            text += "; the mass is sized with the full force model, which puts it at the target"
        return text

    def _set_key(self, s, selected):
        """The legend strip: the DART the markers describe and the marker classes; the selected window and the
        linear caveat."""
        if s is None:
            dart = "DART launch windows"
        else:
            beta = f"β {s['beta_eval']:.2f}" + (" (90 % confidence)" if s["sizing"] == "p90" else
                                                " (measured)" if s.get("live") else "")
            dart = (f"{'Remaining windows for the next DART' if s.get('live') else 'Windows for this DART'} "
                    f"({_fmt_mass(s['mass'])} · ≤ {s['v_max']:.2f} km/s · {beta})")
        goal = f"≥ {s['goal_ld']:.2f} LD" if s else "≥ goal"
        classes = "&nbsp;&nbsp;&nbsp;".join(f"<span style='color:{col}'>{char}</span>&nbsp;{text.format(goal=goal)}"
                                            for _, col, _, char, text in WINDOW_STATUS)
        line2 = (f"<span style='color:{INK}'>○</span>&nbsp;{selected} · " if selected else "") + \
            f"<span style='color:{MUTED}'>{LINEAR_NOTE}</span>"
        self._key_lines = [f"{dart}:&nbsp;&nbsp; {classes}", line2]
        self._update_title()

    def _update_title(self):
        item = self.getPlotItem()
        lines = [self._title] + ([f"<span style='color:{INK2}; font-size:8.5pt'>{t}</span>" for t in self._key_lines]
                                 if self._markers_on else [])
        item.setTitle("<div align='center'>" + "<br>".join(lines) + "</div>")
        height = 30 + 17 * (len(lines) - 1)
        item.titleLabel.setMaximumHeight(height)
        item.layout.setRowFixedHeight(0, height)

    def set_markers_visible(self, on):
        """Show or hide the window markers, their legend and the "selected" label (display only: the selected
        window and the mission do not change; hidden markers take no clicks)."""
        self._markers_on = bool(on)
        self.opps.setVisible(self._markers_on)
        self.sel.setVisible(self._markers_on)
        self._update_title()

    def legend_shown(self):
        return self._markers_on and bool(self._key_lines)

    def set_deflected(self, t, d):
        if t is None:
            self.defl.setData([], [])
        else:
            self.defl.setData(*minmax_decimate(t_to_year(np.asarray(t)), np.maximum(np.asarray(d) / AU, 1e-5)))

    TRACTOR_NAME = "Gravity tractor (plan): launch window and flight"

    def set_tractor(self, window, path_t=None, path=None, eph=None, live=False):
        """The gravity tractor's launch window (core/planner.py tractor_launch_window: the blue band over the launch
        dates that still reach Bennu in time; its dates go in the legend entry, where they never cover a curve) and
        its planned flight (barycentric path samples with their times: Earth distance from the launch to the
        intercept). window None: no band (no plan yet, or the tractor has launched: its planned flight alone)."""
        self._tractor = (window, path_t, path, live)
        if path is not None and path_t is not None and eph is not None:
            d = np.linalg.norm(np.asarray(path) - eph.pos("earth", np.asarray(path_t)), axis=1) / AU
            self.tr_curve.setData(t_to_year(np.asarray(path_t)), np.maximum(d, 1e-5))
        else:
            self.tr_curve.setData([], [])
        band = window is not None and bool(np.any(window["ok"]))
        self.tr_band.setVisible(band)
        if band:
            t0, t1 = float(window["t"][window["ok"]].min()), float(window["t_last"])
            self.tr_band.setRegion((t_to_year(t0), t_to_year(t1)))
            what = "the tow closes the gap" if window.get("gap", 0.0) > 0 else "the tow holds the aim"
            name = f"Gravity tractor: launch window {t_to_utc_str(t0)[:7]} → {t_to_utc_str(t1)[:7]} ({what}), planned flight"
        elif window is not None:
            name = "Gravity tractor: no launch reaches Bennu in time to close the gap"
        elif self.tr_curve.xData is not None and len(self.tr_curve.xData):
            name = "Gravity tractor: planned flight"
        else:
            name = self.TRACTOR_NAME
        lab = self.leg.getLabel(self.tr_curve)
        if lab is not None and lab.text != name:
            lab.setText(name)
            self.leg.updateSize()

    def set_time(self, t):
        self.cursor.setPos(t_to_year(t))

    def _on_opp(self, item, points, ev=None):
        if len(points):
            self._opp_clicked = True
            self.opportunity_clicked.emit(int(points[0].data()))

    def _on_click(self, ev):
        if self._opp_clicked:
            self._opp_clicked = False
            return
        x = self.getPlotItem().vb.mapSceneToView(ev.scenePos()).x()
        self.clicked_time.emit(((x - 2000.0) * 365.25 - 0.5) * DAY)


class BPlanePlot(pg.PlotWidget):
    """The 2135 keyhole flyby in its B-plane (thousands of km): Earth, the Moon's distance and the goal circle,
    the keyhole point (no deflection), the DART target, the planned shifts and the beta uncertainty cloud."""

    def __init__(self):
        super().__init__()
        _style(self, "ξ [1000 km]", "ζ (timing axis) [1000 km]", "B-plane of the 2135 keyhole flyby (closest-approach plane)")
        self.setAspectLocked(True)
        self.leg = _legend(self)
        self.earth = pg.PlotDataItem(pen=pg.mkPen(MUTED, width=1), name="Earth (to scale)")
        self.moon = pg.PlotDataItem(pen=pg.mkPen(INK2, width=1, style=QtCore.Qt.DashLine), name="Moon's distance")
        self.goal = pg.PlotDataItem(pen=pg.mkPen(CRITICAL, width=1.5), name="Goal")
        self.keyhole = pg.ScatterPlotItem(size=14, symbol="star", pen=pg.mkPen(SURFACE, width=1), brush=pg.mkBrush(CRITICAL),
                                          name="Keyhole flyby (no deflection)")
        self.cloud = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(57, 135, 229, 110), name="β uncertainty")
        self.ki = pg.PlotDataItem(pen=pg.mkPen(S2, width=2), name="DART shift")
        self.gt = pg.PlotDataItem(pen=pg.mkPen(S3, width=2), name="Gravity-tractor trim")
        self.total = pg.ScatterPlotItem(size=12, pen=pg.mkPen(SURFACE, width=2), brush=pg.mkBrush(INK), name="Planned flyby")
        self.aim = pg.ScatterPlotItem(size=16, symbol="+", pen=pg.mkPen(INK, width=2), brush=None, name="DART target (aim)")
        self.live_pred = pg.PlotDataItem(pen=pg.mkPen(S4, width=1.5), symbol="o", symbolSize=4, symbolBrush=S4,
                                         symbolPen=None, name="Predicted final point (weekly)")
        self.live_now = pg.ScatterPlotItem(size=11, pen=pg.mkPen(SURFACE, width=2), brush=pg.mkBrush(S3),
                                           name="Flyby with the pushes so far")
        self.verified = pg.ScatterPlotItem(size=13, symbol="d", pen=pg.mkPen(SURFACE, width=1), brush=pg.mkBrush(GOOD),
                                           name="Full-model physics")
        for it in (self.earth, self.moon, self.goal, self.keyhole, self.cloud, self.ki, self.gt, self.total, self.aim,
                   self.live_pred, self.live_now, self.verified):
            self.addItem(it)
        self._ranged = False

    def update_plan(self, res):
        th = np.linspace(0, 2 * np.pi, 240)
        k = 1e-6  # m -> 1000 km
        c = np.column_stack([np.cos(th), np.sin(th)])
        self.earth.setData(*(RADIUS["earth"] * k * c).T)
        self.moon.setData(*(LUNAR_DISTANCE * k * c).T)
        self.goal.setData(*(res.b_goal * k * c).T)
        b0 = res.b0 * k
        self.keyhole.setData([b0[0]], [b0[1]])
        self.cloud.setData(res.b_samples[:, 0] * k, res.b_samples[:, 1] * k)
        bk = b0 + res.db_ki * k
        self.ki.setData([b0[0], bk[0]], [b0[1], bk[1]])
        self.gt.setData([bk[0], bk[0] + res.db_gt[0] * k], [bk[1], bk[1] + res.db_gt[1] * k])
        self.total.setData([res.b_final[0] * k], [res.b_final[1] * k])
        self.aim.setData([res.b_aim[0] * k], [res.b_aim[1] * k])
        if not self._ranged:
            lim = 1.2 * max(res.b_goal * k, np.abs(res.b_samples).max() * k, np.linalg.norm(res.b_aim) * k)
            self.setRange(xRange=(-lim, lim), yRange=(-lim, lim), padding=0)

    def update_live(self, out):
        k = 1e-6
        hist = out.get("trim_history") or []
        if hist:
            P = np.array([q for _, _, q in hist]) * k
            self.live_pred.setData(P[:, 0], P[:, 1])
        if out.get("b_now") is not None:
            self.live_now.setData([out["b_now"][0] * k], [out["b_now"][1] * k])
        fc = out.get("final_check")
        if fc is not None and "b_nonlinear" in fc:
            self.verified.setData([fc["b_nonlinear"][0] * k], [fc["b_nonlinear"][1] * k])
        if out.get("b_aim") is not None:
            self.aim.setData([out["b_aim"][0] * k], [out["b_aim"][1] * k])
        if out.get("b_final") is not None:
            self.total.setData([out["b_final"][0] * k], [out["b_final"][1] * k])

    def clear_live(self):
        for it in (self.live_pred, self.live_now, self.verified):
            it.setData([], [])


class PorkchopPlot(pg.PlotWidget):
    """The selected opportunity: launch date x flight time, coloured by the 2135 shift each arc gives."""

    def __init__(self):
        super().__init__()
        _style(self, "Launch date [year]", "Flight time [days]",
               "DART arcs of the selected opportunity within the departure-speed limit, coloured by deflection")
        self.img = pg.ImageItem()
        self.img.setLookupTable(pg.ColorMap(np.linspace(0, 1, len(BLUE_RAMP_DARK)), BLUE_RAMP_DARK).getLookupTable(nPts=256))
        self.addItem(self.img)
        self.leg = _legend(self)
        self.choice = pg.ScatterPlotItem(size=14, pen=pg.mkPen(INK, width=2), brush=pg.mkBrush(0, 0, 0, 0),
                                         name="Chosen arc")
        self.addItem(self.choice)
        self.scale_text = pg.TextItem("", color=INK2, anchor=(1, 0))
        self.addItem(self.scale_text)

    def update_plan(self, res):
        ip = res.impactor
        pc = ip.porkchop if ip.feasible else None
        if pc is None:
            self.img.clear()
            self.choice.setData([], [])
            return
        shift = pc["shift"] / 1e6  # 1000 km
        x0, x1 = t_to_year(pc["launch"][0]), t_to_year(pc["launch"][-1] + 4 * DAY)
        y0, y1 = pc["tof"][0] / DAY, pc["tof"][-1] / DAY
        vmax = max(float(shift.max()), 1e-9)
        self.img.setImage(shift, levels=(0, vmax))
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.choice.setData([t_to_year(ip.t_launch)], [ip.tof_days])
        self.scale_text.setText(f"brightest = {vmax:,.0f} thousand km for {ip.mass:,.0f} kg (head-on)")
        self.scale_text.setPos(x1, y1)
        self.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)


class EtaPlot(pg.PlotWidget):
    """Tractor arrival date as a function of thrust for a launch right after the DART result (unlimited fuel)."""

    def __init__(self):
        super().__init__()
        _style(self, "Tractor thrust [N] (log scale)", "Arrival at Bennu [year]",
               "More thrust -> earlier arrival (10 t tractor launched after the DART result, unlimited fuel)")
        self.setLogMode(x=True)
        self.leg = _legend(self)
        self.curve = self.plot([], [], pen=pg.mkPen(S1, width=2), symbol="o", symbolSize=6, symbolBrush=S1,
                               symbolPen=pg.mkPen(SURFACE, width=2), name="Earliest arrival")
        self.current = self.plot([], [], pen=None, symbol="o", symbolSize=14, symbolBrush=S2,
                                 symbolPen=pg.mkPen(SURFACE, width=2), name="Current setting")
        self.status = pg.TextItem("computing…", color=INK2, anchor=(0, 0))
        self.addItem(self.status)
        ticks = [(np.log10(v), f"{v:g} N") for v in (0.3, 1, 3, 10, 30, 100, 300, 1000, 3000)]
        self.getPlotItem().getAxis("bottom").setTicks([ticks, []])

    def update_curve(self, curve, current_thrust, current_arrival):
        T, t_arr = curve[:, 0], curve[:, 1]
        ok = np.isfinite(t_arr)
        self.curve.setData(T[ok], t_to_year(t_arr[ok]))
        self.status.setText("" if ok.any() else "no feasible thrust in range")
        if np.isfinite(current_arrival):
            self.current.setData([current_thrust], [t_to_year(current_arrival)])
        else:
            self.current.setData([], [])
        if ok.any():
            self.status.setPos(np.log10(T[ok][0]), t_to_year(t_arr[ok]).max())


def decade_ticks(axis, lo, hi):
    """Label only powers of ten on a log axis (pyqtgraph's minor labels collide)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
        return
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    axis.setTicks([[(e, f"1e{e}") for e in range(e0, e1 + 1)], []])


class LivePlots(pg.GraphicsLayoutWidget):
    """Three small multiples during a live run: ETA history, tractor range to Bennu, navigation error."""

    def __init__(self):
        super().__init__()
        self.setBackground(SURFACE)
        self.eta = self.addPlot(title="<span style='color:#f0efec'>Arrival date as the flight progresses</span>")
        self.rng = self.addPlot(title="<span style='color:#f0efec'>Tractor distance to Bennu [km]</span>")
        self.nav = self.addPlot(title="<span style='color:#f0efec'>Navigation error vs 3σ [km]</span>")
        for plot, xl, yl in [(self.eta, "Mission date [year]", "Planned arrival [year]"),
                             (self.rng, "Mission date [year]", "km"), (self.nav, "Mission date [year]", "km")]:
            for name in ("left", "bottom"):
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(AXIS))
                ax.setTextPen(pg.mkPen(INK2))
            plot.setLabel("bottom", xl, color=INK2)
            plot.setLabel("left", yl, color=INK2)
            plot.showGrid(x=True, y=True, alpha=0.25)
        self.rng.setLogMode(y=True)
        self.nav.setLogMode(y=True)
        self.eta_curve = self.eta.plot(pen=pg.mkPen(S1, width=2), symbol="o", symbolSize=5, symbolBrush=S1,
                                       symbolPen=pg.mkPen(SURFACE, width=1), stepMode=None)
        self.storms = {}
        self.rng_curve = self.rng.plot(pen=pg.mkPen(S1, width=2))
        leg = self.nav.addLegend(offset=(10, 10), labelTextColor=INK2)
        self.err_curve = self.nav.plot(pen=pg.mkPen(S1, width=1.5), name="Estimate error")
        self.sig_curve = self.nav.plot(pen=pg.mkPen(S2, width=2), name="3σ bound")
        self.reset()

    def reset(self):
        self.t, self.range_, self.err, self.sig = [], [], [], []
        self.eta_curve.setData([], [])
        self.rng_curve.setData([], [])
        self.err_curve.setData([], [])
        self.sig_curve.setData([], [])
        for it in self.storms.values():
            self.eta.removeItem(it)
        self.storms = {}

    def add_snapshot(self, s):
        tr = s["tractor"]
        if tr is None:
            return
        y = t_to_year(s["t"])
        if "range" in tr and (not self.t or y - self.t[-1] > 0.002):
            self.t.append(y)
            self.range_.append(tr["range"] / 1e3)
            self.err.append(max(tr.get("nav_err", np.nan) / 1e3, 1e-3))
            self.sig.append(3 * tr.get("nav_sigma", np.nan) / 1e3)
            self.rng_curve.setData(self.t, self.range_)
            self.err_curve.setData(self.t, self.err)
            self.sig_curve.setData(self.t, self.sig)
            decade_ticks(self.rng.getAxis("left"), min(self.range_), max(self.range_))
            decade_ticks(self.nav.getAxis("left"), min(min(self.err), np.nanmin(self.sig)), max(max(self.err), np.nanmax(self.sig)))
        hist = [(a, b) for a, b in s["eta_history"] if np.isfinite(b)]
        if hist:
            ys = t_to_year(np.array([b for _, b in hist]))
            self.eta_curve.setData(t_to_year(np.array([a for a, _ in hist])), ys)
            mid, span = 0.5 * (ys.min() + ys.max()), max(ys.max() - ys.min(), 0.25)  # don't magnify a 2-day change
            self.eta.setYRange(mid - 0.6 * span, mid + 0.6 * span, padding=0)
        if s["t"] < s["safe_until"]:  # shade each storm (identified by its end time) on the ETA plot
            key = round(s["safe_until"])
            if key not in self.storms:
                region = pg.LinearRegionItem(values=(y, t_to_year(s["safe_until"])), movable=False,
                                             brush=pg.mkBrush(250, 178, 25, 60), pen=pg.mkPen(None))
                self.eta.addItem(region)
                self.storms[key] = region


class ProxPlots(pg.GraphicsLayoutWidget):
    """Proximity operations (Step 5): distance to the surface, navigation error vs 3σ, thrust vs limit."""

    def __init__(self):
        super().__init__()
        self.setBackground(SURFACE)
        self.alt = self.addPlot(title="<span style='color:#f0efec'>Altitude above Bennu's surface [m]</span>")
        self.nav = self.addPlot(title="<span style='color:#f0efec'>Navigation error vs 3σ [m] (LIDAR + camera EKF)</span>")
        self.thr = self.addPlot(title="<span style='color:#f0efec'>Thrust [N]</span>")
        for plot, yl in [(self.alt, "m"), (self.nav, "m"), (self.thr, "N")]:
            for name in ("left", "bottom"):
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(AXIS))
                ax.setTextPen(pg.mkPen(INK2))
            plot.setLabel("bottom", "Hours since arrival at Bennu", color=INK2)
            plot.setLabel("left", yl, color=INK2)
            plot.showGrid(x=True, y=True, alpha=0.25)
        self.alt.setLogMode(y=True)
        self.nav.setLogMode(y=True)
        self.alt_curve = self.alt.plot(pen=pg.mkPen(S1, width=2))
        self.nav.addLegend(offset=(-10, -10), labelTextColor=INK2)
        self.err_curve = self.nav.plot(pen=pg.mkPen(S1, width=1.5), name="Estimate error")
        self.sig_curve = self.nav.plot(pen=pg.mkPen(S2, width=2), name="3σ bound")
        self.thr.addLegend(offset=(-10, 10), labelTextColor=INK2)
        self.thr_curve = self.thr.plot(pen=pg.mkPen(S1, width=1.5), name="Commanded")
        self.max_curve = self.thr.plot(pen=pg.mkPen(S2, width=2, style=QtCore.Qt.DashLine), name="Available")
        self.reset()

    def reset(self):
        self.t0 = None
        self.data = {k: [] for k in ("t", "alt", "nav_err", "nav_sigma", "thrust", "a_max")}
        for c in (self.alt_curve, self.err_curve, self.sig_curve, self.thr_curve, self.max_curve):
            c.setData([], [])

    def add_snapshot(self, s):
        pr = (s.get("tractor") or {}).get("prox")
        if pr is None:
            return
        h = pr["hist"]
        if len(h["t"]) == 0:
            return
        if self.t0 is None:
            self.t0 = float(h["t"][0])
        last = self.data["t"][-1] if self.data["t"] else -np.inf
        new = h["t"] > last
        if not new.any():
            return
        for k in self.data:
            self.data[k].extend(np.asarray(h[k])[new].tolist())
        D = {k: np.asarray(v) for k, v in self.data.items()}
        x = (D["t"] - self.t0) / 3600.0
        self.alt_curve.setData(x, np.maximum(D["alt"], 0.1))
        self.err_curve.setData(x, np.maximum(D["nav_err"], 1e-3))
        self.sig_curve.setData(x, 3 * D["nav_sigma"])
        self.thr_curve.setData(x, D["thrust"])
        self.max_curve.setData(x, D["a_max"])
        decade_ticks(self.alt.getAxis("left"), max(D["alt"].min(), 0.1), D["alt"].max())
        near = pr["phase"] in ("survey", "descent", "capture", "ascent", "transfer", "station")
        for plot in (self.alt, self.nav, self.thr):  # near the surface: follow the last 12 hours
            if near and x[-1] > 12.0:
                plot.setXRange(x[-1] - 12.0, x[-1], padding=0.01)
                plot.setAutoVisible(y=True)
            else:
                plot.enableAutoRange(axis="x")
        lo = min(np.maximum(D["nav_err"], 1e-3).min(), (3 * D["nav_sigma"]).min())
        decade_ticks(self.nav.getAxis("left"), lo, max(D["nav_err"].max(), (3 * D["nav_sigma"]).max()))


class SunPlot(pg.GraphicsLayoutWidget):
    """The Sun's center around the Solar System barycenter: path (ecliptic plane, solar radii) and distance."""

    def __init__(self):
        super().__init__()
        self.setBackground(SURFACE)
        self.path = self.addPlot(title="<span style='color:#f0efec'>Sun's center seen from the ecliptic pole "
                                       "[solar radii]</span>")
        self.dist = self.addPlot(title="<span style='color:#f0efec'>Distance of the Sun's center from the "
                                       "barycenter [solar radii]</span>")
        for plot, (xl, yl) in [(self.path, ("x", "y")), (self.dist, ("Year", "solar radii"))]:
            for name in ("left", "bottom"):
                ax = plot.getAxis(name)
                ax.setPen(pg.mkPen(AXIS))
                ax.setTextPen(pg.mkPen(INK2))
            plot.setLabel("bottom", xl, color=INK2)
            plot.setLabel("left", yl, color=INK2)
            plot.showGrid(x=True, y=True, alpha=0.25)
        self.path.setAspectLocked(True)
        self.path.addLegend(offset=(10, 10), labelTextColor=INK2)
        self.dist.addLegend(offset=(-10, 10), labelTextColor=INK2)
        self.jpl_curve = self.path.plot(pen=pg.mkPen(S1, width=1.5), name="JPL DE441")
        self.nb_curve = self.path.plot(pen=pg.mkPen(S2, width=1.5, style=QtCore.Qt.DashLine),
                                       name="our N-body (Sun + 9 planets)")
        th = np.linspace(0, 2 * np.pi, 120)
        self._circle = np.column_stack([np.cos(th), np.sin(th)])
        self.disk = self.path.plot(pen=pg.mkPen("#ffcc66", width=2))
        self.center = self.path.plot(pen=None, symbol="o", symbolSize=8, symbolBrush="#ffcc66")
        self.ssb = self.path.plot([0], [0], pen=None, symbol="+", symbolSize=14, symbolPen=pg.mkPen("#ffffff", width=2))
        self.d_jpl = self.dist.plot(pen=pg.mkPen(S1, width=1.5), name="all planets (JPL)")
        self.d_js = self.dist.plot(pen=pg.mkPen(S3, width=1.5), name="Jupiter + Saturn only")
        self.d_j = self.dist.plot(pen=pg.mkPen(S4, width=1.2), name="Jupiter only")
        self.now = pg.InfiniteLine(angle=90, pen=pg.mkPen(INK2, width=1, style=QtCore.Qt.DashLine))
        self.dist.addItem(self.now)
        for c in (self.d_jpl, self.d_js, self.d_j, self.jpl_curve, self.nb_curve):  # static once loaded
            c.curve.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.data = None

    def set_data(self, d):
        """d: years, xy_jpl, xy_nbody (solar radii), dist_jpl, dist_js, dist_j, times, xy_now function."""
        self.data = d
        self.jpl_curve.setData(d["xy_jpl"][:, 0], d["xy_jpl"][:, 1])
        if d.get("xy_nbody") is not None:
            self.nb_curve.setData(d["xy_nbody"][:, 0], d["xy_nbody"][:, 1])
        self.d_jpl.setData(d["years"], d["dist_jpl"])
        self.d_js.setData(d["years"], d["dist_js"])
        self.d_j.setData(d["years"], d["dist_j"])

    def set_time(self, t):
        if self.data is None:
            return
        y = t_to_year(t)
        self.now.setPos(y)
        xy = np.array([np.interp(y, self.data["years"], self.data["xy_jpl"][:, k]) for k in range(2)])
        self.center.setData([xy[0]], [xy[1]])
        self.disk.setData(xy[0] + self._circle[:, 0], xy[1] + self._circle[:, 1])  # the Sun's disk, to scale
