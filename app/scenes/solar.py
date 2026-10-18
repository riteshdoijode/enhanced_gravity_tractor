"""Solar System view (center of the window, always): AU, ecliptic J2000 axes, origin = Solar System Barycenter.

The Sun, the eight planets and Pluto at their JPL positions (DE441). An orbit longer than the ephemerides (Neptune
165 yr, Pluto 248 yr, against 2026-2185) is drawn from them as far as they go and closed with a two-body arc from
their last state; the bodies themselves are always on the ephemeris part. 🪐 zooms out to the whole system.

The Sun moves around the barycenter (its offset can be drawn exaggerated). Bennu follows the scenario orbit
(through the 2135 keyhole flyby to the Earth impact), or the deflected orbit during a live run.
Galactic side view (the 🏠 button): the Solar System seen from the side as it moves around the Milky Way (camera
toward the Galactic center, north Galactic pole up, the motion to the left), with the paths the Sun and planets
trace in that motion: their barycentric positions (JPL ephemerides) plus the barycenter's Galactic velocity
(core/galaxy.py) times the time since. The motion can be slowed for display so the planets' helices show.
"""

import numpy as np
from pyvista.examples import planets

from app.scenes.base import EQ_TO_ECL, LIVE_COLORS, SceneBase, line, pose
from core.constants import AU, DAY, GM, YEAR

PLANETS = {  # name: (marker size px, color, orbital period days)
    "mercury": (11, "#9c8f86", 87.97),
    "venus": (11, "#e3c48d", 224.70),
    "earth": (14, None, 365.256),
    "mars_bary": (11, "#c1440e", 686.98),
    "jupiter_bary": (13, "#c9a27a", 4332.6),
    "saturn_bary": (12, "#e3d19c", 10759.2),
    "uranus_bary": (11, "#9fd8e0", 30688.5),
    "neptune_bary": (11, "#5b7fe8", 60182.0),
    "pluto_bary": (8, "#c8b29a", 90560.0),  # the dwarf planet (Pluto-Charon barycenter)
}
NAMES = {"mars_bary": "Mars", "jupiter_bary": "Jupiter", "saturn_bary": "Saturn", "uranus_bary": "Uranus",
         "neptune_bary": "Neptune", "pluto_bary": "Pluto"}
BENNU_PERIOD_D = 436.65
TRACTOR_COLOR = "#3987e5"  # the tractor's plan: the same blue as its launch window on the distance plot
CAMERA_START = (0.3, -3.6, 2.5)  # AU from the barycenter, ecliptic north up
CAMERA_WIDE = (8.0, -150.0, 95.0)  # 🪐: the whole system, out to Pluto's orbit (aphelion 49 AU)
GALACTIC_SLOW = 50.0  # display option: the Galactic motion slowed x1/50 so the planets' helices show
PATH_YEARS = 4332.6 / 365.25  # traced paths: one orbit of Jupiter behind and ahead
PATH_POINTS = 1500  # per path (~30 per orbit of Mercury)
SIDE_DISTANCE = 11.0  # AU: the side view frames the inner planets (scroll out for Jupiter)


def ecl(P):
    return (EQ_TO_ECL @ np.atleast_2d(P).T).T / AU


class SolarScene(SceneBase):
    SKY_FRAME = EQ_TO_ECL  # the scene is in ecliptic axes
    SKY_FULL = True  # the main view: the 8k Milky Way

    def __init__(self, plotter, eph):
        super().__init__(plotter)
        import pyvista as pv

        self.pv = pv
        self.eph = eph
        self.plan = {}  # name -> (times, barycentric positions, color, label)
        self.sun_wobble = 1.0
        self._caption = None  # the side view's caption, until the camera is moved
        self.galactic = None  # core/galaxy.solar_motion() while the Galactic side view is on
        self.galactic_slow = 1.0  # display slow-down of the Galactic motion (1 = true speed)
        self._paths_t = None
        self.build()
        for event in ("LeftButtonPressEvent", "MiddleButtonPressEvent", "RightButtonPressEvent",
                      "MouseWheelForwardEvent", "MouseWheelBackwardEvent"):
            self.pl.iren.add_observer(event, lambda *_: self._set_caption(None))

    def build(self):
        pv = self.pv
        self.reset_scene()
        light = pv.Light(position=(0, 0, 0), focal_point=(1, 0, 0), light_type="scene light", intensity=1.4)
        light.positional = True
        light.cone_angle = 180.0  # point light at the Sun
        self.pl.add_light(light)
        self._sun_light_actor = light
        self._a["sun"] = self.pl.add_mesh(planets.load_planet(radius=0.045), texture=self.tex["sun"], ambient=1.0,
                                          diffuse=0.0, specular=0.0, name="sun", render=False)
        self.marker("ssb", "#ffffff", 7)
        self._sun_trail_t = None
        t0 = self.eph.span("earth")[0]
        for name, (size, color, period) in PLANETS.items():
            self.pl.add_mesh(line(self._orbit(name, period, t0)), color=color or "#4f8fd9", opacity=0.45,
                             line_width=1.5, name=f"orbit_{name}", render=False)
            self.marker(name, color or "#4f8fd9", size)
        self.marker("bennu", "#ffb347", 11)
        self.labels("labels", ["Sun", *(NAMES.get(n, n.title()) for n in PLANETS), "Bennu"])
        self.labels("ssb_label", ["barycenter" + (f" (Sun's wobble ×{self.sun_wobble:g})" if self.sun_wobble > 1 else "")],
                    color="#c3c2b7", size=11)
        self._bennu_orbit_t = None
        self._add_plan()
        if self.live is not None:
            self._update_live()
        if self.galactic is not None:
            self._add_galactic()
        self.reset_camera()

    def _orbit(self, name, period_days, t0):
        """One orbit (scene AU) from the ephemerides; when it is longer than they are, closed with a two-body arc
        (Sun + planet) flown on from their last state."""
        from core.guidance.kepler import propagate

        lo, hi = self.eph.span(name)
        lo = max(lo, t0)
        P = period_days * DAY
        if lo + P <= hi:
            return ecl(self.eph.pos(name, lo + np.linspace(0.0, P, 400)))
        arc = self.eph.pos(name, np.linspace(lo, hi, 400))
        x = np.atleast_2d(self.eph.state(name, hi))[0] - np.atleast_2d(self.eph.state("sun", hi))[0]
        r, _ = propagate(x[:3], x[3:], np.linspace(0.0, lo + P - hi, 200), mu=GM["sun"] + GM[name])
        return ecl(np.vstack([arc, r + self.eph.pos("sun", hi)]))

    def reset_camera(self, wide=False):
        c = self.pl.camera
        c.position, c.focal_point, c.up = (CAMERA_WIDE if wide else CAMERA_START), (0, 0, 0), (0, 0, 1)
        self.pl.reset_camera_clipping_range()

    # ------------------------------------------------------------------ Galactic side view
    def set_galactic(self, motion):
        """motion (core/galaxy.solar_motion): turn the Galactic side view on, with the traced paths; None: off (the
        caller resets the camera)."""
        self.galactic = motion
        self._paths_t = None
        if motion is None:
            for key in [k for k in self._a if k.startswith("gal_")]:
                self._a[key].SetVisibility(False)
            for key in [k for k in self._labels if k.startswith("gal_")]:
                self._labels[key].actor.SetVisibility(False)
            self._set_caption(None)
            return
        self._add_galactic()
        self.side_view()
        if self.t is not None:
            self._update_paths(self.t)

    def set_galactic_slow(self, factor):
        self.galactic_slow = float(factor)
        self._paths_t = None
        if self.galactic is not None and self.t is not None:
            self._update_paths(self.t)
            self._set_caption(self._side_caption())

    def _axes(self):
        """The side view's axes in scene coordinates: motion (screen left), Galactic north (up), view (toward the
        Galactic center)."""
        m = self.galactic
        fwd = EQ_TO_ECL @ m["forward"]
        up = EQ_TO_ECL @ m["up"]
        up = up - (up @ fwd) * fwd
        up /= np.linalg.norm(up)
        view = np.cross(up, fwd)
        return fwd, up, view if view @ (EQ_TO_ECL @ m["center"]) > 0 else -view

    def side_view(self):
        """The camera beside the Solar System, looking toward the Galactic center with the north Galactic pole up:
        the motion runs to the left and the traced paths stream to the right."""
        fwd, up, view = self._axes()
        focal = -1.2 * fwd  # the Solar System left of center, its paths on the right
        c = self.pl.camera
        c.position, c.focal_point, c.up = tuple(focal - SIDE_DISTANCE * view), tuple(focal), tuple(up)
        self.pl.reset_camera_clipping_range()
        self._set_caption(self._side_caption())

    def _side_caption(self):
        m = self.galactic
        k = self.galactic_slow
        lo = self.eph.span("earth")[0]
        past = min(PATH_YEARS, (self.t - lo) / YEAR) if self.t is not None else PATH_YEARS
        au_yr = m["speed_kms"] * 1e3 * YEAR / AU
        speed = (f"{m['speed_kms']:.0f} km/s ({au_yr:.0f} AU per year) toward {m['constellation']}" if k == 1 else
                 f"{m['speed_kms']:.0f} km/s, drawn slowed ×1/{k:g} so the helices show")
        return (f"Side view: the Solar System moves left around the Milky Way\n{speed}\n"
                f"Up: north Galactic pole · Galactic center behind · orbits tilted {m['ecliptic_tilt_deg']:.0f}°\n"
                f"Paths: solid, the last {past:.1f} years; faint, the next {PATH_YEARS:.1f} years")

    def _add_galactic(self):
        """The Galactic plane (parallel, through the barycenter) and an arrow along the motion."""
        pv = self.pv
        fwd, up, view = self._axes()
        L = 40.0
        corners = np.array([L * (fwd + view), L * (fwd - view), L * (-fwd - view), L * (-fwd + view)])
        for key in ("gal_plane", "gal_plane_edge", "gal_arrow"):
            if key in self._a:
                self.pl.remove_actor(self._a.pop(key), render=False)
        self._a["gal_plane"] = self.pl.add_mesh(pv.PolyData(corners, faces=[4, 0, 1, 2, 3]), color="#7aa7e0",
                                                opacity=0.05, name="gal_plane", render=False)
        self._a["gal_plane_edge"] = self.pl.add_mesh(line(np.vstack([corners, corners[:1]])), color="#7aa7e0",
                                                     opacity=0.5, line_width=1.0, name="gal_plane_edge", render=False)
        start = -2.2 * up - 1.4 * fwd
        self._a["gal_arrow"] = self.pl.add_mesh(pv.Arrow(start=start, direction=fwd, scale=1.6, tip_length=0.2,
                                                         tip_radius=0.05, shaft_radius=0.015),
                                                color="#e6e6e6", ambient=1.0, name="gal_arrow", render=False)
        m = self.galactic
        self.labels("gal_labels", ["parallel to the Galactic plane",
                                   f"motion around the Milky Way: {m['speed_kms']:.0f} km/s"], color="#c3c2b7", size=11)
        self.set_label_points("gal_labels", [-2.6 * fwd + 0.08 * up, start + 1.6 * fwd + 0.15 * up])
        self._labels["gal_labels"].actor.SetVisibility(True)

    def _update_paths(self, t):
        """The paths of the Sun and the planets in the Galactic motion, behind (solid) and ahead (faint) of now:
        position(t') + v (t' - t), with the barycenter at the origin now."""
        if self._paths_t is not None and abs(t - self._paths_t) < DAY:
            return
        self._paths_t = t
        fwd, _, _ = self._axes()
        v = self.galactic["speed_kms"] * 1e3 / AU / self.galactic_slow * fwd  # AU per second, scene axes
        lo, hi = self.eph.span("earth")
        bodies = [("sun", "#ffcc66")] + [(n, c or "#4f8fd9") for n, (_, c, _) in PLANETS.items()]
        for part, (a, b), opacity in (("past", (max(lo, t - PATH_YEARS * YEAR), t), 0.9),
                                      ("next", (t, min(hi, t + PATH_YEARS * YEAR)), 0.3)):
            tt = np.linspace(a, b, PATH_POINTS)
            for name, color in bodies:
                key = f"gal_path_{part}_{name}"
                if b - a < DAY:
                    self.hide(key)
                    continue
                p = ecl(self.eph.pos(name, tt)) * (self.sun_wobble if name == "sun" else 1.0)
                if key in self._a:
                    self._a[key].SetVisibility(True)
                self.update_line(key, p + (tt - t)[:, None] * v, color=color, opacity=opacity,
                                 line_width=2.0 if name == "sun" else 1.5)

    def _set_caption(self, text):
        if text != self._caption:
            self._caption = text
            self.text("galactic", text, size=8, color="#c3c2b7", backing=0.7)

    def set_sun_wobble(self, scale):
        self.sun_wobble = float(scale)
        cam, caption = self.pl.camera_position, self._caption
        self._caption = None
        self._paths_t = None
        self.build()
        self.update(self.t)
        self.pl.camera_position = cam
        self._set_caption(caption)

    # ------------------------------------------------------------------ plan
    def set_plan(self, res):
        plan = {}
        if res is not None and res.tractor.feasible and res.tractor.path is not None:
            plan["tractor"] = (res.tractor.path_t, res.tractor.path, TRACTOR_COLOR, "Gravity tractor (plan)")
        if res is not None and res.impactor.feasible and res.impactor.path is not None:
            plan["impactor"] = (res.impactor.path_t, res.impactor.path, "#d95926", f"DART #{res.impactor.number} (plan)")
        self.plan = plan
        self._add_plan()
        if self.t is not None:
            self._update_plan(self.t)

    def _add_plan(self):
        for name in ("tractor", "impactor"):
            for key in (f"plan_line_{name}", f"plan_{name}"):
                if key in self._a:
                    self.pl.remove_actor(self._a.pop(key), render=False)
                    self._d.pop(key, None)
        for name, (times, path, color, _) in self.plan.items():
            live = self.live is not None  # during a live run the plan is only a faint reference
            self._a[f"plan_line_{name}"] = self.pl.add_mesh(line(ecl(path)), color=color, line_width=1.2 if live else 2.5,
                                                            opacity=0.25 if live else 1.0, name=f"plan_line_{name}",
                                                            render=False)
            self.marker(f"plan_{name}", color, 12)

    def _update_plan(self, t):
        names, pts = [], []
        for name, (times, path, color, label) in self.plan.items():
            actor = self._a.get(f"plan_{name}")
            if actor is None:
                continue
            inside = bool(times[0] <= t <= times[-1]) and self.live is None
            actor.SetVisibility(bool(inside))
            if inside:
                p = ecl(np.array([np.interp(t, times, path[:, i]) for i in range(3)]))[0]
                self.move_marker(f"plan_{name}", p)
                names.append(label)
                pts.append(p + [0, 0, -0.06])
        if pts:
            self._d["plan_labels"] = (names, "#c3c2b7", 12)
            self.set_label_points("plan_labels", pts)
        else:
            self.hide("plan_labels")

    # ------------------------------------------------------------------ time
    def update(self, t):
        self.t = float(t)
        pos = {n: EQ_TO_ECL @ self.eph.pos(n, t) / AU for n in [*PLANETS, "sun"]}
        b = self._bennu_pos(t)
        pos["bennu"] = EQ_TO_ECL @ b / AU
        for n in [*PLANETS, "bennu"]:
            self.move_marker(n, pos[n])
        sun = pos["sun"] * self.sun_wobble
        self._a["sun"].user_matrix = pose(p=sun)
        self._sun_light_actor.position = tuple(sun)
        self.move_marker("ssb", np.zeros(3))
        if self._sun_trail_t is None or abs(t - self._sun_trail_t) > 30 * DAY:
            lo, hi = self.eph.span("sun")
            tt = np.clip(t + np.linspace(-12.0, 0.0, 600) * 365.25 * DAY, lo, hi)
            self.update_line("sun_trail", self.sun_wobble * ecl(self.eph.pos("sun", tt)), color="#ffcc66",
                             line_width=1.5, opacity=0.7)
            self._sun_trail_t = t
        lab = [sun + [0, 0, 0.08]] + [pos[n] + [0, 0, 0.05] for n in PLANETS] + [pos["bennu"] + [0, 0, 0.04]]
        self.set_label_points("labels", lab)
        self.set_label_points("ssb_label", [np.array([0.0, 0.0, -0.09])])
        if self._bennu_orbit_t is None or abs(t - self._bennu_orbit_t) > 20 * DAY:
            lo, hi = self.eph.span("bennu")
            tt = np.clip(t + np.linspace(-0.5, 0.5, 500) * BENNU_PERIOD_D * DAY, lo, hi)
            self.update_line("bennu_orbit", ecl(self.eph.pos("bennu", tt)), color="#ffb347", line_width=2.0)
            self._bennu_orbit_t = t
        self._update_plan(t)
        if self.galactic is not None:
            self._update_paths(t)

    def _bennu_pos(self, t):
        if self.live is not None and self.live.get("bennu_pos") is not None and abs(self.live["t"] - t) < 1.0:
            return np.asarray(self.live["bennu_pos"])
        return self.eph.pos("bennu", t)

    # ------------------------------------------------------------------ live mission
    def set_live(self, snap):
        first = self.live is None
        self.live = snap
        if first:
            self._add_plan()  # re-draw the plan lines dimmed
        self._update_live()

    def clear_live(self):
        self.live = None
        self.clear_live_actors()
        self._add_plan()
        if self.t is not None:
            self.update(self.t)

    def _update_live(self):
        snap = self.live
        names, pts = [], []
        crafts = [("tractor", snap.get("tractor"))] + [("impactor", snap.get("impactor"))]
        for name, c in crafts:
            color = LIVE_COLORS[name]
            hidden = c is None or (name == "tractor" and not c.get("launched")) or c["phase"] in ("impacted",)
            if hidden:
                self.live_marker(f"live_{name}", None, color)
                self.live_line(f"live_trail_{name}", None, color)
                continue
            p = ecl(c["pos"])[0]
            self.live_marker(f"live_{name}", p, color)
            self.live_line(f"live_trail_{name}", None if c["trail"] is None else ecl(c["trail"]), color)
            if name == "tractor":
                ahead = c.get("plan_ahead")
                self.live_line("live_plan_ahead", None if ahead is None else ecl(ahead), color, width=1.2, opacity=0.45)
            names.append(f"{'Gravity tractor' if name == 'tractor' else c.get('name', 'DART')} ({c['phase']})")
            pts.append(p + [0, 0, -0.07])
        if pts:
            self._d["live_labels"] = (names, "#e6e6e6", 12)
            self.set_label_points("live_labels", pts)
        else:
            self.hide("live_labels")
