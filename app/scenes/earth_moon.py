"""Earth-Moon view (bottom tab): km, ICRF axes, origin = Earth's center.

Shows the Moon's orbit (the goal: Bennu's 2135 flyby must stay outside it), Bennu's track over +-20 days,
the keyhole point where the scenario orbit makes its 2135 closest approach, and during a live run the DART
or the tractor while they are near Earth.
"""

import numpy as np
from pyvista.examples import planets

from app.scenes.base import LIVE_COLORS, SceneBase, gmst, pose
from core.constants import DAY, LUNAR_DISTANCE, RADIUS


class EarthMoonScene(SceneBase):
    def __init__(self, plotter, eph, scenario=None):
        super().__init__(plotter)
        self.eph = eph
        self.sc = scenario
        self.build()

    def build(self):
        self.reset_scene()
        R_e, R_m = RADIUS["earth"] / 1e3, RADIUS["moon"] / 1e3
        self._a["earth"] = self.pl.add_mesh(planets.load_planet(radius=R_e, lat_resolution=180, lon_resolution=360),
                                            texture=self.tex["earth"], ambient=0.06, diffuse=1.0, specular=0.15,
                                            specular_power=20, name="earth", render=False)
        self._a["moon"] = self.pl.add_mesh(planets.load_planet(radius=R_m, lat_resolution=90, lon_resolution=180),
                                           texture=self.tex["moon"], ambient=0.04, diffuse=1.0, name="moon", render=False)
        self.marker("bennu", "#ffb347", 12)
        self.labels("labels", ["Moon", "Bennu"])
        self.marker("keyhole", "#ff3b6b", 11)
        self.labels("keyhole_label", ["2135 keyhole flyby"], color="#ff8fa8", size=11)
        self.update_line("ld_ring", self._ring(), color="#d03b3b", line_width=1.0, opacity=0.45)
        self.labels("ld_label", ["goal: the Moon's mean distance (1 LD)"], color="#e58a8a", size=10)
        self.reset_camera()

    def reset_camera(self):
        c = self.pl.camera
        c.position, c.focal_point, c.up = (0, -8.0e5, 4.5e5), (0, 0, 0), (0, 0, 1)
        self.pl.reset_camera_clipping_range()

    def update(self, t):
        self.t = float(t)
        sun = self.eph.rel_pos("sun", "earth", t)
        self.sun_light(sun / np.linalg.norm(sun))
        self._a["earth"].user_matrix = pose(R=np.array([[np.cos(gmst(t)), -np.sin(gmst(t)), 0],
                                                        [np.sin(gmst(t)), np.cos(gmst(t)), 0], [0, 0, 1.0]]))
        r_moon = self.eph.rel_pos("moon", "earth", t) / 1e3
        x = -r_moon / np.linalg.norm(r_moon)
        y = np.cross([0, 0, 1.0], x)
        y /= np.linalg.norm(y)
        self._a["moon"].user_matrix = pose(R=np.column_stack([x, y, np.cross(x, y)]), p=r_moon)
        tt = t + np.linspace(-13.66, 13.66, 300) * DAY
        self.update_line("moon_orbit", self.eph.rel_pos("moon", "earth", tt) / 1e3, color="#9aa4b1", opacity=0.5,
                         line_width=1.5)
        lo, hi = self.eph.span("bennu")
        tb = np.clip(t + np.linspace(-20, 20, 600) * DAY, lo, hi)
        self.update_line("bennu_track", self.eph.rel_pos("bennu", "earth", tb) / 1e3, color="#ffb347", line_width=2.0)
        live_b = self.live.get("bennu_pos") if self.live and abs(self.live["t"] - t) < 1.0 else None
        r_b = ((np.asarray(live_b) - self.eph.pos("earth", t)) if live_b is not None
               else self.eph.rel_pos("bennu", "earth", np.clip(t, lo, hi))) / 1e3
        self.move_marker("bennu", r_b)
        self.set_label_points("labels", [r_moon + [0, 0, 4000.0], r_b + [0, 0, 6000.0]])
        self.set_label_points("ld_label", [self._ring()[150] * 1.04])
        if self.sc is not None:
            kp = self.eph.rel_pos("bennu", "earth", self.sc.t_ca) / 1e3
            self.move_marker("keyhole", kp)
            self.set_label_points("keyhole_label", [kp + [0, 0, -9000.0]])
        if self.live is not None:
            self._update_live()

    def set_live(self, snap):
        self.live = snap

    def clear_live(self):
        self.live = None
        self.clear_live_actors()

    def set_track(self, t, d_track):
        """Deflected orbit near the 2135 flyby (Earth-relative positions [m]) from the verification, or None."""
        if d_track is None:
            self.live_line("live_defl_track", None, "#0ca30c")
            return
        self.live_line("live_defl_track", np.asarray(d_track) / 1e3, "#0ca30c", width=2.0, opacity=0.9)

    def _update_live(self):
        for name in ("tractor", "impactor"):
            c = self.live.get(name)
            geo = None if c is None else c.get("geo")
            if name == "tractor" and c is not None and not c.get("launched"):
                geo = None
            self.live_marker(f"live_{name}", None if geo is None else np.asarray(geo) / 1e3, LIVE_COLORS[name], 10)

    def _ring(self):
        """Circle of radius 1 LD in the Moon's orbital plane at the keyhole flyby."""
        t = self.sc.t_ca if self.sc is not None else self.eph.span("earth")[0]
        r, v = self.eph.rel_pos("moon", "earth", t), self.eph.rel_vel("moon", "earth", t)
        u = r / np.linalg.norm(r)
        w = np.cross(np.cross(r, v), u)
        w /= np.linalg.norm(w)
        th = np.linspace(0, 2 * np.pi, 200)
        return LUNAR_DISTANCE / 1e3 * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
