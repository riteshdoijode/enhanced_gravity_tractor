"""Bennu hits Earth: the last two days of the approach, the impact and the plume (core/earth_impact.py).

Earth-centred ICRF in km (as the Earth-Moon view), the textured Earth turning with sidereal time, the real sky behind
it. Bennu (a marker: 0.5 km is far below a pixel here) with the track it has flown and the one ahead, the impact
site, a short flash and the ballistic plume after the impact. The camera starts on the whole Earth and closes in on
the site as Bennu comes down (display only); the caption carries the impact numbers.
"""

import numpy as np
import pyvista as pv
from pyvista.examples import planets

from app.scenes.base import SceneBase, pose, rot_z
from core.constants import RADIUS
from core.earth_impact import SHOW_BEFORE, EarthImpact, gmst
from core.timeutil import t_to_utc_str

R_KM = RADIUS["earth"] / 1e3
BENNU_ORANGE = "#ffb347"


def surface_is_water(texture, lat, lon):
    """Ocean or land at (lat, lon) [deg] from the Earth texture of the views (blue-dominant pixel = water)."""
    img = texture.to_image()
    nx, ny = img.dimensions[0], img.dimensions[1]
    a = np.asarray(img.point_data.active_scalars).reshape(ny, nx, -1)
    i = int(round((lat + 90.0) / 180.0 * (ny - 1)))
    j = int(round((lon + 180.0) / 360.0 * (nx - 1))) % nx
    r, g, b = (float(c) for c in a[i, j, :3])
    return b > 1.4 * r and b > 1.2 * g


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class EarthImpactScene(SceneBase):
    SKY_FULL = True  # a main (centre) view: the 8k Milky Way

    def __init__(self, plotter, eph, scenario=None):
        super().__init__(plotter)
        self.eph = eph
        self.sc = scenario
        self.imp = None
        self.label = ""
        self.build()

    def build(self):
        self.reset_scene()
        self._a["earth"] = self.pl.add_mesh(planets.load_planet(radius=R_KM, lat_resolution=180, lon_resolution=360),
                                            texture=self.tex["earth"], ambient=0.28, diffuse=0.9, specular=0.15,
                                            specular_power=20, name="earth", render=False)
        self.marker("bennu", BENNU_ORANGE, 12)
        self.labels("bennu_label", ["Bennu"], color=BENNU_ORANGE, size=13)
        self.marker("site", "#ff3b6b", 9)
        self.labels("site_label", ["impact site"], color="#ff8fa8", size=11)
        self._d["plume"] = pv.PolyData(np.zeros((1, 3)))
        self._a["plume"] = self.pl.add_points(self._d["plume"], color="#dfe9f5", point_size=3.0, opacity=0.85,
                                              render_points_as_spheres=True, name="plume", render=False)
        self._a["plume"].GetProperty().LightingOff()  # (seen at night too: a display model, drawn unlit)
        self._a["plume"].SetVisibility(False)
        self._a["flash"] = self.pl.add_mesh(pv.Sphere(radius=1.0, theta_resolution=32, phi_resolution=16),
                                            color="#fff4c8", ambient=1.0, diffuse=0.0, opacity=0.0, name="flash",
                                            render=False)

    def set_impact(self, imp: EarthImpact, label="The scenario without a deflection"):
        """The impact to show (scenario or live track): site, surface type, numbers for the caption."""
        if imp is self.imp:
            return
        self.imp = imp
        self.label = label
        imp.set_surface(surface_is_water(self.tex["earth"], imp.site_lat, imp.site_lon))
        tt = np.linspace(imp.t_hit - SHOW_BEFORE, imp.t_hit, 600)
        self.track = imp.geo_state(tt)[:, :3] / 1e3
        self.track_t = tt
        x = imp.x_track[-1]
        self.n = x[:3] / np.linalg.norm(x[:3])
        v = x[3:] / np.linalg.norm(x[3:])
        h = v - (v @ self.n) * self.n
        self.h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-6 else np.cross(self.n, [0, 0, 1.0])
        self.s = np.cross(self.n, self.h)

    # ------------------------------------------------------------------ frame
    def update(self, t, slowmo=None):
        imp = self.imp
        if imp is None:
            return
        self.t = t = float(t)
        sun = self.eph.rel_pos("sun", "earth", t)
        self.sun_light(sun / np.linalg.norm(sun))
        g = gmst(t)
        self._a["earth"].user_matrix = pose(R=rot_z(g))
        since = t - imp.t_hit
        x = imp.geo_state(t)[0]
        p = x[:3] / 1e3
        alt = np.linalg.norm(p) - R_KM
        before = since < 0
        self.live_marker("bennu", p if before else None, BENNU_ORANGE)
        self.set_label_points("bennu_label", [p] if before else np.zeros((0, 3)), ["Bennu"] if before else [])
        past = self.track[self.track_t <= t]
        self.live_line("track_past", np.vstack([past, p]) if before and len(past) else None, BENNU_ORANGE, width=2.5)
        ahead = self.track[self.track_t > t]
        self.live_line("track_ahead", np.vstack([p, ahead]) if before and len(ahead) else None, "#c98b3a", width=1.2,
                       opacity=0.45)
        # the site turns with the Earth until the impact (it is where the track meets the surface then)
        site = rot_z(g - gmst(imp.t_hit)) @ (self.n * R_KM)
        self.move_marker("site", site)
        self.set_label_points("site_label", [site * 1.002])
        # flash: a short bright glow at the site (display), then the ballistic plume (core/earth_impact.py)
        fl = self._a["flash"]
        if 0 <= since < 90.0:
            fl.user_matrix = pose(p=site, scale=60.0 * min(1.0, (since + 0.5) / 6.0) ** 0.5)
            fl.GetProperty().SetOpacity(float(0.95 * np.exp(-since / 18.0)))
            fl.SetVisibility(True)
        else:
            fl.SetVisibility(False)
        pts = imp.plume_at(t)
        if pts is not None:
            self._d["plume"].copy_from(pv.PolyData(pts / 1e3))  # (ICRF already: landed ones turn with the Earth)
            self._a["plume"].SetVisibility(True)
        else:
            self._a["plume"].SetVisibility(False)
        self._camera(alt if before else 0.0, site)
        self._caption(t, alt, x, slowmo)

    def _camera(self, alt_km, site):
        """Whole Earth while Bennu is far; closing in on the site, seen from the side of the track, as it comes."""
        w = _smooth((60_000.0 - alt_km) / 57_000.0)  # 0 above 60,000 km, 1 below 3,000 km
        focal = w * (site + 500.0 * site / np.linalg.norm(site))
        e = 0.3 * self.n + 0.95 * self.s - 0.1 * self.h  # low, from the side of the steep track: the plume rises in view
        e /= np.linalg.norm(e)
        dist = (1 - w) * 48_000.0 + w * 6_500.0
        c = self.pl.camera
        c.focal_point = tuple(focal)
        c.position = tuple(focal + dist * e)
        up = self.n - (self.n @ e) * e
        c.up = tuple(up / np.linalg.norm(up))
        self.pl.reset_camera_clipping_range()

    def _caption(self, t, alt_km, x, slowmo):
        nb = self.imp.numbers
        ent = nb["entry"]
        since = t - self.imp.t_hit
        when = (f"T − {self._dur(-since)}" if since < 0 else f"T + {self._dur(since)}")
        state = (f"Bennu {alt_km:,.0f} km above the surface, {np.linalg.norm(x[3:]) / 1e3:.2f} km/s" if since < 0
                 else "Impact" + (": the plume (ballistic display model)" if since > 5 else ""))
        where = "ocean" if nb["water"] else "land"
        crater = (f"water cavity ~{nb['transient_m'] / 1e3:.1f} km wide (no tsunami model)"
                  if nb["water"] else f"crater ~{nb['final_m'] / 1e3:.1f} km rim to rim")
        if ent.get("airburst"):
            how = [f"Breaks up at {ent['z_star'] / 1e3:.0f} km; airburst at {ent['z_b'] / 1e3:.0f} km"]
        elif ent.get("z_star") is not None:
            how = [f"Breaks up at {ent['z_star'] / 1e3:.0f} km, no airburst: the fragments hit together",
                   f"at {nb['v_surface'] / 1e3:.2f} km/s; {crater}"]
        else:
            how = [f"Reaches the surface intact at {nb['v_surface'] / 1e3:.2f} km/s", crater]
        lines = [f"BENNU HITS EARTH — {t_to_utc_str(self.imp.t_hit)[:16]} UTC", f"({self.label})",
                 f"{t_to_utc_str(t)[:19]} UTC   {when}", state,
                 f"Site {abs(nb['lat']):.1f}°{'S' if nb['lat'] < 0 else 'N'} {abs(nb['lon']):.1f}°"
                 f"{'W' if nb['lon'] < 0 else 'E'}: {where}, local solar time {self._solar_time()}",
                 f"Entry {nb['v_entry'] / 1e3:.2f} km/s at {nb['angle_deg']:.0f}° to the horizon; "
                 f"{nb['energy_mt']:,.0f} Mt TNT", *how,
                 "Entry and crater: Collins, Melosh & Marcus (2005)"]
        if slowmo:
            lines.insert(4, f"SLOW MOTION ({slowmo[0]:g} s/s): {slowmo[1]}")
        self.text("caption", "\n".join(lines), size=9, color="#e6e6e6", backing=0.65)

    def _solar_time(self):
        """Local solar time at the site at the impact (hh:mm)."""
        t = self.imp.t_hit
        sun = rot_z(-gmst(t)) @ self.eph.rel_pos("sun", "earth", t)
        h = (12.0 + (self.imp.site_lon - np.degrees(np.arctan2(sun[1], sun[0]))) / 15.0) % 24.0
        return f"{int(h):02d}:{int(round(60 * (h % 1))) % 60:02d}"

    @staticmethod
    def _dur(s):
        s = abs(float(s))
        if s >= 3600:
            return f"{s / 3600:.1f} h"
        if s >= 60:
            return f"{int(s // 60)} min {int(s % 60):02d} s"
        return f"{s:.1f} s"
