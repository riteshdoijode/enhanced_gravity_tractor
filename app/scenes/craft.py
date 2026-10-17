"""Craft view (right column, below Bennu): a chase camera on the DART or on the gravity tractor.

Craft-centered metres, ICRF axes. The spacecraft sits at the origin at true size; distant bodies keep their
true direction and angular size but are drawn nearer: the Earth system (Earth, Moon, the current orbit and
the trail around Earth) is scaled about the craft by s = min(1, 5 km / distance to Earth), Bennu likewise
with its own factor. This keeps the depth range small (float32, depth buffer) while the view looks right:
Earth's limb below the craft in the parking orbit, the orbit ellipse growing during the orbit raising, Bennu
growing in the DART's camera during the terminal approach, and Bennu's surface below the tractor.
"""

import numpy as np
from pyvista.examples import planets

from app.models import attitude, dart_model
from app.scenes.base import LIVE_COLORS, SceneBase, bennu_attitude_nominal, geom_mesh, gmst, line, pose, rot_z
from app.scenes.bennu import bennu_meshes
from core.assets import spacecraft_glb
from core.constants import RADIUS
from core.dart import orbit_points

D_RENDER = 5e3  # m
PARK_R = RADIUS["earth"] + 400e3
PHASE_TEXT = {"waiting": "in the parking orbit", "parking": "in the parking orbit", "raising": "raising its orbit",
              "departing": "leaving Earth", "spiral": "spiraling out of Earth orbit", "escape": "final escape revolutions",
              "cruise": "heliocentric cruise", "terminal": "terminal approach", "impacted": "impacted Bennu",
              "missed": "missed Bennu", "proximity": "at Bennu"}


class CraftScene(SceneBase):
    def __init__(self, plotter, eph):
        super().__init__(plotter)
        self.eph = eph
        self.kind = "dart"  # which craft is shown: "dart" or "tractor"
        self.follow = True  # camera follows the craft until the user moves it
        self._geom_actors = {}
        self.build()
        try:
            self.pl.iren.add_observer("StartInteractionEvent", lambda *a: setattr(self, "follow", False))
        except Exception:  # off-screen plotters have no interactor
            pass

    def build(self):
        self.reset_scene()
        self._a["earth"] = self.pl.add_mesh(planets.load_planet(radius=RADIUS["earth"], lat_resolution=120,
                                                                lon_resolution=240), texture=self.tex["earth"],
                                            ambient=0.08, diffuse=1.0, specular=0.1, name="earth", culling="back", render=False)
        self._a["moon"] = self.pl.add_mesh(planets.load_planet(radius=RADIUS["moon"], lat_resolution=60,
                                                               lon_resolution=120), texture=self.tex["moon"],
                                           ambient=0.05, diffuse=1.0, name="moon", culling="back", render=False)
        self._a["bennu"] = self.pl.add_mesh(bennu_meshes()["full"], color="#6e675f", ambient=0.03, diffuse=1.0,
                                            specular=0.05, smooth_shading=True, name="bennu", render=False)
        self._dart = []
        for k, (mesh, color, spec) in enumerate(dart_model()):
            self._dart.append(self.pl.add_mesh(mesh, color=color, specular=spec, ambient=0.15, name=f"dart_{k}", render=False))
        before = set(self.pl.renderer.actors)
        self.pl.import_gltf(str(spacecraft_glb()))
        self._osiris = []
        for key in set(self.pl.renderer.actors) - before:
            actor = self.pl.renderer.actors[key]
            m = actor.GetUserMatrix()
            import pyvista as pv

            self._osiris.append((actor, np.eye(4) if m is None else pv.array_from_vtkmatrix(m)))
        for key, color, size in [("m_earth", "#4f8fd9", 9), ("m_moon", "#c8c8c8", 7), ("m_bennu", "#ffb347", 9),
                                 ("m_sun", "#ffd75e", 12)]:
            self.marker(key, color, size)
        self.labels("body_labels", ["Earth", "Moon", "Bennu", "Sun"], color="#c3c2b7", size=11)
        self._geom_actors = {}
        self.follow = True

    def set_kind(self, kind):
        if kind != self.kind:
            self.kind = kind
            self.follow = True
            self.clear_live_geoms()
            if self.t is not None:
                self.update(self.t)

    def reset_view(self):
        self.follow = True
        if self.t is not None:
            self.update(self.t)

    def clear_live_geoms(self):
        for name in list(self._geom_actors):
            self.pl.remove_actor(self._geom_actors.pop(name), render=False)

    def set_live(self, snap):
        self.live = snap

    def clear_live(self):
        self.live = None
        self.clear_live_actors()
        self.clear_live_geoms()
        self.follow = True

    # ------------------------------------------------------------------ craft state
    def _parking_state(self, t, phase0):
        """Planning mode: both craft circle in the 400 km parking orbit (inclination 28.5 deg)."""
        incl = np.radians(28.5)
        ex, ey = np.array([1.0, 0, 0]), np.array([0, np.cos(incl), np.sin(incl)])
        n = np.sqrt(3.986004418e14 / PARK_R**3)
        th = n * t + phase0
        r = PARK_R * (np.cos(th) * ex + np.sin(th) * ey)
        v = PARK_R * n * (-np.sin(th) * ex + np.cos(th) * ey)
        return r, v

    def _craft(self, t):
        """(barycentric position, forward direction, phase, extra) of the shown craft."""
        snap = self.live
        earth = self.eph.pos("earth", t)
        if self.kind == "dart":
            c = snap.get("impactor") if snap else None
            if c is None:
                r, v = self._parking_state(t, np.pi)
                return {"pos": earth + r, "geo": r, "geo_vel": v, "phase": "parking", "name": "DART (not launched)",
                        "orbit": None}
            return c
        c = snap.get("tractor") if snap else None
        if c is None or c.get("phase") == "waiting":
            r, v = self._parking_state(t, 0.0)
            return {"pos": earth + r, "geo": r, "geo_vel": v, "phase": "waiting", "name": "Gravity tractor",
                    "launched": bool(c and c.get("launched"))}
        return c

    # ------------------------------------------------------------------ per frame
    def update(self, t):
        self.t = float(t)
        c = self._craft(t)
        pos = np.asarray(c["pos"], float)
        earth, moon, sun = self.eph.pos("earth", t), self.eph.pos("moon", t), self.eph.pos("sun", t)
        lo, hi = self.eph.span("bennu")
        live_b = self.live.get("bennu_pos") if self.live and abs(self.live["t"] - t) < 1.0 else None
        bennu = np.asarray(live_b, float) if live_b is not None else self.eph.pos("bennu", np.clip(t, lo, hi))
        s_dir = (sun - pos) / np.linalg.norm(sun - pos)
        self.sun_light(s_dir, intensity=1.3)
        phase = c["phase"]
        prox = c.get("prox")
        key = (self.kind, phase, prox["phase"] if prox is not None else None)
        if key != getattr(self, "_phase_key", None):  # a new phase: the chase camera re-frames the craft
            self._phase_key = key
            self.follow = True
        if prox is not None:  # tractor at Bennu: exact relative position from the simulation
            bennu = pos - np.asarray(prox["rho"], float)
        # ---- Earth system, scaled about the craft
        e_rel, m_rel = earth - pos, moon - pos
        de = max(np.linalg.norm(e_rel), 1.0)
        s_e = min(1.0, D_RENDER / de)
        near_earth = de < 3e9
        self._a["earth"].SetVisibility(bool(near_earth))
        self._a["moon"].SetVisibility(bool(near_earth))
        if near_earth:
            self._a["earth"].user_matrix = pose(R=rot_z(gmst(t)), p=s_e * e_rel, scale=s_e)
            x = -(moon - earth) / np.linalg.norm(moon - earth)  # tidally locked Moon
            y = np.cross([0, 0, 1.0], x)
            y /= np.linalg.norm(y)
            self._a["moon"].user_matrix = pose(R=np.column_stack([x, y, np.cross(x, y)]), p=s_e * m_rel, scale=s_e)
        geo = c.get("geo")
        orbit = c.get("orbit")
        if orbit is not None and near_earth:
            pts = orbit_points(orbit["r_p"], orbit["v_p"], np.asarray(orbit["P"]), np.asarray(orbit["Q"]), n=240,
                               r_max=2.4e8)
            self.live_line("live_orbit", s_e * (pts + e_rel), LIVE_COLORS["impactor"], width=1.5, opacity=0.8)
        elif geo is not None and near_earth and phase in ("waiting", "parking"):
            th = np.linspace(0, 2 * np.pi, 180)
            r0 = np.asarray(geo, float)
            v0 = np.asarray(c.get("geo_vel", np.cross([0, 0, 1.0], r0)), float)
            u, w = r0 / np.linalg.norm(r0), v0 / np.linalg.norm(v0)
            ring = np.linalg.norm(r0) * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
            self.live_line("live_orbit", s_e * (ring + e_rel), "#9aa4b1", width=1.2, opacity=0.6)
        else:
            self.live_line("live_orbit", None, "#9aa4b1")
        # ---- Bennu, scaled about the craft
        b_rel = bennu - pos
        db = max(np.linalg.norm(b_rel), 1.0)
        s_b = min(1.0, D_RENDER / db)
        R_b = np.asarray(self.live["bennu_R"]) if self.live and self.live.get("bennu_R") is not None \
            and abs(self.live["t"] - t) < 1.0 else bennu_attitude_nominal(t)
        bennu_vis = db < 5e8
        self._a["bennu"].SetVisibility(bool(bennu_vis))
        if bennu_vis:
            self._a["bennu"].user_matrix = pose(R=R_b, p=s_b * b_rel, scale=s_b)
        # ---- markers + labels (far bodies)
        far = 0.8 * 2e5
        pts, show = {}, {}
        for key, rel, vis in [("m_earth", e_rel, True), ("m_moon", m_rel, de < 3e9), ("m_bennu", b_rel, True),
                              ("m_sun", sun - pos, True)]:
            d = np.linalg.norm(rel)
            p = rel / d * min(far, D_RENDER * 1.05 if key != "m_sun" else far) if d > D_RENDER else rel * 1.02
            if key == "m_earth" and near_earth:
                p = s_e * e_rel
            if key == "m_moon" and near_earth:
                p = s_e * m_rel
            if key == "m_bennu" and db < 1e6:
                p = s_b * b_rel
            pts[key], show[key] = p, vis
            self._a[key].SetVisibility(bool(vis))
            self.move_marker(key, p)
        self.set_label_points("body_labels", [pts[k] for k in ("m_earth", "m_moon", "m_bennu", "m_sun")])
        # ---- the craft itself
        forward = self._forward(c, e_rel, b_rel)
        if self.kind == "dart":
            visible = phase not in ("impacted",)
            R = attitude(forward, s_dir, "z")
            for a in self._dart:
                a.SetVisibility(bool(visible))
                a.user_matrix = pose(R=R)
            for a, _ in self._osiris:
                a.SetVisibility(False)
            self.clear_live_geoms()
        else:
            for a in self._dart:
                a.SetVisibility(False)
            if prox is not None:
                for a, _ in self._osiris:
                    a.SetVisibility(False)
                self._prox_geoms(prox)
            else:
                self.clear_live_geoms()
                R = attitude(forward, s_dir, "y")
                for a, node in self._osiris:
                    import pyvista as pv

                    a.SetVisibility(True)
                    a.SetUserMatrix(pv.vtkmatrix_from_array(pose(R=R) @ node))
        self._camera(c, forward, e_rel, b_rel, prox)
        self._overlay(c, de, db)

    def _forward(self, c, e_rel, b_rel):
        phase = c["phase"]
        if phase in ("waiting", "parking", "raising", "departing", "spiral", "escape") and c.get("geo_vel") is not None:
            return np.asarray(c["geo_vel"], float)
        if phase in ("waiting", "parking", "raising", "departing", "spiral", "escape"):
            g = np.asarray(c.get("geo", -e_rel), float)
            return np.cross([0, 0, 1.0], g)
        return b_rel if np.linalg.norm(b_rel) > 0 else np.array([1.0, 0, 0])

    def _prox_geoms(self, pr):
        rho = np.asarray(pr["rho"], float)
        names = set()
        for name, gtype, size, rgba, p, R in pr["geoms"]:
            names.add(name)
            if name not in self._geom_actors:
                mesh = geom_mesh(gtype, size, pr.get("boulder_mesh"))
                if mesh is None:
                    continue
                self._geom_actors[name] = self.pl.add_mesh(mesh, color=tuple(float(x) for x in rgba[:3]), ambient=0.15,
                                                           diffuse=0.9, specular=0.3, smooth_shading=True,
                                                           name=f"craft_g_{name}")
            a = self._geom_actors[name]
            a.SetVisibility(True)
            a.user_matrix = pose(R=R, p=np.asarray(p, float) - rho)
        for name, a in self._geom_actors.items():
            if name not in names:
                a.SetVisibility(False)

    def _camera(self, c, forward, e_rel, b_rel, prox):
        cam = self.pl.camera
        if self.follow:
            f = np.asarray(forward, float)
            f /= np.linalg.norm(f)
            phase = c["phase"]
            if prox is not None:
                u = np.asarray(prox["rho"], float)
                u /= np.linalg.norm(u)
                side = np.cross(u, [0, 0, 1.0])
                side /= max(np.linalg.norm(side), 1e-9)
                cam.position = tuple(u * 22.0 + side * 14.0 + np.array([0, 0, 6.0]))
                cam.focal_point, cam.up = tuple(-u * 6.0), tuple(u)
            elif phase in ("waiting", "parking", "raising", "departing", "spiral", "escape"):
                up = -e_rel / np.linalg.norm(e_rel)  # away from Earth
                side = np.cross(f, up)
                # above the craft looking down at it, Earth behind it (the orbit runs across the view)
                cam.position = tuple(16.0 * up - 9.0 * f + 5.0 * side)
                cam.focal_point, cam.up = (0.0, 0.0, 0.0), tuple(f)
            else:
                up = np.cross(f, np.cross([0, 0, 1.0], f))
                up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-9 else np.array([0, 0, 1.0])
                back = 16.0 if phase == "terminal" else 26.0
                cam.position = tuple(-back * f + 4.0 * up)
                cam.focal_point, cam.up = tuple(60.0 * f), tuple(up)
        cam.clipping_range = (0.3, 2.5e5)

    def _overlay(self, c, de, db):
        phase = c["phase"]
        name = c.get("name", "DART" if self.kind == "dart" else "Gravity tractor")
        if self.kind == "tractor" and phase == "waiting":
            name = "Gravity tractor" + ("" if c.get("launched") else " (awaiting a successful DART)")
        lines = [f"{name}: {PHASE_TEXT.get(phase, phase)}"]
        if c.get("speed_geo") is not None:
            lines.append(f"speed {c['speed_geo'] / 1e3:.2f} km/s · perigee speed {c.get('v_perigee', np.nan) / 1e3:.2f} km/s"
                         + (f" · apogee {c['r_apogee'] / 1e3:,.0f} km" if np.isfinite(c.get('r_apogee', np.inf)) else ""))
        if "prop_used" in c and (c.get("prop_used") or 0) > 0:
            lines.append(f"fuel used {c['prop_used']:,.0f} kg · Δv {c['dv_used'] / 1e3:,.2f} km/s")
        if c.get("range") is not None:
            lines.append(f"Bennu {c['range'] / 1e3:,.0f} km · closing {c['rel_speed'] / 1e3:.2f} km/s")
        elif de < 3e9:
            lines.append(f"Earth's center {de / 1e3:,.0f} km")
        if phase == "impacted":
            lines.append("IMPACT — loss of signal")
        self.text("overlay", "\n".join(lines), size=9)
