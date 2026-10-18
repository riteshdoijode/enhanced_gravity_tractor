"""Bennu view (top right, always): Bennu at true scale, metres, ICRF axes, origin = Bennu's center of mass.

Planning: the recommended point of contact of the selected DART, rotating with Bennu. Live: the incoming DART
in its last minutes, the impact site and ejecta tracers, and the gravity tractor (bus, arm, boulder from the
simulation) with its trail, thrust plume, LIDAR beam, the capture site and the tractor station. During the
tractor's scan for boulders (core/proximity/boulders.py): the green fan of laser shots, the part of the surface
mapped so far (green), and the stones found, numbered, with their estimated size and mass (★ recommended).
"""

import numpy as np

from app.scenes.base import LIVE_COLORS, SceneBase, bennu_attitude_nominal, geom_mesh, pose
from core import plotstyle
from core.assets import bennu_display_mesh, bennu_surface_gravity, tagsite, tagsite_display_mesh

DART_SHOW_RANGE = 150e3  # m: the incoming DART is drawn inside this range (its last ~13 s)
FLASH_S = 25.0  # s: the impact flash expands and fades over this much mission time
SCAN_GREEN = "#3cff6e"
STONE_COLOR = "#9a8f84"


def _cut_site(bennu):
    """Global Bennu mesh with a hole where the 5 cm Nightingale model is inset (avoids z-fighting)."""
    s = tagsite()
    site = tagsite_display_mesh()
    loc = (np.asarray(bennu.cell_centers().points) - s["origin"]) @ s["R"]
    lo = (np.asarray(site.points) - s["origin"]) @ s["R"]
    (x0, y0), (x1, y1) = lo[:, :2].min(0) + 1.5, lo[:, :2].max(0) - 1.5
    inside = (loc[:, 0] > x0) & (loc[:, 0] < x1) & (loc[:, 1] > y0) & (loc[:, 1] < y1) & (np.abs(loc[:, 2]) < 25)
    return bennu.extract_cells(np.where(~inside)[0]).extract_surface(), site


_MESH = {}


def bennu_meshes():
    if not _MESH:
        m = bennu_display_mesh()
        m["g_eff"] = bennu_surface_gravity(m) * 1e6  # micro-m/s^2
        _MESH["full"] = m
        _MESH["holed"], _MESH["site"] = _cut_site(m)
    return _MESH


class BennuScene(SceneBase):
    def __init__(self, plotter, eph):
        super().__init__(plotter)
        self.eph = eph
        self.coloring = "regolith"
        self.contact_b = None  # recommended point of contact (body frame) of the planned DART
        self._geom_actors = {}
        self._sc_pos = None
        self._boulder_key = None
        self._stone_actors = {}  # stone key -> actor (body-frame mesh, posed with Bennu's rotation)
        self._cover = None  # (version, grid) of the scan's coverage shell
        self.highlight = None  # stone selected in the list
        self.build()

    def build(self):
        self.reset_scene()
        meshes = bennu_meshes()
        self._geom_actors = {}
        if self.coloring == "gravity":
            from matplotlib.colors import LinearSegmentedColormap

            self._a["bennu"] = self.pl.add_mesh(
                meshes["full"], scalars="g_eff", cmap=LinearSegmentedColormap.from_list("g", plotstyle.BLUE_RAMP, N=256),
                ambient=0.35, diffuse=0.8, smooth_shading=True, name="bennu",
                scalar_bar_args=dict(title="Surface gravity\nmicro-m/s^2\n", color="white", fmt="%.0f", vertical=True,
                                     position_x=0.82, position_y=0.2, width=0.06, height=0.5, title_font_size=12,
                                     label_font_size=11), render=False)
        else:  # global model with the 5 cm Nightingale (capture-site) model inset
            self._a["bennu"] = self.pl.add_mesh(meshes["holed"], color="#6e675f", ambient=0.02, diffuse=1.0,
                                                specular=0.05, smooth_shading=True, name="bennu", render=False)
            self._a["tagsite"] = self.pl.add_mesh(meshes["site"], color="#6e675f", ambient=0.02, diffuse=1.0,
                                                  specular=0.05, smooth_shading=True, name="tagsite", render=False)
        self._stone_actors, self._cover, self._geom_actors = {}, None, {}
        self.marker("contact", "#ff3b6b", 12)
        self.hide("contact")
        self.labels("contact_label", ["DART aim point"], color="#ff8fa8", size=11)
        self.text("title", "Bennu (true scale)", size=9, color="#c3c2b7")
        if self.live is not None:
            self._update_live()
        self.reset_camera()

    def set_coloring(self, mode):
        self.coloring = mode
        cam = self.pl.camera_position
        self.build()
        if self.t is not None:
            self.update(self.t)
        self.pl.camera_position = cam

    def set_contact(self, point_b):
        self.contact_b = None if point_b is None else np.asarray(point_b, float)
        if self.t is not None:
            self.update(self.t)

    def _R(self, t):
        if self.live is not None and self.live.get("bennu_R") is not None and abs(self.live["t"] - t) < 1.0:
            return np.asarray(self.live["bennu_R"])  # simulated rotation (an impact can change the spin)
        return bennu_attitude_nominal(t)

    def reset_camera(self):
        c = self.pl.camera
        pr = ((self.live or {}).get("tractor") or {}).get("prox")
        if pr is not None and self._sc_pos is not None:
            r = np.linalg.norm(self._sc_pos)
            e1 = self._sc_pos / r
            side = np.cross(e1, [0, 0, 1.0])
            side /= max(np.linalg.norm(side), 1e-9)
            d = max(r, 1500.0)
            mid = 0.5 * self._sc_pos if r > 3e3 else np.zeros(3)
            c.position = tuple(mid + (1.3 * side + 0.35 * e1 + 0.45 * np.array([0, 0, 1.0])) * 1.5 * d)
            c.focal_point, c.up = tuple(mid), (0, 0, 1)
        else:
            c.position, c.focal_point, c.up = (1100.0, -900.0, 600.0), (0, 0, 0), (0, 0, 1)
        self.pl.reset_camera_clipping_range()

    def frame(self, distance=4000.0, toward=None):
        """Camera on Bennu from `distance` m (e.g. to watch the ejecta right after the impact)."""
        c = self.pl.camera
        u = np.asarray(toward, float) if toward is not None else np.array([1.0, 0.3, 0.4])
        u /= np.linalg.norm(u)
        side = np.cross(u, [0, 0, 1.0])
        side /= max(np.linalg.norm(side), 1e-9)
        c.position = tuple(distance * (0.8 * u + 0.6 * side) / np.linalg.norm(0.8 * u + 0.6 * side))
        c.focal_point, c.up = (0.0, 0.0, 0.0), (0, 0, 1)
        self.pl.reset_camera_clipping_range()

    def frame_scan(self):
        """The tractor's scan: the camera beside the tractor, the fan of laser shots and Bennu in view."""
        if self._sc_pos is None:
            return
        rho = np.asarray(self._sc_pos, float)
        r = np.linalg.norm(rho)
        e1 = rho / r
        side = np.cross(e1, [0, 0, 1.0])
        side /= max(np.linalg.norm(side), 1e-9)
        c = self.pl.camera
        c.position = tuple(1.2 * rho + 1.6 * r * side + 0.5 * r * np.array([0, 0, 1.0]))
        c.focal_point, c.up = tuple(0.55 * rho), (0, 0, 1)
        self.pl.reset_camera_clipping_range()

    def set_highlight(self, key):
        self.highlight = key
        if self.live is not None:
            self._update_scan(((self.live.get("tractor") or {}).get("prox")) or {})

    def look_at_stone(self, key, distance=1100.0):
        """Turn the camera to a stone the scan found (the list's selection), Bennu's center in the middle."""
        pr = ((self.live or {}).get("tractor") or {}).get("prox") or {}
        st = next((x for x in pr.get("stones") or [] if x["key"] == key), None)
        if st is not None:
            self.frame(distance, toward=self._R(self.live["t"]) @ np.asarray(st["point_b"], float))

    def set_banner(self, text):
        """A second line under the title (e.g. the slow-motion note)."""
        self.text("title", "Bennu (true scale)" + (f"\n{text}" if text else ""), size=9, color="#c3c2b7")

    def frame_impact(self, site, distance=2500.0):
        """Side-on view of the impact site: the ejecta cone rises across the view, Bennu's limb below it."""
        c = self.pl.camera
        n = np.asarray(site, float) / np.linalg.norm(site)
        side = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
        side /= np.linalg.norm(side)
        focus = 0.3 * np.asarray(site, float) + 150.0 * n  # Bennu and the rising plume both in view
        c.position = tuple(focus + distance * (0.85 * side + 0.35 * n + 0.2 * np.cross(n, side)))
        c.focal_point, c.up = tuple(focus), tuple(n)
        self.pl.reset_camera_clipping_range()

    # ------------------------------------------------------------------ time
    def update(self, t):
        self.t = float(t)
        lo, hi = self.eph.span("bennu")
        sun = self.eph.pos("sun", t) - self.eph.pos("bennu", np.clip(t, lo, hi))
        self.sun_light(sun / np.linalg.norm(sun))
        R = self._R(t)
        self._a["bennu"].user_matrix = pose(R=R)
        if "tagsite" in self._a:
            self._a["tagsite"].user_matrix = pose(R=R)
        show = self.contact_b is not None and not self._impacted()
        self._a["contact"].SetVisibility(bool(show))
        if show:
            p = R @ self.contact_b
            self.move_marker("contact", p * 1.01)
            self.set_label_points("contact_label", [p * 1.01 + R @ self.contact_b / np.linalg.norm(self.contact_b) * 60])
        else:
            self.hide("contact_label")

    def _impacted(self):
        ki = (self.live or {}).get("impactor")
        return ki is not None and ki.get("impact") is not None and ki["phase"] == "impacted"

    # ------------------------------------------------------------------ live
    def set_live(self, snap):
        self.live = snap
        self._update_live()

    def clear_live(self):
        self.live = None
        self.clear_live_actors()
        self._geom_actors = {}
        self._stone_actors = {}
        self._cover = None
        self._boulder_key = None
        self.highlight = None
        self._sc_pos = None
        if self.t is not None:
            self.update(self.t)

    def _update_ejecta(self):
        """Ejecta tracers as sunlit dust: fine round points, brightest near the impact site, thinning as the cloud
        expands; tracers that fell back are part of the surface again (not drawn). A flash marks the impact."""
        import pyvista as pv

        ej = self.live.get("ejecta")
        if ej is None or ej.get("pos") is None:
            for key in ("live_ejecta", "live_crater", "live_flash"):
                self.hide(key)
            return
        pos, code = np.asarray(ej["pos"], float), np.asarray(ej["code"])
        keep = code != 1
        pos, code = pos[keep], code[keep]
        if not len(pos):
            pos, code = np.zeros((1, 3)), np.array([1])  # nothing aloft: one invisible point keeps the actor
        r = np.linalg.norm(pos, axis=1)
        alpha = np.clip(1.15 - r / 6000.0, 0.25, 1.0) * np.where(code == 2, 0.6, 1.0) * (code != 1)
        rgba = np.empty((len(pos), 4), np.uint8)
        rgba[:, :3] = (238, 226, 200)  # sunlit regolith dust
        rgba[:, 3] = (255 * alpha).astype(np.uint8)
        cloud = pv.PolyData(pos)
        cloud["rgba"] = rgba
        if "live_ejecta" in self._d:
            self._d["live_ejecta"].copy_from(cloud)
            self._a["live_ejecta"].SetVisibility(True)
        else:
            self._d["live_ejecta"] = cloud
            self._a["live_ejecta"] = self.pl.add_points(cloud, scalars="rgba", rgba=True, point_size=3,
                                                        render_points_as_spheres=True, lighting=False,
                                                        name="live_ejecta", render=False)  # self-lit: also on the night side
        if ej.get("site_b") is None:
            return
        site = self._R(self.live["t"]) @ np.asarray(ej["site_b"])
        self.live_marker("live_crater", site, "#ff3b6b", 7)
        since = self.live["t"] - ej["t0"]
        if 0 <= since < FLASH_S:  # impact flash: expands and fades over the first seconds
            k = since / FLASH_S
            if "live_flash" not in self._a:
                self._d["live_flash"] = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=16)
                self._a["live_flash"] = self.pl.add_mesh(self._d["live_flash"], color="#fff3d1", lighting=False,
                                                         name="live_flash", render=False)
            a = self._a["live_flash"]
            rad = 12.0 + 70.0 * k
            m = np.diag([rad, rad, rad, 1.0])
            m[:3, 3] = site * 1.01
            a.user_matrix = m
            a.GetProperty().SetOpacity(float(1.0 - k) ** 1.5)
            a.SetVisibility(True)
        else:
            self.hide("live_flash")

    def _update_dart(self):
        ki = self.live.get("impactor")
        rel = None if ki is None else ki.get("rel_bennu")
        if rel is None or np.linalg.norm(rel) > DART_SHOW_RANGE or ki["phase"] not in ("terminal",):
            self.live_marker("live_dart", None, LIVE_COLORS["impactor"])
            self.live_line("live_dart_path", None, LIVE_COLORS["impactor"])
            self.hide("live_dart_label")
            return
        rel = np.asarray(rel, float)
        self.live_marker("live_dart", rel, LIVE_COLORS["impactor"], 12)
        self.live_line("live_dart_path", np.array([rel, np.zeros(3)]), LIVE_COLORS["impactor"], width=1.5, opacity=0.6)
        self._d["live_dart_label"] = ([f"{ki.get('name', 'DART')}: {np.linalg.norm(rel) / 1e3:,.1f} km"], "#ffb39e", 11)
        self.set_label_points("live_dart_label", [rel + np.array([0, 0, 0.05 * np.linalg.norm(rel)])])

    def _update_live(self):
        self._update_ejecta()
        self._update_dart()
        pr = (self.live.get("tractor") or {}).get("prox")
        if pr is None:
            return
        if pr.get("boulder_key") != self._boulder_key:  # a stone was chosen: its rock replaces the plan's
            self._boulder_key = pr.get("boulder_key")
            for name in list(self._geom_actors):
                self.pl.remove_actor(self._geom_actors.pop(name), render=False)
                self._a.pop(f"live_g_{name}", None)
        self._update_scan(pr)
        names = set()
        for name, gtype, size, rgba, p, R in pr["geoms"]:
            names.add(name)
            if name not in self._geom_actors:
                mesh = geom_mesh(gtype, size, pr.get("boulder_mesh"))
                if mesh is None:
                    continue
                color = tuple(float(c) for c in rgba[:3])
                self._geom_actors[name] = self.pl.add_mesh(mesh, color=color, ambient=0.15, diffuse=0.9, specular=0.3,
                                                           smooth_shading=True, name=f"live_g_{name}", render=False)
                self._a[f"live_g_{name}"] = self._geom_actors[name]
            actor = self._geom_actors[name]
            actor.SetVisibility(True)
            actor.user_matrix = pose(R=R, p=p)
        for name, actor in self._geom_actors.items():
            if name not in names:
                actor.SetVisibility(False)
        rho = np.asarray(pr["rho"], float)
        self._sc_pos = rho
        self.live_marker("live_sc_marker", rho if np.linalg.norm(rho) > 1.5e3 else None, LIVE_COLORS["tractor"], 9)
        self.live_line("live_prox_trail", pr["trail"], LIVE_COLORS["tractor"], width=2.0, opacity=0.8)
        T = np.asarray(pr["thrust_vec"], float)
        if np.linalg.norm(T) > 1e-6 and pr["thrust_max"] > 0:
            plume = rho - T / np.linalg.norm(T) * (4.0 + 18.0 * min(1.0, np.linalg.norm(T) / pr["thrust_max"]))
            self.live_line("live_plume", np.array([rho, plume]), "#ffb347", width=5.0, opacity=0.9)
        else:
            self.live_line("live_plume", None, "#ffb347")
        if pr.get("lidar") is not None:
            b = np.asarray(pr["boresight_target"], float) - rho
            b /= np.linalg.norm(b)
            self.live_line("live_lidar", np.array([rho, rho + b * pr["lidar"]]), "#ff3b6b", width=1.5, opacity=0.8)
        else:
            self.live_line("live_lidar", None, "#ff3b6b")
        self.live_marker("live_site", None if pr.get("site") is None else np.asarray(pr["site"], float), "#ff5a36", 9)
        leaving = pr["phase"] in ("release", "depart", "departed")  # the tow is over: no station to mark
        self.live_marker("live_station", None if leaving else np.asarray(pr["station"], float), "#3987e5", 9)
        free = pr.get("boulder_free")
        self.live_marker("live_boulder_free", None if free is None or np.linalg.norm(rho) < 2e3 else np.asarray(free, float),
                         "#c8b89a", 7)  # the released boulder, seen from afar (the mesh itself is in the geoms)
        phase = pr["phase"] + (f" · {pr['capture_stage']}" if pr.get("capture_stage") else "")
        self._d["live_labels"] = ([f"Gravity tractor ({phase})"], "white", 11)
        self.set_label_points("live_labels", [rho + np.array([0, 0, max(8.0, 0.02 * np.linalg.norm(rho))])])

    # ------------------------------------------------------------------ the tractor's scan for boulders
    def _coverage(self, scan, R):
        """The surface mapped so far: a shell 1.5 m above the surface, its 2 x 2 deg cells green once hit."""
        import pyvista as pv

        hits = scan.get("hits")
        if hits is None:
            self.hide("live_cover")
            return
        if self._cover is None:
            from core.impact import radius_map, surface_radius

            n_lat, n_lon = hits.shape
            lat = np.radians(np.linspace(-90, 90, n_lat + 1))
            lon = np.radians(np.linspace(0, 360, n_lon + 1))
            LA, LO = np.meshgrid(lat, lon, indexing="ij")
            U = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], axis=-1)
            r = surface_radius(radius_map(), U.reshape(-1, 3)).reshape(LA.shape) + 1.5
            X = U * r[..., None]
            grid = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
            grid.cell_data["rgba"] = np.zeros((grid.n_cells, 4), np.uint8)
            self._d["live_cover"] = grid
            self._a["live_cover"] = self.pl.add_mesh(grid, scalars="rgba", rgba=True, preference="cell", lighting=False,
                                                     name="live_cover", render=False)
            self._cover = [-1, grid]
        if scan.get("version") != self._cover[0]:
            self._cover[0] = scan.get("version")
            rgba = np.zeros((hits.size, 4), np.uint8)
            on = hits.ravel(order="F") > 0  # (structured-grid cells: the first index runs fastest)
            rgba[on] = (60, 255, 110, 60)
            self._cover[1].cell_data["rgba"][:] = rgba
            self._cover[1].Modified()
        self._a["live_cover"].user_matrix = pose(R=R)
        self._a["live_cover"].SetVisibility(True)

    def _stone_actor(self, st):
        import pyvista as pv

        from core.proximity.boulders import _local_frame
        from core.proximity.capture import boulder_mesh

        hull = boulder_mesh(st["mass"], seed=st["seed"], aspect=st["aspect"], density=st["density"])
        L = _local_frame(-np.asarray(st["up_b"], float))
        v = np.asarray(hull.vertices, float)
        v[:, 2] -= v[:, 2].min()  # resting on the ground
        vb = np.asarray(st["point_b"], float) + v @ L.T
        mesh = pv.PolyData.from_regular_faces(vb, np.asarray(hull.faces)).compute_normals()
        return self.pl.add_mesh(mesh, color=STONE_COLOR, ambient=0.12, diffuse=0.9, smooth_shading=False,
                                name=f"live_stone_{st['key']}", render=False)

    def _update_scan(self, pr):
        scan = pr.get("scan")
        stones = pr.get("stones") or []
        if not scan:
            for key in ("live_beams", "live_cover", "live_stones_ok", "live_stones_no", "live_stone_sel",
                        "live_stone_labels"):
                self.hide(key)
            return
        R = self._R(self.live["t"])
        self._segments("live_beams", scan.get("beams") or [], SCAN_GREEN, width=1.5, opacity=0.85)
        self._coverage(scan, R)
        chosen = pr.get("chosen")
        for st in stones:
            if st["key"] not in self._stone_actors:
                self._stone_actors[st["key"]] = act = self._stone_actor(st)
                self._a[f"live_stone_{st['key']}"] = act
            act = self._stone_actors[st["key"]]
            act.user_matrix = pose(R=R)
            act.SetVisibility(st["key"] != chosen)  # the chosen stone is the capture's own boulder
        picking = chosen is None
        tops = {st["key"]: R @ (np.asarray(st["point_b"], float) + np.asarray(st["up_b"], float) * (st["d_est"] + 4.0))
                for st in stones}
        ok = [tops[st["key"]] for st in stones if st["feasible"]] if picking else []
        no = [tops[st["key"]] for st in stones if not st["feasible"]] if picking else []
        self._points("live_stones_ok", ok, SCAN_GREEN, 9)
        self._points("live_stones_no", no, "#8b8f97", 7)
        sel = self.highlight if picking else None
        self.live_marker("live_stone_sel", tops.get(sel), "#ffb347", 14)
        if picking and stones:
            rec = scan.get("recommended")
            names = []
            for st in stones:  # (short: size and mass are in the list; ten long labels on Bennu overlap. Plain
                big = st["key"] == self.highlight  # glyphs only: the overlay font has no star or ballot x)
                names.append(f"#{st['index']}" + (" recommended" if st["key"] == rec else "")
                             + ("" if st["feasible"] else " ×")
                             + (f" {st['d_est']:.1f} m ~{st['mass_est'] / 1e3:.0f} t" if big else ""))
            self.labels("live_stone_labels", names, color="#c9ffd6", size=11)
            self.set_label_points("live_stone_labels", [tops[st["key"]] + 6.0 * tops[st["key"]] /
                                                        np.linalg.norm(tops[st["key"]]) for st in stones], names)
        else:
            self.hide("live_stone_labels")
