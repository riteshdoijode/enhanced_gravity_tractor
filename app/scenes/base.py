"""Shared helpers for the 3D scenes: textures, markers, labels, lines and lights.

Each scene keeps its own origin and units so float32 rendering stays precise. Bodies too small to see at a
view's scale are drawn as screen-space markers (constant pixel size) and labeled.
"""

import numpy as np
import pyvista as pv
from pyvista import examples
from pyvista.examples import planets

from core.constants import BENNU
from core.timeutil import t_to_jd

OBLIQUITY = np.radians(23.4392911)  # J2000 mean obliquity of the ecliptic
EQ_TO_ECL = np.array([[1, 0, 0],
                      [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
                      [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])
LIVE_COLORS = {"tractor": "#00e5ff", "impactor": "#ff5a36"}
_TEX = {}


def textures():
    """Planet textures, loaded once per process (shared by every scene)."""
    if not _TEX:
        _TEX["earth"] = examples.load_globe_texture()
        _TEX["moon"] = planets.download_moon_surface(texture=True)
        _TEX["sun"] = planets.download_sun_surface(texture=True)
    return _TEX


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def pose(R=np.eye(3), p=(0, 0, 0), scale=1.0):
    M = np.eye(4)
    M[:3, :3] = R * scale
    M[:3, 3] = p
    return M


def gmst(t):
    """Greenwich mean sidereal angle [rad] (IAU 1982 linear term; TDB used for UT1, <0.3 deg error)."""
    d = t_to_jd(t) - 2451545.0
    return np.radians((280.46061837 + 360.98564736629 * d) % 360.0)


def bennu_attitude_nominal(t):
    """Body-fixed -> ICRF rotation with the nominal spin (planning mode; the live run sends its own)."""
    a, d = np.radians(BENNU["pole_ra_deg"]), np.radians(BENNU["pole_dec_deg"])
    return rot_z(a + np.pi / 2) @ rot_x(np.pi / 2 - d) @ rot_z(BENNU["omega"] * t)


def line(points):
    return pv.lines_from_points(np.asarray(points, float))


def geom_mesh(gtype, size, mesh=None):
    """PyVista mesh for a MuJoCo geom in its own frame."""
    if gtype == 6:  # box
        return pv.Box(bounds=(-size[0], size[0], -size[1], size[1], -size[2], size[2]))
    if gtype == 5:  # cylinder
        return pv.Cylinder(radius=size[0], height=2 * size[1], direction=(0, 0, 1), resolution=24)
    if gtype == 3:  # capsule
        cyl = pv.Cylinder(radius=size[0], height=2 * size[1], direction=(0, 0, 1), resolution=16)
        caps = [pv.Sphere(radius=size[0], center=(0, 0, z), theta_resolution=16, phi_resolution=8) for z in (-size[1], size[1])]
        return cyl.merge(caps).triangulate()
    if gtype == 2:
        return pv.Sphere(radius=size[0])
    if gtype == 4:
        return pv.ParametricEllipsoid(*size[:3])
    if gtype == 7 and mesh is not None:
        v, f = mesh
        return pv.PolyData.from_regular_faces(np.asarray(v, float), np.asarray(f)).compute_normals()
    return None


class SceneBase:
    SKY_FRAME = None  # rotation ICRF -> scene axes of the star sky (None: the scene is in ICRF axes)

    def __init__(self, plotter):
        from app.scenes.sky import SkyLayer

        self.pl = plotter
        self.sky = SkyLayer(plotter, self.SKY_FRAME)  # the real sky behind the scene (app/scenes/sky.py)
        self.t = None
        self.live = None
        self._a = {}  # actors
        self._d = {}  # mutable datasets
        self._labels, self._texts, self._light = {}, {}, None
        self.tex = textures()

    def reset_scene(self):
        self.pl.clear_actors()
        for prop in [lab.actor for lab in self._labels.values()] + list(self._texts.values()):
            self.pl.renderer.RemoveViewProp(prop)  # labels and overlay text are not PyVista actors: a rebuild left
        self.pl.remove_all_lights()                # them on screen, frozen, beside the new ones
        self._a.clear()
        self._d.clear()
        self._labels = {}
        self._texts = {}
        self._light = None
        self.pl.set_background("black")  # (drawn behind it: the star sky, in its own renderer)

    # ---- labels: one persistent 2-D overlay per key, moved in place (re-creating label actors every frame
    # cost ~1 ms each plus a render on the GUI thread)
    def labels(self, key, names, color="white", size=14):
        self._d[key] = (list(names), color, size)

    def set_label_points(self, key, points, names=None):
        spec = self._d.get(key)
        if names is None and isinstance(spec, tuple):
            names = spec[0]
        color, size = (spec[1], spec[2]) if isinstance(spec, tuple) else ("white", 12)
        lab = self._labels.get(key)
        if lab is None:
            lab = self._labels[key] = LabelSet(self.pl.renderer, names, color, size)
        lab.set(np.asarray(points, float), names)

    def hide(self, key):
        if key in self._a:
            self._a[key].SetVisibility(False)
        if key in getattr(self, "_labels", {}):
            self._labels[key].actor.SetVisibility(False)

    def marker(self, key, color, size):
        """Screen-space marker (constant pixel size at any zoom), moved by editing its point."""
        pts = pv.PolyData(np.zeros((1, 3)))
        self._d[key] = pts
        self._a[key] = self.pl.add_points(pts, color=color, point_size=size, render_points_as_spheres=True,
                                          ambient=1.0, name=key, render=False)

    def move_marker(self, key, p):
        self._d[key].points = np.asarray(p, float).reshape(1, 3)

    def live_marker(self, key, p, color, size=13):
        if key not in self._a:
            self.marker(key, color, size)
        self._a[key].SetVisibility(bool(p is not None))
        if p is not None:
            self.move_marker(key, p)

    def update_line(self, key, points, **kw):
        poly = line(points)
        if key in self._d:
            self._d[key].copy_from(poly)
        else:
            self._d[key] = poly
            self._a[key] = self.pl.add_mesh(poly, name=key, render=False, **kw)

    def live_line(self, key, pts, color, width=2.5, opacity=1.0):
        if pts is None or len(pts) < 2:
            self.hide(key)
            return
        if key in self._a:
            self._a[key].SetVisibility(True)
        self.update_line(key, pts, color=color, line_width=width, opacity=opacity)

    def sun_light(self, direction, intensity=1.2):
        """Directional light shining along -direction (direction points from the scene toward the Sun); one
        light per scene, moved in place."""
        pos = tuple(np.asarray(direction, float) * 1e3)
        if self._light is None:
            self.pl.remove_all_lights()
            light = pv.Light(position=pos, focal_point=(0, 0, 0), color="white", intensity=intensity,
                             light_type="scene light")
            light.positional = False
            self.pl.add_light(light)
            self._light = light
        else:
            self._light.position = pos

    def text(self, key, message, position="upper_left", size=11, color="#e6e6e6", backing=0.0):
        """Overlay text in the top-left corner, one persistent actor per key (SetInput on updates); backing: opacity
        of a dark box behind it (keeps it readable over labels and orbits)."""
        import vtk

        act = self._texts.get(key)
        if act is None:
            act = vtk.vtkTextActor()
            tp = act.GetTextProperty()
            tp.SetFontSize(int(size * 1.6))
            tp.SetColor(pv.Color(color).float_rgb)
            tp.SetVerticalJustificationToTop()
            tp.SetBackgroundColor(0.04, 0.05, 0.07)
            tp.SetBackgroundOpacity(backing)
            act.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            act.SetPosition(0.008, 0.985)
            self.pl.renderer.AddViewProp(act)
            self._texts[key] = act
        act.SetInput(message or "")
        act.SetVisibility(bool(message))

    def clear_live_actors(self):
        for key in [k for k in self._a if k.startswith("live_")]:
            self.pl.remove_actor(self._a.pop(key), render=False)
            self._d.pop(key, None)
        for key in [k for k in self._labels if k.startswith("live_")]:
            self.pl.renderer.RemoveViewProp(self._labels.pop(key).actor)
            self._d.pop(key, None)


class LabelSet:
    """Text labels at 3-D points drawn as an always-visible 2-D overlay (the same VTK pipeline PyVista's
    add_point_labels builds), kept alive and moved by editing its points."""

    def __init__(self, renderer, names, color, size):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk  # noqa: F401  (import check)

        self.vtk = vtk
        self.names = None
        self.poly = vtk.vtkPolyData()
        self.hier = vtk.vtkPointSetToLabelHierarchy()
        self.hier.SetInputData(self.poly)
        self.hier.SetLabelArrayName("labels")
        tp = self.hier.GetTextProperty()
        tp.SetFontSize(int(size))
        tp.SetColor(pv.Color(color).float_rgb)
        tp.SetJustificationToLeft()
        tp.SetVerticalJustificationToBottom()
        mapper = vtk.vtkLabelPlacementMapper()
        mapper.SetInputConnection(self.hier.GetOutputPort())
        mapper.SetPlaceAllLabels(True)
        mapper.SetShapeToNone()
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)
        renderer.AddViewProp(self.actor)
        self.set(np.zeros((len(names or []), 3)), names)

    def set(self, points, names=None):
        from vtk.util.numpy_support import numpy_to_vtk

        vtk = self.vtk
        points = np.atleast_2d(np.asarray(points, float))
        if names is not None and list(names) != self.names:
            self.names = list(names)
            arr = vtk.vtkStringArray()
            arr.SetName("labels")
            for n in self.names:
                arr.InsertNextValue(str(n))
            self.poly.GetPointData().RemoveArray("labels")
            self.poly.GetPointData().AddArray(arr)
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        self.poly.SetPoints(pts)
        self.poly.Modified()
        self.actor.SetVisibility(bool(len(points)))
