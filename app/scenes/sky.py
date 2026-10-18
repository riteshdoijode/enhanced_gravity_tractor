"""The real sky behind a 3-D view, in one of two looks the Display toggle picks (cross-faded over SKY_FADE_S):
  - the Milky Way (default): NASA SVS's Deep Star Maps 2020 (Gaia DR2, Hipparcos-2, Tycho-2), 8192 x 4096, on a
    sphere around the camera (core/sky.py load_milkyway);
  - constellations: the catalogue stars (XHIP, to magnitude 8) with the 88 IAU constellation figures and names.

A second VTK renderer in layer 0 of the view's render window draws the sky; the view's own renderer moves to layer 1
(VTK then keeps the colour buffer and clears only the depth for it, so the scene is drawn over the sky). Before every
render the sky's camera takes the view camera's orientation and field of view, at the centre of a unit sphere: the
stars are infinitely far, so panning or zooming the scene never moves them, only turning the camera does. `frame`
rotates ICRF into the view's axes (the identity for the ICRF views, EQ_TO_ECL for the ecliptic Solar System view).
Catalogue stars are points in magnitude classes (size and brightness from the magnitude, colour from B-V).
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from core.sky import constellations, load_milkyway, load_sky, unit_vectors

# magnitude classes: (upper magnitude, point size [px], brightness 0..1)
MAG_CLASSES = [(0.5, 5.0, 1.0), (1.5, 4.2, 1.0), (2.5, 3.4, 0.95), (3.5, 2.7, 0.85), (4.5, 2.2, 0.72),
               (5.5, 1.8, 0.58), (6.5, 1.5, 0.45), (99.0, 1.2, 0.32)]
LINE_COLOR = (0.30, 0.47, 0.70)
NAME_COLOR = (0.55, 0.70, 0.90)
SKY_FADE_S = 3.0  # seconds of cross-fade between the Milky Way and the constellations
R_MILKYWAY = 2.0  # the image sphere lies behind the catalogue stars (unit sphere) and the figures
_TEXTURE = {}


def _milkyway_image(half):
    """The Milky Way image as vtkImageData (shared by every view that uses that resolution), or None."""
    if half not in _TEXTURE:
        img = load_milkyway(half)
        if img is None:
            _TEXTURE[half] = None
        else:
            data = vtk.vtkImageData()
            data.SetDimensions(img.shape[1], img.shape[0], 1)
            rows_up = np.ascontiguousarray(img[::-1]).reshape(-1, 3)  # VTK images start at the bottom row (south)
            arr = numpy_to_vtk(rows_up, deep=True)
            arr.SetName("rgb")
            data.GetPointData().SetScalars(arr)
            _TEXTURE[half] = data
    return _TEXTURE[half]


def _milkyway_sphere(frame, n_lon=256, n_lat=128):
    """Sphere of radius R_MILKYWAY with the image's plate carree coordinates: s = 0.5 - RA/360 (0h in the middle,
    RA to the left), t = (Dec + 90)/180; the seam (RA 180 deg) has its own column at each end."""
    S, T = np.meshgrid(np.linspace(0.0, 1.0, n_lon + 1), np.linspace(0.0, 1.0, n_lat + 1))
    pts = R_MILKYWAY * unit_vectors((0.5 - S) * 360.0, T * 180.0 - 90.0).reshape(-1, 3) @ frame.T
    poly = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    poly.SetPoints(vp)
    cells = vtk.vtkCellArray()
    w = n_lon + 1
    for i in range(n_lat):
        for j in range(n_lon):
            k = i * w + j
            cells.InsertNextCell(4)
            for q in (k, k + 1, k + w + 1, k + w):
                cells.InsertCellPoint(q)
    poly.SetPolys(cells)
    tc = numpy_to_vtk(np.ascontiguousarray(np.column_stack([S.ravel(), T.ravel()])), deep=True)
    tc.SetName("tcoords")
    poly.GetPointData().SetTCoords(tc)
    return poly


def _points(u, rgb=None):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(u, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for i in range(len(u)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    if rgb is not None:
        c = numpy_to_vtk(np.ascontiguousarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)), deep=True)
        c.SetName("rgb")
        poly.GetPointData().SetScalars(c)
    return poly


def _actor(poly, size=None, color=None, width=None, opacity=1.0):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    if color is not None:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.LightingOff()
    if size is not None:
        prop.SetPointSize(size)
        prop.SetRenderPointsAsSpheres(size > 1.6)
    if color is not None:
        prop.SetColor(*color)
    if width is not None:
        prop.SetLineWidth(width)
    prop.SetOpacity(opacity)
    actor.PickableOff()
    return actor


class SkyLayer:
    """Stars (and optionally the constellation figures and names) behind one PyVista plotter's scene."""

    def __init__(self, plotter, frame=None, half=True):
        self.pl = plotter
        self.frame = np.eye(3) if frame is None else np.asarray(frame, float)
        self.half = half  # the 4k image (small views) or the full 8k one (the main views)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.InteractiveOff()
        self._con_actors = []
        self._star_actors = []
        self._labels = None
        self._milkyway = None
        self.mix = None
        self.ok = self._build()
        self.set_mix(0.0)  # the Milky Way
        rw = plotter.ren_win
        n = rw.GetNumberOfLayers()
        renderers = [rw.GetRenderers().GetItemAsObject(i) for i in range(rw.GetRenderers().GetNumberOfItems())]
        rw.SetNumberOfLayers(n + 1)
        for r in renderers:  # the view's renderers move up one layer (keeping their colour buffer: see the top)
            r.SetLayer(r.GetLayer() + 1)
        self.ren.SetLayer(0)
        rw.AddRenderer(self.ren)
        self.ren.SetViewport(*plotter.renderer.GetViewport())
        self._obs = rw.AddObserver("StartEvent", lambda *a: self.sync())

    def _build(self):
        img = _milkyway_image(self.half)
        if img is not None:  # first in the renderer: drawn first, behind everything
            tex = vtk.vtkTexture()
            tex.SetInputData(img)
            tex.InterpolateOn()
            tex.MipmapOn()
            tex.EdgeClampOn()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(_milkyway_sphere(self.frame))
            self._milkyway = vtk.vtkActor()
            self._milkyway.SetMapper(mapper)
            self._milkyway.SetTexture(tex)
            prop = self._milkyway.GetProperty()
            prop.LightingOff()
            prop.SetColor(1.0, 1.0, 1.0)
            self._milkyway.PickableOff()
            self.ren.AddActor(self._milkyway)
        sky = load_sky()
        if sky is None:  # assets/sky.npz not fetched: the Milky Way alone (or a plain black background)
            return img is not None
        u = sky["star_u"].astype(float) @ self.frame.T
        mag = sky["star_mag"]
        lo = -np.inf
        for hi, size, bright in MAG_CLASSES:
            sel = (mag > lo) & (mag <= hi)
            lo = hi
            if sel.any():
                act = _actor(_points(u[sel], sky["star_rgb"][sel] * bright), size=size)
                self._star_actors.append(act)
                self.ren.AddActor(act)
        segs = []
        names, where = [], []
        for c in constellations().values():
            segs.append(c["segments"].astype(float) @ self.frame.T)
            names += c["names"]
            where.append(c["label_u"].astype(float) @ self.frame.T)
        seg = np.concatenate(segs) if segs else np.zeros((0, 2, 3))
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(0.999 * seg.reshape(-1, 3)), deep=True))
        poly.SetPoints(pts)
        lines = vtk.vtkCellArray()
        for i in range(len(seg)):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * i)
            lines.InsertCellPoint(2 * i + 1)
        poly.SetLines(lines)
        act = _actor(poly, color=LINE_COLOR, width=1.2, opacity=0.75)
        self.ren.AddActor(act)
        self._con_actors.append(act)
        if names:
            from app.scenes.base import LabelSet

            self._labels = LabelSet(self.ren, names, "#8cb3e6", 12)
            self._labels.set(0.998 * np.concatenate(where), names)
        return True

    def sync(self):
        """The sky camera takes the view camera's orientation and field of view (it stays at the centre)."""
        cam, sc = self.pl.renderer.GetActiveCamera(), self.ren.GetActiveCamera()
        p, f = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
        d = f - p
        n = np.linalg.norm(d)
        if not np.isfinite(n) or n == 0:
            return
        sc.SetPosition(0.0, 0.0, 0.0)
        sc.SetFocalPoint(*(d / n))
        sc.SetViewUp(*cam.GetViewUp())
        sc.SetViewAngle(cam.GetViewAngle())
        sc.SetParallelProjection(False)
        sc.SetClippingRange(0.05, 5.0)
        self.ren.SetViewport(*self.pl.renderer.GetViewport())

    @property
    def constellations_on(self):
        return bool(self.mix is not None and self.mix >= 0.5)

    def set_mix(self, c):
        """The background between the Milky Way (c = 0) and the constellations (c = 1): the image fades out as the
        catalogue stars, the figures and the names fade in (without the image, the stars are always shown)."""
        c = float(np.clip(c, 0.0, 1.0))
        if c == self.mix:
            return
        self.mix = c
        if self._milkyway is not None:
            self._milkyway.GetProperty().SetOpacity(1.0 - c)
            self._milkyway.SetVisibility(c < 0.999)
        stars = c if self._milkyway is not None else 1.0
        for a in self._star_actors:
            a.GetProperty().SetOpacity(stars)
            a.SetVisibility(stars > 0.001)
        for a in self._con_actors:
            a.GetProperty().SetOpacity(0.75 * c)
            a.SetVisibility(c > 0.001)
        if self._labels is not None:
            self._labels.hier.GetTextProperty().SetOpacity(c)
            self._labels.hier.Modified()
            self._labels.actor.SetVisibility(c > 0.05)

    def set_constellations(self, on):
        """The constellation sky (on) or the Milky Way (off), at once (main.py cross-fades with set_mix)."""
        self.set_mix(1.0 if on else 0.0)
