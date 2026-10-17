"""The real star sky behind a 3-D view (core/sky.py: XHIP stars to magnitude 8, the 88 IAU constellations).

A second VTK renderer in layer 0 of the view's render window draws the sky; the view's own renderer moves to layer 1
(VTK then keeps the colour buffer and clears only the depth for it, so the scene is drawn over the sky). Before every
render the sky's camera takes the view camera's orientation and field of view, at the centre of a unit sphere: the
stars are infinitely far, so panning or zooming the scene never moves them, only turning the camera does. `frame`
rotates ICRF into the view's axes (the identity for the ICRF views, EQ_TO_ECL for the ecliptic Solar System view).
Stars are points in magnitude classes (size and brightness from the magnitude, colour from B-V); the Milky Way
shows as the density of the faint stars along the Galactic plane.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from core.sky import constellations, load_sky

# magnitude classes: (upper magnitude, point size [px], brightness 0..1)
MAG_CLASSES = [(0.5, 5.0, 1.0), (1.5, 4.2, 1.0), (2.5, 3.4, 0.95), (3.5, 2.7, 0.85), (4.5, 2.2, 0.72),
               (5.5, 1.8, 0.58), (6.5, 1.5, 0.45), (99.0, 1.2, 0.32)]
LINE_COLOR = (0.30, 0.47, 0.70)
NAME_COLOR = (0.55, 0.70, 0.90)


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

    def __init__(self, plotter, frame=None):
        self.pl = plotter
        self.frame = np.eye(3) if frame is None else np.asarray(frame, float)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.InteractiveOff()
        self._con_actors = []
        self._labels = None
        self.ok = self._build()
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
        sky = load_sky()
        if sky is None:  # assets/sky.npz not fetched: a plain black background
            return False
        u = sky["star_u"].astype(float) @ self.frame.T
        mag = sky["star_mag"]
        lo = -np.inf
        for hi, size, bright in MAG_CLASSES:
            sel = (mag > lo) & (mag <= hi)
            lo = hi
            if sel.any():
                self.ren.AddActor(_actor(_points(u[sel], sky["star_rgb"][sel] * bright), size=size))
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
        act.SetVisibility(False)
        self.ren.AddActor(act)
        self._con_actors.append(act)
        if names:
            from app.scenes.base import LabelSet

            self._labels = LabelSet(self.ren, names, "#8cb3e6", 12)
            self._labels.set(0.998 * np.concatenate(where), names)
            self._labels.actor.SetVisibility(False)
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
        return bool(self._con_actors and self._con_actors[0].GetVisibility())

    def set_constellations(self, on):
        """The constellation figures and names on or off."""
        for a in self._con_actors:
            a.SetVisibility(bool(on))
        if self._labels is not None:
            self._labels.actor.SetVisibility(bool(on))
