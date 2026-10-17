"""Proximity sensors: laser altimeter (LIDAR) range and camera bearing.

LIDAR: range from the spacecraft along a boresight b to the first surface hit, ray-cast on the OSIRIS-REx
shape model (400k-face global model; the 5 cm Nightingale model near the capture site). The same ray cast
from the *estimated* position is the filter's measurement model, and its Jacobian is
    dh/dp = -n^T / (n^T b)      (n = surface normal at the hit, b held fixed for the measurement)
Beyond LIDAR_MAX the range comes from Bennu's apparent size in the camera (1 % of the range).
Camera: unit vector to a landmark (Bennu's center of figure, or the capture site), expressed in a 2-D
tangent basis E of the predicted direction u: y = E^T u, H = -E^T / |l - p|  (since E^T u_pred = 0).
"""

import numpy as np
import vtk

from core.assets import bennu_display_mesh, tagsite_display_mesh

LIDAR_MAX = 7e3  # m (OSIRIS-REx OLA-class)
SIGMA_LIDAR = 0.1  # m
SIGMA_RANGE_OPTICAL = 0.01  # fraction of range
SIGMA_BEARING = 0.5e-3  # rad


class ShapeRaycaster:
    """First intersection of a ray with Bennu (body frame)."""

    def __init__(self):
        self.global_mesh = bennu_display_mesh()
        self.site_mesh = tagsite_display_mesh()
        self._loc = []
        for mesh in (self.site_mesh, self.global_mesh):  # the fine site model has priority
            loc = vtk.vtkStaticCellLocator()
            loc.SetDataSet(mesh)
            loc.BuildLocator()
            self._loc.append((loc, mesh))
        self._normals = [np.asarray(m.cell_normals) for m in (self.site_mesh, self.global_mesh)]

    def cast(self, p, b, max_range=LIDAR_MAX):
        """Distance and outward normal of the first hit from p along unit b (body frame), or (None, None)."""
        t, x, pc, sid, cid = vtk.reference(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], vtk.reference(0), vtk.reference(0)
        end = p + b * max_range
        best = (None, None)
        for k, (loc, _) in enumerate(self._loc):
            if loc.IntersectWithLine(tuple(p), tuple(end), 1e-9, t, x, pc, sid, cid):
                dist = float(t) * max_range
                if best[0] is None or dist < best[0] - 0.05:
                    best = (dist, self._normals[k][int(cid)])
        return best


def tangent_basis(u):
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(u, e1)])
