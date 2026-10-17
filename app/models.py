"""Procedural 3D model of the DART-style impactor (display only).

Body axes: +x = boresight (the camera looks ahead along the flight direction), the two roll-out solar arrays
(ROSA-like, 8.5 m x 2.3 m each, as on NASA's DART) extend along +-y with their cells facing +z, the engine
nozzle points to -x. Returned as a few meshes grouped by colour so a single pose moves them together.
"""

import numpy as np
import pyvista as pv


def dart_model():
    """[(mesh, color, specular)] of the impactor in its body frame (metres)."""
    bus = pv.Box(bounds=(-0.6, 0.6, -0.65, 0.65, -0.65, 0.65))
    camera = pv.Cylinder(center=(0.85, 0.0, 0.25), direction=(1, 0, 0), radius=0.16, height=0.5, resolution=24)
    lens = pv.Disc(center=(1.101, 0.0, 0.25), inner=0.0, outer=0.12, normal=(1, 0, 0), c_res=24)
    antenna = pv.Box(bounds=(-0.45, 0.45, -0.45, 0.45, -0.70, -0.66))  # flat radial-line slot antenna
    nozzle = pv.Cone(center=(-0.85, 0, 0), direction=(-1, 0, 0), height=0.5, radius=0.28, resolution=24)
    arrays, booms = [], []
    for s in (-1, 1):
        y0, y1 = s * 0.75, s * (0.75 + 8.5)
        arrays.append(pv.Box(bounds=(-1.15, 1.15, min(y0, y1), max(y0, y1), -0.02, 0.02)))
        booms.append(pv.Cylinder(center=(0, s * 0.70, 0), direction=(0, 1, 0), radius=0.05, height=0.12))
        # rolled cylinders at the tips of the ROSA wings
        booms.append(pv.Cylinder(center=(0, y1, 0), direction=(1, 0, 0), radius=0.08, height=2.4, resolution=16))
    return [(bus.merge(antenna), "#c9a24c", 0.5), (arrays[0].merge(arrays[1]), "#1f3f86", 0.6),
            (camera.merge(booms), "#b9bcc2", 0.4), (lens, "#101418", 0.9), (nozzle, "#5d6066", 0.3)]


def attitude(forward, sun_dir, sun_axis="z"):
    """Rotation body -> ICRF with body +x along `forward` and the Sun as close as possible to body `sun_axis`."""
    x = np.asarray(forward, float)
    x = x / np.linalg.norm(x)
    s = np.asarray(sun_dir, float)
    s = s - (s @ x) * x
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(x, [0, 0, 1.0]) if abs(x[2]) < 0.9 else np.cross(x, [0, 1.0, 0])
    s = s / np.linalg.norm(s)
    if sun_axis == "z":
        z = s
        y = np.cross(z, x)
    else:  # sun on +y (OSIRIS-REx model: solar-cell side of the arrays = +y)
        y = s
        z = np.cross(x, y)
    return np.column_stack([x, y, z])
