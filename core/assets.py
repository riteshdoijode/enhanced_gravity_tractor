"""3D asset processing and loading.

Bennu: NASA SVS #5069 OSIRIS-REx OLA global shape model v20 (88 cm, 3.37 M faces, watertight).
The raw glTF geometry is in Bennu's body-fixed frame (z = spin pole, km); the file's scene node
only adds the glTF Y-up rotation, so we read the raw vertices and skip that node transform.
Outputs (meters, origin shifted to the center of mass of a uniform-density body):
    assets/bennu_display.vtp    ~400k faces, for rendering
    assets/bennu_physics.npz    5120 faces, watertight (radially remeshed), for polyhedron gravity
    assets/bennu_props.json     volume, implied GM, inertia tensor, principal axes, dimensions
Nightingale TAG site (OLA v20 local model, 5 cm, 56 m x 56 m), the boulder-capture site (Step 5):
    assets/tagsite_display.vtp  ~300k faces, Bennu body frame, for close-up rendering and the LIDAR
    assets/tagsite.npz          local site frame (east, north, up = -effective gravity) and a 10 cm
                                height grid of the central 40 m x 40 m (MuJoCo heightfield)
"""

import json
import shutil

import numpy as np
import pyvista as pv
import trimesh

from core.constants import ASSET_DIR, BENNU, G

DISPLAY_FACES = 400_000
PHYSICS_ICO_SUBDIV = 4  # geodesic icosphere: 2562 vertices, 5120 faces


def _decimate(mesh: pv.PolyData, n_faces: int) -> pv.PolyData:
    reduction = 1.0 - n_faces / mesh.n_cells
    return mesh.decimate(reduction).clean().triangulate()


def _radial_remesh(surface: pv.PolyData, subdivisions: int) -> trimesh.Trimesh:
    """Watertight genus-0 mesh: project icosphere vertices radially onto the outermost surface.

    VTK quadric decimation leaves a few non-manifold edges on this model, which breaks the
    polyhedron gravity formulas; a radially projected icosphere is closed by construction.
    """
    import vtk

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    ico = trimesh.creation.icosphere(subdivisions=subdivisions)
    far = 2.0 * float(np.linalg.norm(surface.points, axis=1).max())
    t, x, pcoords, sub_id = vtk.reference(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], vtk.reference(0)
    rng = np.random.default_rng(0)
    verts = np.empty_like(ico.vertices)
    for i, d0 in enumerate(ico.vertices):
        d = d0
        for _ in range(10):  # ray from outside inward: first hit = outermost surface
            if locator.IntersectWithLine(tuple(far * d), (0.0, 0.0, 0.0), 1e-9, t, x, pcoords, sub_id):
                break
            d = d0 + 1e-5 * rng.standard_normal(3)  # grazing an edge can miss; jitter and retry
            d /= np.linalg.norm(d)
        else:
            raise RuntimeError(f"ray {i} never hit the surface")
        verts[i] = d0 * np.linalg.norm(x)
    tm = trimesh.Trimesh(verts, ico.faces, process=True)
    trimesh.repair.fix_normals(tm)
    return tm


def process_bennu(raw_glb):
    scene = trimesh.load(raw_glb)
    raw = list(scene.geometry.values())[0]  # body-fixed frame, km (no node transform)
    full = trimesh.Trimesh(raw.vertices * 1e3, raw.faces, process=True)
    full.density = BENNU["density"]
    com = full.center_mass.copy()
    full.apply_translation(-com)
    inertia = full.moment_inertia
    moments, axes = np.linalg.eigh(inertia)
    along = full.vertices @ axes

    pv_full = pv.PolyData.from_regular_faces(full.vertices, full.faces)
    display = _decimate(pv_full, DISPLAY_FACES).compute_normals(auto_orient_normals=True, split_vertices=False)
    display.save(ASSET_DIR / "bennu_display.vtp")

    tm = _radial_remesh(pv_full, PHYSICS_ICO_SUBDIV)
    np.savez(ASSET_DIR / "bennu_physics.npz", vertices=tm.vertices, faces=tm.faces)

    props = {
        "source": "NASA SVS 5069: g_00880mm_alt_ptm_0000n00000_v020 (OSIRIS-REx OLA v20)",
        "frame": "Bennu body-fixed, z = spin pole, origin = center of mass (uniform density)",
        "com_offset_from_model_origin_m": com.tolist(),
        "volume_m3": float(full.volume),
        "density_kg_m3": BENNU["density"],
        "mass_kg": float(full.mass),
        "gm_from_shape": float(G * full.mass),
        "gm_measured": BENNU["gm"],
        "inertia_kg_m2": inertia.tolist(),
        "principal_moments_kg_m2": moments.tolist(),
        "principal_axes_columns": axes.tolist(),
        "dimensions_along_principal_axes_m": (along.max(0) - along.min(0)).tolist(),
        "faces": {"full": int(len(full.faces)), "display": int(display.n_cells), "physics": int(len(tm.faces))},
        "physics_mesh": {"watertight": bool(tm.is_watertight), "volume_m3": float(tm.volume),
                         "is_volume": bool(tm.is_volume)},
    }
    (ASSET_DIR / "bennu_props.json").write_text(json.dumps(props, indent=2))
    return props


def process_spacecraft(raw_glb):
    """Copy the OSIRIS-REx glTF (meters, glTF Y-up) and record its size."""
    dest = ASSET_DIR / "osiris_rex.glb"
    shutil.copy(raw_glb, dest)
    bounds = trimesh.load(raw_glb).bounds
    props = {
        "source": "NASA Science 3D Resources: OSIRIS-REx 3D model (VTAD), glTF",
        "bounds_m": bounds.tolist(),
        "extent_m": (bounds[1] - bounds[0]).tolist(),
    }
    (ASSET_DIR / "spacecraft_props.json").write_text(json.dumps(props, indent=2))
    return props


TAGSITE_DISPLAY_FACES = 300_000
TAGSITE_GRID_HALF = 20.0  # m
TAGSITE_GRID_STEP = 0.1  # m


def process_tagsite(raw_glb):
    import vtk

    from core.dynamics import rotating
    from core.dynamics.polyhedron import Polyhedron

    scene = trimesh.load(raw_glb)
    raw = list(scene.geometry.values())[0]  # same body-fixed frame and units (km) as the global model
    com = np.array(bennu_props()["com_offset_from_model_origin_m"])
    verts = raw.vertices * 1e3 - com
    patch = pv.PolyData.from_regular_faces(verts, raw.faces)
    center = verts.mean(0)
    poly = Polyhedron(*bennu_physics_mesh())
    g_eff = rotating.effective_gravity(center, lambda x: poly.acceleration(x))
    up = -g_eff / np.linalg.norm(g_eff)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    R = np.column_stack([east, north, up])  # local -> body

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(patch)
    locator.BuildLocator()
    xs = np.arange(-TAGSITE_GRID_HALF, TAGSITE_GRID_HALF + 1e-9, TAGSITE_GRID_STEP)
    H = np.full((len(xs), len(xs)), np.nan)  # H[iy, ix]
    t, x, pc, sid = vtk.reference(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], vtk.reference(0)
    for iy, yy in enumerate(xs):
        for ix, xx in enumerate(xs):
            a = center + R @ np.array([xx, yy, 30.0])
            b = center + R @ np.array([xx, yy, -30.0])
            if locator.IntersectWithLine(tuple(a), tuple(b), 1e-9, t, x, pc, sid):
                H[iy, ix] = (np.asarray(x) - center) @ up
    if np.isnan(H).any():  # fill rare misses from neighbours
        from scipy.ndimage import generic_filter

        H = np.where(np.isnan(H), generic_filter(H, np.nanmean, size=3), H)
    display = _decimate(patch, TAGSITE_DISPLAY_FACES).compute_normals(auto_orient_normals=True, split_vertices=False)
    display.save(ASSET_DIR / "tagsite_display.vtp")
    np.savez(ASSET_DIR / "tagsite.npz", origin=center, R=R, xs=xs, heights=H, g_eff=g_eff)
    return {"origin_m": center.tolist(), "radius_m": float(np.linalg.norm(center)),
            "lat_deg": float(np.degrees(np.arcsin(center[2] / np.linalg.norm(center)))),
            "lon_deg": float(np.degrees(np.arctan2(center[1], center[0]))),
            "g_eff_m_s2": float(np.linalg.norm(g_eff)), "height_range_m": [float(H.min()), float(H.max())]}


def tagsite():
    return dict(np.load(ASSET_DIR / "tagsite.npz"))


def tagsite_display_mesh() -> pv.PolyData:
    return pv.read(ASSET_DIR / "tagsite_display.vtp")


def bennu_display_mesh() -> pv.PolyData:
    return pv.read(ASSET_DIR / "bennu_display.vtp")


def bennu_physics_mesh():
    d = np.load(ASSET_DIR / "bennu_physics.npz")
    return d["vertices"], d["faces"]


def bennu_props() -> dict:
    return json.loads((ASSET_DIR / "bennu_props.json").read_text())


def spacecraft_glb():
    return ASSET_DIR / "osiris_rex.glb"


def bennu_surface_gravity(display=None, cache=True):
    """Effective surface acceleration magnitude (polyhedron gravity + spin) on the display mesh [m/s^2].

    Evaluated 1 m above each vertex of the 5120-face physics mesh, then interpolated to the display
    mesh (closest point); cached to assets/bennu_display_geff.npy.
    """
    path = ASSET_DIR / "bennu_display_geff.npy"
    if cache and path.exists():
        return np.load(path)
    from core.dynamics import rotating
    from core.dynamics.polyhedron import Polyhedron

    poly = Polyhedron(*bennu_physics_mesh())
    ico = pv.PolyData.from_regular_faces(poly.V, poly.F).compute_normals(auto_orient_normals=True)
    pts = np.asarray(ico.points) + np.asarray(ico.point_normals)
    g = poly.acceleration(pts)
    ico["g_eff"] = np.linalg.norm(g - np.cross(rotating.OMEGA_BODY, np.cross(rotating.OMEGA_BODY, pts)), axis=1)
    display = bennu_display_mesh() if display is None else display
    values = np.asarray(display.interpolate(ico, sharpness=4, radius=60.0, strategy="closest_point")["g_eff"])
    if cache:
        np.save(path, values)
    return values
