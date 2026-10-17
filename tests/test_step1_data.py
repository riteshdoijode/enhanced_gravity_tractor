"""Step 1 validation: JPL ephemeris pipeline and Bennu shape-model physics."""

import json

import numpy as np
import pytest
import trimesh
from scipy.optimize import minimize_scalar

from core.assets import bennu_physics_mesh, bennu_props
from core.constants import AU, BENNU, DATA_DIR, DAY, HORIZONS_BODIES
from core.ephemeris import Ephemeris
from core.timeutil import iso_to_t

MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


@pytest.fixture(scope="module")
def eph():
    return Ephemeris()


# Max allowed interpolation error vs off-node Horizons samples [m]
@pytest.mark.parametrize("key,tol", [("earth", 100.0), ("jupiter_bary", 10.0), ("moon", 30.0),
                                     ("bennu", 50.0), ("bennu_ca2135", 50.0)])
def test_interpolation_vs_horizons(eph, key, tol):
    chk = np.load(DATA_DIR / "ephemeris_check.npz")
    body = key.split("_ca")[0]
    center = {"500@0": "ssb", "500@399": "earth", "500@10": "sun"}[HORIZONS_BODIES[body][1]]
    t, r = chk[f"{key}_t"], chk[f"{key}_r"]
    r_ours = eph.pos(body, t) if center == "ssb" else eph.rel_pos(body, center, t)
    assert np.linalg.norm(r_ours - r, axis=1).max() < tol


def _sbdb_approaches():
    rec = json.loads((DATA_DIR / "sbdb_bennu.json").read_text())
    out = []
    for ca in rec["ca_data"]:
        y, mon, rest = ca["cd"].split("-", 2)
        out.append((iso_to_t(f"{y}-{MONTHS[mon]}-{rest}"), float(ca["dist"]), ca["cd"]))
    return out


def test_close_approaches_match_sbdb(eph):
    """Earth-Bennu minimum distance and time vs SBDB's independently published approach table."""
    t_lo, t_hi = eph.span("bennu")
    checked = 0
    for tc, d_sb, label in _sbdb_approaches():
        if not (t_lo + 2 * DAY < tc < t_hi - 2 * DAY):
            continue
        res = minimize_scalar(lambda t: np.linalg.norm(eph.rel_pos("bennu", "earth", t)),
                              bounds=(tc - DAY, tc + DAY), method="bounded", options={"xatol": 1.0})
        assert abs(res.x - tc) < 90.0, label  # SBDB times are rounded to the minute
        assert abs(res.fun - d_sb * AU) < 3e3, label  # within 3 km
        checked += 1
    assert checked >= 30


def test_2135_keyhole_approach(eph):
    """The 2135 encounter that gates the 2182 impact possibility: 0.00136 au on 2135-09-25."""
    tc = iso_to_t("2135-09-25 07:49")
    res = minimize_scalar(lambda t: np.linalg.norm(eph.rel_pos("bennu", "earth", t)),
                          bounds=(tc - DAY, tc + DAY), method="bounded")
    assert abs(res.fun / AU - 0.00136393) < 1e-7
    assert res.fun < 384_400e3  # passes inside the Moon's mean distance


def test_shape_model_mass_matches_measured_gm():
    """Volume of the OSIRIS-REx shape model x SBDB bulk density reproduces the measured GM."""
    props = bennu_props()
    assert abs(props["gm_from_shape"] / BENNU["gm"] - 1.0) < 0.002
    # principal axis of maximum inertia is the spin axis (body z) for a principal-axis rotator
    axes = np.array(props["principal_axes_columns"])
    assert abs(axes[:, 2] @ np.array([0, 0, 1.0])) > 0.999


def test_physics_mesh_is_closed_volume():
    v, f = bennu_physics_mesh()
    tm = trimesh.Trimesh(v, f, process=False)
    assert tm.is_watertight and tm.is_winding_consistent and tm.is_volume
    assert tm.volume > 0  # outward normals
    assert abs(tm.volume / bennu_props()["volume_m3"] - 1.0) < 0.01
