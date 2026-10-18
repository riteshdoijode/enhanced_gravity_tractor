"""Step 5 validation: gravity near Bennu, hover control (LQR vs MPC), navigation (EKF), boulder capture
(MuJoCo), and the proximity-operations timeline flown in the live mission."""

import numpy as np
import pytest

from core.constants import BENNU, DAY, G
from core.control.dare import c2d, dare, kalman_gain
from core.control.prox import project_thrust, relative_model, weights
from core.dynamics.fastgrav import BennuGravity
from core.proximity.capture import Capture, merge_mass
from core.proximity.scenarios import station_keeping

MU = BENNU["gm"]


@pytest.fixture(scope="module")
def grav():
    return BennuGravity()


def test_compiled_polyhedron_matches_reference(grav):
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20, 3))
    pts *= (rng.uniform(300, 2500, 20) / np.linalg.norm(pts, axis=1))[:, None]
    ref = grav.poly.acceleration(pts)
    for x, a_ref in zip(pts, ref):
        a, Gm, lap, _ = grav.eval_body(x)
        assert np.linalg.norm(a - a_ref) < 1e-12 * np.linalg.norm(a_ref)
        h = 1e-3
        G_fd = np.column_stack([(grav.accel_body(x + h * e) - grav.accel_body(x - h * e)) / (2 * h) for e in np.eye(3)])
        assert np.abs(Gm - G_fd).max() < 1e-6 * np.abs(Gm).max()
        assert np.allclose(Gm, Gm.T, atol=1e-20) and abs(np.trace(Gm)) < 1e-18 and abs(lap) < 1e-18
    _, Gin, lap_in, _ = grav.eval_body(np.array([20.0, -30.0, 10.0]))
    assert np.isclose(lap_in, -4 * np.pi * G * grav.poly.density, rtol=1e-9)
    assert np.isclose(np.trace(Gin), lap_in, rtol=1e-9)


def test_hover_linearization_eigenvalues():
    """Point-mass hover: eigenvalues +-sqrt(2 mu/d^3) (unstable, radial) and +-i sqrt(mu/d^3) (twice)."""
    d = 485.0
    e = np.array([0.0, 0.6, 0.8])
    G = MU / d**3 * (3 * np.outer(e, e) - np.eye(3))
    Ac = np.block([[np.zeros((3, 3)), np.eye(3)], [G, np.zeros((3, 3))]])
    lam = np.linalg.eigvals(Ac)
    w = np.sqrt(MU / d**3)
    expected = np.array([-np.sqrt(2) * w, np.sqrt(2) * w, 1j * w, 1j * w, -1j * w, -1j * w])
    assert all(np.min(np.abs(lam - x)) < 1e-12 for x in expected)
    assert all(np.min(np.abs(expected - x)) < 1e-12 for x in lam)
    assert 50 < 1 / (np.sqrt(2) * w) / 60 < 70  # e-folding ~ 1 hour at 2 Bennu radii


def test_lqr_kalman_separation_principle():
    """Lecture 3: rho(A - BK) < 1, rho(A - LC) < 1, and the closed loop with the observer has exactly
    eig(A - BK) U eig(A - LC)."""
    d, dt = 485.0, 60.0
    e = np.array([1.0, 0, 0])
    A, B = relative_model(MU / d**3 * (3 * np.outer(e, e) - np.eye(3)), dt)
    Q, R = weights(3e-3)
    _, K = dare(A, B, Q, R)
    C = np.hstack([np.eye(3), np.zeros((3, 3))])
    L = kalman_gain(A, C, 1e-10 * np.eye(6), 0.1**2 * np.eye(3))
    assert max(abs(np.linalg.eigvals(A - B @ K))) < 1 and max(abs(np.linalg.eigvals(A - L @ C))) < 1
    big = np.block([[A - B @ K, B @ K], [np.zeros((6, 6)), A - L @ C]])  # [x; estimation error]
    lam = np.linalg.eigvals(big)
    expected = np.concatenate([np.linalg.eigvals(A - B @ K), np.linalg.eigvals(A - L @ C)])
    assert all(np.min(np.abs(lam - x)) < 1e-9 for x in expected)
    assert all(np.min(np.abs(expected - x)) < 1e-9 for x in lam)
    # ZOH discretization equals the matrix exponential series for a double integrator
    Ad, Bd = c2d(np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 6))]]), np.vstack([np.zeros((3, 3)), np.eye(3)]), dt)
    assert np.allclose(Bd[:3], dt**2 / 2 * np.eye(3)) and np.allclose(Ad[:3, 3:], dt * np.eye(3))


def test_thrust_projection_onto_plume_cone():
    e = np.array([0.0, 0.0, 1.0])
    g = np.radians(40)
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.normal(size=3)
        p = project_thrust(u, e, g, 10.0)
        if np.linalg.norm(p) > 1e-12:
            assert np.arccos(p @ e / np.linalg.norm(p)) <= g + 1e-9
        inside = np.arccos(np.clip(u @ e / np.linalg.norm(u), -1, 1)) <= g
        if inside:
            assert np.allclose(p, u)


def test_station_keeping_lqr_vs_mpc(grav):
    """Full rotating polyhedron field + solar pressure, 2 N on 30 t, plume cone 56 deg: both settle; the
    MPC respects the cone by construction, settles faster with a lower quadratic cost."""
    lqr = station_keeping("LQR", hours=6, grav=grav)
    mpc = station_keeping("MPC", hours=6, grav=grav)
    for r in (lqr, mpc):
        assert r["final_err"] < 0.5
        assert r["angle"].max() <= r["gamma_deg"] + 1e-6
        assert r["min_range"] > grav.R_bound + 30
    assert np.mean(lqr["raw_angle"] > lqr["gamma_deg"]) > 0.2  # unconstrained LQR wants forbidden directions
    assert mpc["cost"] < 0.6 * lqr["cost"] and mpc["settle_h"] < lqr["settle_h"]
    assert mpc["fallbacks"] == 0


def test_navigation_filter_consistent_in_hover(grav):
    """EKF on raw LIDAR range + camera bearing, flying the loop on its own estimate."""
    r = station_keeping("LQR", hours=10, grav=grav, navigation=True, seed=3)
    inside = r["nav_err"] < 3 * r["nav_sigma"]
    assert inside.mean() > 0.95
    assert r["nav_err"][-60:].max() < 0.5 and r["final_err"] < 1.0


def test_merge_mass_parallel_axis():
    rng = np.random.default_rng(2)
    clouds = [rng.normal(c, s, (4000, 3)) for c, s in [((0, 0, 0), 1.0), ((0, 0, -4), 0.6), ((2, 1, -6), 0.3)]]
    masses = [10_000.0, 300.0, 20_000.0]

    def inertia(pts, m, about):
        r = pts - about
        w = m / len(pts)
        return w * (np.sum(r * r) * np.eye(3) - r.T @ r)

    parts = [(m, P.mean(0), inertia(P, m, P.mean(0))) for P, m in zip(clouds, masses)]
    merged = merge_mass(parts)
    allp = np.vstack([P for P in clouds])
    w = np.concatenate([np.full(len(P), m / len(P)) for P, m in zip(clouds, masses)])
    com = (w[:, None] * allp).sum(0) / w.sum()
    r = allp - com
    J = np.einsum("n,nij->ij", w, np.einsum("ni,ni->n", r, r)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", r, r))
    assert np.isclose(merged["mass"], sum(masses)) and np.allclose(merged["com"], com)
    assert np.allclose(merged["inertia"], J, rtol=1e-9)
    ev = np.linalg.eigvalsh(merged["inertia"])
    assert ev.min() > 0 and ev[0] + ev[1] >= ev[2]


def test_free_floating_capture_conserves_momentum():
    """No gravity, no terrain, no Coriolis, thrusters off: arm deploys, grasps and stows a floating
    boulder; every force is internal, so total linear momentum stays zero."""
    cap = Capture(10_000.0, 20_000.0, thrust_max=0.0, gravity=np.zeros(3), terrain=False, coriolis=False,
                  omega_local=np.zeros(3))
    p_max = 0.0
    for _ in range(60):
        cap.advance(10.0)
        p_max = max(p_max, np.linalg.norm(cap.momentum()))
        if cap.stage == "lift":
            break
    assert cap.attached  # grasped
    m_tot = 30_000.0
    assert p_max < 1e-6 * m_tot  # < 1 micrometre/s of center-of-mass drift


def test_capture_lifts_boulder_with_computed_torque():
    cap = Capture(10_000.0, 20_000.0, thrust_max=20.0)
    res = cap.run()
    assert res["success"] and res["lifted_m"] > 3.0 and res["grip_slip_m"] < 0.1
    assert res["peak_tracking_rad"] < 0.02  # free-space arm motion tracks its min-jerk reference
    assert np.isclose(res["mass"]["mass"], 30_000.0, rtol=1e-6)
    assert cap.retries == 0


@pytest.fixture(scope="module")
def planner():
    from tests.keyhole import setup

    return setup()[2]


def test_proximity_timeline_in_live_mission(planner):
    """Hand-over 30 km out -> approach -> LIDAR scan for boulders -> the recommended stone (headless run) ->
    survey -> MPC descent in the glide-slope cone -> MuJoCo capture -> ascent -> transfer -> tractor station, all
    on the EKF estimate in the full model."""
    import time

    from core.dynamics.asteroid import bennu_attitude
    from core.mission import LiveMission
    from core.planner import MissionParams
    from core.proximity.ops import GLIDESLOPE, ProximityOps

    from core.timeutil import iso_to_t

    p = MissionParams(tractor_thrust_n=20.0)
    tp, _ = planner.plan_tractor(p, iso_to_t("2031-01-01"))
    t0 = tp.t_arrive
    m = LiveMission(planner.eph, planner.nom, planner, p, t0, seed=4)  # proximity only (no DART on its way)
    m.tractor_plan = tp
    tr = m.tractor
    m.t = tr.t = t0
    xb = m.bennu_helio(t0)[0]
    rho0 = -30e3 * xb[:3] / np.linalg.norm(xb[:3])
    v0 = np.array([0.2, -0.1, 0.15])
    tr.phase, tr.t_arrived, tr.arrival_offset = "proximity", t0, rho0
    tr.prox = pr = ProximityOps(m, tr, t0, rho0, v0)
    wall = time.time()
    while not m.on_station() and m.t < t0 + 10 * DAY:
        m.advance(m.t + DAY / 4, max_chunk=600.0)
    assert time.time() - wall < 240
    assert pr.phase == "station" and pr.has_boulder and pr.capture_result["success"]
    texts = [e.text for e in m.events]
    order = ["approach to the scan point", "LIDAR scan", "Scan complete", "approach to the survey point", "Surveying",
             "Descent", "Capture: hovering", "Boulder captured", "Ascent", "Transfer", "On station"]
    idx = [next(i for i, s in enumerate(texts) if key in s) for key in order]
    assert idx == sorted(idx)
    assert np.linalg.norm(pr.x[:3] - pr.d_station * pr.e_station) < 1.0
    H = pr.hist
    inside = np.array(H["nav_err"]) < 3 * np.array(H["nav_sigma"])
    assert inside.mean() > 0.95
    # descent stayed inside the glide-slope cone over the site
    t_desc = next(e.t for e in m.events if e.text.startswith("Descent"))
    t_cap = next(e.t for e in m.events if e.text.startswith("Capture: hovering"))
    ang = []
    for t, rho in pr.trail:
        if t_desc + 600 < t < t_cap:
            rel = bennu_attitude(t).T @ rho - pr.site_body
            ang.append(np.degrees(np.arccos(rel @ pr.site.up / np.linalg.norm(rel))))
    assert len(ang) > 50 and max(ang) < np.degrees(GLIDESLOPE)
