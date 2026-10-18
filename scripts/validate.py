"""Validation figures for the report -> outs/validation/*.png

    uv run python -m scripts.validate              # all figures, in parallel processes (~2 min)
    uv run python -m scripts.validate --only integrators bennu_jpl stm gravity checks      (Step 2)
    uv run python -m scripts.validate --only prox_hover prox_ops capture                   (Step 5)
    uv run python -m scripts.validate --only keyhole opportunities orbit_raising tow       (keyhole mission)
"""

import argparse
import json

import numpy as np

from core import plotstyle as ps
from core.assets import bennu_display_mesh, bennu_physics_mesh
from core.constants import AU, BENNU, DATA_DIR, DAY, EARTH_J2, GM, RADIUS, ROOT
from core.dynamics import attitude as att
from core.dynamics import rotating
from core.dynamics.asteroid import LIT_A2, LIT_AMRAT, BennuDynamics
from core.dynamics.nominal import NominalBennu
from core.dynamics.perturbations import air_density
from core.dynamics.polyhedron import Polyhedron
from core.dynamics.spacecraft import SpacecraftDynamics, SpacecraftParams, initial_state
from core.ephemeris import Ephemeris
from core.integrate import euler_step, propagate, rk4_step
from core.timeutil import iso_to_t, t_to_year

OUT = ROOT / "outs" / "validation"
MU = GM["earth"]
T0 = iso_to_t("2030-03-20 00:00")


def _two_body(eph, **kw):
    opts = dict(perturbers=[], j2=False, use_drag=False, use_srp=False, gravity_gradient_torque=False)
    opts.update(kw)
    return SpacecraftDynamics(eph, "earth", **opts)


def _circular(alt, inc):
    a = RADIUS["earth"] + alt
    v = np.sqrt(MU / a)
    return a, np.array([a, 0, 0.0]), np.array([0, v * np.cos(np.radians(inc)), v * np.sin(np.radians(inc))])


def fig_integrators(eph):
    a, r0, v0 = _circular(400e3, 51.6)
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    f = _two_body(eph).rhs
    fig, ax = ps.figure(figsize=(7.2, 4.2))
    runs = [("RK4, 10 s step", rk4_step, 10.0, "-"), ("Forward Euler, 1 s step", euler_step, 1.0, "-"),
            ("RK4, 60 s step", rk4_step, 60.0, "--"), ("Forward Euler, 10 s step", euler_step, 10.0, "--")]
    for (label, step, dt, ls), color in zip(runs, ps.SERIES):
        n = int(10 * T / dt)
        ts, xs, _ = propagate(f, T0, initial_state(r0, v0), dt, n, step=step, every=max(1, n // 400))
        E = 0.5 * np.sum(xs[:, 3:6] ** 2, 1) - MU / np.linalg.norm(xs[:, :3], axis=1)
        err = np.abs(E / E[0] - 1)[1:]
        orbits = (ts[1:] - T0) / T
        ax.plot(orbits, err, ls=ls, color=color, label=label)
        ps.end_label(ax, orbits[-1], err[-1], f"{err[-1]:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("Orbits (400 km circular, 51.6° inclination)")
    ax.set_ylabel("Relative energy error |ΔE/E|")
    ax.set_title("Forward Euler (lecture code) gains energy every orbit; RK4 holds it", pad=20)
    ps.subtitle(ax, "Two-body LEO propagation, 10 orbits. The simulator steps with RK4.")
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.80))
    ax.set_xlim(0, 11.3)
    return fig


def fig_bennu_jpl(eph):
    t0 = iso_to_t("2026-01-01 00:00")
    x0 = eph.state("bennu", t0)
    lo, hi = eph.span("bennu")
    ts = np.arange(t0 + 30 * DAY, hi, 30 * DAY)
    truth = eph.pos("bennu", ts)
    fit = json.loads((DATA_DIR / "bennu_nongrav_fit.json").read_text())
    runs = [("Our model, A2 & SRP fitted to JPL 2026–2059", {}),
            ("Published A2 (Chesley et al. 2014), no fit", {"a2": LIT_A2, "amrat": LIT_AMRAT}),
            ("Without Yarkovsky drift", {"a2": 0.0}),
            ("Without general relativity", {"gr": False})]
    fig, ax = ps.figure(figsize=(8.0, 4.6))
    years = t_to_year(ts)
    for (label, kw), color in zip(runs, ps.SERIES):
        xs, _ = BennuDynamics(eph, **kw).propagate(t0, x0, ts)
        err = np.linalg.norm(xs[:, :3] - truth, axis=1) / 1e3
        ax.plot(years, np.maximum(err, 1e-2), color=color, label=label, lw=2.0 if not kw else 1.6)
    sb = {c["cd"][:4]: c for c in json.loads((DATA_DIR / "sbdb_bennu.json").read_text())["ca_data"]}
    sig_x, sig_y = [], []
    for year in ["2060", "2080", "2087", "2135"]:
        c = sb[year]
        y = t_to_year(iso_to_t(f"{year}-{c['cd'][5:]}".replace("Sep", "09").replace("Oct", "10")))
        sig_x.append(y)
        sig_y.append((float(c["dist_max"]) - float(c["dist_min"])) / 2 * AU / 1e3)
    ax.scatter(sig_x, sig_y, marker="_", s=260, color=ps.INK, linewidths=2.2, zorder=5,
               label="JPL's own 3σ uncertainty at Earth approaches")
    for y, lab in [(2060.73, "2060 flyby, 0.005 au"), (2135.73, "2135 flyby,\n0.0014 au")]:
        ax.axvline(y, color=ps.AXIS, lw=0.8, zorder=0)
        ax.text(y - 0.8, 0.15, lab, ha="right", va="bottom", fontsize=8, color=ps.INK2)
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 2e8)
    ax.set_xlim(2026, 2137)
    ax.set_xlabel("Year")
    ax.set_ylabel("Position difference from JPL Horizons [km]")
    ax.set_title("Bennu propagated 110 years agrees with JPL within JPL's own uncertainty", pad=20)
    ps.subtitle(ax, f"Fitted A2 = {fit['a2']:.3e} au/d² (published −4.618e-14). Earth flybys amplify "
                    "any model difference ~100×.")
    ax.legend(loc="upper left", fontsize=8.5)
    return fig


def fig_stm(eph):
    nom = NominalBennu()
    t1 = iso_to_t("2032-06-01")
    x1 = nom.state(t1)
    dv_mag = 1e-4
    dv = dv_mag * x1[3:] / np.linalg.norm(x1[3:])
    ts = np.arange(t1 + 30 * DAY, nom.span[1] - 2 * DAY, 30 * DAY)
    dyn = BennuDynamics(eph)
    xa, _ = dyn.propagate(t1, x1, ts)
    xb, _ = dyn.propagate(t1, x1 + np.r_[0, 0, 0, dv], ts)
    nonlin = np.linalg.norm(xb[:, :3] - xa[:, :3], axis=1) / 1e3
    lin = np.array([np.linalg.norm(nom.dr_from_dv(t, t1, dv)) for t in ts]) / 1e3
    cw = 3 * dv_mag * (ts - t1) / 1e3
    years = t_to_year(ts)
    fig, ax = ps.figure(figsize=(7.6, 4.4))
    ax.plot(years, nonlin, color=ps.SERIES[0], label="Full nonlinear N-body (two propagations)")
    ax.plot(years, lin, color=ps.SERIES[1], ls="--", label="State-transition matrix (linearized)")
    ax.plot(years, cw, color=ps.SERIES[2], ls=":", lw=2.2, label="Clohessy–Wiltshire secular drift 3Δv·t")
    ax.axvline(2060.73, color=ps.AXIS, lw=0.8, zorder=0)
    ax.text(2061.5, 1.5e1, "2060 Earth flyby\namplifies the shift ~100×", fontsize=8, color=ps.INK2, va="bottom")
    ax.text(2093, 3e6, "shift ~10⁶ km: linearization\nbreaks down", fontsize=8, color=ps.INK2, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Change in Bennu's position [km]")
    ax.set_title("A 0.1 mm/s push in 2032: the linearized STM tracks full N-body through two flybys", pad=20)
    ps.subtitle(ax, "Along-track Δv on 2032-06-01. Error 1.5e-4 at 2060; Clohessy–Wiltshire 3Δv·t holds until the "
                    "first flyby.")
    ax.legend(loc="lower right")
    return fig


def fig_gravity():
    """Effective surface acceleration (polyhedron gravity + spin) rendered on the OSIRIS-REx shape model."""
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap

    from core.assets import bennu_surface_gravity

    disp = bennu_display_mesh()
    disp["g_eff"] = bennu_surface_gravity(disp) * 1e6  # micro-m/s^2
    mag = np.asarray(disp["g_eff"])
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ps.BLUE_RAMP, N=256)  # continuous, one hue
    pl = pv.Plotter(off_screen=True, window_size=(1600, 980), shape=(1, 2), border=False)
    for col, (view, title) in enumerate([("xz", "Equatorial view"), ("xy", "View down the spin axis")]):
        pl.subplot(0, col)
        pl.set_background(ps.SURFACE)
        pl.add_mesh(disp, scalars="g_eff", cmap=cmap, clim=(float(mag.min()), float(mag.max())),
                    smooth_shading=True, show_scalar_bar=(col == 1), ambient=0.25,
                    scalar_bar_args=dict(title="Effective surface gravity [micro-m/s^2]\n", color=ps.INK,
                                         fmt="%.0f", vertical=False, position_x=0.2, position_y=0.03,
                                         width=0.6, height=0.12, title_font_size=16, label_font_size=14,
                                         n_labels=5))
        pl.add_text(title, font_size=12, color=ps.INK)
        pl.camera_position = view
        pl.reset_camera()
        pl.camera.zoom(1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bennu_surface_gravity.png"
    pl.screenshot(str(path))
    pl.close()
    z = np.asarray(disp.points)[:, 2]
    stats = {"min": float(mag.min()), "max": float(mag.max()), "equator_mean": float(mag[np.abs(z) < 40].mean()),
             "pole_mean": float(mag[np.abs(z) > 200].mean())}
    return path, stats


def fig_checks(eph):
    """Small multiples: J2 node drift, drag decay, Jacobi constant, torque-free invariants."""
    fig, axs = ps.figure(2, 2, figsize=(9.0, 6.6))
    # J2 node drift
    a, r0, v0 = _circular(400e3, 51.6)
    ts, xs, _ = propagate(_two_body(eph, j2=True).rhs, T0, initial_state(r0, v0), 30.0, int(5 * DAY / 30), every=10)
    h = np.cross(xs[:, :3], xs[:, 3:6])
    node = np.degrees(np.unwrap(np.arctan2(h[:, 0], -h[:, 1])))
    rate = -1.5 * np.sqrt(MU / a**3) * EARTH_J2 * (RADIUS["earth"] / a) ** 2 * np.cos(np.radians(51.6))
    d = (ts - T0) / DAY
    ax = axs[0, 0]
    ax.plot(d, node - node[0], color=ps.SERIES[0], label="Simulated (osculating)")
    ax.plot(d, np.degrees(rate) * (ts - T0), color=ps.SERIES[1], ls="--", label="Analytic −1.5 n J₂ (R/p)² cos i")
    ax.set_title("J2 nodal regression, ISS-like orbit")
    ax.set_xlabel("Days")
    ax.set_ylabel("Node change [deg]")
    ax.legend(loc="lower left")
    # drag decay
    a, r0, v0 = _circular(400e3, 90.0)
    p = SpacecraftParams(mass=1000.0, drag_area=10.0, cd=2.2)
    ts, xs, _ = propagate(_two_body(eph, params=p, use_drag=True).rhs, T0, initial_state(r0, v0, mass=1000.0),
                          10.0, int(DAY / 10), every=6)
    E = 0.5 * np.sum(xs[:, 3:6] ** 2, 1) - MU / np.linalg.norm(xs[:, :3], axis=1)
    sma = -MU / (2 * E)
    rate = -air_density(400e3) * p.cd * p.drag_area / p.mass * np.sqrt(MU * a)
    ax = axs[0, 1]
    hrs = (ts - T0) / 3600
    ax.plot(hrs, sma - a, color=ps.SERIES[0], lw=1.2, label="Simulated semi-major axis")
    ax.plot(hrs, rate * (ts - T0), color=ps.SERIES[1], ls="--", label="Analytic −ρ (C_D A/m) √(μa) t")
    ax.set_title("Atmospheric drag decay at 400 km")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Δa [m]")
    ax.legend(loc="lower left")
    # Jacobi constant around Bennu
    poly = Polyhedron(*bennu_physics_mesh())
    U = lambda r: poly.potential(r)[0]
    r0 = np.array([0.0, 1000.0, 120.0])
    x = np.concatenate([r0, [-np.sqrt(BENNU["gm"] / 1000.0), 0, 0] - np.cross(rotating.OMEGA_BODY, r0)])
    C0, dt, Cs, tt = rotating.jacobi(x, U), 30.0, [], []
    for k in range(int(12 * 3600 / dt)):
        x = rk4_step(lambda t, x, u: rotating.rhs(t, x, poly.acceleration), k * dt, x, dt)
        if k % 20 == 0:
            Cs.append(abs(rotating.jacobi(x, U) / C0 - 1))
            tt.append((k + 1) * dt / 3600)
    ax = axs[1, 0]
    ax.plot(tt, np.maximum(Cs, 1e-16), color=ps.SERIES[0])
    ax.set_yscale("log")
    ax.set_title("Bennu orbit, polyhedron gravity: Jacobi constant")
    ax.set_xlabel("Hours (rotating frame, 1 km orbit)")
    ax.set_ylabel("Relative drift")
    # torque-free rigid body
    J = np.diag([2.4e4, 2.1e4, 1.8e4])
    Ji = np.linalg.inv(J)
    y = np.concatenate([[1.0, 0, 0, 0], [0.10, 0.02, 0.05]])
    H0, K0 = att.q_to_R(y[:4]) @ J @ y[4:], 0.5 * y[4:] @ J @ y[4:]
    dH, dK, tq = [], [], []
    for k in range(12000):
        y = rk4_step(lambda t, y, u: np.concatenate([att.q_dot(y[:4], y[4:]), att.omega_dot(y[4:], J, np.zeros(3), Ji)]),
                     0, y, 0.05)
        y[:4] /= np.linalg.norm(y[:4])
        if k % 50 == 0:
            dH.append(np.linalg.norm(att.q_to_R(y[:4]) @ J @ y[4:] - H0) / np.linalg.norm(H0))
            dK.append(abs(0.5 * y[4:] @ J @ y[4:] / K0 - 1))
            tq.append(k * 0.05)
    ax = axs[1, 1]
    ax.plot(tq, np.maximum(dH, 1e-17), color=ps.SERIES[0], label="Angular momentum (inertial)")
    ax.plot(tq, np.maximum(dK, 1e-17), color=ps.SERIES[1], ls="--", label="Rotational kinetic energy")
    ax.set_yscale("log")
    ax.set_title("Tumbling spacecraft, quaternion attitude")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Relative drift")
    ax.set_ylim(1e-18, 1e-11)
    ax.legend(loc="lower right")
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    return fig


def _stage_bands(ax, t, stages, x_scale=1.0):
    """Shade alternate phases lightly and name them inside the top of the plot (staggered)."""
    t = np.asarray(t) * x_scale
    edges = [0] + [i for i in range(1, len(stages)) if stages[i] != stages[i - 1]] + [len(stages)]
    for k in range(len(edges) - 1):
        a, b = t[edges[k]], t[min(edges[k + 1], len(t) - 1)]
        if k % 2 == 0:
            ax.axvspan(a, b, color=ps.GRID, alpha=0.45, lw=0)
        ax.text(0.5 * (a + b), 0.97 if k % 2 == 0 else 0.91, stages[edges[k]], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, color=ps.INK2)


def fig_prox_hover(eph):
    """LQR vs MPC acquiring the gravity-tractor station (2 N, 30 t, rotating polyhedron field)."""
    from core.dynamics.fastgrav import BennuGravity
    from core.proximity.scenarios import station_keeping

    g = BennuGravity()
    runs = {k: station_keeping(k, hours=6, grav=g) for k in ("LQR", "MPC")}
    fig, axs = ps.figure(1, 3, figsize=(13.0, 3.9))
    for (k, r), c in zip(runs.items(), ps.SERIES):
        th = r["t"] / 3600
        err = np.linalg.norm(r["dp"], axis=1)
        axs[0].plot(th, err, color=c, label=f"{k}: settles in {r['settle_h']:.1f} h, cost {r['cost']:.2g} m²s")
        axs[1].plot(th, r["angle"], color=c, label=f"{k} applied")
        axs[2].plot(th, np.linalg.norm(r["u"], axis=1), color=c, label=f"{k}: Δv {r['dv']:.2f} m/s")
    r = runs["LQR"]
    axs[1].plot(r["t"] / 3600, r["raw_angle"], color=ps.SERIES[0], lw=1.0, ls=":", label="LQR before clipping")
    axs[1].axhline(r["gamma_deg"], color=ps.INK2, lw=1.0, ls="--")
    ps.end_label(axs[1], 0.0, r["gamma_deg"], "plume cone (56°)", dx=0, dy=8)
    axs[2].axhline(2.0, color=ps.INK2, lw=1.0, ls="--")
    ps.end_label(axs[2], 0.1, 2.0, "2 N available", dx=0, dy=-8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Distance from the station [m]")
    axs[1].set_ylabel("Thrust angle from the outward axis [deg]")
    axs[2].set_ylabel("Thrust [N]")
    titles = ["Station acquisition", "Plume-cone constraint", "Thrust used"]
    axs[0].set_ylim(1e-6, 1e3)
    for ax, ttl, loc in zip(axs, titles, ["lower left", "upper right", "center right"]):
        ax.set_xlabel("Time [h]")
        ax.set_title(ttl, pad=16)
        ax.legend(loc=loc)
    ps.subtitle(axs[0], "485 m from Bennu, 30 t, 2 N: start 135 m off, drifting 7 cm/s")
    ps.subtitle(axs[1], "thrusters canted so the plume misses the surface")
    ps.subtitle(axs[2], "same field, same thrust limit")
    fig.tight_layout(w_pad=2.0)
    return fig


def _keyhole():
    from core.planner import Planner
    from core.scenario import KeyholeScenario

    sc = KeyholeScenario.load()
    eph = Ephemeris().with_bennu(sc.nominal)
    return sc, eph, Planner(eph, sc.nominal, sc)


def _prox_timeline(eph):
    from core.mission import LiveMission
    from core.planner import MissionParams
    from core.proximity.ops import ProximityOps

    sc, eph, pl = _keyhole()
    p = MissionParams(tractor_thrust_n=20.0)
    tp, _ = pl.plan_tractor(p, iso_to_t("2031-01-01"))
    t0 = tp.t_arrive
    m = LiveMission(eph, pl.nom, pl, p, t0, seed=4)  # proximity only
    m.tractor_plan = tp
    tr = m.tractor
    xb = m.bennu_helio(t0)[0]
    rho0 = -30e3 * xb[:3] / np.linalg.norm(xb[:3])
    tr.phase, tr.t_arrived, tr.arrival_offset = "proximity", t0, rho0
    tr.prox = ProximityOps(m, tr, t0, rho0, np.array([0.2, -0.1, 0.15]))
    while not m.on_station() and m.t < t0 + 10 * DAY:
        m.advance(m.t + DAY / 4, max_chunk=600.0)
    return m, tr.prox


def fig_prox_ops(eph):
    """The proximity timeline flown on the EKF estimate: altitude, navigation, glide slope, thrust."""
    from core.dynamics.asteroid import bennu_attitude
    from core.proximity.ops import GLIDESLOPE

    m, pr = _prox_timeline(eph)
    H = {k: np.asarray(v) for k, v in pr.hist.items()}
    t0 = H["t"][0]
    th = (H["t"] - t0) / 3600
    fig, axs = ps.figure(2, 2, figsize=(12.0, 7.2))
    ax = axs[0, 0]
    ax.plot(th, H["alt"], color=ps.SERIES[0])
    ax.set_yscale("log")
    ax.set_ylabel("Altitude above the surface [m]")
    ax.set_title("Approach, scan, survey, descent, capture, ascent, station")
    _stage_bands(ax, th, list(H["phase"]))
    ax.set_ylim(top=ax.get_ylim()[1] * 8)
    ax = axs[0, 1]
    ax.plot(th, np.maximum(H["nav_err"], 1e-3), color=ps.SERIES[0], lw=1.2, label="Estimate error")
    ax.plot(th, 3 * H["nav_sigma"], color=ps.SERIES[1], label="3σ bound")
    ax.set_yscale("log")
    ax.set_ylabel("Position [m]")
    ax.set_title("EKF on LIDAR range + camera bearing", pad=16)
    inside = np.mean(H["nav_err"] < 3 * H["nav_sigma"])
    ps.subtitle(ax, f"error inside 3σ {100 * inside:.1f} % of the time")
    ax.legend(loc="upper right")
    ax = axs[1, 0]
    t_desc = next(e.t for e in m.events if e.text.startswith("Descent"))
    t_cap = next(e.t for e in m.events if e.text.startswith("Capture: hovering"))
    hh, lat = [], []
    for t, rho in pr.trail:
        if t_desc <= t <= t_cap:
            rel = bennu_attitude(t).T @ rho - pr.site_body
            h = rel @ pr.site.up
            hh.append(h)
            lat.append(np.linalg.norm(rel - h * pr.site.up))
    hh, lat = np.array(hh), np.array(lat)
    hg = np.geomspace(max(hh.min(), 1.0), hh.max(), 50)
    ax.fill_between(hg, hg * np.tan(GLIDESLOPE), 1e4, color=ps.GRID, alpha=0.6, lw=0)
    ax.plot(hg, hg * np.tan(GLIDESLOPE), color=ps.INK2, lw=1.0, ls="--")
    ps.end_label(ax, hg[len(hg) // 2], hg[len(hg) // 2] * np.tan(GLIDESLOPE), "glide-slope cone edge (60°)",
                 dx=6, dy=-10)
    ax.plot(hh, np.maximum(lat, 1e-3), color=ps.SERIES[0], label="Spacecraft (descending, right to left)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e3)
    ax.set_xlabel("Height above the boulder site [m]")
    ax.set_ylabel("Distance from the site's vertical [m]")
    ax.set_title("Descent in Bennu's rotating frame", pad=16)
    ps.subtitle(ax, f"{hh[0]:.0f} m to {hh[-1]:.1f} m; shaded: outside the cone (MPC constraint)")
    ax.legend(loc="upper left")
    ax = axs[1, 1]
    ax.plot(th, H["thrust"], color=ps.SERIES[0], lw=1.2, label="Commanded")
    ax.plot(th, H["a_max"], color=ps.INK2, lw=1.0, ls="--", label="Available")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 50)
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust")
    ax.legend(loc="lower right")
    for a in (axs[0, 0], axs[0, 1], axs[1, 1]):
        a.set_xlabel("Hours since the hand-over (30 km)")
    fig.tight_layout(h_pad=2.6, w_pad=2.0)
    return fig


def fig_capture(eph):
    """MuJoCo boulder capture: computed-torque tracking, heights, bus thrust."""
    from core.proximity.capture import Capture

    cap = Capture(10_000.0, 20_000.0, thrust_max=20.0)
    res = cap.run()
    L = cap.log
    t = np.asarray(L.t)
    q, qd = np.asarray(L.q), np.asarray(L.q_des)
    fig, axs = ps.figure(1, 3, figsize=(13.0, 3.9))
    names = ["yaw", "shoulder", "elbow", "wrist"]
    for j, c in zip(range(4), ps.SERIES):
        axs[0].plot(t / 60, np.degrees(q[:, j]), color=c, label=names[j])
        axs[0].plot(t / 60, np.degrees(qd[:, j]), color=c, lw=1.0, ls=":")
    axs[0].set_ylabel("Joint angle [deg] (dotted: reference)")
    axs[0].set_title("Computed-torque arm control")
    axs[0].legend(loc="center left", ncol=2)
    bus, boulder, palm = np.asarray(L.bus), np.asarray(L.boulder), np.asarray(L.palm)
    axs[1].plot(t / 60, bus[:, 2], color=ps.SERIES[0], label="Bus center")
    axs[1].plot(t / 60, palm[:, 2], color=ps.SERIES[1], label="Palm")
    axs[1].plot(t / 60, boulder[:, 2], color=ps.SERIES[2], label=f"Boulder ({cap.m_boulder / 1e3:.0f} t) center")
    axs[1].set_ylabel("Height above the site [m]")
    axs[1].set_title("Grasp and lift-off", pad=16)
    ps.subtitle(axs[1], f"lifted {res['lifted_m']:.1f} m, grip slip {100 * res['grip_slip_m']:.0f} cm")
    axs[1].legend(loc="center left")
    F = np.linalg.norm(np.asarray(L.bus_force), axis=1)
    axs[2].plot(t / 60, F, color=ps.SERIES[0], label="Bus thrusters")
    axs[2].axhline(cap.thrust_max, color=ps.INK2, lw=1.0, ls="--")
    ps.end_label(axs[2], t[-1] / 60, cap.thrust_max, f"{cap.thrust_max:g} N available", dx=-90, dy=-22)
    axs[2].set_ylabel("Force [N]")
    axs[2].set_ylim(0, cap.thrust_max * 1.35)
    axs[1].set_ylim(top=axs[1].get_ylim()[1] + 2.0)
    axs[2].set_title("Holding the bus while the arm works")
    for ax in axs:
        ax.set_xlabel("Minutes")
        _stage_bands(ax, t / 60, L.stage)
    fig.tight_layout(w_pad=2.0)
    return fig


def fig_impact(eph):
    """DART check and the impact physics of Step 6 (momentum geometry, spin change, ejecta fates)."""
    from core.constants import DART
    from core.deflection import dart_period_change_minutes
    from core.impact import EjectaCloud, locate_impact, momentum_transfer, spin_change
    from core.proximity.ops import ProxEnvironment

    fig, axs = ps.figure(2, 2, figsize=(12.0, 8.0))
    # (a) DART: period change of Dimorphos' orbit from the along-track velocity change, dP/P = 3 dv_T / v
    ax = axs[0, 0]
    dvt = np.linspace(-4.0, -1.5, 50)
    ax.plot(dvt, [dart_period_change_minutes(x * 1e-3) for x in dvt], color=ps.SERIES[0],
            label="Model: ΔP/P = 3 Δv_T / v (v, P from lecture 1)")
    pred = dart_period_change_minutes()
    ax.axhspan(-34.0, -32.0, color=ps.GRID, alpha=0.7, lw=0)
    ax.axvspan(-2.80, -2.60, color=ps.GRID, alpha=0.7, lw=0)
    ax.plot([-2.70], [pred], "o", ms=8, color=ps.SERIES[1], mec=ps.SURFACE, mew=2,
            label=f"Δv_T = −2.70 mm/s (Cheng et al. 2023) gives {pred:.1f} min")
    ps.end_label(ax, -3.95, -33.0, "measured −33.0 ± 1.0 min", dx=0, dy=8)
    ax.set_xlabel("Along-track Δv of Dimorphos [mm/s]")
    ax.set_ylabel("Orbit period change [min]")
    ax.set_title("DART validation", pad=16)
    ps.subtitle(ax, f"β = {DART['beta']} from the same momentum model (Cheng et al. 2023 eq. 1)")
    ax.legend(loc="lower right")
    # (b) Bennu dv vs impact angle: ejecta along the normal vs straight back along U
    ax = axs[0, 1]
    m, Umag, beta = 40_000.0, 11e3, 3.6
    th = np.linspace(0, 75, 76)
    n = np.array([0, 0, 1.0])
    for model, c in (("head-on", ps.SERIES[0]), ("normal", ps.SERIES[1])):
        vals = []
        for a in np.radians(th):
            U = -Umag * np.array([np.sin(a), 0, np.cos(a)])
            E = -U / Umag if model == "head-on" else n
            vals.append(np.linalg.norm(momentum_transfer(m, U, beta, E)[0]) * 1e3)
        ax.plot(th, vals, color=c, label="ejecta straight back along the path (planner)" if model == "head-on"
                else "ejecta along the surface normal (live run)")
    ax.set_xlabel("Impact angle from the local vertical [deg]")
    ax.set_ylabel("Bennu Δv [mm/s]")
    ax.set_title("Momentum transfer depends on where it hits", pad=16)
    ps.subtitle(ax, "40 t at 11 km/s, β = 3.6: |Δv| = (mU/M) √(β² cos²θ + sin²θ) for normal ejecta")
    ax.legend(loc="lower left")
    # (c) spin-period change vs offset of a prograde equatorial hit
    ax = axs[1, 0]
    env = ProxEnvironment.shared()
    offs = np.linspace(5, 240, 40)
    for (mi, Ui, label), c in zip([(40_000.0, 11e3, "40 t at 11 km/s"), (DART["mass_impact"], DART["speed"],
                                   "DART-class: 579 kg at 6.1 km/s")], ps.SERIES):
        dps = []
        for b in offs:
            hit = locate_impact(env.ray, np.array([2000.0, b, 0.0]), np.array([-2000.0, b, 0.0]))
            U = np.array([-Ui, 0, 0])
            dv, _ = momentum_transfer(mi, U, 3.6, hit["normal"])
            dps.append(abs(spin_change(hit["point"], M_BENNU * dv)["period_change_s"]))
        ax.plot(offs, dps, color=c, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Offset of the impact line from Bennu's center [m] (equatorial)")
    ax.set_ylabel("|Change of Bennu's day| [s]")
    ax.set_title("Spin change (angular momentum r × MΔv)", pad=16)
    ps.subtitle(ax, "β = 3.6, ejecta along the normal; Bennu's day is 4.296 h")
    ax.legend(loc="lower right")
    # (d) ejecta fates by launch speed
    ax = axs[1, 1]
    hit = locate_impact(env.ray, np.array([2000.0, 40.0, 30.0]), np.array([-2000.0, 40.0, 30.0]))
    _, p_ej = momentum_transfer(40_000.0, np.array([-11e3, 0, 0]), 3.6, hit["normal"])
    cloud = EjectaCloud(0.0, hit["point"], hit["normal"], np.linalg.norm(p_ej), env.grav, env.ray, n=1500)
    cloud.thread.join()
    code = cloud.frames[-1][2]
    bins = np.geomspace(0.05, 50, 31)
    names = ["still in flight after 2 days", "landed back on Bennu", "escaped"]
    order = (1, 0, 2)
    ax.hist([cloud.v_launch[code == k] for k in order], bins=bins, stacked=True, label=[names[k] for k in order],
            color=[ps.SERIES[0], ps.SERIES[3], ps.SERIES[1]], edgecolor=ps.SURFACE, linewidth=0.8)
    ax.axvline(cloud.v_esc, color=ps.INK2, lw=1.0, ls="--")
    ps.end_label(ax, cloud.v_esc, ax.get_ylim()[1] * 0.9, f"escape speed {100 * cloud.v_esc:.0f} cm/s", dx=4)
    ax.set_xscale("log")
    ax.set_xlabel("Launch speed [m/s]")
    ax.set_ylabel("Tracers")
    ax.set_title("Ejecta in Bennu's polyhedron gravity", pad=16)
    ps.subtitle(ax, f"1,500 tracers, M(>v) ~ v^−1.23 (μ = 0.41); total ≈ {cloud.total_mass:.1e} kg (model)")
    ax.legend(loc="upper right")
    fig.tight_layout(h_pad=2.6, w_pad=2.0)
    return fig


M_BENNU = BENNU["mass"]


def fig_tow(eph):
    """Step 7 in the keyhole mission: tow drift vs Clohessy-Wiltshire, steered tow, trim to the DART aim point in
    the 2135 B-plane, and the linear prediction vs the full nonlinear propagation."""
    from core.constants import G, YEAR
    from core.tow import Deflection, TrimGuidance, cw_along_track, verify_nonlinear

    sc, eph, pl = _keyhole()
    nom = sc.nominal
    fig, axs = ps.figure(2, 2, figsize=(12.0, 8.2))
    # (a) along-track drift from a constant along-track pull: STM vs CW
    ax = axs[0, 0]
    t0 = iso_to_t("2030-01-01")
    a_T = G * 30_000 / 485.0**2
    d = Deflection(nom, t0)
    vhat = lambda t: nom.state(t)[3:] / np.linalg.norm(nom.state(t)[3:])
    d.advance(t0 + 10 * YEAR, lambda t: a_T * vhat(t))
    ts = t0 + np.linspace(0.01, 10, 400) * YEAR
    y = np.array([d.state_at(t)[:3] @ vhat(t) for t in ts]) / 1e3
    n = 2 * np.pi / (436.65 * DAY)
    ax.plot((ts - t0) / YEAR, -y, color=ps.SERIES[0], label="Bennu's orbit (STM, e = 0.20)")
    ax.plot((ts - t0) / YEAR, -cw_along_track(a_T, n, ts - t0) / 1e3, color=ps.SERIES[1], ls="--",
            label="Clohessy–Wiltshire (circular orbit)")
    ax.set_xlabel("Years of towing")
    ax.set_ylabel("Along-track drift behind the nominal [km]")
    ax.set_title("Tow drift: y = (4a/n²)(1 − cos nt) − (3/2) a t²", pad=16)
    ps.subtitle(ax, f"constant along-track pull a = {a_T:.1e} m/s² (30 t at 485 m)")
    ax.legend(loc="upper left")
    # (b) steered vs fixed tow: shift of the 2135 flyby vs tow length
    ax = axs[0, 1]
    yrs = np.linspace(0.25, 10, 40)
    steered, fixed = [], []
    tt10 = np.linspace(t0, t0 + 10 * YEAR, 600)
    w = np.linalg.svd(np.trapezoid(pl.B(tt10), tt10, axis=0))[0][:, 0]
    for Y in yrs:
        tt = np.linspace(t0, t0 + Y * YEAR, 300)
        B = pl.B(tt)
        Bi = np.trapezoid(B, tt, axis=0)
        steered.append(a_T * np.trapezoid(np.linalg.norm(np.einsum("nij,i->nj", B, w), axis=1), tt) / 1e3)
        fixed.append(a_T * np.linalg.norm(Bi.T @ w) / 1e3)
    ax.plot(yrs, steered, color=ps.SERIES[0], label="steered: e(t) = B(t)^T w / |B(t)^T w| (trim law)")
    ax.plot(yrs, fixed, color=ps.SERIES[1], label="best fixed inertial direction")
    ax.set_yscale("log")
    ax.set_xlabel("Tow duration from 2030 [years]")
    ax.set_ylabel("Shift of the 2135 flyby [km]")
    ax.set_title("Steering the tow with Bennu's orbit", pad=16)
    ps.subtitle(ax, "before 2060 the flyby there amplifies every push ~4,000×")
    ax.legend(loc="lower right")
    # (c) B-plane trim: an impact that fell short, trimmed to the aim point
    ax = axs[1, 0]
    t_imp = iso_to_t("2030-06-22")
    dv = np.array([1.5e-6, -2.5e-6, 1e-6])
    d = Deflection(nom, t_imp - DAY)
    d.impulse(t_imp, dv)
    b_ki = pl.bplane.project((d.Phi(pl.t_ca, t_imp)[:, 3:] @ dv)[:3])
    trim = TrimGuidance(pl, "aim", aim_shift=1.5 * b_ki)
    t, t_end = t_imp + 5 * DAY, t_imp + 10 * YEAR
    path = []
    while t < t_end:
        on, e = trim.update(d, t, a_T, t_end)
        path.append(trim.b_now(d, t))
        t1 = min(trim.next, t_end)
        d.advance(t1, (lambda tau, e=e: a_T * e) if on else None)
        t = t1
    path = (np.array(path) + pl.b0) / 1e6
    k0 = pl.b0 / 1e6
    ax.plot(path[:, 0], path[:, 1], color=ps.SERIES[0], label="flyby point, weekly (with trim)")
    ax.plot([k0[0]], [k0[1]], "*", color=ps.CRITICAL, ms=12, label="keyhole flyby (no deflection)")
    ax.plot([k0[0] + b_ki[0] / 1e6], [k0[1] + b_ki[1] / 1e6], "o", color=ps.SERIES[1], ms=8, label="impact only")
    ax.plot([k0[0] + trim.b_aim[0] / 1e6], [k0[1] + trim.b_aim[1] / 1e6], "+", color=ps.INK, ms=14, mew=2,
            label="aim point")
    err = np.linalg.norm(trim.b_now(d, t) - trim.b_aim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("ξ [1000 km]")
    ax.set_ylabel("ζ [1000 km]")
    ax.set_title("Trim guidance in the 2135 B-plane", pad=16)
    ps.subtitle(ax, f"impact delivers 2/3 of the aim; the 30 t tractor adds the rest: final error {err / 1e3:.0f} km")
    ax.legend(loc="lower left", fontsize=8)
    # (d) linear (STM) vs full nonlinear propagation at the 2135 flyby, vs the size of the shift
    ax = axs[1, 1]
    sizes, rels = [], []
    for scale in (0.003, 0.01, 0.03, 0.1, 0.3):
        dd = Deflection(nom, t_imp - DAY)
        dd.impulse(t_imp, dv * scale * 100)
        r = verify_nonlinear(eph, nom, dd, pl, to_end=False)
        sizes.append(np.linalg.norm(r["b_linear"] - pl.b0) / 1e3)
        rels.append(100 * r["relative"])
    ax.loglog(sizes, rels, "o-", color=ps.SERIES[0], label="|nonlinear − linear| / shift")
    ax.loglog(sizes, rels[-2] * np.array(sizes) / sizes[-2], ls="--", color=ps.SERIES[1],
              label="proportional to the shift (second order)")
    from core.planner import MissionParams

    ax.axvline(pl.push_direction(MissionParams())[1] / 1e3, color=ps.AXIS, lw=0.8)
    ax.set_xlabel("Shift of the 2135 flyby (linear prediction) [km]")
    ax.set_ylabel("Linear-prediction error [%]")
    ax.set_title("Linear prediction vs full nonlinear check", pad=16)
    ps.subtitle(ax, "impact 2030-06-22; vertical line: the shift the goal needs; small shifts: the ~150 km noise floor")
    ax.legend(loc="upper left")
    fig.tight_layout(h_pad=2.6, w_pad=2.0)
    return fig


def fig_keyhole(eph):
    """The scenario: the line-of-variations scan for resonant returns, and the Earth distance to the impact."""
    from core.constants import LUNAR_DISTANCE

    sc, eph, pl = _keyhole()
    scan = json.loads((DATA_DIR / "scenario_scan.json").read_text())
    if "c_km" in scan[0]:
        scan = [{"c": r["c_km"] * 1e3, "approaches": r["approaches"]} for r in scan]
    fig, axs = ps.figure(1, 2, figsize=(14.0, 4.8))
    ax = axs[0]
    yr = lambda t: 2000.0 + (np.asarray(t) / DAY + 0.5) / 365.25
    c_imp = sc.c_lov
    xs, ys, cs = [], [], []
    for r in scan:
        for a in r["approaches"]:
            if a[0] > sc.t_ca + 300 * DAY and a[1] < 0.05 * AU:
                xs.append(r["c"] / 1e6), ys.append(a[1] / AU), cs.append(yr(a[0]))
    scat = ax.scatter(xs, ys, c=cs, s=10, cmap="viridis")
    ax.set_yscale("log")
    ax.axhline(RADIUS["earth"] / AU, color=ps.CRITICAL, lw=0.8, ls="--")
    ax.axvline(c_imp / 1e6, color=ps.INK2, lw=0.8)
    ps.end_label(ax, c_imp / 1e6, 0.03, f" keyhole: {int(yr(sc.t_impact))} impact", dx=4)
    cb = fig.colorbar(scat, ax=ax)
    cb.set_label("year of the later approach")
    ax.set_xlabel("Offset of the 2135 flyby along the line of variations [1000 km]")
    ax.set_ylabel("Later Earth approach [au]")
    ax.set_title("Resonant returns after the 2135 flyby", pad=16)
    ps.subtitle(ax, f"scan of {len(scan)} full propagations; keyhole {sc.keyhole_width / 1e3:.1f} km wide; "
                    f"2026 change {sc.meta['dv0_ms']:.1e} m/s")
    ax = axs[1]
    from core.dynamics.nominal import NominalBennu

    real = NominalBennu(eph)
    t = np.linspace(sc.nominal.t[0], sc.t_impact - 60, 40_000)
    d_vi = np.linalg.norm(sc.nominal.state(t)[:, :3] - eph.pos("earth", t), axis=1) / AU
    tr = t[t < real.span[1]]
    d_re = np.linalg.norm(real.state(tr)[:, :3] - eph.pos("earth", tr), axis=1) / AU
    ax.semilogy(yr(tr), d_re, color=ps.SERIES[1], lw=0.8, label="Bennu's nominal orbit (our propagation)")
    ax.semilogy(yr(t), d_vi, color=ps.SERIES[0], lw=1.0, label="scenario: through the keyhole")
    ax.axhline(LUNAR_DISTANCE / AU, color=ps.INK2, lw=0.8)
    ps.end_label(ax, 2028, LUNAR_DISTANCE / AU, "Moon's distance (goal)", dy=4)
    ax.axhline(RADIUS["earth"] / AU, color=ps.CRITICAL, lw=0.8, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Earth–Bennu distance [au]")
    ax.set_title(f"Keyhole flyby in {int(yr(sc.t_ca))}, Earth impact on 2182-09-24", pad=16)
    ps.subtitle(ax, f"2135 flyby {sc.q_ca / 1e3:,.0f} km ({sc.q_ca / LUNAR_DISTANCE:.2f} LD); the two orbits differ "
                    "by km before 2060")
    ax.legend(loc="lower left")
    fig.tight_layout(w_pad=2.0)
    return fig


def fig_opportunities(eph):
    """Recommended DART mass at every close pass (head-on, planned beta), and the speed-mass trade."""
    from core.planner import MissionParams

    sc, eph, pl = _keyhole()
    opps = [o for o in pl.opportunities(MissionParams()) if o.feasible]
    yr = lambda t: 2000.0 + (np.asarray(t) / DAY + 0.5) / 365.25
    fig, axs = ps.figure(1, 2, figsize=(14.0, 4.8))
    ax = axs[0]
    ax.semilogy([yr(o.t_arrive) for o in opps], [o.mass_nominal for o in opps], "o-", color=ps.SERIES[0],
                label="planned β = 3.6")
    ax.semilogy([yr(o.t_arrive) for o in opps], [o.mass_p90 for o in opps], "s--", color=ps.SERIES[1], ms=4,
                label="90 % confidence over β")
    ax.axvline(yr(iso_to_t("2060-09-23")), color=ps.AXIS, lw=0.8)
    ps.end_label(ax, yr(iso_to_t("2060-09-23")), min(o.mass_nominal for o in opps), " 2060 flyby", dx=4)
    ax.axhline(579.4, color=ps.INK2, lw=0.8, ls=":")
    ps.end_label(ax, yr(opps[-1].t_arrive), 579.4, "DART (579 kg)", dx=-90, dy=-12)
    ax.set_xlabel("Impact date")
    ax.set_ylabel("Impactor mass at impact [kg]")
    ax.set_title("DART opportunities at Bennu's close passes", pad=16)
    ps.subtitle(ax, f"linear estimate for the {pl.target_shift(MissionParams()) / 1e3:,.0f} km target (1.25 LD), "
                    "head-on, departure ≤ 12.5 km/s")
    ax.legend(loc="upper left")
    ax = axs[1]
    for k, o in enumerate(opps[:6]):
        if not o.trade:
            continue
        v, spd, m = zip(*o.trade)
        ax.semilogy(v, m, "o-", color=ps.SERIES[k % len(ps.SERIES)], label=f"impact {yr(o.t_arrive):.1f}")
    ax.set_xlabel("Orbit-raising cut-off speed (departure perigee speed) [km/s]")
    ax.set_ylabel("Recommended mass [kg]")
    ax.set_title("Faster departure: faster arcs, lighter impactor", pad=16)
    ps.subtitle(ax, "more departure energy opens arcs with higher impact speed")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    return fig


def fig_orbit_raising(eph):
    """The recommended DART's orbit raising: ellipses in the orbit plane, perigee speed and fuel per burn."""
    from core.dart import orbit_points
    from core.planner import MissionParams

    sc, eph, pl = _keyhole()
    ip = pl.plan(MissionParams()).impactor
    rp = ip.raising
    fig, axs = ps.figure(1, 3, figsize=(15.0, 4.8))
    ax = axs[0]
    for k, b in enumerate(rp.burns):
        pts = orbit_points(rp.r_park, b.v_perigee, rp.P_hat, rp.Q_hat, n=400, r_max=4e8)
        x, y = pts @ rp.P_hat / 1e6, pts @ rp.Q_hat / 1e6
        ax.plot(x, y, color=ps.SERIES[1] if b.kind == "inject" else ps.SERIES[0], lw=1.0 if b.kind == "raise" else 1.6,
                alpha=0.4 + 0.6 * (k + 1) / len(rp.burns))
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(RADIUS["earth"] / 1e6 * np.cos(th), RADIUS["earth"] / 1e6 * np.sin(th), color=ps.SERIES[0], alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("along the perigee direction [1000 km]")
    ax.set_ylabel("in the orbit plane [1000 km]")
    ax.set_title("Orbit raising, then injection toward Bennu", pad=16)
    ps.subtitle(ax, f"{len(rp.burns) - 1} burns at perigee, apogee up to 300,000 km; injection to v∞ "
                    f"{rp.v_inf / 1e3:.2f} km/s")
    ax = axs[1]
    t_days = (np.array([b.t for b in rp.burns]) - rp.t_start) / DAY
    ax.step(t_days, [b.v_perigee / 1e3 for b in rp.burns], where="post", color=ps.SERIES[0])
    ax.plot(t_days, [b.v_perigee / 1e3 for b in rp.burns], "o", color=ps.SERIES[0], ms=4)
    ax.set_xlabel("Days since the first burn")
    ax.set_ylabel("Perigee speed after the burn [km/s]")
    ax.set_title("Gaining speed before leaving Earth", pad=16)
    ps.subtitle(ax, f"cut-off {MissionParams().dart_max_speed_kms:.1f} km/s; this arc departs at {rp.v_dep / 1e3:.2f} km/s")
    ax = axs[2]
    ax.plot(t_days, np.cumsum([b.propellant_kg for b in rp.burns]), "o-", color=ps.SERIES[1])
    ax.set_xlabel("Days since the first burn")
    ax.set_ylabel("Fuel spent so far [kg]")
    ax.set_title("Fuel", pad=16)
    ps.subtitle(ax, f"{ip.mass:,.0f} kg impactor (linear sizing), Isp 320 s: {rp.propellant_kg:,.0f} kg for "
                    f"Δv {rp.dv_total / 1e3:.2f} km/s")
    fig.tight_layout(w_pad=2.0)
    return fig


def fig_sun_barycenter(eph):
    """The Sun around the Solar System barycenter: path, distance, our N-body vs JPL, planet contributions."""
    from core.dynamics import solar_system as SS

    d = SS.plot_data(eph, n=4000)
    fig, axs = ps.figure(1, 3, figsize=(14.0, 4.6))
    ax = axs[0]
    k = d["years"] < 2060
    ax.plot(d["xy_jpl"][k, 0], d["xy_jpl"][k, 1], color=ps.SERIES[0], lw=1.2, label="Sun's center, 2026–2060 (JPL)")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=ps.SERIES[3], lw=1.0, label="the Sun's disk (centered on the barycenter)")
    ax.plot([0], [0], "+", color=ps.INK, ms=12, mew=2, label="barycenter")
    ax.set_aspect("equal")
    ax.set_xlabel("ecliptic x [solar radii]")
    ax.set_ylabel("ecliptic y [solar radii]")
    ax.set_title("The Sun circles the barycenter", pad=16)
    ps.subtitle(ax, "pulled mostly by Jupiter and Saturn")
    ax.legend(loc="lower left", fontsize=8)
    ax = axs[1]
    ax.plot(d["years"], d["dist_jpl"], color=ps.SERIES[0], label="all planets (JPL)")
    ax.plot(d["years"], d["dist_js"], color=ps.SERIES[2], lw=1.2, label="Jupiter + Saturn")
    ax.plot(d["years"], d["dist_j"], color=ps.SERIES[1], lw=1.2, label="Jupiter only")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sun's center from the barycenter [solar radii]")
    ax.set_title("Distance from the barycenter", pad=16)
    ps.subtitle(ax, f"our N-body (Sun + 9 planets) matches JPL to {d['nbody_max_err_km']:.0f} km over 114 years")
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[2]
    ts = np.linspace(iso_to_t("2026-01-02"), iso_to_t("2139-12-30"), 2000)
    _, c = SS.sun_from_planets(eph, ts)
    names = sorted(c, key=lambda b: -np.linalg.norm(c[b], axis=1).max())
    vals = [np.linalg.norm(c[b], axis=1).max() / 1e3 for b in names]
    ax.barh(range(len(names)), vals, color=ps.SERIES[0])
    ax.set_yticks(range(len(names)), [SS.PLANET_LABELS[b] for b in names])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Largest pull on the Sun's position [km]  (m_i / m_sun) |r_i|")
    ax.set_title("Each planet's share", pad=16)
    ps.subtitle(ax, "Pluto (dwarf planet) included as the 9th body")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:,.0f}", va="center", fontsize=8, color=ps.INK2)
    fig.tight_layout(w_pad=2.0)
    return fig

FIGURES = {"integrators": "fig_integrators", "bennu_jpl": "fig_bennu_jpl", "stm": "fig_stm", "checks": "fig_checks",
           "gravity": "fig_gravity", "prox_hover": "fig_prox_hover", "prox_ops": "fig_prox_ops",
           "capture": "fig_capture", "impact": "fig_impact", "tow": "fig_tow", "sun_barycenter": "fig_sun_barycenter",
           "keyhole": "fig_keyhole", "opportunities": "fig_opportunities", "orbit_raising": "fig_orbit_raising"}


def _render(name):
    """One figure, in its own process (each worker loads its own ephemerides)."""
    import matplotlib

    matplotlib.use("Agg")
    if name == "gravity":
        path, stats = fig_gravity()
        return f"{path}  {json.dumps({k: round(v, 1) for k, v in stats.items()})} [µm/s²]"
    fig = globals()[FIGURES[name]](Ephemeris())
    path = OUT / f"{name}.png"
    fig.savefig(path)
    return str(path)


def main():
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from core.parallel import N_THREADS

    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(FIGURES), default=list(FIGURES))
    p.add_argument("--jobs", type=int, default=N_THREADS, help="figures rendered in parallel processes")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = max(1, min(args.jobs, len(args.only)))
    if jobs == 1:
        for name in args.only:
            print(_render(name))
    else:  # one process per figure; compute threads shared out between them
        os.environ["BENNU_THREADS"] = str(max(1, N_THREADS // jobs))
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as ex:
            for line in ex.map(_render, args.only):
                print(line)
    print(f"{len(args.only)} figures in {time.time() - t0:.0f} s ({jobs} processes)")


if __name__ == "__main__":
    main()
