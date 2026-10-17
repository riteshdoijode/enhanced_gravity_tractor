"""Build the keyhole scenario: data/scenario_keyhole.json + data/bennu_vi.npz (see core/scenario.py).

    uv run python -m scripts.make_scenario            # ~2 min with a cached scan, ~12 min with --rescan

1. The real nominal's 2135 flyby, its B-plane (modified target plane) and the line of variations (LOV):
   the direction in which small early changes of Bennu's orbit move the flyby point (the sensitivity is
   rank one), and the 2026 state change v1 per metre along it.
2. Scan the LOV (+-300,000 km in 2,000 km steps, in parallel processes): for every post-flyby Earth
   approach inside 0.2 au, its miss vector (xi', zeta'). A zero crossing of the timing coordinate zeta' is
   a resonant return; if xi' at the crossing is inside the Earth, it is an impact keyhole.
3. Pick the impact keyhole whose 2135 flyby lies inside the Moon's distance and nearest the nominal
   (the 2182 return - JPL's most likely Bennu impact date - is taken if it is an impact in this model).
4. Refine it with the secant method on the asymptotic B-plane of the return (geocentric hyperbola from
   0.01 au) to zeta' = 0, and estimate the keyhole width (|B'| below Earth's capture radius, zeta' linear in c).
5. Build the trajectory file (linear before t_pre, nonlinear with the STM after it, stopped at Earth's
   surface), its close approaches and the impact site, and write the JSON.
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from core.constants import AU, DATA_DIR, DAY, LUNAR_DISTANCE, RADIUS, YEAR

SCAN_FILE = DATA_DIR / "scenario_scan.json"
_G = {}


def _init():
    from core.dynamics.asteroid import BennuDynamics
    from core.ephemeris import Ephemeris

    _G["eph"] = Ephemeris()
    _G["dyn"] = BennuDynamics(_G["eph"])


def _scan_one(args):
    """All Earth approaches inside 0.2 au after t_pre, with their closest-approach miss vectors."""
    x_pre, t_pre, t_end = args
    eph, dyn = _G["eph"], _G["dyn"]
    ts = np.arange(t_pre + DAY, t_end, 0.25 * DAY)
    xs, _ = dyn.propagate(t_pre, x_pre, ts, rtol=1e-12, atol=1e-6)
    d = np.linalg.norm(xs[:, :3] - eph.pos("earth", ts), axis=1)
    i = np.where((d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < 0.2 * AU))[0] + 1
    out = []
    for k in i:
        r = xs[k, :3] - eph.pos("earth", ts[k])
        v = xs[k, 3:] - eph.vel("earth", ts[k])
        s = v / np.linalg.norm(v)
        ve = eph.vel("earth", ts[k]) - eph.vel("sun", ts[k])
        z = -(ve - (ve @ s) * s)
        z /= np.linalg.norm(z)
        xi = np.cross(s, z)
        rp = r - (r @ s) * s
        out.append((float(ts[k]), float(d[k]), float(rp @ xi), float(rp @ z)))
    return out


def _return_b(args):
    """Asymptotic B-plane vector of the return near t_ret (2-D, m), or None if Bennu stays beyond 0.01 au."""
    from scipy.integrate import solve_ivp

    from core.scenario import b_from_hyperbola

    x_pre, t_pre, t_ret = args
    eph, dyn = _G["eph"], _G["dyn"]
    t_a = t_ret - 60 * DAY
    x_a = dyn.propagate(t_pre, x_pre, [t_a], rtol=1e-12, atol=1e-6)[0][-1]

    def near(t, x):
        return np.linalg.norm(x[:3] - eph.pos("earth", t)) - 0.01 * AU

    near.terminal, near.direction = True, -1
    sol = solve_ivp(dyn.rhs, (t_a, t_ret + 60 * DAY), x_a, method="DOP853", rtol=1e-12, atol=1e-6, events=near)
    if not len(sol.t_events[0]):
        return None
    t, x = sol.t_events[0][0], sol.y_events[0][0]
    b, B, vinf = b_from_hyperbola(x[:3] - eph.pos("earth", t), x[3:] - eph.vel("earth", t),
                                  eph.vel("earth", t) - eph.vel("sun", t))
    return b, B, vinf


def _refine(args):
    """Secant on zeta' (asymptotic B-plane of the return) along the LOV around one candidate crossing."""
    from core.scenario import earth_capture_b

    x_nom_pre, dxpre, t_pre, cand = args
    t_ret = cand["t_ret"]
    f = lambda cc: _return_b((x_nom_pre + cc * dxpre, t_pre, t_ret))
    c0, c1 = cand["c"] - 0.002 * abs(cand["dc"]), cand["c"] + 0.002 * abs(cand["dc"])
    r0, r1 = f(c0), f(c1)
    if r0 is None or r1 is None:
        return None
    for _ in range(12):
        z0, z1 = r0[0][1], r1[0][1]
        if z1 == z0:
            break
        c2 = c1 - z1 * (c1 - c0) / (z1 - z0)
        c0, r0, c1 = c1, r1, c2
        r1 = f(c1)
        if r1 is None:
            return None
        if abs(r1[0][1]) < 50e3 and abs(c1 - c0) < 0.5:
            break
    bE = earth_capture_b(r1[2])
    slope = (r1[0][1] - r0[0][1]) / (c1 - c0)
    return {**cand, "c": float(c1), "B": [float(r1[0][0]), float(r1[0][1])], "B_norm": float(r1[1]),
            "v_inf": float(r1[2]), "b_capture": float(bE), "impact": bool(r1[1] < bE),
            "width": float(2.0 * np.sqrt(max(bE**2 - r1[0][0] ** 2, 0.0)) / abs(slope))}


def year(t):
    return 2000.0 + (t / DAY + 0.5) / 365.25


def crossings(scan):
    """Zero crossings of zeta' per return (approaches grouped by quarter of a year) with xi' interpolated."""
    from collections import defaultdict

    br = defaultdict(dict)
    for i, r in enumerate(scan):
        for a in r["approaches"]:
            br[round(year(a[0]) * 4)][i] = a
    out = []
    for m in br.values():
        idx = sorted(m)
        for i0, i1 in zip(idx[:-1], idx[1:]):
            if i1 != i0 + 1:
                continue
            a0, a1 = m[i0], m[i1]
            if np.sign(a0[3]) != np.sign(a1[3]):
                f = a0[3] / (a0[3] - a1[3])
                out.append({"c": scan[i0]["c"] + f * (scan[i1]["c"] - scan[i0]["c"]), "t_ret": a0[0] + f * (a1[0] - a0[0]),
                            "xi": a0[2] + f * (a1[2] - a0[2]), "dc": scan[i1]["c"] - scan[i0]["c"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true", help="redo the line-of-variations scan")
    ap.add_argument("--prefer-year", type=int, default=2182)
    args = ap.parse_args()
    from core.dynamics.nominal import NominalBennu
    from core.ephemeris import Ephemeris
    from core.scenario import (KEYHOLE_DATE, PRE_DAYS, SCENARIO_FILE, VI_FILE, build_vi_file, earth_capture_b,
                               find_close_approaches, impact_site, lov_vector, mtp)
    from core.timeutil import t_to_utc_str

    wall = time.time()
    eph = Ephemeris()
    nom = NominalBennu(eph)
    t_ca, r_rel, v_rel = nom.close_approach(eph, KEYHOLE_DATE)
    bp, b_nom = mtp(eph, t_ca, nom.state(t_ca))
    lov, v1 = lov_vector(nom, t_ca, bp.basis)
    t_pre = t_ca - PRE_DAYS * DAY
    dxpre = nom.Phi0(t_pre) @ v1
    x_nom_pre = nom.state(t_pre)
    t_end = eph.span("earth")[1] - 20 * DAY
    print(f"Nominal 2135 flyby {t_to_utc_str(t_ca)}: {np.linalg.norm(b_nom) / 1e3:,.0f} km "
          f"({np.linalg.norm(b_nom) / LUNAR_DISTANCE:.2f} LD), LOV direction {lov.round(4)}, "
          f"2026 velocity change {np.linalg.norm(v1[3:]) * 1e6:.2e} m/s per km")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(6, mp_context=ctx, initializer=_init) as ex:
        if args.rescan or not SCAN_FILE.exists():
            cs = np.arange(-300e3, 300e3 + 1, 2e3) * 1e3
            res = list(ex.map(_scan_one, [(x_nom_pre + c * dxpre, t_pre, t_end) for c in cs]))
            SCAN_FILE.write_text(json.dumps([{"c": float(c), "approaches": a} for c, a in zip(cs, res)]))
            print(f"  scan: {len(cs)} propagations in {time.time() - wall:.0f} s")
        scan = json.loads(SCAN_FILE.read_text())
        if "c_km" in scan[0]:  # scan written by the exploratory script (km)
            scan = [{"c": r["c_km"] * 1e3, "approaches": r["approaches"]} for r in scan]
        cands = []
        for x in crossings(scan):
            q35 = np.linalg.norm(b_nom + x["c"] * lov)
            if abs(x["xi"]) < 4 * RADIUS["earth"] and q35 < LUNAR_DISTANCE and x["t_ret"] > t_ca + YEAR:
                cands.append({**x, "q35": float(q35)})
        print(f"  {len(cands)} resonant returns pass within 4 Earth radii (closest-approach estimate); refining…")
        refined = [r for r in ex.map(_refine, [(x_nom_pre, dxpre, t_pre, x) for x in cands]) if r is not None]
    for x in sorted(refined, key=lambda x: abs(x["c"])):
        print(f"  return {t_to_utc_str(x['t_ret'])[:10]}: c = {x['c'] / 1e3:,.3f} km, |B'| {x['B_norm'] / 1e3:,.0f} km vs "
              f"capture radius {x['b_capture'] / 1e3:,.0f} km -> {'IMPACT' if x['impact'] else 'miss'}, "
              f"2135 flyby {x['q35'] / LUNAR_DISTANCE:.2f} LD")
    hits = [x for x in refined if x["impact"]]
    if not hits:
        raise SystemExit("no impact keyhole inside the Moon's distance on the scanned line of variations")
    pref = [x for x in hits if int(year(x["t_ret"])) == args.prefer_year]
    pick = min(pref or hits, key=lambda x: abs(x["c"]))
    if not pref:
        print(f"  (no {args.prefer_year} impact keyhole in this force model; using the nearest one)")
    c_star, width = pick["c"], pick["width"]
    cands = refined
    print(f"  keyhole at c = {c_star / 1e3:,.3f} km, width {width / 1e3:.2f} km along the LOV")
    # 5. trajectory file
    t_imp, _ = build_vi_file(eph, nom, c_star, v1, t_pre)
    if not np.isfinite(t_imp):
        raise SystemExit("the scenario trajectory built with the STM does not hit Earth; widen the search")
    from core.dynamics.nominal import NominalBennu as NB

    vi = NB(path=VI_FILE)
    t_ca_vi, r_vi, v_vi = vi.close_approach(eph, KEYHOLE_DATE)
    bp_vi, b0 = mtp(eph, t_ca_vi, vi.state(t_ca_vi))
    lov_vi, _ = lov_vector(vi, t_ca_vi, bp_vi.basis)
    apps = find_close_approaches(vi.state, eph, vi.t[0], t_imp - 2 * DAY, max_dist=0.5 * AU, step=0.25 * DAY)
    lat, lon, v_imp = impact_site(eph, vi, t_imp - 60.0)
    out = {"t0": float(vi.t[0]), "t_pre": float(t_pre), "t_ca": float(t_ca_vi), "b0": b0.tolist(),
           "basis": bp_vi.basis.tolist(), "v_rel": v_vi.tolist(), "lov": lov_vi.tolist(), "c_lov": float(c_star),
           "t_impact": float(t_imp), "impact_latlon": [lat, lon], "impact_v": v_imp, "keyhole_width": float(width),
           "approaches": apps,
           "meta": {"nominal_q_2135": float(np.linalg.norm(b_nom)), "return_year": int(year(t_imp)),
                    "candidates": [{k: v for k, v in x.items()} for x in cands],
                    "dv0_ms": float(np.linalg.norm(c_star * v1[3:])), "dr0_m": float(np.linalg.norm(c_star * v1[:3])),
                    "built": time.strftime("%Y-%m-%d %H:%M")}}
    SCENARIO_FILE.write_text(json.dumps(out, indent=1))
    print(f"Scenario: 2135 flyby {t_to_utc_str(t_ca_vi)} at {np.linalg.norm(b0) / 1e3:,.0f} km "
          f"({np.linalg.norm(b0) / LUNAR_DISTANCE:.3f} LD); Earth impact {t_to_utc_str(t_imp)} UTC at "
          f"{lat:+.1f}° {lon:+.1f}°, {v_imp / 1e3:.1f} km/s; keyhole {width / 1e3:.2f} km wide; "
          f"2026 velocity change {out['meta']['dv0_ms']:.2e} m/s. ({time.time() - wall:.0f} s)")


if __name__ == "__main__":
    main()
