"""Full-model sizing of a DART: the recommended mass corrected for the nonlinearity of large deflections.

The planner sizes the impactor with the linear (STM) map of its push to the 2135 flyby. For the shifts this
mission needs (hundreds of thousands of km, amplified ~4,000x by the 2060 flyby), the full force model gives a
noticeably different result: the second-order part grows with the square of the shift (5 % at 90,000 km,
tens of percent at 300,000 km). Here the mass is found so that the *nonlinear* shift along the push direction
equals the DART target: each evaluation propagates Bennu with and without the pushes (earlier DARTs included)
to 60 days before the flyby and flies through it (core/tow.py; the run without pushes is computed once).

Without a pool: the secant method on log(mass), 2-4 evaluations one after another (~1.5 min).
With a core/fullmodel.py pool of n workers: n - 1 candidate masses spanning the likely answer go at once
(with the run without pushes on the last worker), the crossing of the target is interpolated, and a second
round checks masses within +-1.5 % of it, again side by side: two evaluation times (~45 s) whatever n is,
one when an earlier refinement for the same opportunity and target narrows the first round.

`refine_job` runs in a separate process (the UI submits it in the background); it builds its own planner and
its pool of full-model workers (budget()["refine_procs"]) once.
"""

from dataclasses import replace

import numpy as np

from core.constants import DAY

_STATE = {}


def refine_dart_mass(pl, p, opportunity=-1, b_now=None, t_after=None, prior_impulses=(), number=1, tol=0.03,
                     max_evals=5, pool=None, guess=None):
    """Mass for which the full-model shift of the flyby along the push direction equals the DART target.
    `pool`: core/fullmodel.py workers (candidate masses in parallel); `guess`: expected ratio to the linear mass.
    Returns {"mass", "linear_mass", "fraction", "evals": [(mass, fraction)], "opportunity", "t_arrive"} or None."""
    from core.tow import Deflection, verify_nonlinear_many

    q = replace(p, use_recommended_mass=True, refined_mass_kg=0.0, dart_sizing="nominal")
    opps = pl.opportunities(q, b_now=b_now, t_after=t_after)
    opp = next((o for o in opps if o.index == opportunity), None) if opportunity >= 0 else pl.recommended(opps)
    if opp is None or not opp.feasible:
        return None
    ip, _ = pl.plan_dart(q, opp, b_now=b_now, number=number)
    if not ip.feasible:
        return None
    d, _ = pl.push_direction(q)
    target = pl.target_shift(q)
    dv_unit = ip.dv_bennu / ip.mass
    prior = [(float(t), np.asarray(dv, float)) for t, dv in prior_impulses]
    t0 = min([t for t, _ in prior] + [ip.t_arrive]) - DAY
    cache = {}

    def deflection(m):
        defl = Deflection(pl.nom, t0)
        for t, dv in prior:
            defl.impulse(t, dv)
        defl.impulse(ip.t_arrive, dv_unit * m)
        return defl

    def fractions(masses):
        rs = verify_nonlinear_many(pl.eph, pl.nom, [deflection(m) for m in masses], pl, to_end=False, cache=cache,
                                   pool=pool)
        return [float((r["b_nonlinear"] - pl.b0) @ d) / target for r in rs]

    m_lin = ip.mass
    if pool is not None and pool.workers >= 3:
        evals = _parallel_search(fractions, m_lin, pool.workers, bool(prior), tol, guess)
    else:
        evals = _secant_search(lambda m: fractions([m])[0], m_lin, bool(prior), tol, max_evals)
    best = min(evals, key=lambda e: abs(e[1] - 1.0))
    return {"mass": best[0], "linear_mass": m_lin, "fraction": best[1], "evals": evals,
            "opportunity": opp.index, "t_arrive": ip.t_arrive}


def _secant_search(fraction, m_lin, has_prior, tol, max_evals):
    m0 = m_lin
    f0 = fraction(m0)
    evals = [(m0, f0)]
    # second guess: scale the remaining part of the shift (earlier DARTs give the part at m = 0)
    f_prior = fraction(0.0) if has_prior else 0.0
    m1 = m0 * (1.0 - f_prior) / max(f0 - f_prior, 1e-6)
    m1 = float(np.clip(m1, 0.1 * m_lin, 10.0 * m_lin))
    f1 = fraction(m1)
    evals.append((m1, f1))
    for _ in range(max_evals - 2):
        if abs(f1 - 1.0) < tol or f1 == f0:
            break
        x0, x1 = np.log(m0), np.log(m1)
        x2 = x1 - (f1 - 1.0) * (x1 - x0) / (f1 - f0)
        m0, f0 = m1, f1
        m1 = float(np.exp(np.clip(x2, np.log(0.1 * m_lin), np.log(10 * m_lin))))
        f1 = fraction(m1)
        evals.append((m1, f1))
    return evals


def _crossing(evals, m_lin):
    """Mass where the evaluated fraction crosses 1: quadratic through the bracketing pair and its nearest
    neighbour, linear if that leaves the bracket, linear extrapolation from the end if nothing brackets it."""
    pts = sorted(evals)
    m = np.array([e[0] for e in pts])
    f = np.array([e[1] for e in pts])
    lo, hi = 0.1 * m_lin, 10.0 * m_lin
    for i in range(len(m) - 1):
        if (f[i] - 1.0) * (f[i + 1] - 1.0) <= 0 and f[i] != f[i + 1]:
            m_lin_x = m[i] + (1.0 - f[i]) * (m[i + 1] - m[i]) / (f[i + 1] - f[i])
            j = [k for k in (i - 1, i + 2) if 0 <= k < len(m)]
            if j:
                k = min(j, key=lambda k: abs(f[k] - 1.0))
                idx = sorted([i, i + 1, k])
                c = np.polyfit(m[idx], f[idx] - 1.0, 2)
                roots = [r.real for r in np.roots(c) if abs(r.imag) < 1e-12 and m[i] <= r.real <= m[i + 1]]
                if roots:
                    return float(roots[0])
            return float(m_lin_x)
    a, b = (0, 1) if f[0] > 1.0 else (len(m) - 2, len(m) - 1)  # all above 1: go lighter; all below: heavier
    if f[a] == f[b]:
        return float(np.clip(m[b] * 2.0 if f[0] < 1.0 else m[a] / 2.0, lo, hi))
    return float(np.clip(m[a] + (1.0 - f[a]) * (m[b] - m[a]) / (f[b] - f[a]), lo, hi))


def _parallel_search(fractions, m_lin, workers, has_prior, tol, guess=None, rounds=3):
    """Round 1: n - 1 masses at once around the expected answer (within +-3 % of the last refinement's ratio to
    the linear mass for the same opportunity and target, else a wide bracket); later rounds: masses within
    +-1.5 % of the interpolated crossing."""
    n = max(2, workers - 1)  # one worker computes the shared run without the pushes
    if guess:
        factors = guess * np.geomspace(0.97, 1.03, n)  # 1.5 % apart: usually done in one round
    else:
        factors = np.geomspace(0.5, 1.25, n)
    masses = [float(m_lin * k) for k in factors]
    if has_prior:
        masses[0] = 0.0  # earlier DARTs alone: anchors the curve at m = 0
    evals = list(zip(masses, fractions(masses)))
    for _ in range(rounds - 1):
        best = min(evals, key=lambda e: abs(e[1] - 1.0))
        if abs(best[1] - 1.0) < tol / 2:
            break
        m_star = _crossing(evals, m_lin)
        k = min(workers, 3)
        steps = [0.0, -0.015, 0.015, -0.03, 0.03][:k]
        masses = [m_star * (1 + s) for s in steps]
        evals += list(zip(masses, fractions(masses)))
        if abs(min(evals, key=lambda e: abs(e[1] - 1.0))[1] - 1.0) < tol / 2:
            break
    return evals


def refine_init(generation):
    """Process-pool initializer: `generation` (shared int) is bumped by the UI for every new request."""
    from core import parallel

    parallel.set_qos()
    parallel.exit_with_parent()
    _STATE["generation"] = generation


def refine_close():
    """Last job before the UI closes the refinement process: stop the full-model workers, so the process can
    then exit normally."""
    pool = _STATE.pop("pool", None)
    if pool is not None:
        pool.close()


def _watch(job_id, pool, done, flags):
    """Abort the running job as soon as a newer request arrives: its full-model workers are stopped (fresh ones
    start with the next job), so the stale evaluations stop using cores at once."""
    gen = _STATE.get("generation")
    while gen is not None and not done.wait(0.2):
        if gen.value != job_id:
            flags["aborted"] = True
            pool.close()
            return


def refine_job(args, job_id=None):
    """Process-pool entry: args = (params, opportunity, b_now, t_after, prior_impulses, number, spin_change).
    Returns None if a newer request (job_id) superseded it."""
    import threading
    import time

    from core import parallel
    from core.fullmodel import FullModelPool

    p, opportunity, b_now, t_after, prior, number, spin = args
    gen = _STATE.get("generation")
    stale = lambda: gen is not None and job_id is not None and gen.value != job_id
    time.sleep(0.5)  # requests come in bursts while a slider moves: only the last one is worth starting
    if stale():
        return None
    if _STATE.get("pool") is None:
        _STATE["pool"] = FullModelPool(parallel.budget()["refine_procs"])
    if "pl" not in _STATE:
        from core.ephemeris import Ephemeris
        from core.planner import Planner
        from core.scenario import KeyholeScenario

        sc = KeyholeScenario.load()
        eph = Ephemeris().with_bennu(sc.nominal)
        _STATE["pl"] = Planner(eph, sc.nominal, sc)
    from core.dynamics.asteroid import reset_spin, set_spin_change

    reset_spin()
    if spin is not None:
        set_spin_change(*spin)
    key = (opportunity, number, len(prior), round(p.dart_target_km, -4))  # what sets the nonlinearity
    pool, done, flags = _STATE["pool"], threading.Event(), {"aborted": False}
    if job_id is not None:
        threading.Thread(target=_watch, args=(job_id, pool, done, flags), daemon=True).start()
    try:
        r = refine_dart_mass(_STATE["pl"], p, opportunity, b_now=b_now, t_after=t_after, prior_impulses=prior,
                             number=number, pool=pool, guess=_STATE.get(("ratio", key)))
    except Exception:
        if not flags["aborted"]:
            raise
        r = None
    finally:
        done.set()
    if flags["aborted"] or stale():
        if flags["aborted"]:
            _STATE["pool"] = None
        return None
    if r is not None and abs(r["fraction"] - 1.0) < 0.03:
        _STATE[("ratio", key)] = r["mass"] / r["linear_mass"]
    return r
