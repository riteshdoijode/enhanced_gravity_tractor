"""Multi-core execution: the core budget, full-model runs in worker processes (identical to running them in the
calling thread) and the parallel search of the DART mass."""

import numpy as np
import pytest

from core import parallel
from core.constants import DAY, YEAR
from core.fullmodel import FullModelPool, Inline
from core.timeutil import iso_to_t
from core.tow import Deflection, deflected_states_at_pre, flyby_and_tail


@pytest.fixture(scope="module")
def setup():
    from tests.keyhole import setup as keyhole

    sc, eph, pl = keyhole()
    return eph, sc.nominal, pl


def test_budget_shares_the_cores():
    b = parallel.budget(12)
    assert b["sim_threads"] + b["sim_fullmodel"] + 3 * b["planner_threads"] + b["refine_procs"] <= 12 + 6
    assert b["refine_procs"] == 6 and b["sim_fullmodel"] == 2
    small = parallel.budget(1)
    assert all(v >= 1 for v in small.values())


def test_full_model_pool_matches_inline(setup):
    """Runs with and without the pushes, for two deflections at once (one with a tow), and the flyby from t_pre:
    the worker processes give bit-identical results to the calling thread."""
    eph, nom, pl = setup
    t_imp = iso_to_t("2030-06-22")
    t_pre = t_imp + YEAR  # a short stretch: the comparison, not the century, is the point
    d1 = Deflection(nom, t_imp - DAY)
    d1.impulse(t_imp, np.array([1.5e-5, -2.5e-5, 1e-5]))
    d2 = Deflection(nom, t_imp - DAY)
    d2.impulse(t_imp, np.array([-1e-5, 0.0, 2e-5]))
    d2.advance(t_imp + 20 * DAY, accel=lambda t: np.array([0.0, 3e-11, 0.0]))
    pool = FullModelPool(3)
    try:
        cache_a, cache_b = {}, {}
        inline = deflected_states_at_pre(eph, nom, [d1, d2], t_pre, cache=cache_a, pool=Inline(eph))
        pooled = deflected_states_at_pre(eph, nom, [d1, d2], t_pre, cache=cache_b, pool=pool)
        args = (nom.state(pl.sc.t_pre), pl.sc.t_pre, pl.t_ca + 2 * DAY, pl.t_ca, pl.bplane.basis, False)
        tail_inline = Inline(eph).submit(flyby_and_tail, *args).result()
        tail_pooled = pool.submit(flyby_and_tail, *args).result()
    finally:
        pool.close()
    for (xa, da), (xb, db) in zip(inline, pooled):
        assert np.array_equal(xa, xb) and np.array_equal(da, db)
    assert np.linalg.norm(inline[0][1][3:]) > 1e-6  # the pushes made a difference
    assert len(cache_a) == len(cache_b) == 2  # one run without pushes per set of segment edges
    assert np.array_equal(tail_inline["b_nonlinear"], tail_pooled["b_nonlinear"])
    assert tail_inline["q"] == tail_pooled["q"]


@pytest.mark.parametrize("has_prior", [False, True])
def test_parallel_mass_search(has_prior):
    """The parallel rounds find the mass whose nonlinear shift equals the target, on a smooth curve with a
    strong second-order part (fraction 1.36 at the linear mass, as for a ~270,000 km shift)."""
    from core.refine import _crossing, _parallel_search

    m_lin, f_prior = 190.0, (0.3 if has_prior else 0.0)

    def frac(m):
        x = (1 - f_prior) * m / m_lin
        return f_prior + x * (1 + 0.36 * x)

    calls = []

    def fractions(masses):
        calls.append(len(masses))
        return [frac(m) for m in masses]

    evals = _parallel_search(fractions, m_lin, 6, has_prior, tol=0.03)
    best = min(evals, key=lambda e: abs(e[1] - 1.0))
    assert abs(best[1] - 1.0) < 0.015
    assert len(calls) <= 2 and calls[0] == 5  # 5 masses + the run without pushes on 6 workers, then a check
    m_true = min((abs(frac(m) - 1.0), m) for m in np.linspace(50, 250, 200001))[1]
    assert abs(_crossing(evals, m_lin) - m_true) / m_true < 0.003
    # a known ratio from an earlier refinement: one round
    calls.clear()
    evals = _parallel_search(fractions, m_lin, 6, has_prior, tol=0.03, guess=m_true / m_lin * 1.01)
    assert len(calls) == 1 and abs(min(evals, key=lambda e: abs(e[1] - 1.0))[1] - 1.0) < 0.015
