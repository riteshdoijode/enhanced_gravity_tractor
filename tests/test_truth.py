"""Bennu's ground truth in the live run (core/truth.py): the full-force-model integration of the deflected orbit
agrees with the independent nonlinear check (core/tow.py), a vanishing push leaves Bennu in the keyhole (it
still hits Earth), and the tractor's pull enters as block means that are re-flown when a block closes."""

import numpy as np
import pytest

from core.constants import DAY, LUNAR_DISTANCE
from core.timeutil import iso_to_t
from tests.keyhole import setup


@pytest.fixture(scope="module")
def ks():
    return setup()


def test_truth_matches_the_nonlinear_check_and_keeps_the_keyhole(ks):
    from core.tow import Deflection, verify_nonlinear
    from core.truth import BennuTruth

    sc, eph, pl = ks
    t_imp = iso_to_t("2030-06-22")
    dv = np.array([6.6e-5, -6.2e-5, -8.0e-5])  # a DART-class push
    tr = BennuTruth(eph, sc.nominal, sc, pl.bplane.basis, pl.t_ca)
    tr.impulse(t_imp, dv)
    assert tr.ensure(t_imp + 30 * DAY)  # inline: integrated when asked for
    assert 0 < np.linalg.norm(tr.delta(t_imp + 30 * DAY)[:3]) < 1e6  # tens to hundreds of km after a month
    pred = tr.request_prediction()
    d = Deflection(sc.nominal, t_imp - DAY)
    d.impulse(t_imp, dv)
    ref = verify_nonlinear(eph, sc.nominal, d, pl, to_end=False)
    shift = np.linalg.norm(ref["b_nonlinear"] - pl.b0)
    assert np.linalg.norm(pred["b_nonlinear"] - ref["b_nonlinear"]) < 1e-3 * shift  # same physics, other stops
    assert pred["q"] == pytest.approx(ref["q"], rel=1e-3) and pred["q"] > LUNAR_DISTANCE
    assert not np.isfinite(pred["t_impact"])  # deflected: no impact
    # a vanishing push: Bennu stays in the 3 km keyhole and still hits Earth on the scenario date
    tiny = BennuTruth(eph, sc.nominal, sc, pl.bplane.basis, pl.t_ca)
    tiny.impulse(t_imp, dv / np.linalg.norm(dv) * 1e-11)
    pt = tiny.request_prediction()
    assert np.isfinite(pt["t_impact"]) and abs(pt["t_impact"] - sc.t_impact) < DAY
    # the state queries are continuous across the segment stops and agree with the samples
    tn, dn = tr.delta_samples()
    k = len(tn) // 2
    assert np.allclose(tr.delta(tn[k])[:3], dn[k][:3], rtol=1e-9, atol=1e-3)


def test_tow_blocks_are_reflown_when_they_close(ks):
    from core.truth import TOW_BLOCK, BennuTruth

    sc, eph, pl = ks
    t0 = iso_to_t("2033-01-01")
    tr = BennuTruth(eph, sc.nominal, sc, pl.bplane.basis, pl.t_ca)
    a = np.array([2e-11, -1e-11, 0.5e-11])
    for k in range(10):  # 10 daily steps of the tractor's pull: a block in progress
        tr.tow(t0 + k * DAY, t0 + (k + 1) * DAY, a)
    assert tr.ensure(t0 + 10 * DAY) and not tr.pushes  # flown with the mean so far, not yet a final push
    for k in range(10, 40):
        tr.tow(t0 + k * DAY, t0 + (k + 1) * DAY, a * (2.0 if k >= 20 else 1.0))
    assert len(tr.pushes) == 1 and tr.pushes[0][0] == "tow"
    blk = tr.pushes[0]
    assert blk[2] == TOW_BLOCK and np.allclose(blk[3], a * 40 / 30)  # the exact block mean (20 d at a, 10 d at 2a)
    assert tr.ensure(t0 + 40 * DAY)
    d = tr.delta(t0 + 40 * DAY)
    assert 0 < np.linalg.norm(d[:3]) < 1e4  # metres to km after a month of a 1e-11 m/s^2 pull
