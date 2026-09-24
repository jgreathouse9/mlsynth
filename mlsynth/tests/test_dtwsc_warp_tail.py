"""DTWSC's counterfactual over the warp's tail, as a root-cause ladder.

The incident: after DTWSC's donor-weight solve moved from cvxpy/ECOS to the
active set, ``test_outcome_only_backends_agree_closely`` failed. The two
``sc_backend`` options returned -1.4117 and -3.1799 against a tolerance of
0.35, where before they had returned -2.9424 and -3.1799.

The weights were not the fault. On that panel the two solvers agree to
2.5e-06 and the active set reaches the lower objective (7.0108369926e+01
against 7.0108369972e+01). What differs is that ECOS leaves an excluded donor
at ~1e-9 while the active set leaves it at exactly 0, and the counterfactual
read support as ``weights[observed].sum() > 0``.

Two faults, one rung each below.

1. Support read at ``> 0``. Dust of 1e-18 on four excluded donors was enough
   to move the reported ATT by 1.53.
2. The synthetic control was rebuilt period by period, renormalising over
   whichever donors were observed. At t=25 that made it a single donor at
   weight one -- a different unit from the one that was fit. Meanwhile the
   VanillaSC path completed the same gap by carrying values forward, so the
   two backends were never the same estimator.

Both live at the counterfactual link, downstream of the weights, so the fit
and the donor weights are untouched by construction and none of the five
standing estimation candidates applies. The decomposition of the observed
move is exact: of the 1.530696 the ATT moved, 1.530704 is the two periods the
active set dropped and -0.000008 is everything else.

A third was suspected and is not one. The post periods without a
counterfactual are already counted, on
``res.metadata["n_post_periods_undefined"]`` -- an estimator-specific field,
which is why a first pass over ``effects`` and ``fit_diagnostics`` read the
ATT as silently averaged over a short window.

The remedy for the second fault is a choice, and it was made by measurement.
``conflictlab/dsc`` reports NA over the tail and takes its ATT over what is
left; ``benchmarks/reference/dtwsc_basque/reference.R`` says so in its own
words -- "three donors' warped series end one period short, so 1997 is NA".
Copying that rule costs a great deal here. Against a planted constant effect
of -3.0 over twelve seeded panels, carrying the tail forward recovers it to a
mean absolute error of 0.52 with a worst case of 1.09; reporting NA gives 1.67
and 3.87, and the wrong sign on two of the twelve. The tail is where a
mis-timed gap path is largest, so dropping it biases the ATT toward zero. The
Basque cross-validation cannot separate the two -- three of its sixteen donors
run short and none of them carries weight -- so it stays bit-identical either
way, and the difference from the reference is recorded instead of hidden.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import DTWSC
from mlsynth.utils.dtwsc_helpers import pipeline as P

from test_dtwsc import base_config, make_panel

SEED = 60          # the panel the incident was found on
T_TREAT = 18
ATT = -3.17993082  # both backends, over all ten post periods


def _fit(sc_backend="simplex"):
    return DTWSC(base_config(make_panel(seed=SEED), sc_backend=sc_backend)).fit()


def _cf(res):
    return np.asarray(res.time_series.counterfactual_outcome, float).ravel()


# --- rung 0: which reported quantity, and over what window ------------------

def test_the_att_is_the_mean_over_the_periods_that_have_a_counterfactual():
    res = _fit()
    y, cf = np.asarray(res.time_series.observed_outcome, float).ravel(), _cf(res)
    gap = (y - cf)[T_TREAT:]
    assert res.effects.att == pytest.approx(float(np.nanmean(gap)), abs=1e-9)
    assert res.effects.att == pytest.approx(ATT, abs=1e-6)


def test_the_tail_is_carried_forward_so_no_post_period_is_lost_here():
    """Reporting NA over the tail loses three of ten periods on this panel.

    Which is the whole cost: the recovery panels put the mis-timed gap path's
    largest values there, and dropping them pulls the ATT toward zero.
    """
    res = _fit()
    assert np.isfinite(_cf(res)[T_TREAT:]).all()
    assert res.metadata["n_post_periods_undefined"] == 0


def test_the_window_the_att_was_taken_over_is_on_the_result():
    """The contract that was already enforced, pinned so it stays enforced.

    ``n_post_periods_undefined`` lives on DTWSC's own ``metadata``, not on
    ``effects`` or ``fit_diagnostics``. Reading only the standard sub-models
    makes a short window look unreported, which is a conclusion this test
    exists to stop anyone reaching twice.
    """
    res = _fit()
    lost = int((~np.isfinite(_cf(res)[T_TREAT:])).sum())
    assert res.metadata["n_post_periods_undefined"] == lost


# --- rung 1: which other outputs move with it -------------------------------

def test_both_backends_return_the_same_counterfactual():
    """The test that failed asserted the ATTs agree to 0.35; they are equal.

    Two backends fitting the same objective on the same donors differ only in
    what they do where a warped series runs short. Once both carry it forward,
    there is nothing left to differ in, and the 0.35 tolerance the old test
    needed was covering a real disagreement that ECOS's dust kept inside it.
    """
    a, b = _fit("simplex"), _fit("outcome-only")
    np.testing.assert_allclose(_cf(a), _cf(b), rtol=0, atol=1e-9)
    assert a.effects.att == pytest.approx(b.effects.att, abs=1e-9)
    assert a.metadata["n_post_periods_undefined"] == \
        b.metadata["n_post_periods_undefined"] == 0


# --- rung 2: which step produced them ---------------------------------------

def test_the_synthetic_control_is_one_combination_in_every_period():
    """Renormalising over the observed donors rebuilt it period by period.

    Donor 0 carries all the weight and runs short from t=2. Reading support
    per period would make the counterfactual donor 1's path at weight one --
    a different synthetic unit from the one that was fit, reported as though
    it were the same one.
    """
    warped = np.array([[1.0, 100.0], [2.0, 200.0], [np.nan, 300.0],
                       [np.nan, 400.0]])
    np.testing.assert_allclose(P._counterfactual(warped, np.array([1.0, 0.0])),
                               [1.0, 2.0, 2.0, 2.0])


def test_carry_forward_fills_a_tail_and_leaves_the_rest():
    block = np.array([[1.0, 10.0], [2.0, np.nan], [np.nan, np.nan],
                      [np.nan, 40.0]])
    np.testing.assert_allclose(
        P._carry_forward(block),
        np.array([[1.0, 10.0], [2.0, 10.0], [2.0, 10.0], [2.0, 40.0]]))


def test_carry_forward_cannot_invent_a_leading_value():
    """Nothing precedes a leading NaN, so that period has no counterfactual."""
    warped = np.array([[np.nan, 1.0], [10.0, 2.0], [11.0, 3.0]])
    out = P._counterfactual(warped, np.array([1.0, 0.0]))
    assert not np.isfinite(out[0])
    np.testing.assert_allclose(out[1:], [10.0, 11.0])


# --- rung 3: which invariant the old behaviour violated ---------------------

@settings(max_examples=20, deadline=None)
@given(dust=st.floats(min_value=0.0, max_value=1e-9))
def test_the_estimate_does_not_turn_on_solver_dust(dust):
    """An excluded donor at 1e-18 or at 0 is the same fit, so the same ATT.

    This is the invariant the incident violated. Dust of 1e-18 on four donors
    moved the ATT from -1.4117 to -2.9631 -- a denormal deciding an estimate.
    """
    real = P._simplex_weights
    try:
        P._simplex_weights = lambda t, d: (
            (lambda w: w / w.sum())(real(t, d) + dust))
        assert _fit().effects.att == pytest.approx(ATT, abs=1e-6)
    finally:
        P._simplex_weights = real


def test_a_negligible_weight_is_not_support():
    """The tolerance, on a design where ``> 0`` and ``> tol`` come apart.

    Donor 0 is dust and runs short. Read at ``> 0`` it is support, so the last
    period has no counterfactual and the window rests on a 1e-15 weight. Read
    at the tolerance it is not, and donor 1 carries the period.
    """
    warped = np.array([[np.nan, 1.0], [10.0, 2.0], [11.0, 3.0]])
    np.testing.assert_allclose(
        P._counterfactual(warped, np.array([1e-15, 1.0 - 1e-15])),
        [1.0, 2.0, 3.0])


def test_a_donor_that_carries_no_weight_cannot_lose_a_period():
    """Three of sixteen Basque donors run short and none carries weight."""
    warped = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
    np.testing.assert_allclose(
        P._counterfactual(warped, np.array([0.0, 1.0])), [1.0, 2.0, 3.0])
