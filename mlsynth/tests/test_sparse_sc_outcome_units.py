"""SparseSC's predictor selection does not depend on the outcome's units (#933).

The outer V-objective trades an outcome MSE, in outcome units squared, against
``lam * ||v||_1`` on a grid fixed in absolute units, so recording the outcome in other
units rescaled every lambda by ``1 / s^2``. On the Basque panel (13 predictors, treated
from 1975) the shipped defaults kept 3 predictors with GDP in thousands of dollars, 7
in dollars, and all 13 at a thousandth. The sweep now divides the outcome blocks by
their pooled pre-treatment standard deviation (``outcome_scale``), so the grid is
relative to the outcome's own variation; ``lambda_units="raw"`` keeps the unit-dependent
grid of Vives-i-Bastida's MATLAB driver, which the Proposition 99 replication uses.

Scaling by a power of two is exact in floating point, so under the unit-free grid the
sweep's inputs, and so its answer, are bit-identical across such scales.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import SparseSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sparse_sc_helpers.inference import run_placebo
from mlsynth.utils.sparse_sc_helpers.objective import outcome_scale
from mlsynth.utils.sparse_sc_helpers.optimization import sweep_lambda

GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
T, T0, T0_TRAIN = 20, 14, 10


def _arrays(seed=0, N=6, P=4):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, P)); L = rng.standard_normal((N + 1, P))
    Y = 50.0 + 10.0 * F @ L.T + 3.0 * rng.standard_normal((T, N + 1))
    Y[T0:, 0] += -25.0
    X = L.T                                            # (P, N + 1) predictors, already on a unit scale
    return X[:, 0], X[:, 1:], Y[:, 0], Y[:, 1:]


def _panel(seed=0, N=6, P=4, scale=1.0):
    X1, X0, Y1, Y0 = _arrays(seed, N, P)
    X = np.column_stack([X1, X0]); Y = np.column_stack([Y1, Y0]) * scale
    rec = []
    for u in range(N + 1):
        for t in range(T):
            rec.append({"unit": f"u{u}", "year": 2000 + t, "y": float(Y[t, u]),
                        "tr": int(u == 0 and t >= T0), **{f"p{p}": float(X[p, u]) for p in range(P)}})
    return pd.DataFrame(rec)


def _fit(df, **kw):
    cfg = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year", covariates=["p0", "p1", "p2", "p3"],
               lambda_grid=GRID, run_inference=False, display_graphs=False, robust_selection=False)
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SparseSC(cfg).fit()


def _sweep(X1, X0, Y1, Y0, scale=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sweep_lambda(X1=X1, X0=X0, Y1=Y1, Y0=Y0, T0_total=T0, T0_train=T0_TRAIN,
                            lambda_grid=np.array(GRID), robust=False, outcome_scale=scale)


# --------------------------------------------------------------------------- smoke
def test_a_fit_records_its_outcome_scale():
    """By default the grid is on the outcome's own scale: the pooled pre-period spread, not 1."""
    res = _fit(_panel())
    X1, X0, Y1, Y0 = _arrays()
    assert res.design.outcome_scale == pytest.approx(outcome_scale(Y1, Y0, T0))
    assert res.design.outcome_scale != pytest.approx(1.0)


# --------------------------------------------------------------------------- unit: the scale
def test_the_scale_is_the_pooled_pre_treatment_spread():
    X1, X0, Y1, Y0 = _arrays()
    Y = np.column_stack([Y1, Y0])[:T0]
    assert outcome_scale(Y1, Y0, T0) == pytest.approx(float(np.std(Y)))


def test_the_scale_follows_the_units_and_ignores_the_level():
    X1, X0, Y1, Y0 = _arrays()
    s = outcome_scale(Y1, Y0, T0)
    assert outcome_scale(1e3 * Y1, 1e3 * Y0, T0) == pytest.approx(1e3 * s)
    assert outcome_scale(Y1 + 1e4, Y0 + 1e4, T0) == pytest.approx(s, rel=1e-9)


def test_the_post_period_does_not_enter_the_scale():
    X1, X0, Y1, Y0 = _arrays()
    Y1b = Y1.copy(); Y1b[T0:] += 1e6
    assert outcome_scale(Y1b, Y0, T0) == outcome_scale(Y1, Y0, T0)


# --------------------------------------------------------------------------- unit: the answer
def _support(v, tol=1e-10):
    return tuple(np.flatnonzero(np.asarray(v) > tol))


def test_selection_and_weights_do_not_depend_on_the_outcome_units():
    """Thousands against units: the Basque panel's case, on a panel that fits in a second.

    A factor of 1000 is not exact in floating point, so the inputs differ in their last
    bits, and L-BFGS-B stops at a different point of the outer objective's flat valley.
    On this panel the validation losses at 1e-3, 1e-2 and 1e-1 agree to about 1e-6
    relative, so which of them is the argmin is rounding: Python 3.13 on CI chose 0.1 in
    one unit and 0.001 in the other. What is identified is the curve, in outcome-SD
    units, to where L-BFGS-B stops (2.1e-4 relative at lambda = 1, measured), and so the
    set of near-minimising lambdas; the support; and the ATT, to the spread among those
    tied lambdas (1.9e-4 relative on CI). The unpenalised end is 19 percent worse, so a
    tolerance of 1e-3 still separates the curve's shape.
    """
    X1, X0, Y1, Y0 = _arrays()
    *_, val_a, _ = _sweep(X1, X0, Y1, Y0)
    *_, val_b, _ = _sweep(X1, X0, 1e3 * Y1, 1e3 * Y0)
    val_a, val_b = np.asarray(val_a), np.asarray(val_b)
    np.testing.assert_allclose(val_b, val_a, rtol=1e-3)
    a, b = _fit(_panel()), _fit(_panel(scale=1e3))
    for lam, val in ((a.design.opt_lambda, val_b), (b.design.opt_lambda, val_a)):
        assert val[GRID.index(lam)] <= val.min() * (1.0 + 1e-3)
    assert _support(a.design.v) == _support(b.design.v)
    assert b.effects.att == pytest.approx(1e3 * a.effects.att, rel=1e-3)


def test_the_raw_grid_selects_differently_in_other_units():
    """The defect on this panel: in raw units a factor of 1000 moves the selected lambda and drops a predictor."""
    a, b = _fit(_panel(), lambda_units="raw"), _fit(_panel(scale=1e3), lambda_units="raw")
    assert (a.design.opt_lambda, _support(a.design.v)) != (b.design.opt_lambda, _support(b.design.v))


def test_raw_units_keep_the_grid_in_outcome_units():
    res = _fit(_panel(), lambda_units="raw")
    assert res.design.outcome_scale == 1.0


def test_raw_units_are_the_sweep_without_a_scale():
    X1, X0, Y1, Y0 = _arrays()
    v_raw, lam_raw, *_ = _sweep(X1, X0, Y1, Y0, scale=1.0)
    v_sd, lam_sd, *_ = _sweep(X1, X0, Y1 / outcome_scale(Y1, Y0, T0), Y0 / outcome_scale(Y1, Y0, T0), scale=1.0)
    v_auto, lam_auto, *_ = _sweep(X1, X0, Y1, Y0)
    assert lam_auto == lam_sd and np.array_equal(v_auto, v_sd)
    assert (lam_raw, tuple(v_raw)) != (lam_sd, tuple(v_sd)) or np.isclose(outcome_scale(Y1, Y0, T0), 1.0)


def test_placebo_refits_use_the_same_scale():
    """At the selected lambda each placebo refit is the treated fit's problem, on the same units."""
    X1, X0, Y1, Y0 = _arrays()
    s = 2.0 ** 10
    kw = dict(T0_total=T0, T0_train=T0_TRAIN, selected_lambda=1e-2, n_placebo=3, seed=0)
    a, pa, _ = run_placebo(Y0=Y0, Y1=Y1, X0=X0, X1=X1, observed_att=-25.0,
                           outcome_scale=outcome_scale(Y1, Y0, T0), **kw)
    b, pb, _ = run_placebo(Y0=s * Y0, Y1=s * Y1, X0=X0, X1=X1, observed_att=-25.0 * s,
                           outcome_scale=outcome_scale(s * Y1, s * Y0, T0), **kw)
    np.testing.assert_array_equal(b, s * a)
    assert pa == pb


# --------------------------------------------------------------------------- edge
def test_a_constant_outcome_has_scale_one():
    X1, X0, Y1, Y0 = _arrays()
    assert outcome_scale(np.full_like(Y1, 3.0), np.full_like(Y0, 3.0), T0) == 1.0


# --------------------------------------------------------------------------- failure
def test_a_non_finite_pre_period_outcome_is_refused():
    X1, X0, Y1, Y0 = _arrays()
    Y0 = Y0.copy(); Y0[2, 1] = np.nan
    with pytest.raises(MlsynthDataError, match="outcome"):
        outcome_scale(Y1, Y0, T0)


def test_an_unknown_lambda_unit_is_refused():
    with pytest.raises(MlsynthConfigError, match="lambda_units"):
        _fit(_panel(), lambda_units="dollars")


# --------------------------------------------------------------------------- property
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(-30, 30))
def test_property_a_power_of_two_change_of_units_leaves_the_sweep_bit_identical(seed, k):
    X1, X0, Y1, Y0 = _arrays(seed)
    s = 2.0 ** k
    v_a, lam_a, _, out_a, val_a, path_a = _sweep(X1, X0, Y1, Y0)
    v_b, lam_b, _, out_b, val_b, path_b = _sweep(X1, X0, s * Y1, s * Y0)
    assert lam_a == lam_b
    assert np.array_equal(v_a, v_b) and np.array_equal(path_a, path_b)
    np.testing.assert_array_equal(val_a, val_b)
