"""The two-group aggregate regression, shared by every estimator that fits one.

Three estimators fit ``y_t = alpha + beta x_t`` on a pair of group aggregates:
FDID's ADID arm on the control mean, TBR (and through it TBRMM) on the control
sum, and PANGEO on the control aggregate with an optional trend. Two of them also
form the prediction variance ``xbar' (X'X)^-1 xbar`` that the interval's width
depends on.

Test-first, per ``agents/agents_tests.md``: this file is written before
``mlsynth/utils/groupfit/`` exists and is RED on the import.

The referees are stated independently of the implementation. Coefficients are
checked against ``np.linalg.lstsq`` on the explicit design, the prediction
variance against ``xbar @ inv(X'X) @ xbar`` formed directly, and the accuracy at
geo-aggregate scale against float128, because the whole reason the package exists
is that the careless route loses digits exactly there.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.groupfit import (
    AGGREGATIONS,
    GroupSums,
    TwoGroupFit,
    aggregate_group,
    fit_on_sums,
    fit_two_group,
    group_sums,
    is_identified,
    prediction_variance,
    unscaled_cov,
)


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def market_matrix(n_periods=90, n_units=8, level=5000.0, spread=0.25, seed=0):
    """Units of differing size sharing a factor, the shape a geo panel has."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(0.0, 0.6, n_units)
    size = level * size / size.mean()
    common = 1.0 + spread * rng.normal(size=(n_periods, 1))
    idio = 1.0 + 0.1 * rng.normal(size=(n_periods, n_units))
    return size * (0.6 * common + 0.4 * idio)


def explicit(y, x):
    """``(alpha, beta)`` from the design ``[1, x]``, the route this replaced."""
    design = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


# --------------------------------------------------------------------------- #
# Layer 4: smoke
# --------------------------------------------------------------------------- #
def test_the_pipeline_runs_end_to_end():
    Y = market_matrix()
    x = aggregate_group(Y, [0, 1, 2], how="sum")
    y = aggregate_group(Y, [5, 6], how="sum")
    fit = fit_two_group(y, x)
    assert isinstance(fit, TwoGroupFit)
    assert isinstance(fit.sums, GroupSums)
    assert np.isfinite([fit.alpha, fit.beta, fit.sigma_sq]).all()
    assert fit.resid.shape == y.shape
    assert np.isfinite(prediction_variance(fit.sums, float(x.mean())))


# --------------------------------------------------------------------------- #
# Layer 1: aggregation
# --------------------------------------------------------------------------- #
def test_the_two_aggregations_are_the_documented_ones():
    assert AGGREGATIONS == ("sum", "mean")


def test_the_sum_and_the_mean_differ_by_the_group_size():
    Y = market_matrix()
    cols = [1, 3, 4, 6]
    s = aggregate_group(Y, cols, how="sum")
    m = aggregate_group(Y, cols, how="mean")
    assert np.allclose(m, s / len(cols), rtol=0, atol=1e-12)
    assert np.allclose(s, Y[:, cols].sum(axis=1), rtol=0, atol=1e-12)


def test_a_one_unit_group_aggregates_to_that_unit():
    Y = market_matrix()
    for how in AGGREGATIONS:
        assert np.allclose(aggregate_group(Y, [2], how=how), Y[:, 2])


def test_an_unknown_aggregation_names_the_accepted_ones():
    Y = market_matrix()
    with pytest.raises(MlsynthDataError, match="sum"):
        aggregate_group(Y, [0, 1], how="total")


def test_an_empty_group_is_refused():
    with pytest.raises(MlsynthDataError):
        aggregate_group(market_matrix(), [])


# --------------------------------------------------------------------------- #
# Layer 1: the sums
# --------------------------------------------------------------------------- #
def test_the_sums_are_the_centred_moments():
    Y = market_matrix(seed=1)
    x = aggregate_group(Y, [0, 1], how="sum")
    y = aggregate_group(Y, [4, 5], how="sum")
    s = group_sums(y, x)
    dx, dy = x - x.mean(), y - y.mean()
    assert s.n == x.size
    assert s.sum_x == pytest.approx(float(x.sum()), rel=1e-15)
    assert s.sum_y == pytest.approx(float(y.sum()), rel=1e-15)
    assert s.s_xx == pytest.approx(float(dx @ dx), rel=1e-12)
    assert s.s_xy == pytest.approx(float(dx @ dy), rel=1e-12)
    assert s.s_yy == pytest.approx(float(dy @ dy), rel=1e-12)


def test_identification_is_exactly_a_varying_regressor():
    Y = market_matrix(seed=2)
    y = aggregate_group(Y, [0, 1], how="sum")
    assert is_identified(group_sums(y, aggregate_group(Y, [2, 3], how="sum")))
    assert not is_identified(group_sums(y, np.full(y.size, 7.0)))


def test_mismatched_lengths_are_refused():
    with pytest.raises(MlsynthDataError):
        group_sums(np.arange(5.0), np.arange(6.0))


# --------------------------------------------------------------------------- #
# Layer 1: the fit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("how", AGGREGATIONS)
def test_the_coefficients_are_the_least_squares_ones(how):
    """The referee is ``lstsq`` on the explicit design, under both aggregations."""
    for seed in range(6):
        Y = market_matrix(seed=seed)
        x = aggregate_group(Y, [0, 1, 2], how=how)
        y = aggregate_group(Y, [5, 6], how=how)
        fit = fit_two_group(y, x)
        a, b = explicit(y, x)
        assert fit.alpha == pytest.approx(a, rel=1e-9, abs=1e-8)
        assert fit.beta == pytest.approx(b, rel=1e-10)


def test_the_residual_and_the_variance_agree_with_each_other():
    Y = market_matrix(seed=3)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    fit = fit_two_group(y, x)
    assert np.allclose(fit.resid, y - (fit.alpha + fit.beta * x), atol=1e-9)
    assert fit.df == y.size - 2
    assert fit.sigma_sq == pytest.approx(float(fit.resid @ fit.resid) / fit.df,
                                        rel=1e-15)


def test_summing_or_averaging_the_groups_is_the_same_regression():
    """The ADID / TBR relation, as a test.

    Li and Van den Bulte regress on the control average and Kerman, Wang and
    Vaver on the control sum. The regressor differs by the group size, so the
    intercept, the fitted values, the residuals and the residual variance are the
    same numbers and only the slope's units change: ``beta_mean = k beta_sum``.
    """
    Y = market_matrix(seed=4)
    ctl, trt = [0, 1, 2, 3], [5, 6]
    k = len(ctl)
    f_sum = fit_two_group(aggregate_group(Y, trt, how="sum"),
                          aggregate_group(Y, ctl, how="sum"))
    f_mean = fit_two_group(aggregate_group(Y, trt, how="sum"),
                           aggregate_group(Y, ctl, how="mean"))
    assert f_mean.beta == pytest.approx(k * f_sum.beta, rel=1e-10)
    assert f_mean.alpha == pytest.approx(f_sum.alpha, rel=1e-9, abs=1e-8)
    assert f_mean.sigma_sq == pytest.approx(f_sum.sigma_sq, rel=1e-10)
    assert np.allclose(f_mean.resid, f_sum.resid, atol=1e-8)


def test_an_unidentified_fit_is_refused_and_says_why():
    """The closed form is ``S_xy / S_xx``, which has no answer at ``S_xx == 0``.

    What to do instead is the caller's: FDID raises and tells the user to hold
    the slope at one, TBR takes the pseudoinverse because a constant regressor is
    its zero-cost case. Neither policy belongs in the mechanism.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = group_sums(y, np.full(5, 3.0))
    with pytest.raises(MlsynthEstimationError, match="constant"):
        fit_on_sums(s, y, np.full(5, 3.0))


def test_a_window_with_no_residual_degrees_of_freedom_is_refused():
    y, x = np.array([1.0, 2.0]), np.array([3.0, 5.0])
    with pytest.raises(MlsynthDataError):
        fit_two_group(y, x)


# --------------------------------------------------------------------------- #
# Layer 1: the covariance and the prediction variance
# --------------------------------------------------------------------------- #
def test_the_unscaled_covariance_inverts_the_gram():
    Y = market_matrix(seed=5)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    design = np.column_stack([np.ones(x.size), x])
    V = unscaled_cov(group_sums(y, x))
    assert np.allclose(V @ (design.T @ design), np.eye(2), atol=1e-10)


def test_the_prediction_variance_is_the_quadratic_form():
    """``xbar' (X'X)^-1 xbar`` with ``xbar = [1, x_bar]``, formed directly."""
    Y = market_matrix(seed=6)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    s = group_sums(y, x)
    design = np.column_stack([np.ones(x.size), x])
    inv = np.linalg.inv(design.T @ design)
    for x_bar in (float(x.mean()), float(x.mean() + 3.0 * x.std()), float(x[0])):
        v = np.array([1.0, x_bar])
        assert prediction_variance(s, x_bar) == pytest.approx(
            float(v @ inv @ v), rel=1e-9)


def test_the_prediction_variance_does_not_care_how_the_groups_were_aggregated():
    """It is a ratio of squares in the regressor, so the group size cancels."""
    Y = market_matrix(seed=7)
    ctl = [0, 1, 2, 3]
    y = aggregate_group(Y, [5, 6], how="sum")
    xs, xm = (aggregate_group(Y, ctl, how=h) for h in ("sum", "mean"))
    n_pre = 70
    got = [prediction_variance(group_sums(y[:n_pre], v[:n_pre]),
                               float(v[n_pre:].mean())) for v in (xs, xm)]
    assert got[0] == pytest.approx(got[1], rel=1e-9)


def test_the_prediction_variance_grows_as_the_test_window_drifts():
    """``(xbar_T - xbar_pre)^2 / S_xx`` is the term that widens the interval."""
    Y = market_matrix(seed=8)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    s = group_sums(y, x)
    at_mean = prediction_variance(s, float(x.mean()))
    drifted = prediction_variance(s, float(x.mean() + 20.0 * x.std()))
    assert drifted > 50.0 * at_mean


def test_an_unidentified_covariance_is_refused():
    y = np.arange(5.0)
    s = group_sums(y, np.full(5, 2.0))
    with pytest.raises(MlsynthEstimationError):
        unscaled_cov(s)
    with pytest.raises(MlsynthEstimationError):
        prediction_variance(s, 2.0)


# --------------------------------------------------------------------------- #
# Layer 3: the invariant the package exists for
# --------------------------------------------------------------------------- #
def test_the_fit_holds_its_digits_at_geo_aggregate_scale():
    """A control aggregate is a large level with a small spread.

    Summing markets gives a series near 44,000 varying by a few hundred, where
    ``X'X`` is conditioned around 1e13 and both the determinant form and a
    decomposition lose digits. Asserted against float128 on the centred formula.
    """
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = 44000.0 + 500.0 * rng.standard_normal(90)
        y = 12.0 + 0.8 * x + 40.0 * rng.standard_normal(90)
        fit = fit_two_group(y, x)

        xl, yl = x.astype(np.longdouble), y.astype(np.longdouble)
        dxl, dyl = xl - xl.mean(), yl - yl.mean()
        beta_exact = (dxl @ dyl) / (dxl @ dxl)
        assert abs(np.longdouble(fit.beta) - beta_exact) / abs(beta_exact) < 1e-14


def test_the_prediction_variance_takes_a_whole_horizon_at_once():
    """A caller projecting over a growing window wants the profile in one call."""
    Y = market_matrix(seed=9)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    s = group_sums(y[:70], x[:70])
    running = np.cumsum(x[70:]) / np.arange(1, x.size - 70 + 1)
    profile = prediction_variance(s, running)
    assert isinstance(profile, np.ndarray) and profile.shape == running.shape
    for i, value in enumerate(running):
        assert profile[i] == pytest.approx(prediction_variance(s, float(value)),
                                           rel=1e-15)
    assert isinstance(prediction_variance(s, float(running[0])), float)


# --------------------------------------------------------------------------- #
# Layer 4: the refusals, and the one convenience
# --------------------------------------------------------------------------- #
def test_a_panel_that_is_not_periods_by_units_is_refused():
    with pytest.raises(MlsynthDataError, match="periods by units"):
        aggregate_group(np.arange(10.0), [0])


def test_a_unit_index_outside_the_panel_is_refused():
    Y = market_matrix(n_units=4)
    with pytest.raises(MlsynthDataError, match="outside"):
        aggregate_group(Y, [0, 9])


def test_an_empty_fitting_window_is_refused():
    with pytest.raises(MlsynthDataError, match="empty"):
        group_sums(np.array([]), np.array([]))


def test_the_fit_predicts_at_regressor_values_it_was_not_fitted_on():
    """The counterfactual every caller of this wants is the line off the window."""
    Y = market_matrix(seed=10)
    x, y = aggregate_group(Y, [0, 1]), aggregate_group(Y, [4, 5])
    n_pre = 70
    fit = fit_two_group(y[:n_pre], x[:n_pre])
    projected = fit.predict(x[n_pre:])
    assert projected.shape == x[n_pre:].shape
    assert np.allclose(projected, fit.alpha + fit.beta * x[n_pre:], atol=1e-9)
    assert fit.predict(float(x[0])) == pytest.approx(fit.alpha + fit.beta * x[0])
