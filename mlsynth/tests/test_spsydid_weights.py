"""SpSyDiD's two SDID weight QPs, against the programs they solve.

``spsydid_helpers.weights`` fits both SDID weight vectors with a free intercept:

.. math::

   \\min_{\\beta_0,\\; \\lambda \\in \\Delta}
       \\lVert \\beta_0 \\mathbf{1} + Y_{0,pre}^\\top \\lambda
              - \\bar y_{0,post} \\rVert_2^2

.. math::

   \\min_{\\omega_0,\\; \\omega \\in \\Delta}
       \\lVert \\omega_0 \\mathbf{1} + Y_{0,pre}\\,\\omega
              - \\bar y_{1,pre} \\rVert_2^2
       + T_0 \\zeta^2 \\lVert \\omega \\rVert_2^2

Two reshapings, both exact. The intercept is unconstrained, so at the optimum it
equals the mean residual, and substituting that back leaves the same program on
centred data -- profiling it out is a projection, not an approximation. The
intercept is then recovered from the weights, which is why these tests assert the
recovered value against its own first-order condition and not only against
cvxpy. The ridge on ``omega`` becomes design rows carrying no target (Zou and
Hastie 2005, Lemma 1).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.spsydid_helpers.weights import (
    compute_regularization,
    fit_time_weights,
    fit_unit_weights,
)

TOL = 1e-6
SHAPES = [(12, 5), (20, 8), (8, 20), (6, 30)]


def _time_oracle(Y0, post):
    T0 = Y0.shape[0]
    b0, lam = cp.Variable(), cp.Variable(T0, nonneg=True)
    cp.Problem(
        cp.Minimize(cp.sum_squares(b0 + (lam @ Y0) - post)), [cp.sum(lam) == 1]
    ).solve(solver=cp.CLARABEL)
    return float(b0.value), np.asarray(lam.value, dtype=float)


def _unit_oracle(Y0, treated, zeta):
    T0, J = Y0.shape
    b0, om = cp.Variable(), cp.Variable(J, nonneg=True)
    pen = T0 * (float(zeta) ** 2) * cp.sum_squares(om)
    cp.Problem(
        cp.Minimize(cp.sum_squares(b0 + Y0 @ om - treated) + pen), [cp.sum(om) == 1]
    ).solve(solver=cp.CLARABEL)
    return float(b0.value), np.asarray(om.value, dtype=float)


def _time_obj(Y0, post, b0, lam):
    r = b0 + lam @ Y0 - post
    return float(r @ r)


def _unit_obj(Y0, treated, zeta, b0, om):
    T0 = Y0.shape[0]
    r = b0 + Y0 @ om - treated
    return float(r @ r + T0 * float(zeta) ** 2 * (om @ om))


def _panel(rng, T0, J):
    return rng.normal(size=(T0, J)), rng.normal(size=J), rng.normal(size=T0)


# ----------------------------------------------------------- time weights
def test_time_weights_return_an_intercept_and_one_weight_per_period():
    rng = np.random.default_rng(0)
    Y0, post, _ = _panel(rng, 12, 5)
    b0, lam = fit_time_weights(Y0, post)
    assert isinstance(b0, float)
    assert lam.shape == (12,)
    assert np.all(np.isfinite(lam))


@pytest.mark.parametrize("T0,J", SHAPES)
def test_time_weights_are_on_the_simplex(T0, J):
    rng = np.random.default_rng(T0 * 100 + J)
    Y0, post, _ = _panel(rng, T0, J)
    _b0, lam = fit_time_weights(Y0, post)
    assert lam.min() >= 0.0
    assert abs(lam.sum() - 1.0) < TOL


@pytest.mark.parametrize("T0,J", SHAPES)
def test_time_weights_solve_the_intercept_program(T0, J):
    rng = np.random.default_rng(1000 + T0 * 100 + J)
    Y0, post, _ = _panel(rng, T0, J)
    b0, lam = fit_time_weights(Y0, post)
    ob0, olam = _time_oracle(Y0, post)
    assert _time_obj(Y0, post, b0, lam) <= _time_obj(Y0, post, ob0, olam) + 1e-6


@pytest.mark.parametrize("T0,J", SHAPES)
def test_the_time_intercept_is_the_mean_residual(T0, J):
    """Its own first-order condition: unconstrained, so it centres the fit."""
    rng = np.random.default_rng(2000 + T0 * 100 + J)
    Y0, post, _ = _panel(rng, T0, J)
    b0, lam = fit_time_weights(Y0, post)
    assert b0 == pytest.approx(float(np.mean(post - lam @ Y0)), abs=1e-8)


# ----------------------------------------------------------- unit weights
@pytest.mark.parametrize("T0,J", SHAPES)
def test_unit_weights_are_on_the_simplex(T0, J):
    rng = np.random.default_rng(3000 + T0 * 100 + J)
    Y0, _post, treated = _panel(rng, T0, J)
    _b0, om = fit_unit_weights(Y0, treated, zeta=0.7)
    assert om.min() >= 0.0
    assert abs(om.sum() - 1.0) < TOL


@pytest.mark.parametrize("T0,J", SHAPES)
@pytest.mark.parametrize("zeta", [0.0, 0.3, 2.0])
def test_unit_weights_solve_the_intercept_and_ridge_program(T0, J, zeta):
    rng = np.random.default_rng(4000 + T0 * 100 + J)
    Y0, _post, treated = _panel(rng, T0, J)
    b0, om = fit_unit_weights(Y0, treated, zeta=zeta)
    ob0, oom = _unit_oracle(Y0, treated, zeta)
    assert _unit_obj(Y0, treated, zeta, b0, om) <= _unit_obj(
        Y0, treated, zeta, ob0, oom
    ) + 1e-6


@pytest.mark.parametrize("T0,J", SHAPES)
def test_the_unit_intercept_is_the_mean_residual(T0, J):
    rng = np.random.default_rng(5000 + T0 * 100 + J)
    Y0, _post, treated = _panel(rng, T0, J)
    b0, om = fit_unit_weights(Y0, treated, zeta=0.7)
    assert b0 == pytest.approx(float(np.mean(treated - Y0 @ om)), abs=1e-8)


def test_a_heavier_ridge_spreads_the_unit_weights():
    rng = np.random.default_rng(11)
    Y0, _post, treated = _panel(rng, 15, 6)
    light = fit_unit_weights(Y0, treated, zeta=1e-6)[1]
    heavy = fit_unit_weights(Y0, treated, zeta=1e3)[1]
    assert float(heavy @ heavy) < float(light @ light)
    assert np.allclose(heavy, np.full(6, 1 / 6), atol=1e-3)


def test_a_zero_ridge_is_the_plain_intercept_program():
    """``zeta = 0`` is reachable only with no post-periods, and is allowed."""
    rng = np.random.default_rng(13)
    Y0, _post, treated = _panel(rng, 10, 4)
    b0, om = fit_unit_weights(Y0, treated, zeta=0.0)
    ob0, oom = _unit_oracle(Y0, treated, 0.0)
    assert _unit_obj(Y0, treated, 0.0, b0, om) <= _unit_obj(
        Y0, treated, 0.0, ob0, oom
    ) + 1e-6
    assert abs(om.sum() - 1.0) < TOL


def test_compute_regularization_is_zero_only_without_post_periods():
    rng = np.random.default_rng(17)
    Y0 = rng.normal(size=(10, 4))
    assert compute_regularization(Y0, num_post_periods=0) == 0.0
    assert compute_regularization(Y0, num_post_periods=4) > 0.0


# ------------------------------------------ shifts, scales and relabelling
@pytest.mark.parametrize("shift", [-5.0, 3.5])
def test_shifting_the_level_moves_only_the_intercept(shift):
    """What the intercept absorbs, the weights should not see."""
    rng = np.random.default_rng(19)
    Y0, _post, treated = _panel(rng, 14, 5)
    b0, om = fit_unit_weights(Y0, treated, zeta=0.5)
    b0s, oms = fit_unit_weights(Y0, treated + shift, zeta=0.5)
    assert np.allclose(om, oms, atol=1e-9)
    assert b0s == pytest.approx(b0 + shift, abs=1e-7)


@pytest.mark.parametrize("c", [0.5, 4.0])
def test_scaling_the_panel_and_the_ridge_together_leaves_the_weights_alone(c):
    """Exact under the active set; the cone form drifts here at 1e-08."""
    rng = np.random.default_rng(23)
    Y0, _post, treated = _panel(rng, 14, 5)
    base = fit_unit_weights(Y0, treated, zeta=1.0)[1]
    moved = fit_unit_weights(c * Y0, c * treated, zeta=c)[1]
    assert np.allclose(base, moved, atol=1e-12)


# ------------------------------------------------------------- edge cases
def test_a_single_donor_takes_all_the_unit_weight():
    rng = np.random.default_rng(29)
    _b0, om = fit_unit_weights(rng.normal(size=(7, 1)), rng.normal(size=7), zeta=0.4)
    assert np.allclose(om, [1.0], atol=TOL)


def test_a_single_pre_period_takes_all_the_time_weight():
    rng = np.random.default_rng(31)
    _b0, lam = fit_time_weights(rng.normal(size=(1, 5)), rng.normal(size=5))
    assert np.allclose(lam, [1.0], atol=TOL)


def test_collinear_donors_do_not_break_the_unit_solve():
    rng = np.random.default_rng(37)
    a = rng.normal(size=(9, 1))
    Y0 = np.hstack([a, 2.0 * a, -a])
    _b0, om = fit_unit_weights(Y0, rng.normal(size=9), zeta=0.2)
    assert np.all(np.isfinite(om)) and abs(om.sum() - 1.0) < TOL


def test_exact_zeros_are_attainable():
    """The active set reaches the boundary where a cone solver stops near it.

    This is the difference that made the WLS rank check meaningful: CLARABEL
    returned minima of order 1e-11 on these panels and never an exact zero, so a
    period carrying no weight still contributed a row of magnitude 3e-06 after
    the square root.
    """
    rng = np.random.default_rng(41)
    Y0, post, _ = _panel(rng, 14, 6)
    _b0, lam = fit_time_weights(Y0, post)
    assert (lam == 0.0).any()
    assert lam.min() == 0.0


# ---------------------------------------------------------------- failures
def test_the_existing_guards_still_refuse_malformed_input():
    with pytest.raises(MlsynthDataError):
        fit_time_weights(np.ones(5), np.ones(5))
    with pytest.raises(MlsynthDataError):
        fit_time_weights(np.ones((5, 3)), np.ones((5, 3)))
    with pytest.raises(MlsynthDataError):
        fit_time_weights(np.ones((5, 3)), np.ones(4))
    with pytest.raises(MlsynthDataError):
        fit_unit_weights(np.ones((5, 3)), np.ones(5), zeta=-1.0)
    with pytest.raises(MlsynthDataError):
        fit_unit_weights(np.ones((5, 0)), np.ones(5), zeta=1.0)
    with pytest.raises(MlsynthDataError):
        fit_unit_weights(np.ones((5, 3)), np.ones(4), zeta=1.0)
