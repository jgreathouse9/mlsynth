"""HSC's donor QP, against the metric-and-ridge program it solves.

``hsc_helpers.formulation.fit_donor_weights`` fits

.. math::

   \\min_{\\omega \\geq 0,\\, \\mathbf{1}^\\top \\omega = 1}
       (Y - X\\omega)^\\top W (Y - X\\omega) + c \\lVert \\omega \\rVert^2

for a symmetric PSD metric ``W`` and a ridge ``c > 0``. Written out, that is
``omega' H omega - 2 f' omega`` up to a constant, with ``H = X'WX + cI`` and
``f = X'WY`` -- the Gram form.

The reshaping does not form ``H``. Factor the metric as ``W = L'L`` and stack
the ridge as extra design rows carrying no target (Zou and Hastie 2005,
Lemma 1):

.. math::

   B = \\begin{bmatrix} LX \\\\ \\sqrt{c}\\,I \\end{bmatrix}, \\qquad
   A = \\begin{bmatrix} LY \\\\ 0 \\end{bmatrix}

Then ``B'B = X'WX + cI = H`` and ``B'A = X'WY = f`` exactly, so
``||A - B omega||^2`` is the same program. Two tests below assert those two
identities directly, which is the algebra and holds whatever solver runs.

Going through ``W`` and not ``H`` matters for accuracy: ``H`` is the Gram of the
design, so forming it squares the design's condition number, while a symmetric
eigendecomposition of ``W`` is backward stable. ``smoother_and_metric`` supplies
``W`` in three forms across its ``rho`` branches, and the tests exercise all of
them.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.hsc_helpers.formulation import (
    fit_donor_weights,
    smoother_and_metric,
)
from mlsynth.utils.solvers.minnorm import simplex_point_is_optimal

TOL = 1e-6
RHOS = (0.0, 0.3, 0.7, 1.0)


def _metric_factor(W: np.ndarray) -> np.ndarray:
    """``L`` with ``L'L = W``, from a symmetric eigendecomposition."""
    W = 0.5 * (W + W.T)
    d, U = np.linalg.eigh(W)
    return np.sqrt(np.maximum(d, 0.0))[:, None] * U.T


def _ridge_coef(X, W, ridge=1e-6, ridge_abs=None) -> float:
    N = X.shape[1]
    return (
        float(ridge_abs)
        if ridge_abs is not None
        else ridge * (float(np.trace(X.T @ W @ X)) / max(N, 1))
    )


def _design(X, Y, W, c):
    N = X.shape[1]
    L = _metric_factor(W)
    B = np.vstack([L @ X, np.sqrt(c) * np.eye(N)])
    A = np.concatenate([L @ Y, np.zeros(N)])
    return B, A


def _oracle(X, Y, W, c) -> np.ndarray:
    """The Gram-form program, solved independently of the module."""
    N = X.shape[1]
    H = 0.5 * (X.T @ W @ X + (X.T @ W @ X).T) + c * np.eye(N)
    f = X.T @ W @ Y
    om = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(om, cp.psd_wrap(H)) - 2.0 * f @ om),
        [om >= 0, cp.sum(om) == 1],
    )
    prob.solve(solver=cp.CLARABEL)
    assert om.value is not None, prob.status
    return np.clip(np.asarray(om.value, dtype=float), 0.0, None)


def _objective(X, Y, W, c, om) -> float:
    r = Y - X @ om
    return float(r @ W @ r + c * (om @ om))


def _panel(rng, T=24, N=8, q=1, rho=0.3):
    X = rng.normal(size=(T, N))
    Y = rng.normal(size=T)
    _S, W = smoother_and_metric(T, q, rho)
    return X, Y, W


# ------------------------------------------------- the reshaping is an identity
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("rho", RHOS)
def test_the_stacked_design_reproduces_the_gram(q, rho):
    """``B'B = X'WX + cI``, which is the first half of the reshaping."""
    rng = np.random.default_rng(q * 10 + int(rho * 10))
    X, Y, W = _panel(rng, q=q, rho=rho)
    c = _ridge_coef(X, W)
    B, _A = _design(X, Y, W, c)
    H = X.T @ W @ X + c * np.eye(X.shape[1])
    assert np.allclose(B.T @ B, H, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("rho", RHOS)
def test_the_stacked_target_reproduces_the_linear_term(q, rho):
    """``B'A = X'WY``, which is the second half."""
    rng = np.random.default_rng(100 + q * 10 + int(rho * 10))
    X, Y, W = _panel(rng, q=q, rho=rho)
    c = _ridge_coef(X, W)
    B, A = _design(X, Y, W, c)
    assert np.allclose(B.T @ A, X.T @ W @ Y, rtol=1e-10, atol=1e-10)


def test_the_ridge_rows_carry_no_target():
    """The augmentation is a penalty because its target is zero."""
    rng = np.random.default_rng(3)
    X, Y, W = _panel(rng)
    c = _ridge_coef(X, W)
    B, A = _design(X, Y, W, c)
    N = X.shape[1]
    assert np.allclose(A[-N:], 0.0)
    assert np.allclose(B[-N:], np.sqrt(c) * np.eye(N))


# ------------------------------------------------------------------ smoke
def test_returns_one_finite_weight_per_donor():
    rng = np.random.default_rng(0)
    X, Y, W = _panel(rng)
    om = fit_donor_weights(X, Y, W)
    assert om.shape == (8,)
    assert np.all(np.isfinite(om))


# -------------------------------------------------------------- invariants
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("rho", RHOS)
def test_the_answer_is_on_the_simplex(q, rho):
    rng = np.random.default_rng(q * 7 + int(rho * 11))
    X, Y, W = _panel(rng, q=q, rho=rho)
    om = fit_donor_weights(X, Y, W)
    assert om.min() >= 0.0
    assert abs(om.sum() - 1.0) < TOL


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("rho", RHOS)
def test_it_solves_the_metric_and_ridge_program(q, rho):
    rng = np.random.default_rng(200 + q * 10 + int(rho * 10))
    X, Y, W = _panel(rng, q=q, rho=rho)
    c = _ridge_coef(X, W)
    got, want = fit_donor_weights(X, Y, W), _oracle(X, Y, W, c)
    assert _objective(X, Y, W, c, got) <= _objective(X, Y, W, c, want) + 1e-6


@pytest.mark.parametrize("rho", RHOS)
def test_the_stacked_point_is_certified_optimal(rho):
    rng = np.random.default_rng(300 + int(rho * 10))
    X, Y, W = _panel(rng, rho=rho)
    c = _ridge_coef(X, W)
    B, A = _design(X, Y, W, c)
    assert simplex_point_is_optimal(B, A, fit_donor_weights(X, Y, W))


def test_an_identity_metric_is_plain_least_squares_with_a_ridge():
    rng = np.random.default_rng(5)
    T, N = 20, 6
    X, Y = rng.normal(size=(T, N)), rng.normal(size=T)
    W = np.eye(T)
    c = _ridge_coef(X, W)
    got = fit_donor_weights(X, Y, W)
    assert _objective(X, Y, W, c, got) <= _objective(X, Y, W, c, _oracle(X, Y, W, c)) + 1e-6


def test_the_projection_branch_metric_is_its_own_factor():
    """At ``rho = 1`` the metric is ``I - P0``, idempotent and symmetric."""
    _S, W = smoother_and_metric(18, 2, 1.0)
    assert np.allclose(W, W.T, atol=1e-12)
    assert np.allclose(W @ W, W, atol=1e-10)
    assert np.allclose(W.T @ W, W, atol=1e-10)


@pytest.mark.parametrize("c", [0.5, 3.0, 20.0])
def test_scaling_the_panel_and_the_ridge_together_leaves_the_weights_alone(c):
    """``ridge_abs`` has to scale with the squared data for this to hold."""
    rng = np.random.default_rng(23)
    X, Y, W = _panel(rng)
    base = fit_donor_weights(X, Y, W, ridge_abs=1.0)
    moved = fit_donor_weights(c * X, c * Y, W, ridge_abs=c**2)
    assert np.allclose(base, moved, atol=1e-12)


# ------------------------------------------------------------- edge cases
def test_a_single_donor_takes_all_the_weight():
    rng = np.random.default_rng(31)
    _S, W = smoother_and_metric(9, 1, 0.3)
    om = fit_donor_weights(rng.normal(size=(9, 1)), rng.normal(size=9), W)
    assert np.allclose(om, [1.0], atol=TOL)


def test_a_singular_metric_still_solves():
    """At ``rho = 0`` the metric is a roughness matrix, which is rank deficient.

    A first-difference roughness matrix annihilates constants, so ``W`` is
    singular and the fit term alone does not pin the weights; the ridge does.
    """
    rng = np.random.default_rng(37)
    T, N = 16, 5
    _S, W = smoother_and_metric(T, 1, 0.0)
    assert np.linalg.matrix_rank(W) < T
    om = fit_donor_weights(rng.normal(size=(T, N)), rng.normal(size=T), W)
    assert np.all(np.isfinite(om))
    assert om.min() >= 0.0
    assert abs(om.sum() - 1.0) < TOL


def test_a_larger_ridge_pulls_the_weights_toward_uniform():
    """The ridge is the only term that prefers spread, so it should spread."""
    rng = np.random.default_rng(41)
    X, Y, W = _panel(rng, T=20, N=6)
    light = fit_donor_weights(X, Y, W, ridge_abs=1e-8)
    heavy = fit_donor_weights(X, Y, W, ridge_abs=1e4)
    assert float(heavy @ heavy) < float(light @ light)
    assert np.allclose(heavy, np.full(6, 1 / 6), atol=1e-3)


# ---------------------------------------------------------------- failures
def test_a_metric_of_the_wrong_size_is_refused():
    rng = np.random.default_rng(43)
    with pytest.raises((MlsynthEstimationError, ValueError)):
        fit_donor_weights(rng.normal(size=(10, 3)), rng.normal(size=10), np.eye(9))


def test_a_target_of_the_wrong_length_is_refused():
    rng = np.random.default_rng(47)
    with pytest.raises((MlsynthEstimationError, ValueError)):
        fit_donor_weights(rng.normal(size=(10, 3)), rng.normal(size=9), np.eye(10))


def test_an_empty_donor_pool_raises_a_translated_error():
    with pytest.raises(MlsynthEstimationError):
        fit_donor_weights(np.ones((6, 0)), np.ones(6), np.eye(6))
