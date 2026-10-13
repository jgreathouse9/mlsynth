"""CSCM's SCM warm-start, against the metric-weighted program it solves.

``cscm_helpers.engine.solve_scm_simplex`` fits

.. math::

   \\min_{w \\geq 0,\\, \\mathbf{1}^\\top w = 1}
       (X_1 - X_0 w)^\\top V (X_1 - X_0 w)

for diagonal ``V``. A diagonal metric on a residual is a row scaling of the
design: with :math:`d = \\sqrt{\\mathrm{diag}(V)}`,

.. math::

   (X_1 - X_0 w)^\\top V (X_1 - X_0 w) = \\| d \\odot X_1 - (d \\odot X_0) w \\|_2^2

so the program is plain simplex least squares on the scaled panel. The
neighbouring ``solve_cscm_penalized`` already builds its design this way, so
these tests hold both to the same identity.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.cscm_helpers.engine import solve_scm_simplex
from mlsynth.utils.solvers.minnorm import simplex_point_is_optimal

TOL = 1e-6


def _cvxpy_oracle(X1: np.ndarray, X0: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The quadratic-form program, solved independently of the module."""
    W = cp.Variable(X0.shape[1])
    r = X1 - X0 @ W
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(r, cp.psd_wrap(np.diag(V)))),
        [W >= 0, cp.sum(W) == 1],
    )
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=100000)
    assert W.value is not None, prob.status
    return np.asarray(W.value, dtype=float)


def _weighted_sse(X1, X0, V, w) -> float:
    r = X1 - X0 @ w
    return float(r @ (np.diag(V) @ r))


def _panel(rng, F: int, J: int):
    X0 = rng.normal(size=(F, J))
    X1 = rng.normal(size=F)
    V = rng.uniform(0.1, 2.0, size=F)
    return X1, X0, V / V.sum()


# ------------------------------------------------------------------ smoke
def test_returns_one_finite_weight_per_donor():
    rng = np.random.default_rng(0)
    X1, X0, V = _panel(rng, 6, 4)
    w = solve_scm_simplex(X1, X0, V)
    assert w.shape == (4,)
    assert np.all(np.isfinite(w))


# -------------------------------------------------------------- invariants
@pytest.mark.parametrize("F,J", [(5, 3), (8, 5), (4, 4), (10, 2)])
def test_the_answer_is_on_the_simplex(F, J):
    rng = np.random.default_rng(F * 100 + J)
    X1, X0, V = _panel(rng, F, J)
    w = solve_scm_simplex(X1, X0, V)
    assert w.min() >= 0.0
    assert abs(w.sum() - 1.0) < TOL


@pytest.mark.parametrize("seed", range(6))
def test_it_solves_the_metric_weighted_program(seed):
    rng = np.random.default_rng(seed)
    X1, X0, V = _panel(rng, 8, 4)
    got, want = solve_scm_simplex(X1, X0, V), _cvxpy_oracle(X1, X0, V)
    assert _weighted_sse(X1, X0, V, got) == pytest.approx(
        _weighted_sse(X1, X0, V, want), abs=1e-6
    )


@pytest.mark.parametrize("seed", range(5))
def test_the_metric_is_exactly_a_row_scaling(seed):
    """The identity the reshaping rests on, asserted on the returned point."""
    rng = np.random.default_rng(100 + seed)
    X1, X0, V = _panel(rng, 9, 4)
    w = solve_scm_simplex(X1, X0, V)
    d = np.sqrt(V)
    scaled = d * X1 - (d[:, None] * X0) @ w
    assert float(scaled @ scaled) == pytest.approx(_weighted_sse(X1, X0, V, w), rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_the_scaled_point_is_certified_optimal(seed):
    rng = np.random.default_rng(200 + seed)
    X1, X0, V = _panel(rng, 9, 5)
    d = np.sqrt(V)
    w = solve_scm_simplex(X1, X0, V)
    assert simplex_point_is_optimal(d[:, None] * X0, d * X1, w)


def test_a_constant_metric_is_the_unweighted_program():
    """A flat V is a positive scalar on the objective, so it moves nothing."""
    rng = np.random.default_rng(21)
    X0 = rng.normal(size=(8, 4))
    X1 = rng.normal(size=8)
    flat = solve_scm_simplex(X1, X0, np.full(8, 1.0 / 8.0))
    ones = solve_scm_simplex(X1, X0, np.ones(8))
    assert np.allclose(flat, ones, atol=1e-9)


@pytest.mark.parametrize("c", [0.25, 4.0, 50.0])
def test_scaling_the_panel_leaves_the_weights_alone(c):
    rng = np.random.default_rng(23)
    X1, X0, V = _panel(rng, 9, 4)
    assert np.allclose(
        solve_scm_simplex(X1, X0, V), solve_scm_simplex(c * X1, c * X0, V), atol=1e-12
    )


def test_a_zero_metric_entry_drops_that_row():
    """Weight 0 on a feature makes it irrelevant to the fit."""
    rng = np.random.default_rng(29)
    X0 = rng.normal(size=(6, 3))
    X1 = rng.normal(size=6)
    V = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    kept = solve_scm_simplex(X1, X0, V)
    dropped = solve_scm_simplex(X1[:5], X0[:5], V[:5])
    assert np.allclose(kept, dropped, atol=1e-9)


# ------------------------------------------------------------- edge cases
def test_a_single_donor_takes_all_the_weight():
    rng = np.random.default_rng(31)
    w = solve_scm_simplex(rng.normal(size=5), rng.normal(size=(5, 1)), np.ones(5))
    assert np.allclose(w, [1.0], atol=TOL)


def test_one_feature_still_solves():
    w = solve_scm_simplex(np.array([2.0]), np.array([[1.0, 3.0]]), np.array([1.0]))
    assert abs(w.sum() - 1.0) < TOL
    assert w.min() >= 0.0


def test_collinear_donors_do_not_break_the_solve():
    rng = np.random.default_rng(37)
    a = rng.normal(size=(7, 1))
    X0 = np.hstack([a, 2.0 * a, -a])
    w = solve_scm_simplex(rng.normal(size=7), X0, np.ones(7))
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < TOL


def test_a_negative_metric_entry_is_refused():
    """A negative feature weight makes the objective non-convex, so it raises.

    No V producer in the library emits one -- ``estimate_V_poisson`` returns
    ``|coef| / sum|coef|`` and ``uniform_V`` returns ``1/F`` -- so this is the
    defensive path, and refusing is what it did before: the cvxpy program
    wrapped its metric in ``cp.psd_wrap``, which suppresses cvxpy's own
    semidefiniteness check, and OSQP then failed with the bare code ``4``. The
    guard keeps the refusal and says which entries caused it.

    The neighbouring ``solve_cscm_penalized`` clamps the same metric with
    ``np.maximum(Vdiag, 0.0)`` instead. That disagreement is left alone here;
    changing it would alter a second function's behaviour on malformed input.
    """
    rng = np.random.default_rng(41)
    X0 = rng.normal(size=(6, 3))
    X1 = rng.normal(size=6)
    V = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -5.0])
    with pytest.raises(MlsynthEstimationError, match="negative"):
        solve_scm_simplex(X1, X0, V)


def test_a_metric_of_all_zeros_is_refused():
    """Every feature weighted zero leaves no objective to minimise."""
    rng = np.random.default_rng(43)
    with pytest.raises(MlsynthEstimationError):
        solve_scm_simplex(rng.normal(size=5), rng.normal(size=(5, 3)), np.zeros(5))


# ---------------------------------------------------------------- failures
def test_a_feature_count_mismatch_is_refused():
    with pytest.raises((MlsynthEstimationError, ValueError)):
        solve_scm_simplex(np.ones(5), np.ones((4, 2)), np.ones(5))


def test_an_empty_donor_pool_raises_a_translated_error():
    with pytest.raises(MlsynthEstimationError):
        solve_scm_simplex(np.ones(5), np.ones((5, 0)), np.ones(5))
