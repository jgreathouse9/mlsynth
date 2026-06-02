"""Unit tests for the self-contained bilevel SCM solver (Malo et al. 2024)."""

import numpy as np
import pytest

from mlsynth.utils.fscm_helpers.bilevel import (
    BilevelProblem,
    BilevelSolution,
    project_simplex,
    simplex_lstsq,
    solve_bilevel,
    lower_level_weights,
)
from mlsynth.utils.fscm_helpers.bilevel.simplex import mspe


# --------------------------------------------------------------------------- #
# project_simplex
# --------------------------------------------------------------------------- #
def test_project_simplex_basic():
    w = project_simplex(np.array([0.2, 0.5, 0.3]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    # Already on the simplex -> unchanged.
    np.testing.assert_allclose(w, [0.2, 0.5, 0.3], atol=1e-9)


def test_project_simplex_negative_and_outside():
    w = project_simplex(np.array([3.0, -1.0, 0.0]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert w[0] == pytest.approx(1.0)  # dominant coordinate absorbs the mass


def test_project_simplex_singleton():
    assert project_simplex(np.array([5.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# simplex_lstsq
# --------------------------------------------------------------------------- #
def test_simplex_lstsq_matches_known_optimum():
    # Target is an exact convex combination of two columns -> recoverable.
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.25, 0.75, 1.0])
    w = simplex_lstsq(A, b)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(w, [0.25, 0.75], atol=1e-4)


def test_simplex_lstsq_optimality_vs_grid():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(15, 4))
    b = rng.normal(size=15)
    w = simplex_lstsq(A, b)
    obj = np.sum((A @ w - b) ** 2)
    # No simplex vertex or random simplex point should beat it.
    for _ in range(200):
        wr = rng.dirichlet(np.ones(4))
        assert obj <= np.sum((A @ wr - b) ** 2) + 1e-6


def test_simplex_lstsq_singleton():
    assert simplex_lstsq(np.array([[2.0], [3.0]]), np.array([1.0, 1.0])) == pytest.approx([1.0])


# --------------------------------------------------------------------------- #
# solve_bilevel
# --------------------------------------------------------------------------- #
def test_bilevel_unconstrained_certificate():
    # Treated outcome is an exact convex combo of donors, and a predictor is
    # matched by that same combination -> Stage 1 certifies optimality.
    Y0 = np.array([[1.0, 3.0], [2.0, 0.0], [0.0, 4.0]])
    w_true = np.array([0.5, 0.5])
    y1 = Y0 @ w_true
    X0 = np.array([[10.0, 20.0]])      # predictor matched by w_true -> X1 = 15
    X1 = X0 @ w_true
    sol = solve_bilevel(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0))
    assert isinstance(sol, BilevelSolution)
    assert sol.stage == "unconstrained"
    assert sol.lower_bound == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(sol.W, w_true, atol=1e-3)


def test_bilevel_upper_loss_at_least_lower_bound():
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(12, 6))
    y1 = rng.normal(size=12)
    X0 = rng.normal(size=(3, 6))
    X1 = rng.normal(size=3)
    sol = solve_bilevel(BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0))
    assert sol.upper_loss >= sol.lower_bound - 1e-6
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.V.sum() == pytest.approx(1.0, abs=1e-6)


def test_lower_level_weights_fixed_V_on_simplex():
    rng = np.random.default_rng(7)
    Y0 = rng.normal(size=(10, 5))
    y1 = rng.normal(size=10)
    X0 = rng.normal(size=(2, 5))
    X1 = rng.normal(size=2)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    w = lower_level_weights(prob, np.array([1.0, 0.0]))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-9)


def test_bilevel_deterministic():
    rng = np.random.default_rng(9)
    Y0 = rng.normal(size=(14, 7)); y1 = rng.normal(size=14)
    X0 = rng.normal(size=(4, 7)); X1 = rng.normal(size=4)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s1, s2 = solve_bilevel(prob), solve_bilevel(prob)
    np.testing.assert_allclose(s1.V, s2.V)
    np.testing.assert_allclose(s1.W, s2.W)


# --------------------------------------------------------------------------- #
# method dispatch: malo (default) vs mscmt
# --------------------------------------------------------------------------- #
def test_unknown_method_raises():
    prob = BilevelProblem(
        y1_pre=np.zeros(3), Y0_pre=np.eye(3)[:, :2],
        X1=np.zeros(1), X0=np.zeros((1, 2)),
    )
    with pytest.raises(ValueError, match="malo|mscmt"):
        solve_bilevel(prob, method="not-a-method")


def test_mscmt_backend_on_simplex_and_bounded():
    rng = np.random.default_rng(5)
    Y0 = rng.normal(size=(12, 6)); y1 = rng.normal(size=12)
    X0 = rng.normal(size=(3, 6)); X1 = rng.normal(size=3)
    sol = solve_bilevel(
        BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0),
        method="mscmt", maxiter=50, seed=0,
    )
    assert isinstance(sol, BilevelSolution)
    assert sol.stage in ("mscmt", "mscmt-feasible")
    assert sol.W.sum() == pytest.approx(1.0, abs=1e-6)
    assert sol.V.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(sol.W >= -1e-9)
    assert sol.upper_loss >= sol.lower_bound - 1e-6


def test_mscmt_matches_malo_on_feasible_certificate():
    # When the unconstrained outcome optimum is predictor-feasible, both
    # backends return the exact global solution and must agree.
    Y0 = np.array([[1.0, 3.0], [2.0, 0.0], [0.0, 4.0]])
    w_true = np.array([0.5, 0.5])
    y1 = Y0 @ w_true
    X0 = np.array([[10.0, 20.0]])
    X1 = X0 @ w_true
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s_malo = solve_bilevel(prob, method="malo")
    s_msc = solve_bilevel(prob, method="mscmt")
    assert s_malo.stage == "unconstrained"
    assert s_msc.stage == "mscmt-feasible"
    np.testing.assert_allclose(s_malo.W, s_msc.W, atol=1e-3)


def test_mscmt_deterministic_with_seed():
    rng = np.random.default_rng(11)
    Y0 = rng.normal(size=(14, 7)); y1 = rng.normal(size=14)
    X0 = rng.normal(size=(4, 7)); X1 = rng.normal(size=4)
    prob = BilevelProblem(y1_pre=y1, Y0_pre=Y0, X1=X1, X0=X0)
    s1 = solve_bilevel(prob, method="mscmt", seed=3, maxiter=40)
    s2 = solve_bilevel(prob, method="mscmt", seed=3, maxiter=40)
    np.testing.assert_allclose(s1.W, s2.W)
    np.testing.assert_allclose(s1.V, s2.V)
