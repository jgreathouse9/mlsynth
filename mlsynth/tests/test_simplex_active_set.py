"""Correctness contract for the pure-NumPy active-set simplex QP (PR #58).

Test-first (per the TDD convention in ``agents/agents_tests.md``): this harness
is written *before* the solver, against a stub that raises
``NotImplementedError``, so every test below is RED until the active-set method
is implemented to satisfy it.

The contract has three pillars, all solver-implementation-agnostic so the same
suite later validates a numba / warm-start variant unchanged:

1. **cvxpy parity** -- the fitted value ``B @ w`` (always unique) and the
   objective match the exact reference, across every problem regime.
2. **KKT certificate** -- the returned ``w`` satisfies reduced-gradient
   optimality, proven *without* trusting any other solver. The certifier is
   cross-validated by also asserting it on cvxpy's own solution.
3. **Fuzz** -- hundreds of seeded random instances spanning the regime grid.

Performance tests ("very very fast": pivot bounds, warm-start ratios) live in a
separate module added once the solver is green -- asserting speed on a
``NotImplementedError`` stub is meaningless.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from mlsynth.utils.bilevel.active_set import solve_simplex_qp


# --------------------------------------------------------------------------- #
# Test infrastructure: the exact reference oracle and the KKT certifier
# --------------------------------------------------------------------------- #
def _reference(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Exact simplex-constrained least squares via cvxpy (the oracle)."""
    J = B.shape[1]
    w = cp.Variable(J)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(B @ w - A)), [w >= 0, cp.sum(w) == 1])
    prob.solve()
    return np.asarray(w.value, dtype=float).ravel()


def _objective(B, A, w):
    r = B @ np.asarray(w, float).ravel() - A
    return float(r @ r)


def assert_feasible(w, J, tol=1e-7):
    w = np.asarray(w, float).ravel()
    assert w.shape == (J,), f"shape {w.shape} != ({J},)"
    assert np.all(np.isfinite(w)), "non-finite weights"
    assert w.min() >= -tol, f"negative weight {w.min():.2e}"
    assert abs(w.sum() - 1.0) <= tol, f"weights sum to {w.sum():.6f}"


def assert_kkt_optimal(B, A, w, tol=1e-6):
    """Reduced-gradient optimality for ``min 1/2||Bw-A||^2`` on the simplex.

    At the optimum there is a multiplier ``mu`` with ``g_i == mu`` on the
    support (``w_i > 0``) and ``g_i >= mu`` off it (``w_i == 0``), where
    ``g = B^T (Bw - A)``. Tolerance is scaled by the gradient magnitude so the
    check is invariant to problem scale. This certifies optimality with no
    reference to any other solver.
    """
    w = np.asarray(w, float).ravel()
    J = B.shape[1]
    assert_feasible(w, J, tol=max(tol, 1e-7))
    g = B.T @ (B @ w - A)
    scale = 1.0 + float(np.max(np.abs(g)))
    support = w > 1e-7
    assert support.any(), "empty support (sum-to-one violated?)"
    mu = float(g[support].mean())
    assert np.all(np.abs(g[support] - mu) <= tol * scale), "support gradients not equalised"
    if (~support).any():
        assert np.all(g[~support] >= mu - tol * scale), "off-support reduced gradient is negative"


def _rand(rng, m, J, scale=1.0):
    B = rng.normal(size=(m, J)) * scale
    A = rng.normal(size=m) * scale
    return B, A


# --------------------------------------------------------------------------- #
# 1. Smoke
# --------------------------------------------------------------------------- #
def test_smoke_returns_feasible_weights():
    rng = np.random.default_rng(0)
    B, A = _rand(rng, 5, 3)
    w = solve_simplex_qp(B, A)
    assert_feasible(w, J=3)


# --------------------------------------------------------------------------- #
# 2. Known-answer / unit
# --------------------------------------------------------------------------- #
def test_target_is_a_donor_recovers_vertex():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(8, 4))
    A = B[:, 2].copy()                      # target equals donor 2 exactly
    w = solve_simplex_qp(B, A)
    assert_feasible(w, 4)
    assert w[2] > 1 - 1e-5 and np.all(np.delete(w, 2) < 1e-5)


def test_midpoint_of_two_donors():
    rng = np.random.default_rng(2)
    B = rng.normal(size=(10, 2))
    A = 0.5 * (B[:, 0] + B[:, 1])
    w = solve_simplex_qp(B, A)
    assert np.allclose(w, [0.5, 0.5], atol=1e-5)


def test_in_hull_recovery_exact_fit():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(12, 5))            # m > J -> full column rank, unique
    w_true = rng.dirichlet(np.ones(5))
    A = B @ w_true
    w = solve_simplex_qp(B, A)
    assert np.allclose(w, w_true, atol=1e-5)
    assert _objective(B, A, w) < 1e-10


# --------------------------------------------------------------------------- #
# 3. Optimality certificate + parity with the oracle
# --------------------------------------------------------------------------- #
def test_kkt_certifier_validates_the_reference():
    """Cross-validate the certifier itself: cvxpy's solution must pass it."""
    rng = np.random.default_rng(4)
    B, A = _rand(rng, 9, 6)
    assert_kkt_optimal(B, A, _reference(B, A))


@pytest.mark.parametrize("seed", range(5))
def test_parity_with_cvxpy(seed):
    rng = np.random.default_rng(100 + seed)
    B, A = _rand(rng, 14, 7)
    w = solve_simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)
    w_ref = _reference(B, A)
    # objective + fitted value are unique even when weights are not
    assert abs(_objective(B, A, w) - _objective(B, A, w_ref)) < 1e-7
    assert np.allclose(B @ w, B @ w_ref, atol=1e-6)


def test_determinism():
    rng = np.random.default_rng(5)
    B, A = _rand(rng, 8, 5)
    assert np.array_equal(solve_simplex_qp(B, A), solve_simplex_qp(B, A))


# --------------------------------------------------------------------------- #
# 4. Edge / degenerate regimes (correctness must hold)
# --------------------------------------------------------------------------- #
def test_singular_gram_more_donors_than_periods():
    # J > m -> B^T B rank-deficient. lstsq free-set solve must cope; assert
    # optimality + objective parity (weights are non-unique here).
    rng = np.random.default_rng(6)
    B, A = _rand(rng, 5, 12)
    w = solve_simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)
    assert abs(_objective(B, A, w) - _objective(B, A, _reference(B, A))) < 1e-6


def test_collinear_donors_nonunique_weights():
    rng = np.random.default_rng(7)
    B = rng.normal(size=(10, 3))
    B[:, 2] = B[:, 0]                       # duplicate donor -> non-unique w
    A = rng.normal(size=10)
    w = solve_simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)
    assert np.allclose(B @ w, B @ _reference(B, A), atol=1e-6)


def test_target_outside_hull_boundary_solution():
    rng = np.random.default_rng(8)
    B = rng.normal(size=(10, 4))
    A = 50.0 * np.ones(10)                  # far from any convex combination
    w = solve_simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)


def test_single_donor():
    rng = np.random.default_rng(9)
    B = rng.normal(size=(6, 1))
    A = rng.normal(size=6)
    assert np.allclose(solve_simplex_qp(B, A), [1.0])


def test_extreme_scales():
    rng = np.random.default_rng(10)
    for scale in (1e-8, 1e8):
        B, A = _rand(rng, 8, 4, scale=scale)
        w = solve_simplex_qp(B, A)
        assert_kkt_optimal(B, A, w, tol=1e-5)


# --------------------------------------------------------------------------- #
# 5. Fuzz -- the differential property test across the regime grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(20, 5), (8, 8), (5, 15), (40, 3), (6, 20)])
def test_fuzz_parity_and_kkt(m, J):
    rng = np.random.default_rng(1000 + m * 100 + J)
    for _ in range(40):
        cond = rng.choice([1.0, 1e3])      # well- vs ill-conditioned
        B = rng.normal(size=(m, J)) * np.linspace(1.0, cond, J)
        A = rng.normal(size=m)
        w = solve_simplex_qp(B, A)
        assert_feasible(w, J)
        assert_kkt_optimal(B, A, w, tol=1e-5)
        assert abs(_objective(B, A, w) - _objective(B, A, _reference(B, A))) < 1e-5


# --------------------------------------------------------------------------- #
# 6. Warm-start: must not change the optimum, only the work
# --------------------------------------------------------------------------- #
def test_warm_start_returns_same_optimum():
    rng = np.random.default_rng(11)
    B, A = _rand(rng, 12, 6)
    cold = solve_simplex_qp(B, A)
    warm = solve_simplex_qp(B, A, warm_start=cold)
    assert np.allclose(B @ cold, B @ warm, atol=1e-7)


# --------------------------------------------------------------------------- #
# 7. Failure reporting (translated / clear, not silent garbage)
# --------------------------------------------------------------------------- #
def test_dimension_mismatch_raises():
    with pytest.raises((ValueError, Exception)):
        solve_simplex_qp(np.ones((5, 3)), np.ones(4))


def test_empty_donor_matrix_raises():
    with pytest.raises((ValueError, Exception)):
        solve_simplex_qp(np.ones((5, 0)), np.ones(5))
