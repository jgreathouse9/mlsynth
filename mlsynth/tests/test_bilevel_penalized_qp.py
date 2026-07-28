"""Exactness of the penalized backend's inner simplex QP.

``penalized_weights`` solves Abadie-L'Hour eq. (5),

    min_w  ||X1 - X0 w||^2 + lambda * sum_j w_j ||X1 - X_j||^2
    s.t.   w >= 0, sum_j w_j = 1,

a convex quadratic program over the simplex. Theorem 1 says that for any
``lambda > 0`` its solution is unique and carries at most ``K + 1`` non-zero
weights, where ``K`` is the number of matching variables. That guarantee is the
entire reason the backend exists -- it is what makes the estimator well defined
when the treated unit sits inside the donor convex hull and the unpenalized
program has a continuum of solutions.

A first-order method does not deliver it at realistic iteration budgets. These
tests pin the exact behaviour instead: the returned weights must attain the
optimal objective value, and hence the sparsity bound.

Written test-first, against an independent cvxpy reference.

Two tolerances need stating, because the guarantee is exact arithmetic and the
solver is numerical. Objective values are compared with a *relative* tolerance:
an interior-point QP lands within ~1e-6 relative of the optimum, so demanding
bit-agreement with another solver run would test the solver, not this module.
And a donor is counted as carrying weight above ``_WEIGHT_FLOOR``: interior-point
methods approach the optimal face from inside and never return exact vertices,
so they always leave dust several orders of magnitude below any weight that
matters (the source paper reports its own weights to three decimals). The
substantive selection is what is asserted, and it agrees with the reference
donor-for-donor.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.penalized import penalized_weights


# --------------------------------------------------------------------------- #
# reference + helpers
# --------------------------------------------------------------------------- #
def _reference(X1: np.ndarray, X0: np.ndarray, lam: float) -> np.ndarray:
    """Independent exact solve of eq. (5), built from the paper's statement."""
    import cvxpy as cp

    J = X0.shape[1]
    d2 = np.sum((X1[:, None] - X0) ** 2, axis=0)
    w = cp.Variable(J, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(X1 - X0 @ w) + lam * (d2 @ w)),
               [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    v = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    return v / v.sum()


def _objective(X1: np.ndarray, X0: np.ndarray, lam: float, w: np.ndarray) -> float:
    d2 = np.sum((X1[:, None] - X0) ** 2, axis=0)
    return float(((X1 - X0 @ w) ** 2).sum() + lam * (d2 @ w))


def _problem(J: int = 12, K: int = 4, seed: int = 0):
    """Treated unit strictly inside the donor hull -> unpenalized non-uniqueness."""
    rng = np.random.default_rng(seed)
    X0 = rng.uniform(1.0, 5.0, size=(K, J))
    w = np.zeros(J)
    w[[0, 1, 2]] = [0.5, 0.3, 0.2]
    return X0 @ w, X0


LAMBDAS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)

# A donor "carries weight" above this; below it is interior-point residue.
_WEIGHT_FLOOR = 1e-4
# Relative objective tolerance: interior-point accuracy, not solver agreement.
_OBJ_RTOL = 1e-4


def _nnz(w: np.ndarray) -> int:
    return int((w > _WEIGHT_FLOOR).sum())


# --------------------------------------------------------------------------- #
# optimality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", LAMBDAS)
def test_attains_the_optimal_objective(lam):
    """The returned weights must not be beaten by an independent exact solve."""
    X1, X0 = _problem()
    got = _objective(X1, X0, lam, penalized_weights(X1, X0, lam))
    ref = _objective(X1, X0, lam, _reference(X1, X0, lam))
    assert got <= ref * (1.0 + _OBJ_RTOL) + 1e-12


@pytest.mark.parametrize("J,K", [(12, 4), (98, 9), (38, 19), (5, 3)])
def test_attains_the_optimum_across_shapes(J, K):
    """Including the wide (J >> K) regime, where the unpenalized program is
    most degenerate and the penalty matters most."""
    X1, X0 = _problem(J=J, K=K)
    lam = 1e-3
    got = _objective(X1, X0, lam, penalized_weights(X1, X0, lam))
    ref = _objective(X1, X0, lam, _reference(X1, X0, lam))
    assert got <= ref * (1.0 + _OBJ_RTOL) + 1e-12


# --------------------------------------------------------------------------- #
# Theorem 1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", LAMBDAS)
def test_sparsity_bound_holds_at_every_positive_lambda(lam):
    """At most ``K + 1`` non-zero weights, for *any* positive lambda -- the
    small-lambda end included, which the first-order solver could not reach."""
    K = 4
    X1, X0 = _problem(K=K)
    w = penalized_weights(X1, X0, lam)
    assert _nnz(w) <= K + 1


def test_solution_is_unique_across_starting_conditions(lam=1e-3):
    """Uniqueness: perturbing the problem's column order must permute the
    solution, not change which donors are selected."""
    X1, X0 = _problem()
    order = np.array([7, 0, 3, 11, 2, 9, 1, 5, 8, 4, 10, 6])
    a = penalized_weights(X1, X0, lam)
    b = penalized_weights(X1, X0[:, order], lam)
    assert np.allclose(a[order], b, atol=1e-6)


def test_selection_matches_an_independent_solver(lam=1e-3):
    """The donors carrying weight must be the same set the reference picks."""
    for J, K in [(12, 4), (98, 9), (38, 19)]:
        X1, X0 = _problem(J=J, K=K)
        mine = set(np.where(penalized_weights(X1, X0, lam) > _WEIGHT_FLOOR)[0])
        ref = set(np.where(_reference(X1, X0, lam) > _WEIGHT_FLOOR)[0])
        assert mine == ref, f"J={J} K={K}"


def test_residue_below_the_floor_stays_negligible(lam=1e-3):
    """Guard on the dust itself: if it ever grows into the range where it could
    be mistaken for a real weight, this fails."""
    X1, X0 = _problem(J=98, K=9)
    w = penalized_weights(X1, X0, lam)
    assert w[w <= _WEIGHT_FLOOR].max() < 1e-5


def test_repeated_calls_agree_exactly(lam=1e-3):
    X1, X0 = _problem()
    assert np.allclose(penalized_weights(X1, X0, lam),
                       penalized_weights(X1, X0, lam), atol=1e-12)


# --------------------------------------------------------------------------- #
# invariants and edges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", (0.0,) + LAMBDAS)
def test_weights_are_on_the_simplex(lam):
    X1, X0 = _problem()
    w = penalized_weights(X1, X0, lam)
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_single_donor_gets_all_the_weight():
    X1, X0 = _problem()
    w = penalized_weights(X1, X0[:, :1], 1e-3)
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0)


def test_large_lambda_approaches_nearest_neighbour_matching():
    """As lambda grows the penalty dominates and the solution collapses onto
    the single closest donor (the matching estimator)."""
    X1, X0 = _problem()
    d2 = np.sum((X1[:, None] - X0) ** 2, axis=0)
    w = penalized_weights(X1, X0, 1e6)
    assert int(np.argmax(w)) == int(np.argmin(d2))
    assert w.max() > 0.99


def test_zero_lambda_still_fits_the_target():
    """lambda = 0 is the unpenalized synthetic control: the fit must be exact
    here because the treated unit is inside the hull."""
    X1, X0 = _problem()
    w = penalized_weights(X1, X0, 0.0)
    assert ((X1 - X0 @ w) ** 2).sum() < 1e-12


def test_collinear_donors_do_not_break_the_solve():
    """A duplicated donor makes Q singular; the solve must still return a valid
    simplex point attaining the optimum."""
    X1, X0 = _problem()
    X0 = np.hstack([X0, X0[:, :2]])           # exact duplicates
    lam = 1e-3
    w = penalized_weights(X1, X0, lam)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    ref = _objective(X1, X0, lam, _reference(X1, X0, lam))
    assert _objective(X1, X0, lam, w) <= ref * (1.0 + _OBJ_RTOL) + 1e-12


def test_negative_lambda_is_rejected():
    X1, X0 = _problem()
    with pytest.raises(ValueError, match="non-negative"):
        penalized_weights(X1, X0, -1.0)


# --------------------------------------------------------------------------- #
# scale invariance
# --------------------------------------------------------------------------- #
def _scaled_problem(scale: float, J: int = 40, K: int = 9, seed: int = 3):
    """The same geometry at an arbitrary unit scale.

    Real panels are not order-1: the motivating case for this backend is
    campaign contributions in dollars, where the pairwise discrepancies reach
    1e13. A solver handed those directly can fail outright.
    """
    rng = np.random.default_rng(seed)
    X0 = rng.uniform(1.0, 5.0, size=(K, J)) * scale
    w = np.zeros(J)
    w[[0, 1, 2]] = [0.5, 0.3, 0.2]
    return X0 @ w, X0


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e5, 1e6])
@pytest.mark.parametrize("lam", [1e-2, 1.0, 100.0])
def test_solve_is_invariant_to_the_unit_scale(scale, lam):
    """Rescaling every input by a constant must not change the weights: the
    objective is homogeneous, so the argmin is scale-free."""
    X1_ref, X0_ref = _scaled_problem(1.0)
    X1, X0 = _scaled_problem(scale)
    assert np.allclose(penalized_weights(X1, X0, lam),
                       penalized_weights(X1_ref, X0_ref, lam), atol=1e-5)


@pytest.mark.parametrize("scale", [1e5, 1e6])
@pytest.mark.parametrize("lam", [1.0, 100.0])
def test_large_scale_data_still_returns_a_real_solution(scale, lam):
    """Guard against a silent fallback: uniform weights over every donor are
    what a failed solve looks like, and must never be returned quietly."""
    X1, X0 = _scaled_problem(scale)
    w = penalized_weights(X1, X0, lam)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(w, 1.0 / len(w), atol=1e-9), "silent uniform fallback"
    assert _nnz(w) <= 9 + 1
