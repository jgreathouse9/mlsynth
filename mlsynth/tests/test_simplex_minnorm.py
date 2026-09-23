"""Correctness contract for the batched Gram-form simplex QP.

Test-first (per ``agents/agents_tests.md``): this harness is written before
:mod:`mlsynth.utils.bilevel.minnorm`, so every test here is RED until the
minimum-norm-point solver satisfies it.

The solver minimises ``w' G w`` over the probability simplex, for a whole stack
of Gram matrices ``G`` at once. With the sum-to-one constraint in force,
``||B w - A||^2 = w' G w`` for ``G = (B - A 1')' (B - A 1')``, so the contract
below is stated in the *design* variables ``(B, A)`` the rest of the library
speaks -- and every claim is checked against the same three pillars the
single-problem active set is held to in ``test_simplex_active_set.py``:

1. KKT certificate -- optimality proven without trusting another solver.
2. cvxpy parity -- the objective never exceeds the exact reference's.
3. Fuzz -- seeded random instances across the regime grid, including the
   rank-deficient and collinear designs a donor pool actually produces.

Plus two contracts the batch form adds: a batch must equal a loop over its own
members, and a warm start must change the work done, never the optimum.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.minnorm import (
    simplex_gram,
    solve_simplex_minnorm,
    solve_simplex_minnorm_batch,
)


# --------------------------------------------------------------------------- #
# Infrastructure: the exact oracle and a Gram-form KKT certifier
# --------------------------------------------------------------------------- #
def _reference(B: np.ndarray, A: np.ndarray):
    """Exact simplex-constrained least squares via cvxpy (best-effort oracle)."""
    w = cp.Variable(B.shape[1])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(B @ w - A)),
                      [w >= 0, cp.sum(w) == 1])
    try:
        prob.solve()
    except Exception:  # pragma: no cover - oracle failure is handled, not asserted
        return None
    if w.value is None:  # pragma: no cover - same
        return None
    return np.asarray(w.value, dtype=float).ravel()


def _objective(G, w):
    w = np.asarray(w, dtype=float).ravel()
    return float(w @ G @ w)


def assert_feasible(w, J, tol=1e-7):
    w = np.asarray(w, dtype=float).ravel()
    assert w.shape == (J,), f"shape {w.shape} != ({J},)"
    assert np.all(np.isfinite(w)), "non-finite weights"
    assert w.min() >= -tol, f"negative weight {w.min():.2e}"
    assert abs(w.sum() - 1.0) <= tol, f"weights sum to {w.sum():.6f}"


def assert_kkt_optimal(G, w, tol=1e-6):
    """Optimality of ``w`` for ``min w'Gw`` on the simplex, from first principles.

    The multiplier on the sum-to-one constraint is ``nu = w'Gw``; stationarity
    then reads ``(Gw)_j == nu`` on the support and ``(Gw)_j >= nu`` off it. The
    tolerance is scaled by the problem's own magnitude so the check is invariant
    to how the donors are scaled.
    """
    w = np.asarray(w, dtype=float).ravel()
    assert_feasible(w, G.shape[0], tol=max(tol, 1e-7))
    g = G @ w
    nu = float(w @ g)
    scale = 1.0 + float(np.max(np.abs(np.diag(G))))
    support = w > 1e-7
    assert support.any(), "empty support (sum-to-one violated?)"
    assert np.all(np.abs(g[support] - nu) <= tol * scale), \
        "support gradients not equalised"
    if (~support).any():
        assert np.all(g[~support] >= nu - tol * scale), \
            "an off-support donor would improve the fit"


def assert_no_worse_than_reference(B, A, w, rtol=1e-6):
    ref_w = _reference(B, A)
    if ref_w is None:  # pragma: no cover - oracle unavailable; KKT covers it
        return
    G = simplex_gram(B, A)
    ours, ref = _objective(G, w), _objective(G, ref_w)
    assert ours <= ref + rtol * (1.0 + abs(ref)), f"objective {ours} vs ref {ref}"


def _rand(rng, m, J, scale=1.0):
    return rng.normal(size=(m, J)) * scale, rng.normal(size=m) * scale


# --------------------------------------------------------------------------- #
# 1. simplex_gram: the design -> Gram reduction the solver rests on
# --------------------------------------------------------------------------- #
def test_gram_reproduces_the_least_squares_objective_on_the_simplex():
    """``w'Gw == ||Bw - A||^2`` for every ``w`` on the simplex -- the identity
    that lets the solver forget the design matrix."""
    rng = np.random.default_rng(0)
    B, A = _rand(rng, 7, 4)
    G = simplex_gram(B, A)
    for _ in range(20):
        w = rng.dirichlet(np.ones(4))
        assert np.isclose(_objective(G, w), float(np.sum((B @ w - A) ** 2)))


def test_gram_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(1)
    B, A = _rand(rng, 6, 5)
    G = simplex_gram(B, A)
    assert np.allclose(G, G.T)
    assert np.linalg.eigvalsh(G).min() > -1e-10


def test_gram_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must equal"):
        simplex_gram(np.ones((4, 3)), np.ones(5))


def test_gram_rejects_a_non_matrix_design():
    with pytest.raises(ValueError, match="2-D"):
        simplex_gram(np.ones(4), np.ones(4))


# --------------------------------------------------------------------------- #
# 1b. When the reduction is safe to use at all
# --------------------------------------------------------------------------- #
def test_gram_reduction_is_safe_only_on_a_full_column_rank_design():
    """Forming ``G`` squares the design's condition number, so the reduction is
    only faithful where the design has full column rank. The guard says so."""
    from mlsynth.utils.bilevel.minnorm import gram_reduction_is_safe

    rng = np.random.default_rng(0)
    assert gram_reduction_is_safe(rng.normal(size=(40, 12))) is True
    assert gram_reduction_is_safe(rng.normal(size=(8, 39))) is False
    collinear = rng.normal(size=(40, 6))
    collinear[:, 5] = collinear[:, 0]
    assert gram_reduction_is_safe(collinear) is False
    assert gram_reduction_is_safe(np.zeros((5, 3))) is False


def test_full_rank_designs_agree_with_the_design_form_on_the_weights():
    """Where the guard passes, the two exact solvers return the same weights and
    not merely the same fit -- which is what makes swapping them safe."""
    rng = np.random.default_rng(1)
    from mlsynth.utils.bilevel.minnorm import gram_reduction_is_safe

    for m, J in [(40, 20), (30, 12), (60, 25)]:
        B, A = _rand(rng, m, J)
        assert gram_reduction_is_safe(B)
        w_gram = solve_simplex_minnorm(simplex_gram(B, A))
        w_design = solve_simplex_qp(B, A)
        assert np.allclose(w_gram, w_design, atol=1e-6)


def test_rank_deficient_designs_agree_on_the_fit_but_not_the_weights():
    """The other side of the guard, stated as a fact and not an aspiration: on a
    face the two solvers land in different places, both optimal. Anything reading
    the weights themselves -- a donor table, a counterfactual built from
    post-period donor outcomes -- would change if it swapped one for the other.

    Which places they land in depends on where each one starts, and the Gram
    form's start moved when it gained a first-order seed (#461). Cold, it began
    at a vertex and certified at a sparse corner of the face: 9 donors, 0.209
    from the design form's answer. Seeded, it begins at a spread point and
    certifies near it: 38 donors -- the design form's own support, exactly -- and
    0.015 away within that face. The two forms therefore agree better than they
    did, which narrows this failure without closing it: they now pick the same
    donors and still disagree on how much each one carries, so a donor table
    still moves when one is swapped for the other, and the guard is still what
    decides whether the reduction is allowed.
    """
    rng = np.random.default_rng(2)
    from mlsynth.utils.bilevel.minnorm import gram_reduction_is_safe

    B = np.cumsum(rng.normal(size=(8, 39)), axis=0) + 10.0
    A = B @ rng.dirichlet(np.ones(39)) + 0.05 * rng.normal(size=8)
    assert not gram_reduction_is_safe(B)
    w_gram = solve_simplex_minnorm(simplex_gram(B, A))
    w_design = solve_simplex_qp(B, A)
    assert np.allclose(B @ w_gram, B @ w_design, atol=1e-7)     # same fit
    assert np.abs(w_gram - w_design).max() > 1e-3               # different point
    assert np.array_equal(w_gram > 1e-9, w_design > 1e-9)       # same donors now


def test_the_cold_gram_solver_still_lands_at_the_sparse_corner():
    """The behaviour above, pinned on both sides of the seed.

    The seed is speed only, so it may move which optimum on a face comes back
    and may not move the fit. Both are asserted here rather than inferred: the
    cold and seeded answers differ from each other by more than either differs
    in fit, and both reproduce the design form's fitted values.
    """
    rng = np.random.default_rng(2)
    B = np.cumsum(rng.normal(size=(8, 39)), axis=0) + 10.0
    A = B @ rng.dirichlet(np.ones(39)) + 0.05 * rng.normal(size=8)
    G = simplex_gram(B, A)
    w_cold = solve_simplex_minnorm(G, accelerate=False)
    w_seeded = solve_simplex_minnorm(G)
    assert (w_cold > 1e-9).sum() < (w_seeded > 1e-9).sum()
    assert np.abs(w_cold - w_seeded).max() > 0.05
    for w in (w_cold, w_seeded):
        assert np.allclose(B @ w, B @ solve_simplex_qp(B, A), atol=1e-7)


# --------------------------------------------------------------------------- #
# 2. Smoke
# --------------------------------------------------------------------------- #
def test_smoke_single_returns_feasible_weights():
    rng = np.random.default_rng(2)
    B, A = _rand(rng, 5, 3)
    w = solve_simplex_minnorm(simplex_gram(B, A))
    assert_feasible(w, J=3)


def test_smoke_batch_returns_one_feasible_row_per_problem():
    rng = np.random.default_rng(3)
    G = np.stack([simplex_gram(*_rand(rng, 5, 4)) for _ in range(6)])
    W = solve_simplex_minnorm_batch(G)
    assert W.shape == (6, 4)
    for w in W:
        assert_feasible(w, J=4)


# --------------------------------------------------------------------------- #
# 3. Known answers
# --------------------------------------------------------------------------- #
def test_target_is_a_donor_recovers_vertex():
    rng = np.random.default_rng(4)
    B = rng.normal(size=(8, 4))
    A = B[:, 2].copy()
    w = solve_simplex_minnorm(simplex_gram(B, A))
    assert w[2] > 1 - 1e-5 and np.all(np.delete(w, 2) < 1e-5)


def test_midpoint_of_two_donors():
    rng = np.random.default_rng(5)
    B = rng.normal(size=(10, 2))
    A = 0.5 * (B[:, 0] + B[:, 1])
    w = solve_simplex_minnorm(simplex_gram(B, A))
    assert np.allclose(w, [0.5, 0.5], atol=1e-6)


def test_in_hull_recovery_exact_fit():
    rng = np.random.default_rng(6)
    B = rng.normal(size=(12, 5))
    w_true = rng.dirichlet(np.ones(5))
    A = B @ w_true
    G = simplex_gram(B, A)
    w = solve_simplex_minnorm(G)
    assert np.allclose(w, w_true, atol=1e-6)
    assert _objective(G, w) < 1e-12


def test_single_donor_is_the_only_answer():
    w = solve_simplex_minnorm(simplex_gram(np.ones((3, 1)), np.zeros(3)))
    assert w.shape == (1,) and w[0] == 1.0


# --------------------------------------------------------------------------- #
# 4. Optimality certificate + parity with the oracle and the incumbent solver
# --------------------------------------------------------------------------- #
def test_kkt_certifier_validates_the_reference():
    """Cross-validate the certifier itself on cvxpy's solution."""
    rng = np.random.default_rng(7)
    B, A = _rand(rng, 9, 6)
    assert_kkt_optimal(simplex_gram(B, A), _reference(B, A))


@pytest.mark.parametrize("seed", range(6))
def test_parity_with_cvxpy(seed):
    rng = np.random.default_rng(100 + seed)
    B, A = _rand(rng, 14, 7)
    G = simplex_gram(B, A)
    w = solve_simplex_minnorm(G)
    assert_kkt_optimal(G, w)
    assert_no_worse_than_reference(B, A, w)
    assert np.allclose(B @ w, B @ _reference(B, A), atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_parity_with_the_design_form_active_set(seed):
    """The incumbent exact solver and this one must agree on the fitted value."""
    rng = np.random.default_rng(200 + seed)
    B, A = _rand(rng, 11, 6)
    w_gram = solve_simplex_minnorm(simplex_gram(B, A))
    w_design = solve_simplex_qp(B, A)
    assert np.allclose(B @ w_gram, B @ w_design, atol=1e-6)


@pytest.mark.parametrize("m,J", [(20, 5), (5, 20), (12, 12), (3, 40), (30, 2)])
def test_fuzz_across_the_regime_grid(m, J):
    rng = np.random.default_rng(1000 + m * 100 + J)
    for _ in range(15):
        B, A = _rand(rng, m, J)
        G = simplex_gram(B, A)
        w = solve_simplex_minnorm(G)
        assert_kkt_optimal(G, w, tol=1e-5)
        assert_no_worse_than_reference(B, A, w, rtol=1e-5)


# --------------------------------------------------------------------------- #
# 5. Batch == loop, and invariances
# --------------------------------------------------------------------------- #
def test_batch_matches_a_loop_over_its_members():
    rng = np.random.default_rng(8)
    probs = [_rand(rng, 9, 6) for _ in range(12)]
    G = np.stack([simplex_gram(B, A) for B, A in probs])
    W = solve_simplex_minnorm_batch(G)
    for Gi, wi in zip(G, W):
        assert np.isclose(_objective(Gi, wi),
                          _objective(Gi, solve_simplex_minnorm(Gi)), atol=1e-12)


def test_batch_members_do_not_interact():
    """Reordering (or padding) a batch cannot change any member's answer."""
    rng = np.random.default_rng(9)
    G = np.stack([simplex_gram(*_rand(rng, 8, 5)) for _ in range(7)])
    W = solve_simplex_minnorm_batch(G)
    order = np.array([3, 0, 6, 1, 5, 2, 4])
    W_shuffled = solve_simplex_minnorm_batch(G[order])
    assert np.allclose(W_shuffled, W[order], atol=1e-9)


def test_scale_invariance():
    """``G`` and ``c G`` define the same minimiser for any ``c > 0``."""
    rng = np.random.default_rng(10)
    G = simplex_gram(*_rand(rng, 9, 6))
    base = solve_simplex_minnorm(G)
    for c in (1e-8, 1e-3, 1e3, 1e8):
        assert np.allclose(solve_simplex_minnorm(c * G), base, atol=1e-7)


def test_permutation_equivariance():
    rng = np.random.default_rng(11)
    B, A = _rand(rng, 10, 6)
    w = solve_simplex_minnorm(simplex_gram(B, A))
    perm = np.array([4, 1, 0, 5, 3, 2])
    w_perm = solve_simplex_minnorm(simplex_gram(B[:, perm], A))
    assert np.allclose(B[:, perm] @ w_perm, B @ w, atol=1e-6)


def test_determinism():
    rng = np.random.default_rng(12)
    G = simplex_gram(*_rand(rng, 8, 5))
    assert np.array_equal(solve_simplex_minnorm(G), solve_simplex_minnorm(G))


# --------------------------------------------------------------------------- #
# 6. Edge / degenerate regimes
# --------------------------------------------------------------------------- #
def test_more_donors_than_periods_rank_deficient():
    rng = np.random.default_rng(13)
    B, A = _rand(rng, 5, 12)
    G = simplex_gram(B, A)
    w = solve_simplex_minnorm(G)
    assert_kkt_optimal(G, w, tol=1e-5)
    assert_no_worse_than_reference(B, A, w, rtol=1e-5)


def test_collinear_donors_nonunique_weights():
    rng = np.random.default_rng(14)
    B = rng.normal(size=(10, 3))
    B[:, 2] = B[:, 0]
    A = rng.normal(size=10)
    G = simplex_gram(B, A)
    w = solve_simplex_minnorm(G)
    assert_kkt_optimal(G, w, tol=1e-5)
    assert np.allclose(B @ w, B @ _reference(B, A), atol=1e-6)


def test_identical_donors_leave_every_weight_optimal():
    """Every donor identical -> ``G`` is the zero matrix and the objective is
    flat. Any simplex point is optimal; the solver must still return one."""
    B = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 4))
    A = B[:, 0].copy()
    G = simplex_gram(B, A)
    assert np.allclose(G, 0.0)
    w = solve_simplex_minnorm(G)
    assert_feasible(w, 4)


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_extreme_scaling(scale):
    rng = np.random.default_rng(15)
    B, A = _rand(rng, 9, 6, scale=scale)
    G = simplex_gram(B, A)
    w = solve_simplex_minnorm(G)
    assert_kkt_optimal(G, w, tol=1e-5)
    assert_no_worse_than_reference(B, A, w, rtol=1e-5)


def test_two_donors_target_outside_the_hull_lands_on_a_vertex():
    B = np.array([[0.0, 1.0], [0.0, 0.0]])
    A = np.array([-5.0, 0.0])                 # nearest simplex point is donor 0
    w = solve_simplex_minnorm(simplex_gram(B, A))
    assert np.allclose(w, [1.0, 0.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# 7. Warm start: changes the work, never the optimum
# --------------------------------------------------------------------------- #
def test_warm_start_at_the_optimum_certifies_immediately():
    rng = np.random.default_rng(16)
    G = np.stack([simplex_gram(*_rand(rng, 10, 6)) for _ in range(5)])
    W = solve_simplex_minnorm_batch(G)
    W2, info = solve_simplex_minnorm_batch(G, warm_start=W, return_info=True)
    assert np.allclose(W2, W, atol=1e-9)
    assert info["iterations"] == 1              # one solve to certify, no pivots


def test_warm_start_does_not_move_the_optimum():
    rng = np.random.default_rng(17)
    G = np.stack([simplex_gram(*_rand(rng, 12, 7)) for _ in range(8)])
    cold = solve_simplex_minnorm_batch(G)
    warm = rng.dirichlet(np.ones(7), size=8)
    hot = solve_simplex_minnorm_batch(G, warm_start=warm)
    for Gi, a, b in zip(G, cold, hot):
        assert np.isclose(_objective(Gi, a), _objective(Gi, b), atol=1e-10)


@pytest.mark.parametrize("bad", ["negative", "zeros", "wrong_shape", "nan"])
def test_warm_start_from_garbage_is_ignored(bad):
    rng = np.random.default_rng(18)
    G = np.stack([simplex_gram(*_rand(rng, 9, 5)) for _ in range(4)])
    cold = solve_simplex_minnorm_batch(G)
    warms = {
        "negative": np.full((4, 5), -1.0),
        "zeros": np.zeros((4, 5)),
        "wrong_shape": np.full((3, 5), 0.2),
        "nan": np.full((4, 5), np.nan),
    }
    W = solve_simplex_minnorm_batch(G, warm_start=warms[bad])
    for Gi, a, b in zip(G, cold, W):
        assert np.isclose(_objective(Gi, a), _objective(Gi, b), atol=1e-10)


# --------------------------------------------------------------------------- #
# 8. Diagnostics and failure reporting
# --------------------------------------------------------------------------- #
def test_info_reports_convergence_per_problem():
    rng = np.random.default_rng(19)
    G = np.stack([simplex_gram(*_rand(rng, 10, 6)) for _ in range(5)])
    _, info = solve_simplex_minnorm_batch(G, return_info=True)
    assert info["converged"].shape == (5,)
    assert info["converged"].all()
    assert info["iterations"] >= 1


def test_truncated_iteration_budget_is_reported_not_hidden():
    """A budget too small to certify must come back flagged, with a feasible
    iterate -- never a silent claim of optimality."""
    rng = np.random.default_rng(20)
    G = np.stack([simplex_gram(*_rand(rng, 12, 20)) for _ in range(4)])
    W, info = solve_simplex_minnorm_batch(G, max_iter=1, return_info=True)
    assert not info["converged"].all()
    for w in W:
        assert_feasible(w, 20)


def test_single_solver_reports_its_own_convergence():
    rng = np.random.default_rng(21)
    G = simplex_gram(*_rand(rng, 10, 6))
    w, info = solve_simplex_minnorm(G, return_info=True)
    assert info["converged"] is True
    assert_kkt_optimal(G, w)


# --------------------------------------------------------------------------- #
# 9. Input validation
# --------------------------------------------------------------------------- #
def test_rejects_non_square_gram():
    with pytest.raises(ValueError, match="square"):
        solve_simplex_minnorm(np.ones((3, 4)))


def test_rejects_wrong_dimensionality():
    with pytest.raises(ValueError, match="2-D|3-D"):
        solve_simplex_minnorm(np.ones(4))
    with pytest.raises(ValueError, match="3-D"):
        solve_simplex_minnorm_batch(np.ones((4, 4)))


def test_rejects_empty_donor_pool():
    with pytest.raises(ValueError, match="donor"):
        solve_simplex_minnorm(np.ones((0, 0)))


def test_rejects_non_finite_gram():
    G = np.eye(3)
    G[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_simplex_minnorm(G)
