"""Simplex least squares for many targets against one shared design.

Test-first (per ``agents/agents_tests.md``).

A cohort of treated units fitted against a common donor pool poses exactly this:
one ``A``, one target per unit. The Gram of each problem is
``A'A - c_j 1' - 1 c_j' + s_j 11'`` with ``c_j = A' b_j`` and ``s_j = b_j' b_j``,
so ``A'A`` and ``A'B`` computed once carry the whole set.

As everywhere else here, the batch has to reproduce the one-at-a-time solve and
not merely tie with it -- these weights are reported per treated unit and the
counterfactuals are built from them -- so ambiguous members are verified and
re-solved.

Which one-at-a-time solve is part of the contract, not a detail. STACKEDSC calls
the primal active set directly; VanillaSC's engine calls it through a wrapper
that escalates to CVXPY when the active set reports failure on itself. The two
agree everywhere the batch runs and can disagree on a design pathological enough
to trip that escape hatch, so the caller names the solver it is standing in for.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.active_set import solve_simplex_qp


def _loop(A, B, solver=solve_simplex_qp):
    return np.stack([np.asarray(solver(A, B[:, j]), dtype=float).ravel()
                     for j in range(B.shape[1])])


def _cohort(rng, m, J, k, in_hull=False):
    A = np.cumsum(rng.normal(size=(m, J)), axis=0) + 40.0
    if in_hull:
        B = A @ rng.dirichlet(np.ones(J), size=k).T
    else:
        B = np.cumsum(rng.normal(size=(m, k)), axis=0) + 40.0
    return A, B


@pytest.mark.parametrize("m,J,k", [(8, 39, 12), (30, 20, 6), (19, 38, 25), (5, 39, 40)])
def test_matches_the_one_at_a_time_solve(m, J, k):
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(m * 7 + J), m, J, k)
    np.testing.assert_allclose(solve_simplex_shared_design(A, B), _loop(A, B),
                               atol=1e-7)


def test_ambiguous_members_reproduce_the_incumbent_solver():
    """Targets the donors reproduce exactly, so the minimisers are faces."""
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(1), 6, 30, 10, in_hull=True)
    got, info = solve_simplex_shared_design(A, B, return_info=True)
    assert info["n_fallback"] > 0
    np.testing.assert_allclose(got, _loop(A, B), atol=1e-7)


def test_rows_are_on_the_simplex():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(2), 12, 25, 8)
    for w in solve_simplex_shared_design(A, B):
        assert w.min() >= -1e-9
        assert abs(w.sum() - 1.0) < 1e-9


def test_info_counts_every_problem():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(3), 20, 15, 9)
    _, info = solve_simplex_shared_design(A, B, return_info=True)
    assert info["n_problems"] == 9
    assert info["n_fallback"] == 0


def test_single_target():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(4), 10, 7, 1)
    np.testing.assert_allclose(solve_simplex_shared_design(A, B), _loop(A, B),
                               atol=1e-7)


def test_single_donor():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A = np.array([[1.0], [2.0], [3.0]])
    B = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    W = solve_simplex_shared_design(A, B)
    assert W.shape == (2, 1) and np.allclose(W, 1.0)


def test_rejects_mismatched_rows():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    with pytest.raises(ValueError, match="rows"):
        solve_simplex_shared_design(np.ones((5, 3)), np.ones((4, 2)))


def test_rejects_a_non_matrix_design():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    with pytest.raises(ValueError, match="2-D"):
        solve_simplex_shared_design(np.ones(5), np.ones((5, 2)))


def test_a_one_dimensional_target_is_treated_as_one_column():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _cohort(np.random.default_rng(5), 14, 9, 1)
    np.testing.assert_allclose(solve_simplex_shared_design(A, B[:, 0]),
                               _loop(A, B), atol=1e-7)


# --------------------------------------------------------------------------- #
# Designs whose Gram loses most of float64
# --------------------------------------------------------------------------- #
def _ill_conditioned(rng, m=8, J=39, k=20):
    """Covariate-matching geometry: predictors in different units, so the design
    is merely awkward at cond ~ 1e7 and its Gram is at the edge of float64."""
    A = np.cumsum(rng.normal(size=(m, J)), axis=0) + 100.0
    A[:3, :] *= 1e5
    A[3:5, :] *= 1e-3
    B = np.cumsum(rng.normal(size=(m, k)), axis=0) + 100.0
    B[:3, :] *= 1e5
    B[3:5, :] *= 1e-3
    return A, B


def test_an_ill_conditioned_design_still_matches_the_loop():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    A, B = _ill_conditioned(np.random.default_rng(11))
    sv = np.linalg.svd(A, compute_uv=False)
    assert sv[-1] / sv[0] < 1e-10                            # precondition
    np.testing.assert_allclose(solve_simplex_shared_design(A, B), _loop(A, B),
                               atol=1e-7)


def test_the_fallback_solver_is_the_callers_own():
    """VanillaSC reaches the active set through a wrapper that escalates to
    CVXPY; STACKEDSC calls it directly. A batch standing in for one must not
    return the other's answer."""
    from mlsynth.utils.bilevel.minnorm import solve_simplex_shared_design

    seen = []

    def spy(Bm, a):
        seen.append(Bm.shape)
        return solve_simplex_qp(Bm, a)

    A, B = _ill_conditioned(np.random.default_rng(14))
    W, info = solve_simplex_shared_design(A, B, fallback=spy, return_info=True)
    assert len(seen) == info["n_fallback"] > 0
    np.testing.assert_allclose(W, _loop(A, B), atol=1e-7)
