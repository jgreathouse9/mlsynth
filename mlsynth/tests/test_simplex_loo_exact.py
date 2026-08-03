"""Exact leave-one-out simplex least squares, for the in-space placebo family.

Test-first (per ``agents/agents_tests.md``): written before
``solve_simplex_loo_exact`` exists.

The in-space placebo casts every donor as treated in turn and refits against the
others, so it poses ``J`` programs that share one matrix. In Gram form the whole
family falls out of a single ``M' M``: deleting column ``j`` deletes a row and a
column of the Gram, and the target is itself a column of ``M``, so the cross term
is a column of that same Gram. No product with the data appears per problem.

What the family gives up is uniqueness in general -- for some ``j`` the minimiser
may be a face, and then the batched and one-at-a-time solvers are both right and
disagree. Since the placebo p-value is a rank statistic over these fits, that has
to be settled per member rather than assumed, so each is verified and the
ambiguous ones fall back to the solver that produced the library's existing
numbers. These tests pin exactly that.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_augment import simplex_qp


def _loop(M, n=None):
    """The family solved one at a time, which the batch must reproduce."""
    J = M.shape[1]
    n = J if n is None else n
    out = np.zeros((n, J))
    for j in range(n):
        others = [k for k in range(J) if k != j]
        out[j, others] = simplex_qp(M[:, others], M[:, j])
    return out


def _panel(rng, T0, J, kind="sparse"):
    M = np.cumsum(rng.normal(size=(T0, J)), axis=0) + 50.0
    if kind == "collinear":
        M[:, -1] = M[:, 0]
    return M


# --------------------------------------------------------------------------- #
# 1. The batch reproduces the loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T0,J", [(19, 38), (30, 12), (89, 49), (12, 60), (8, 20)])
def test_matches_the_one_at_a_time_solve(T0, J):
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(T0 * 100 + J), T0, J)
    got = solve_simplex_loo_exact(M)
    want = _loop(M)
    assert got.shape == (J, J)
    np.testing.assert_allclose(got, want, atol=1e-7)


def test_each_row_is_on_the_simplex_and_excludes_itself():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(1), 20, 15)
    W = solve_simplex_loo_exact(M)
    for j, w in enumerate(W):
        assert w[j] == 0.0                       # a donor never fits itself
        assert w.min() >= -1e-9
        assert abs(w.sum() - 1.0) < 1e-9


def test_fitted_values_match_the_loop_even_where_weights_need_not():
    """On a family member whose minimiser is a face the weights may legitimately
    differ; the fit never may."""
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    rng = np.random.default_rng(2)
    M = _panel(rng, 6, 25)                        # wide and short: faces likely
    W = solve_simplex_loo_exact(M)
    want = _loop(M)
    for j in range(M.shape[1]):
        others = [k for k in range(M.shape[1]) if k != j]
        got_fit = M[:, others] @ W[j, others]
        want_fit = M[:, others] @ want[j, others]
        assert np.allclose(got_fit, want_fit, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Ambiguous members fall back, so existing numbers are preserved
# --------------------------------------------------------------------------- #
def test_non_unique_members_reproduce_the_incumbent_solver_exactly():
    """A panel built so several members have a face for a minimiser. Those must
    come back bit-comparable to the one-at-a-time solve, not merely optimal."""
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(size=(5, 4)), axis=0) + 20.0
    mix = rng.dirichlet(np.ones(4), size=16).T
    M = np.hstack([base, base @ mix])             # 20 donors spanning 4 rows
    W = solve_simplex_loo_exact(M)
    np.testing.assert_allclose(W, _loop(M), atol=1e-7)


def test_collinear_donors_reproduce_the_incumbent_solver():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(4), 15, 10, kind="collinear")
    np.testing.assert_allclose(solve_simplex_loo_exact(M), _loop(M), atol=1e-7)


def test_info_reports_how_many_members_fell_back():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    rng = np.random.default_rng(5)
    base = np.cumsum(rng.normal(size=(5, 4)), axis=0) + 20.0
    M = np.hstack([base, base @ rng.dirichlet(np.ones(4), size=16).T])
    _, info = solve_simplex_loo_exact(M, return_info=True)
    assert set(info) >= {"n_fallback", "n_problems"}
    assert info["n_problems"] == M.shape[1]
    assert 0 <= info["n_fallback"] <= info["n_problems"]

    clean = _panel(np.random.default_rng(6), 40, 12)
    _, info_clean = solve_simplex_loo_exact(clean, return_info=True)
    assert info_clean["n_fallback"] == 0


# --------------------------------------------------------------------------- #
# 3. Edges
# --------------------------------------------------------------------------- #
def test_two_donors():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(7), 10, 2)
    W = solve_simplex_loo_exact(M)
    assert np.allclose(W, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_rejects_a_single_donor():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    with pytest.raises(ValueError, match="at least two"):
        solve_simplex_loo_exact(np.ones((5, 1)))


def test_rejects_a_non_matrix():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    with pytest.raises(ValueError, match="2-D"):
        solve_simplex_loo_exact(np.ones(5))


@pytest.mark.parametrize("bad", [0, -1, 9])
def test_rejects_an_out_of_range_target_count(bad):
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(8), 10, 6)
    with pytest.raises(ValueError, match="n_targets"):
        solve_simplex_loo_exact(M, n_targets=bad)


def test_a_target_border_leaves_later_columns_as_donors_only():
    """A permutation placebo needs the treated unit in every pool without being
    a placebo itself, which is what ``n_targets`` short of ``J`` gives."""
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = _panel(np.random.default_rng(9), 14, 8)
    W = solve_simplex_loo_exact(M, n_targets=5)
    assert W.shape == (5, 8)
    np.testing.assert_allclose(W, _loop(M, n=5), atol=1e-7)


def test_identical_donors_still_returns_simplex_rows():
    from mlsynth.utils.bilevel.minnorm import solve_simplex_loo_exact

    M = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 4))
    W = solve_simplex_loo_exact(M)
    for j, w in enumerate(W):
        assert w[j] == 0.0
        assert abs(w.sum() - 1.0) < 1e-9
