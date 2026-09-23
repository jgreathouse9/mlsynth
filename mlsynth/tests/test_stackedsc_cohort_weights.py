"""A cohort's weight programs, solved as one shared-design family.

Test-first (per ``agents/agents_tests.md``).

Every treated unit in a cohort faces the same donor block and differs only in
its own pre-treatment target, so the cohort is one design against many targets
and its Grams differ by rank-one pieces of ``A'B``. A donor predicate that binds
breaks that: each unit then gets its own design. The units sharing a pool are
still a family, so the loop becomes one batch per distinct pool.

What these tests pin is that the change is invisible in the output. STACKEDSC
reports per-unit weights and builds per-unit counterfactuals from them, and on
these designs the optimum is often a face, so an equally optimal but different
point is a changed answer. The batch has to return what the loop returned.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.stackedsc_helpers.pipeline import _weights_for_cohort


class _Cohort:
    def __init__(self, n_donors, n_units):
        self.donors = [f"d{k}" for k in range(n_donors)]
        self.units = [f"x{i}" for i in range(n_units)]


def _loop(cohort, A, B, allowed):
    """The incumbent one-at-a-time solve, which the batch must reproduce."""
    cols = []
    for j, keep in enumerate(allowed):
        w = np.zeros(len(cohort.donors))
        w[keep] = np.asarray(solve_simplex_qp(A[:, keep], B[:, j]),
                             dtype=float).ravel()
        cols.append(w)
    return np.column_stack(cols)


def _design(rng, m, J, k, in_hull=False):
    A = np.cumsum(rng.normal(size=(m, J)), axis=0) + 100.0
    if in_hull:
        B = A @ rng.dirichlet(np.ones(J), size=k).T
    else:
        B = np.cumsum(rng.normal(size=(m, k)), axis=0) + 100.0
    return A, B


# --------------------------------------------------------------------------- #
# 1. Shared pool: one batch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J,k", [(5, 39, 12), (10, 39, 4), (8, 20, 25), (7, 12, 1)])
def test_shared_pool_matches_the_loop(m, J, k):
    cohort = _Cohort(J, k)
    A, B = _design(np.random.default_rng(m * 31 + J), m, J, k)
    allowed = [list(range(J))] * k
    np.testing.assert_allclose(_weights_for_cohort(cohort, A, B, allowed),
                               _loop(cohort, A, B, allowed), atol=1e-7)


def test_shared_pool_with_a_face_for_an_optimum_matches_the_loop():
    """Targets the donors reproduce exactly: the minimisers are faces, so an
    equally optimal but different point is what the batch must not return."""
    cohort = _Cohort(30, 10)
    A, B = _design(np.random.default_rng(1), 6, 30, 10, in_hull=True)
    allowed = [list(range(30))] * 10
    np.testing.assert_allclose(_weights_for_cohort(cohort, A, B, allowed),
                               _loop(cohort, A, B, allowed), atol=1e-7)


def test_weights_are_on_the_simplex_and_shaped_by_donor_then_unit():
    cohort = _Cohort(18, 6)
    A, B = _design(np.random.default_rng(2), 9, 18, 6)
    W = _weights_for_cohort(cohort, A, B, [list(range(18))] * 6)
    assert W.shape == (18, 6)
    assert W.min() >= -1e-9
    np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 2. A binding predicate: one batch per distinct pool
# --------------------------------------------------------------------------- #
def test_a_binding_predicate_matches_the_loop():
    cohort = _Cohort(20, 8)
    A, B = _design(np.random.default_rng(3), 9, 20, 8)
    allowed = [[k for k in range(20) if k % 2 == j % 2] for j in range(8)]
    np.testing.assert_allclose(_weights_for_cohort(cohort, A, B, allowed),
                               _loop(cohort, A, B, allowed), atol=1e-7)


def test_excluded_donors_get_exactly_zero():
    cohort = _Cohort(15, 5)
    A, B = _design(np.random.default_rng(4), 8, 15, 5)
    allowed = [list(range(j + 2, 15)) for j in range(5)]
    W = _weights_for_cohort(cohort, A, B, allowed)
    for j, keep in enumerate(allowed):
        dropped = [k for k in range(15) if k not in keep]
        assert np.all(W[dropped, j] == 0.0)
        assert abs(W[:, j].sum() - 1.0) < 1e-9


def test_every_unit_with_its_own_pool_matches_the_loop():
    """No two units share a pool, so every group is a batch of one."""
    cohort = _Cohort(12, 6)
    A, B = _design(np.random.default_rng(5), 7, 12, 6)
    allowed = [[k for k in range(12) if k != j] for j in range(6)]
    np.testing.assert_allclose(_weights_for_cohort(cohort, A, B, allowed),
                               _loop(cohort, A, B, allowed), atol=1e-7)


def test_pools_are_grouped_by_content_not_by_position():
    """Units 0 and 2 share a pool while unit 1 does not; the grouping has to put
    each unit's weights back in its own column."""
    cohort = _Cohort(10, 3)
    A, B = _design(np.random.default_rng(6), 6, 10, 3)
    allowed = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
    W = _weights_for_cohort(cohort, A, B, allowed)
    np.testing.assert_allclose(W, _loop(cohort, A, B, allowed), atol=1e-7)
    assert np.all(W[5:, 0] == 0.0) and np.all(W[:5, 1] == 0.0)


# --------------------------------------------------------------------------- #
# 3. Edges
# --------------------------------------------------------------------------- #
def test_a_single_donor_takes_all_the_weight():
    cohort = _Cohort(3, 2)
    A, B = _design(np.random.default_rng(7), 5, 3, 2)
    W = _weights_for_cohort(cohort, A, B, [[1], [1]])
    np.testing.assert_allclose(W, np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))


def test_one_treated_unit():
    cohort = _Cohort(9, 1)
    A, B = _design(np.random.default_rng(8), 6, 9, 1)
    np.testing.assert_allclose(_weights_for_cohort(cohort, A, B, [list(range(9))]),
                               _loop(cohort, A, B, [list(range(9))]), atol=1e-7)
