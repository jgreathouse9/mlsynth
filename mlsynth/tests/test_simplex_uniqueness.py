"""When a simplex least-squares optimum is unique, and how to know after solving.

Test-first (per ``agents/agents_tests.md``): written before
``simplex_optimum_is_unique`` exists.

Two exact solvers of the same program return the same weights when the optimum
is a point and different weights when it is a face. The library needed to know
which case it was in before batching a family of these, and the condition it
used -- full column rank of the design -- is sound but far stronger than
necessary. It rejects the ordinary synthetic-control shape, where donors
outnumber pre-treatment periods, even though those optima are almost always
unique.

Rank deficiency is only a precondition. The optimum moves along a direction
``d`` only if ``B d = 0`` and ``1' d = 0`` *and* ``d`` is feasible where the
solution sits -- which needs the support to be large relative to the design's
rank. Synthetic-control solutions are sparse, so it usually is not: on the
Proposition 99 placebo family every one of the 38 solves has a unique optimum
despite 37 donors against 19 pre-periods.

So the question is settled after solving, on the support, not before solving,
on the shape. These tests pin that condition against what the two solvers
actually do.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.minnorm import (
    gram_reduction_is_safe,
    simplex_gram,
    solve_simplex_minnorm,
)
from mlsynth.utils.bilevel.ridge_augment import simplex_qp


def _solvers_agree(B, A, atol=1e-6):
    """Do the design-form and Gram-form active sets return the same weights?"""
    w_design = simplex_qp(B, A)
    w_gram = solve_simplex_minnorm(simplex_gram(B, A))
    return float(np.abs(w_design - w_gram).max()) <= atol, w_design


def _in_hull_target(rng, m, J):
    """A target the donors reproduce exactly, so the optimal set is a face."""
    B = np.cumsum(rng.normal(size=(m, J)), axis=0) + 10.0
    return B, B @ rng.dirichlet(np.ones(J))


def _outside_hull_target(rng, m, J):
    """A target no convex combination reaches, so the optimum is a vertex."""
    B = np.cumsum(rng.normal(size=(m, J)), axis=0) + 10.0
    return B, B[:, 0] * 3.0 + 50.0


def _sparse_fit_target(rng, m, J):
    """The ordinary case: an imperfect fit off a handful of donors."""
    B = np.cumsum(rng.normal(size=(m, J)), axis=0) + 50.0
    return B[:, 1:], B[:, 0]


# --------------------------------------------------------------------------- #
# 1. The predicate agrees with what the solvers do
# --------------------------------------------------------------------------- #
def test_unique_when_the_target_is_outside_the_hull():
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(0)
    B, A = _outside_hull_target(rng, 8, 39)
    agree, w = _solvers_agree(B, A)
    assert simplex_optimum_is_unique(B, A, w) is True
    assert agree


def test_not_unique_when_the_target_is_inside_the_hull():
    """More donors than rows *and* a reachable target: the optimal set is a face
    and the two exact solvers land on different points of it."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(0)
    B, A = _in_hull_target(rng, 8, 39)
    agree, w = _solvers_agree(B, A)
    assert simplex_optimum_is_unique(B, A, w) is False
    assert not agree


@pytest.mark.parametrize("m,J", [(19, 37), (30, 119), (8, 60), (89, 48)])
def test_sparse_fits_are_unique_however_wide_the_donor_pool(m, J):
    """The shape the old guard rejected. A donor pool far wider than the
    pre-period still gives a unique optimum when the fit is imperfect, because
    the solution is sparse and no null direction is feasible at it."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(m * 100 + J)
    B, A = _sparse_fit_target(rng, m, J)
    agree, w = _solvers_agree(B, A)
    assert simplex_optimum_is_unique(B, A, w) is True
    assert agree


@pytest.mark.parametrize("seed", range(12))
def test_the_predicate_never_disagrees_with_the_solvers(seed):
    """Fuzz across regimes: whenever the predicate says unique, the two exact
    solvers must agree. That is the only direction that matters -- a false
    'unique' would silently change an answer."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(seed)
    m = int(rng.integers(4, 40))
    J = int(rng.integers(3, 60))
    kind = seed % 3
    if kind == 0:
        B, A = _in_hull_target(rng, m, J)
    elif kind == 1:
        B, A = _outside_hull_target(rng, m, J)
    else:
        B, A = _sparse_fit_target(rng, m, J + 1)
    agree, w = _solvers_agree(B, A)
    if simplex_optimum_is_unique(B, A, w):
        assert agree, "predicate said unique but the solvers disagree"


# --------------------------------------------------------------------------- #
# 2. It is strictly more permissive than the shape test, and never less safe
# --------------------------------------------------------------------------- #
def test_full_column_rank_designs_are_always_unique():
    """Anything the old shape guard admits, this admits too."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(3)
    for m, J in [(40, 20), (30, 12), (60, 25)]:
        B = rng.normal(size=(m, J))
        A = rng.normal(size=m)
        assert gram_reduction_is_safe(B)
        assert simplex_optimum_is_unique(B, A, simplex_qp(B, A)) is True


def test_it_admits_shapes_the_rank_test_rejects():
    """The point of the change: the ordinary synthetic-control geometry."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(4)
    B, A = _sparse_fit_target(rng, 19, 38)
    assert not gram_reduction_is_safe(B)          # the shape test says no
    assert simplex_optimum_is_unique(B, A, simplex_qp(B, A)) is True


# --------------------------------------------------------------------------- #
# 3. Edges
# --------------------------------------------------------------------------- #
def test_single_donor():
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    B = np.array([[2.0], [3.0]])
    assert simplex_optimum_is_unique(B, np.array([1.0, 1.0]), np.array([1.0])) is True


def test_duplicate_donors_carrying_weight_are_not_unique():
    """Two identical donors both in the support: weight can slide between them."""
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    rng = np.random.default_rng(5)
    B = rng.normal(size=(10, 3))
    B[:, 2] = B[:, 1]
    A = B[:, 1:].mean(axis=1)                     # reachable off the duplicates
    w = simplex_qp(B, A)
    if (w[1] > 1e-9) and (w[2] > 1e-9):
        assert simplex_optimum_is_unique(B, A, w) is False


def test_all_donors_identical():
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    B = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 4))
    A = B[:, 0].copy()
    w = simplex_qp(B, A)
    assert simplex_optimum_is_unique(B, A, w) is False


def test_rejects_mismatched_weights():
    from mlsynth.utils.bilevel.minnorm import simplex_optimum_is_unique

    with pytest.raises(ValueError):
        simplex_optimum_is_unique(np.ones((4, 3)), np.ones(4), np.ones(2))
