r"""Abadie and Zhao's Unit-level design, their equation (10), in Stage 1.

LEXSCM chooses treated units so the treated aggregate reproduces the population
target. That says nothing about whether each chosen unit is itself reproducible
by the donors left over, and the two come apart: on a 62-market panel at m = 8
the design selected the two largest markets, neither of which any convex
combination of the remainder can reach, gave them 58 per cent of the weight, and
the aggregate readout missed the truth. Capping eligible size -- a crude proxy
for the same constraint -- cut the aggregate half-width from 16.9 to 5.7 per cent
and restored coverage.

Equation (10) adds the missing term, ``xi * sum_j w_j ||x_j - sum_i v_ij x_i||^2``,
weighting each treated unit's own reproducibility by its share of the aggregate.
Two facts make it cheap. The inner weights ``v_.j`` appear in exactly one term
multiplied by the non-negative scalar ``w_j``, so the optimal ``v_.j`` does not
depend on ``w`` at all -- it is the ordinary synthetic control for unit ``j``
against the donors outside the tuple. And on the simplex a linear term folds
into the quadratic exactly, ``c'w == w'(c1' + 1c')w / 2``, so the existing
minimum-norm-point solver handles it unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.unit_level import (
    fold_linear_into_gram,
    per_unit_imbalance,
)


def _simplex_points(rng, m, n=40):
    w = rng.dirichlet(np.ones(m), size=n)
    return np.vstack([w, np.eye(m)])          # interior draws plus the vertices


# ------------------------------------------------- folding the linear term
def test_folding_returns_a_symmetric_gram_and_an_offset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4)); Q = X.T @ X
    M, kappa = fold_linear_into_gram(Q, np.array([1.0, 2.0, 0.5, 3.0]))
    assert M.shape == Q.shape
    assert np.allclose(M, M.T)
    assert np.isfinite(kappa)


def test_the_folded_form_equals_the_original_on_the_simplex():
    """The identity the whole reformulation rests on."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 5)); Q = X.T @ X
    c = rng.uniform(0.0, 4.0, 5)
    M, kappa = fold_linear_into_gram(Q, c)
    for w in _simplex_points(rng, 5):
        assert (w @ M @ w - kappa) == pytest.approx(w @ Q @ w + c @ w, rel=1e-10, abs=1e-10)


def test_the_folded_gram_is_positive_semidefinite():
    """The solver is a minimum-norm-point method, so an indefinite form would
    not merely be slower -- its active-set systems have no meaning."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        m = int(rng.integers(2, 9))
        X = rng.normal(size=(14, m)); Q = X.T @ X
        M, _ = fold_linear_into_gram(Q, rng.uniform(0.0, 5.0, m))
        assert np.linalg.eigvalsh(M).min() > -1e-9


def test_a_zero_penalty_leaves_the_gram_untouched():
    """xi = 0 has to recover the current objective exactly, since that is what
    keeps every pinned LEXSCM value unchanged."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 4)); Q = X.T @ X
    M, kappa = fold_linear_into_gram(Q, np.zeros(4))
    assert kappa == 0.0
    assert np.array_equal(M, Q)


def test_a_single_treated_unit_folds_to_the_scalar_case():
    M, kappa = fold_linear_into_gram(np.array([[2.0]]), np.array([3.0]))
    assert (1.0 * M[0, 0] - kappa) == pytest.approx(2.0 + 3.0)


def test_folding_rejects_a_mismatched_linear_term():
    with pytest.raises(MlsynthConfigError):
        fold_linear_into_gram(np.eye(3), np.array([1.0, 2.0]))


def test_folding_rejects_a_non_square_gram():
    with pytest.raises(MlsynthConfigError):
        fold_linear_into_gram(np.ones((3, 4)), np.array([1.0, 2.0, 3.0]))


def test_folding_rejects_non_finite_input():
    with pytest.raises(MlsynthDataError):
        fold_linear_into_gram(np.eye(2), np.array([1.0, np.inf]))


# ------------------------------------------------ per-unit reproducibility
def test_a_unit_inside_the_donor_hull_is_reproducible():
    """x_3 is built as a convex combination of the donors, so its own synthetic
    control reaches it and the penalty it contributes is ~0."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 6))
    X[:, 3] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    d = per_unit_imbalance(X, treated=[3], donors=[0, 1, 2, 4, 5])
    assert d[0] < 1e-6


def test_a_unit_outside_the_donor_hull_is_not():
    """Scaled beyond every donor, so no convex combination reaches it."""
    rng = np.random.default_rng(5)
    X = np.abs(rng.normal(size=(20, 5))) + 1.0
    X[:, 2] = X[:, [0, 1, 3, 4]].max(axis=1) * 3.0
    d = per_unit_imbalance(X, treated=[2], donors=[0, 1, 3, 4])
    assert d[0] > 1.0


def test_treated_units_are_not_donors_for_each_other():
    """Equation (10) sets v_ij = 0 for i in the treated set. Were unit 1 allowed
    to serve unit 0, a pair of near-identical treated markets would each look
    perfectly reproducible while the donors could reach neither."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(25, 6)) * 0.1
    X[:, 0] = 20.0 + rng.normal(size=25) * 0.01
    X[:, 1] = X[:, 0] + rng.normal(size=25) * 0.01      # a twin of unit 0
    d = per_unit_imbalance(X, treated=[0, 1], donors=[2, 3, 4, 5])
    assert d.min() > 1.0, "a treated twin was used as a donor"


def test_the_imbalance_is_returned_per_treated_unit_in_order():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(18, 7))
    d = per_unit_imbalance(X, treated=[1, 4, 6], donors=[0, 2, 3, 5])
    assert d.shape == (3,)
    assert np.all(np.isfinite(d))
    assert np.all(d >= 0.0)


def test_a_single_donor_is_the_only_available_combination():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(15, 3))
    d = per_unit_imbalance(X, treated=[0], donors=[2])
    assert d[0] == pytest.approx(float(np.sum((X[:, 0] - X[:, 2]) ** 2)), rel=1e-8)


def test_a_unit_listed_as_both_treated_and_donor_is_refused():
    """The guard that enforces v_ij = 0 for i in the treated set.

    Every other test here passes disjoint sets, so the guard was never
    exercised and the mutant that deleted it survived a first run. A unit
    allowed to donate to itself reproduces itself exactly, which would report
    the least approximable market in the panel as the most.
    """
    rng = np.random.default_rng(10)
    X = rng.normal(size=(15, 5))
    with pytest.raises(MlsynthConfigError, match="treated and donor"):
        per_unit_imbalance(X, treated=[0, 1], donors=[1, 2, 3])


def test_a_unit_donating_to_itself_would_look_perfectly_reproducible():
    """Why the guard earns its place: the refused call is not a formality.

    Fitted against a pool containing itself, a market is reproduced exactly and
    its penalty is zero, so the design would rank an unreachable market as its
    safest choice. Measured here by handing the same unit in as its own donor
    through the donors argument alone.
    """
    rng = np.random.default_rng(11)
    X = np.abs(rng.normal(size=(20, 5))) + 1.0
    X[:, 0] = X[:, 1:].max(axis=1) * 4.0          # far outside the donor hull
    honest = per_unit_imbalance(X, treated=[0], donors=[1, 2, 3, 4])
    with_self = per_unit_imbalance(X, treated=[], donors=[0, 1, 2, 3, 4])
    assert honest[0] > 1.0
    assert with_self.size == 0


def test_no_donors_left_is_reported_not_silently_zero():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(10, 2))
    with pytest.raises(MlsynthDataError):
        per_unit_imbalance(X, treated=[0, 1], donors=[])


# ------------------------------------------------- generative properties
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@given(st.integers(0, 2 ** 31 - 1), st.integers(2, 8), st.floats(0.0, 20.0))
@SETTINGS
def test_the_folding_identity_holds_over_the_whole_simplex(seed, m, scale):
    """``w'Mw - kappa == w'Qw + c'w`` for every point of the simplex, not only
    the fixture's. The identity is what licenses reusing a pure-quadratic solver
    for a penalised objective, so it has to hold over the domain."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(max(m + 2, 6), m))
    Q = X.T @ X
    c = rng.uniform(0.0, 1.0, m) * scale
    M, kappa = fold_linear_into_gram(Q, c)
    for w in _simplex_points(rng, m, n=12):
        assert (w @ M @ w - kappa) == pytest.approx(w @ Q @ w + c @ w,
                                                    rel=1e-9, abs=1e-9)


@given(st.integers(0, 2 ** 31 - 1), st.integers(2, 8))
@SETTINGS
def test_the_folded_form_is_always_usable_by_a_minimum_norm_solver(seed, m):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(max(m + 2, 6), m))
    M, _ = fold_linear_into_gram(X.T @ X, rng.uniform(0.0, 8.0, m))
    assert np.linalg.eigvalsh(M).min() > -1e-9


@given(st.integers(0, 2 ** 31 - 1), st.floats(0.25, 6.0))
@SETTINGS
def test_reproducibility_scales_with_the_square_of_the_outcome_scale(seed, k):
    """The penalty is a squared norm, so rescaling the panel rescales it by the
    square -- which is why xi has units and cannot be transferred between panels
    without thought."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 6))
    base = per_unit_imbalance(X, treated=[0, 1], donors=[2, 3, 4, 5])
    scaled = per_unit_imbalance(X * k, treated=[0, 1], donors=[2, 3, 4, 5])
    assert scaled == pytest.approx(base * k ** 2, rel=1e-6, abs=1e-9)


@given(st.integers(0, 2 ** 31 - 1))
@SETTINGS
def test_a_convex_combination_of_its_donors_is_always_reproducible(seed):
    """Build the treated unit inside the hull on purpose: whatever the donors
    are, a unit assembled from them has to come back at zero."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 5))
    a = rng.dirichlet(np.ones(4))
    target = X[:, 1:] @ a
    Y = np.column_stack([target, X[:, 1:]])
    d = per_unit_imbalance(Y, treated=[0], donors=[1, 2, 3, 4])
    assert d[0] < 1e-6
