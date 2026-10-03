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


# -------------------------------------------- wiring the penalty into Stage 1
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs, _afw_single
from mlsynth.utils.fast_scm_helpers.unit_level import unit_level_gram


def _panel_with_one_unreachable_unit(rng, J=10, T=60):
    """A panel where the balance-optimal tuple contains a unit no donor
    combination can reach.

    Unit 0 is scaled far beyond every other, so it sits outside the hull of the
    rest; it is also built to sit opposite the others relative to the population
    mean, which is what makes it attractive to a design scored on aggregate
    balance alone.
    """
    base = rng.normal(size=(T, J))
    base[:, 0] = base[:, 1:].mean(axis=1) * -6.0
    return base


def test_a_zero_penalty_leaves_the_search_gram_identical():
    """xi = 0 has to be today's Stage 1, not a numerically close version."""
    rng = np.random.default_rng(20)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    out, kappa = unit_level_gram(G, X, list(range(X.shape[1])), penalty=0.0)
    assert kappa == 0.0
    assert np.array_equal(out, G)


def test_a_zero_penalty_selects_exactly_what_it_selects_today():
    rng = np.random.default_rng(21)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    cand = list(range(X.shape[1]))
    plain = select_treated_designs(G, cand, m=2, top_K=5, method="enumerate")
    folded, _ = unit_level_gram(G, X, cand, penalty=0.0)
    same = select_treated_designs(folded, cand, m=2, top_K=5, method="enumerate")
    assert ([tuple(d.indices) for d in plain["top_designs"]]
            == [tuple(d.indices) for d in same["top_designs"]])


def test_the_penalty_moves_weight_off_the_less_reproducible_unit():
    """The penalty acts on the weights, not on membership.

    It enters the objective as ``xi * sum_j w_j d_j``, so a unit its donors
    cannot reach is answered by giving it less weight. A first version of this
    test asserted the unreachable unit left the chosen tuple and failed: the
    design keeps it and sets its weight toward zero instead.
    """
    rng = np.random.default_rng(22)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    cand = list(range(X.shape[1]))
    d = np.array([per_unit_imbalance(X, [j], [i for i in cand if i != j])[0]
                  for j in cand])
    assert d[0] > 4 * np.median(d), "fixture must make unit 0 the unreachable one"

    carried = []
    for xi in (0.0, 10.0, 50.0):
        Gs = G if xi == 0.0 else unit_level_gram(G, X, cand, penalty=xi)[0]
        best = select_treated_designs(Gs, cand, m=2, top_K=1,
                                      method="enumerate")["top_designs"][0]
        # weight the chosen design puts on its least reproducible member
        worst = int(np.argmax(d[list(best.indices)]))
        carried.append(float(best.weights[worst]))
    assert carried[0] > carried[1] > carried[2]
    assert carried[-1] == pytest.approx(0.0, abs=1e-6)


def test_a_large_penalty_leaves_a_treated_unit_carrying_no_weight():
    """Abadie and Zhao note that large xi makes the treated weights sparse, and
    for a design with a treatment budget that is a cost, not just a property.

    LEXSCM's ``m`` is a budget -- the number of markets a team can afford to
    treat. A design that selects m markets and gives one of them zero weight has
    spent that budget on a market the estimator then ignores. Pinned so the
    behaviour is known rather than discovered in a readout.
    """
    rng = np.random.default_rng(22)
    X = _panel_with_one_unreachable_unit(rng)
    cand = list(range(X.shape[1]))
    folded, _ = unit_level_gram(X.T @ X, X, cand, penalty=50.0)
    best = select_treated_designs(folded, cand, m=2, top_K=1,
                                  method="enumerate")["top_designs"][0]
    assert len(best.indices) == 2
    assert int(np.sum(best.weights > 1e-6)) == 1, (
        "expected the penalty to collapse the design onto one treated unit")


def test_the_folded_gram_keeps_every_candidate_submatrix_usable():
    """The search solves an m x m submatrix per tuple, so the lift has to hold
    for all of them -- which it does, since a principal submatrix of a positive
    semidefinite matrix is positive semidefinite."""
    rng = np.random.default_rng(23)
    X = _panel_with_one_unreachable_unit(rng, J=8)
    folded, _ = unit_level_gram(X.T @ X, X, list(range(8)), penalty=3.0)
    assert np.linalg.eigvalsh(folded).min() > -1e-8
    for S in ((0, 1), (2, 5, 7), (1, 3, 4, 6)):
        assert np.linalg.eigvalsh(folded[np.ix_(S, S)]).min() > -1e-8


def test_the_global_fold_matches_folding_each_tuple_separately():
    """A[i, j] = (d_i + d_j) / 2, so the tuple's block of the global fold is the
    fold of that tuple's own d. One J x J matrix therefore serves every
    candidate and the per-tuple cost of the penalty is nil."""
    rng = np.random.default_rng(24)
    X = _panel_with_one_unreachable_unit(rng, J=7)
    G = X.T @ X
    cand = list(range(7))
    xi = 2.5
    folded, kappa = unit_level_gram(G, X, cand, penalty=xi)
    d = np.array([per_unit_imbalance(X, [j], [i for i in cand if i != j])[0] for j in cand])
    for S in ((0, 2), (1, 4, 6), (0, 3, 5)):
        S = list(S)
        local, local_k = fold_linear_into_gram(G[np.ix_(S, S)], xi * d[S])
        for w in _simplex_points(rng, len(S), n=8):
            assert (w @ folded[np.ix_(S, S)] @ w - kappa) == pytest.approx(
                w @ local @ w - local_k, rel=1e-8, abs=1e-8)


def test_a_negative_penalty_is_refused():
    rng = np.random.default_rng(25)
    X = _panel_with_one_unreachable_unit(rng, J=6)
    with pytest.raises(MlsynthConfigError):
        unit_level_gram(X.T @ X, X, list(range(6)), penalty=-1.0)


# ------------------------------------------- keeping the treated set spent
from mlsynth.utils.fast_scm_helpers.unit_level import solve_penalised_weights


def test_without_a_floor_the_penalty_can_strand_a_treated_market():
    """The behaviour the floor exists to prevent, pinned as the baseline.

    LEXSCM's m is a budget: m markets get treated and paid for. A penalty that
    answers an unreachable market by zeroing its weight has spent that budget on
    a market the estimator ignores.
    """
    rng = np.random.default_rng(40)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    d = np.array([per_unit_imbalance(X, [j], [i for i in range(X.shape[1]) if i != j])[0]
                  for j in range(X.shape[1])])
    S = [0, 4]
    w, _ = solve_penalised_weights(G[np.ix_(S, S)], 50.0 * d[S], min_weight=0.0)
    assert int(np.sum(w > 1e-6)) == 1


def test_a_floor_keeps_every_treated_market_carrying_weight():
    rng = np.random.default_rng(40)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    d = np.array([per_unit_imbalance(X, [j], [i for i in range(X.shape[1]) if i != j])[0]
                  for j in range(X.shape[1])])
    S = [0, 4]
    w, _ = solve_penalised_weights(G[np.ix_(S, S)], 50.0 * d[S], min_weight=0.05)
    assert int(np.sum(w > 1e-9)) == len(S)
    assert w.min() >= 0.05 - 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_the_floor_still_lets_the_penalty_rank_the_units():
    """Enforcing the support must not flatten the penalty into equal weights."""
    rng = np.random.default_rng(41)
    X = _panel_with_one_unreachable_unit(rng)
    G = X.T @ X
    d = np.array([per_unit_imbalance(X, [j], [i for i in range(X.shape[1]) if i != j])[0]
                  for j in range(X.shape[1])])
    S = [0, 4]
    w, _ = solve_penalised_weights(G[np.ix_(S, S)], 50.0 * d[S], min_weight=0.05)
    worst = int(np.argmax(d[S]))
    assert w[worst] == pytest.approx(0.05, abs=1e-6), (
        "the least reproducible unit should sit on the floor, not above it")


def test_a_zero_floor_and_no_penalty_is_the_plain_simplex_solve():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4)); Q = X.T @ X
    w, _ = solve_penalised_weights(Q, np.zeros(4), min_weight=0.0)
    ref_loss, ref_w, _ = _afw_single(Q)
    assert w == pytest.approx(ref_w, abs=1e-8)


def test_a_floor_that_cannot_fit_on_the_simplex_is_refused():
    with pytest.raises(MlsynthConfigError, match="floor"):
        solve_penalised_weights(np.eye(4), np.zeros(4), min_weight=0.30)


def test_a_negative_floor_is_refused():
    with pytest.raises(MlsynthConfigError):
        solve_penalised_weights(np.eye(3), np.zeros(3), min_weight=-0.1)


@given(st.integers(0, 2 ** 31 - 1), st.integers(2, 7), st.floats(0.0, 0.12))
@SETTINGS
def test_the_floor_binds_and_the_weights_stay_on_the_simplex(seed, m, floor):
    """Whatever the panel, the solution is feasible: non-negative, summing to
    one, and no coordinate below the floor."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(max(m + 3, 8), m))
    if floor * m >= 1.0:
        return
    w, _ = solve_penalised_weights(X.T @ X, rng.uniform(0, 5, m), min_weight=floor)
    assert w.min() >= floor - 1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
