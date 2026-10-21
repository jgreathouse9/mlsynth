"""Generative properties of the exact one-way SYNDES backend (#807).

Over panels, sizes, scales and levels:

* the ridge bound lies between the two-way bound and ``f(S)`` for every set;
* the design is the optimum, against brute force over every treated set;
* scaling ``Y`` by ``s`` scales the objective by ``s^2`` and leaves the design
  optimal for the original panel (the default penalty scales with ``s^2``);
* a level common to every unit changes nothing, since the treated average and
  the control combination both carry weights summing to one;
* relabelling the units relabels the design;
* a donor exclusion never lowers the optimum.

A design is compared by its value on the original panel, not by index set, so
a tie at rounding level does not read as a failure.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.gram import (BoundEngine, TwoWayProblem,
                                               signs_from_subsets)
from mlsynth.utils.syndes_helpers.one_way_exact import (
    OneWayFits,
    ridge_bound,
    solve_one_way_exact,
)
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


@st.composite
def cases(draw):
    N = draw(st.integers(min_value=2, max_value=7))
    T = draw(st.integers(min_value=3, max_value=12))
    K = draw(st.integers(min_value=1, max_value=N - 1))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2 ** 32 - 1)))
    Y = (rng.standard_normal((T, 2)) @ rng.standard_normal((2, N))
         + 0.4 * rng.standard_normal((T, N)))
    return Y, K


def _value(Y, S, lam, excluded=None):
    """``f(S)`` on ``Y`` from a fresh fit."""
    return OneWayFits(Y, lam, excluded).price(tuple(S))[0]


@settings(max_examples=60, deadline=None)
@given(cases())
def test_the_ridge_bound_lies_between_the_two_way_bound_and_the_value(case):
    Y, K = case
    N = Y.shape[1]
    lam = estimate_lambda(Y)
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))
    sets = np.array(list(combinations(range(N), K)))
    ridge = ridge_bound(engine, sets)
    two_way = engine.bound(signs_from_subsets(N, sets)).lower
    values = np.array([_value(Y, S, lam) for S in sets])
    assert np.all(ridge <= values * (1 + 1e-9))
    assert np.all(ridge >= two_way * (1 - 1e-9))


@settings(max_examples=60, deadline=None)
@given(cases())
def test_the_design_is_the_optimum_over_every_treated_set(case):
    Y, K = case
    lam = estimate_lambda(Y)
    best = min(_value(Y, S, lam) for S in combinations(range(Y.shape[1]), K))
    d = solve_one_way_exact(Y, K)
    assert abs(d.objective_value - best) <= 1e-9 * best
    assert d.raw_results["certified"] is True


@settings(max_examples=40, deadline=None)
@given(cases(), st.integers(min_value=-6, max_value=6).map(lambda k: 10.0 ** k))
def test_a_scaled_outcome_gets_an_optimal_design(case, s):
    Y, K = case
    ref = solve_one_way_exact(Y, K)
    got = solve_one_way_exact(Y * s, K)
    assert abs(got.objective_value / s ** 2 - ref.objective_value) <= (
        1e-7 * ref.objective_value)
    assert _value(Y, got.selected_unit_indices, ref.lambda_value) <= (
        ref.objective_value * (1 + 1e-7))


@settings(max_examples=40, deadline=None)
@given(cases(), st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False))
def test_a_common_level_changes_neither_design_nor_objective(case, c):
    Y, K = case
    ref = solve_one_way_exact(Y, K)
    got = solve_one_way_exact(Y + c, K, lam=ref.lambda_value)
    assert abs(got.objective_value - ref.objective_value) <= 1e-6 * ref.objective_value
    assert _value(Y, got.selected_unit_indices, ref.lambda_value) <= (
        ref.objective_value * (1 + 1e-6))


@settings(max_examples=40, deadline=None)
@given(cases(), st.randoms(use_true_random=False))
def test_relabelling_the_units_relabels_the_design(case, rnd):
    Y, K = case
    order = list(range(Y.shape[1]))
    rnd.shuffle(order)
    ref = solve_one_way_exact(Y, K)
    got = solve_one_way_exact(Y[:, order], K, lam=ref.lambda_value)
    assert abs(got.objective_value - ref.objective_value) <= 1e-9 * ref.objective_value
    back = [order[i] for i in got.selected_unit_indices]
    assert _value(Y, back, ref.lambda_value) <= ref.objective_value * (1 + 1e-9)


@settings(max_examples=40, deadline=None)
@given(cases(), st.data())
def test_a_donor_exclusion_never_lowers_the_optimum(case, data):
    Y, K = case
    N = Y.shape[1]
    i = data.draw(st.integers(min_value=0, max_value=N - 1))
    j = data.draw(st.integers(min_value=0, max_value=N - 1).filter(lambda v: v != i))
    # One pair empties only the pool of the set that leaves ``j`` its sole
    # control and treats ``i``; the set treating ``j`` stays admissible.
    ref = solve_one_way_exact(Y, K)
    got = solve_one_way_exact(Y, K, lam=ref.lambda_value,
                              restrictions=DesignRestrictions(donor_exclusion=[(i, j)]))
    assert got.objective_value >= ref.objective_value * (1 - 1e-12)
