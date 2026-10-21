"""Generative properties of the exact per-unit SYNDES backend (#806).

Over panels, sizes, scales and levels:

* the design is the optimum, against brute force over every treated set;
* scaling ``Y`` by ``s`` scales the objective by ``s^2`` and leaves the design
  optimal for the original panel (the default penalty scales with ``s^2``);
* a level common to every unit changes nothing, since each unit's fit is a
  simplex combination and the level cancels from its residual;
* relabelling the units relabels the design.

A design is compared by its value on the original panel, not by index set, so
a tie at rounding level does not read as a failure.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.optimization import estimate_lambda
from mlsynth.utils.syndes_helpers.per_unit_exact import (
    UnitFits,
    solve_per_unit_exact,
)


@st.composite
def cases(draw):
    N = draw(st.integers(min_value=3, max_value=7))
    T = draw(st.integers(min_value=3, max_value=12))
    K = draw(st.integers(min_value=1, max_value=N - 1))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2 ** 32 - 1)))
    Y = (rng.standard_normal((T, 2)) @ rng.standard_normal((2, N))
         + 0.4 * rng.standard_normal((T, N)))
    return Y, K


def _value(Y, S, lam):
    """``f(S)`` on ``Y`` from fresh fits."""
    fits = UnitFits(Y, lam)
    removed = frozenset(S)
    return sum(fits.phi(i, removed) for i in S) / len(S)


@settings(max_examples=60, deadline=None)
@given(cases())
def test_the_design_is_the_optimum_over_every_treated_set(case):
    Y, K = case
    lam = estimate_lambda(Y)
    best = min(_value(Y, S, lam) for S in combinations(range(Y.shape[1]), K))
    d = solve_per_unit_exact(Y, K)
    assert abs(d.objective_value - best) <= 1e-9 * best
    assert d.raw_results["certified"] is True


@settings(max_examples=40, deadline=None)
@given(cases(), st.integers(min_value=-6, max_value=6).map(lambda k: 10.0 ** k))
def test_a_scaled_outcome_gets_an_optimal_design(case, s):
    Y, K = case
    ref = solve_per_unit_exact(Y, K)
    got = solve_per_unit_exact(Y * s, K)
    assert abs(got.objective_value / s ** 2 - ref.objective_value) <= (
        1e-7 * ref.objective_value)
    assert _value(Y, tuple(got.selected_unit_indices), ref.lambda_value) <= (
        ref.objective_value * (1 + 1e-7))


@settings(max_examples=40, deadline=None)
@given(cases(), st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False))
def test_a_common_level_changes_neither_design_nor_objective(case, c):
    Y, K = case
    ref = solve_per_unit_exact(Y, K)
    got = solve_per_unit_exact(Y + c, K, lam=ref.lambda_value)
    assert abs(got.objective_value - ref.objective_value) <= 1e-6 * ref.objective_value
    assert _value(Y, tuple(got.selected_unit_indices), ref.lambda_value) <= (
        ref.objective_value * (1 + 1e-6))


@settings(max_examples=40, deadline=None)
@given(cases(), st.randoms(use_true_random=False))
def test_relabelling_the_units_relabels_the_design(case, rnd):
    Y, K = case
    N = Y.shape[1]
    order = list(range(N))
    rnd.shuffle(order)
    ref = solve_per_unit_exact(Y, K)
    got = solve_per_unit_exact(Y[:, order], K, lam=ref.lambda_value)
    assert abs(got.objective_value - ref.objective_value) <= 1e-9 * ref.objective_value
    back = tuple(sorted(order[i] for i in got.selected_unit_indices))
    assert _value(Y, back, ref.lambda_value) <= ref.objective_value * (1 + 1e-9)
