"""Generative properties: the two-way exact backend ignores units and level (#802).

Scaling ``Y`` by ``s`` scales the two-way objective by ``s^2``, and a level common
to every unit leaves it unchanged, because the objective reads only contrasts
whose weights sum to zero. The backend's design is then the same problem's
answer at every scale and level. Asserted over panels, sizes, scales from
``1e-8`` to ``1e8`` and levels up to ``1e8``, on the enumeration and on the
forced swap-search fallback.

The design under a transformation is checked as optimal for the original panel,
not as the same index set, so a tie between two designs at rounding level does
not read as a failure.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
from mlsynth.utils.syndes_helpers.gram import (
    BoundEngine,
    TwoWayProblem,
    maxcut_candidates,
)
from mlsynth.utils.syndes_helpers.partition import (
    solve_partition_batch,
    split_indices,
)


@st.composite
def cases(draw):
    """A factor panel with ``T > N`` and a treated-set size in range."""
    N = draw(st.integers(min_value=4, max_value=8))
    T = draw(st.integers(min_value=N + 2, max_value=N + 10))
    K = draw(st.integers(min_value=1, max_value=N - 1))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2 ** 32 - 1)))
    Y = (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
         + 0.35 * rng.standard_normal((T, N)))
    return Y, K


_scale = st.integers(min_value=-8, max_value=8).map(lambda k: 10.0 ** k)
_level = st.floats(min_value=-1e8, max_value=1e8,
                   allow_nan=False, allow_infinity=False)
_limit = st.sampled_from([20_000_000, 1])


def _value(Y, treated):
    """``f(S)`` on ``Y`` for one treated set, solved to the inner solver's limit."""
    problem = TwoWayProblem.from_panel(Y)
    t, c = split_indices(problem.N, np.asarray([sorted(treated)], dtype=np.int64))
    return float(solve_partition_batch(problem, t, c, max_iter=20_000,
                                       tol=1e-14).value[0])


@settings(max_examples=40, deadline=None)
@given(cases(), _scale, _limit)
def test_a_scaled_outcome_gets_the_same_design(case, s, limit):
    Y, K = case
    ref = solve_synthetic_design_exact(Y, K, candidate_limit=limit)
    got = solve_synthetic_design_exact(Y * s, K, candidate_limit=limit)
    assert abs(got.objective_value / s ** 2 - ref.objective_value) <= (
        1e-7 * ref.objective_value)
    assert _value(Y, got.selected_unit_indices) <= ref.objective_value * (1 + 1e-7)


@settings(max_examples=40, deadline=None)
@given(cases(), _level, _limit)
def test_a_common_level_changes_neither_design_nor_objective(case, c, limit):
    Y, K = case
    ref = solve_synthetic_design_exact(Y, K, candidate_limit=limit)
    got = solve_synthetic_design_exact(Y + c, K, candidate_limit=limit)
    assert abs(got.objective_value - ref.objective_value) <= 1e-6 * ref.objective_value
    assert _value(Y, got.selected_unit_indices) <= ref.objective_value * (1 + 1e-6)


@settings(max_examples=60, deadline=None)
@given(cases(), _scale)
def test_the_swap_search_makes_the_same_swaps_at_every_scale(case, s):
    Y, K = case
    ref = maxcut_candidates(BoundEngine.from_problem(TwoWayProblem.from_panel(Y)), K)
    got = maxcut_candidates(
        BoundEngine.from_problem(TwoWayProblem.from_panel(Y * s)), K)
    np.testing.assert_array_equal(got, ref)


@settings(max_examples=60, deadline=None)
@given(cases(), _level)
def test_the_gram_matrix_reads_contrasts_only(case, c):
    Y, _ = case
    T, N = Y.shape
    G = TwoWayProblem.from_panel(Y + c).G
    u = np.random.default_rng(N * T).standard_normal(N)
    u -= u.mean()
    raw = float(u @ (Y.T @ Y / T) @ u)
    assert abs(float(u @ G @ u) - raw) <= 1e-6 * raw
