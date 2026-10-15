"""Generative checks on the capped-weight water-filling.

:func:`cap_control_weights` is the one piece of the weight cap that is not a
constraint handed to a solver. The relaxed program rounds its continuous
selection to an integer design and renormalizes the control weights over the
units it did not treat, which can lift a weight back above a cap the program
respected, so the rounded column is water-filled back under the cap.

Four relations have to hold for the cap to mean anything after rounding: the
result is a weight vector (non-negative, summing to 1), nothing exceeds the
cap, nothing outside the support carries weight, and the operation is a no-op
on an input that already satisfies all three. The last is what makes the cap
composable with the exact program, where the weights arrive already feasible.

The helper is deliberately not the Euclidean projection onto the capped
simplex, so these assert feasibility and idempotence, not minimal distance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.marex_helpers.formulation import cap_control_weights

RAW = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
              allow_infinity=False),
    min_size=2, max_size=20,
)
CAPS = st.floats(min_value=0.05, max_value=1.0, allow_nan=False)


def _feasible(raw, cap):
    """A vector and a cap the cap can actually be met at."""
    v = np.asarray(raw, dtype=float)
    assume(v.sum() > 1e-6)
    assume(v.size * cap >= 1.0 + 1e-9)
    return v / v.sum()


@given(RAW, CAPS)
@settings(max_examples=400, deadline=None)
def test_the_result_is_a_weight_vector_under_the_cap(raw, cap):
    x = cap_control_weights(_feasible(raw, cap), cap)
    assert np.all(x >= 0.0)
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    assert x.max() <= cap + 1e-9


@given(RAW, CAPS)
@settings(max_examples=300, deadline=None)
def test_water_filling_is_idempotent(raw, cap):
    v = _feasible(raw, cap)
    once = cap_control_weights(v, cap)
    twice = cap_control_weights(once, cap)
    np.testing.assert_allclose(twice, once, atol=1e-12)


@given(RAW, CAPS)
@settings(max_examples=300, deadline=None)
def test_an_already_feasible_vector_is_returned_unchanged(raw, cap):
    v = _feasible(raw, cap)
    assume(v.max() <= cap - 1e-9)
    np.testing.assert_allclose(cap_control_weights(v, cap), v, atol=1e-12)


@given(RAW, CAPS, st.integers(min_value=1, max_value=8))
@settings(max_examples=300, deadline=None)
def test_nothing_outside_the_support_carries_weight(raw, cap, n_out):
    v = _feasible(raw, cap)
    mask = np.ones(v.size, dtype=bool)
    mask[:min(n_out, v.size - 1)] = False
    assume(int(mask.sum()) * cap >= 1.0 + 1e-9)
    x = cap_control_weights(v, cap, support=mask)
    assert np.all(x[~mask] == 0.0)
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    assert x.max() <= cap + 1e-9


@given(RAW, CAPS)
@settings(max_examples=300, deadline=None)
def test_the_order_of_the_weights_is_preserved(raw, cap):
    """Water-filling clips from the top and redistributes proportionally.

    So a market the relaxed solve liked more never ends up below one it liked
    less. Clipping alone could not promise this; proportional redistribution
    among the unclipped entries does.
    """
    v = _feasible(raw, cap)
    x = cap_control_weights(v, cap)
    for i in range(v.size):
        for j in range(v.size):
            if v[i] > v[j] + 1e-9:
                assert x[i] >= x[j] - 1e-9


@given(RAW, st.floats(min_value=0.02, max_value=0.4, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_a_support_too_small_for_the_cap_is_refused(raw, cap):
    v = np.asarray(raw, dtype=float)
    assume(v.sum() > 1e-6)
    assume(v.size * cap < 1.0 - 1e-6)
    with pytest.raises(MlsynthEstimationError, match="at least"):
        cap_control_weights(v / v.sum(), cap)


def test_an_all_zero_column_becomes_uniform_over_the_support():
    mask = np.array([True, True, True, True, False])
    x = cap_control_weights(np.zeros(5), 0.3, support=mask)
    np.testing.assert_allclose(x[mask], np.full(4, 0.25))
    assert x[4] == 0.0


def test_the_mass_of_a_treated_unit_is_redistributed_not_lost():
    """The case the relaxed path actually hits.

    The relaxed solve put 0.7 on a unit that rounding then selected treated.
    That unit leaves the support, and its mass goes to the controls under the
    cap instead of being dropped.
    """
    v = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    mask = np.array([False, True, True, True, True])
    x = cap_control_weights(v, 0.3, support=mask)
    assert x[0] == 0.0
    assert x.sum() == pytest.approx(1.0)
    assert x.max() <= 0.3 + 1e-12
