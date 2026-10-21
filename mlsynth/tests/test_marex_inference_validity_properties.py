"""Properties of MAREX's corrected inference, over its input domain.

Each property is an invariant the example tests check at one point:

* the blank floor is exactly where the split-conformal quantile starts to
  exist, at every level;
* the pointwise half-width is finite exactly at or above the floor, and is then
  one of the absolute blank effects;
* per-period p-values lie in ``[1 / (Tb + 1), 1]`` and do not rise with the size
  of the effect;
* the global p-value is a multiple of ``1 / |Pi|`` and at least that when exact,
  and at least ``1 / (1 + B)`` when drawn;
* the effective rank ignores the order of the units, their scale, and unit and
  period effects, and recovers an exact rank;
* the default blank window lies between the two floors, and is refused exactly
  when they do not fit in the pre-period.
"""
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.conformal.quantile import split_conformal_quantile
from mlsynth.utils.marex_helpers.inference import compute_inference
from mlsynth.utils.marex_helpers.windows import (default_blank_periods,
                                                 effective_rank,
                                                 min_blank_periods)

_EFFECTS = st.floats(min_value=-50, max_value=50, allow_nan=False,
                     allow_infinity=False)


@st.composite
def _pair(draw, max_tb=30, max_post=8):
    tb = draw(st.integers(0, max_tb))
    h = draw(st.integers(1, max_post))
    te = draw(st.integers(1, 5))
    T = te + tb + h
    y_t = draw(arrays(np.float64, T, elements=_EFFECTS))
    y_c = draw(arrays(np.float64, T, elements=_EFFECTS))
    return y_t, y_c, te + tb, te, tb, h


def _quiet(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_inference(*args, **kw)


@settings(max_examples=300, deadline=None)
@given(st.floats(min_value=0.005, max_value=0.95))
def test_the_floor_is_where_the_quantile_starts_to_exist(alpha):
    n = min_blank_periods(alpha)
    assert np.isfinite(split_conformal_quantile(np.arange(1.0, n + 1), alpha))
    assert np.isinf(split_conformal_quantile(np.arange(1.0, n), alpha))
    assert n == max(math.ceil(1 / alpha) - 1, 1) or abs(1 / alpha - round(1 / alpha)) < 1e-9


@settings(max_examples=150, deadline=None)
@given(_pair(), st.sampled_from([0.05, 0.1, 0.2, 0.3]))
def test_the_half_width_is_finite_exactly_at_the_floor(case, alpha):
    y_t, y_c, T0, te, tb, h = case
    inf = _quiet(y_t, y_c, T0, te, tb, alpha=alpha, random_state=0)
    if tb == 0:
        assert np.isnan(inf.ci[T0:]).all()
        return
    half = inf.ci[T0:, 1] - inf.treated_effects
    if tb >= min_blank_periods(alpha):
        assert np.all(np.isfinite(half))
        assert np.any(np.isclose(half[0], np.abs(inf.placebo_effects),
                                 rtol=1e-12, atol=1e-12))
    else:
        assert np.all(np.isinf(inf.ci[T0:]))


@settings(max_examples=150, deadline=None)
@given(_pair())
def test_per_period_p_values_are_bounded_and_monotone(case):
    y_t, y_c, T0, te, tb, h = case
    if tb == 0:
        return
    inf = _quiet(y_t, y_c, T0, te, tb, random_state=0)
    p = inf.per_period_pvals
    assert np.all(p >= 1 / (tb + 1)) and np.all(p <= 1)
    order = np.argsort(np.abs(inf.treated_effects), kind="stable")
    assert np.all(np.diff(p[order]) <= 1e-15)


@settings(max_examples=150, deadline=None)
@given(_pair(max_tb=8, max_post=4), st.integers(0, 2 ** 31 - 1))
def test_the_exact_global_p_value_is_on_the_arrangement_grid(case, seed):
    y_t, y_c, T0, te, tb, h = case
    n = math.comb(tb + h, h)
    inf = _quiet(y_t, y_c, T0, te, tb, random_state=seed,
                 max_combinations=max(n, 1))
    p = inf.global_p_value
    assert p >= 1 / n
    assert abs(p * n - round(p * n)) < 1e-9


@settings(max_examples=100, deadline=None)
@given(_pair(), st.integers(2, 60), st.integers(0, 2 ** 31 - 1))
def test_a_drawn_global_p_value_never_falls_below_one_over_b_plus_one(case, B,
                                                                       seed):
    y_t, y_c, T0, te, tb, h = case
    inf = _quiet(y_t, y_c, T0, te, tb, random_state=seed, max_combinations=B)
    assert inf.global_p_value >= 1 / (B + 1) - 1e-15
    assert inf.global_p_value <= 1


@st.composite
def _low_rank(draw):
    N = draw(st.integers(4, 25))
    T = draw(st.integers(4, 40))
    r = draw(st.integers(0, min(N, T) - 2))
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, r)) @ rng.normal(size=(r, T)) if r else np.zeros((N, T))
    return Y, r, rng


@settings(max_examples=200, deadline=None)
@given(_low_rank(), st.floats(min_value=1e-3, max_value=1e3))
def test_the_effective_rank_is_invariant_and_exact(case, c):
    Y, r, rng = case
    assert effective_rank(Y) == r
    shifted = c * Y + rng.normal(size=(Y.shape[0], 1)) + rng.normal(size=(1, Y.shape[1]))
    assert effective_rank(shifted) == r
    assert effective_rank(Y[rng.permutation(Y.shape[0])]) == r


@settings(max_examples=100, deadline=None)
@given(st.integers(12, 80), st.integers(0, 3), st.sampled_from([0.05, 0.1, 0.2, 0.5]),
       st.integers(0, 2 ** 31 - 1))
def test_the_default_lies_between_the_floors_or_is_refused(T0, r, alpha, seed):
    rng = np.random.default_rng(seed)
    N = 12
    Y = (5 * rng.normal(size=(N, r)) @ rng.normal(size=(r, T0)) if r
         else np.zeros((N, T0))) + 0.1 * rng.standard_normal((N, T0))
    tb, te = min_blank_periods(alpha), max(2 * effective_rank(Y), 10)
    if T0 < tb + te:
        with pytest.raises(MlsynthConfigError):
            default_blank_periods(Y, alpha)
    else:
        b = default_blank_periods(Y, alpha)
        assert tb <= b <= T0 - te


@settings(max_examples=200, deadline=None)
@given(_pair(), st.sampled_from([0.05, 0.1, 0.2, 0.3]))
def test_the_band_excludes_zero_exactly_when_the_p_value_is_at_most_alpha(case,
                                                                          alpha):
    """Split-conformal duality: the band and the per-period p-value agree."""
    y_t, y_c, T0, te, tb, h = case
    if tb == 0:
        return
    inf = _quiet(y_t, y_c, T0, te, tb, alpha=alpha, random_state=0)
    q = split_conformal_quantile(inf.placebo_effects, alpha)
    excluded = np.abs(inf.treated_effects) > q
    np.testing.assert_array_equal(excluded, inf.per_period_pvals <= alpha)
