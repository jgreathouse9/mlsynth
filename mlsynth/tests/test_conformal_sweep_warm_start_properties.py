"""Metamorphic invariants of the chained conformal sweep.

The unit suite pins the sweep on fixed panels. These properties assert what has
to hold for every panel and every grid, and the first of them is the one the
whole optimisation rests on:

* equivalence -- the chained sweep reproduces one cold call per candidate
  exactly. A seed is a starting point for the active set, never a parameter of
  the estimand, so a discrepancy of any size means the optimisation changed an
  answer. Exactness is the right bar here and ``allclose`` is not: a conformal
  p-value is a rank among a finite reference set, so the smallest difference is
  a different rank and a moved interval;
* order invariance -- permuting the grid permutes the results and changes
  nothing else, since the sweep sorts internally but must report in the
  caller's order;
* seed irrelevance -- any feasible seed handed to a single call leaves its
  p-value alone, which is the same statement one candidate at a time;
* range -- every p-value is a share of a finite reference set, so it lies in
  ``(0, 1]`` and can never be zero: the identity permutation is always in the
  set and always ties with itself.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.bilevel.ridge_inference import (conformal_pvalue,
                                                   conformal_pvalue_sweep)

_SETTINGS = settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

_KW = dict(refit="sc", conformal_type="block", q=1.0)


def _panel(n_donors, pre, horizon, effect, seed):
    """A donor pool the treated unit sits inside, with a constant post effect."""
    rng = np.random.default_rng(seed)
    T = pre + horizon
    f = np.column_stack([np.cumsum(rng.normal(0, 1, T)) for _ in range(2)])
    lam = rng.dirichlet(np.ones(n_donors), size=2).T * 2.0
    Y0 = f @ lam.T + 0.5 * rng.standard_normal((T, n_donors))
    y = Y0 @ rng.dirichlet(np.ones(n_donors)) + 0.5 * rng.standard_normal(T)
    y[pre:] += effect
    return y, Y0, pre


def _unchained(y, Y0, pre, grid, **kw):
    out = np.empty(len(grid))
    for i, tau in enumerate(grid):
        shifted = y.copy()
        shifted[pre:] -= tau
        out[i] = conformal_pvalue(shifted, Y0, pre, **kw)
    return out


_PANELS = st.tuples(
    st.integers(min_value=3, max_value=12),      # donors
    st.integers(min_value=14, max_value=40),     # pre-periods
    st.integers(min_value=2, max_value=10),      # horizon
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=10_000),  # panel seed
)


@given(spec=_PANELS,
       n_grid=st.integers(min_value=2, max_value=9),
       lo=st.floats(min_value=-3.0, max_value=0.0, allow_nan=False),
       hi=st.floats(min_value=0.1, max_value=3.0, allow_nan=False))
@_SETTINGS
def test_the_chain_never_changes_a_p_value(spec, n_grid, lo, hi):
    y, Y0, pre = _panel(*spec)
    grid = np.linspace(lo, hi, n_grid)
    assert np.array_equal(conformal_pvalue_sweep(y, Y0, pre, grid, **_KW),
                          _unchained(y, Y0, pre, grid, **_KW))


@given(spec=_PANELS, n_grid=st.integers(min_value=2, max_value=7),
       shuffle_seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_permuting_the_grid_permutes_the_results(spec, n_grid, shuffle_seed):
    y, Y0, pre = _panel(*spec)
    grid = np.linspace(-1.5, 2.0, n_grid)
    order = np.random.default_rng(shuffle_seed).permutation(n_grid)
    straight = conformal_pvalue_sweep(y, Y0, pre, grid, **_KW)
    permuted = conformal_pvalue_sweep(y, Y0, pre, grid[order], **_KW)
    assert np.array_equal(permuted, straight[order])


@given(spec=_PANELS, seed_weight=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_any_feasible_seed_leaves_the_p_value_alone(spec, seed_weight):
    y, Y0, pre = _panel(*spec)
    cold = conformal_pvalue(y, Y0, pre, **_KW)
    w = np.random.default_rng(seed_weight).dirichlet(np.ones(Y0.shape[1]))
    assert conformal_pvalue(y, Y0, pre, warm_start=w, **_KW) == cold


@given(spec=_PANELS, n_grid=st.integers(min_value=1, max_value=7))
@_SETTINGS
def test_every_p_value_is_a_share_of_a_finite_reference_set(spec, n_grid):
    """It can reach 1 but never 0: the identity permutation is always in the
    reference set and always ties with the observed statistic."""
    y, Y0, pre = _panel(*spec)
    p = conformal_pvalue_sweep(y, Y0, pre, np.linspace(-1.0, 1.0, n_grid), **_KW)
    assert np.isfinite(p).all()
    assert (p > 0.0).all() and (p <= 1.0).all()
    n_ref = y.size
    assert np.allclose(p * n_ref, np.round(p * n_ref))
