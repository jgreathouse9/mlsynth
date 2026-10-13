"""Metamorphic invariants of the HCW rolling-origin holdout.

The unit suite pins the diagnostic on fixed panels. These assert what must hold
for every panel:

* structure -- one error per period per origin, and every origin leaves the
  widest donor model a residual degree of freedom;
* honesty -- every error comes from a refit that never saw the period it scores,
  so the series cannot drift back towards the in-sample fit;
* separation -- the post-period is not part of the diagnostic, so moving it
  cannot move a single error;
* equivariance -- rescaling the outcome rescales every error by the same factor
  and leaves the ratio alone, which is what makes the ratio readable across
  panels measured in different units;
* invariance -- shifting the outcome moves the intercept and nothing else;
* monotonicity -- a longer horizon never buys more origins.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.pda_helpers.hcw import MIN_HOLDOUT_TRAIN, holdout_schedule
from mlsynth.utils.pda_helpers.hcw.estimation import fit_hcw
from mlsynth.utils.pda_helpers.hcw.holdout import rolling_origin_holdout

_SETTINGS = settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def _fit(y, X, origin):
    return fit_hcw(y, X, int(origin))[3]


def _panel(seed, T0, n_post, n_donors, noise):
    rng = np.random.default_rng(seed)
    T = T0 + n_post
    X = rng.normal(5.0, 1.0, size=(T, n_donors))
    beta = rng.normal(0.4, 0.2, size=n_donors)
    y = 1.0 + X @ beta + noise * rng.normal(size=T)
    return y, X, T0


_PANELS = dict(
    seed=st.integers(0, 2**16),
    T0=st.integers(8, 16),
    n_post=st.integers(1, 5),
    n_donors=st.integers(1, 3),
    noise=st.floats(0.01, 0.5),
)


@_SETTINGS
@given(horizon=st.integers(1, 3), **_PANELS)
def test_structure_one_error_per_period_per_origin(horizon, **kw):
    y, X, T0 = _panel(**kw)
    h = rolling_origin_holdout(y, X, T0, horizon, _fit)
    assert h.errors.size == h.n_origins * h.horizon
    assert h.origins.size == h.n_origins
    if h.n_origins:
        assert h.origins.min() >= X.shape[1] + 2
        assert h.origins.max() + horizon <= T0


@_SETTINGS
@given(horizon=st.integers(1, 3), **_PANELS)
def test_every_error_is_out_of_sample(horizon, **kw):
    y, X, T0 = _panel(**kw)
    h = rolling_origin_holdout(y, X, T0, horizon, _fit)
    for i, o in enumerate(h.origins):
        cf = _fit(y, X, int(o))
        block = slice(int(o), int(o) + horizon)
        assert np.allclose(h.errors[i * horizon:(i + 1) * horizon],
                           y[block] - cf[block])


@_SETTINGS
@given(horizon=st.integers(1, 3), bump=st.floats(-50, 50), **_PANELS)
def test_the_post_period_is_not_part_of_the_diagnostic(horizon, bump, **kw):
    y, X, T0 = _panel(**kw)
    h = rolling_origin_holdout(y, X, T0, horizon, _fit)
    y2 = y.copy()
    y2[T0:] += bump
    h2 = rolling_origin_holdout(y2, X, T0, horizon, _fit)
    assert np.allclose(h.errors, h2.errors, atol=1e-8)
    assert np.allclose(h.in_sample_rmse, h2.in_sample_rmse, atol=1e-8)


@_SETTINGS
@given(horizon=st.integers(1, 3), scale=st.floats(0.25, 4.0), **_PANELS)
def test_rescaling_the_outcome_rescales_errors_and_leaves_the_ratio(
        horizon, scale, **kw):
    y, X, T0 = _panel(**kw)
    h = rolling_origin_holdout(y, X, T0, horizon, _fit)
    h2 = rolling_origin_holdout(scale * y, X, T0, horizon, _fit)
    if h.n_origins == 0:
        pytest.skip("no origin on this panel")
    assert np.allclose(h2.errors, scale * h.errors, atol=1e-6)
    assert h2.held_out_rmse == pytest.approx(scale * h.held_out_rmse, rel=1e-6)
    if np.isfinite(h.ratio):
        assert h2.ratio == pytest.approx(h.ratio, rel=1e-5)


@_SETTINGS
@given(horizon=st.integers(1, 3), shift=st.floats(-25, 25), **_PANELS)
def test_shifting_the_outcome_moves_the_intercept_and_nothing_else(
        horizon, shift, **kw):
    y, X, T0 = _panel(**kw)
    h = rolling_origin_holdout(y, X, T0, horizon, _fit)
    h2 = rolling_origin_holdout(y + shift, X, T0, horizon, _fit)
    assert np.allclose(h2.errors, h.errors, atol=1e-6)


@_SETTINGS
@given(T0=st.integers(8, 24), a=st.integers(1, 4), b=st.integers(1, 4),
       min_train=st.integers(MIN_HOLDOUT_TRAIN, 6))
def test_a_longer_horizon_never_buys_more_origins(T0, a, b, min_train):
    lo, hi = sorted((a, b))
    assert len(holdout_schedule(T0, hi, min_train)) \
        <= len(holdout_schedule(T0, lo, min_train))


@_SETTINGS
@given(T0=st.integers(4, 30), horizon=st.integers(1, 5),
       min_train=st.integers(MIN_HOLDOUT_TRAIN, 8))
def test_schedule_windows_never_overlap_and_stay_in_the_pre_period(
        T0, horizon, min_train):
    sched = list(holdout_schedule(T0, horizon, min_train))
    assert all(o >= min_train for o in sched)
    assert all(o + horizon <= T0 for o in sched)
    assert all(b >= a + horizon for a, b in zip(sched, sched[1:]))
