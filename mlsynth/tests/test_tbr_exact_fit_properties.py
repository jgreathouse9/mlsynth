"""Properties of the exact-fit rule (#1000), over the input domain.

* any treated aggregate that is an affine function of the control aggregate
  (nonzero or zero slope, any intercept, any scale) is an exact fit and is
  scored infeasible under every objective, with no exception raised;
* the rule is scale-free: rescaling both aggregates by one positive constant
  never changes whether a split fits exactly;
* a residual well above rounding is never flagged, and an infeasible key ranks
  below the feasible key of the same aggregates with that residual added.
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.tbr_helpers.design.objective import score_split
from mlsynth.utils.tbr_helpers.posterior import fit_pretest, fits_exactly

_SETTINGS = dict(max_examples=60, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow])
_OBJECTIVES = ("reference", "relative", "paper")


def _control(seed, n):
    rng = np.random.default_rng(seed)
    return 50.0 + rng.normal(size=n).cumsum() + 5.0 * rng.normal(size=n)


@settings(**_SETTINGS)
@given(seed=st.integers(0, 10_000), n=st.integers(20, 120),
       slope=st.one_of(st.just(0.0), st.floats(0.05, 20.0), st.floats(-20.0, -0.05)),
       intercept=st.floats(-1e4, 1e4), scale=st.sampled_from([1e-6, 1.0, 1e6]))
def test_every_affine_split_is_infeasible_under_every_objective(seed, n, slope,
                                                               intercept, scale):
    x = scale * _control(seed, n)
    y = slope * x + scale * intercept
    # the relative key needs a positive level; shift so it has one
    y = y - min(0.0, float(y.min())) + scale
    assert fits_exactly(fit_pretest(y, x))
    for objective in _OBJECTIVES:
        s = score_split(y, x, objective=objective, n_test=8)
        assert s.detail["exact_fit"] is True
        assert all(v == -np.inf for v in s.key)


@settings(**_SETTINGS)
@given(seed=st.integers(0, 10_000), n=st.integers(20, 120),
       noise=st.floats(1e-7, 10.0), k=st.integers(-8, 8))
def test_the_rule_is_scale_free_and_a_real_residual_is_never_flagged(seed, n,
                                                                     noise, k):
    x = _control(seed, n)
    e = np.random.default_rng(seed + 1).normal(size=n)
    y = 2.0 * x + 10.0 + noise * np.std(x) * e
    c = 10.0 ** k
    assert not fits_exactly(fit_pretest(y, x))
    assert not fits_exactly(fit_pretest(c * y, c * x))
    assert fits_exactly(fit_pretest(c * (2.0 * x + 10.0), c * x))
    for objective in _OBJECTIVES:
        feasible = score_split(y, x, objective=objective, n_test=8)
        exact = score_split(2.0 * x + 10.0, x, objective=objective, n_test=8)
        assert not feasible.detail.get("exact_fit", False)
        assert exact.key < feasible.key
