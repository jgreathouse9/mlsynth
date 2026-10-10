"""Generative checks on the contamination triage arithmetic.

The example-based tests in ``test_contamination.py`` pin each relation at a
fixture. These assert the same relations over the input domain, which is where
a weight vector that happens to be uniform, or an effect that happens to be
positive, stops standing in for the general case.

Four relations carry the module. The exposure a market has is the weight the
design gave it, so it lies in the unit interval. The estimation error an event
there produces is that weight times the event, with a sign flip, and it is
linear in the event. The breakdown event is the one that exactly consumes the
effect, so multiplying it back by the exposure returns the effect. And the
concentration summaries agree with each other: the Herfindahl index is the
reciprocal of the effective sample size, which is itself bounded above by the
number of markets carrying any weight.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.contamination import WEIGHT_TOL, contamination_report

WEIGHTS = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=24,
)
FINITE = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False,
                   allow_infinity=False)


def _simplex(raw):
    v = np.asarray(raw, dtype=float)
    total = v.sum()
    assume(total > 1e-6)
    return v / total


@given(WEIGHTS, st.integers(min_value=0, max_value=23))
@settings(max_examples=300, deadline=None)
def test_exposure_is_the_weight_and_lies_in_the_unit_interval(raw, k):
    v = _simplex(raw)
    k %= v.size
    rep = contamination_report(v, market=k)
    assert rep.exposure == v[k]
    assert 0.0 <= rep.exposure <= 1.0
    assert rep.carries_weight == (rep.exposure > WEIGHT_TOL)


@given(WEIGHTS, st.integers(min_value=0, max_value=23), FINITE)
@settings(max_examples=300, deadline=None)
def test_the_error_is_minus_the_weight_times_the_event(raw, k, shock):
    v = _simplex(raw)
    k %= v.size
    rep = contamination_report(v, market=k, shock=shock)
    assert rep.bias == -v[k] * shock


@given(WEIGHTS, st.integers(min_value=0, max_value=23), FINITE,
       st.floats(min_value=0.1, max_value=10.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_the_error_scales_linearly_with_the_event(raw, k, shock, c):
    v = _simplex(raw)
    k %= v.size
    one = contamination_report(v, market=k, shock=shock).bias
    scaled = contamination_report(v, market=k, shock=c * shock).bias
    assert scaled == np.float64(c * one) or abs(scaled - c * one) <= 1e-9 * (
        1.0 + abs(scaled))


@given(WEIGHTS, st.integers(min_value=0, max_value=23), FINITE)
@settings(max_examples=300, deadline=None)
def test_the_breakdown_event_consumes_exactly_the_effect(raw, k, att):
    v = _simplex(raw)
    k %= v.size
    rep = contamination_report(v, market=k, att=att)
    if not rep.carries_weight:
        assert rep.breakdown_shock == float("inf")
        return
    recovered = rep.breakdown_shock * rep.exposure
    assert abs(recovered - abs(att)) <= 1e-8 * (1.0 + abs(att))


@given(WEIGHTS, st.integers(min_value=0, max_value=23))
@settings(max_examples=300, deadline=None)
def test_the_concentration_summaries_agree(raw, k):
    v = _simplex(raw)
    k %= v.size
    rep = contamination_report(v, market=k)
    n_nz = rep.n_carrying_weight
    assert 1 <= n_nz <= v.size
    assert abs(rep.herfindahl - 1.0 / rep.effective_sample_size) <= 1e-12
    assert rep.effective_sample_size <= n_nz + 1e-9
    assert rep.max_weight >= 1.0 / n_nz - 1e-9
    assert rep.exposure <= rep.max_weight


@given(WEIGHTS, st.integers(min_value=0, max_value=23), FINITE, FINITE)
@settings(max_examples=200, deadline=None)
def test_a_market_with_no_weight_is_inert(raw, k, shock, att):
    v = _simplex(raw)
    k %= v.size
    v = v.copy()
    v[k] = 0.0
    total = v.sum()
    assume(total > 1e-6)
    v /= total
    rep = contamination_report(v, market=k, shock=shock, att=att)
    assert rep.exposure == 0.0
    assert rep.bias == 0.0
    assert rep.carries_weight is False
