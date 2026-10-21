"""Properties of MAREX's inversion band, over the shape of the design.

Each property is an invariant the example tests check at one design:

* the band is the library's band on the synthetic treated series and the donor
  outcomes, with the reference starting at the blank window and MAREX's own
  total as its point;
* shifting the post outcomes by ``d`` moves the point and both endpoints by
  ``H * d``;
* the band warns, and is unbounded, exactly when the reference is too short for
  its smallest p-value to reach ``alpha``.
"""
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import cumulative_conformal_by_inversion
from mlsynth.utils.marex_helpers.inference import compute_inference


@st.composite
def _design(draw):
    TE = draw(st.integers(4, 20))
    Tb = draw(st.integers(1, 24))
    H = draw(st.integers(1, 10))
    J = draw(st.integers(1, 6))
    seed = draw(st.integers(0, 2 ** 31 - 1))
    effect = draw(st.floats(-3.0, 3.0, allow_nan=False))
    rng = np.random.default_rng(seed)
    T = TE + Tb + H
    F = rng.normal(0, 1, (T, 2))
    lam = rng.uniform(0.2, 1.0, (J, 2))
    controls = lam @ F.T + 0.3 * rng.standard_normal((J, T))
    y_t = lam.mean(axis=0) @ F.T + 0.3 * rng.standard_normal(T)
    y_t[TE + Tb:] += effect
    return y_t, controls.mean(axis=0), controls, TE + Tb, TE, Tb


def _band(case, alpha=0.05, shift=0.0):
    y_t, y_c, controls, T0, te, tb = case
    y_t = y_t.copy()
    y_t[T0:] += shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_inference(y_t, y_c, T0, te, tb, alpha=alpha,
                                 cumulative_band=True,
                                 cumulative_method="inversion",
                                 control_outcomes=controls).cumulative_band


@settings(max_examples=40, deadline=None)
@given(_design())
def test_the_band_is_the_library_band_with_marex_point(case):
    y_t, y_c, controls, T0, te, tb = case
    band = _band(case)
    want = cumulative_conformal_by_inversion(
        y_t, controls.T, T0, alpha=0.05, refit="sc", conformal_type="block",
        statistic="signed", reference_start=te)
    assert (band.lower, band.upper) == (want.lower, want.upper)
    assert band.reference_start == te
    assert band.n_reference == tb + (y_t.size - T0)
    assert band.point == pytest.approx(float((y_t - y_c)[T0:].sum()),
                                       rel=1e-12, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(_design(), st.floats(-5.0, 5.0, allow_nan=False))
def test_a_post_shift_moves_everything_by_the_horizon_times_it(case, d):
    H = case[0].size - case[3]
    base, moved = _band(case), _band(case, shift=d)
    assert moved.point == pytest.approx(base.point + H * d, rel=1e-9, abs=1e-9)
    for a, b in ((base.lower, moved.lower), (base.upper, moved.upper)):
        if np.isfinite(a):
            assert b == pytest.approx(a + H * d, rel=1e-6, abs=1e-6)
        else:
            assert b == a


@settings(max_examples=40, deadline=None)
@given(_design(), st.sampled_from([0.01, 0.05, 0.1, 0.2]))
def test_it_warns_exactly_when_no_p_value_can_reach_alpha(case, alpha):
    y_t, y_c, controls, T0, te, tb = case
    n = tb + (y_t.size - T0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        band = compute_inference(y_t, y_c, T0, te, tb, alpha=alpha,
                                 cumulative_band=True,
                                 cumulative_method="inversion",
                                 control_outcomes=controls).cumulative_band
    warned = any("cannot reject" in str(w.message) for w in rec)
    assert warned == (1.0 / n > alpha)
    if warned:
        assert band.lower == -np.inf and band.upper == np.inf
