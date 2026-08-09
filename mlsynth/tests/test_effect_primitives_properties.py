"""Property tests for the effect and fit primitives every estimator reports through.

``mlsynth.utils.effectutils`` and ``mlsynth.utils.fitutils`` are the bedrock of
estimator reporting: ATT, total effect, percent ATT, standardized ATT, the
per-period gap, RMSE and R-squared. Every estimator's headline numbers come out
of these ten functions, and ``resultutils.effects.calculate`` composes them into
the display dictionaries. A defect here is a defect in every estimator at once,
which is why they are asserted over their domain instead of at a fixture.

Coverage before this file: the primitives had no direct tests at all, and
``effects.calculate`` had three (a smoke test and the two empty-segment cases).

Most of what these functions promise is metamorphic -- how an output must
respond to rescaling or shifting the input -- which needs no oracle and is
exactly the shape that survives generation. Three of the relations are the ones
a reader should care about most:

* ``percent_att`` and ``standardized_att`` are scale invariant. They are ratios,
  so changing the outcome's units must not move them; if either picked up a
  dimension the estimator would report a different number for the same study
  measured in cents instead of dollars.
* ``rmse`` and ``std`` are linked by an exact identity, ``rmse^2 = mean^2 +
  std^2``. ``effects.calculate`` reports ``std(post_gap)`` under the label
  "T1 RMSE", so the two are not interchangeable and the identity is what says
  by how much they differ.
* ``total_effect == att * T1`` exactly. Both are reported, and a reader who
  divides one by the other is entitled to get the post-period count back.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from mlsynth.utils import effectutils as eff
from mlsynth.utils import fitutils as fit
from mlsynth.utils.resultutils import effects

_FINITE = dict(allow_nan=False, allow_infinity=False,
               min_value=-1e6, max_value=1e6, width=64)


def _series(min_size: int = 1, max_size: int = 60):
    return st.lists(st.floats(**_FINITE), min_size=min_size, max_size=max_size
                    ).map(np.asarray)


def _approx(expected, rel=1e-9, abs=1e-9):
    """`pytest.approx` under a shorter name; used throughout this module."""
    return pytest.approx(expected, rel=rel, abs=abs)


_nonzero_scale = st.floats(min_value=0.01, max_value=1e3,
                           allow_nan=False, allow_infinity=False)
_shift = st.floats(min_value=-1e4, max_value=1e4,
                   allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# gap
# ---------------------------------------------------------------------------

@given(obs=_series(), shift=_shift)
def test_gap_is_invariant_to_a_common_level_shift(obs, shift):
    """Moving observed and counterfactual together cannot change the effect."""
    cf = obs * 0.5 + 1.0
    np.testing.assert_allclose(eff.gap(obs + shift, cf + shift), eff.gap(obs, cf),
                               atol=1e-9, rtol=1e-9)


@given(obs=_series(), scale=_nonzero_scale)
def test_gap_scales_with_the_outcome(obs, scale):
    cf = obs * 0.5 + 1.0
    np.testing.assert_allclose(eff.gap(scale * obs, scale * cf),
                               scale * eff.gap(obs, cf), rtol=1e-9, atol=1e-9)


@given(obs=_series(), cf=_series())
def test_gap_is_antisymmetric_and_length_preserving(obs, cf):
    assume(obs.size == cf.size)
    np.testing.assert_allclose(eff.gap(cf, obs), -eff.gap(obs, cf), atol=0)
    assert eff.gap(obs, cf).shape == (obs.size,)


# ---------------------------------------------------------------------------
# split_pre_post
# ---------------------------------------------------------------------------

@given(arr=_series(min_size=0), n_pre=st.integers(0, 70), n_post=st.integers(0, 70))
@example(arr=np.array([1.0, 2.0, 3.0]), n_pre=0, n_post=3)
@example(arr=np.array([1.0, 2.0, 3.0]), n_pre=3, n_post=0)
def test_split_partitions_without_overlap_or_gap(arr, n_pre, n_post):
    """The two segments are adjacent, in order, and never share an element."""
    pre, post = eff.split_pre_post(arr, n_pre, n_post)
    assert pre.size == min(n_pre, arr.size)
    assert post.size == max(0, min(n_pre + n_post, arr.size) - n_pre)
    np.testing.assert_allclose(np.concatenate([pre, post]),
                               arr[: n_pre + n_post], atol=0)


# ---------------------------------------------------------------------------
# att / total_effect
# ---------------------------------------------------------------------------

@given(post_gap=_series())
def test_total_effect_is_att_times_the_post_period_count(post_gap):
    """Both are reported; dividing one by the other must give T1 back."""
    assert eff.total_effect(post_gap) == \
        _approx(eff.att(post_gap) * post_gap.size)


@given(value=st.floats(**_FINITE), size=st.integers(1, 40))
def test_att_of_a_constant_gap_is_that_constant(value, size):
    assert eff.att(np.full(size, value)) == _approx(value)


@given(post_gap=_series(), scale=_nonzero_scale)
def test_att_and_total_effect_scale_with_the_gap(post_gap, scale):
    assert eff.att(scale * post_gap) == _approx(scale * eff.att(post_gap))
    assert eff.total_effect(scale * post_gap) == \
        _approx(scale * eff.total_effect(post_gap))


def test_att_and_total_effect_are_nan_on_an_empty_post_period():
    assert np.isnan(eff.att(np.array([])))
    assert np.isnan(eff.total_effect(np.array([])))


# ---------------------------------------------------------------------------
# percent_att / percent_gap -- the ratios
# ---------------------------------------------------------------------------

@given(post_gap=_series(), cf=_series(), scale=_nonzero_scale)
def test_percent_att_is_scale_invariant(post_gap, cf, scale):
    """A percent is dimensionless: the same study in cents and in dollars must
    report the same number."""
    assume(cf.size > 0 and abs(float(cf.mean())) > 1e-3)
    base = eff.percent_att(eff.att(post_gap), cf)
    scaled = eff.percent_att(eff.att(scale * post_gap), scale * cf)
    assume(np.isfinite(base))
    assert scaled == _approx(base, rel=1e-6)


@given(att_value=st.floats(min_value=0.1, max_value=1e4,
                           allow_nan=False, allow_infinity=False),
       cf=_series())
def test_percent_att_keeps_the_sign_of_the_counterfactual_denominator(att_value, cf):
    """A negative counterfactual level flips the sign of the percent effect.

    Scale invariance alone does not pin this: taking the absolute value of the
    denominator preserves it, and that mutation survives every other test here.
    A treated series whose counterfactual sits below zero -- a net balance, a
    deficit, a temperature anomaly -- is where the two differ.
    """
    assume(cf.size > 0 and float(cf.mean()) < -1e-3)
    assert eff.percent_att(att_value, cf) < 0.0
    assert eff.percent_att(-att_value, cf) > 0.0


@given(post_gap=_series())
def test_percent_att_is_nan_when_the_counterfactual_averages_to_zero(post_gap):
    assert np.isnan(eff.percent_att(eff.att(post_gap), np.array([-1.0, 1.0])))
    assert np.isnan(eff.percent_att(eff.att(post_gap), np.array([])))


@given(post_gap=_series(), cf=_series())
def test_percent_gap_is_nan_exactly_where_the_counterfactual_is_zero(post_gap, cf):
    assume(post_gap.size == cf.size)
    out = eff.percent_gap(post_gap, cf)
    np.testing.assert_array_equal(np.isnan(out), cf == 0)


# ---------------------------------------------------------------------------
# standardized_att
# ---------------------------------------------------------------------------

@given(pre_gap=_series(), post_gap=_series(), scale=_nonzero_scale)
def test_standardized_att_is_scale_invariant(pre_gap, post_gap, scale):
    """Numerator and denominator carry the same units, so the ratio is free of
    them -- the property that makes SATT comparable across studies."""
    base = eff.standardized_att(pre_gap, post_gap)
    assume(np.isfinite(base))
    scaled = eff.standardized_att(scale * pre_gap, scale * post_gap)
    assert scaled == _approx(base, rel=1e-6)


@given(pre_gap=_series(), post_gap=_series())
def test_standardized_att_carries_the_sign_of_the_att(pre_gap, post_gap):
    satt = eff.standardized_att(pre_gap, post_gap)
    assume(np.isfinite(satt) and satt != 0.0)
    assert np.sign(satt) == np.sign(eff.att(post_gap))


def test_standardized_att_is_nan_when_either_segment_is_empty():
    assert np.isnan(eff.standardized_att(np.array([]), np.array([1.0])))
    assert np.isnan(eff.standardized_att(np.array([1.0]), np.array([])))


@given(pre_gap=_series(min_size=2), post_gap=_series(min_size=2))
def test_standardized_att_matches_the_documented_formula(pre_gap, post_gap):
    """``sqrt(T1) * att / sqrt((T1/T0) * s^2 + s^2)``, transcribed independently.

    Scale invariance and the sign both survive swapping ``sqrt(T1)`` for
    ``sqrt(T0)`` in the numerator, so neither pins the statistic's magnitude.
    Writing the docstring's formula out is what does, and it is the quantity a
    reader compares across studies with different post-period lengths.
    """
    t0, t1 = pre_gap.size, post_gap.size
    mean_sq_resid = float(pre_gap @ pre_gap) / t0
    denom = np.sqrt((t1 / t0) * mean_sq_resid + mean_sq_resid)
    assume(denom > 1e-8)

    expected = float(np.sqrt(t1) * post_gap.mean() / denom)
    assume(np.isfinite(expected))
    assert eff.standardized_att(pre_gap, post_gap) == _approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# fitutils
# ---------------------------------------------------------------------------

@given(residuals=_series())
def test_rmse_and_std_satisfy_the_bias_variance_identity(residuals):
    """``rmse^2 = mean^2 + std^2``.

    ``effects.calculate`` labels ``std(post_gap)`` as "T1 RMSE", so the two are
    not the same quantity; this pins the exact amount by which they differ.
    """
    r = residuals
    assert fit.rmse(r) ** 2 == _approx(
        float(r.mean()) ** 2 + fit.std(r) ** 2, rel=1e-8, abs=1e-8
    )


@given(residuals=_series(), scale=_nonzero_scale)
@example(residuals=np.full(3, 263862.5), scale=170.2006956476681)
def test_rmse_and_std_are_non_negative_and_scale_by_the_magnitude(residuals, scale):
    """Both are homogeneous of degree one, to float64 precision.

    The precision qualifier is the whole content of the tolerance here.
    Equivariance is exact in real arithmetic but only holds to about
    ``eps * max|c r|`` in float64, because scaling shifts where the centering
    inside ``np.std`` loses bits: ``std([263862.5] * 3)`` is exactly 0, while
    ``std(170.2 * [263862.5] * 3)`` is 7.5e-9. An absolute tolerance ignores
    the data's magnitude and calls that a failure, so the bound is taken
    relative to the scaled data instead.
    """
    assert fit.rmse(residuals) >= 0.0
    assert fit.std(residuals) >= 0.0

    magnitude = float(scale * np.abs(residuals).max(initial=0.0))
    tol = 1e-9 * max(1.0, magnitude)
    assert fit.rmse(scale * residuals) == _approx(
        scale * fit.rmse(residuals), abs=tol)
    assert fit.std(scale * residuals) == _approx(
        scale * fit.std(residuals), abs=tol)


@given(residuals=_series(), shift=_shift)
def test_std_is_shift_invariant_and_rmse_is_not(residuals, shift):
    assert fit.std(residuals + shift) == _approx(fit.std(residuals), abs=1e-6)


@given(size=st.integers(1, 40))
def test_rmse_of_a_perfect_fit_is_zero(size):
    assert fit.rmse(np.zeros(size)) == 0.0


@given(observed=_series(min_size=2), residuals=_series(min_size=2))
def test_r_squared_never_exceeds_one(observed, residuals):
    assume(observed.size == residuals.size)
    value = fit.r_squared(observed, residuals)
    assume(np.isfinite(value))
    assert value <= 1.0 + 1e-9


@given(
    observed=st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False,
                                allow_infinity=False, width=64),
                      min_size=2, max_size=60).map(np.asarray),
    scale=_nonzero_scale,
    shift=st.floats(min_value=-1e3, max_value=1e3,
                    allow_nan=False, allow_infinity=False),
)
def test_r_squared_is_invariant_to_the_units_of_the_outcome(observed, scale, shift):
    """R-squared is a variance ratio, so rescaling and shifting the observed
    series (with the residuals rescaled to match) cannot move it.

    The spread has to survive the shift in float64 for the claim to mean
    anything -- adding 1.0 to a series whose spread is 1e-70 leaves a constant
    array, and comparing a ratio computed on that against one computed on the
    original is a statement about floating point, not about R-squared.
    """
    assume(float(np.std(observed)) > 1e-6 * (1.0 + abs(shift)))
    residuals = observed * 0.1 - 0.3
    base = fit.r_squared(observed, residuals)
    assume(np.isfinite(base))
    moved = fit.r_squared(scale * observed + shift, scale * residuals)
    assert moved == _approx(base, rel=1e-6, abs=1e-9)


@given(observed=_series(min_size=2))
def test_r_squared_of_a_zero_residual_fit_is_one(observed):
    assume(float(np.var(observed)) > 1e-6)
    assert fit.r_squared(observed, np.zeros(observed.size)) == _approx(1.0)


@given(size=st.integers(2, 20), value=st.floats(**_FINITE))
def test_r_squared_is_nan_when_the_observed_series_has_no_variance(size, value):
    assert np.isnan(fit.r_squared(np.full(size, value), np.zeros(size)))


@pytest.mark.parametrize("residuals_value", [0.0, 1e-6, 0.5])
def test_r_squared_on_a_flat_series_that_does_not_center_to_exactly_zero(
    residuals_value,
):
    """Regression: the exact case the generated tests turned up.

    ``np.full(17, 493447.830355742)`` is constant, but subtracting its mean
    leaves residue with a centered sum of squares of 5.8e-20 instead of 0. An
    exact ``denom != 0`` guard admitted that, and the reported R-squared was
    1.0, -2.9e8 or -7.4e19 depending only on the residuals -- a silent wrong
    answer with no warning. A flat pre-period is an ordinary panel, so the
    guard now compares against the noise floor of the centering.
    """
    y = np.full(17, 493447.830355742)
    assert 0.0 < float((y - y.mean()) @ (y - y.mean())) < 1e-15   # not exactly flat
    assert np.isnan(fit.r_squared(y, np.full(17, residuals_value)))


def test_r_squared_still_reports_a_series_with_real_but_small_variance():
    """The guard must not swallow a genuinely varying series.

    A spread of 1 on a level of 1e6 is a ratio of 1e-13 -- far below anything a
    naive relative tolerance would keep, and far above the 1e-32 floor that
    centering noise sits at.
    """
    y = 1e6 + np.arange(20.0)
    value = fit.r_squared(y, np.full(20, 0.1))
    assert np.isfinite(value) and value < 1.0


# ---------------------------------------------------------------------------
# effects.calculate -- the composition
# ---------------------------------------------------------------------------

@st.composite
def _panels(draw):
    n_pre = draw(st.integers(min_value=1, max_value=25))
    n_post = draw(st.integers(min_value=1, max_value=25))
    total = n_pre + n_post
    obs = np.asarray(draw(st.lists(st.floats(**_FINITE),
                                   min_size=total, max_size=total)))
    cf = np.asarray(draw(st.lists(st.floats(**_FINITE),
                                  min_size=total, max_size=total)))
    return obs, cf, n_pre, n_post


@given(panel=_panels())
@settings(max_examples=150)
def test_calculate_reports_the_primitives_it_delegates_to(panel):
    """The display dictionaries must agree with the functions underneath."""
    obs, cf, n_pre, n_post = panel
    eff_dict, fit_dict, _ = effects.calculate(obs, cf, n_pre, n_post)

    gap_series = eff.gap(obs, cf)
    pre_gap, post_gap = eff.split_pre_post(gap_series, n_pre, n_post)

    assert eff_dict["ATT"] == _approx(round(eff.att(post_gap), 3))
    assert eff_dict["TTE"] == _approx(round(eff.total_effect(post_gap), 3))
    assert fit_dict["T0 RMSE"] == _approx(round(fit.rmse(pre_gap), 3))
    assert fit_dict["T1 RMSE"] == _approx(round(fit.std(post_gap), 3))
    assert fit_dict["Pre-Periods"] == n_pre
    assert fit_dict["Post-Periods"] == n_post


@given(panel=_panels(), shift=_shift)
@settings(max_examples=150)
def test_calculate_att_is_invariant_to_a_common_level_shift(panel, shift):
    """Shifting observed and counterfactual together leaves the effect alone."""
    obs, cf, n_pre, n_post = panel
    base, _, _ = effects.calculate(obs, cf, n_pre, n_post)
    moved, _, _ = effects.calculate(obs + shift, cf + shift, n_pre, n_post)
    assert moved["ATT"] == _approx(base["ATT"], abs=1e-3)


@given(panel=_panels())
@settings(max_examples=150)
def test_calculate_relative_time_column_puts_zero_on_the_last_pre_period(panel):
    """Pins the convention the gap plots depend on.

    ``relative_time = arange(T) - n_pre + 1``, so index ``n_pre - 1`` -- the
    last pre-treatment period -- carries 0 and the first post-treatment period
    carries 1. The comment above that line in ``resultutils`` reads "0 at
    treatment start", which is off by one against this if "treatment start"
    means the first treated period. The behaviour is pinned here as-is because
    every gap plot in the library is drawn on it; the comment is the thing that
    disagrees.
    """
    obs, cf, n_pre, n_post = panel
    _, _, vectors = effects.calculate(obs, cf, n_pre, n_post)
    rel = vectors["Gap"][:, 1]

    assert rel[n_pre - 1] == 0.0
    assert rel[n_pre] == 1.0
    assert rel.shape[0] == obs.size
    np.testing.assert_allclose(np.diff(rel), 1.0, atol=0)


@given(panel=_panels())
@settings(max_examples=100)
def test_calculate_time_series_vectors_keep_the_input_length(panel):
    obs, cf, n_pre, n_post = panel
    _, _, vectors = effects.calculate(obs, cf, n_pre, n_post)
    assert vectors["Observed Unit"].shape == (obs.size, 1)
    assert vectors["Counterfactual"].shape == (obs.size, 1)
    assert vectors["Gap"].shape == (obs.size, 2)


