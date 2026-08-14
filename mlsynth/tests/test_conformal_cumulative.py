"""Calibrated conformal interval for the cumulative (total) treatment effect.

The cumulative effect over an ``L``-period post-window is
``C = sum_k (y_{T0+k} - synthetic_{T0+k})``. mlsynth already reports ``C`` as a
point estimate (``effectutils.total_effect``), and ``fast_scm`` reports a "total
lift" band that is the per-period ATE interval rescaled by the horizon -- no
conformity score is ever formed on the sum. This suite pins the missing object: a
band calibrated *for the sum*.

Construction (``mlsynth.utils.conformal``):

* ``scores.rolling_origin_block_sums`` refits the control at sliding,
  NON-OVERLAPPING origins across the pre-period. Each origin contributes one
  conformity score: the sum of its ``L`` out-of-sample residuals. Because every
  refit sees only data strictly before its origin, the scores carry no in-sample
  optimism.
* ``cumulative.cumulative_conformal_interval`` centres those scores and takes the
  split-conformal half-width (``quantile.split_conformal_quantile``, the
  ``ceil((m+1)(1-alpha))``-th order statistic), giving ``C_hat +/- q``.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (
    CumulativeConformalBand,
    cumulative_conformal_interval,
    cumulative_conformal_from_refit,
    rolling_origin_block_sums,
    split_conformal_quantile,
)


# --- deterministic fixtures (no estimator dependency) -----------------------
def _ols_weight_fn(y, Y0):
    """Stand-in for an estimator's refit: unconstrained donor regression.

    Takes period indices, the contract ``debiased_sc_ttest`` already uses, so a
    covariate-aware estimator can subset its covariates by the same periods.
    """
    def _fn(keep_idx):
        idx = np.asarray(keep_idx)
        beta, *_ = np.linalg.lstsq(Y0[idx], y[idx], rcond=None)
        return np.asarray(beta, float)
    return _fn


def _panel(T=120, J=5, pre=100, seed=0, effect=0.0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, J))
    y = Y0 @ rng.dirichlet(np.ones(J)) + rng.normal(scale=0.1, size=T)
    y[pre:] += effect
    return y, Y0


def _fit(**kw):
    y, Y0 = kw.pop("panel", None) or _panel()
    base = dict(pre_periods=100, horizon=4, weight_fn=_ols_weight_fn(y, Y0),
                alpha=0.1, min_train_frac=0.2)
    base.update(kw)
    return cumulative_conformal_from_refit(y, Y0, **base)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_smoke_returns_band_of_the_declared_type():
    band = _fit()
    assert isinstance(band, CumulativeConformalBand)
    assert np.isfinite(band.point) and np.isfinite(band.half_width)
    assert band.lower <= band.point <= band.upper
    assert band.n_scores >= 9          # enough blocks for a finite 90% order statistic
    assert band.alpha == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Unit invariants
# ---------------------------------------------------------------------------

def test_point_is_the_summed_out_of_sample_gap_not_its_average():
    """Guards the estimand: the band is for the SUM, not the per-period mean."""
    y, Y0 = _panel()
    W = _ols_weight_fn(y, Y0)(np.arange(100))
    gap = y[100:104] - Y0[100:104] @ W
    band = _fit(panel=(y, Y0))
    assert band.point == pytest.approx(float(gap.sum()))
    assert band.point != pytest.approx(float(gap.mean()))


def test_band_is_symmetric_about_the_point():
    band = _fit()
    assert band.point - band.lower == pytest.approx(band.half_width)
    assert band.upper - band.point == pytest.approx(band.half_width)


def test_half_width_is_the_split_conformal_quantile_of_centred_block_sums():
    """The seam: scores from rolling origins, half-width from the shared primitive."""
    y, Y0 = _panel()
    scores = rolling_origin_block_sums(
        y, Y0, pre_periods=100, horizon=4, weight_fn=_ols_weight_fn(y, Y0), min_train_frac=0.2
    )
    expected = split_conformal_quantile(scores - scores.mean(), alpha=0.1)
    band = _fit(panel=(y, Y0))
    assert band.half_width == pytest.approx(expected)
    assert band.n_scores == scores.size


def test_scores_are_block_sums_computed_from_hand():
    """Pins the score's scale and construction without reusing the helper."""
    y, Y0 = _panel()
    expected = []
    for origin in range(20, 100 - 4 + 1, 4):
        w = _ols_weight_fn(y, Y0)(np.arange(origin))
        expected.append(float(np.sum(y[origin:origin + 4] - Y0[origin:origin + 4] @ w)))
    scores = rolling_origin_block_sums(
        y, Y0, pre_periods=100, horizon=4, weight_fn=_ols_weight_fn(y, Y0), min_train_frac=0.2
    )
    np.testing.assert_allclose(scores, np.asarray(expected), rtol=1e-9, atol=1e-12)


def test_scores_come_from_non_overlapping_origins():
    """Overlapping blocks would reuse periods and break exchangeability."""
    y, Y0 = _panel()
    scores = rolling_origin_block_sums(
        y, Y0, pre_periods=100, horizon=4, weight_fn=_ols_weight_fn(y, Y0), min_train_frac=0.2
    )
    start = max(4, int(100 * 0.2))
    expected_origins = len(range(start, 100 - 4 + 1, 4))   # stride == horizon
    assert scores.size == expected_origins


def test_scores_are_out_of_sample_each_refit_excludes_its_own_block():
    """A refit that saw its own block would score ~0; genuine OOS scores do not."""
    y, Y0 = _panel()
    scores = rolling_origin_block_sums(
        y, Y0, pre_periods=100, horizon=4, weight_fn=_ols_weight_fn(y, Y0), min_train_frac=0.2
    )
    in_sample = []
    for o in range(max(4, int(100 * 0.2)), 100 - 4 + 1, 4):
        W = _ols_weight_fn(y, Y0)(np.arange(o + 4))          # trained INCLUDING the block
        in_sample.append(float(np.sum(y[o:o + 4] - Y0[o:o + 4] @ W)))
    assert np.mean(np.abs(scores)) > np.mean(np.abs(in_sample))


def test_smaller_alpha_never_narrows_the_band():
    y, Y0 = _panel()
    assert _fit(panel=(y, Y0), alpha=0.05).half_width >= _fit(panel=(y, Y0), alpha=0.20).half_width


def test_pure_combiner_accepts_precomputed_scores():
    """PPSCM-style callers build scores their own way and reuse the combiner."""
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 0.5, -0.5, 1.5])
    band = cumulative_conformal_interval(point=4.0, scores=scores, alpha=0.1)
    assert band.point == pytest.approx(4.0)
    assert band.half_width == pytest.approx(
        split_conformal_quantile(scores - scores.mean(), alpha=0.1)
    )
    assert band.n_scores == scores.size


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_too_few_blocks_yields_an_honest_infinite_band():
    """A short pre-period cannot support the order statistic; say so, don't guess."""
    y, Y0 = _panel(T=60, J=4, pre=40)
    band = _fit(panel=(y, Y0), pre_periods=40, horizon=8, min_train_frac=0.3)
    assert band.n_scores < 9
    assert not np.isfinite(band.half_width)
    assert band.lower == -np.inf and band.upper == np.inf


def test_horizon_of_one_is_valid():
    band = _fit(horizon=1)
    assert np.isfinite(band.point) and band.n_scores >= 9


def test_no_origins_available_yields_infinite_band():
    band = _fit(min_train_frac=0.99)
    assert band.n_scores == 0
    assert not np.isfinite(band.half_width)


def test_single_donor_panel_is_supported():
    y, Y0 = _panel(J=1)
    band = _fit(panel=(y, Y0))
    assert np.isfinite(band.point)


# ---------------------------------------------------------------------------
# Failure -- translated errors, reported not swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_alpha_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(alpha=bad)


@pytest.mark.parametrize("bad", ["0.1", None, True])
def test_non_numeric_alpha_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(alpha=bad)


@pytest.mark.parametrize("bad", [0, -3])
def test_nonpositive_horizon_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(horizon=bad)


@pytest.mark.parametrize("bad", [4.0, "4", None, True])
def test_non_integer_horizon_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(horizon=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
def test_invalid_min_train_frac_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(min_train_frac=bad)


@pytest.mark.parametrize("bad", ["0.3", None, True])
def test_non_numeric_min_train_frac_raises_config_error(bad):
    with pytest.raises(MlsynthConfigError):
        _fit(min_train_frac=bad)


def test_nonpositive_pre_periods_raises_config_error():
    with pytest.raises(MlsynthConfigError):
        _fit(pre_periods=0)


def test_horizon_beyond_the_post_window_raises_data_error():
    y, Y0 = _panel(T=120)
    with pytest.raises(MlsynthDataError):
        _fit(panel=(y, Y0), pre_periods=118, horizon=5)      # 118 + 5 > 120


def test_donor_row_mismatch_raises_data_error():
    y, Y0 = _panel(T=120)
    with pytest.raises(MlsynthDataError):
        _fit(panel=(y[:-1], Y0))


def test_one_dimensional_donor_matrix_raises_data_error():
    y, Y0 = _panel(T=120)
    with pytest.raises(MlsynthDataError):
        _fit(panel=(y, Y0.ravel()))


def test_empty_scores_raise_data_error_in_the_pure_combiner():
    with pytest.raises(MlsynthDataError):
        cumulative_conformal_interval(point=1.0, scores=np.array([]), alpha=0.1)


def test_non_finite_scores_raise_data_error():
    with pytest.raises(MlsynthDataError):
        cumulative_conformal_interval(point=1.0, scores=np.array([1.0, np.nan, 2.0]), alpha=0.1)


# ---------------------------------------------------------------------------
# Move regression: the shared primitive keeps its behaviour and its old import
# ---------------------------------------------------------------------------

def test_split_conformal_quantile_is_reexported_from_inferutils():
    from mlsynth.utils.inferutils import split_conformal_quantile as legacy
    assert legacy is split_conformal_quantile


def test_split_conformal_quantile_behaviour_is_unchanged():
    r = np.array([-3.0, 1.0, -2.0, 0.5, 4.0])
    # k = ceil((5+1)*0.9) = 6 > n -> uninformative
    assert not np.isfinite(split_conformal_quantile(r, alpha=0.1))
    # k = ceil((5+1)*0.5) = 3 -> third smallest absolute value
    assert split_conformal_quantile(r, alpha=0.5) == pytest.approx(2.0)
    assert not np.isfinite(split_conformal_quantile(np.array([]), alpha=0.1))
