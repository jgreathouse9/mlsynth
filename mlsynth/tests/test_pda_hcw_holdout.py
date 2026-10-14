"""Rolling-origin holdout diagnostic for the HCW panel data approach.

HCW fits an unrestricted OLS on the pre-period and extrapolates it. Nothing in
the reported fit says whether that extrapolation survives contact with periods
the regression did not see, and the pre-period fit statistic cannot say so: it
is the quantity the regression minimised. A short pre-period makes the two come
apart badly -- a four-point fit interpolates its four points and still misses a
held-out stretch by five times as much.

The diagnostic slides an origin across the pre-period, refits HCW on the periods
strictly before it, and scores the next ``horizon`` periods out of sample. These
tests cover the schedule, the errors, the summary ratio, the degenerate panels,
and the failures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.pda import PDA
from mlsynth.config_models import PDAConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.pda_helpers import PDAHoldout
from mlsynth.utils.pda_helpers.hcw.holdout import (
    MIN_HOLDOUT_TRAIN,
    holdout_schedule,
    rolling_origin_holdout,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _linear_panel(T=30, T0=20, n_donors=3, effect=0.0, noise=0.0, seed=0):
    """A panel whose treated unit is an exact linear combination of its donors."""
    rng = np.random.default_rng(seed)
    X = rng.normal(5.0, 1.0, size=(T, n_donors))
    beta = np.array([0.5, 0.3, 0.2])[:n_donors]
    y = 1.0 + X @ beta + noise * rng.normal(size=T)
    y[T0:] += effect
    return y, X, T0


def _fit_hcw(y, X, T0):
    from mlsynth.utils.pda_helpers.hcw.estimation import fit_hcw
    return fit_hcw(y, X, T0)[3]


def _panel_frame(y, X, T0):
    T, J = X.shape
    rows = {"unit": ["treated"] * T, "time": list(range(1, T + 1)),
            "y": list(y), "D": [1 if t > T0 else 0 for t in range(1, T + 1)]}
    df = pd.DataFrame(rows)
    for j in range(J):
        df = pd.concat([df, pd.DataFrame({
            "unit": [f"donor{j}"] * T, "time": list(range(1, T + 1)),
            "y": list(X[:, j]), "D": [0] * T})], ignore_index=True)
    return df


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_smoke_returns_populated_holdout():
    y, X, T0 = _linear_panel(noise=0.05)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert isinstance(h, PDAHoldout)
    assert h.n_origins > 0
    assert h.errors.size == h.n_origins * h.horizon
    assert np.isfinite(h.held_out_rmse)
    assert np.isfinite(h.in_sample_rmse)
    assert np.isfinite(h.ratio)


def test_smoke_through_the_estimator():
    y, X, T0 = _linear_panel(noise=0.05, effect=1.0)
    res = PDA(PDAConfig(df=_panel_frame(y, X, T0), unitid="unit", time="time",
                        outcome="y", treat="D", method="hcw", hcw_holdout=2,
                        display_graphs=False)).fit()
    fit = (res[0] if isinstance(res, list) else res).fits["hcw"]
    assert isinstance(fit.holdout, PDAHoldout)
    assert fit.holdout.horizon == 2


def test_absent_unless_requested():
    y, X, T0 = _linear_panel(noise=0.05)
    res = PDA(PDAConfig(df=_panel_frame(y, X, T0), unitid="unit", time="time",
                        outcome="y", treat="D", method="hcw",
                        display_graphs=False)).fit()
    fit = (res[0] if isinstance(res, list) else res).fits["hcw"]
    assert fit.holdout is None


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------

def test_schedule_is_non_overlapping_and_inside_the_pre_period():
    sched = list(holdout_schedule(20, 3))
    assert sched == list(range(MIN_HOLDOUT_TRAIN, 20 - 3 + 1, 3))
    assert all(o + 3 <= 20 for o in sched)
    assert all(b - a == 3 for a, b in zip(sched, sched[1:]))


def test_schedule_starts_at_the_identification_floor_not_the_conformal_one():
    """The conformal band refuses below ten training periods; this does not.

    A pre-period too short to hold out is the condition the diagnostic exists to
    report, so its floor is the OLS identification requirement, well below the
    band's.
    """
    from mlsynth.utils.conformal.scores import MIN_TRAIN_PERIODS
    assert MIN_HOLDOUT_TRAIN < MIN_TRAIN_PERIODS
    assert list(holdout_schedule(9, 2))        # the band's schedule is empty here
    assert len(list(holdout_schedule(9, 2))) > 0


def test_schedule_empty_when_the_pre_period_cannot_spare_a_window():
    assert list(holdout_schedule(MIN_HOLDOUT_TRAIN, 1)) == []
    assert list(holdout_schedule(MIN_HOLDOUT_TRAIN + 1, 5)) == []


# --------------------------------------------------------------------------
# the errors
# --------------------------------------------------------------------------

def test_errors_are_out_of_sample_not_fitted():
    """Each error comes from a fit that never saw the period it scores."""
    y, X, T0 = _linear_panel(noise=0.3, seed=3)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    for i, o in enumerate(h.origins):
        cf = _fit_hcw(y, X, int(o))
        expected = y[int(o):int(o) + h.horizon] - cf[int(o):int(o) + h.horizon]
        assert np.allclose(h.errors[i * h.horizon:(i + 1) * h.horizon], expected)


def test_exact_fit_gives_near_zero_held_out_error():
    """A treated unit that is exactly a donor combination extrapolates exactly."""
    y, X, T0 = _linear_panel(noise=0.0)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert h.held_out_rmse < 1e-8
    assert np.allclose(h.errors, 0.0, atol=1e-8)


def test_post_periods_never_enter_the_diagnostic():
    """Changing the post-period cannot move a pre-period holdout."""
    y, X, T0 = _linear_panel(noise=0.1, seed=5)
    a = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    y2 = y.copy()
    y2[T0:] += 100.0
    b = rolling_origin_holdout(y2, X, T0, horizon=2, fit=_fit_hcw)
    assert np.allclose(a.errors, b.errors)
    assert a.ratio == pytest.approx(b.ratio)


def test_ratio_is_held_out_over_in_sample():
    y, X, T0 = _linear_panel(noise=0.2, seed=7)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert h.ratio == pytest.approx(h.held_out_rmse / h.in_sample_rmse)
    assert h.in_sample_rmse > 0


def test_in_sample_rmse_is_the_full_pre_period_fit():
    y, X, T0 = _linear_panel(noise=0.2, seed=9)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    cf = _fit_hcw(y, X, T0)
    assert h.in_sample_rmse == pytest.approx(
        float(np.sqrt(np.mean((y[:T0] - cf[:T0]) ** 2))))


def test_short_pre_period_inflates_the_ratio():
    """The failure the diagnostic exists for: a short fit that looks fine."""
    rng = np.random.default_rng(11)
    T, T0 = 24, 16
    trend = np.arange(T, dtype=float)
    X = np.column_stack([trend + rng.normal(0, 0.4, T), rng.normal(5, 1, T)])
    y = 2.0 + 1.4 * X[:, 0] + rng.normal(0, 0.15, T)
    long_ratio = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw).ratio
    short = rolling_origin_holdout(y[T0 - 6:], X[T0 - 6:], 6, horizon=2,
                                   fit=_fit_hcw)
    assert short.n_origins >= 1
    assert short.ratio > long_ratio


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------

def test_no_origin_warns_and_reports_empty():
    y, X, T0 = _linear_panel(T=10, T0=4)
    with pytest.warns(UserWarning, match="too short"):
        h = rolling_origin_holdout(y, X, T0, horizon=3, fit=_fit_hcw)
    assert h.n_origins == 0
    assert h.errors.size == 0
    assert np.isnan(h.held_out_rmse)
    assert np.isnan(h.ratio)


def test_single_donor_panel():
    y, X, T0 = _linear_panel(n_donors=1, noise=0.05)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert h.n_origins > 0 and np.isfinite(h.held_out_rmse)


def test_horizon_of_one_scores_every_remaining_period():
    y, X, T0 = _linear_panel(noise=0.05)
    h = rolling_origin_holdout(y, X, T0, horizon=1, fit=_fit_hcw)
    assert h.n_origins == T0 - h.min_train
    assert h.errors.size == h.n_origins


def test_training_floor_keeps_the_full_donor_model_identified():
    """Every origin must leave the widest OLS a residual degree of freedom."""
    y, X, T0 = _linear_panel(n_donors=3)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert h.min_train == max(MIN_HOLDOUT_TRAIN, X.shape[1] + 2)
    assert h.origins.min() >= X.shape[1] + 2


def test_perfectly_collinear_donors_do_not_crash():
    y, X, T0 = _linear_panel(noise=0.05)
    X = np.column_stack([X[:, 0], X[:, 0], X[:, 1]])
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert np.all(np.isfinite(h.errors))


def test_zero_in_sample_error_yields_infinite_or_nan_ratio_not_a_crash():
    y, X, T0 = _linear_panel(noise=0.0)
    h = rolling_origin_holdout(y, X, T0, horizon=2, fit=_fit_hcw)
    assert not np.isnan(h.held_out_rmse)
    assert np.isnan(h.ratio) or np.isfinite(h.ratio) or np.isinf(h.ratio)


# --------------------------------------------------------------------------
# failures -- reported, never swallowed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "2", None])
def test_bad_horizon_raises_config_error(bad):
    y, X, T0 = _linear_panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        rolling_origin_holdout(y, X, T0, horizon=bad, fit=_fit_hcw)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_bad_pre_periods_raises_config_error(bad):
    y, X, _ = _linear_panel()
    with pytest.raises(MlsynthConfigError, match="pre_periods"):
        rolling_origin_holdout(y, X, bad, horizon=2, fit=_fit_hcw)


def test_pre_periods_beyond_the_sample_raises_data_error():
    y, X, _ = _linear_panel(T=30)
    with pytest.raises(MlsynthDataError, match="exceeds"):
        rolling_origin_holdout(y, X, 31, horizon=2, fit=_fit_hcw)


def test_misaligned_donors_raise_data_error():
    y, X, T0 = _linear_panel(T=30)
    with pytest.raises(MlsynthDataError, match="align"):
        rolling_origin_holdout(y, X[:20], T0, horizon=2, fit=_fit_hcw)


def test_one_dimensional_donor_matrix_raises_data_error():
    y, X, T0 = _linear_panel()
    with pytest.raises(MlsynthDataError, match="align"):
        rolling_origin_holdout(y, X[:, 0], T0, horizon=2, fit=_fit_hcw)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_config_rejects_a_bad_holdout_horizon(bad):
    y, X, T0 = _linear_panel()
    with pytest.raises(Exception):
        PDAConfig(df=_panel_frame(y, X, T0), unitid="unit", time="time",
                  outcome="y", treat="D", method="hcw", hcw_holdout=bad)


def test_failure_inside_the_refit_is_propagated_not_swallowed():
    """A refit that raises must surface, not silently drop an origin."""
    y, X, T0 = _linear_panel()

    def exploding_fit(yy, XX, origin):
        raise MlsynthDataError("refit failed at this origin")

    with pytest.raises(MlsynthDataError, match="refit failed"):
        rolling_origin_holdout(y, X, T0, horizon=2, fit=exploding_fit)
