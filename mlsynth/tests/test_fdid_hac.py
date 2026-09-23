"""Serial-correlation-robust inference for FDID.

Li (2023)'s Proposition 2.1 standard error studentises by the residual's
marginal variance. The estimator's error is a difference of two block means,
whose variance is the long-run variance, so the formula is exact only when
the residual is serially uncorrelated -- the case Online Appendix A's
Assumptions 2(ii) and 3(i) impose. ``benchmarks/cases/fdid_serial_correlation_mc``
measures what that costs under the main text's weaker Assumption 2.1.

These tests cover the optional ``inference="hac"`` path that estimates the
autocovariances from the pre-period residuals and puts them through the exact
finite-block variance of both means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import FDID
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.fdid_helpers.config import FDIDConfig
from mlsynth.utils.fdid_helpers.inference import (
    block_mean_variance,
    did_inference,
    hac_lag,
    residual_autocovariances,
)
from mlsynth.utils.fdid_helpers.population import long_run_inflation
from mlsynth.utils.fdid_helpers.simulation import simulate_fdid_serial_sample


# ---------------------------------------------------------------------------
# residual_autocovariances
# ---------------------------------------------------------------------------

def test_autocovariances_lag_zero_is_the_centred_second_moment():
    v = np.array([1.0, -2.0, 3.0, 0.5, -1.5])
    g = residual_autocovariances(v, max_lag=0)
    assert g.shape == (1,)
    assert g[0] == pytest.approx(np.mean((v - v.mean()) ** 2))


def test_autocovariances_match_an_explicit_loop():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    g = residual_autocovariances(v, max_lag=6)
    vc = v - v.mean()
    for k in range(7):
        expected = sum(vc[t] * vc[t - k] for t in range(k, 50)) / 50
        assert g[k] == pytest.approx(expected)


def test_autocovariances_recover_the_ar1_decay():
    rho, n = 0.7, 200_000
    rng = np.random.default_rng(1)
    e = rng.normal(scale=np.sqrt(1 - rho ** 2), size=n)
    v = np.empty(n)
    v[0] = rng.normal()
    for t in range(1, n):
        v[t] = rho * v[t - 1] + e[t]
    g = residual_autocovariances(v, max_lag=3)
    assert g[0] == pytest.approx(1.0, abs=0.02)
    for k in (1, 2, 3):
        assert g[k] == pytest.approx(rho ** k, abs=0.02)


def test_autocovariances_clamp_the_lag_to_the_sample():
    v = np.arange(5.0)
    assert residual_autocovariances(v, max_lag=99).shape == (5,)


def test_autocovariances_reject_a_negative_lag():
    with pytest.raises(ValueError, match="max_lag"):
        residual_autocovariances(np.arange(5.0), max_lag=-1)


def test_autocovariances_reject_an_empty_sample():
    with pytest.raises(ValueError, match="at least one"):
        residual_autocovariances(np.array([]), max_lag=0)


# ---------------------------------------------------------------------------
# block_mean_variance
# ---------------------------------------------------------------------------

def _brute_force_block_variance(gamma: np.ndarray, T: int) -> float:
    """Var of a length-``T`` block mean by the full double sum."""
    total = 0.0
    for i in range(T):
        for j in range(T):
            k = abs(i - j)
            total += gamma[k] if k < gamma.size else 0.0
    return total / T ** 2


def test_block_variance_matches_the_double_sum():
    gamma = np.array([1.4, 0.6, 0.25, 0.1])
    for T in (1, 2, 4, 9, 30):
        assert block_mean_variance(gamma, T) == pytest.approx(
            _brute_force_block_variance(gamma, T)
        )


def test_block_variance_with_no_lags_is_the_iid_formula():
    gamma = np.array([2.5])
    for T in (1, 7, 100):
        assert block_mean_variance(gamma, T) == pytest.approx(2.5 / T)


def test_block_variance_grows_with_positive_dependence():
    iid = block_mean_variance(np.array([1.0]), 40)
    dependent = block_mean_variance(np.array([1.0, 0.5, 0.25]), 40)
    assert dependent > iid


def test_block_variance_is_floored_at_the_iid_value():
    """Negative autocovariance shrinks the truth; the estimator does not follow
    it down. Truncating an alternating series can drive the sum negative, and a
    negative variance is not a usable standard error."""
    gamma = np.array([1.0, -0.9, 0.1])
    assert block_mean_variance(gamma, 20) == pytest.approx(1.0 / 20)
    assert block_mean_variance(gamma, 20) >= 1.0 / 20


def test_block_variance_rejects_a_nonpositive_block():
    with pytest.raises(ValueError, match="block_length"):
        block_mean_variance(np.array([1.0]), 0)


def test_block_variance_rejects_an_empty_autocovariance_sequence():
    with pytest.raises(ValueError, match="gamma_0"):
        block_mean_variance(np.array([]), 10)


def test_block_variance_reproduces_the_population_inflation_factor():
    """The closed form the serial-correlation property case scores against.

    Feeding the AR(1) population autocovariances into the two block variances
    must return exactly what ``long_run_inflation`` predicts."""
    rho, n, T1, T2 = 0.7, 10, 400, 10
    gamma_0 = 1.0 + 1.0 / n
    gamma = np.array([gamma_0] + [rho ** k for k in range(1, max(T1, T2))])
    true = block_mean_variance(gamma, T1) + block_mean_variance(gamma, T2)
    reported = gamma_0 * (1.0 / T1 + 1.0 / T2)
    assert np.sqrt(true / reported) == pytest.approx(
        long_run_inflation(rho, n=n, T1=T1, T2=T2)
    )


# ---------------------------------------------------------------------------
# hac_lag
# ---------------------------------------------------------------------------

def test_hac_lag_never_exceeds_what_the_post_block_can_use():
    """Lag ``k`` enters a length-``T`` block mean with weight ``1 - k/T``, which
    is zero at ``k = T``. Nothing past ``T2 - 1`` can move the post block."""
    for T1 in (50, 100, 400, 2000):
        for T2 in (2, 5, 10, 40, 200):
            assert hac_lag(T1, T2) <= T2 - 1


def test_hac_lag_never_exceeds_what_the_pre_block_can_estimate():
    for T1 in (10, 50, 100, 400, 2000):
        for T2 in (2, 5, 10, 40, 200):
            assert hac_lag(T1, T2) <= T1 // 10


def test_hac_lag_is_nonnegative_on_the_shortest_panels():
    assert hac_lag(1, 1) == 0
    assert hac_lag(5, 2) == 0


def test_hac_lag_is_monotone_in_both_lengths():
    assert hac_lag(400, 40) >= hac_lag(400, 10)
    assert hac_lag(400, 40) >= hac_lag(100, 40)


def test_hac_lag_pins_the_measured_configurations():
    assert hac_lag(400, 10) == 9
    assert hac_lag(400, 40) == 39
    assert hac_lag(100, 10) == 9
    assert hac_lag(100, 100) == 10


# ---------------------------------------------------------------------------
# did_inference
# ---------------------------------------------------------------------------

def test_hac_at_zero_lag_reproduces_the_analytic_standard_error():
    """With no lags there is no long-run correction, so the two paths must give
    the same number to floating point. The residuals are centred because the
    difference-in-differences intercept is the pre-period mean difference, so
    the series the estimator actually feeds in has mean zero exactly."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=200)
    v -= v.mean()
    se_a, *_ = did_inference(1.0, v, 200, 10, method="analytic")
    se_h, *_ = did_inference(1.0, v, 200, 10, method="hac", lrvar_lag=0)
    assert se_h == pytest.approx(se_a)


def test_hac_on_white_noise_barely_moves_the_standard_error():
    rng = np.random.default_rng(3)
    v = rng.normal(size=400)
    se_a, *_ = did_inference(1.0, v, 400, 10, method="analytic")
    se_h, *_ = did_inference(1.0, v, 400, 10, method="hac")
    assert se_h == pytest.approx(se_a, rel=0.15)


def test_hac_widens_the_standard_error_under_positive_dependence():
    rho, n = 0.9, 400
    rng = np.random.default_rng(4)
    e = rng.normal(scale=np.sqrt(1 - rho ** 2), size=n)
    v = np.empty(n)
    v[0] = rng.normal()
    for t in range(1, n):
        v[t] = rho * v[t - 1] + e[t]
    se_a, *_ = did_inference(1.0, v, n, 10, method="analytic")
    se_h, *_ = did_inference(1.0, v, n, 10, method="hac")
    assert se_h > 1.5 * se_a


def test_hac_is_never_narrower_than_the_analytic_standard_error():
    """The floor in ``block_mean_variance`` guarantees this even when the
    residual is negatively autocorrelated. Centred, as the estimator's own
    difference-in-differences residual is by construction."""
    rng = np.random.default_rng(5)
    e = rng.normal(size=301)
    v = e[1:] - 0.95 * e[:-1]                 # MA(1), gamma_1 < 0
    v -= v.mean()
    se_a, *_ = did_inference(1.0, v, 300, 10, method="analytic")
    se_h, *_ = did_inference(1.0, v, 300, 10, method="hac")
    assert se_h >= se_a


def test_hac_respects_an_explicit_lag():
    rho, n = 0.8, 400
    rng = np.random.default_rng(6)
    e = rng.normal(scale=np.sqrt(1 - rho ** 2), size=n)
    v = np.empty(n)
    v[0] = rng.normal()
    for t in range(1, n):
        v[t] = rho * v[t - 1] + e[t]
    ses = [did_inference(1.0, v, n, 20, method="hac", lrvar_lag=L)[0]
           for L in (0, 1, 4, 12)]
    assert ses == sorted(ses)


def test_hac_leaves_the_point_estimate_and_its_sign_alone():
    rng = np.random.default_rng(7)
    v = rng.normal(size=100)
    _, ci_a, p_a, _ = did_inference(3.0, v, 100, 10, method="analytic")
    _, ci_h, p_h, _ = did_inference(3.0, v, 100, 10, method="hac")
    assert np.mean(ci_a) == pytest.approx(3.0)
    assert np.mean(ci_h) == pytest.approx(3.0)
    assert p_h >= p_a                          # a wider interval cannot sharpen


def test_hac_degenerate_periods_return_nan_like_the_analytic_path():
    se, ci, pval, satt = did_inference(2.0, np.array([0.0, 0.0]), 0, 0, method="hac")
    assert np.isnan(se) and np.isnan(pval) and np.isnan(satt)
    assert all(np.isnan(b) for b in ci)


def test_hac_on_two_pre_periods_has_no_lag_to_correct_with():
    """``hac_lag(2, 4)`` is zero, so the correction is a no-op and the two paths
    coincide."""
    v = np.array([1.5, -1.5])
    se_a, *_ = did_inference(1.0, v, 2, 4, method="analytic")
    se_h, *_ = did_inference(1.0, v, 2, 4, method="hac")
    assert hac_lag(2, 4) == 0
    assert se_h == pytest.approx(se_a)


def test_a_single_pre_period_leaves_the_standard_error_undefined():
    """One pre-period fits the intercept exactly, so the residual is zero and
    neither path has a variance to report."""
    for method in ("analytic", "hac"):
        se, ci, pval, satt = did_inference(1.0, np.array([0.0]), 1, 4, method=method)
        assert se == 0.0
        assert np.isnan(pval) and np.isnan(satt)


def test_did_inference_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        did_inference(1.0, np.ones(10), 10, 2, method="bootstrap")


def test_did_inference_rejects_a_negative_lag():
    with pytest.raises(ValueError, match="lrvar_lag"):
        did_inference(1.0, np.ones(10), 10, 2, method="hac", lrvar_lag=-2)


# ---------------------------------------------------------------------------
# FDIDConfig
# ---------------------------------------------------------------------------

@pytest.fixture
def serial_panel() -> pd.DataFrame:
    return simulate_fdid_serial_sample(
        rho=0.9, N=12, T1=60, T2=10, rng=np.random.default_rng(11)
    ).df


def _cfg(df: pd.DataFrame, **kw) -> dict:
    return {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "display_graphs": False, "verbose": False, **kw}


def test_config_defaults_to_the_analytic_standard_error(serial_panel):
    assert FDIDConfig(**_cfg(serial_panel)).inference == "analytic"


def test_config_accepts_hac(serial_panel):
    cfg = FDIDConfig(**_cfg(serial_panel, inference="hac", lrvar_lag=3))
    assert cfg.inference == "hac"
    assert cfg.lrvar_lag == 3


def test_config_rejects_an_unknown_inference_method(serial_panel):
    with pytest.raises(Exception):
        FDIDConfig(**_cfg(serial_panel, inference="wild-bootstrap"))


def test_config_rejects_a_negative_lag(serial_panel):
    with pytest.raises(Exception):
        FDIDConfig(**_cfg(serial_panel, lrvar_lag=-1, inference="hac"))


def test_config_rejects_a_lag_without_the_hac_method(serial_panel):
    """A lag under the analytic formula would be silently ignored."""
    with pytest.raises(MlsynthConfigError, match="lrvar_lag"):
        FDIDConfig(**_cfg(serial_panel, lrvar_lag=3))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_fit_with_hac_returns_a_finite_wider_interval(serial_panel):
    a = FDID(_cfg(serial_panel)).fit()
    h = FDID(_cfg(serial_panel, inference="hac")).fit()

    assert np.isfinite(h.att_se) and h.att_se > 0
    assert h.att == pytest.approx(a.att)                    # estimate unmoved
    assert h.att_se > a.att_se                              # interval widened
    assert h.att_ci[1] - h.att_ci[0] > a.att_ci[1] - a.att_ci[0]


def test_fit_with_hac_widens_both_reported_variants(serial_panel):
    a = FDID(_cfg(serial_panel)).fit()
    h = FDID(_cfg(serial_panel, inference="hac")).fit()
    for name in ("FDID", "DID"):
        assert h.methods[name].att == pytest.approx(a.methods[name].att)
        assert h.methods[name].att_se >= a.methods[name].att_se


def test_fit_records_which_standard_error_it_reported(serial_panel):
    a = FDID(_cfg(serial_panel)).fit()
    h = FDID(_cfg(serial_panel, inference="hac", lrvar_lag=5)).fit()
    assert "Li 2023" in a.inference.method
    assert "HAC" in h.inference.method
    assert "5" in h.inference.method
    assert h.fdid.lrvar_lag == 5


def test_hac_selects_the_same_donors_as_the_analytic_fit(serial_panel):
    """Inference is downstream of selection; the subset must not move."""
    a = FDID(_cfg(serial_panel)).fit()
    h = FDID(_cfg(serial_panel, inference="hac")).fit()
    assert h.fdid.selected_names == a.fdid.selected_names
