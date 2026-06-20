"""Tests for the HCW (Hsiao, Ching & Wan 2012) best-subset panel data approach.

The original HCW method constructs the treated unit's counterfactual by
*unrestricted OLS* (with intercept) on a best-subset-selected set of control
series, the subset chosen by a model-selection criterion (AICc by default,
matching the ``pampe`` R package and HCW Section 5).

The headline cross-validation is HCW (2012) Table XVI -- the Hong Kong
sovereignty study (estimation window 1993:Q1-1997:Q2, T0 = 18, ten candidate
controls). The published AICc model selects {Japan, Korea, Taiwan, USA} with
OLS weights (const 0.0263, Japan -0.676, Korea -0.4323, Taiwan 0.7926,
USA 0.486), R^2 = 0.9314, AICc = -171.771. We pin the port to those numbers.

Reference implementation: pampe (https://github.com/cran/pampe), which uses
``leaps::regsubsets`` best-subset + AICc + ``lm`` with intercept.
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PDA
from mlsynth.config_models import PDAConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError

from mlsynth.utils.pda_helpers.hcw.estimation import (
    best_subset_select,
    fit_hcw,
    info_criterion,
)
from mlsynth.utils.pda_helpers.hcw.inference import hcw_ate_inference


_HK = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")
_HCW_CANDS = ["China", "Indonesia", "Japan", "Korea", "Malaysia",
              "Philippines", "Singapore", "Taiwan", "Thailand", "United States"]


@pytest.fixture
def hk_table16():
    """HK sovereignty study at the HCW Table XVI setup: y, X, labels, T0=18."""
    d = pd.read_csv(os.path.abspath(_HK))
    W = d.pivot(index="Time", columns="Country", values="GDP").sort_index()
    y = W["Hong Kong"].to_numpy(dtype=float)
    X = W[_HCW_CANDS].to_numpy(dtype=float)
    return y, X, _HCW_CANDS, 18


# =========================================================================
# INFORMATION CRITERION
# =========================================================================

class TestInfoCriterion:

    def test_aicc_matches_pampe_convention(self):
        # K = donor regressors + intercept + error variance = p + 2.
        # Built so the HK Table XVI model (4 donors, n=18) lands on -171.771.
        n, p = 18, 4
        # Reconstruct the RSS that yields HCW's reported AICc.
        # AICc = n*log(rss/n) + 2K + 2K(K+1)/(n-K-1), K = p+2 = 6.
        K = p + 2
        target = -171.771
        base = target - (2 * K + 2 * K * (K + 1) / (n - K - 1))   # = n*log(rss/n)
        rss = n * np.exp(base / n)
        got = info_criterion(rss, n, p + 1, "AICc")   # n_regressors = p + intercept
        assert abs(got - target) < 1e-2

    def test_bic_penalises_more_than_aic_for_large_n(self):
        n, nreg = 50, 3
        rss = 10.0
        aic = info_criterion(rss, n, nreg, "AIC")
        bic = info_criterion(rss, n, nreg, "BIC")
        assert bic > aic   # log(n) > 2 for n = 50

    def test_unknown_criterion_raises(self):
        with pytest.raises(MlsynthEstimationError):
            info_criterion(1.0, 10, 2, "ZIGZAG")

    def test_perfect_fit_returns_neg_inf(self):
        # rss == 0 (perfect in-sample fit) -> -inf, the best possible score.
        assert info_criterion(0.0, 10, 2, "AIC") == -np.inf

    def test_aicc_undefined_correction_returns_inf(self):
        # When n - K - 1 <= 0 the small-sample correction is undefined; AICc
        # must return +inf so such an over-parameterised model is never chosen.
        # K = n_regressors + 1; pick n_regressors so K + 1 >= n.
        assert info_criterion(1.0, 6, 6, "AICc") == float("inf")


# =========================================================================
# BEST-SUBSET SELECTION
# =========================================================================

class TestBestSubset:

    def test_matches_bruteforce_oracle(self):
        rng = np.random.default_rng(0)
        T0, N = 25, 7
        X = rng.standard_normal((T0, N))
        y = X[:, [1, 4]] @ np.array([1.5, -2.0]) + 0.1 * rng.standard_normal(T0)
        got = best_subset_select(y, X, T0, criterion="AICc")

        # Brute force: best subset of each size by RSS, then min AICc over sizes.
        def rss_of(cols):
            Z = np.column_stack([np.ones(T0), X[:T0, cols]]) if cols else np.ones((T0, 1))
            coef, *_ = np.linalg.lstsq(Z, y[:T0], rcond=None)
            e = y[:T0] - Z @ coef
            return float(e @ e)
        best, best_ic = None, np.inf
        for r in range(0, N + 1):
            for combo in combinations(range(N), r):
                ic = info_criterion(rss_of(list(combo)), T0, len(combo) + 1, "AICc")
                if ic < best_ic:
                    best_ic, best = ic, list(combo)
        assert sorted(got) == sorted(best)

    def test_recovers_true_support(self):
        rng = np.random.default_rng(3)
        T0, N = 40, 6
        X = rng.standard_normal((T0, N))
        y = X[:, [0, 3]] @ np.array([2.0, 1.0]) + 0.05 * rng.standard_normal(T0)
        sel = best_subset_select(y, X, T0, criterion="AICc")
        assert set(sel) == {0, 3}

    def test_nvmax_caps_model_size(self):
        rng = np.random.default_rng(1)
        T0, N = 30, 8
        X = rng.standard_normal((T0, N))
        y = X @ rng.standard_normal(N) + 0.01 * rng.standard_normal(T0)
        sel = best_subset_select(y, X, T0, criterion="AIC", nvmax=2)
        assert len(sel) <= 2

    def test_unknown_criterion_raises(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        with pytest.raises(MlsynthEstimationError):
            best_subset_select(y, X, 20, criterion="ZIGZAG")

    def test_nvmax_below_one_raises(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        with pytest.raises(MlsynthEstimationError):
            best_subset_select(y, X, 20, criterion="AIC", nvmax=0)

    def test_pool_too_large_raises(self):
        # Exhaustive best subset is combinatorial; an oversized pool with a
        # high nvmax must raise a clear, translated error (not hang).
        rng = np.random.default_rng(2)
        T0, N = 60, 40
        X = rng.standard_normal((T0, N))
        y = rng.standard_normal(T0)
        with pytest.raises((MlsynthEstimationError, MlsynthConfigError)):
            best_subset_select(y, X, T0, criterion="AICc", nvmax=40)


# =========================================================================
# GOLD CROSS-VALIDATION: HCW (2012) TABLE XVI
# =========================================================================

class TestHCWTable16:

    def test_selects_japan_korea_taiwan_usa(self, hk_table16):
        y, X, labels, T0 = hk_table16
        sel = best_subset_select(y, X, T0, criterion="AICc")
        chosen = sorted(labels[i] for i in sel)
        assert chosen == ["Japan", "Korea", "Taiwan", "United States"]

    def test_ols_weights_match_table16(self, hk_table16):
        y, X, labels, T0 = hk_table16
        sel, beta_full, intercept, cf = fit_hcw(y, X, T0, criterion="AICc")
        w = {labels[i]: beta_full[i] for i in sel}
        assert abs(intercept - 0.0263) < 5e-4
        assert abs(w["Japan"] - (-0.676)) < 5e-4
        assert abs(w["Korea"] - (-0.4323)) < 5e-4
        assert abs(w["Taiwan"] - 0.7926) < 5e-4
        assert abs(w["United States"] - 0.486) < 5e-4

    def test_r2_and_aicc_match_table16(self, hk_table16):
        y, X, labels, T0 = hk_table16
        sel, beta_full, intercept, cf = fit_hcw(y, X, T0, criterion="AICc")
        resid = y[:T0] - cf[:T0]
        rss = float(resid @ resid)
        r2 = 1.0 - rss / float(np.sum((y[:T0] - y[:T0].mean()) ** 2))
        aicc = info_criterion(rss, T0, len(sel) + 1, "AICc")
        assert abs(r2 - 0.9314) < 1e-3
        assert abs(aicc - (-171.771)) < 0.1


# =========================================================================
# FIT CONTRACT
# =========================================================================

class TestFitHCW:

    def test_return_contract(self, hk_table16):
        y, X, labels, T0 = hk_table16
        sel, beta_full, intercept, cf = fit_hcw(y, X, T0, criterion="AICc")
        assert isinstance(sel, list)
        assert beta_full.shape == (X.shape[1],)
        assert isinstance(intercept, float)
        assert cf.shape == (X.shape[0],)
        # counterfactual is the OLS extrapolation X @ beta + intercept.
        np.testing.assert_allclose(cf, X @ beta_full + intercept, atol=1e-8)
        # off-support weights are exactly zero.
        off = [i for i in range(X.shape[1]) if i not in sel]
        np.testing.assert_array_equal(beta_full[off], 0.0)

    def test_single_donor(self):
        rng = np.random.default_rng(5)
        T0 = 20
        x = rng.standard_normal((30, 1))
        y = 2.0 * x[:, 0] + 0.01 * rng.standard_normal(30)
        sel, beta_full, intercept, cf = fit_hcw(y, x, T0, criterion="AICc")
        assert beta_full.shape == (1,)
        assert cf.shape == (30,)

    def test_intercept_only_when_donors_useless(self):
        # Nearly-constant treated series, donors uncorrelated with it: no donor
        # earns its AICc penalty, so the counterfactual is the pre-period mean.
        rng = np.random.default_rng(11)
        T0 = 15
        X = rng.standard_normal((25, 4))
        y = 5.0 + 0.001 * rng.standard_normal(25)
        sel, beta_full, intercept, cf = fit_hcw(y, X, T0, criterion="AICc")
        assert sel == []
        np.testing.assert_array_equal(beta_full, 0.0)
        assert abs(intercept - float(np.mean(y[:T0]))) < 1e-9
        np.testing.assert_allclose(cf, intercept, atol=1e-9)


# =========================================================================
# INFERENCE
# =========================================================================

class TestHCWInference:

    def test_ate_inference_contract(self, hk_table16):
        y, X, labels, T0 = hk_table16
        _, _, _, cf = fit_hcw(y, X, T0, criterion="AICc")
        att, se, ci, p = hcw_ate_inference(y, cf, T0, alpha=0.05)
        assert np.isfinite(att)
        assert se > 0
        assert ci[0] <= att <= ci[1]
        assert 0.0 <= p <= 1.0

    def test_ate_inference_fixed_lag_bartlett(self, hk_table16):
        # Supplying lrvar_lag switches to the fixed-lag Bartlett HAC (HCW's
        # Newey-West, Lemma 4) rather than the prewhitened default.
        y, X, labels, T0 = hk_table16
        _, _, _, cf = fit_hcw(y, X, T0, criterion="AICc")
        att, se, ci, p = hcw_ate_inference(y, cf, T0, alpha=0.05, lrvar_lag=2)
        assert np.isfinite(att)
        assert se > 0
        assert ci[0] <= att <= ci[1]


# =========================================================================
# CONFIG
# =========================================================================

class TestHCWConfig:

    def _df(self):
        return pd.read_csv(os.path.abspath(_HK))

    def test_method_hcw_accepted(self):
        cfg = PDAConfig(df=self._df(), outcome="GDP", treat="Integration",
                        unitid="Country", time="Time", method="hcw")
        assert cfg.method == "hcw"

    def test_default_criterion_is_aicc(self):
        cfg = PDAConfig(df=self._df(), outcome="GDP", treat="Integration",
                        unitid="Country", time="Time", method="hcw")
        assert cfg.hcw_criterion == "AICc"

    def test_invalid_criterion_rejected(self):
        with pytest.raises(Exception):
            PDAConfig(df=self._df(), outcome="GDP", treat="Integration",
                      unitid="Country", time="Time", method="hcw",
                      hcw_criterion="ZIGZAG")

    def test_invalid_nvmax_rejected(self):
        with pytest.raises(Exception):
            PDAConfig(df=self._df(), outcome="GDP", treat="Integration",
                      unitid="Country", time="Time", method="hcw", hcw_nvmax=0)


# =========================================================================
# ESTIMATOR SMOKE (public API)
# =========================================================================

class TestHCWEstimator:

    def _df_subset(self):
        # Restrict to the HCW candidate pool so best-subset is tractable.
        d = pd.read_csv(os.path.abspath(_HK))
        keep = ["Hong Kong"] + _HCW_CANDS
        return d[d["Country"].isin(keep)].copy()

    def test_fit_smoke(self):
        res = PDA({
            "df": self._df_subset(), "outcome": "GDP", "treat": "Integration",
            "unitid": "Country", "time": "Time", "method": "hcw",
            "display_graphs": False,
        }).fit()
        assert np.isfinite(res.att)
        assert res.inference.p_value is None or 0.0 <= res.inference.p_value <= 1.0

    def test_selected_donors_populated(self):
        res = PDA({
            "df": self._df_subset(), "outcome": "GDP", "treat": "Integration",
            "unitid": "Country", "time": "Time", "method": "hcw",
            "display_graphs": False,
        }).fit()
        # HCW selects a sparse donor set; it must be reported.
        sub = res.fits["hcw"]
        assert sub.selected_donors is not None
        assert len(sub.selected_donors) >= 1
