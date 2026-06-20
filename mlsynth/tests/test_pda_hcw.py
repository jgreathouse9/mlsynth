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
    _best_subset_bnb,
    _best_subset_exhaustive,
    _best_subset_fw,
    _gram,
    _rss,
    _all_in_rss,
    _augmented,
    _subset_rss,
    _sweep,
    _unsweep,
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

class TestGramRSS:
    """The Gram-based per-subset RSS must equal the direct OLS RSS exactly.

    best_subset_select precomputes Z'Z, Z'y and y'y once and gets each subset's
    residual sum of squares from the relevant submatrix solve, instead of a
    fresh SVD per subset. This pins that fast path to the lstsq reference so the
    selection is provably unchanged -- only faster.
    """

    @pytest.mark.parametrize("seed", range(5))
    def test_subset_rss_matches_direct_lstsq(self, seed):
        rng = np.random.default_rng(seed)
        T0, N = 22, 6
        X = rng.standard_normal((T0, N))
        y = rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        ones = np.ones((T0, 1))
        subsets = [[], [0], [2, 4], [0, 1, 3, 5], list(range(N))]
        for cols in subsets:
            Z = np.column_stack([ones, X[:, cols]]) if cols else ones
            got = _subset_rss(G, Zty, yty, cols)
            ref = _rss(y, Z)
            assert abs(got - ref) < 1e-7 * max(ref, 1.0)

    def test_subset_rss_collinear_subset_is_finite(self):
        # A collinear donor block makes the Gram submatrix singular; the fast
        # path must fall back gracefully (no exception, finite RSS).
        rng = np.random.default_rng(1)
        T0 = 20
        x = rng.standard_normal((T0, 1))
        X = np.column_stack([x[:, 0], x[:, 0], rng.standard_normal(T0)])  # cols 0,1 identical
        y = rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        rss = _subset_rss(G, Zty, yty, [0, 1, 2])
        assert np.isfinite(rss)
        # Same fitted subspace as the full-rank {0, 2} design.
        ref = _rss(y, np.column_stack([np.ones(T0), X[:, [0, 2]]]))
        assert abs(rss - ref) < 1e-6 * max(ref, 1.0)


class TestBranchAndBound:
    """Branch-and-bound must return the *same* optimum as exhaustive search.

    The bound (min RSS = include all remaining donors; min penalty = stop now)
    is a true lower bound on the criterion over each subtree, so pruning never
    discards the optimum. We pin B&B to the exhaustive optimum across many
    random designs, sizes, and criteria -- both the selected set and the
    achieved criterion value.
    """

    @pytest.mark.parametrize("criterion", ["AICc", "AIC", "BIC"])
    @pytest.mark.parametrize("seed", range(12))
    def test_bnb_matches_exhaustive(self, criterion, seed):
        rng = np.random.default_rng(seed)
        T0 = int(rng.integers(14, 35))
        N = int(rng.integers(3, 11))
        X = rng.standard_normal((T0 + 6, N))
        # A genuine signal on a random sparse support, plus noise.
        k_true = int(rng.integers(1, min(N, 4) + 1))
        support = rng.choice(N, size=k_true, replace=False)
        y = X[:, support] @ rng.standard_normal(k_true) + 0.4 * rng.standard_normal(T0 + 6)

        G, Zty, yty = _gram(y[:T0], X[:T0])
        r_max = min(N, max(T0 - 2, 0))
        exhaustive = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, criterion)
        bnb = _best_subset_bnb(G, Zty, yty, N, T0, r_max, criterion)
        assert sorted(bnb) == sorted(exhaustive)

    @pytest.mark.parametrize("seed", range(6))
    def test_bnb_respects_nvmax(self, seed):
        rng = np.random.default_rng(50 + seed)
        T0, N = 30, 9
        X = rng.standard_normal((T0, N))
        y = X @ rng.standard_normal(N) + 0.1 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, 3, "AIC")
        bb = _best_subset_bnb(G, Zty, yty, N, T0, 3, "AIC")
        assert len(bb) <= 3
        assert sorted(bb) == sorted(ex)

    def test_public_select_matches_exhaustive_on_hk(self, hk_table16):
        # The public path (now B&B) must still select HCW's Table XVI set.
        y, X, labels, T0 = hk_table16
        G, Zty, yty = _gram(y[:T0], X[:T0])
        ex = _best_subset_exhaustive(G, Zty, yty, X.shape[1], T0,
                                     min(X.shape[1], T0 - 2), "AICc")
        got = best_subset_select(y, X, T0, criterion="AICc")
        assert sorted(got) == sorted(ex)


class TestSweepOperator:
    """The sweep operator is the FW regression engine.

    Sweeping the intercept and a donor subset into the augmented cross-product
    matrix must read off the OLS residual sum of squares in the y diagonal, and
    a forward sweep followed by a reverse sweep on the same pivot must restore
    the matrix exactly. These two invariants are what make the O(p^2) reversible
    add/remove descent of :func:`_best_subset_fw` correct.
    """

    def _augmented(self, y, X):
        # M = [1, X, y]' [1, X, y]: intercept at 0, donors at 1..N, y at N+1.
        N = X.shape[1]
        G, Zty, yty = _gram(y, X)
        M = np.empty((N + 2, N + 2))
        M[: N + 1, : N + 1] = G
        M[: N + 1, N + 1] = Zty
        M[N + 1, : N + 1] = Zty
        M[N + 1, N + 1] = yty
        return M, N

    @pytest.mark.parametrize("seed", range(5))
    def test_swept_y_diagonal_is_rss(self, seed):
        rng = np.random.default_rng(100 + seed)
        T0, N = 24, 6
        X = rng.standard_normal((T0, N))
        y = rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        for cols in ([], [0], [2, 4], [0, 1, 3, 5], list(range(N))):
            M, _ = self._augmented(y, X)
            _sweep(M, 0)                       # intercept always in
            for c in cols:
                _sweep(M, c + 1)
            got = M[N + 1, N + 1]
            ref = _subset_rss(G, Zty, yty, cols)
            assert abs(got - ref) < 1e-7 * max(ref, 1.0)

    def test_unsweep_restores_matrix(self):
        # A forward sweep then a reverse sweep on the same pivot is the identity
        # (the forward sweep alone is not -- sweeping twice negates the pivot
        # row/column, which is why removal must go through _unsweep).
        rng = np.random.default_rng(7)
        T0, N = 18, 4
        X = rng.standard_normal((T0, N))
        y = rng.standard_normal(T0)
        M, _ = self._augmented(y, X)
        original = M.copy()
        _sweep(M, 2)
        assert not np.allclose(M, original)    # actually changed something
        twice = M.copy()
        _sweep(M, 2)                           # forward twice != identity
        assert not np.allclose(M, original)
        M = twice.copy()
        _unsweep(M, 2)                         # reverse sweep undoes it
        np.testing.assert_allclose(M, original, atol=1e-9)

    def test_all_in_rss_matches_full_subset(self):
        # With the intercept and a chosen subset swept in, _all_in_rss must give
        # the RSS of chosen + every remaining donor (the subtree's RSS floor).
        rng = np.random.default_rng(13)
        T0, N = 26, 6
        X = rng.standard_normal((T0, N))
        y = X[:, [1, 4]] @ np.array([1.0, -0.5]) + 0.3 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        chosen, remaining = [0, 2], [1, 3, 4, 5]
        M = _augmented(G, Zty, yty, N)
        _sweep(M, 0)
        for c in chosen:
            _sweep(M, c + 1)
        cur = M[N + 1, N + 1]
        got = _all_in_rss(M, [r + 1 for r in remaining], N + 1, cur)
        ref = _subset_rss(G, Zty, yty, chosen + remaining)
        assert abs(got - ref) < 1e-6 * max(ref, 1.0)
        # Empty remainder: nothing to add, RSS is unchanged.
        assert _all_in_rss(M, [], N + 1, cur) == cur


class TestFurnivalWilson:
    """The Furnival-Wilson leaps-and-bounds search must equal exhaustive search.

    ``_best_subset_fw`` is the canonical regsubsets-style engine: a sweep-driven
    best-subset-of-each-size search whose size is then chosen by the information
    criterion (HCW Section 5 / pampe two-step). It is validated against the
    brute-force oracle across the same battery as branch-and-bound -- the IC
    optimum is always the best-RSS subset at its own size, so the per-size search
    plus IC selection returns the identical set.
    """

    @pytest.mark.parametrize("criterion", ["AICc", "AIC", "BIC"])
    @pytest.mark.parametrize("seed", range(12))
    def test_fw_matches_exhaustive(self, criterion, seed):
        rng = np.random.default_rng(seed)
        T0 = int(rng.integers(14, 35))
        N = int(rng.integers(3, 11))
        X = rng.standard_normal((T0 + 6, N))
        k_true = int(rng.integers(1, min(N, 4) + 1))
        support = rng.choice(N, size=k_true, replace=False)
        y = X[:, support] @ rng.standard_normal(k_true) + 0.4 * rng.standard_normal(T0 + 6)

        G, Zty, yty = _gram(y[:T0], X[:T0])
        r_max = min(N, max(T0 - 2, 0))
        exhaustive = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, criterion)
        fw = _best_subset_fw(G, Zty, yty, N, T0, r_max, criterion)
        assert sorted(fw) == sorted(exhaustive)

    @pytest.mark.parametrize("criterion", ["AICc", "AIC", "BIC"])
    @pytest.mark.parametrize("seed", range(8))
    def test_fw_agrees_with_bnb(self, criterion, seed):
        # Two independent exact searches must always concur.
        rng = np.random.default_rng(200 + seed)
        T0 = int(rng.integers(16, 32))
        N = int(rng.integers(3, 10))
        X = rng.standard_normal((T0, N))
        y = X @ rng.standard_normal(N) + 0.3 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        r_max = min(N, max(T0 - 2, 0))
        bnb = _best_subset_bnb(G, Zty, yty, N, T0, r_max, criterion)
        fw = _best_subset_fw(G, Zty, yty, N, T0, r_max, criterion)
        assert sorted(fw) == sorted(bnb)

    @pytest.mark.parametrize("seed", range(6))
    def test_fw_respects_nvmax(self, seed):
        rng = np.random.default_rng(60 + seed)
        T0, N = 30, 9
        X = rng.standard_normal((T0, N))
        y = X @ rng.standard_normal(N) + 0.1 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, 3, "AIC")
        fw = _best_subset_fw(G, Zty, yty, N, T0, 3, "AIC")
        assert len(fw) <= 3
        assert sorted(fw) == sorted(ex)

    def test_fw_r_max_zero_is_intercept_only(self):
        rng = np.random.default_rng(1)
        T0, N = 20, 5
        X = rng.standard_normal((T0, N))
        y = rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        assert _best_subset_fw(G, Zty, yty, N, T0, 0, "AICc") == []

    def test_fw_single_donor(self):
        rng = np.random.default_rng(2)
        T0 = 20
        x = rng.standard_normal((T0, 1))
        y = 2.0 * x[:, 0] + 0.05 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, x)
        ex = _best_subset_exhaustive(G, Zty, yty, 1, T0, 1, "AICc")
        fw = _best_subset_fw(G, Zty, yty, 1, T0, 1, "AICc")
        assert sorted(fw) == sorted(ex)

    def test_fw_collinear_pool_matches_oracle_objective(self):
        # A duplicated donor makes some subset Grams singular; FW must skip the
        # redundant pivot (no RSS reduction, min-norm behaviour) and reach the
        # same criterion value as the exhaustive search.
        rng = np.random.default_rng(3)
        T0 = 22
        x = rng.standard_normal((T0, 1))
        X = np.column_stack([x[:, 0], x[:, 0], rng.standard_normal(T0),
                             rng.standard_normal(T0)])     # cols 0,1 identical
        y = 1.5 * x[:, 0] + 0.05 * rng.standard_normal(T0)
        G, Zty, yty = _gram(y, X)
        N, r_max = X.shape[1], min(X.shape[1], T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, "AICc")
        fw = _best_subset_fw(G, Zty, yty, N, T0, r_max, "AICc")
        ic_ex = info_criterion(_subset_rss(G, Zty, yty, ex), T0, len(ex) + 1, "AICc")
        ic_fw = info_criterion(_subset_rss(G, Zty, yty, fw), T0, len(fw) + 1, "AICc")
        assert abs(ic_fw - ic_ex) < 1e-7 * max(abs(ic_ex), 1.0)

    def test_fw_matches_exhaustive_on_hk(self, hk_table16):
        y, X, labels, T0 = hk_table16
        G, Zty, yty = _gram(y[:T0], X[:T0])
        N, r_max = X.shape[1], min(X.shape[1], T0 - 2)
        ex = _best_subset_exhaustive(G, Zty, yty, N, T0, r_max, "AICc")
        fw = _best_subset_fw(G, Zty, yty, N, T0, r_max, "AICc")
        assert sorted(fw) == sorted(ex)


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
