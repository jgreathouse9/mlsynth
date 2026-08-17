"""``lrvar_lag`` reaches the LASSO, and switches it to the fsPDA t-test.

The ladder from ``agents/agents_tests.md``, walked on this incident: the LASSO's
null rejection rate came out at 0.005-0.125 against Shi & Huang's (2023) Table 1
cells of 0.056-0.244, in all six cells and always low, while every other cell of
the table matched.

Rung 2 put it in the standard error. ``lasso_ate_inference`` builds Li & Bell's
(2017) two-component variance -- a post-period HAC term plus a first-stage OLS
prediction term -- and reads the HAC term at the Newey-West automatic lag. On the
Table 1 design the first-stage term is 44-46% of the total variance and the
automatic lag is 3 or 4 where the caller asked for 1 or 2, so the standard error
runs about a third large and the test under-rejects.

Rung 3 settled it on their own panels: against ``lasso.BIC`` with glmnet
converged, mlsynth's reported t-statistic sits up to 4.5e-01 away, and the
fixed-lag Bartlett t-statistic sits 7.4e-06 away, on all eight.

Rung 4 is the contract this file enforces. ``lrvar_lag`` was accepted by the
config, read by forward selection and HCW, and ignored by the LASSO. A supplied
lag that reaches nothing produces a well-formed standard error and a p-value in
range, so nothing downstream can tell it from a lag that was used.

The default is unchanged: with no lag the LASSO keeps Li & Bell's variance, which
is the inference its own paper prescribes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlsynth.utils.pda_helpers.inference import hac_lrv, newey_west_lag
from mlsynth.utils.pda_helpers.lasso.inference import lasso_ate_inference


def _panel(seed: int = 0, T0: int = 60, T2: int = 40, N: int = 8, effect: float = 0.4):
    """Treated series driven by two factors the donors also load on."""
    rng = np.random.default_rng(seed)
    T = T0 + T2
    f = rng.standard_normal((T, 2))
    lam = rng.uniform(0.5, 1.5, size=(N, 2))
    X = f @ lam.T + 0.3 * rng.standard_normal((T, N))
    y = f @ np.array([1.0, 0.8]) + 0.3 * rng.standard_normal(T)
    y[T0:] += effect
    return y, X, T0


def _fit(y, X, T0, **kw):
    from mlsynth import PDA
    import pandas as pd
    T, N = X.shape
    recs = [{"unit": "treated", "time": t + 1, "y": float(y[t]),
             "treat": int(t >= T0)} for t in range(T)]
    recs += [{"unit": f"d{j + 1:02d}", "time": t + 1, "y": float(X[t, j]),
              "treat": 0} for t in range(T) for j in range(N)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({"df": pd.DataFrame(recs), "outcome": "y", "treat": "treat",
                    "unitid": "unit", "time": "time", "methods": ["LASSO"],
                    "display_graphs": False, **kw}).fit()


class TestTheSuppliedLagIsWhatTheLassoUses:
    """Rung 2, at the helper."""

    def test_a_supplied_lag_gives_the_fspda_standard_error(self):
        """``se = sqrt(lrvar(d, h) / T2)`` -- their ``lasso.BIC``, exactly.

        No first-stage term. Supplying the lag is the request for the released
        package's t-test, and that test has one component.
        """
        y, X, T0 = _panel()
        cf = X @ np.linalg.lstsq(X[:T0], y[:T0], rcond=None)[0]
        support = np.ones(X.shape[1], dtype=bool)
        for lag in (0, 1, 3):
            att, se, _, _ = lasso_ate_inference(y, X, cf, support, T0, lrvar_lag=lag)
            d = y[T0:] - cf[T0:]
            expected = float(np.sqrt(hac_lrv(d, lag=lag) / d.size))
            assert se == pytest.approx(expected, rel=1e-12), f"lag {lag}"

    def test_no_lag_keeps_li_and_bell_s_two_component_variance(self):
        """The default is untouched: the first-stage term is still there and the
        HAC term is still read at the Newey-West automatic lag."""
        y, X, T0 = _panel()
        cf = X @ np.linalg.lstsq(X[:T0], y[:T0], rcond=None)[0]
        support = np.ones(X.shape[1], dtype=bool)
        att, se, _, _ = lasso_ate_inference(y, X, cf, support, T0)
        d = y[T0:] - cf[T0:]
        var_post = hac_lrv(d - d.mean(), lag=newey_west_lag(d.size)) / d.size
        assert se ** 2 > var_post * 1.05, (
            "the first-stage term is what makes this Li & Bell's variance and "
            "not the fsPDA one")

    def test_the_two_conventions_disagree_on_this_design(self):
        """Otherwise the tests above would pass against either implementation."""
        y, X, T0 = _panel()
        cf = X @ np.linalg.lstsq(X[:T0], y[:T0], rcond=None)[0]
        support = np.ones(X.shape[1], dtype=bool)
        se_default = lasso_ate_inference(y, X, cf, support, T0)[1]
        se_fixed = lasso_ate_inference(y, X, cf, support, T0, lrvar_lag=1)[1]
        assert abs(se_default - se_fixed) / se_fixed > 0.05

    @pytest.mark.parametrize("lag", [-1, 999])
    def test_a_lag_outside_the_cap_raises(self, lag):
        y, X, T0 = _panel()
        cf = X @ np.linalg.lstsq(X[:T0], y[:T0], rcond=None)[0]
        support = np.ones(X.shape[1], dtype=bool)
        with pytest.raises(ValueError, match="lrvar_lag"):
            lasso_ate_inference(y, X, cf, support, T0, lrvar_lag=lag)


class TestTheLagReachesTheEstimator:
    """Rung 4: the contract that was never enforced.

    A configuration field the estimator accepts and does not read is invisible
    downstream -- the standard error is finite, positive and the right order of
    magnitude whatever the field said. What separates a lag that was used from
    one that was dropped is that the answer moves when the lag moves, so that is
    what gets asserted, for every method that documents the field.
    """

    @pytest.mark.parametrize("method", ["LASSO", "fs"])
    def test_the_reported_standard_error_responds_to_the_lag(self, method):
        y, X, T0 = _panel()
        ses = [float(_fit(y, X, T0, methods=[method], lrvar_lag=lag)
                     .fits[method.lower()].att_se) for lag in (0, 1, 4)]
        assert len(set(np.round(ses, 12))) == 3, (
            f"{method} reports {ses} for three different lags")

    def test_the_lasso_matches_the_fspda_formula_end_to_end(self):
        """Through the public API, on the gap the estimator itself reports."""
        y, X, T0 = _panel()
        res = _fit(y, X, T0, lasso_criterion="mbic", lrvar_lag=2)
        las = res.fits["lasso"]
        d = np.asarray(las.gap, dtype=float)[T0:]
        assert float(las.att_se) == pytest.approx(
            float(np.sqrt(hac_lrv(d, lag=2) / d.size)), rel=1e-10)

    def test_omitting_the_lag_leaves_the_default_path(self):
        y, X, T0 = _panel()
        with_lag = float(_fit(y, X, T0, lrvar_lag=1).fits["lasso"].att_se)
        without = float(_fit(y, X, T0).fits["lasso"].att_se)
        assert without != pytest.approx(with_lag, rel=1e-6)
        assert np.isfinite(without) and without > 0
