"""Jiang et al. (2025) bootstrap prediction intervals for PDA: engine tests.

``inferutils.pda_prediction_intervals`` is a general, estimator-agnostic engine:
given the treated series, control matrix, a fitted counterfactual + support, and
a ``refit`` callback (fit the point estimator on a bootstrap pre-period), it
returns equal-tailed (EQ) and symmetric (SY) prediction intervals for the
post-period counterfactual ``Y_t`` and the treatment effect ``Delta_t``, via the
dependent-wild / residual bootstrap of Algorithm 2.1.

These tests pin the contract TDD-first:

1. shape / structure of the returned intervals (EQ + SY, effect + counterfactual),
2. interval ordering (lower <= point <= upper),
3. the studentization label: post-selection OLS HAC sandwich when the support is
   low-rank-feasible, sigma^2-only fallback otherwise,
4. reproducibility under a fixed seed,
5. nominal coverage on a known DGP (the method's central claim),
6. input-validation failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.inferutils import pda_prediction_intervals


# ---------------------------------------------------------------------------
# A simple OLS PDA estimator to drive the engine (refit callback)
# ---------------------------------------------------------------------------

def _ols_fit(y, X, T0, support):
    """Post-selection OLS on ``support`` (with intercept); full-T counterfactual."""
    cols = np.asarray(support, dtype=int)
    Z = np.column_stack([np.ones(T0), X[:T0][:, cols]])
    coef, *_ = np.linalg.lstsq(Z, y[:T0], rcond=None)
    Zf = np.column_stack([np.ones(X.shape[0]), X[:, cols]])
    return Zf @ coef


def _make_refit(X, T0, support):
    def refit(y_boot):
        cf = _ols_fit(y_boot, X, T0, support)
        return cf, np.asarray(support, dtype=int)
    return refit


def _dgp(seed, T0=40, T1=6, p=4, effect=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0 + T1, p))
    beta = np.array([1.0, -0.5, 0.3, 0.0])[:p]
    y0 = X @ beta + rng.standard_normal(T0 + T1) * 0.5
    y = y0.copy()
    y[T0:] += effect
    return y, X


def _fit(y, X, T0, support):
    return _ols_fit(y, X, T0, support)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_keys_and_shapes(self):
        y, X = _dgp(0)
        T0, T1 = 40, 6
        support = [0, 1, 2, 3]
        cf = _fit(y, X, T0, support)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=_make_refit(X, T0, support), alpha=0.05, n_boot=199, seed=1,
        )
        assert out["studentization"] in ("sandwich", "sigma2")
        for block in ("effect", "counterfactual"):
            b = out[block]
            for key in ("point", "eq_lower", "eq_upper", "sy_lower", "sy_upper"):
                assert b[key].shape == (T1,)
        assert out["se"].shape == (T1,)
        # effect point == y_post - cf_post
        np.testing.assert_allclose(out["effect"]["point"], y[T0:] - cf[T0:], atol=1e-12)

    def test_interval_ordering(self):
        y, X = _dgp(2)
        T0, support = 40, [0, 1, 2, 3]
        cf = _fit(y, X, T0, support)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=_make_refit(X, T0, support), n_boot=199, seed=3,
        )
        for block in ("effect", "counterfactual"):
            b = out[block]
            assert (b["eq_lower"] <= b["eq_upper"] + 1e-9).all()
            assert (b["sy_lower"] <= b["sy_upper"] + 1e-9).all()
            # symmetric interval is centered on the point estimate
            mid = 0.5 * (b["sy_lower"] + b["sy_upper"])
            np.testing.assert_allclose(mid, b["point"], atol=1e-9)


# ---------------------------------------------------------------------------
# Studentization label
# ---------------------------------------------------------------------------

class TestStudentization:
    def test_sandwich_when_support_feasible(self):
        y, X = _dgp(0)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=support,
            refit=_make_refit(X, 40, support), n_boot=99, seed=1,
        )
        assert out["studentization"] == "sandwich"

    def test_sigma2_fallback_when_support_empty(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        # empty support -> no sandwich -> sigma^2-only studentization
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=[],
            refit=lambda yb: (cf, np.array([], dtype=int)), n_boot=99, seed=1,
        )
        assert out["studentization"] == "sigma2"
        # se reduces to a single sigma across periods (no V_t term)
        assert np.allclose(out["se"], out["se"][0])

    def test_sigma2_fallback_when_support_exceeds_T0(self):
        # |support| >= T0 -> sandwich Gram is singular -> fallback.
        rng = np.random.default_rng(5)
        T0, T1, p = 8, 4, 10
        X = rng.standard_normal((T0 + T1, p))
        y = rng.standard_normal(T0 + T1)
        support = list(range(p))
        cf = np.zeros(T0 + T1)
        out = pda_prediction_intervals(
            y, X, T0, counterfactual=cf, support=support,
            refit=lambda yb: (cf, np.asarray(support)), n_boot=49, seed=1,
        )
        assert out["studentization"] == "sigma2"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_intervals(self):
        y, X = _dgp(7)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        kw = dict(counterfactual=cf, support=support,
                  refit=_make_refit(X, 40, support), n_boot=199)
        a = pda_prediction_intervals(y, X, 40, seed=42, **kw)
        b = pda_prediction_intervals(y, X, 40, seed=42, **kw)
        np.testing.assert_allclose(a["effect"]["eq_lower"], b["effect"]["eq_lower"])
        np.testing.assert_allclose(a["effect"]["sy_upper"], b["effect"]["sy_upper"])


# ---------------------------------------------------------------------------
# Coverage (the method's central claim)
# ---------------------------------------------------------------------------

class TestCoverage:
    @pytest.mark.slow
    def test_nominal_coverage_of_zero_effect(self):
        # 95% EQ prediction interval for Delta_{T0+1} should cover the true
        # zero effect about 95% of the time across DGP draws.
        T0, T1, p = 45, 1, 4
        support = [0, 1, 2, 3]
        covered = 0
        reps = 200
        for s in range(reps):
            y, X = _dgp(1000 + s, T0=T0, T1=T1, p=p, effect=0.0)
            cf = _fit(y, X, T0, support)
            out = pda_prediction_intervals(
                y, X, T0, counterfactual=cf, support=support,
                refit=_make_refit(X, T0, support), alpha=0.05, n_boot=199, seed=s,
            )
            lo, hi = out["effect"]["eq_lower"][0], out["effect"]["eq_upper"][0]
            covered += int(lo <= 0.0 <= hi)
        cov = covered / reps
        # nominal 0.95; MC SE ~ sqrt(.95*.05/200) ~ 1.5pp -> allow [0.90, 0.99]
        assert 0.90 <= cov <= 0.995, f"coverage {cov:.3f} off nominal 0.95"


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_length_mismatch(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthDataError):
            pda_prediction_intervals(y[:-1], X, 40, counterfactual=cf,
                                     support=[0], refit=lambda yb: (cf, [0]))

    def test_no_post_periods(self):
        y, X = _dgp(0, T0=46, T1=0)
        cf = _fit(y, X, 46, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 46, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]))

    def test_bad_n_boot(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=1)

    def test_bad_alpha(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError):
            pda_prediction_intervals(y, X, 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=10, alpha=1.5)

    def test_x_not_2d(self):
        y, X = _dgp(0)
        cf = _fit(y, X, 40, [0, 1, 2, 3])
        with pytest.raises(MlsynthDataError):
            pda_prediction_intervals(y, X[:, 0], 40, counterfactual=cf, support=[0],
                                     refit=lambda yb: (cf, [0]), n_boot=10)


class TestIndependentBootstrap:
    def test_iid_multipliers_path(self):
        y, X = _dgp(0)
        support = [0, 1, 2, 3]
        cf = _fit(y, X, 40, support)
        out = pda_prediction_intervals(
            y, X, 40, counterfactual=cf, support=support,
            refit=_make_refit(X, 40, support), n_boot=199, dependent=False, seed=1,
        )
        assert out["effect"]["eq_lower"].shape == (6,)
        assert (out["effect"]["eq_lower"] <= out["effect"]["eq_upper"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Wiring into the PDA estimator (all three methods)
# ---------------------------------------------------------------------------

def _pda_panel(n_units=8, n_periods=40, t_start=30, effect=5.0, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)
    n_donors = n_units - 1
    donors = {}
    for j in range(n_donors):
        donors[j] = rng.normal(20 + 4 * j, 3) + np.linspace(0, 6 + j, n_periods) \
            + rng.normal(0, 0.6, n_periods)
    w = rng.uniform(0.2, 1.0, n_donors)
    w /= w.sum()
    treated = sum(w[j] * donors[j] for j in range(n_donors)) + rng.normal(0, 0.3, n_periods)
    rows = []
    for i, t in enumerate(periods):
        v = treated[i] + (effect if t >= t_start else 0.0)
        rows.append({"ID": 1, "Period": int(t), "Value": float(v),
                     "IsTreated": int(t >= t_start)})
        for j in range(n_donors):
            rows.append({"ID": j + 2, "Period": int(t),
                         "Value": float(donors[j][i]), "IsTreated": 0})
    return pd.DataFrame(rows)


def _pda_cfg(df, **kw):
    cfg = dict(df=df, outcome="Value", treat="IsTreated", unitid="ID",
               time="Period", display_graphs=False)
    cfg.update(kw)
    return cfg


class TestPDAWiring:
    @pytest.mark.parametrize("method", ["l2", "LASSO", "fs"])
    def test_prediction_intervals_attached(self, method):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        res = PDA(PDAConfig(**_pda_cfg(
            df, method=method, prediction_intervals=True, pi_n_boot=99))).fit()
        fit = next(iter(res.fits.values()))
        pi = fit.prediction_intervals
        assert pi is not None
        assert pi["studentization"] in ("sandwich", "sigma2")
        T1 = res.inputs.T2
        assert pi["effect"]["eq_lower"].shape == (T1,)
        assert pi["counterfactual"]["sy_upper"].shape == (T1,)
        assert (pi["effect"]["eq_lower"] <= pi["effect"]["eq_upper"] + 1e-9).all()

    def test_off_by_default(self):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        res = PDA(PDAConfig(**_pda_cfg(df, method="fs"))).fit()
        fit = next(iter(res.fits.values()))
        assert fit.prediction_intervals is None

    def test_reproducible_via_pi_seed(self):
        from mlsynth import PDA
        from mlsynth.config_models import PDAConfig

        df = _pda_panel()
        cfg = _pda_cfg(df, method="fs", prediction_intervals=True, pi_n_boot=99, pi_seed=7)
        a = PDA(PDAConfig(**cfg)).fit().fits["fs"].prediction_intervals
        b = PDA(PDAConfig(**cfg)).fit().fits["fs"].prediction_intervals
        np.testing.assert_allclose(a["effect"]["eq_lower"], b["effect"]["eq_lower"])
