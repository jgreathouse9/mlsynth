"""Fixed penalty for the penalized (Abadie-L'Hour) VanillaSC backend.

``VanillaSCConfig.penalized_lambda`` lets the user inject their own penalty
``lambda`` instead of cross-validating it: a numeric value is used as a fixed
penalty (skipping CV entirely), ``None`` preserves the CV behavior selected by
``penalized_cv``. Written test-first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.exceptions import MlsynthConfigError


def _panel(J: int = 8, T_pre: int = 15, T_post: int = 6, seed: int = 0):
    """Factor panel with a unit-level covariate; unit 0 is treated."""
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, 2))
    lam = rng.standard_normal((J + 1, 2))
    Y = F @ lam.T + 0.2 * rng.standard_normal((T, J + 1))
    x = lam[:, 0] * 3.0 + rng.standard_normal(J + 1) * 0.1   # unit-level covariate
    rows = [
        {"unit": j, "time": t, "y": float(Y[t, j]), "x": float(x[j]),
         "D": int(j == 0 and t >= T_pre)}
        for j in range(J + 1) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _weights(res) -> dict:
    return {k: v for k, v in (res.weights.donor_weights or {}).items()}


def _fit(df, **over):
    cfg = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
               covariates=["x"], backend="penalized", display_graphs=False)
    cfg.update(over)
    return VanillaSC(cfg).fit()


@pytest.fixture
def df():
    return _panel()


# ----------------------------------------------------------------------
# The fixed penalty is honored (skips CV)
# ----------------------------------------------------------------------

class TestFixedLambda:
    def test_used_lambda_equals_injected(self, df):
        res = _fit(df, penalized_lambda=0.7)
        assert res.method_details.parameters_used["penalized_lambda"] == pytest.approx(0.7)

    def test_lambda_zero_allowed(self, df):
        res = _fit(df, penalized_lambda=0.0)
        assert res.method_details.parameters_used["penalized_lambda"] == pytest.approx(0.0)
        assert res.att is not None and np.isfinite(res.att)

    def test_fixed_lambda_overrides_cv(self, df):
        # A numeric penalty must ignore penalized_cv entirely.
        w_holdout = _fit(df, penalized_lambda=0.5, penalized_cv="holdout")
        w_loo = _fit(df, penalized_lambda=0.5, penalized_cv="loo")
        a = np.array(list(_weights(w_holdout).values()))
        b = np.array(list(_weights(w_loo).values()))
        assert np.allclose(a, b, atol=1e-8)

    def test_deterministic(self, df):
        a = np.array(list(_weights(_fit(df, penalized_lambda=0.3)).values()))
        b = np.array(list(_weights(_fit(df, penalized_lambda=0.3)).values()))
        assert np.allclose(a, b, atol=1e-10)

    def test_larger_lambda_more_concentrated(self, df):
        # Abadie-L'Hour: larger lambda -> nearer-neighbour -> more concentrated.
        small = _weights(_fit(df, penalized_lambda=1e-6))
        large = _weights(_fit(df, penalized_lambda=50.0))
        assert max(large.values()) >= max(small.values()) - 1e-9

    def test_infinitesimal_lambda_is_lexicographic(self, df):
        # An infinitesimal-surrogate penalty stays feasible and finite.
        res = _fit(df, penalized_lambda=1e-10)
        w = np.array(list(_weights(res).values()))
        assert w.min() >= -1e-9
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(res.att)


# ----------------------------------------------------------------------
# None preserves cross-validation
# ----------------------------------------------------------------------

class TestNonePreservesCV:
    def test_none_runs_cv(self, df):
        res = _fit(df, penalized_lambda=None)      # default
        lam = res.method_details.parameters_used["penalized_lambda"]
        assert lam is not None and np.isfinite(lam) and lam >= 0.0

    def test_fixed_differs_from_cv_when_far(self, df):
        # A deliberately large fixed penalty should not coincide with the
        # CV-selected weights.
        cv = np.array(list(_weights(_fit(df, penalized_lambda=None)).values()))
        fixed = np.array(list(_weights(_fit(df, penalized_lambda=100.0)).values()))
        assert not np.allclose(cv, fixed, atol=1e-6)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

class TestValidation:
    def test_negative_lambda_rejected(self, df):
        with pytest.raises(MlsynthConfigError):
            VanillaSC(dict(df=df, outcome="y", treat="D", unitid="unit",
                           time="time", covariates=["x"], backend="penalized",
                           penalized_lambda=-0.5))

    def test_fixed_lambda_ignored_without_covariates(self, df):
        # The Abadie-L'Hour pairwise penalty needs covariate predictors; with
        # none, the engine routes backend='penalized' to the exact outcome-only
        # simplex, so penalized_lambda has no effect and is surfaced as None.
        base = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                    display_graphs=False)
        res = VanillaSC({**base, "backend": "penalized", "penalized_lambda": 0.2}).fit()
        assert res.method_details.parameters_used["penalized_lambda"] is None
        oo = VanillaSC({**base, "backend": "outcome-only"}).fit()
        a = np.array(list(_weights(res).values()))
        b = np.array(list(_weights(oo).values()))
        assert np.allclose(a, b, atol=1e-8)
