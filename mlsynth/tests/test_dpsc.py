"""Tests for DPSC (differentially private synthetic control, Rho et al. 2023).

Layered per ``agents/agents_tests.md``:
  Layer 1 (mechanisms): closed-form ridge == sklearn; sensitivity formula;
    high-dimensional Laplace structure + reproducibility; epsilon -> infinity
    converges to the non-private ridge SC.
  Layer 2 (setup): single-treated ingestion; multi-cohort rejected.
  Layer 3 (integration): DPSC.fit on Prop 99 -- shapes, reproducibility, the
    non-private reference, both mechanisms, plotting smoke.
  Layer 4 (config): every validator raises the translated error.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth import DPSC  # noqa: E402
from mlsynth.config_models import BaseEstimatorResults, DPSCConfig  # noqa: E402
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError  # noqa: E402
from mlsynth.utils.dpsc_helpers.mechanisms import (  # noqa: E402
    _ridge_coefficients,
    _high_dim_laplace,
    non_private_counterfactual,
    output_sensitivity,
    run_objective_perturbation,
    run_output_perturbation,
)

_SMOKING = Path(__file__).resolve().parents[2] / "basedata" / "smoking_data.csv"


@pytest.fixture(scope="module")
def prop99() -> pd.DataFrame:
    df = pd.read_csv(_SMOKING)
    df["treat"] = df["Proposition 99"].astype(int)
    return df[["state", "year", "cigsale", "treat"]]


def _cfg(df, **over):
    base = dict(df=df, outcome="cigsale", treat="treat", unitid="state", time="year",
                n_draws=50, seed=0, display_graphs=False)
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Layer 1: mechanisms
# ---------------------------------------------------------------------------
class TestMechanisms:
    def test_ridge_closed_form_matches_sklearn(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 6))
        y = rng.standard_normal(15)
        w = _ridge_coefficients(X, y, ridge_alpha=5.0)
        sk = Ridge(alpha=5.0, fit_intercept=False).fit(X, y).coef_
        assert np.max(np.abs(w - sk)) < 1e-10

    def test_output_sensitivity_formula(self):
        # Delta = 4 T0 sqrt(8 + N0) / lambda
        assert output_sensitivity(19, 38, 10.0) == pytest.approx(4 * 19 * np.sqrt(46) / 10.0)

    def test_high_dim_laplace_reproducible_and_isotropic(self):
        a = _high_dim_laplace(np.random.RandomState(3), 10, 2.0)
        b = _high_dim_laplace(np.random.RandomState(3), 10, 2.0)
        assert np.array_equal(a, b)                       # seed -> identical
        # one magnitude x unit direction: recover a unit vector
        assert np.linalg.norm(a) > 0

    def test_seeded_mechanisms_are_reproducible(self):
        rng = np.random.default_rng(1)
        pre_d = rng.standard_normal((10, 5)); pre_t = rng.standard_normal(10)
        full = rng.standard_normal((13, 5))
        for run in (run_output_perturbation, run_objective_perturbation):
            cf1, w1, _ = run(np.random.RandomState(7), pre_d, pre_t, full, 10.0, 5.0, 5.0)
            cf2, w2, _ = run(np.random.RandomState(7), pre_d, pre_t, full, 10.0, 5.0, 5.0)
            assert np.array_equal(cf1, cf2) and np.array_equal(w1, w2)

    def test_objective_gaussian_branch_runs(self):
        # delta > 0 selects Gaussian ((epsilon, delta)-DP) noise for the
        # objective mechanism; the fit must still be finite and reproducible.
        rng = np.random.default_rng(4)
        pre_d = rng.standard_normal((10, 5)); pre_t = rng.standard_normal(10)
        full = rng.standard_normal((13, 5))
        cf1, w1, _ = run_objective_perturbation(
            np.random.RandomState(9), pre_d, pre_t, full, 10.0, 5.0, 5.0, delta=0.1)
        cf2, w2, _ = run_objective_perturbation(
            np.random.RandomState(9), pre_d, pre_t, full, 10.0, 5.0, 5.0, delta=0.1)
        assert np.all(np.isfinite(cf1)) and np.array_equal(w1, w2)

    def test_epsilon_to_infinity_converges_to_non_private(self):
        rng = np.random.default_rng(2)
        pre_d = rng.standard_normal((12, 4)); pre_t = rng.standard_normal(12)
        full = rng.standard_normal((16, 4))
        cf_np, _ = non_private_counterfactual(pre_d, pre_t, full, 10.0)
        for run in (run_output_perturbation, run_objective_perturbation):
            cf, _, _ = run(np.random.RandomState(0), pre_d, pre_t, full, 10.0, 1e7, 1e7)
            assert np.max(np.abs(cf - cf_np)) < 1e-2  # noise ~ 1/eps -> 0


# ---------------------------------------------------------------------------
# Layer 2 / 3: setup + integration on Prop 99
# ---------------------------------------------------------------------------
class TestProp99Integration:
    def test_smoke_returns_contract(self, prop99):
        res = DPSC(_cfg(prop99, epsilon1=100.0, epsilon2=100.0)).fit()
        assert isinstance(res, BaseEstimatorResults)
        assert np.isfinite(res.effects.att)
        assert res.time_series.counterfactual_outcome.shape[0] == 31
        assert res.method_details.method_name == "DPSC"

    def test_non_private_reference_recorded(self, prop99):
        res = DPSC(_cfg(prop99, epsilon1=100.0, epsilon2=100.0)).fit()
        npv = res.inference.details["att_non_private"]
        # Non-private ridge SC on Prop 99 lands in the canonical negative range.
        assert -25 < npv < -8

    def test_privacy_noise_reported_as_se(self, prop99):
        res = DPSC(_cfg(prop99, epsilon1=1.0, epsilon2=1.0, n_draws=100)).fit()
        assert res.inference.standard_error is not None and res.inference.standard_error > 0
        lo, hi = res.inference.ci_lower, res.inference.ci_upper
        assert lo < res.effects.att < hi

    def test_output_mechanism_runs(self, prop99):
        res = DPSC(_cfg(prop99, mechanism="output", epsilon1=50.0, epsilon2=50.0)).fit()
        assert np.isfinite(res.effects.att)

    def test_seed_reproducible_end_to_end(self, prop99):
        a = DPSC(_cfg(prop99, epsilon1=5.0, epsilon2=5.0, seed=42)).fit()
        b = DPSC(_cfg(prop99, epsilon1=5.0, epsilon2=5.0, seed=42)).fit()
        assert a.effects.att == b.effects.att
        assert np.array_equal(a.time_series.counterfactual_outcome,
                              b.time_series.counterfactual_outcome)

    def test_different_seed_gives_different_release(self, prop99):
        a = DPSC(_cfg(prop99, epsilon1=5.0, epsilon2=5.0, seed=1)).fit()
        b = DPSC(_cfg(prop99, epsilon1=5.0, epsilon2=5.0, seed=2)).fit()
        assert a.effects.att != b.effects.att

    def test_multi_treated_panel_rejected(self, prop99):
        df = prop99.copy()
        extra = {"California", "Nevada", "Utah"}
        df["treat"] = ((df["state"].isin(extra)) & (df["year"] >= 1989)).astype(int)
        with pytest.raises(MlsynthDataError):
            DPSC(_cfg(df, epsilon1=5.0, epsilon2=5.0)).fit()

    def test_plotting_smoke(self, prop99):
        res = DPSC(_cfg(prop99, epsilon1=100.0, epsilon2=100.0, display_graphs=True)).fit()
        assert res is not None


# ---------------------------------------------------------------------------
# Layer 4: config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def _base(self, prop99, **over):
        return _cfg(prop99, **over)

    @pytest.mark.parametrize("field,value", [
        ("epsilon1", 0.0), ("epsilon1", -1.0),
        ("epsilon2", 0.0), ("epsilon2", -2.0),
        ("ridge_lambda", 0.0), ("ridge_lambda", -1.0),
        ("delta", -0.1), ("delta", 1.0),
        ("n_draws", 0), ("alpha", 0.0), ("alpha", 1.0),
        ("mechanism", "banana"),
    ])
    def test_invalid_config_rejected(self, prop99, field, value):
        with pytest.raises(MlsynthConfigError):
            DPSC(self._base(prop99, **{field: value}))

    def test_extra_field_forbidden(self, prop99):
        with pytest.raises(MlsynthConfigError):
            DPSC(self._base(prop99, not_a_field=1))

    def test_config_defaults(self, prop99):
        cfg = DPSCConfig(**self._base(prop99))
        assert cfg.mechanism == "objective" and cfg.delta == 0.0
