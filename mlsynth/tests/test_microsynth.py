"""Tests for the MicroSynth estimator.

Robbins-Davenport (2021) user-level balancing synthetic control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mlsynth import MicroSynth
from mlsynth.config_models import MicroSynthConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.microsynth_helpers.diagnostics import (
    effective_sample_size,
    feasibility_check,
    standardized_mean_difference,
)
from mlsynth.utils.microsynth_helpers.dual_solver import solve_microsynth_dual
from mlsynth.utils.microsynth_helpers.raking import solve_raking_weights
from mlsynth.utils.microsynth_helpers.setup import prepare_microsynth_inputs
from mlsynth.utils.microsynth_helpers.structures import (
    MicroSynthDesign,
    MicroSynthInference,
    MicroSynthInputs,
    MicroSynthResults,
)


def _contamination_panel(
    *, seed: int = 0, n: int = 1000, n_exposed: int = 600,
    n_contam: int = 150, true_lift: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Contaminated-holdout DGP folded into a long df."""
    rng = np.random.default_rng(seed)
    age              = rng.standard_normal(n)
    prior_engagement = rng.standard_normal(n)
    device           = rng.binomial(1, 0.4, n).astype(float)
    gender           = rng.binomial(1, 0.5, n).astype(float)
    country_tier     = rng.standard_normal(n)
    logit_p0 = (
        -1.5 + 0.30 * age + 0.60 * prior_engagement + 0.20 * device
        - 0.10 * gender + 0.20 * country_tier
    )
    p0 = expit(logit_p0)
    p1 = np.clip(p0 + true_lift, 0, 1)
    Y0 = rng.binomial(1, p0)
    Y1 = rng.binomial(1, p1)

    perm = rng.permutation(n)
    assigned_exposed = np.zeros(n, dtype=bool)
    assigned_exposed[perm[:n_exposed]] = True
    holdout_idx = np.where(~assigned_exposed)[0]
    contam_score = expit(
        0.8 * prior_engagement[holdout_idx] + 0.5 * age[holdout_idx]
        + 0.4 * country_tier[holdout_idx]
    )
    probs = contam_score / contam_score.sum()
    contam = rng.choice(len(holdout_idx), size=n_contam, replace=False, p=probs)
    saw_ads = assigned_exposed.copy()
    saw_ads[holdout_idx[contam]] = True
    Y_obs = np.where(saw_ads, Y1, Y0)

    rows = []
    for i in range(n):
        base = dict(
            user_id=f"u{i:05d}", age=age[i], device=device[i], gender=gender[i],
            country_tier=country_tier[i], prior_engagement=prior_engagement[i],
        )
        rows.append({**base, "week": 0, "converted": 0, "saw_ad": 0})
        rows.append({
            **base, "week": 1,
            "converted": int(Y_obs[i]), "saw_ad": int(saw_ads[i]),
        })
    return pd.DataFrame(rows), float(true_lift)


@pytest.fixture(scope="module")
def small_panel():
    df, lift = _contamination_panel(n=600, n_exposed=400, n_contam=80)
    return df, lift


COVS = ["age", "device", "gender", "country_tier", "prior_engagement"]


# ---------------------------------------------------------------------------
# Layer 1: dual solver + diagnostics
# ---------------------------------------------------------------------------

class TestDualSolver:
    def test_balance_constraints_met(self):
        rng = np.random.default_rng(0)
        n_C, d = 200, 4
        X_C = rng.standard_normal((n_C, d))
        xbar_T = X_C.mean(axis=0) + 0.3 * rng.standard_normal(d)
        res = solve_microsynth_dual(X_C, xbar_T)
        assert res.converged
        assert res.w.shape == (n_C,)
        assert res.w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (res.w >= -1e-8).all()
        # Balance to numerical precision.
        balanced = X_C.T @ res.w
        assert np.allclose(balanced, xbar_T, atol=1e-4)

    def test_dual_dim_independent_of_nC(self):
        rng = np.random.default_rng(1)
        d = 3
        for n_C in (50, 500):
            X_C = rng.standard_normal((n_C, d))
            xbar_T = X_C.mean(axis=0)
            res = solve_microsynth_dual(X_C, xbar_T)
            assert res.dual_lambda.shape == (d,)
            assert np.isfinite(res.dual_nu)

    def test_uniform_when_treated_matches_control_mean(self):
        # If treated mean equals control mean, uniform weights satisfy
        # all constraints; the QP picks those.
        rng = np.random.default_rng(2)
        n_C, d = 100, 3
        X_C = rng.standard_normal((n_C, d))
        xbar_T = X_C.mean(axis=0)
        res = solve_microsynth_dual(X_C, xbar_T)
        assert np.allclose(res.w, 1 / n_C, atol=1e-4)


class TestRakingCalibration:
    """The microsynth (Robbins et al.) raking/GREG calibration weight solve."""

    def test_exact_balance_and_entropy_form(self):
        rng = np.random.default_rng(0)
        n_C, n_T, d = 300, 40, 4
        X_C = np.column_stack([np.ones(n_C), rng.standard_normal((n_C, d - 1))])
        X_T = np.column_stack([np.ones(n_T), 0.8 * rng.standard_normal((n_T, d - 1))])
        targets = X_T.sum(axis=0)                         # treated column totals
        sol = solve_raking_weights(X_C, targets)
        assert sol.converged
        # exact calibration: weighted control totals == treated targets
        assert np.allclose(X_C.T @ sol.w, targets, atol=1e-6)
        # raking weights are a positive exponential tilt of the base weights
        assert np.all(sol.w > 0)
        assert np.allclose(sol.w, np.exp(np.clip(X_C @ sol.dual_lambda, -50, 50)), atol=1e-9)
        # intercept target makes the weights sum to the treated count
        assert sol.w.sum() == pytest.approx(float(n_T), abs=1e-5)

    def test_base_weight_tilt(self):
        # With non-uniform base weights, w_i = base_i * exp(x_i . lambda).
        rng = np.random.default_rng(1)
        n_C, d = 150, 3
        X_C = np.column_stack([np.ones(n_C), rng.standard_normal((n_C, d - 1))])
        targets = np.array([20.0, 1.5, -0.7])
        base = rng.uniform(0.5, 2.0, n_C)
        sol = solve_raking_weights(X_C, targets, base_weight=base)
        # calibration is exact to microsynth's tolerance (cal.epsilon = 1e-4)
        assert np.allclose(X_C.T @ sol.w, targets, atol=1e-3)
        assert np.allclose(sol.w, base * np.exp(np.clip(X_C @ sol.dual_lambda, -50, 50)), atol=1e-9)


class TestDiagnostics:
    def test_smd_zero_when_balanced(self):
        rng = np.random.default_rng(0)
        X_T = rng.standard_normal((50, 3))
        X_C = X_T.copy()
        rng.shuffle(X_C)
        smd = standardized_mean_difference(X_T, X_C)
        # Same distribution; SMDs should be small.
        assert np.max(np.abs(smd)) < 0.5

    def test_ess_bounds(self):
        w = np.ones(100) / 100
        assert effective_sample_size(w) == pytest.approx(100.0)
        w_degen = np.zeros(100); w_degen[0] = 1.0
        assert effective_sample_size(w_degen) == pytest.approx(1.0)

    def test_feasibility_check(self):
        ok = feasibility_check(np.array([1e-6, 2e-6]), balance_tol=1e-4)
        assert ok[0] is True
        bad = feasibility_check(np.array([0.5, 1e-6]), balance_tol=1e-4)
        assert bad[0] is False
        assert "NOT achieved" in bad[1]


# ---------------------------------------------------------------------------
# Layer 2: setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_basic_shapes(self, small_panel):
        df, _ = small_panel
        inputs = prepare_microsynth_inputs(
            df=df, outcome="converted", treat="saw_ad",
            unitid="user_id", time="week", covariates=COVS,
        )
        assert inputs.X_T.shape[1] == 5
        assert inputs.X_C.shape[1] == 5
        assert inputs.n_T + inputs.n_C == 600
        assert inputs.T_post == 1
        assert inputs.cohort_time == 1
        assert list(inputs.covariate_names) == COVS

    def test_time_varying_covariate_rejected(self, small_panel):
        df, _ = small_panel
        df_bad = df.copy()
        # Make 'age' time-varying by perturbing the week-1 rows.
        df_bad.loc[df_bad["week"] == 1, "age"] += 0.01
        with pytest.raises(MlsynthDataError, match="time-invariant"):
            prepare_microsynth_inputs(
                df=df_bad, outcome="converted", treat="saw_ad",
                unitid="user_id", time="week", covariates=COVS,
            )

    def test_no_treated_units_rejected(self, small_panel):
        df, _ = small_panel
        df_no_t = df.copy()
        df_no_t["saw_ad"] = 0
        with pytest.raises(MlsynthDataError, match="No treated"):
            prepare_microsynth_inputs(
                df=df_no_t, outcome="converted", treat="saw_ad",
                unitid="user_id", time="week", covariates=COVS,
            )

    def test_outcome_lag_periods(self, small_panel):
        df, _ = small_panel
        # Week 0 is pre-treatment (cohort time = 1).
        inputs = prepare_microsynth_inputs(
            df=df, outcome="converted", treat="saw_ad",
            unitid="user_id", time="week",
            covariates=COVS, outcome_lag_periods=[0],
        )
        assert inputs.X_T.shape[1] == 6
        assert inputs.covariate_names[-1] == "converted@0"

    def test_post_period_lag_rejected(self, small_panel):
        df, _ = small_panel
        with pytest.raises(MlsynthDataError, match="pre-treatment window"):
            prepare_microsynth_inputs(
                df=df, outcome="converted", treat="saw_ad",
                unitid="user_id", time="week",
                covariates=COVS, outcome_lag_periods=[1],
            )

    def test_no_covariates_rejected(self, small_panel):
        df, _ = small_panel
        with pytest.raises(MlsynthDataError):
            prepare_microsynth_inputs(
                df=df, outcome="converted", treat="saw_ad",
                unitid="user_id", time="week", covariates=[],
            )

    def test_unknown_covariate_rejected(self, small_panel):
        df, _ = small_panel
        with pytest.raises(MlsynthDataError):
            prepare_microsynth_inputs(
                df=df, outcome="converted", treat="saw_ad",
                unitid="user_id", time="week",
                covariates=["does_not_exist"],
            )


# ---------------------------------------------------------------------------
# Layer 3: integration / synthetic recovery
# ---------------------------------------------------------------------------

class TestRakingPanel:
    """The microsynth panel method (weight_method='raking'): balance the full
    pre-period outcome trajectory, report per-period TOTAL effects."""

    def _panel(self, n_c=80, n_t=8, T0=6, T_post=3, eff=2.0, seed=0):
        rng = np.random.default_rng(seed)
        F = rng.normal(size=(T0 + T_post, 2))            # 2 latent time factors
        rows = []
        for i in range(n_c + n_t):
            load = rng.normal(size=2)
            base = F @ load + 0.05 * rng.normal(size=T0 + T_post)
            treated = i >= n_c
            for t in range(T0 + T_post):
                post = treated and t >= T0
                rows.append({"unit": i, "time": t,
                             "y": float(base[t] + (eff if post else 0.0)),
                             "treat": int(post)})
        return pd.DataFrame(rows), eff, n_t

    def test_raking_balance_weights_and_effect(self):
        df, eff, n_t = self._panel()
        res = MicroSynth({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "covariates": [],
            "outcome_lag_periods": list(range(6)),       # full pre-period window
            "weight_method": "raking", "run_inference": False,
            "display_graphs": False,
        }).fit()
        # raking weights sum to the treated count and exactly balance the pre-fit
        assert res.design.w.sum() == pytest.approx(float(n_t), abs=1e-3)
        assert np.max(np.abs(res.design.smd_after)) < 1e-2
        # per-period contrast is on totals -> ~ n_t * per-unit effect
        assert res.att == pytest.approx(n_t * eff, rel=0.25)
        assert res.gap_trajectory.shape == (3,)          # one per post period

    def test_simplex_default_unchanged(self):
        df, eff, n_t = self._panel()
        cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                   covariates=[], outcome_lag_periods=list(range(6)),
                   run_inference=False, display_graphs=False)
        res = MicroSynth(cfg).fit()                       # default simplex
        assert res.design.w.sum() == pytest.approx(1.0, abs=1e-6)  # simplex weights
        assert res.att == pytest.approx(eff, rel=0.25)            # per-unit-mean effect


class TestSyntheticRecovery:
    def test_recovers_true_lift(self, small_panel):
        df, true_lift = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": False, "display_graphs": False,
        }).fit()
        # Single-seed integration test; tolerance reflects finite-sample noise.
        assert res.att == pytest.approx(true_lift, abs=0.04)

    def test_balance_achieved(self, small_panel):
        df, _ = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert res.design.feasible
        assert np.max(np.abs(res.design.smd_after)) < 1e-3

    def test_weights_on_simplex(self, small_panel):
        df, _ = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert res.design.w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (res.design.w >= -1e-8).all()

    def test_bootstrap_inference_runs(self, small_panel):
        df, true_lift = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": True, "n_bootstrap": 60, "seed": 42,
            "display_graphs": False,
        }).fit()
        assert res.inference.method == "paired_bootstrap"
        assert np.isfinite(res.inference.se)
        assert res.inference.se > 0
        lo, hi = res.inference.ci
        assert lo <= res.inference.att <= hi
        assert res.inference.n_bootstrap > 0


# ---------------------------------------------------------------------------
# Layer 3b: ad-holdout simulation (ITT vs as-treated vs CACE)
# ---------------------------------------------------------------------------

class TestAdHoldoutSimulation:
    """Spillover/contamination DGP with an unobserved confounder.

    The as-treated comparison (regroup by exposure) is biased because
    exposure is selected on latent intent that balancing on observed
    covariates cannot remove; ITT on the randomized arms is unbiased for
    the campaign effect; and CACE = ITT / compliance-gap recovers the
    true per-exposure effect.
    """

    @staticmethod
    def _att(df, treat_col):
        return MicroSynth({
            "df": df, "outcome": "sales", "treat": treat_col,
            "unitid": "user_id", "time": "time",
            "covariates": ["age", "income"],
            "run_inference": False, "display_graphs": False,
        }).fit()

    def test_as_treated_is_biased_itt_and_cace_recover(self):
        from mlsynth.utils.microsynth_helpers import simulate_ad_holdout

        df, truth = simulate_ad_holdout(n_per_arm=6000, delta=1.0, seed=1)
        gap = truth["compliance_gap"]
        delta = truth["delta_per_exposure"]

        as_treated = self._att(df, "D_att").att
        itt = self._att(df, "D_itt").att
        cace = itt / gap

        # As-treated overstates the per-exposure effect (residual
        # unobserved-intent confounding survives covariate balancing).
        assert as_treated > delta + 0.10
        # ITT lands on the (diluted) campaign effect delta * gap.
        assert itt == pytest.approx(truth["itt_effect"], abs=0.08)
        # The covariate-balanced Wald ratio recovers the per-exposure effect.
        assert cace == pytest.approx(delta, abs=0.12)
        # And it beats the as-treated estimate it is meant to replace.
        assert abs(cace - delta) < abs(as_treated - delta)

    def test_covariates_balanced_under_both_labellings(self):
        from mlsynth.utils.microsynth_helpers import simulate_ad_holdout

        df, _ = simulate_ad_holdout(n_per_arm=4000, seed=2)
        for treat_col in ("D_itt", "D_att"):
            res = self._att(df, treat_col)
            assert res.design.feasible
            assert np.max(np.abs(res.design.smd_after)) < 1e-3


# ---------------------------------------------------------------------------
# Layer 4: public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_import(self):
        from mlsynth import MicroSynth as Imported  # noqa: F401
        assert Imported is MicroSynth

    def test_results_types(self, small_panel):
        df, _ = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": False, "display_graphs": False,
        }).fit()
        assert isinstance(res, MicroSynthResults)
        assert isinstance(res.inputs, MicroSynthInputs)
        assert isinstance(res.design, MicroSynthDesign)
        assert isinstance(res.inference, MicroSynthInference)

    def test_dict_vs_config_object(self, small_panel):
        df, _ = small_panel
        cfg_dict = dict(
            df=df, outcome="converted", treat="saw_ad",
            unitid="user_id", time="week", covariates=COVS,
            run_inference=False, display_graphs=False,
        )
        cfg_obj = MicroSynthConfig(**cfg_dict)
        r1 = MicroSynth(cfg_dict).fit()
        r2 = MicroSynth(cfg_obj).fit()
        assert r1.att == pytest.approx(r2.att, abs=1e-9)

    def test_invalid_config_raises(self):
        with pytest.raises(MlsynthConfigError):
            MicroSynth({"df": "not a dataframe", "covariates": ["x"]})

    def test_donor_weights_only_active(self, small_panel):
        df, _ = small_panel
        res = MicroSynth({
            "df": df, "outcome": "converted", "treat": "saw_ad",
            "unitid": "user_id", "time": "week", "covariates": COVS,
            "run_inference": False, "display_graphs": False,
        }).fit()
        # All keys in donor_weights are control users with w > 0.
        active = sum(1 for w in res.design.w if w > 0)
        assert len(res.donor_weights) == active
        for k, v in res.donor_weights.items():
            assert isinstance(k, str)
            assert v > 0
