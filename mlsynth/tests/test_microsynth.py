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
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.microsynth_helpers.dual_solver import solve_microsynth_dual
from mlsynth.utils.microsynth_helpers.panel_qp import solve_panel_qp
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


class TestPanelQP:
    """The microsynth (Robbins et al.) panel weight QP.

    Faithful port of ``microsynth::my.qp`` (LowRankQP): exactly balance the
    hard constraints (intercept + covariates), least-squares-fit the soft
    constraints (lagged outcomes), with ``w >= 0``. A strictly-convex ridge
    makes the otherwise rank-deficient solution unique (the maximum-ESS /
    minimum-norm point on microsynth's optimal face).
    """

    def test_exact_balance_and_lag_fit(self):
        # Construct a panel where exact balance on both blocks is achievable
        # (treated totals lie in the cone spanned by control rows).
        rng = np.random.default_rng(0)
        n_C, n_T = 300, 40
        cov_C = rng.uniform(0.0, 5.0, (n_C, 3))
        lag_C = rng.uniform(0.0, 3.0, (n_C, 4))
        # pick a known nonneg combination summing to n_T as the "truth"
        w_true = rng.uniform(0.0, 1.0, n_C)
        w_true *= n_T / w_true.sum()
        hard_C = np.column_stack([np.ones(n_C), cov_C])
        hard_t = np.array([float(n_T), *(cov_C.T @ w_true)])
        soft_t = lag_C.T @ w_true
        sol = solve_panel_qp(hard_C, hard_t, lag_C, soft_t, ridge=1e-8)
        assert sol.converged
        assert (sol.w >= -1e-7).all()
        # hard block exactly balanced; weights sum to the treated count
        assert np.allclose(hard_C.T @ sol.w, hard_t, atol=1e-5)
        assert sol.w.sum() == pytest.approx(float(n_T), abs=1e-4)
        # soft (lagged-outcome) block fit to ~0 residual since achievable
        assert sol.soft_residual < 1e-3

    def test_ridge_makes_solution_unique_and_deterministic(self):
        # Heavily under-determined: many w achieve exact balance, so the
        # ridge term must pin a single reproducible answer.
        rng = np.random.default_rng(1)
        n_C, n_T = 500, 20
        cov_C = rng.standard_normal((n_C, 2))
        hard_C = np.column_stack([np.ones(n_C), cov_C])
        # feasible target: totals of a known nonneg weighting
        w0 = rng.uniform(0, 1, n_C); w0 *= n_T / w0.sum()
        hard_t = hard_C.T @ w0
        s1 = solve_panel_qp(hard_C, hard_t, None, None, ridge=1e-6)
        s2 = solve_panel_qp(hard_C, hard_t, None, None, ridge=1e-6)
        assert np.allclose(s1.w, s2.w, atol=1e-7)          # deterministic
        # min-norm property: the ridge solution is no larger in L2 than the
        # arbitrary feasible w0 that generated the targets.
        assert float(s1.w @ s1.w) <= float(w0 @ w0) + 1e-6

    def test_infeasible_hard_targets_raise(self):
        # Demand a covariate total no nonneg, count-constrained weighting can
        # reach (all controls below the per-unit target) -> infeasible QP.
        rng = np.random.default_rng(2)
        n_C, n_T = 100, 5
        cov_C = rng.uniform(0.0, 1.0, (n_C, 1))
        hard_C = np.column_stack([np.ones(n_C), cov_C])
        # mean covariate per treated unit forced above every control value
        hard_t = np.array([float(n_T), float(n_T) * 10.0])
        with pytest.raises(MlsynthEstimationError):
            solve_panel_qp(hard_C, hard_t, None, None, ridge=1e-6)

    def test_nonpositive_ridge_rejected(self):
        rng = np.random.default_rng(3)
        hard_C = np.column_stack([np.ones(20), rng.standard_normal(20)])
        hard_t = np.array([5.0, 0.0])
        with pytest.raises(MlsynthEstimationError, match="ridge"):
            solve_panel_qp(hard_C, hard_t, None, None, ridge=0.0)


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

class TestPanelMethod:
    """The microsynth panel method (weight_method='panel'): exactly balance
    covariates, least-squares-fit the pre-period outcome trajectory with
    w >= 0, and report per-period TOTAL effects on the treated area."""

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

    def test_panel_balance_weights_and_effect(self):
        df, eff, n_t = self._panel()
        res = MicroSynth({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "covariates": [],
            "outcome_lag_periods": list(range(6)),       # full pre-period window
            "weight_method": "panel", "run_inference": False,
            "display_graphs": False,
        }).fit()
        # panel weights are non-negative and sum to the treated count
        assert res.design.w.sum() == pytest.approx(float(n_t), abs=1e-3)
        assert (res.design.w >= -1e-7).all()
        # pre-period outcome trajectory fit to ~0 -> balanced
        assert np.max(np.abs(res.design.smd_after)) < 1e-2
        # per-period contrast is on totals -> ~ n_t * per-unit effect
        assert res.att == pytest.approx(n_t * eff, rel=0.25)
        assert res.gap_trajectory.shape == (3,)          # one per post period

    def test_panel_deterministic(self):
        df, _, _ = self._panel()
        cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                   covariates=[], outcome_lag_periods=list(range(6)),
                   weight_method="panel", run_inference=False, display_graphs=False)
        r1 = MicroSynth(cfg).fit()
        r2 = MicroSynth(cfg).fit()
        assert np.allclose(r1.design.w, r2.design.w, atol=1e-7)
        assert r1.att == pytest.approx(r2.att, abs=1e-9)

    def test_panel_with_covariates_only(self):
        # No lagged outcomes: pure hard covariate balance + ridge selection.
        df, _, n_t = self._panel()
        # add a time-invariant covariate
        rng = np.random.default_rng(7)
        per_unit = {u: rng.standard_normal() for u in df["unit"].unique()}
        df = df.assign(z=df["unit"].map(per_unit))
        res = MicroSynth({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "covariates": ["z"],
            "weight_method": "panel", "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.design.w.sum() == pytest.approx(float(n_t), abs=1e-3)
        assert np.max(np.abs(res.design.smd_after)) < 1e-2

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
