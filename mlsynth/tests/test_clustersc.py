"""Tests for the modernized CLUSTERSC estimator.

Layered along agents_tests.md:

* Layer 1 (numerical helpers): per-method run_pcr / run_rpca on tiny
  synthetic panels.
* Layer 2 (data utilities): prepare_clustersc_inputs pivot,
  validation paths.
* Layer 3 (estimator integration): CLUSTERSC.fit across the three
  methods, the two estimator modes, both RPCA variants, primary
  selection.
* Layer 4 (public API contracts): top-level import, frozen
  dataclasses, legacy-field accommodation in CLUSTERSCConfig.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC
from mlsynth.config_models import CLUSTERSCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.clustersc_helpers.pcr import run_pcr
from mlsynth.utils.clustersc_helpers.rpca import run_rpca
from mlsynth.utils.clustersc_helpers.setup import prepare_clustersc_inputs
from mlsynth.utils.clustersc_helpers.structures import (
    CLUSTERSCInference,
    CLUSTERSCInputs,
    CLUSTERSCResults,
    MethodFit,
)


# ----------------------------------------------------------------------
# Shared synthetic-panel fixture
# ----------------------------------------------------------------------

def _factor_panel(
    J: int = 12, T_pre: int = 14, T_post: int = 6, r: int = 2,
    tau_true: float = 1.0, seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r))
    lam = rng.standard_normal((J + 1, r))
    eps = rng.standard_normal((T, J + 1)) * 0.4
    Y0 = F @ lam.T + eps
    Y = Y0.copy()
    Y[T_pre:, 0] += tau_true
    rows = [
        {"unit": j, "time": t, "y": float(Y[t, j]),
         "D": int(j == 0 and t >= T_pre)}
        for j in range(J + 1) for t in range(T)
    ]
    return pd.DataFrame(rows), tau_true


@pytest.fixture
def panel():
    return _factor_panel()


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestPCRHelper:
    def test_run_pcr_ols_recovers_att(self, panel):
        df, tau = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit, credible = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=True,
            estimator="frequentist",
        )
        assert isinstance(fit, MethodFit)
        assert fit.name == "pcr_ols"
        assert credible is None
        assert abs(fit.att - tau) < 0.5
        # Paper-aligned metadata: rank, rank_method, clustering must be set.
        assert "rank" in fit.metadata and fit.metadata["rank"] >= 1
        assert fit.metadata["rank_method"] == "cumvar"
        assert fit.metadata["clustering"] is True

    def test_standardize_for_rank_changes_cumvar_pick(self, panel):
        """On a panel with non-trivial intercepts, standardising the
        donor matrix before the cumvar comparison must pick a higher
        rank than the raw rule (because the leading singular value no
        longer absorbs the level information).
        """
        df, _ = panel
        # Add a unit-specific intercept to the donor outcomes so the
        # uncentered SVD is dominated by a single direction.
        df = df.copy()
        df["y"] = df["y"] + df["unit"].astype(float) * 100.0
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit_std, _ = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=False,
            estimator="frequentist",
            rank_method="cumvar", cumvar_threshold=0.95,
            standardize_for_rank=True,
        )
        fit_raw, _ = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=False,
            estimator="frequentist",
            rank_method="cumvar", cumvar_threshold=0.95,
            standardize_for_rank=False,
        )
        # The standardised path should pick a strictly higher rank
        # than the raw path on this artificial-intercept panel.
        assert fit_std.metadata["rank"] > fit_raw.metadata["rank"]

    def test_project_denoised_flag_recorded(self, panel):
        """`project_denoised` is a genuine knob (not a no-op): with
        HSVT applied to the pre-period only, projecting through raw
        vs denoised post-period donors gives different counterfactuals.
        Just verify both paths run and metadata records the choice.
        """
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit_raw, _ = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=False,
            estimator="frequentist", rank=3, project_denoised=False,
        )
        fit_den, _ = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=False,
            estimator="frequentist", rank=3, project_denoised=True,
        )
        assert fit_raw.metadata["project_denoised"] is False
        assert fit_den.metadata["project_denoised"] is True
        # Both produce a finite counterfactual of the right shape.
        assert fit_raw.counterfactual.shape == fit_den.counterfactual.shape
        assert np.all(np.isfinite(fit_raw.counterfactual))
        assert np.all(np.isfinite(fit_den.counterfactual))

    def test_explicit_rank_overrides_default(self, panel):
        """Passing `rank=k` must produce a rank-k truncation regardless
        of the default `rank_method='cumvar'`.
        """
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit, _ = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=False,
            estimator="frequentist", rank=3,
        )
        assert fit.metadata["rank"] == 3
        assert fit.metadata["rank_method"] == "cumvar"  # original method preserved

    def test_run_pcr_simplex_returns_simplex_weights(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit, credible = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="SIMPLEX", clustering=False,
            estimator="frequentist",
        )
        assert fit.name == "pcr_simplex"
        assert credible is None
        weights = np.array(list(fit.donor_weights.values()))
        # Simplex weights must be non-negative and sum to 1.
        assert np.all(weights >= -1e-6)
        assert abs(float(weights.sum()) - 1.0) < 1e-4

    def test_run_pcr_bayesian_returns_per_period_bands(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit, credible = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=True,
            estimator="bayesian", alpha=0.10,
        )
        assert fit.name == "pcr_bayesian"
        assert credible is not None
        cf_lo, cf_hi = credible
        # Paper Algorithm 4 Step 5: bands are per-period, shape (T,).
        assert cf_lo.shape == (inputs.T,)
        assert cf_hi.shape == (inputs.T,)
        assert np.all(cf_lo <= cf_hi + 1e-9)
        assert np.all(np.isfinite(cf_lo)) and np.all(np.isfinite(cf_hi))

    def test_invalid_estimator_rejected(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        with pytest.raises(MlsynthEstimationError):
            run_pcr(
                treated_outcome=inputs.treated_outcome,
                donor_outcomes=inputs.donor_outcomes,
                donor_names=inputs.donor_names,
                T0=inputs.T0, estimator="bogus",
            )


class TestRPCAHelper:
    @pytest.mark.parametrize("rpca_method", ["PCP", "HQF"])
    def test_run_rpca_runs(self, panel, rpca_method):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit = run_rpca(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0,
            rpca_method=rpca_method,
            k_clusters=1,  # No cluster structure in the test panel.
        )
        assert isinstance(fit, MethodFit)
        # Paper-aligned name carries the solver variant.
        assert fit.name == f"rpca_{rpca_method.lower()}"
        assert fit.counterfactual.shape == (inputs.T,)
        assert fit.metadata["rpca_method"] == rpca_method
        # Algorithm 4 metadata: FPCA rank + treated cluster id + RPCA solver knobs.
        assert "fpca_rank" in fit.metadata
        assert "treated_cluster" in fit.metadata
        assert "k_clusters" in fit.metadata
        # Bayani's NNLS is non-negative.
        weights = np.array(list(fit.donor_weights.values()))
        assert np.all(weights >= -1e-9)

    def test_unknown_rpca_method_rejected(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        with pytest.raises(MlsynthEstimationError):
            run_rpca(
                treated_outcome=inputs.treated_outcome,
                donor_outcomes=inputs.donor_outcomes,
                donor_names=inputs.donor_names,
                T0=inputs.T0, rpca_method="BOGUS",
            )


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_pivot_assembles_inputs(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        assert isinstance(inputs, CLUSTERSCInputs)
        assert inputs.T == 20
        assert inputs.T0 == 14
        assert inputs.J == 12

    def test_missing_values_rejected(self, panel):
        df, _ = panel
        df.loc[5, "y"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_clustersc_inputs(
                df, outcome="y", treat="D", unitid="unit", time="time",
            )


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_default_pcr_fit(self, panel):
        df, tau = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        assert isinstance(res, CLUSTERSCResults)
        assert res.pcr is not None
        assert res.rpca is None
        assert res.selected_variant == "pcr"
        assert abs(res.att - tau) < 0.5

    def test_rpca_only_fit(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "rpca", "k_clusters": 1,
        }).fit()
        assert res.rpca is not None
        assert res.pcr is None
        assert res.selected_variant == "rpca"

    def test_both_methods_populated(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "both", "primary": "pcr",
            "k_clusters": 1,
        }).fit()
        assert res.pcr is not None and res.rpca is not None
        assert res.selected_variant == "pcr"
        # Primary alias picks PCR.
        assert res.att == res.pcr.att

    def test_primary_rpca_when_method_both(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "both", "primary": "rpca",
            "k_clusters": 1,
        }).fit()
        assert res.selected_variant == "rpca"
        assert res.att == res.rpca.att

    def test_bayesian_pcr_populates_credible_interval(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "pcr", "estimator": "bayesian", "alpha": 0.10,
        }).fit()
        assert res.inference.method == "bayesian_credible"
        lo, hi = res.inference.credible_interval
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi

    def test_shen_ci_for_frequentist_ols_pcr(self, panel):
        """Frequentist OLS PCR returns Shen-Ding-Sekhon-Yu (2023) CIs by default."""
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "pcr", "estimator": "frequentist",
        }).fit()
        assert res.inference.method == "shen_homoskedastic"
        shen = res.inference.shen
        assert shen is not None
        # Per-period structure: one CI per post-period.
        n_post = res.inputs.T - res.inputs.T0
        assert shen.per_period_gap.shape == (n_post,)
        assert shen.per_period_ci_vt.shape == (n_post, 2)
        # ATT CIs are finite and ordered.
        for ci in (shen.att_ci_hz, shen.att_ci_vt, shen.att_ci_dr):
            lo, hi = ci
            assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi

    def test_cft_pi_for_rpca(self, panel):
        """`compute_cft_pi=True` returns CFT prediction intervals for RPCA-SC."""
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "rpca", "rpca_method": "PCP", "k_clusters": 1,
            "compute_cft_pi": True, "cft_sims": 30,
        }).fit()
        assert res.inference.method == "cft_gaussian"
        cft = res.inference.cft
        assert cft is not None
        n_post = res.inputs.T - res.inputs.T0
        assert cft.per_period_gap.shape == (n_post,)
        assert cft.per_period_pi.shape == (n_post, 2)
        # PI bounds are ordered (lower <= upper).
        assert np.all(cft.per_period_pi[:, 0] <= cft.per_period_pi[:, 1] + 1e-9)
        att_lo, att_hi = cft.att_pi
        assert np.isfinite(att_lo) and np.isfinite(att_hi) and att_lo <= att_hi
        assert cft.sims == 30
        assert cft.sigma_e > 0

    def test_shen_ci_disabled(self, panel):
        """`compute_shen_ci=False` falls back to method='none'."""
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "pcr", "estimator": "frequentist",
            "compute_shen_ci": False,
        }).fit()
        assert res.inference.method == "none"
        assert res.inference.shen is None

    @pytest.mark.parametrize("objective", ["OLS", "SIMPLEX"])
    def test_pcr_objectives(self, panel, objective):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "pcr", "pcr_objective": objective,
        }).fit()
        assert np.isfinite(res.att)

    @pytest.mark.parametrize("rpca_method", ["PCP", "HQF"])
    def test_rpca_methods(self, panel, rpca_method):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "rpca", "rpca_method": rpca_method,
            "k_clusters": 1,
        }).fit()
        assert res.rpca.metadata["rpca_method"] == rpca_method

    def test_rpca_clustering_picks_treated_cohort(self):
        """Algorithm 4 Step 2: treated unit's cluster drives donor pool."""
        # Two-cluster panel: cluster A = the treated unit and donors 0-5 share
        # one factor; cluster B = donors 6-11 share another. Bayani's clustering
        # should isolate cluster A as the donor pool.
        rng = np.random.default_rng(42)
        J, T_pre, T_post = 12, 14, 6
        T = T_pre + T_post
        t = np.arange(T)
        clust_a = np.sin(t * 0.4)
        clust_b = np.cos(t * 0.4) * 5.0  # very different signature
        rows = []
        for j in range(J + 1):
            base = clust_a if j <= 6 else clust_b
            y = base + 0.05 * rng.standard_normal(T)
            if j == 0:
                y[T_pre:] += 1.0  # planted ATT on the treated unit
            for ti in range(T):
                rows.append({
                    "unit": j, "time": ti, "y": float(y[ti]),
                    "D": int(j == 0 and ti >= T_pre),
                })
        df = pd.DataFrame(rows)
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "rpca", "rpca_method": "PCP",
        }).fit()
        # All selected donors should come from cluster A (indices 1-6).
        selected = res.rpca.selected_donors.tolist()
        assert all(int(d) <= 6 for d in selected), selected
        assert res.rpca.metadata["k_clusters"] >= 2


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import CLUSTERSC as _CSC
        assert _CSC is CLUSTERSC

    def test_results_dataclasses_frozen(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        with pytest.raises(Exception):
            res.selected_variant = "rpca"
        with pytest.raises(Exception):
            res.pcr.att = 0.0

    def test_legacy_field_names_accepted(self, panel):
        # Old field names should still parse: objective, cluster,
        # Frequentist, ROB (and upper-case method tags).
        df, _ = panel
        cfg = CLUSTERSCConfig(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            method="BOTH", objective="OLS", cluster=True,
            Frequentist=False, ROB="HQF",
        )
        assert cfg.method == "both"
        assert cfg.pcr_objective == "OLS"
        assert cfg.clustering is True
        assert cfg.estimator == "bayesian"
        assert cfg.rpca_method == "HQF"

    def test_invalid_method_rejected(self, panel):
        df, _ = panel
        with pytest.raises(Exception):
            CLUSTERSCConfig(
                df=df, outcome="y", treat="D",
                unitid="unit", time="time", method="bogus",
            )
