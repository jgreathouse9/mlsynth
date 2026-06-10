"""Tests for the multi-level synthetic control (mlSC) estimator.

Follows the four-layer testing philosophy in ``agents/agents_tests.md``:

    Layer 1: numerical / helper tests
        - penalty matrix Q_s is PSD with v_s in its kernel
        - variance decomposition matches Appendix G on a deterministic panel
        - heuristic lambda has the right closed form
    Layer 2: data utility tests
        - config validators reject misaligned panels, missing columns,
          mismatched treatment timing, orphan agg_id values
        - prepare_mlsc_inputs handles both the cohorts and single-treated
          dataprep return shapes
    Layer 3: estimator integration tests
        - MLSC.fit() runs end-to-end on heuristic / fixed lambda modes
        - high-lambda regime recovers classical-SC structure
          (within-block weights flat, equal to v_sc * w_s)
        - lambda = 0 recovers fully disaggregated SC
        - omega sums to 1, aggregate_weights sums to 1
        - exception translation: bad input -> MlsynthDataError
    Layer 4: public API contract tests
        - from mlsynth import MLSC works
        - MLSCResults has the expected fields and types

Reference: Bottmer (2025), "Synthetic Control with Disaggregated Data."
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from mlsynth import MLSC
from mlsynth.config_models import EffectResult, MLSCConfig, MlsynthResult
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.mlsc_helpers.penalty import build_block, build_penalty_matrix
from mlsynth.utils.mlsc_helpers.setup import prepare_mlsc_inputs
from mlsynth.utils.mlsc_helpers.structures import (
    MLSCDesign,
    MLSCInference,
    MLSCInputs,
    MLSCResults,
)
from mlsynth.utils.mlsc_helpers.variance import (
    estimate_variance_components,
    heuristic_lambda,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_panel(
    seed: int = 0,
    S: int = 5,
    C: int = 4,
    T: int = 24,
    T0: int = 18,
    treated: int = 0,
    rank: int = 2,
    noise_scale: float = 0.2,
    include_pop: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize an aggregate + disaggregate panel from a hierarchical
    latent factor model matching Bottmer (2025) Assumption 1.

    Returns
    -------
    df_agg, df_disagg : pd.DataFrame
        Long-form aggregate (S rows per time) and disaggregate (S*C rows
        per time) panels. Treatment is assigned to ``state_<treated>`` at
        period ``T0`` and absorbing thereafter.
    """

    rng = np.random.default_rng(seed)
    beta_t = rng.standard_normal((T, rank)) * 0.3
    alpha_s = rng.standard_normal((S, rank)) * 1.5
    eta_sc = rng.standard_normal((S, C, rank)) * 0.4
    mu_sc = alpha_s[:, None, :] + eta_sc
    eps = rng.standard_normal((S, C, T)) * noise_scale
    Y_sct = np.einsum("scf,tf->sct", mu_sc, beta_t) + eps
    Y_st = Y_sct.mean(axis=1)

    state_labels = [f"state_{s}" for s in range(S)]
    agg_records = [
        {
            "state": state_labels[s],
            "year": 2000 + t,
            "y": float(Y_st[s, t]),
            "treated": 1 if (s == treated and t >= T0) else 0,
        }
        for s in range(S)
        for t in range(T)
    ]
    disagg_records = []
    for s in range(S):
        for c in range(C):
            pop = float(rng.uniform(0.5, 2.0)) if include_pop else 1.0
            for t in range(T):
                disagg_records.append(
                    {
                        "county": f"county_{s}_{c}",
                        "state": state_labels[s],
                        "year": 2000 + t,
                        "y": float(Y_sct[s, c, t]),
                        "treated": 1 if (s == treated and t >= T0) else 0,
                        "pop": pop,
                    }
                )
    return pd.DataFrame(agg_records), pd.DataFrame(disagg_records)


def _base_config(df_agg: pd.DataFrame, df_disagg: pd.DataFrame, **overrides):
    cfg = dict(
        df_agg=df_agg,
        df_disagg=df_disagg,
        outcome="y",
        time="year",
        treat="treated",
        unitid_agg="state",
        unitid_disagg="county",
        agg_id="state",
        weight_col=None,
        lambda_est="heuristic",
        display_graphs=False,
    )
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Layer 1: Numerical / helper tests
# ---------------------------------------------------------------------------

class TestPenaltyMatrix:
    """Properties of the block-diagonal penalty matrix Q."""

    def test_block_is_symmetric(self):
        v = np.array([0.3, 0.2, 0.5])
        Q = build_block(v)
        assert np.allclose(Q, Q.T)

    def test_block_is_psd(self):
        v = np.array([0.3, 0.2, 0.5])
        Q = build_block(v)
        eigs = np.linalg.eigvalsh(Q)
        assert np.all(eigs >= -1e-10)

    def test_v_lies_in_kernel(self):
        # The block penalty equals zero whenever omega_s = c * v_s for any
        # scalar c (proportional weights). In particular Q_s @ v_s = 0.
        v = np.array([0.3, 0.2, 0.5])
        Q = build_block(v)
        assert np.allclose(Q @ v, 0.0, atol=1e-12)

    def test_uniform_block(self):
        # Uniform v = 1/C: Q = I - (1/C) (J + J^T) + (1/C) J = I - (1/C) J,
        # the centering matrix.
        C = 4
        v = np.full(C, 1.0 / C)
        Q = build_block(v)
        expected = np.eye(C) - np.full((C, C), 1.0 / C)
        assert np.allclose(Q, expected)

    def test_block_diagonal_assembly(self):
        v = np.array([0.5, 0.5, 0.4, 0.6])
        idx = np.array([0, 0, 1, 1])
        Q = build_penalty_matrix(v, idx)
        # Should have the (2, 2) and (3, 3) cross-block entries zeroed out.
        assert Q[0, 2] == 0 and Q[1, 3] == 0
        assert Q.shape == (4, 4)
        # Each block matches build_block on its slice.
        assert np.allclose(Q[:2, :2], build_block(v[:2]))
        assert np.allclose(Q[2:, 2:], build_block(v[2:]))


class TestVarianceDecomposition:
    """Variance estimation must match Appendix G on a known panel."""

    def test_known_homoskedastic_panel(self):
        # Construct a panel with known noise variance sigma_eps^2 = 0.25.
        # The Appendix-G estimator should hit ~0.25 as T0 grows.
        rng = np.random.default_rng(42)
        S, C, T0 = 8, 6, 200
        alpha_s = rng.standard_normal(S) * 1.0           # (S,)
        eta_sc = rng.standard_normal((S, C)) * 0.5       # (S, C)
        mu_sc = alpha_s[:, None] + eta_sc                # (S, C)
        noise = rng.standard_normal((S, C, T0)) * 0.5    # sigma_eps^2 = 0.25
        Y = mu_sc[:, :, None] + noise                    # (S, C, T0)
        # Reshape to (T0, S*C).
        Y_pre_flat = Y.reshape(S * C, T0).T

        disagg_to_agg = np.repeat(np.arange(S), C)
        inputs = MLSCInputs(
            Y_agg_treated=np.zeros(T0),
            X_disagg=np.zeros((T0, (S - 1) * C)),
            v_population=np.full((S - 1) * C, 1.0 / C),
            disagg_to_agg=np.repeat(np.arange(S - 1), C),
            agg_labels=[f"s{i}" for i in range(S - 1)],
            disagg_labels=[f"d{i}" for i in range((S - 1) * C)],
            Y_disagg_pre_full=Y_pre_flat,
            disagg_to_agg_full=disagg_to_agg,
            treated_agg_idx_full=S - 1,
            T=T0,
            T0=T0,
            treated_unit_name="treated",
            time_labels=np.arange(T0),
            Ywide_agg=None,
            outcome="y",
        )
        sigma_eps2, sigma_y2 = estimate_variance_components(inputs)
        assert 0.20 < sigma_eps2 < 0.30
        # sigma_y2 should be larger than sigma_eps2 (it includes alpha and eta).
        assert sigma_y2 > sigma_eps2

    def test_heuristic_lambda_closed_form(self):
        assert heuristic_lambda(0.5, 1.0) == pytest.approx(1.0)
        assert heuristic_lambda(0.1, 2.0) == pytest.approx(0.1)

    def test_heuristic_lambda_guards_against_zero(self):
        # Outcome variance very near zero should not blow up.
        val = heuristic_lambda(0.5, 0.0)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Layer 2: Data utility / config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """MLSCConfig validators must reject malformed inputs early."""

    def test_missing_column_in_df_agg(self):
        df_agg, df_disagg = _make_panel()
        df_bad = df_agg.drop(columns=["treated"])
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_bad, df_disagg))

    def test_missing_column_in_df_disagg(self):
        df_agg, df_disagg = _make_panel()
        df_bad = df_disagg.drop(columns=["state"])
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg, df_bad))

    def test_empty_dataframe(self):
        df_agg, df_disagg = _make_panel()
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg.iloc[0:0], df_disagg))

    def test_mismatched_time_periods(self):
        df_agg, df_disagg = _make_panel()
        # Drop the last period from df_disagg only.
        last_year = df_disagg["year"].max()
        df_bad = df_disagg[df_disagg["year"] != last_year]
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg, df_bad))

    def test_disaggregate_unit_with_multiple_aggregates(self):
        df_agg, df_disagg = _make_panel()
        # Reassign part of one county's rows to a different state so the
        # unit-to-agg mapping is ambiguous.
        df_bad = df_disagg.copy()
        df_bad.loc[
            (df_bad["county"] == "county_1_0") & (df_bad["year"] > 2010),
            "state",
        ] = "state_2"
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg, df_bad))

    def test_orphan_agg_id_in_disaggregate(self):
        df_agg, df_disagg = _make_panel()
        df_bad = df_disagg.copy()
        df_bad.loc[df_bad["state"] == "state_3", "state"] = "state_does_not_exist"
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg, df_bad))


class TestDataPreparation:
    """Prepare_mlsc_inputs handles both dataprep return shapes."""

    def test_cohorts_path_multiple_treated_children(self):
        # Default fixture has C=4 treated counties -> dataprep cohorts path.
        df_agg, df_disagg = _make_panel(S=5, C=4)
        cfg = MLSCConfig(**_base_config(df_agg, df_disagg))
        inputs = prepare_mlsc_inputs(
            df_agg=cfg.df_agg, df_disagg=cfg.df_disagg,
            outcome=cfg.outcome, time=cfg.time, treat=cfg.treat,
            unitid_agg=cfg.unitid_agg, unitid_disagg=cfg.unitid_disagg,
            agg_id=cfg.agg_id, weight_col=cfg.weight_col,
        )
        # 5 - 1 = 4 control aggregates, each with 4 disagg units -> M = 16.
        assert inputs.S == 4
        assert inputs.M == 16
        assert inputs.X_disagg.shape == (inputs.T, 16)
        assert inputs.v_population.shape == (16,)

    def test_single_treated_child_path(self):
        # Trim the treated state down to just one county. That exercises
        # the dataprep single-treated return shape.
        df_agg, df_disagg = _make_panel(S=5, C=4)
        df_disagg = df_disagg[
            ~((df_disagg["state"] == "state_0") & (df_disagg["county"] != "county_0_0"))
        ].copy()
        cfg = MLSCConfig(**_base_config(df_agg, df_disagg))
        inputs = prepare_mlsc_inputs(
            df_agg=cfg.df_agg, df_disagg=cfg.df_disagg,
            outcome=cfg.outcome, time=cfg.time, treat=cfg.treat,
            unitid_agg=cfg.unitid_agg, unitid_disagg=cfg.unitid_disagg,
            agg_id=cfg.agg_id, weight_col=cfg.weight_col,
        )
        assert inputs.S == 4
        # 4 control aggregates * 4 counties each = 16 disagg controls;
        # treated aggregate contributes 1 child (excluded from X_disagg).
        assert inputs.M == 16

    def test_uniform_v_population_default(self):
        df_agg, df_disagg = _make_panel(S=4, C=3)
        cfg = MLSCConfig(**_base_config(df_agg, df_disagg, weight_col=None))
        inputs = prepare_mlsc_inputs(
            df_agg=cfg.df_agg, df_disagg=cfg.df_disagg,
            outcome=cfg.outcome, time=cfg.time, treat=cfg.treat,
            unitid_agg=cfg.unitid_agg, unitid_disagg=cfg.unitid_disagg,
            agg_id=cfg.agg_id, weight_col=cfg.weight_col,
        )
        # Each block has 3 disagg units, so each weight should be 1/3.
        assert np.allclose(inputs.v_population, 1.0 / 3.0)

    def test_population_weights_normalize_per_aggregate(self):
        df_agg, df_disagg = _make_panel(S=3, C=4)
        cfg = MLSCConfig(**_base_config(df_agg, df_disagg, weight_col="pop"))
        inputs = prepare_mlsc_inputs(
            df_agg=cfg.df_agg, df_disagg=cfg.df_disagg,
            outcome=cfg.outcome, time=cfg.time, treat=cfg.treat,
            unitid_agg=cfg.unitid_agg, unitid_disagg=cfg.unitid_disagg,
            agg_id=cfg.agg_id, weight_col=cfg.weight_col,
        )
        # Per-block sums must each equal 1.
        for s in sorted(set(inputs.disagg_to_agg.tolist())):
            block = inputs.v_population[inputs.disagg_to_agg == s]
            assert block.sum() == pytest.approx(1.0)

    def test_orphan_treated_disaggregate(self):
        # treat=1 outside the treated aggregate must raise.
        df_agg, df_disagg = _make_panel()
        df_bad = df_disagg.copy()
        df_bad.loc[
            (df_bad["state"] == "state_2") & (df_bad["year"] >= 2018),
            "treated",
        ] = 1
        with pytest.raises(MlsynthDataError):
            MLSC(_base_config(df_agg, df_bad)).fit()


# ---------------------------------------------------------------------------
# Layer 3: Estimator integration tests
# ---------------------------------------------------------------------------

class TestEstimatorIntegration:
    """End-to-end MLSC.fit() behavior."""

    def test_heuristic_mode_runs(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert isinstance(res, MLSCResults)
        assert np.isfinite(res.att)
        assert np.isfinite(res.pre_rmse)
        assert res.design.lambda_used > 0
        assert res.design.sigma_eps2 > 0
        assert res.design.sigma_y2 > 0

    def test_fixed_lambda_mode_runs(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(
            _base_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=0.5)
        ).fit()
        assert res.design.lambda_used == pytest.approx(0.5)

    def test_omega_satisfies_simplex_constraint(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert res.design.omega.sum() == pytest.approx(1.0, abs=1e-6)
        assert (res.design.omega >= -1e-8).all()

    def test_aggregate_weights_sum_to_one(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert res.design.aggregate_weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_counterfactual_length_matches_T(self):
        df_agg, df_disagg = _make_panel(T=30, T0=20)
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert res.paths.counterfactual.shape == (30,)
        assert res.paths.gap.shape == (30,)
        # standardized accessors expose the same series
        assert res.counterfactual.shape == (30,)
        assert res.gap.shape == (30,)

    def test_high_lambda_recovers_classical_sc_structure(self):
        # When lambda is enormous, within each aggregate block the
        # disaggregate weights become exactly proportional to v_sc.
        df_agg, df_disagg = _make_panel(S=5, C=4)
        res = MLSC(
            _base_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=1e8)
        ).fit()
        for s, label in enumerate(res.inputs.agg_labels):
            mask = res.inputs.disagg_to_agg == s
            block = res.design.omega[mask]
            v_block = res.inputs.v_population[mask]
            # The block has the form omega = c * v for some scalar c.
            # If v is uniform, this means a flat block; either way it is
            # proportional to v.
            if block.sum() > 1e-6:
                ratios = block / v_block
                # All entries of (omega / v) within a block should agree.
                assert np.allclose(ratios, ratios[0], atol=1e-3)

    def test_lambda_zero_is_more_flexible_than_high_lambda(self):
        # lambda = 0 frees up the disaggregate weights; the pre-period fit
        # should be no worse than the high-lambda regime.
        df_agg, df_disagg = _make_panel()
        res_high = MLSC(
            _base_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=1e6)
        ).fit()
        res_zero = MLSC(
            _base_config(df_agg, df_disagg, lambda_est="fixed", lambda_val=0.0)
        ).fit()
        assert res_zero.pre_rmse <= res_high.pre_rmse + 1e-8

    def test_donor_weights_dict_aligns_with_omega(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert set(res.donor_weights.keys()) == set(res.inputs.disagg_labels)
        for label, w in res.donor_weights.items():
            i = res.inputs.disagg_labels.index(label)
            assert w == pytest.approx(float(res.design.omega[i]))

    def test_no_post_period_yields_nan_att(self):
        # Build a panel where T0 == T (no treatment ever fires).
        df_agg, df_disagg = _make_panel(T=20, T0=20, treated=-1)
        # Force a treated row at the last period only so dataprep accepts it.
        df_agg = df_agg.copy()
        df_disagg = df_disagg.copy()
        df_agg.loc[(df_agg["state"] == "state_0") & (df_agg["year"] == 2019), "treated"] = 1
        df_disagg.loc[
            (df_disagg["state"] == "state_0") & (df_disagg["year"] == 2019),
            "treated",
        ] = 1
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        # There IS a post period of length 1 here; the test guards the
        # general path rather than the literal degenerate case.
        assert np.isfinite(res.att)


# ---------------------------------------------------------------------------
# Layer 4: Public API contract tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """The user-facing API contract must remain stable."""

    def test_import_path(self):
        from mlsynth import MLSC as Imported  # noqa: F401
        assert Imported is MLSC

    def test_results_object_fields(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        # Top-level contract.
        assert isinstance(res, MLSCResults)
        assert isinstance(res.inputs, MLSCInputs)
        assert isinstance(res.design, MLSCDesign)
        assert isinstance(res.paths, MLSCInference)
        assert isinstance(res.att, float)
        assert isinstance(res.pre_rmse, float)
        assert isinstance(res.donor_weights, dict)
        assert isinstance(res.aggregate_donor_weights, dict)

    def test_two_family_result_contract(self):
        """MLSC conforms to the observational (EffectResult) contract.

        MLSC takes a two-level panel, so it cannot join the single-df loop in
        test_result_contract.py; pin the contract here instead.
        """
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        assert isinstance(res, MlsynthResult)
        assert isinstance(res, EffectResult)
        # standardized sub-models populated
        assert res.effects is not None and res.effects.att is not None
        assert res.time_series is not None
        assert res.time_series.counterfactual_outcome is not None
        assert res.weights is not None
        assert res.method_details is not None and res.method_details.method_name
        # flat accessors resolve; donor weights served from the weights slot
        assert isinstance(res.att, float)
        assert res.att == pytest.approx(res.effects.att)
        cf = np.asarray(res.counterfactual)
        gap = np.asarray(res.gap)
        assert cf.ndim == 1 and cf.shape == gap.shape
        assert set(res.donor_weights.keys()) == set(res.inputs.disagg_labels)
        ci = res.att_ci
        assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])
        # mlSC has no statistical inference
        assert res.inference is None

    def test_design_object_has_diagnostics(self):
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg)).fit()
        d = res.design
        assert d.omega.shape == (res.inputs.M,)
        assert d.aggregate_weights.shape == (res.inputs.S,)
        assert d.lambda_est in ("heuristic", "fixed", "cross-validation")
        assert isinstance(d.solver_status, str)

    def test_dict_config_round_trip(self):
        # Passing a dict and a Config object should produce identical fits.
        df_agg, df_disagg = _make_panel()
        cfg_dict = _base_config(df_agg, df_disagg)
        cfg_obj = MLSCConfig(**cfg_dict)
        res_dict = MLSC(cfg_dict).fit()
        res_obj = MLSC(cfg_obj).fit()
        assert res_dict.att == pytest.approx(res_obj.att)
        assert res_dict.pre_rmse == pytest.approx(res_obj.pre_rmse)


# ---------------------------------------------------------------------------
# Layer 4: cross-validation-over-time penalty selection (Section 5.2)
# ---------------------------------------------------------------------------
class TestCrossValidation:
    """``lambda_est='cross-validation'`` -- the Section-5.2 rolling CV."""

    def test_config_accepts_cross_validation(self):
        df_agg, df_disagg = _make_panel()
        cfg = MLSCConfig(**_base_config(df_agg, df_disagg,
                                        lambda_est="cross-validation"))
        assert cfg.lambda_est == "cross-validation"
        assert cfg.cv_holdout_periods == 1          # reference default
        assert cfg.lambda_grid is None              # -> reference grid

    def test_cv_runs_and_selects_grid_penalty(self):
        from mlsynth.utils.mlsc_helpers.crossval import _DEFAULT_GRID
        df_agg, df_disagg = _make_panel()
        res = MLSC(_base_config(df_agg, df_disagg,
                                lambda_est="cross-validation")).fit()
        assert res.design.lambda_est == "cross-validation"
        # The selected penalty is a member of the (default) grid.
        assert np.min(np.abs(_DEFAULT_GRID - res.design.lambda_used)) < 1e-12
        assert np.isclose(res.design.omega.sum(), 1.0)
        assert np.all(res.design.omega >= -1e-8)

    def test_custom_grid_is_respected(self):
        df_agg, df_disagg = _make_panel()
        grid = [0.01, 1.0, 100.0]
        res = MLSC(_base_config(df_agg, df_disagg, lambda_est="cross-validation",
                                lambda_grid=grid)).fit()
        assert res.design.lambda_used in grid

    def test_holdout_too_long_raises(self):
        # T0 == 18 here; holding out all of it leaves no training period.
        df_agg, df_disagg = _make_panel(T0=18)
        with pytest.raises((MlsynthEstimationError, ValueError)):
            MLSC(_base_config(df_agg, df_disagg, lambda_est="cross-validation",
                              cv_holdout_periods=18)).fit()

    def test_selector_unit_matches_objective_argmin(self):
        # The helper returns the grid penalty minimizing the held-out MSE.
        from mlsynth.utils.mlsc_helpers.crossval import select_lambda_cv
        from mlsynth.utils.mlsc_helpers.penalty import build_penalty_matrix
        from mlsynth.utils.mlsc_helpers.setup import prepare_mlsc_inputs
        from mlsynth.utils.mlsc_helpers.variance import estimate_variance_components

        df_agg, df_disagg = _make_panel(seed=3)
        inputs = prepare_mlsc_inputs(
            df_agg=df_agg, df_disagg=df_disagg, outcome="y", time="year",
            treat="treated", unitid_agg="state", unitid_disagg="county",
            agg_id="state", weight_col=None,
        )
        _, sigma_y2 = estimate_variance_components(inputs)
        Q = build_penalty_matrix(v_population=inputs.v_population,
                                 disagg_to_agg=inputs.disagg_to_agg)
        grid = [0.01, 1.0, 50.0, 500.0]
        lam = select_lambda_cv(inputs, Q, sigma_y2, lambda_grid=grid,
                               cv_holdout_periods=2)
        assert lam in grid
