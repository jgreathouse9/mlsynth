"""Tests for the SPILLSYNTH estimator (Cao & Dowd 2023, method='cd').

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): demeaned simplex SCM, leave-one-out
  batch, A-matrix construction, M-builder, and the spillover-adjusted
  closed-form (eq. 5 of the paper).
* Layer 2 (data utilities): panel ingestion + validation (single
  treated unit, no NaNs, no overlap between treated and affected,
  unknown labels rejected, pre/post non-empty).
* Layer 3 (estimator integration): end-to-end run on a synthetic
  factor-model panel; identifies (a) the treatment effect on the
  treated unit, (b) the spillover effect injected on a designated
  donor, (c) returns a frozen :class:`SpillSynthResults` with internally-
  consistent shapes.
* Layer 4 (public API contracts): top-level import, dict-vs-config
  equivalence, exception translation, the ``method='cd'`` dispatcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import SPILLSYNTHConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spillsynth_helpers import (
    CDFit,
    SpillSynthInputs,
    SpillSynthResults,
    build_A_example3,
    prepare_spillsynth_inputs,
    run_cd,
)
from mlsynth.utils.spillsynth_helpers.cd import (
    build_M,
    fit_demeaned_sc,
    fit_leave_one_out_sc,
    sp_estimate,
    vanilla_scm_path,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _spill_panel(
    *,
    N: int = 8,
    T: int = 40,
    T0: int = 30,
    treatment: float = -3.0,
    spillover: float = 1.5,
    spillover_idx: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """One-factor panel: unit 0 treated (effect ``treatment``), unit
    ``spillover_idx`` receives a spillover of ``spillover`` from t=T0.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=N)
    f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
    intercept = rng.uniform(-1, 1, size=N)
    Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
    Y[0, T0:] += treatment
    Y[spillover_idx, T0:] += spillover
    D = np.zeros((N, T))
    D[0, T0:] = 1
    rows = [
        {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _spill_panel()


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="year",
                display_graphs=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# Layer 1 -- numerical helpers
# ----------------------------------------------------------------------
class TestDemeanedSC:
    def test_simplex_and_perfect_fit(self):
        # Treated == convex combination of two controls -> zero residual.
        rng = np.random.default_rng(0)
        T = 30
        Y_c1 = rng.standard_normal(T)
        Y_c2 = rng.standard_normal(T)
        Y_t = 0.3 * Y_c1 + 0.7 * Y_c2
        Y_block = np.vstack([Y_t, Y_c1, Y_c2])
        a, b = fit_demeaned_sc(Y_block)
        assert b.shape == (3,)
        assert b[0] == 0.0
        assert b[1:].sum() == pytest.approx(1.0, abs=1e-6)
        assert (b[1:] >= -1e-9).all()
        assert np.allclose(b[1:], [0.3, 0.7], atol=5e-3)
        assert a == pytest.approx(0.0, abs=1e-6)

    def test_intercept_absorbs_mean_shift(self):
        # Shifting the treated unit by a constant should be absorbed
        # by the intercept, not the weights.
        rng = np.random.default_rng(1)
        T = 25
        donors = rng.standard_normal((3, T))
        weights = np.array([0.2, 0.3, 0.5])
        y = donors.T @ weights + 17.0
        Y_block = np.vstack([y, donors])
        a, b = fit_demeaned_sc(Y_block)
        assert a == pytest.approx(17.0, abs=1e-4)
        assert np.allclose(b[1:], weights, atol=1e-3)


class TestLeaveOneOut:
    def test_batch_shapes_and_self_weight(self, panel):
        inputs = prepare_spillsynth_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="year",
            affected_units=["u1"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        N = inputs.N
        assert a.shape == (N,)
        assert B.shape == (N, N)
        np.testing.assert_allclose(np.diag(B), 0.0, atol=1e-9)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-5)
        assert (B >= -1e-6).all()


class TestAMatrix:
    @pytest.mark.parametrize("N, p", [(5, 0), (5, 2), (10, 4)])
    def test_shape_and_structure(self, N, p):
        A = build_A_example3(N, p)
        assert A.shape == (N, 1 + p)
        assert A[0, 0] == 1.0
        for k in range(p):
            assert A[1 + k, 1 + k] == 1.0
        # Outside the leading (1+p) x (1+p) block must be zero.
        assert (A[1 + p:, :] == 0).all()

    def test_rejects_bad_dimensions(self):
        with pytest.raises(MlsynthDataError):
            build_A_example3(N=3, p=5)


class TestSpEstimate:
    def test_zero_spillover_returns_scalar_per_period(self, panel):
        # With p=0, A is N x 1 and gamma is the single treatment-effect
        # parameter per post-period, drawn from the full residual system
        # (not just the treated unit's own leave-one-out fit, so it does
        # NOT equal the vanilla SCM gap in general).
        inputs = prepare_spillsynth_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="year",
            affected_units=None,
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        M = build_M(B)
        gamma, alpha, _ = sp_estimate(
            inputs.Y_post, a=a, B=B, M=M, A=inputs.A,
        )
        assert gamma.shape == (1, inputs.T1)
        assert alpha.shape == (inputs.N, inputs.T1)
        # Only the treated row is nonzero by construction of A.
        assert (alpha[1:] == 0.0).all()

    def test_cond_AMA_finite_and_reasonable(self, panel):
        inputs = prepare_spillsynth_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="year",
            affected_units=["u1", "u2"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        M = build_M(B)
        _, _, cond = sp_estimate(
            inputs.Y_post, a=a, B=B, M=M, A=inputs.A,
        )
        assert np.isfinite(cond)
        assert cond < 1e6


# ----------------------------------------------------------------------
# Layer 2 -- data utility tests
# ----------------------------------------------------------------------
class TestPrepareInputs:
    def test_identifies_treated_and_order(self, panel):
        inputs = prepare_spillsynth_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="year",
            affected_units=["u2", "u5"],
        )
        assert inputs.treated_label == "u0"
        assert inputs.affected_labels == ("u2", "u5")
        # Row order: treated, affected, clean (sorted).
        assert inputs.Y.shape[0] == 8
        clean = inputs.clean_labels
        assert "u0" not in clean and "u2" not in clean and "u5" not in clean
        assert tuple(sorted(clean)) == clean

    def test_rejects_missing_column(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="nope", treat="treat",
                unitid="unit", time="year",
            )

    def test_rejects_multiple_treated(self, panel):
        panel = panel.copy()
        panel.loc[(panel["unit"] == "u3") & (panel["year"] >= 30), "treat"] = 1
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="y", treat="treat",
                unitid="unit", time="year",
            )

    def test_rejects_zero_treated(self, panel):
        panel = panel.copy()
        panel["treat"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="y", treat="treat",
                unitid="unit", time="year",
            )

    def test_rejects_treated_in_affected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="y", treat="treat",
                unitid="unit", time="year",
                affected_units=["u0", "u1"],
            )

    def test_rejects_unknown_affected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="y", treat="treat",
                unitid="unit", time="year",
                affected_units=["unknown_unit"],
            )

    def test_rejects_duplicate_affected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_spillsynth_inputs(
                panel, outcome="y", treat="treat",
                unitid="unit", time="year",
                affected_units=["u1", "u1"],
            )


# ----------------------------------------------------------------------
# Layer 3 -- estimator integration tests
# ----------------------------------------------------------------------
class TestEstimatorPipeline:
    def test_fit_returns_results(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        assert isinstance(res, SpillSynthResults)
        assert isinstance(res.cd, CDFit)
        assert isinstance(res.inputs, SpillSynthInputs)

    def test_internally_consistent_shapes(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1", "u2"])).fit()
        T1 = res.inputs.T1
        assert res.gap.shape == (T1,)
        assert res.gap_scm.shape == (T1,)
        assert res.counterfactual.shape == (T1,)
        assert res.counterfactual_scm.shape == (T1,)
        # gap = y_treated_post - counterfactual
        y_post = res.inputs.Y_post[0]
        np.testing.assert_allclose(res.gap, y_post - res.counterfactual, atol=1e-9)
        np.testing.assert_allclose(res.gap_scm,
                                   y_post - res.counterfactual_scm, atol=1e-9)
        # SP gap equals row 0 of alpha.
        np.testing.assert_allclose(res.gap, res.cd.alpha[0], atol=1e-9)
        # Clean-control rows of alpha are identically zero.
        assert np.allclose(res.cd.alpha[res.inputs.p + 1:], 0.0, atol=1e-12)

    def test_recovers_treatment_effect_with_correct_structure(self):
        # Synthetic DGP: treatment = -3 on unit 0, spillover = +1.5 on u1.
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=1)
        res = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        assert res.att == pytest.approx(-3.0, abs=0.6)
        spill_u1 = res.spillover_effects["u1"].mean()
        assert spill_u1 == pytest.approx(1.5, abs=0.6)

    def test_misspecified_spillover_biases_scm_but_not_sp(self):
        # Inject spillover but tell vanilla SCM nothing about it.
        df = _spill_panel(treatment=-3.0, spillover=2.0,
                          spillover_idx=1, seed=2)
        # Vanilla SCM (no spillover declared) -- expect bias toward 0 because
        # the spillover-affected donor inflates the synthetic California.
        sc_only = SPILLSYNTH(_cfg(df, affected_units=None)).fit()
        sp_aware = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        # SP-aware should be closer to the true -3.0 than vanilla SCM.
        assert abs(sp_aware.att + 3.0) <= abs(sc_only.att + 3.0)

    def test_no_affected_units_runs_and_returns_finite_att(self, panel):
        # p=0 is the corner case where SP and vanilla SCM disagree --
        # SP aggregates across the full residual system via M and (I-B),
        # vanilla SCM uses only the treated unit's own leave-one-out fit.
        # We require both to be finite, but they need not coincide.
        res = SPILLSYNTH(_cfg(panel, affected_units=None)).fit()
        assert np.isfinite(res.att)
        assert np.isfinite(res.att_scm)
        assert res.gap.shape == (res.inputs.T1,)
        assert res.gap_scm.shape == (res.inputs.T1,)


class TestPTestInference:
    """Cao-Dowd Section 4.2 P-test wiring."""

    def test_pipeline_populates_inference_fields(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1", "u2"])).fit()
        tt = res.cd.treatment_test
        T1 = res.inputs.T1
        T0 = res.inputs.T0
        assert tt is not None
        assert tt.P_post.shape == (T1,)
        assert tt.P_pre.shape == (T0,)
        assert tt.p_value.shape == (T1,)
        assert isinstance(tt.cutoff_05, float)
        assert tt.reject_05.shape == (T1,)
        assert tt.reject_05.dtype == bool
        # p_values lie in [0, 1].
        assert ((tt.p_value >= 0.0) & (tt.p_value <= 1.0)).all()

    def test_spillover_test_keys_match_affected_labels(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u2", "u3"])).fit()
        assert set(res.cd.spillover_tests.keys()) == {"u2", "u3"}
        for label, t in res.cd.spillover_tests.items():
            assert t.P_post.shape == (res.inputs.T1,)
            assert ((t.p_value >= 0.0) & (t.p_value <= 1.0)).all()

    def test_rejects_under_large_treatment_effect(self):
        # Treatment = -8 on unit 0, no spillover; SP test should reject.
        df = _spill_panel(treatment=-8.0, spillover=0.0,
                          spillover_idx=1, seed=11)
        res = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        tt = res.cd.treatment_test
        # All post-period p-values should be small under a large true effect.
        assert (tt.p_value < 0.10).all(), tt.p_value
        assert tt.reject_05.all()

    def test_does_not_reject_under_zero_effect(self):
        # Treatment = 0, no spillover; per-period p-values should not be
        # systematically tiny. Tests behaviour, not formal size.
        df = _spill_panel(treatment=0.0, spillover=0.0,
                          spillover_idx=1, seed=23)
        res = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        tt = res.cd.treatment_test
        # Mean p-value across post-period bounded away from 0.
        assert tt.p_value.mean() > 0.05, tt.p_value


# ----------------------------------------------------------------------
# Layer 4 -- public API contracts
# ----------------------------------------------------------------------
class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import SPILLSYNTH as _S
        from mlsynth.config_models import SPILLSYNTHConfig as _C
        assert _S is SPILLSYNTH
        assert _C is SPILLSYNTHConfig

    def test_dict_config_equivalent_to_typed_config(self, panel):
        r1 = SPILLSYNTH(SPILLSYNTHConfig(**_cfg(panel, affected_units=["u1"]))).fit()
        r2 = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        assert r1.att == pytest.approx(r2.att)
        np.testing.assert_allclose(r1.cd.B, r2.cd.B, atol=1e-9)

    def test_invalid_config_translates(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(panel, method="not-a-method"))

    def test_unknown_affected_unit_via_dict_path(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(panel, affected_units=["totally_unknown"]))

    def test_treated_in_affected_via_dict_path(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(panel, affected_units=["u0", "u1"]))
