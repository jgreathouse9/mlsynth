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

from pathlib import Path

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
    build_A_distance_decay,
    build_A_example3,
    build_A_homogeneous,
    build_A_per_unit,
    kappa_A_test,
    prepare_spillsynth_inputs,
    run_cd,
    select_A_by_kappa,
)
from mlsynth.utils.spillsynth_helpers.cd import (
    build_M,
    fit_demeaned_sc,
    fit_leave_one_out_sc,
    sp_estimate,
    vanilla_scm_path,
)
from mlsynth.utils.spillsynth_helpers.cd.estimation import (
    estimate_omega_from_pre_residuals, sp_estimate_weighted,
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

    def test_accepts_multiple_treated_same_time(self, panel):
        # Cao-Dowd v3 Section S.1.2: multiple treated units sharing a
        # common intervention time are supported.
        panel = panel.copy()
        panel.loc[(panel["unit"] == "u3") & (panel["year"] >= 30), "treat"] = 1
        inputs = prepare_spillsynth_inputs(
            panel, outcome="y", treat="treat",
            unitid="unit", time="year",
        )
        assert inputs.n_treated == 2
        assert set(inputs.treated_labels) == {"u0", "u3"}
        # Treated units are rows 0 and 1 of the reordered panel.
        assert inputs.A.shape[1] == 2                  # n_treated + p=0

    def test_rejects_multiple_treated_different_times(self, panel):
        # Different intervention times = staggered adoption = out of
        # scope for S.1.2.
        panel = panel.copy()
        panel.loc[(panel["unit"] == "u3") & (panel["year"] >= 25), "treat"] = 1
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


class TestAlternativeAStructures:
    """v3 Examples 1, 2, 3 of Cao-Dowd."""

    def test_build_A_per_unit_shapes(self):
        # v3 Example 1 (default, per-unit free coefficients).
        A = build_A_per_unit(N=6, p=2)
        assert A.shape == (6, 3)
        # Treated row is e_1.
        np.testing.assert_array_equal(A[0], [1, 0, 0])
        # Affected unit basis vectors.
        np.testing.assert_array_equal(A[1], [0, 1, 0])
        np.testing.assert_array_equal(A[2], [0, 0, 1])
        # Clean controls all-zero.
        assert (A[3:] == 0).all()

    def test_build_A_homogeneous_shapes(self):
        # v3 Example 2 (single shared b).
        A = build_A_homogeneous(N=6, p=3)
        assert A.shape == (6, 2)
        np.testing.assert_array_equal(A[0], [1, 0])
        # All p affected rows share column 1.
        np.testing.assert_array_equal(A[1:4, 1], [1, 1, 1])
        # Clean rows zero.
        assert (A[4:] == 0).all()

    def test_build_A_homogeneous_rejects_p_zero(self):
        with pytest.raises(MlsynthDataError):
            build_A_homogeneous(N=5, p=0)

    def test_build_A_distance_decay_shapes(self):
        # v3 Example 3 (exponential decay).
        w = np.array([1.0, 0.5, 0.25, 0.0])
        A = build_A_distance_decay(w)
        assert A.shape == (5, 2)
        np.testing.assert_array_equal(A[0], [1, 0])
        np.testing.assert_allclose(A[1:, 1], w)

    def test_build_A_distance_decay_rejects_all_zero(self):
        with pytest.raises(MlsynthDataError):
            build_A_distance_decay(np.zeros(4))

    def test_homogeneous_recovers_shared_b(self):
        # DGP injects the SAME spillover (+1.5) on u1 and u2, so the
        # homogeneous assumption matches the truth and the shared b
        # should recover ~1.5.
        rng = np.random.default_rng(5)
        N, T, T0 = 8, 40, 30
        loadings = rng.uniform(0.5, 1.5, size=N)
        f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
        intercept = rng.uniform(-1, 1, size=N)
        Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
        Y[0, T0:] += -3.0
        Y[1, T0:] += 1.5
        Y[2, T0:] += 1.5
        D = np.zeros((N, T)); D[0, T0:] = 1
        df = pd.DataFrame([
            {"unit": f"u{i}", "year": t, "y": float(Y[i, t]),
             "treat": int(D[i, t])}
            for i in range(N) for t in range(T)
        ])

        res = SPILLSYNTH(_cfg(
            df,
            spillover_structure="homogeneous",
            affected_units=["u1", "u2"],
        )).fit()
        assert res.att == pytest.approx(-3.0, abs=0.8)
        # Both keys map to the same shared coefficient series.
        np.testing.assert_allclose(
            res.spillover_effects["u1"], res.spillover_effects["u2"],
        )
        # Shared b averaged over post is in the right ballpark.
        assert res.cd.gamma[1].mean() == pytest.approx(1.5, abs=0.6)

    def test_distance_decay_runs_end_to_end(self, panel):
        # Build a distance dict; closer unit -> bigger decay weight.
        distances = {f"u{i}": 0.3 * i for i in range(1, 8)}
        res = SPILLSYNTH(_cfg(
            panel,
            spillover_structure="distance_decay",
            unit_distances=distances,
        )).fit()
        assert res.inputs.A.shape == (8, 2)
        # All controls with positive decay weight appear in the
        # spillover panel.
        assert set(res.spillover_effects.keys()) == {f"u{i}" for i in range(1, 8)}

    def test_distance_decay_requires_dict(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(panel, spillover_structure="distance_decay"))

    def test_homogeneous_requires_affected_units(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(panel, spillover_structure="homogeneous"))


class TestEfficientWeighting:
    """v3 Proposition S.1 -- GMM-weighted variant."""

    def test_efficient_fit_populated(self, panel):
        res = SPILLSYNTH(_cfg(
            panel, affected_units=["u1"], weighting="efficient",
        )).fit()
        eff = res.cd.efficient_fit
        assert eff is not None
        T1 = res.inputs.T1
        N = res.inputs.N
        assert eff["alpha_W"].shape == (N, T1)
        assert eff["W"].shape == (N, N)
        # ATT under efficient weighting is finite and close to identity ATT
        # (same panel; weighting only changes finite-sample variance).
        assert np.isfinite(eff["att_sp_W"])

    def test_identity_weighting_returns_no_efficient(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        assert res.cd.efficient_fit is None

    def test_sp_estimate_weighted_identity_matches_unweighted(self, panel):
        # With W = I the weighted closed form equals sp_estimate.
        inputs = prepare_spillsynth_inputs(
            df=panel, outcome="y", treat="treat",
            unitid="unit", time="year",
            affected_units=["u1"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        M = build_M(B)
        g0, a0, _ = sp_estimate(inputs.Y_post, a=a, B=B, M=M, A=inputs.A)
        g1, a1, _ = sp_estimate_weighted(
            inputs.Y_post, a=a, B=B, A=inputs.A, W=np.eye(inputs.N),
        )
        np.testing.assert_allclose(g0, g1, atol=1e-8)
        np.testing.assert_allclose(a0, a1, atol=1e-8)


class TestKappaASpecificationTest:
    """v3 Section 5.1.2 -- kappa_A specification test."""

    def test_kappa_A_fields_present(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        kA = res.cd.kappa_A_test
        assert kA is not None
        T1 = res.inputs.T1
        T0 = res.inputs.T0
        assert kA.kappa_A.shape == (T1,)
        assert kA.kappa_pre.shape == (T0,)
        assert kA.p_value.shape == (T1,)
        assert ((kA.p_value >= 0.0) & (kA.p_value <= 1.0)).all()
        assert isinstance(kA.cutoff_05, float)

    def test_select_A_by_kappa_prefers_correct_structure(self):
        # DGP injects spillover ONLY on u1; the per-unit-with-just-u1 A
        # should win over a homogeneous A that assumes shared spillover
        # across u1, u2, u3.
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=7)
        inputs = prepare_spillsynth_inputs(
            df=df, outcome="y", treat="treat",
            unitid="unit", time="year",
            affected_units=["u1"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        A_correct = build_A_per_unit(inputs.N, 1)
        A_wrong = build_A_homogeneous(inputs.N, 3)
        idx, kappas = select_A_by_kappa(
            Y_post=inputs.Y_post, Y_pre=inputs.Y_pre,
            a=a, B=B, candidates=[A_correct, A_wrong],
        )
        # The correctly-specified structure should have a smaller kappa_A.
        assert idx == 0, (kappas, "expected per-unit correct spec to win")


class TestSignedCIs:
    """v3 page 27 / R reference -- CI via test inversion."""

    def test_ci_brackets_alpha(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        ci = res.cd.treatment_ci_95
        T1 = res.inputs.T1
        assert ci.shape == (T1, 2)
        # The point estimate lies inside its CI by construction (CI is
        # alpha + [q_{0.025}, q_{0.975}]; q_{0.025} <= 0 <= q_{0.975} when
        # pre-period residuals straddle zero, which holds generically).
        # We just check the CI is non-degenerate (lower < upper).
        assert (ci[:, 0] < ci[:, 1]).all()

    def test_spillover_ci_keys(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1", "u2"])).fit()
        assert set(res.cd.spillover_ci_95.keys()) == {"u1", "u2"}
        for ci in res.cd.spillover_ci_95.values():
            assert ci.shape == (res.inputs.T1, 2)


class TestJointSpilloverTest:
    """MATLAB reference -- joint H_0: alpha_2 = ... = alpha_{1+p} = 0."""

    def test_joint_test_populated_when_affected_present(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1", "u2"])).fit()
        jt = res.cd.joint_spillover_test
        T1 = res.inputs.T1
        assert jt is not None
        assert jt.P_post.shape == (T1,)
        assert jt.p_value.shape == (T1,)
        assert ((jt.p_value >= 0) & (jt.p_value <= 1)).all()

    def test_joint_test_none_when_no_affected(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=None)).fit()
        assert res.cd.joint_spillover_test is None

    def test_joint_rejects_under_strong_spillover(self):
        # +3 spillover on u1 with N=8: joint test should fire in every
        # post-period.
        df = _spill_panel(treatment=-3.0, spillover=3.0,
                          spillover_idx=1, seed=11)
        res = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        assert res.cd.joint_spillover_test.reject_05.all()


class TestPureDonorSensitivity:
    """v3 Section 5.2 -- misspecification-bias bounds."""

    def test_sensitivity_populated_with_clean_controls(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1", "u2"])).fit()
        pds = res.cd.pure_donor_sensitivity
        assert pds is not None
        n_clean = res.inputs.N - 1 - res.inputs.p          # = N - 1 - p
        assert pds.n_clean == n_clean
        assert pds.w_sp.shape == (n_clean,)
        assert pds.w_pd.shape == (n_clean,)
        # Weights are sorted descending by |.|.
        assert (np.diff(pds.w_sp) <= 1e-10).all()
        assert (np.diff(pds.w_pd) <= 1e-10).all()

    def test_bias_bounds_linear(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        pds = res.cd.pure_donor_sensitivity
        grid = np.array([0.0, 1.0, 5.0])
        sp, pd = pds.bias_bounds(p=1, alpha_bar_grid=grid)
        assert sp[0] == 0 and pd[0] == 0                  # bound at alpha=0
        assert sp[2] == pytest.approx(5.0 * sp[1], abs=1e-9)
        assert pd[2] == pytest.approx(5.0 * pd[1], abs=1e-9)

    def test_bias_bounds_rejects_bad_p(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        with pytest.raises(ValueError):
            res.cd.pure_donor_sensitivity.bias_bounds(
                p=res.cd.pure_donor_sensitivity.n_clean + 1,
                alpha_bar_grid=np.array([1.0]),
            )


class TestAlgebraicIdentities:
    """Closed-form identities that should hold for every fit."""

    def test_alpha_equals_A_times_gamma(self, panel):
        # alpha = A @ gamma by construction; verify for all three structures.
        for cfg_extras in [
            dict(spillover_structure="per_unit", affected_units=["u1", "u2"]),
            dict(spillover_structure="homogeneous", affected_units=["u1", "u2"]),
            dict(spillover_structure="distance_decay",
                 unit_distances={f"u{i}": 0.3 * i for i in range(1, 8)}),
        ]:
            res = SPILLSYNTH(_cfg(panel, **cfg_extras)).fit()
            expected = res.inputs.A @ res.cd.gamma
            np.testing.assert_allclose(res.cd.alpha, expected, atol=1e-12)

    def test_vanilla_scm_matches_a_plus_B_row0(self, panel):
        # counterfactual_scm == a[0] + B[0] @ Y_post by definition.
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        manual = res.cd.a[0] + res.cd.B[0] @ res.inputs.Y_post
        np.testing.assert_allclose(res.cd.counterfactual_scm, manual, atol=1e-12)

    def test_gap_sp_equals_alpha_row0(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        np.testing.assert_allclose(res.cd.gap_sp, res.cd.alpha[0], atol=1e-12)

    def test_att_sp_equals_gap_mean(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        assert res.att == pytest.approx(float(res.cd.gap_sp.mean()))
        assert res.att_scm == pytest.approx(float(res.cd.gap_scm.mean()))

    def test_homogeneous_equals_per_unit_when_p_is_one(self):
        # With a single affected unit, homogeneous (k=2 cols) and per_unit
        # (k=2 cols) are mathematically the same: A is the same matrix.
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=4)
        r_per = SPILLSYNTH(_cfg(df, spillover_structure="per_unit",
                                affected_units=["u1"])).fit()
        r_hom = SPILLSYNTH(_cfg(df, spillover_structure="homogeneous",
                                affected_units=["u1"])).fit()
        np.testing.assert_allclose(r_per.inputs.A, r_hom.inputs.A)
        assert r_per.att == pytest.approx(r_hom.att, abs=1e-9)

    def test_distance_decay_alpha_proportional_to_decay(self):
        # alpha[i] = exp(-d_i) * gamma[1]; verify the proportionality
        # against the recovered gamma vector.
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=9)
        distances = {f"u{i}": 0.2 * i for i in range(1, 8)}
        res = SPILLSYNTH(_cfg(
            df, spillover_structure="distance_decay",
            unit_distances=distances,
        )).fit()
        decay = res.inputs.A[1:, 1]                       # (N-1,)
        gamma_b = res.cd.gamma[1]                         # (T1,)
        # For each control unit i, alpha[1+i] should equal decay[i] * gamma_b.
        for i in range(decay.size):
            np.testing.assert_allclose(
                res.cd.alpha[1 + i], decay[i] * gamma_b, atol=1e-9,
            )


class TestEdgeCaseStructures:
    """Edge cases at the boundaries of A-matrix construction."""

    def test_per_unit_with_p_zero_runs(self, panel):
        # No affected units declared: reduces to vanilla demeaned SCM.
        res = SPILLSYNTH(_cfg(panel, affected_units=None)).fit()
        assert res.inputs.A.shape == (res.inputs.N, 1)    # only treated col
        assert res.cd.gamma.shape == (1, res.inputs.T1)
        # SP gap equals treated unit's own residual (which is what
        # vanilla SCM does on the demeaned variant).
        assert np.isfinite(res.att)

    def test_singular_AMA_all_units_affected_raises(self, panel):
        # Declare every control as affected -> (I-B)A has rank deficiency
        # for many panels (per Example 4 in Section 3.4.1). The pipeline
        # should still produce numbers, but condition number must be flagged.
        all_controls = [f"u{i}" for i in range(1, 8)]
        try:
            res = SPILLSYNTH(_cfg(panel, affected_units=all_controls)).fit()
            # If it ran, cond_AMA should be huge (near singular).
            assert res.cd.cond_AMA > 1e8 or np.isnan(res.cd.cond_AMA)
        except (MlsynthEstimationError, np.linalg.LinAlgError):
            # Either outcome is acceptable; we just don't want a silent
            # garbage answer with cond_AMA O(1).
            pass

    def test_distance_decay_ignores_treated_in_dict(self, panel):
        # Treated unit u0's distance is ignored (its row is always (1, 0)).
        distances = {"u0": 99.0, **{f"u{i}": 0.5 for i in range(1, 8)}}
        res = SPILLSYNTH(_cfg(
            panel, spillover_structure="distance_decay",
            unit_distances=distances,
        )).fit()
        np.testing.assert_array_equal(res.inputs.A[0], [1, 0])

    def test_distance_decay_missing_controls_get_zero_weight(self, panel):
        # Only u1 has a finite distance; the rest get exp(-d)=0 weight.
        res = SPILLSYNTH(_cfg(
            panel, spillover_structure="distance_decay",
            unit_distances={"u1": 0.0},
        )).fit()
        decay = res.inputs.A[1:, 1]
        # Exactly one control row should have weight 1; rest 0.
        assert (decay > 0).sum() == 1
        assert decay[0] == pytest.approx(1.0)

    def test_distance_decay_rejects_negative_distance(self, panel):
        with pytest.raises(MlsynthDataError):
            SPILLSYNTH(_cfg(
                panel, spillover_structure="distance_decay",
                unit_distances={"u1": -0.5},
            )).fit()

    def test_distance_decay_rejects_infinite_distance(self, panel):
        with pytest.raises(MlsynthDataError):
            SPILLSYNTH(_cfg(
                panel, spillover_structure="distance_decay",
                unit_distances={"u1": float("inf")},
            )).fit()

    def test_distance_decay_rejects_unknown_label(self, panel):
        with pytest.raises(MlsynthDataError):
            SPILLSYNTH(_cfg(
                panel, spillover_structure="distance_decay",
                unit_distances={"ghost_unit": 1.0, "u1": 0.5},
            )).fit()

    def test_unknown_spillover_structure_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(
                panel, spillover_structure="per-unit",   # hyphen not underscore
                affected_units=["u1"],
            ))


class TestMultiplePostPeriods:
    """Shapes scale correctly with T1 > 1."""

    @pytest.mark.parametrize("T1", [1, 3, 10])
    def test_shapes_scale_with_T1(self, T1):
        df = _spill_panel(N=8, T=20 + T1, T0=20, treatment=-3.0,
                          spillover=1.5, spillover_idx=1, seed=2)
        res = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        assert res.inputs.T1 == T1
        assert res.cd.alpha.shape == (res.inputs.N, T1)
        assert res.cd.gamma.shape == (2, T1)
        assert res.cd.gap_sp.shape == (T1,)
        assert res.cd.gap_scm.shape == (T1,)
        assert res.cd.treatment_test.P_post.shape == (T1,)
        assert res.cd.treatment_ci_95.shape == (T1, 2)
        assert res.cd.kappa_A_test.kappa_A.shape == (T1,)
        if T1 > 0:
            assert res.cd.joint_spillover_test.P_post.shape == (T1,)


class TestStatisticalBehaviour:
    """Statistical sanity checks: p-values, CIs, reject patterns."""

    def test_treatment_pvalue_monotone_in_effect_size(self):
        # Bigger true effect -> smaller p-value (in expectation; we use
        # the same seed so the comparison is deterministic).
        small = SPILLSYNTH(_cfg(
            _spill_panel(treatment=-0.5, spillover=0.0,
                         spillover_idx=1, seed=42),
            affected_units=["u1"],
        )).fit()
        big = SPILLSYNTH(_cfg(
            _spill_panel(treatment=-8.0, spillover=0.0,
                         spillover_idx=1, seed=42),
            affected_units=["u1"],
        )).fit()
        # Mean p-value across post-periods should be lower for the
        # larger true effect.
        assert big.cd.treatment_test.p_value.mean() <= \
            small.cd.treatment_test.p_value.mean()

    def test_ci_contains_point_estimate_when_residuals_symmetric(self, panel):
        # The CI is alpha + [q_{0.025}, q_{0.975}]. As long as the
        # pre-period residual distribution straddles 0 (generic), the
        # point estimate should lie inside the CI.
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        ci = res.cd.treatment_ci_95
        point = res.cd.alpha[0]
        # Allow the point to coincide with an endpoint of the CI
        # (happens with degenerate residual quantiles); strict
        # containment is checked by the q_lo<0<q_hi pattern.
        assert (ci[:, 0] <= point + 1e-9).all()
        assert (point - 1e-9 <= ci[:, 1]).all()

    def test_kappa_A_small_under_correct_spec(self):
        # With the correct structure and a clean DGP, kappa_A should be
        # smaller than a deliberately wrong structure on the same panel.
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=15)
        r_right = SPILLSYNTH(_cfg(df, affected_units=["u1"])).fit()
        r_wrong = SPILLSYNTH(_cfg(df, affected_units=["u2"])).fit()
        # Right spec: kappa_A averaged across post-periods is smaller
        # than the misspecified version.
        assert r_right.cd.kappa_A_test.kappa_A.mean() < \
            r_wrong.cd.kappa_A_test.kappa_A.mean()


class TestReproducibility:
    """Same inputs must produce identical numerical outputs."""

    def test_same_panel_same_seed_same_results(self):
        df1 = _spill_panel(seed=99)
        df2 = _spill_panel(seed=99)
        r1 = SPILLSYNTH(_cfg(df1, affected_units=["u1"])).fit()
        r2 = SPILLSYNTH(_cfg(df2, affected_units=["u1"])).fit()
        assert r1.att == pytest.approx(r2.att)
        np.testing.assert_allclose(r1.cd.alpha, r2.cd.alpha, atol=1e-12)
        np.testing.assert_allclose(r1.cd.B, r2.cd.B, atol=1e-9)
        np.testing.assert_allclose(
            r1.cd.treatment_test.P_post, r2.cd.treatment_test.P_post, atol=1e-9,
        )

    def test_dict_path_passes_through_new_options(self, panel):
        # Verify the dict-config code path honours spillover_structure,
        # unit_distances, and weighting.
        res = SPILLSYNTH(_cfg(
            panel,
            spillover_structure="distance_decay",
            unit_distances={f"u{i}": 0.4 for i in range(1, 8)},
            weighting="efficient",
        )).fit()
        assert res.inputs.A.shape == (8, 2)
        assert res.cd.efficient_fit is not None


class TestInferenceEdgeCases:
    """Inference module edge cases."""

    def test_p_test_multi_row_C_runs(self, panel):
        # Direct call to p_test with a multi-row C (matches MATLAB's
        # joint-spillover selector).
        from mlsynth.utils.spillsynth_helpers.cd.inference import (
            G_matrix, compute_pre_residuals, p_test,
        )
        inputs = prepare_spillsynth_inputs(
            df=panel, outcome="y", treat="treat",
            unitid="unit", time="year", affected_units=["u1", "u2"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        M = build_M(B)
        _gamma, alpha, _cond = sp_estimate(
            inputs.Y_post, a=a, B=B, M=M, A=inputs.A,
        )
        U_pre = compute_pre_residuals(inputs.Y_pre, a, B)
        G_hat = G_matrix(inputs.A, B)
        # Multi-row C selecting both affected rows.
        C = np.zeros((2, inputs.N))
        C[0, 1] = 1; C[1, 2] = 1
        result = p_test(alpha_hat=alpha, U_pre=U_pre, G_hat=G_hat, C=C)
        assert result.P_post.shape == (inputs.T1,)
        assert (result.P_post >= 0).all()                # quadratic form

    def test_signed_ci_rejects_multi_row_C(self, panel):
        from mlsynth.utils.spillsynth_helpers.cd.inference import (
            G_matrix, compute_pre_residuals, signed_ci,
        )
        inputs = prepare_spillsynth_inputs(
            df=panel, outcome="y", treat="treat",
            unitid="unit", time="year", affected_units=["u1", "u2"],
        )
        a, B = fit_leave_one_out_sc(inputs.Y_pre)
        M = build_M(B)
        _g, alpha, _c = sp_estimate(inputs.Y_post, a=a, B=B, M=M, A=inputs.A)
        U_pre = compute_pre_residuals(inputs.Y_pre, a, B)
        G_hat = G_matrix(inputs.A, B)
        C_multi = np.eye(2, inputs.N)
        with pytest.raises(ValueError):
            signed_ci(alpha_hat=alpha, U_pre=U_pre, G_hat=G_hat, C=C_multi)

    def test_kappa_A_test_with_zero_p(self, panel):
        # No affected units: A is N x 1 (just treated unit basis).
        # kappa_A_test should still compute (it doesn't depend on p>0).
        res = SPILLSYNTH(_cfg(panel, affected_units=None)).fit()
        kA = res.cd.kappa_A_test
        assert kA is not None
        assert kA.kappa_A.shape == (res.inputs.T1,)


class TestEfficientWeightingEdgeCases:
    """GMM-efficient edge cases."""

    def test_efficient_with_small_T0_runs(self):
        # T0 = 5, N = 8 -> sample Omega is rank 5 < N. Ridge keeps
        # it invertible.
        df = _spill_panel(N=8, T=15, T0=5, treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=7)
        res = SPILLSYNTH(_cfg(
            df, affected_units=["u1"], weighting="efficient",
        )).fit()
        eff = res.cd.efficient_fit
        assert eff is not None
        assert np.isfinite(eff["att_sp_W"])
        # Omega_hat must be symmetric.
        np.testing.assert_allclose(
            eff["Omega_hat"], eff["Omega_hat"].T, atol=1e-12,
        )

    def test_efficient_recovers_treatment_under_strong_signal(self):
        df = _spill_panel(treatment=-3.0, spillover=1.5,
                          spillover_idx=1, seed=23)
        res = SPILLSYNTH(_cfg(
            df, affected_units=["u1"], weighting="efficient",
        )).fit()
        # ATT under W should land near truth as well.
        assert res.cd.efficient_fit["att_sp_W"] == pytest.approx(-3.0, abs=1.0)


class TestPureDonorEdgeCases:
    def test_no_clean_controls_returns_none(self, panel):
        # Declare every control as affected. No clean controls remain,
        # so the SP misspec analysis is undefined -- pipeline returns
        # None.
        try:
            res = SPILLSYNTH(_cfg(
                panel, affected_units=[f"u{i}" for i in range(1, 8)],
            )).fit()
        except MlsynthEstimationError:
            # Singular A'MA path; accepted, not a sensitivity test.
            return
        assert res.cd.pure_donor_sensitivity is None

    def test_bias_bounds_p_zero_rejected(self, panel):
        res = SPILLSYNTH(_cfg(panel, affected_units=["u1"])).fit()
        with pytest.raises(ValueError):
            res.cd.pure_donor_sensitivity.bias_bounds(
                p=0, alpha_bar_grid=np.array([1.0]),
            )


class TestMultipleTreatedUnits:
    """Cao-Dowd v3 Section S.1.2 -- multiple treated units, common time."""

    def _multi_treated_panel(
        self, *, treated_idx, spillover_idx, treatment_effects,
        spillover, N=6, T=40, T0=30, seed=0,
    ):
        """Build a panel with multiple treated units at the same time."""
        rng = np.random.default_rng(seed)
        loadings = rng.uniform(0.5, 1.5, size=N)
        f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
        intercept = rng.uniform(-1, 1, size=N)
        Y = intercept[:, None] + np.outer(loadings, f) + 0.10 * rng.standard_normal((N, T))
        for j, eff in zip(treated_idx, treatment_effects):
            Y[j, T0:] += eff
        if spillover_idx is not None:
            Y[spillover_idx, T0:] += spillover
        D = np.zeros((N, T))
        for j in treated_idx:
            D[j, T0:] = 1
        return pd.DataFrame([
            {"unit": f"u{i}", "year": t, "y": float(Y[i, t]),
             "treat": int(D[i, t])}
            for i in range(N) for t in range(T)
        ])

    def test_two_treated_one_affected_recovers_all(self):
        # Section S.1.2 N=6 mimic: u0, u1 treated (effects -3, -2), u2
        # affected by spillover +1.5, u3-u5 clean.
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=2,
            treatment_effects=[-3.0, -2.0], spillover=1.5, seed=7,
        )
        res = SPILLSYNTH(_cfg(df, affected_units=["u2"])).fit()
        assert res.inputs.n_treated == 2
        assert res.inputs.treated_labels == ("u0", "u1")
        # A is (N, n_treated + p) = (6, 3).
        assert res.inputs.A.shape == (6, 3)
        # Per-treated ATT recovery.
        assert res.cd.atts_sp_by_unit["u0"] == pytest.approx(-3.0, abs=0.6)
        assert res.cd.atts_sp_by_unit["u1"] == pytest.approx(-2.0, abs=0.6)
        # Spillover on u2.
        assert res.spillover_effects["u2"].mean() == pytest.approx(1.5, abs=0.6)
        # Back-compat: res.att = first-treated ATT.
        assert res.att == pytest.approx(res.cd.atts_sp_by_unit["u0"])

    def test_per_treated_inference_fields_present(self):
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=2,
            treatment_effects=[-3.0, -2.0], spillover=1.5, seed=11,
        )
        res = SPILLSYNTH(_cfg(df, affected_units=["u2"])).fit()
        # Per-treated dicts.
        assert set(res.cd.treatment_tests.keys()) == {"u0", "u1"}
        assert set(res.cd.treatment_cis_95.keys()) == {"u0", "u1"}
        assert set(res.cd.atts_sp_by_unit.keys()) == {"u0", "u1"}
        assert set(res.cd.gaps_sp_by_unit.keys()) == {"u0", "u1"}
        # Shapes.
        for label in ("u0", "u1"):
            assert res.cd.treatment_tests[label].p_value.shape == (res.inputs.T1,)
            assert res.cd.treatment_cis_95[label].shape == (res.inputs.T1, 2)
            assert res.cd.gaps_sp_by_unit[label].shape == (res.inputs.T1,)
        # Strong true effects -> rejection at every period for both treated.
        assert res.cd.treatment_tests["u0"].reject_05.all()
        assert res.cd.treatment_tests["u1"].reject_05.all()

    def test_back_compat_att_equals_first_treated(self):
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=None,
            treatment_effects=[-3.0, -2.0], spillover=0.0, seed=1,
        )
        res = SPILLSYNTH(_cfg(df, affected_units=None)).fit()
        # Back-compat fields point at the first treated unit.
        assert res.att == pytest.approx(res.cd.atts_sp_by_unit["u0"])
        np.testing.assert_allclose(
            res.gap, res.cd.gaps_sp_by_unit["u0"], atol=1e-12,
        )
        np.testing.assert_allclose(
            res.cd.treatment_ci_95, res.cd.treatment_cis_95["u0"], atol=1e-12,
        )

    def test_multi_treated_homogeneous_structure(self):
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=2,
            treatment_effects=[-3.0, -2.0], spillover=1.5, seed=2,
        )
        res = SPILLSYNTH(_cfg(
            df, spillover_structure="homogeneous", affected_units=["u2"],
        )).fit()
        # A has n_treated + 1 columns under homogeneous.
        assert res.inputs.A.shape == (6, 3)
        # gamma is shape (3, T1): 2 treatment-effect rows + 1 shared b.
        assert res.cd.gamma.shape == (3, res.inputs.T1)

    def test_multi_treated_per_unit_alpha_partition(self):
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=2,
            treatment_effects=[-3.0, -2.0], spillover=1.5, seed=3,
        )
        res = SPILLSYNTH(_cfg(df, affected_units=["u2"])).fit()
        # alpha rows 0..n_treated-1 = per-treated gaps; row n_treated = spillover.
        np.testing.assert_allclose(res.cd.alpha[0], res.cd.gaps_sp_by_unit["u0"])
        np.testing.assert_allclose(res.cd.alpha[1], res.cd.gaps_sp_by_unit["u1"])
        np.testing.assert_allclose(res.cd.alpha[2], res.spillover_effects["u2"])
        # Clean controls -> identically zero.
        assert (np.abs(res.cd.alpha[3:]) < 1e-12).all()

    def test_treated_in_affected_via_dict_path_multi(self):
        # u0 and u1 both treated; declaring u1 as affected must be rejected.
        df = self._multi_treated_panel(
            treated_idx=[0, 1], spillover_idx=2,
            treatment_effects=[-3.0, -2.0], spillover=1.5, seed=4,
        )
        with pytest.raises(MlsynthConfigError):
            SPILLSYNTH(_cfg(df, affected_units=["u1"]))


class TestEmpiricalPathARegression:
    """Path-A contract: pin paper numbers as a permanent regression check.

    Drives the 51-unit Prop 99 + DC panel through the public
    ``SPILLSYNTH(config).fit()`` API and asserts that the published
    Cao-Dowd v3 values come back to four decimal places.

    The data file (``basedata/prop99_with_dc.csv``) is the augmented
    panel shipped with mlsynth -- 50 states from
    ``prop99_packsales.csv`` plus Washington DC sourced from the CDC
    Tax Burden on Tobacco compilation -- and the 13 spillover-affected
    states are Cao-Dowd Section 5 footnote 5 (AK, AZ, DC, FL, HI, MA,
    MD, MI, NJ, NV, NY, OR, WA).
    """

    PROP99_AFFECTED = [
        "Alaska", "Arizona", "District of Columbia", "Florida", "Hawaii",
        "Massachusetts", "Maryland", "Michigan", "New Jersey", "Nevada",
        "New York", "Oregon", "Washington",
    ]

    @pytest.fixture(scope="class")
    def prop99_fit(self):
        repo_root = Path(__file__).resolve().parents[2]
        csv = repo_root / "basedata" / "prop99_with_dc.csv"
        if not csv.exists():
            pytest.skip(f"prop99_with_dc.csv not present at {csv}")
        df = pd.read_csv(csv)
        df = df[(df["year"] >= 1970) & (df["year"] <= 2000)].copy()
        df["treat"] = (
            (df["state"] == "California") & (df["year"] >= 1989)
        ).astype(int)
        return SPILLSYNTH({
            "df": df, "outcome": "cigsale", "treat": "treat",
            "unitid": "state", "time": "year",
            "method": "cd",
            "affected_units": self.PROP99_AFFECTED,
            "display_graphs": False,
        }).fit()

    def test_att_sp_matches_paper_four_decimals(self, prop99_fit):
        # Paper's headline figure for Cao-Dowd v3 Section 6.
        assert prop99_fit.att == pytest.approx(-9.4399, abs=1e-4)

    def test_att_scm_matches_paper_four_decimals(self, prop99_fit):
        assert prop99_fit.att_scm == pytest.approx(-10.8120, abs=1e-4)

    def test_per_year_att_sp_matches_paper(self, prop99_fit):
        # v3 Figure 4(b) annotations, year by year.
        expected = {
            1989: +0.0827,  1990: +3.7144,  1991: -3.7584,  1992: -3.4271,
            1993: -7.6146,  1994: -10.9137, 1995: -12.8346, 1996: -13.0843,
            1997: -14.9136, 1998: -16.0812, 1999: -18.9588, 2000: -15.4901,
        }
        post_years = sorted(prop99_fit.inputs.post_time.tolist())
        alpha_treated = prop99_fit.cd.alpha[0]
        for year, expected_val in expected.items():
            idx = post_years.index(year)
            assert alpha_treated[idx] == pytest.approx(expected_val, abs=1e-4), (
                f"{year}: expected {expected_val}, got {alpha_treated[idx]}"
            )

    def test_pure_donor_alphabar_matches_figure_3a(self, prop99_fit):
        # Cao-Dowd v3 Figure 3(a) annotation: alpha_bar = 17.07 is the
        # smallest worst-case missed spillover capable of invalidating
        # the largest observed ATT, under p = 1.
        pds = prop99_fit.cd.pure_donor_sensitivity
        assert pds is not None
        alpha_max = float(prop99_fit.cd.alpha[0].max())
        c_sp_p1 = float(pds.w_sp[:1].sum())
        ratio = alpha_max / c_sp_p1
        assert ratio == pytest.approx(17.07, abs=0.01)


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
