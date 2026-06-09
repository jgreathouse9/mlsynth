"""Tests for the modernized NSC estimator (Tian 2023).

Layered along agents_tests.md:

* Layer 1 (numerical helpers): eigenvalue scaling of (a, b),
  weight QP solves on tiny known-answer problems.
* Layer 2 (data utilities): prepare_nsc_inputs pivot + Z_0 stacking,
  covariate handling, missing-data rejection.
* Layer 3 (estimator integration): NSC.fit on a nonlinear DGP,
  ATT recovery, inference behaviour, CV vs fixed (a, b).
* Layer 4 (public API contracts): top-level import, NSCResults
  shape, frozen dataclasses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import NSC
from mlsynth.config_models import NSCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.nsc_helpers.crossval import cv_select
from mlsynth.utils.nsc_helpers.inference import doudchenko_imbens_inference
from mlsynth.utils.nsc_helpers.optimization import (
    design_eigenvalues,
    fit_nsc,
    scale_a,
    scale_b,
    solve_nsc_weights,
)
from mlsynth.utils.nsc_helpers.setup import prepare_nsc_inputs
from mlsynth.utils.nsc_helpers.structures import (
    NSCCVTrace,
    NSCDesign,
    NSCInference,
    NSCInputs,
    NSCResults,
)


# ----------------------------------------------------------------------
# Shared synthetic-panel fixtures
# ----------------------------------------------------------------------

def _nonlinear_panel(
    J: int = 10, T: int = 16, T0: int = 12,
    tau_true: float = 0.1, seed: int = 0, r: int = 2,
) -> tuple[pd.DataFrame, float]:
    """Nonlinear DGP matching Tian (2023, Section 4).

    Latent linear outcome -> rescaled to [0, 1] -> raised to power ``r``
    (r=1 linear, r=2 nonlinear). Unit 0 is the treated unit and gets
    +tau in the post-period.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.sqrt(12), size=(J + 1, 2))
    mu = rng.uniform(0, np.sqrt(12), size=(J + 1, 4))
    beta_t = rng.normal(10, 1, size=(T, 2))
    lam_t = rng.normal(10, 1, size=(T, 4))
    eps = rng.normal(0, 1, size=(T, J + 1))
    Y_star = (X @ beta_t.T).T + (mu @ lam_t.T).T + eps
    Yn = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
    Y0 = Yn ** r
    Y = Y0.copy()
    Y[T0:, 0] += tau_true
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= T0)}
            for j in range(J + 1) for t in range(T)]
    return pd.DataFrame(rows), tau_true


@pytest.fixture
def panel():
    df, tau = _nonlinear_panel(seed=0)
    return df, tau


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestOptimization:
    def test_design_eigenvalues_sorted_and_positive(self):
        rng = np.random.default_rng(0)
        Z0 = rng.standard_normal((6, 4))
        vals = design_eigenvalues(Z0)
        assert vals.ndim == 1
        assert (vals > 0).all()
        assert (np.diff(vals) >= -1e-12).all()

    @pytest.mark.parametrize("b_star,expected_zero", [
        (0.0, True),
        (0.5, False),
    ])
    def test_scale_b_zero_handling(self, b_star, expected_zero):
        eigvals = np.array([0.1, 0.5, 1.0])
        b = scale_b(b_star, eigvals)
        if expected_zero:
            assert b == 0.0
        else:
            assert b > 0.0

    def test_scale_a_uses_combined_design(self):
        rng = np.random.default_rng(0)
        Z0 = rng.standard_normal((5, 4))
        a_no_b = scale_a(0.5, Z0, 0.0)
        a_with_b = scale_a(0.5, Z0, 1.0)
        assert a_with_b >= a_no_b

    def test_solve_nsc_weights_sum_to_one(self):
        rng = np.random.default_rng(0)
        Z1 = rng.standard_normal(8)
        Z0 = rng.standard_normal((6, 8))
        w = solve_nsc_weights(Z1, Z0, a=0.0, b=0.0)
        assert w.shape == (6,)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_solve_nsc_weights_can_be_negative(self):
        Z0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        Z1 = np.array([2.0, -0.5])
        w = solve_nsc_weights(Z1, Z0, a=0.0, b=0.0)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w < 0).any()

    def test_l2_penalty_spreads_weights(self):
        rng = np.random.default_rng(0)
        Z1 = rng.standard_normal(8)
        Z0 = rng.standard_normal((6, 8))
        eigvals = design_eigenvalues(Z0)
        w_no = solve_nsc_weights(Z1, Z0, a=0.0, b=0.0)
        b_raw = scale_b(0.8, eigvals)
        w_l2 = solve_nsc_weights(Z1, Z0, a=0.0, b=b_raw)
        assert w_l2.var(ddof=0) <= w_no.var(ddof=0) + 1e-9

    def test_l1_penalty_concentrates_to_neighbors(self):
        Z1 = np.array([1.0, 0.0, 0.0])
        Z0 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        w_no = solve_nsc_weights(Z1, Z0, a=0.0, b=0.0)
        w_l1 = solve_nsc_weights(Z1, Z0, a=10.0, b=0.0)
        assert w_l1[0] > w_no[0]


class TestFitNSC:
    def test_a_star_equals_one_gives_nearest_neighbor(self):
        Z1 = np.array([1.0, 0.0])
        Z0 = np.array([
            [1.1, 0.0],
            [3.0, 0.0],
            [-5.0, 2.0],
        ])
        w, *_ = fit_nsc(Z1, Z0, a_star=1.0, b_star=0.0)
        assert abs(w.sum() - 1.0) < 1e-6
        assert int(np.argmax(w)) == 0


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_panel_pivot_assembles_inputs(self, panel):
        df, _ = panel
        inputs = prepare_nsc_inputs(
            df, outcome="y", treat="D",
            unitid="unit", time="time",
        )
        assert isinstance(inputs, NSCInputs)
        assert inputs.T == 16 and inputs.T0 == 12
        assert inputs.J == 10
        assert inputs.matching_matrix.shape == (10, 12)
        assert inputs.treated_matching_vector.shape == (12,)

    def test_covariate_stacking(self, panel):
        df, _ = panel
        df["x_static"] = np.repeat(np.arange(11), 16).astype(float)
        inputs = prepare_nsc_inputs(
            df, outcome="y", treat="D",
            unitid="unit", time="time",
            covariates=["x_static"],
        )
        assert inputs.matching_matrix.shape == (10, 13)
        assert inputs.treated_matching_vector.shape == (13,)

    def test_missing_covariate_raises(self, panel):
        df, _ = panel
        with pytest.raises(MlsynthConfigError):
            prepare_nsc_inputs(
                df, outcome="y", treat="D",
                unitid="unit", time="time",
                covariates=["not_a_column"],
            )


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_fit_with_explicit_a_b(self, panel):
        df, tau_true = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "display_graphs": False,
        }).fit()
        assert isinstance(res, NSCResults)
        assert res.cv_trace is None
        assert abs(res.att - tau_true) < 0.05

    def test_fit_with_cv(self, panel):
        df, tau_true = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "cv_max_iterations": 2,
            "display_graphs": False,
        }).fit()
        assert res.cv_trace is not None
        assert isinstance(res.cv_trace, NSCCVTrace)
        assert 0.0 <= res.design.a_star <= 1.0
        assert 0.0 <= res.design.b_star <= 1.0
        assert abs(res.att - tau_true) < 0.08

    def test_inference_populated_when_enabled(self, panel):
        df, _ = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "run_inference": True,
            "display_graphs": False,
        }).fit()
        inf = res.inference_detail
        assert inf.method == "doudchenko_imbens"
        assert inf.period_variance.size == res.inputs.T
        assert inf.gap_lower.size == res.inputs.T
        assert inf.gap_upper.size == res.inputs.T
        assert (inf.gap_upper >= inf.gap_lower).all()
        assert np.isfinite(inf.att)
        assert np.isfinite(inf.att_lower) and np.isfinite(inf.att_upper)
        assert inf.att_lower <= inf.att <= inf.att_upper

    def test_inference_skipped_when_disabled(self, panel):
        df, _ = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "run_inference": False,
            "display_graphs": False,
        }).fit()
        assert res.inference_detail.method == "none"

    def test_cv_target_treated_is_rejected(self, panel):
        """The legacy ``target='treated'`` CV mode (training error on the
        treated unit's pretreatment fit) was removed when the CV
        objective was made R-faithful: the only supported target is
        ``'controls'``, which scores held-out post-period MSPE.
        """
        df, _ = panel
        with pytest.raises(MlsynthConfigError):
            NSC({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "cv_target": "treated",
                "display_graphs": False,
            })

    def test_covariates_passed_through(self, panel):
        df, _ = panel
        df["x_static"] = np.repeat(np.arange(11), 16).astype(float)
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "covariates": ["x_static"],
            "display_graphs": False,
        }).fit()
        assert res.metadata["matching_dim"] == res.inputs.T0 + 1

    def test_invalid_config_raises(self, panel):
        df, _ = panel
        with pytest.raises(MlsynthConfigError):
            NSC({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "a": 1.5,
            })

    def test_unknown_covariate_raises_through_config(self, panel):
        df, _ = panel
        with pytest.raises(MlsynthConfigError):
            NSC({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "covariates": ["nope"],
            })


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import NSC as _NSC
        assert _NSC is NSC

    def test_results_dataclasses_frozen(self, panel):
        df, _ = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "display_graphs": False,
        }).fit()
        with pytest.raises(Exception):
            res.att = 0.0
        with pytest.raises(Exception):
            res.design.w = np.zeros_like(res.design.w)

    def test_donor_weights_alias(self, panel):
        df, _ = panel
        res = NSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "a": 0.0, "b": 0.0,
            "display_graphs": False,
        }).fit()
        assert res.donor_weights is res.design.donor_weights
        assert len(res.donor_weights) == res.inputs.J
