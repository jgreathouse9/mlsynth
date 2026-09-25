"""Tests for the modernized FMA estimator (Li & Sonnier 2023).

Layered along agents_tests.md:

* Layer 1 (numerical helpers): factor extraction, loading projection,
  asymptotic / bootstrap / placebo inference subroutines.
* Layer 2 (data utilities): prepare_fma_inputs pivot, validation paths.
* Layer 3 (estimator integration): FMA.fit on a factor DGP, ATT
  recovery, all three inference modes, validator paths.
* Layer 4 (public API contracts): top-level import, frozen
  dataclasses, FMAResults shape.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import FMA
from mlsynth.config_models import FMAConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.fma_helpers.factors import extract_factors
from mlsynth.utils.fma_helpers.fit import estimate_loading_and_counterfactual
from mlsynth.utils.fma_helpers.inference import (
    asymptotic_inference,
    bootstrap_inference,
    placebo_inference,
)
from mlsynth.utils.fma_helpers.setup import prepare_fma_inputs
from mlsynth.utils.fma_helpers.structures import (
    FMADesign,
    FMAInference,
    FMAInputs,
    FMAResults,
)


# ----------------------------------------------------------------------
# Shared factor-model panel fixture
# ----------------------------------------------------------------------

def _factor_panel(
    J: int = 20, T_pre: int = 30, T_post: int = 10, r_true: int = 2,
    tau_true: float = 1.0, seed: int = 0, non_stationary: bool = True,
) -> tuple[pd.DataFrame, float, int]:
    """Standard factor-model DGP. Unit 0 is the treated unit."""
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    if non_stationary:
        F = rng.standard_normal((T, r_true)).cumsum(axis=0)
    else:
        F = rng.standard_normal((T, r_true))
    lam = rng.standard_normal((J + 1, r_true))
    eps = rng.standard_normal((T, J + 1)) * 0.5
    Y0 = F @ lam.T + eps
    Y = Y0.copy()
    Y[T_pre:, 0] += tau_true
    rows = [
        {"unit": j, "time": t, "y": float(Y[t, j]),
         "D": int(j == 0 and t >= T_pre)}
        for j in range(J + 1)
        for t in range(T)
    ]
    return pd.DataFrame(rows), tau_true, T_pre


@pytest.fixture
def panel():
    df, tau, T0 = _factor_panel()
    return df, tau, T0


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestFactorExtraction:
    def test_recover_true_factor_count_nonstationary(self):
        rng = np.random.default_rng(0)
        T, N, r = 50, 30, 2
        F = rng.standard_normal((T, r)).cumsum(axis=0)
        lam = rng.standard_normal((N, r))
        Y = F @ lam.T + 0.3 * rng.standard_normal((T, N))
        n, _, F_hat, source = extract_factors(
            Y, stationarity="nonstationary", preprocessing="demean",
        )
        assert n == r
        assert source == "IPC1"
        assert F_hat.shape == (T, r)

    def test_user_override_n_factors(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((40, 20))
        n, _, F_hat, source = extract_factors(
            Y, stationarity="nonstationary", preprocessing="demean",
            n_factors=3,
        )
        assert n == 3
        assert source == "user"
        assert F_hat.shape == (40, 3)

    def test_invalid_n_factors_rejected(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((10, 5))
        with pytest.raises(MlsynthConfigError):
            extract_factors(Y, n_factors=99)


class TestLoadingFit:
    def test_perfect_fit_recovers_loading(self):
        rng = np.random.default_rng(0)
        T = 40
        T0 = 30
        F = rng.standard_normal((T, 2))
        lam_true = np.array([1.0, 2.0, -0.5])
        y = (np.column_stack([np.ones(T), F]) @ lam_true)
        lam_hat, cf, F_aug, var_e = estimate_loading_and_counterfactual(
            treated_outcome=y, factors=F, T0=T0,
        )
        assert np.allclose(lam_hat, lam_true, atol=1e-8)
        assert var_e < 1e-12
        assert np.allclose(cf, y, atol=1e-8)


class TestAsymptoticInference:
    def test_returns_finite_values(self):
        rng = np.random.default_rng(0)
        T, T0 = 40, 30
        F = rng.standard_normal((T, 2))
        y = F @ rng.standard_normal(2) + 0.5 * rng.standard_normal(T)
        _, cf, F_aug, var_e = estimate_loading_and_counterfactual(y, F, T0)
        se, lo, hi, p = asymptotic_inference(y, cf, F_aug, T0)
        assert np.isfinite(se) and se > 0
        assert lo <= hi
        assert 0.0 <= p <= 1.0


class TestAsymptoticMatchesTheAuthorsImplementation:
    """Theorem 3.1's variance, as Li & Sonnier themselves define and code it.

    Four independent statements of the same estimator agree:

    * the main article's Appendix, "Variance Estimator for Theorem 3.1";
    * Web Appendix A, whose Newey-West form (W.3) reduces to it at lag 0;
    * the authors' MATLAB in Web Appendix I;
    * Wang, Racine & Wang (2025) Appendix A.1, citing Li & Sonnier.

    All four give

        Omega1 = (T2/T1) eta' A^-1 V A^-1 eta,  V = T1^-1 sum_t e_t^2 F_t F_t'
        Omega2 = T1^-1 sum_t e_t^2

    with A = T1^-1 sum_t F_t F_t'. None of them applies a degrees-of-freedom
    correction to the residual mean square, and none of them replaces V with
    sigma^2 A. These tests exist because nothing previously pinned this
    function's output to any of the four, which is how it came to compute
    neither: on the panel below it returned a standard error 1.3465 times the
    reference, from a T0 - (r + 1) correction (x1.1212) and the homoskedastic
    substitution (x1.2009).
    """

    def test_omega_is_the_sandwich_with_the_T1_normalisation(self):
        """Assembled independently from the published formula."""
        rng = np.random.default_rng(3)
        T, T0 = 50, 35
        r = 3
        F = rng.standard_normal((T, r))
        y = F @ rng.standard_normal(r) + rng.standard_normal(T) * (
            1.0 + np.arange(T) / T)          # heteroskedastic, to give power
        _, cf, F_aug, _ = estimate_loading_and_counterfactual(y, F, T0)
        se, lo, hi, _ = asymptotic_inference(y, cf, F_aug, T0)

        T2 = T - T0
        e = (y - cf)[:T0]
        F_pre, F_post = F_aug[:T0], F_aug[T0:]
        eta = F_post.mean(axis=0)
        A_inv = np.linalg.inv(F_pre.T @ F_pre / T0)
        V = F_pre.T @ np.diag(e ** 2) @ F_pre / T0
        omega = (T2 / T0) * float(eta @ A_inv @ V @ A_inv @ eta) + float(
            np.mean(e ** 2)
        )
        assert se == pytest.approx(float(np.sqrt(omega / T2)), rel=1e-12)

    def test_it_is_not_the_homoskedastic_plug_in(self):
        """The check above needs heteroskedasticity to separate the two."""
        rng = np.random.default_rng(3)
        T, T0, r = 50, 35, 3
        F = rng.standard_normal((T, r))
        y = F @ rng.standard_normal(r) + rng.standard_normal(T) * (
            1.0 + np.arange(T) / T
        )
        _, cf, F_aug, _ = estimate_loading_and_counterfactual(y, F, T0)
        se, _, _, _ = asymptotic_inference(y, cf, F_aug, T0)

        T2 = T - T0
        e = (y - cf)[:T0]
        F_pre, F_post = F_aug[:T0], F_aug[T0:]
        eta = F_post.mean(axis=0)
        A_inv = np.linalg.inv(F_pre.T @ F_pre / T0)
        s2 = float(np.mean(e ** 2))
        homoskedastic = (T2 / T0) * s2 * float(eta @ A_inv @ eta) + s2
        assert se != pytest.approx(float(np.sqrt(homoskedastic / T2)), rel=1e-6)

    def test_omega2_has_no_degrees_of_freedom_correction(self):
        """Omega2 is T1^-1 sum e^2, not SSR/(T1 - r - 1)."""
        rng = np.random.default_rng(5)
        T, T0, r = 24, 16, 4          # T0 close to r + 1, so the two differ a lot
        F = rng.standard_normal((T, r))
        y = F @ rng.standard_normal(r) + 0.4 * rng.standard_normal(T)
        _, cf, F_aug, dof_corrected = estimate_loading_and_counterfactual(
            y, F, T0
        )
        se, _, _, _ = asymptotic_inference(y, cf, F_aug, T0)
        e = (y - cf)[:T0]
        plain = float(np.mean(e ** 2))
        assert dof_corrected == pytest.approx(
            plain * T0 / (T0 - F_aug.shape[1]), rel=1e-10
        )
        T2 = T - T0
        F_pre, F_post = F_aug[:T0], F_aug[T0:]
        eta = F_post.mean(axis=0)
        A_inv = np.linalg.inv(F_pre.T @ F_pre / T0)
        V = F_pre.T @ np.diag(e ** 2) @ F_pre / T0
        omega = (T2 / T0) * float(eta @ A_inv @ V @ A_inv @ eta) + plain
        assert se == pytest.approx(float(np.sqrt(omega / T2)), rel=1e-12)

    def test_hong_kong_matches_the_web_appendix_i_code(self):
        """Cross-validation against the authors' own MATLAB, on their own data.

        Web Appendix I runs on HCW's (2012) Hong Kong GDP panel, which ships
        in ``basedata/HongKong.csv``. The expected values come from a
        line-for-line port of that code, not from this library's output.
        """
        import pandas as pd

        from mlsynth import FMA

        df = pd.read_csv("basedata/HongKong.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = FMA({"df": df, "outcome": "GDP", "treat": "Integration",
                       "unitid": "Country", "time": "Time",
                       "display_graphs": False,
                       "stationarity": "stationary"}).fit()
        d = res.inference_detail
        assert res.metadata["n_factors"] == 8
        assert d.att == pytest.approx(0.03108769584467289, rel=1e-8)
        assert d.asymptotic_att_se == pytest.approx(
            0.004605949474592333, rel=1e-6
        )
        assert d.asymptotic_att_lower == pytest.approx(
            0.022060200759860732, rel=1e-6
        )
        assert d.asymptotic_att_upper == pytest.approx(
            0.040115190929485046, rel=1e-6
        )


class TestBootstrapInference:
    def test_bootstrap_returns_correct_shapes(self):
        rng = np.random.default_rng(0)
        T, T0 = 40, 30
        F = rng.standard_normal((T, 2))
        y = F @ rng.standard_normal(2) + 0.5 * rng.standard_normal(T)
        _, cf, _, _ = estimate_loading_and_counterfactual(y, F, T0)
        out = bootstrap_inference(
            treated_outcome=y, factors=F, counterfactual=cf,
            T0=T0, n_replicates=100, seed=0,
        )
        assert out["lower"].shape == (T - T0,)
        assert out["upper"].shape == (T - T0,)
        assert out["replicates"].shape == (100, T - T0)
        assert (out["upper"] >= out["lower"]).all()
        assert out["n_replicates"] == 100


class TestPlaceboInference:
    def test_placebo_curves_shape(self):
        rng = np.random.default_rng(0)
        T, N_co, T0 = 30, 8, 22
        F = rng.standard_normal((T, 2))
        controls = F @ rng.standard_normal((2, N_co)) + 0.5 * rng.standard_normal((T, N_co))
        treated = F @ rng.standard_normal(2) + 0.5 * rng.standard_normal(T)
        out = placebo_inference(
            control_outcomes=controls, treated_outcome=treated, T0=T0,
            n_factors=2, stationarity="nonstationary", preprocessing="demean",
        )
        assert out["curves"].shape == (N_co + 1, T)
        assert out["q_lower"].shape == (T,)
        assert out["q_upper"].shape == (T,)


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_pivot_assembles_inputs(self, panel):
        df, _, T0 = panel
        inputs = prepare_fma_inputs(
            df, outcome="y", treat="D",
            unitid="unit", time="time",
        )
        assert isinstance(inputs, FMAInputs)
        assert inputs.T == 40 and inputs.T0 == T0
        assert inputs.N_co == 20
        assert inputs.preprocessing == "demean"
        assert inputs.stationarity == "nonstationary"

    def test_invalid_preprocessing_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError):
            prepare_fma_inputs(
                df, outcome="y", treat="D",
                unitid="unit", time="time",
                preprocessing="bogus",
            )

    def test_missing_values_rejected(self, panel):
        df, _, _ = panel
        df.loc[5, "y"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_fma_inputs(
                df, outcome="y", treat="D",
                unitid="unit", time="time",
            )


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_default_fit_recovers_att(self, panel):
        df, tau_true, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        assert isinstance(res, FMAResults)
        assert res.design.n_factors >= 1
        assert abs(res.att - tau_true) < 0.5
        # Default inference is asymptotic only.
        assert "asymptotic" in res.metadata["inference_methods"]
        assert np.isfinite(res.inference_detail.asymptotic_att_se)
        assert np.isfinite(res.inference_detail.asymptotic_att_lower)

    def test_user_n_factors_override(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "n_factors": 3,
        }).fit()
        assert res.design.n_factors == 3
        assert res.design.n_factors_source == "user"

    def test_bootstrap_inference(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "inference_methods": ["asymptotic", "bootstrap"],
            "n_bootstrap": 200,
        }).fit()
        assert res.inference_detail.bootstrap_n_replicates == 200
        assert res.inference_detail.bootstrap_att_t_lower.size == res.inputs.n_post
        assert (
            res.inference_detail.bootstrap_att_t_upper
            >= res.inference_detail.bootstrap_att_t_lower
        ).all()

    def test_placebo_inference(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "inference_methods": ["placebo"],
            "n_factors": 2,
        }).fit()
        assert res.inference_detail.placebo_att_curves.shape[0] == res.inputs.N_co + 1
        assert res.inference_detail.placebo_att_curves.shape[1] == res.inputs.T

    def test_all_three_inference_modes(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "inference_methods": ["asymptotic", "bootstrap", "placebo"],
            "n_bootstrap": 100,
            "n_factors": 2,
        }).fit()
        assert np.isfinite(res.inference_detail.asymptotic_att_se)
        assert res.inference_detail.bootstrap_replicates.shape[0] == 100
        assert res.inference_detail.placebo_att_curves.size > 0

    def test_no_inference(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "inference_methods": [],
        }).fit()
        assert res.inference_detail.method == "none"

    def test_invalid_inference_method_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError):
            FMA({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "inference_methods": ["bogus"],
            })


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import FMA as _FMA
        assert _FMA is FMA

    def test_results_dataclasses_frozen(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        with pytest.raises(Exception):
            res.att = 0.0
        with pytest.raises(Exception):
            res.design.lambda_hat = np.zeros_like(res.design.lambda_hat)

    def test_pre_rmse_finite(self, panel):
        df, _, _ = panel
        res = FMA({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        assert np.isfinite(res.pre_rmse)
        assert res.pre_rmse >= 0.0
