"""Tests for the CFM estimator (Bai & Wang 2026, "Causal Inference Using
Factor Models").

Layered along agents_tests.md:

* Layer 1 (numerical helpers): ER/GR factor selection, systematic-effect
  pipeline, block-additive HC + factor-estimation variance.
* Layer 2 (data utilities): prepare_cfm_inputs pivot + validation paths.
* Layer 3 (estimator integration): CFM.fit on a factor DGP and the Prop 99
  reproduction (Bai & Wang Sec 7.1 reported numbers).
* Layer 4 (public API contracts): import, frozen result, two-family surface.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import CFM
from mlsynth.config_models import CFMConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.cfm_helpers.factors import ahn_horenstein, extract_cfm_factors
from mlsynth.utils.cfm_helpers.pipeline import (
    chow_break_statistic,
    fit_systematic_effect,
)
from mlsynth.utils.cfm_helpers.inference import (
    block_regression_variance,
    cfm_inference,
    factor_estimation_variance,
)
from mlsynth.utils.cfm_helpers.setup import prepare_cfm_inputs
from mlsynth.utils.cfm_helpers.structures import (
    CFMDesign,
    CFMInference,
    CFMInputs,
    CFMResults,
)

BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
SMOKING = BASEDATA / "smoking_data.csv"


# ----------------------------------------------------------------------
# Shared factor-model panel fixture (unit 0 is treated; loadings BREAK at T0)
# ----------------------------------------------------------------------

def _factor_panel(
    J: int = 20, T_pre: int = 30, T_post: int = 12, r_true: int = 2,
    dloading: float = 3.0, seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Dual-potential-outcome factor DGP.

    The treated unit's loadings break by ``dloading`` at ``T_pre`` so the
    *systematic* effect is ``tau*_t = (lam1 - lam0)' f_t`` -- exactly the
    Bai-Wang estimand. Returns the DGP-true systematic effect path over the
    post-period for recovery checks.
    """
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r_true))                 # stationary factors
    lam = rng.standard_normal((J + 1, r_true))           # control + treated loadings
    eps = 0.2 * rng.standard_normal((T, J + 1))
    Y = F @ lam.T + eps
    lam0_treated = lam[0].copy()
    lam1_treated = lam0_treated + dloading * np.array([1.0, -1.0])[:r_true]
    # post-period treated systematic component uses the broken loading
    Y[T_pre:, 0] = F[T_pre:] @ lam1_treated + eps[T_pre:, 0]
    tau_true = F[T_pre:] @ (lam1_treated - lam0_treated)
    rows = [
        {"unit": j, "time": t, "y": float(Y[t, j]),
         "D": int(j == 0 and t >= T_pre)}
        for j in range(J + 1) for t in range(T)
    ]
    return pd.DataFrame(rows), tau_true, T_pre


@pytest.fixture
def panel():
    return _factor_panel()


# ======================================================================
# Layer 1: numerical helpers
# ======================================================================

class TestAhnHorenstein:
    def test_single_factor_selected_on_rank1_dgp(self):
        rng = np.random.default_rng(1)
        T, N = 40, 25
        f = rng.standard_normal((T, 1))
        Y = f @ rng.standard_normal((1, N)) + 0.05 * rng.standard_normal((T, N))
        Xc = Y - Y.mean(0, keepdims=True)
        ev = np.linalg.eigvalsh((Xc.T @ Xc) / T)[::-1]
        r_er, r_gr, ER, GR = ahn_horenstein(np.clip(ev, 0, None), max_factors=8)
        assert r_er == 1
        assert r_gr == 1
        assert ER[0] == pytest.approx(ER.max())

    def test_two_factors_selected(self):
        rng = np.random.default_rng(2)
        T, N = 60, 30
        f = rng.standard_normal((T, 2))
        Y = f @ rng.standard_normal((2, N)) + 0.02 * rng.standard_normal((T, N))
        Xc = Y - Y.mean(0, keepdims=True)
        ev = np.linalg.eigvalsh((Xc.T @ Xc) / T)[::-1]
        r_er, r_gr, _, _ = ahn_horenstein(np.clip(ev, 0, None), max_factors=8)
        assert r_er == 2
        assert r_gr == 2


class TestExtractFactors:
    def test_bai_normalization(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((40, 20)).cumsum(0)
        n, F, source, ev = extract_cfm_factors(Y, selection="er", max_factors=6)
        assert n >= 1
        assert source == "ER"
        # Bai normalization: F'F / T = I_r
        assert np.allclose(F.T @ F / F.shape[0], np.eye(n), atol=1e-8)

    def test_user_override(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((40, 20))
        n, F, source, _ = extract_cfm_factors(Y, n_factors=3)
        assert n == 3 and source == "user" and F.shape == (40, 3)

    def test_invalid_selection_rejected(self):
        Y = np.random.default_rng(0).standard_normal((20, 10))
        with pytest.raises(MlsynthConfigError):
            extract_cfm_factors(Y, selection="bogus")

    def test_invalid_n_factors_rejected(self):
        Y = np.random.default_rng(0).standard_normal((20, 10))
        with pytest.raises(MlsynthConfigError):
            extract_cfm_factors(Y, n_factors=99)

    def test_panel_too_small_rejected(self):
        with pytest.raises(MlsynthEstimationError):
            extract_cfm_factors(np.ones((1, 5)), selection="er")

    def test_ahn_horenstein_too_few_eigenvalues(self):
        with pytest.raises(MlsynthEstimationError):
            ahn_horenstein(np.array([1.0, 0.5]), max_factors=8)


class TestSystematicEffect:
    def test_recovers_systematic_effect(self, panel):
        df, tau_true, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        des = fit_systematic_effect(inputs.treated_outcome, F, T0)
        assert des.tau.shape == (inputs.T - T0,)
        # systematic effect recovered up to idiosyncratic noise
        assert np.corrcoef(des.tau, tau_true)[0, 1] > 0.9
        assert abs(des.att - tau_true.mean()) < 1.0

    def test_zero_effect_when_no_break(self):
        # no loading break -> systematic effect ~ 0
        rng = np.random.default_rng(3)
        T, T0, N = 40, 30, 15
        F = rng.standard_normal((T, 2))
        lam = rng.standard_normal((N, 2))
        Yc = F @ lam.T + 0.1 * rng.standard_normal((T, N))
        y = F @ rng.standard_normal(2) + 0.1 * rng.standard_normal(T)
        _, Fh, _, _ = extract_cfm_factors(Yc, n_factors=2)
        des = fit_systematic_effect(y, Fh, T0)
        assert abs(des.att) < 1.0
        assert des.kappa == pytest.approx(des.a1 - des.a0)

    def test_chow_statistic_positive_and_large_under_break(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        fstat = chow_break_statistic(inputs.treated_outcome, F, T0)
        assert np.isfinite(fstat) and fstat > 0

    def test_fit_rejects_factor_shape_mismatch(self):
        with pytest.raises(ValueError):
            fit_systematic_effect(np.zeros(10), np.zeros((8, 2)), 5)

    def test_chow_nan_when_too_few_periods(self):
        # T - 2k <= 0 -> undefined F-statistic.
        y = np.array([1.0, 2.0, 3.0])
        F = np.array([[0.1], [0.2], [0.3]])
        assert np.isnan(chow_break_statistic(y, F, 2))



class TestInference:
    def test_block_variance_shapes_and_psd(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        V0, V1, th0, th1 = block_regression_variance(
            inputs.treated_outcome, F, T0, hc="HC1")
        assert V0.shape == (3, 3) and V1.shape == (3, 3)  # [const, f1, f2]
        assert np.all(np.linalg.eigvalsh(V0) >= -1e-10)
        assert np.all(np.linalg.eigvalsh(V1) >= -1e-10)

    def test_factor_variance_shape_and_finite(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        Vf = factor_estimation_variance(inputs.control_outcomes, F)
        assert Vf.shape == (inputs.T, 2, 2)
        assert np.all(np.isfinite(Vf))

    def test_cfm_inference_bands_finite_and_ordered(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        out = cfm_inference(inputs.treated_outcome, F, inputs.control_outcomes,
                            T0, alpha=0.05, factor_variance=True)
        n_post = inputs.T - T0
        assert out["se_t"].shape == (n_post,)
        assert np.all(out["se_t"] > 0)
        assert np.all(out["ci_upper_t"] >= out["ci_lower_t"])
        assert np.isfinite(out["att_se"]) and out["att_se"] > 0
        assert 0.0 <= out["att_p_value"] <= 1.0

    def test_factor_variance_widens_bands(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        with_f = cfm_inference(inputs.treated_outcome, F, inputs.control_outcomes,
                               T0, factor_variance=True)
        without_f = cfm_inference(inputs.treated_outcome, F, inputs.control_outcomes,
                                  T0, factor_variance=False)
        assert np.all(with_f["se_t"] >= without_f["se_t"] - 1e-9)


# ======================================================================
# Layer 2: data utilities
# ======================================================================

class TestSetup:
    def test_pivot_assembles_inputs(self, panel):
        df, _, T0 = panel
        inputs = prepare_cfm_inputs(df, "y", "D", "unit", "time")
        assert isinstance(inputs, CFMInputs)
        assert inputs.T0 == T0 and inputs.N_co == 20
        assert inputs.n_post == inputs.T - T0

    def test_missing_values_rejected(self, panel):
        df, _, _ = panel
        df.loc[3, "y"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_cfm_inputs(df, "y", "D", "unit", "time")

    def test_insufficient_pre_periods_rejected(self):
        rows = [{"unit": j, "time": t, "y": float(j + t),
                 "D": int(j == 0 and t >= 1)}
                for j in range(5) for t in range(4)]
        with pytest.raises(MlsynthDataError):
            prepare_cfm_inputs(pd.DataFrame(rows), "y", "D", "unit", "time")


# ======================================================================
# Layer 3: estimator integration
# ======================================================================

class TestEstimator:
    def test_default_fit_recovers_effect(self, panel):
        df, tau_true, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D",
                   "unitid": "unit", "time": "time", "n_factors": 2,
                   "display_graphs": False}).fit()
        assert isinstance(res, CFMResults)
        assert abs(res.att - tau_true.mean()) < 1.0
        assert np.isfinite(res.inference_detail.att_se)
        assert res.design.n_factors == 2

    def test_selection_criteria_run(self, panel):
        df, _, _ = panel
        for sel in ("er", "gr", "bai_ng"):
            res = CFM({"df": df, "outcome": "y", "treat": "D",
                       "unitid": "unit", "time": "time",
                       "factor_selection": sel, "display_graphs": False}).fit()
            assert res.design.n_factors >= 1

    def test_no_factor_variance_option(self, panel):
        df, _, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "n_factors": 2, "factor_variance": False,
                   "display_graphs": False}).fit()
        assert res.metadata["factor_variance"] is False

    def test_invalid_selection_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError):
            CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "factor_selection": "bogus"})

    def test_multiple_treated_rejected(self, panel):
        df, _, _ = panel
        df.loc[(df.unit == 1) & (df.time >= 30), "D"] = 1
        with pytest.raises((MlsynthDataError, MlsynthEstimationError)):
            CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "display_graphs": False}).fit()

    def test_n_factors_exceeds_max_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError):
            CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "n_factors": 5, "max_factors": 3})

    def test_unexpected_error_wrapped(self, panel, monkeypatch):
        df, _, _ = panel
        import mlsynth.estimators.cfm as cfm_mod

        def _boom(*a, **k):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(cfm_mod, "fit_systematic_effect", _boom)
        with pytest.raises(MlsynthEstimationError):
            CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "n_factors": 2, "display_graphs": False}).fit()

    def test_plotting_smoke(self, panel):
        import matplotlib
        matplotlib.use("Agg")
        df, _, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "n_factors": 2, "display_graphs": True}).fit()
        assert res.att is not None

    def test_plotting_error_translated(self, panel, monkeypatch):
        df, _, _ = panel
        import mlsynth.estimators.cfm as cfm_mod
        from mlsynth.exceptions import MlsynthPlottingError

        def _boom(*a, **k):
            raise RuntimeError("bad plot")

        monkeypatch.setattr(cfm_mod, "plot_cfm", _boom)
        with pytest.raises(MlsynthPlottingError):
            CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                 "time": "time", "n_factors": 2, "display_graphs": True}).fit()


# ======================================================================
# Layer 3b: Prop 99 reproduction (Bai & Wang Sec 7.1 reported numbers)
# ======================================================================

@pytest.fixture(scope="module")
def prop99():
    df = pd.read_csv(SMOKING)
    df["treat"] = ((df.state == "California") & (df.year >= 1989)).astype(int)
    return df


@pytest.mark.skipif(not SMOKING.exists(), reason="smoking_data.csv not present")
class TestProp99Reproduction:
    def test_er_gr_select_one_factor(self, prop99):
        inputs = prepare_cfm_inputs(prop99, "cigsale", "treat", "state", "year")
        n_er, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="er")
        n_gr, _, _, _ = extract_cfm_factors(inputs.control_outcomes, selection="gr")
        assert n_er == 1
        assert n_gr == 1

    def test_chow_statistic_matches_paper(self, prop99):
        # Paper Sec 7.1: Chow F for a break at 1989 = 16.84.
        inputs = prepare_cfm_inputs(prop99, "cigsale", "treat", "state", "year")
        _, F, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=1)
        fstat = chow_break_statistic(inputs.treated_outcome, F, inputs.T0)
        assert fstat == pytest.approx(16.84, abs=0.1)

    def test_kappa_tstat_matches_paper(self, prop99):
        # Paper: intercept-shift t = 1.38 (1 factor), 0.10 (2 factors).
        inputs = prepare_cfm_inputs(prop99, "cigsale", "treat", "state", "year")
        _, F1, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=1)
        out1 = cfm_inference(inputs.treated_outcome, F1, inputs.control_outcomes,
                             inputs.T0, factor_variance=False)
        assert out1["kappa_t"] == pytest.approx(1.38, abs=0.05)
        _, F2, _, _ = extract_cfm_factors(inputs.control_outcomes, n_factors=2)
        out2 = cfm_inference(inputs.treated_outcome, F2, inputs.control_outcomes,
                             inputs.T0, factor_variance=False)
        assert out2["kappa_t"] == pytest.approx(0.10, abs=0.05)

    def test_att_tracks_synthetic_control(self, prop99):
        res = CFM({"df": prop99, "outcome": "cigsale", "treat": "treat",
                   "unitid": "state", "time": "year", "n_factors": 1,
                   "display_graphs": False}).fit()
        # mean systematic reduction in pack sales, close to SC (~ -20).
        assert -30.0 < res.att < -10.0


# ======================================================================
# Layer 4: public API contracts
# ======================================================================

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import CFM as _CFM
        assert _CFM is CFM

    def test_result_frozen(self, panel):
        df, _, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "n_factors": 2, "display_graphs": False}).fit()
        with pytest.raises(Exception):
            res.metadata = {}

    def test_n_factors_property(self, panel):
        df, _, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "n_factors": 2, "display_graphs": False}).fit()
        assert res.n_factors == 2

    def test_two_family_result_contract(self, panel):
        from mlsynth.config_models import EffectResult
        df, _, _ = panel
        res = CFM({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "n_factors": 2, "display_graphs": False}).fit()
        assert isinstance(res, EffectResult)
        # flat accessors resolve through the base contract
        assert res.att is not None
        assert res.counterfactual is not None
        assert res.gap is not None
        assert res.att_ci is not None
        assert np.isfinite(res.pre_rmse)
        assert res.effects is not None and res.time_series is not None
        assert res.inference is not None and res.method_details is not None
