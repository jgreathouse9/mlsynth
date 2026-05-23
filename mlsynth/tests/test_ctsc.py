"""Tests for the Continuous-Treatment Synthetic Control (CTSC) estimator.

Powell (2022). Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): simplex LS, biconvex descent, restricted fit.
* Layer 2 (data utilities): panel ingestion.
* Layer 3 (estimator integration): the calibrated simulation study --
  CTSC recovers the (zero) average effect while two-way FE is badly biased
  (paper Table 1, Models 1-4); inference size; constrained re-fit.
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import CTSC
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.ctsc_helpers import (
    CTSCInputs,
    CTSCResults,
    fit_ctsc,
    generate_model,
    prepare_ctsc_inputs,
    run_simulation,
    twoway_fe_effect,
)
from mlsynth.utils.ctsc_helpers.estimate import _simplex_ls


def _panel_from_model(model: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Y, D, true_ae = generate_model(model, rng)
    n, T = Y.shape
    rows = [{"state": f"s{i}", "qtr": t, "emp": Y[i, t], "minwage": D[i, t, 0]}
            for i in range(n) for t in range(T)]
    return pd.DataFrame(rows), true_ae


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestHelpers:
    def test_simplex_ls(self):
        rng = np.random.default_rng(0)
        donors = rng.standard_normal((40, 5))
        true_w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        target = donors @ true_w
        w = _simplex_ls(target, donors)
        assert w.shape == (5,)
        assert np.isclose(w.sum(), 1.0, atol=1e-4)
        assert (w >= -1e-6).all()
        np.testing.assert_allclose(donors @ w, target, atol=1e-3)

    def test_fit_returns_valid_weights(self):
        rng = np.random.default_rng(1)
        Y, D, _ = generate_model(1, rng)
        fit = fit_ctsc(Y, D)
        n = Y.shape[0]
        assert fit["alpha"].shape == (n, 1)
        assert fit["weights"].shape == (n, n)
        np.testing.assert_allclose(np.diag(fit["weights"]), 0.0, atol=1e-9)
        np.testing.assert_allclose(fit["weights"].sum(1), 1.0, atol=1e-3)

    def test_restricted_fit_honors_constraint(self):
        rng = np.random.default_rng(2)
        Y, D, _ = generate_model(1, rng)
        pi = np.full(Y.shape[0], 1.0 / Y.shape[0])
        fit = fit_ctsc(Y, D, population_weights=pi, restrict_ae=np.array([0.5]))
        assert fit["average_effect"][0] == pytest.approx(0.5, abs=1e-4)


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs(self):
        df, _ = _panel_from_model(1)
        inp = prepare_ctsc_inputs(df, "emp", ["minwage"], "state", "qtr")
        assert isinstance(inp, CTSCInputs)
        assert inp.n == 10 and inp.T == 50 and inp.K == 1
        np.testing.assert_allclose(inp.population_weights.sum(), 1.0)

    def test_missing_treatment_var_rejected(self):
        df, _ = _panel_from_model(1)
        with pytest.raises(MlsynthDataError):
            prepare_ctsc_inputs(df, "emp", ["nope"], "state", "qtr")

    def test_empty_treatment_vars_rejected(self):
        df, _ = _panel_from_model(1)
        with pytest.raises(MlsynthConfigError):
            prepare_ctsc_inputs(df, "emp", [], "state", "qtr")


# ----------------------------------------------------------------------
# Layer 3: estimator integration + calibrated simulation (paper Table 1)
# ----------------------------------------------------------------------

class TestCalibration:
    def test_model1_ctsc_near_zero_bias_beats_fe(self):
        """Paper Table 1, Model 1: CTSC mean bias ~0 (paper 0.011), while
        two-way FE is badly biased (paper 0.85). True average effect = 0."""
        summ = run_simulation(model=1, n_sims=40, seed=0)
        assert abs(summ.ctsc_mean_bias) < 0.10        # CTSC ~ unbiased
        assert summ.ctsc_mad < 0.10
        assert abs(summ.fe_mean_bias) > 0.5            # FE badly biased
        # CTSC dramatically less biased than FE.
        assert abs(summ.ctsc_mean_bias) < 0.2 * abs(summ.fe_mean_bias)

    def test_model3_within_unit_variation(self):
        summ = run_simulation(model=3, n_sims=40, seed=1)
        assert abs(summ.ctsc_mean_bias) < 0.10
        assert abs(summ.ctsc_mean_bias) < abs(summ.fe_mean_bias)

    def test_fe_baseline_biased(self):
        rng = np.random.default_rng(5)
        Y, D, _ = generate_model(1, rng)
        assert abs(twoway_fe_effect(Y, D)) > 0.3

    def test_inference_does_not_overreject_true_null(self):
        """Panel B: rejection rate of the true null (AE=0) near nominal 0.05."""
        rng = np.random.default_rng(9)
        from mlsynth.utils.ctsc_helpers.estimate import _per_unit_fit
        from mlsynth.utils.ctsc_helpers.inference import sign_flip_wald_inference
        n_sims, rejects = 20, 0
        for s in range(n_sims):
            Y, D, _ = generate_model(1, rng)
            omega = _per_unit_fit(Y, D)
            inf = sign_flip_wald_inference(
                Y, D, pi=np.full(Y.shape[0], 1 / Y.shape[0]), omega=omega,
                null_value=np.zeros(1), n_draws=500, random_state=s)
            rejects += int(inf.p_value[0] < 0.05)
        assert rejects / n_sims <= 0.20    # not grossly over-rejecting


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_end_to_end(self):
        df, true_ae = _panel_from_model(1, seed=3)
        res = CTSC({"df": df, "outcome": "emp", "treat": "minwage",
                    "treatment_vars": ["minwage"], "unitid": "state",
                    "time": "qtr", "display_graphs": False, "inference": True, "n_draws": 500}).fit()
        assert isinstance(res, CTSCResults)
        assert res.average_effect.shape == (1,)
        assert abs(res.average_effect[0] - true_ae) < 0.2
        assert res.inference is not None

    def test_top_level_import(self):
        from mlsynth import CTSC as _C
        assert _C is CTSC

    def test_results_frozen(self):
        df, _ = _panel_from_model(1, seed=4)
        res = CTSC({"df": df, "outcome": "emp", "treat": "minwage",
                    "treatment_vars": ["minwage"], "unitid": "state",
                    "time": "qtr", "display_graphs": False, "inference": False}).fit()
        with pytest.raises(Exception):
            res.average_effect = np.zeros(1)

    def test_missing_treatment_vars_config_rejected(self):
        df, _ = _panel_from_model(1, seed=5)
        with pytest.raises(MlsynthConfigError):
            CTSC({"df": df, "outcome": "emp", "treat": "minwage",
                  "unitid": "state", "time": "qtr", "display_graphs": False})


def test_weights_results_exposed():
    """CTSC exposes WeightsResults (cross-unit average) + the per-unit matrix."""
    from mlsynth.config_models import WeightsResults
    df, _ = _panel_from_model(1)
    res = CTSC({"df": df, "outcome": "emp", "treat": "minwage",
                "treatment_vars": ["minwage"], "unitid": "state", "time": "qtr",
                "display_graphs": False, "inference": False}).fit()
    assert isinstance(res.weights, WeightsResults)
    assert res.unit_weight_matrix.shape == (res.inputs.n, res.inputs.n)
