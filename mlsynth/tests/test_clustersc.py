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
    def test_run_pcr_frequentist_recovers_att(self, panel):
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
        assert fit.name == "pcr_frequentist"
        assert credible is None
        assert abs(fit.att - tau) < 0.5

    def test_run_pcr_bayesian_returns_credible(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        fit, credible = run_pcr(
            treated_outcome=inputs.treated_outcome,
            donor_outcomes=inputs.donor_outcomes,
            donor_names=inputs.donor_names,
            T0=inputs.T0, objective="OLS", clustering=True,
            estimator="bayesian",
        )
        assert fit.name == "pcr_bayesian"
        assert credible is not None
        lo, hi = credible
        assert lo <= hi
        assert np.isfinite(lo) and np.isfinite(hi)

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
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            inputs=inputs, rpca_method=rpca_method,
        )
        assert isinstance(fit, MethodFit)
        assert fit.name == "rpca"
        assert fit.counterfactual.shape == (inputs.T,)
        assert fit.metadata["rpca_method"] == rpca_method

    def test_unknown_rpca_method_rejected(self, panel):
        df, _ = panel
        inputs = prepare_clustersc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        with pytest.raises(MlsynthEstimationError):
            run_rpca(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                inputs=inputs, rpca_method="BOGUS",
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
            "method": "rpca",
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

    def test_no_inference_for_frequentist_pcr(self, panel):
        df, _ = panel
        res = CLUSTERSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "method": "pcr", "estimator": "frequentist",
        }).fit()
        assert res.inference.method == "none"

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
        }).fit()
        assert res.rpca.metadata["rpca_method"] == rpca_method


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
