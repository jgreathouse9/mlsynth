"""Tests for the modernized Two-Step Synthetic Control (TSSC) estimator.

Covers:
    * TSSCConfig validation.
    * prepare_tssc_inputs (shapes, donor/time consistency, guards).
    * Variant estimation (the four SC-class fits, intercept handling).
    * Step-1 subsampling selection (decision tree, test records,
      reproducibility, restriction-matrix construction).
    * TSSC estimator class (smoke, recommended-variant recovery,
      standardized summary, error wrapping).
    * Plotter (smoke).
    * Immutability of frozen dataclasses.

Reference: Li & Shankar (2023), Management Science,
https://doi.org/10.1287/mnsc.2023.4878.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import TSSC
from mlsynth.config_models import TSSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.tssc_helpers.estimation import fit_variant
from mlsynth.utils.tssc_helpers.selection import _restriction_matrices, select_method
from mlsynth.utils.tssc_helpers.setup import prepare_tssc_inputs
from mlsynth.utils.tssc_helpers.structures import (
    METHODS,
    TSSCInputs,
    TSSCResults,
    TSSCVariantFit,
)


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(n_units=6, T=30, T0=20, seed=0, treated_effect=0.0,
                treated_intercept=0.0, rho=0.6):
    """Factor + AR(1) panel with a single treated unit (unit 0)."""
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = rho * common[t - 1] + rng.standard_normal()

    Y = np.zeros((T, n_units))
    for i in range(n_units):
        load = rng.standard_normal()
        Y[:, i] = 10.0 + load * common + rng.standard_normal(T) * 0.4
    Y[:, 0] += treated_intercept
    if treated_effect:
        Y[T0:, 0] += treated_effect

    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "treat": int(i == 0 and t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel_df():
    return _make_panel()


@pytest.fixture
def inputs(panel_df):
    return prepare_tssc_inputs(df=panel_df, outcome="y", unitid="unitid",
                               time="time", treat="treat")


# =========================================================================
# CONFIG
# =========================================================================

class TestTSSCConfig:

    def test_defaults(self, panel_df):
        cfg = TSSCConfig(df=panel_df, outcome="y", unitid="unitid",
                         time="time", treat="treat")
        assert cfg.alpha == 0.05
        assert cfg.subsample_size is None
        assert cfg.draws == 500
        assert cfg.ci == 0.95

    def test_alpha_bounds(self, panel_df):
        with pytest.raises(Exception):
            TSSCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", alpha=0.0)
        with pytest.raises(Exception):
            TSSCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", alpha=1.0)

    def test_subsample_size_floor(self, panel_df):
        with pytest.raises(Exception):
            TSSCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", subsample_size=1)

    def test_invalid_dict_wrapped(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid TSSC configuration"):
            TSSC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                  "time": "time", "treat": "treat", "alpha": -1})


# =========================================================================
# DATA PREP
# =========================================================================

class TestPrepareInputs:

    def test_shapes(self, inputs):
        assert inputs.donor_matrix.shape == (30, 5)
        assert inputs.y.shape == (30,)
        assert inputs.n_donors == 5
        assert inputs.T0 == 20
        assert inputs.T2 == 10
        assert inputs.T == 30

    def test_too_few_pre_periods(self):
        rows = []
        for i in range(3):
            for t in range(2):
                rows.append({"unitid": f"u{i}", "time": t, "y": float(t + i),
                             "treat": int(i == 0 and t >= 1)})
        df = pd.DataFrame(rows)
        with pytest.raises(MlsynthDataError, match="two pre-treatment"):
            prepare_tssc_inputs(df=df, outcome="y", unitid="unitid",
                                time="time", treat="treat")


# =========================================================================
# VARIANT ESTIMATION
# =========================================================================

class TestVariantEstimation:

    def test_all_four_fit(self, inputs):
        for method in METHODS:
            fit = fit_variant(inputs, method, n_bootstrap=50)
            assert isinstance(fit, TSSCVariantFit)
            assert np.isfinite(fit.att)
            assert fit.counterfactual.shape == (inputs.T,)
            assert fit.gap.shape == (inputs.T,)

    def test_intercept_presence(self, inputs):
        # SC and MSCb impose a zero intercept; MSCa and MSCc carry one.
        assert fit_variant(inputs, "SC", 50).intercept is None
        assert fit_variant(inputs, "MSCb", 50).intercept is None
        assert fit_variant(inputs, "MSCa", 50).intercept is not None
        assert fit_variant(inputs, "MSCc", 50).intercept is not None

    def test_sc_weights_sum_to_one(self, inputs):
        # SC (SIMPLEX) donor weights should sum to ~1.
        fit = fit_variant(inputs, "SC", 50)
        assert abs(sum(fit.donor_weights.values()) - 1.0) < 1e-2

    def test_att_ci_brackets(self, inputs):
        fit = fit_variant(inputs, "MSCc", 100)
        lo, hi = fit.att_ci
        assert lo <= hi


# =========================================================================
# STEP-1 SELECTION
# =========================================================================

class TestSelection:

    def test_restriction_matrices(self):
        R = _restriction_matrices(4)  # p = N = 4 (intercept + 3 donors)
        R_joint, q_joint = R["joint"]
        assert R_joint.shape == (2, 4)
        # Row 0: sum of donor slopes; row 1: intercept.
        np.testing.assert_array_equal(R_joint[0], [0, 1, 1, 1])
        np.testing.assert_array_equal(R_joint[1], [1, 0, 0, 0])
        np.testing.assert_array_equal(q_joint, [1.0, 0.0])
        Ra, qa = R["sum_to_one"]
        np.testing.assert_array_equal(Ra[0], [0, 1, 1, 1]); assert qa[0] == 1.0
        Rb, qb = R["zero_intercept"]
        np.testing.assert_array_equal(Rb[0], [1, 0, 0, 0]); assert qb[0] == 0.0

    def test_returns_valid_method(self, inputs):
        sel = select_method(inputs, alpha=0.05, n_subsamples=200, seed=0)
        assert sel.recommended in METHODS
        assert "joint" in sel.tests
        assert len(sel.decision_path) >= 1
        assert sel.mscc_beta.shape == (inputs.n_donors + 1,)

    def test_reproducible(self, inputs):
        a = select_method(inputs, n_subsamples=150, seed=42)
        b = select_method(inputs, n_subsamples=150, seed=42)
        assert a.recommended == b.recommended
        assert a.tests["joint"].statistic == b.tests["joint"].statistic

    def test_tree_short_circuits(self, inputs):
        # If joint is not rejected, only the joint test is recorded.
        sel = select_method(inputs, n_subsamples=200, seed=0)
        if not sel.tests["joint"].rejected:
            assert sel.recommended == "SC"
            assert set(sel.tests) == {"joint"}


# =========================================================================
# ESTIMATOR
# =========================================================================

class TestTSSCEstimator:

    def test_smoke(self, panel_df):
        res = TSSC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                    "time": "time", "treat": "treat", "draws": 200,
                    "seed": 0, "display_graphs": False}).fit()
        assert isinstance(res, TSSCResults)
        assert res.mode == "tssc"
        assert set(res.variants) == set(METHODS)
        assert res.recommended_method in METHODS
        assert np.isfinite(res.att)
        # Summary mirrors the recommended variant.
        assert res.summary.effects.att == res.recommended.att
        assert res.summary.method_details.method_name.startswith("TSSC (")

    def test_recovers_effect_sign(self):
        df = _make_panel(treated_effect=3.0, seed=3)
        res = TSSC({"df": df, "outcome": "y", "unitid": "unitid",
                    "time": "time", "treat": "treat", "draws": 200,
                    "seed": 1, "display_graphs": False}).fit()
        assert res.att > 1.0  # injected +3.0 effect

    def test_reproducible(self, panel_df):
        kw = {"df": panel_df, "outcome": "y", "unitid": "unitid",
              "time": "time", "treat": "treat", "draws": 150, "seed": 7,
              "display_graphs": False}
        assert TSSC(kw).fit().recommended_method == TSSC(kw).fit().recommended_method

    def test_att_ci_per_method(self, panel_df):
        res = TSSC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                    "time": "time", "treat": "treat", "draws": 200,
                    "seed": 2, "display_graphs": False}).fit()
        cis = res.att_ci_by_method()
        atts = res.att_by_method()
        # Every variant carries a finite, ordered CI bracketing nothing absurd.
        assert set(cis) == set(METHODS)
        for method in METHODS:
            lo, hi = cis[method]
            assert np.isfinite(lo) and np.isfinite(hi)
            assert lo <= hi
        # The recommended variant's CI matches the summary inference block.
        rec_lo, rec_hi = cis[res.recommended_method]
        assert res.summary.inference.ci_lower == rec_lo
        assert res.summary.inference.ci_upper == rec_hi


# =========================================================================
# PLOTTER
# =========================================================================

class TestPlotter:

    def test_plot_smoke(self, panel_df):
        import matplotlib
        matplotlib.use("Agg")
        from mlsynth.utils.tssc_helpers.plotter import plot_tssc
        res = TSSC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                    "time": "time", "treat": "treat", "draws": 100,
                    "seed": 0, "display_graphs": False}).fit()
        plot_tssc(res)


# =========================================================================
# IMMUTABILITY
# =========================================================================

class TestImmutability:

    def test_inputs_frozen(self, inputs):
        with pytest.raises(FrozenInstanceError):
            inputs.T0 = 99

    def test_results_frozen(self, panel_df):
        res = TSSC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                    "time": "time", "treat": "treat", "draws": 80,
                    "seed": 0, "display_graphs": False}).fit()
        # TSSCResults is now a frozen Pydantic EffectResult; attribute
        # assignment raises pydantic's ValidationError ("frozen_instance").
        with pytest.raises(ValidationError):
            res.summary = None
