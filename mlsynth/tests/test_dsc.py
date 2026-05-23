"""Tests for the Distributional Synthetic Control (DSC) estimator.

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): empirical_quantile, sample_quantile_grid,
  solve_simplex_weights, build_lambda_weights.
* Layer 2 (data utilities): prepare_dsc_inputs.
* Layer 3 (estimator integration): DSC.fit on a synthetic micro-panel
  with a planted location-shift treatment effect.
* Layer 4 (public API contracts): top-level import, frozen dataclasses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import DSC
from mlsynth.config_models import DSCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.dsc_helpers import (
    DSCInputs,
    DSCResults,
    QTECurve,
    aggregate_period_weights,
    build_lambda_weights,
    empirical_quantile,
    prepare_dsc_inputs,
    sample_quantile_grid,
    solve_simplex_weights,
)


# ----------------------------------------------------------------------
# Shared synthetic micro-panel fixture
# ----------------------------------------------------------------------

def _micro_panel(
    J: int = 4,
    T_pre: int = 8,
    T_post: int = 4,
    n_per_cell: int = 100,
    delta_post: float = 1.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """A balanced micro-panel where unit 0 has a *location shift* of
    ``delta_post`` in the post-period."""
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    # Unit-specific location, time-specific shift; donors all share the
    # same DGP so the treated unit's pre-period distribution is exactly
    # a convex combination of the donors' distributions.
    unit_loc = rng.standard_normal(J + 1) * 0.5
    time_shift = np.linspace(0.0, 1.0, T)
    rows = []
    for j in range(J + 1):
        for t in range(T):
            loc = unit_loc[j] + time_shift[t]
            if j == 0 and t >= T_pre:
                loc += delta_post
            sample = rng.normal(loc=loc, scale=1.0, size=n_per_cell)
            for y in sample:
                rows.append({
                    "unit": j,
                    "time": t,
                    "y": float(y),
                    "D": int(j == 0 and t >= T_pre),
                })
    return pd.DataFrame(rows), delta_post


@pytest.fixture
def micro_panel():
    return _micro_panel()


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestQuantiles:
    def test_empirical_quantile_matches_numpy(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = np.array([0.1, 0.5, 0.9])
        out = empirical_quantile(sample, q)
        expected = np.quantile(sample, q, method="inverted_cdf")
        np.testing.assert_array_equal(out, expected)

    def test_quantile_grid_qmc_in_unit_interval(self):
        V = sample_quantile_grid(M=100, method="halton", random_state=0)
        assert V.shape == (100,)
        assert (V > 0.0).all() and (V < 1.0).all()

    def test_quantile_grid_unknown_method_rejected(self):
        with pytest.raises(MlsynthEstimationError):
            sample_quantile_grid(M=100, method="bogus")


class TestWeights:
    def test_simplex_weights_recover_exact_mixture(self):
        # Donor 1 = quantile fn of N(0, 1); donor 2 = quantile fn of N(2, 1).
        # Treated = 0.3 * donor 1 + 0.7 * donor 2 (a valid mixture in
        # quantile space because both donors share scale).
        rng = np.random.default_rng(0)
        V = np.linspace(0.05, 0.95, 200)
        from scipy.stats import norm
        d1 = norm.ppf(V, loc=0.0, scale=1.0)
        d2 = norm.ppf(V, loc=2.0, scale=1.0)
        donor_matrix = np.column_stack([d1, d2])
        treated = 0.3 * d1 + 0.7 * d2
        w = solve_simplex_weights(donor_matrix, treated)
        np.testing.assert_allclose(w, [0.3, 0.7], atol=1e-3)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-9).all()


class TestLambdaWeights:
    def test_uniform_weights(self):
        lam = build_lambda_weights(T0=5, method="uniform")
        np.testing.assert_allclose(lam, np.full(5, 0.2))

    def test_recency_weights_peak_at_last_period(self):
        lam = build_lambda_weights(T0=5, method="recency", decay=0.5)
        assert lam[-1] > lam[0]
        assert abs(lam.sum() - 1.0) < 1e-9

    def test_aggregate_period_weights_average(self):
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        lam = np.array([0.4, 0.6])
        agg = aggregate_period_weights(W, lam)
        np.testing.assert_allclose(agg, [0.4, 0.6])


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_dsc_inputs_balanced(self, micro_panel):
        df, _ = micro_panel
        inputs = prepare_dsc_inputs(
            df, outcome="y", treat="D", unitid="unit", time="time",
        )
        assert isinstance(inputs, DSCInputs)
        assert inputs.T == 12
        assert inputs.T0 == 8
        assert inputs.J == 4
        assert inputs.treated_unit_name == 0
        # Each cell carries 100 individual observations.
        for arr in inputs.cell_samples.values():
            assert arr.shape == (100,)

    def test_missing_outcome_rejected(self, micro_panel):
        df, _ = micro_panel
        df.loc[0, "y"] = np.nan
        with pytest.raises(MlsynthDataError):
            prepare_dsc_inputs(
                df, outcome="y", treat="D", unitid="unit", time="time",
            )

    def test_unbalanced_panel_rejected(self, micro_panel):
        df, _ = micro_panel
        # Drop all observations for (unit=2, time=3) -- creates a hole.
        df = df[~((df["unit"] == 2) & (df["time"] == 3))]
        with pytest.raises(MlsynthDataError):
            prepare_dsc_inputs(
                df, outcome="y", treat="D", unitid="unit", time="time",
            )


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_default_fit_recovers_location_shift(self, micro_panel):
        df, delta = micro_panel
        res = DSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "M": 400,
        }).fit()
        assert isinstance(res, DSCResults)
        # Donor weights are on the simplex.
        ws = np.array(list(res.donor_weights.values()))
        assert (ws >= -1e-9).all()
        assert abs(ws.sum() - 1.0) < 1e-4
        # Average QTE across quantiles is close to the planted shift.
        assert abs(res.att - delta) < 0.4
        # QTE is roughly constant across quantiles for a pure location shift.
        spread = float(res.average_qte.max() - res.average_qte.min())
        assert spread < 0.6

    def test_per_period_qte_curves(self, micro_panel):
        df, _ = micro_panel
        res = DSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
        }).fit()
        assert len(res.qte_curves) == 4  # T - T0 = 12 - 8
        for curve in res.qte_curves:
            assert isinstance(curve, QTECurve)
            assert curve.observed.shape == curve.counterfactual.shape
            np.testing.assert_allclose(
                curve.qte, curve.observed - curve.counterfactual,
                atol=1e-12,
            )

    def test_custom_lambda_weights(self, micro_panel):
        df, _ = micro_panel
        custom = [0.0] * 7 + [1.0]  # T0 = 8; put all weight on last pre-period
        res = DSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "lambda_weights": custom,
        }).fit()
        np.testing.assert_allclose(res.lambda_weights, custom)

    def test_invalid_lambda_weights_rejected(self, micro_panel):
        df, _ = micro_panel
        with pytest.raises(MlsynthEstimationError):
            DSC({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "lambda_weights": [0.5, 0.5],  # length != T0
            }).fit()


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import DSC as _DSC
        assert _DSC is DSC

    def test_results_dataclasses_frozen(self, micro_panel):
        df, _ = micro_panel
        res = DSC({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "M": 200,
        }).fit()
        with pytest.raises(Exception):
            res.att = 0.0
        with pytest.raises(Exception):
            res.qte_curves[0].time_label = 99

    def test_unknown_grid_method_rejected(self, micro_panel):
        df, _ = micro_panel
        with pytest.raises(MlsynthConfigError):
            DSC({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "grid_method": "bogus",
            })
