"""Regression tests for DSCAR placebo and FDR inference (issue #1160)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import DSCAR
from mlsynth.utils.dscar_helpers import prepare_dsc_inputs
from mlsynth.utils.dscar_helpers.inference import (
    fdr_unconfoundedness_test,
    normalised_placebo_test,
)
from mlsynth.tests.test_dscar import _ar1_panel


class TestDSCARPlaceboInference:
    """Normalised placebo test unpacking, complete-case donor mask, and invariance."""

    def test_placebo_reps_returns_finite_se_and_atts(self):
        df = _ar1_panel(N=8, T=18, T0=12, tau=2.0, seed=0)
        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "placebo_reps": 5,
            "seed": 7,
            "display_graphs": False,
        }).fit()

        assert res.fit.placebo_atts is not None
        assert res.fit.placebo_atts.shape == (5,)
        assert np.all(np.isfinite(res.fit.placebo_atts))
        assert res.se is not None
        assert np.isfinite(res.se)
        assert res.se >= 0.0

    def test_placebo_applies_complete_case_donor_mask(self):
        """Missing donor cell at a placebo period excludes that donor from weights."""
        df = _ar1_panel(N=8, T=16, T0=10, tau=1.5, seed=1)
        # Drop one donor's outcome at a pre-period and one at a post-period.
        df.loc[(df["unit"] == "u2") & (df["year"] == 5), "y"] = np.nan
        df.loc[(df["unit"] == "u3") & (df["year"] == 12), "y"] = np.nan

        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "placebo_reps": 6,
            "seed": 3,
            "display_graphs": False,
        }).fit()

        assert res.fit.placebo_atts is not None
        assert res.fit.placebo_atts.shape == (6,)
        assert np.all(np.isfinite(res.fit.placebo_atts))
        assert res.se is not None
        assert np.isfinite(res.se)

    def test_placebo_excludes_inactive_donor_outcome(self):
        """An inactive donor at the final period gets zero weight in placebo draws."""
        df_a = _ar1_panel(N=7, T=14, T0=9, tau=0.0, seed=4)
        # u0 is never sampled as placebo_treated, so it is always a placebo donor.
        # Marking its covariate missing at the final period t=13 makes it inactive
        # at t=13 (with no t+1 lag reading it); changing y[u0, 13] from 0 to 1e6
        # must leave every placebo ATT identical.
        df_a.loc[(df_a["unit"] == "u0") & (df_a["year"] == 13), "x1"] = np.nan
        df_b = df_a.copy()
        df_a.loc[(df_a["unit"] == "u0") & (df_a["year"] == 13), "y"] = 0.0
        df_b.loc[(df_b["unit"] == "u0") & (df_b["year"] == 13), "y"] = 1e6

        inp_a = prepare_dsc_inputs(
            df=df_a,
            outcome="y",
            treat="treat",
            unitid="unit",
            time="year",
            exog_covariates=["x1"],
            lagged_outcome="y_lag1",
        )
        inp_b = prepare_dsc_inputs(
            df=df_b,
            outcome="y",
            treat="treat",
            unitid="unit",
            time="year",
            exog_covariates=["x1"],
            lagged_outcome="y_lag1",
        )
        zeros = np.zeros((inp_a.T, inp_a.N - inp_a.n_treated))
        atts_a, se_a = normalised_placebo_test(
            inputs=inp_a,
            weights=zeros,
            att_observed=0.0,
            placebo_reps=4,
            el_tolerance=1e-2,
            seed=11,
        )
        atts_b, se_b = normalised_placebo_test(
            inputs=inp_b,
            weights=zeros,
            att_observed=0.0,
            placebo_reps=4,
            el_tolerance=1e-2,
            seed=11,
        )
        assert atts_a.shape == (4,)
        assert np.all(np.isfinite(atts_a))
        np.testing.assert_allclose(atts_a, atts_b, rtol=1e-10, atol=1e-10)
        assert se_a == pytest.approx(se_b, rel=1e-10, abs=1e-10)

    def test_placebo_without_initial_lagged_outcome(self):
        """Placebo loop aligns V_diag when lagged_outcome is None (no lag at t=0)."""
        df = _ar1_panel(N=7, T=14, T0=9, tau=1.0, seed=2)
        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": None,
            "placebo_reps": 4,
            "seed": 5,
            "display_graphs": False,
        }).fit()

        assert res.fit.placebo_atts is not None
        assert np.all(np.isfinite(res.fit.placebo_atts))
        assert res.se is not None
        assert np.isfinite(res.se)

    def test_placebo_uniform_fallback_when_no_matching_constraints(self):
        """When Z1 is empty at t=0 (no covariates, no initial lag), placebo uses finite donors."""
        df = _ar1_panel(N=7, T=14, T0=9, tau=1.0, seed=8)
        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": None,
            "lagged_outcome": None,
            "placebo_reps": 3,
            "seed": 4,
            "display_graphs": False,
        }).fit()

        assert res.fit.placebo_atts is not None
        assert np.all(np.isfinite(res.fit.placebo_atts))
        assert res.se is not None
        assert np.isfinite(res.se)

    @settings(max_examples=8, deadline=None)
    @given(perm=st.permutations([1, 2, 3, 4, 5, 6]))
    def test_placebo_invariant_to_donor_row_order(self, perm):
        """On a complete panel, placebo statistics are invariant to donor order."""
        df = _ar1_panel(N=7, T=12, T0=8, tau=1.5, seed=0)
        ordered_units = ["u0"] + [f"u{i}" for i in perm]
        df_perm = df.copy()
        df_perm["_order"] = df_perm["unit"].map({u: idx for idx, u in enumerate(ordered_units)})
        df_perm = df_perm.sort_values(["_order", "year"]).drop(columns=["_order"]).reset_index(drop=True)

        base = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "placebo_reps": 3,
            "seed": 9,
            "display_graphs": False,
        }).fit()
        reordered = DSCAR({
            "df": df_perm,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "placebo_reps": 3,
            "seed": 9,
            "display_graphs": False,
        }).fit()

        np.testing.assert_allclose(reordered.fit.placebo_atts, base.fit.placebo_atts, rtol=1e-10, atol=1e-10)
        assert reordered.se == pytest.approx(base.se, rel=1e-10, abs=1e-10)


class TestDSCARFDRInference:
    """FDR unconfoundedness test filters incomplete donor rows before OLS."""

    def test_fdr_returns_finite_pvalues_with_missing_donor_cells(self):
        df = _ar1_panel(N=8, T=18, T0=12, tau=2.0, seed=5)
        df.loc[(df["unit"] == "u2") & (df["year"] == 5), "y"] = np.nan
        df.loc[(df["unit"] == "u3") & (df["year"] == 7), "x1"] = np.nan

        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "display_graphs": False,
        }).fit()

        pvals = res.fit.pre_period_pvalues
        min_p = res.fit.pre_period_min_pvalue_adj
        assert pvals is not None
        assert pvals.shape == (12,)
        assert np.all(np.isfinite(pvals))
        assert np.all((pvals >= 0.0) & (pvals <= 1.0))
        assert min_p is not None
        assert np.isfinite(min_p)
        assert min_p >= 0.0

    def test_fdr_handles_missing_treated_pre_period_outcome(self):
        df = _ar1_panel(N=8, T=18, T0=12, tau=2.0, seed=6)
        df.loc[(df["unit"] == "u0") & (df["year"] == 3), "y"] = np.nan

        inputs = prepare_dsc_inputs(
            df=df,
            outcome="y",
            treat="treat",
            unitid="unit",
            time="year",
            exog_covariates=["x1"],
            lagged_outcome="y_lag1",
        )
        res = DSCAR({
            "df": df,
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "year",
            "exog_covariates": ["x1"],
            "lagged_outcome": "y_lag1",
            "display_graphs": False,
        }).fit()

        pvals, min_p = fdr_unconfoundedness_test(
            inputs=inputs, weights=res.fit.weights, gap=res.fit.gap
        )
        assert np.isnan(pvals[3])
        observed = np.delete(pvals, 3)
        assert np.all(np.isfinite(observed))
        assert np.isfinite(min_p)
