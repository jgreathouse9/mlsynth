"""Tests for the Modified Unbiased Synthetic Control estimator.

Layered following ``agents/agents_tests.md``:

* Layer 1 (numerical helpers): the QP and the unbiased-variance kernel.
* Layer 2 (data utilities): the panel preprocessor.
* Layer 3 (estimator integration): the public ``MUSC.fit()`` pipeline.
* Layer 4 (public API contracts): result-object surface, error semantics.
* Replication: Bottmer et al. (2024) Lemma 1 verified by Monte Carlo;
  Proposition 1 variance estimator verified against empirical
  Var_U[τ̂].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from mlsynth import MUSC
from mlsynth.config_models import MUSCConfig
from mlsynth.exceptions import (
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.musc_helpers import (
    MUSC as MUSC_NAME,
    SC as SC_NAME,
    MUSCInputs,
    MUSCResults,
    MUSCVariantFit,
    att_for_unit,
    prepare_musc_inputs,
    randomization_ci,
    run_musc,
    solve_musc_qp,
    unbiased_variance,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _factor_panel(rng, N=10, T_pre=20, T_post=3,
                   rho=0.7, sigma=1.0, mu_std=0.5):
    """Linear-factor panel under H0 (no treatment effect anywhere)."""
    T = T_pre + T_post
    mu = rng.normal(0.0, mu_std, size=N)
    eta = rng.normal(0.0, 1.0, size=T)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = rho * f[t - 1] + eta[t]
    lam = rng.normal(1.0, 0.3, size=N)
    eps = rng.normal(0.0, sigma, size=(T, N))
    return mu[None, :] + f[:, None] * lam[None, :] + eps


def _long_panel(Y, treated_idx=0, T0=None, unit_prefix="u"):
    """Pivot a (T, N) matrix into a long DataFrame with a sharp intervention."""
    T, N = Y.shape
    T0 = T - 3 if T0 is None else T0
    rows = []
    for j in range(N):
        for t in range(T):
            rows.append({
                "unit": f"{unit_prefix}{j:02d}",
                "time": t,
                "y": float(Y[t, j]),
                "treat": int(j == treated_idx and t >= T0),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def small_panel():
    Y = _factor_panel(np.random.default_rng(0), N=8, T_pre=12, T_post=3)
    return _long_panel(Y, treated_idx=0, T0=12)


@pytest.fixture
def medium_panel():
    Y = _factor_panel(np.random.default_rng(7), N=12, T_pre=20, T_post=4)
    return _long_panel(Y, treated_idx=0, T0=20)


# ---------------------------------------------------------------------------
# Layer 1 — QP and variance kernels
# ---------------------------------------------------------------------------

class TestQP:
    def test_qp_returns_shapes(self, medium_panel):
        Y_pre = (medium_panel.loc[medium_panel["time"] < 20]
                 .pivot(index="time", columns="unit", values="y")
                 .sort_index().to_numpy())
        M, status = solve_musc_qp(Y_pre, column_balance=True)
        assert status in ("optimal", "optimal_inaccurate")
        assert M.shape == (Y_pre.shape[1], Y_pre.shape[1] + 1)

    def test_diagonal_constraint_binds(self, medium_panel):
        """``M[i, i+1] = 1`` must hold for every i."""
        Y_pre = (medium_panel.loc[medium_panel["time"] < 20]
                 .pivot(index="time", columns="unit", values="y")
                 .sort_index().to_numpy())
        M, _ = solve_musc_qp(Y_pre, column_balance=True)
        np.testing.assert_allclose(np.diag(M[:, 1:]), 1.0, atol=1e-7)

    def test_offdiagonal_in_unit_interval(self, medium_panel):
        Y_pre = (medium_panel.loc[medium_panel["time"] < 20]
                 .pivot(index="time", columns="unit", values="y")
                 .sort_index().to_numpy())
        M, _ = solve_musc_qp(Y_pre, column_balance=True)
        off = np.ones_like(M[:, 1:], dtype=bool)
        np.fill_diagonal(off, False)
        W_off = M[:, 1:][off]
        assert W_off.max() <= 1e-7
        assert W_off.min() >= -1.0 - 1e-7

    def test_column_balance_binds_exactly_when_enabled(self, medium_panel):
        Y_pre = (medium_panel.loc[medium_panel["time"] < 20]
                 .pivot(index="time", columns="unit", values="y")
                 .sort_index().to_numpy())
        M_musc, _ = solve_musc_qp(Y_pre, column_balance=True)
        M_sc, _ = solve_musc_qp(Y_pre, column_balance=False)
        col_sums_musc = M_musc[:, 1:].sum(axis=0)
        col_sums_sc = M_sc[:, 1:].sum(axis=0)
        # MUSC: column sums == 0 to QP tolerance.
        np.testing.assert_allclose(col_sums_musc, 0.0, atol=1e-7)
        # SC: at least one column sum visibly non-zero.
        assert np.abs(col_sums_sc).max() > 1e-3

    def test_qp_rejects_degenerate_shape(self):
        with pytest.raises(MlsynthEstimationError):
            solve_musc_qp(np.zeros((1, 5)), column_balance=True)
        with pytest.raises(MlsynthEstimationError):
            solve_musc_qp(np.zeros((5, 2)), column_balance=True)


class TestUnbiasedVariance:
    def test_returns_finite_scalar(self, medium_panel):
        Y_wide = (medium_panel.pivot(index="time", columns="unit", values="y")
                   .sort_index().to_numpy())                     # (T, N)
        Y_pre = Y_wide[:20, :]
        M, _ = solve_musc_qp(Y_pre, column_balance=True)
        v = unbiased_variance(M, Y_wide[20, :])
        assert np.isfinite(v)

    def test_requires_minimum_N(self):
        with pytest.raises(ValueError):
            unbiased_variance(np.zeros((3, 4)), np.zeros(3))

    def test_shape_consistency(self):
        with pytest.raises(ValueError):
            unbiased_variance(np.zeros((4, 4)), np.zeros(4))      # missing intercept col


# ---------------------------------------------------------------------------
# Layer 2 — data preprocessor
# ---------------------------------------------------------------------------

class TestPrepareInputs:
    def test_round_trip_shapes(self, medium_panel):
        inputs = prepare_musc_inputs(
            medium_panel, unitid="unit", time="time", outcome="y",
            treated_unit="u00", intervention_time=20,
        )
        assert isinstance(inputs, MUSCInputs)
        assert inputs.N == 12
        assert inputs.T == 24
        assert inputs.T0 == 20
        assert inputs.treated_idx == 0
        assert inputs.donor_idx.tolist() == list(range(1, 12))

    def test_missing_treated_unit_raises(self, medium_panel):
        with pytest.raises(MlsynthDataError, match="treated_unit"):
            prepare_musc_inputs(
                medium_panel, unitid="unit", time="time", outcome="y",
                treated_unit="not-a-real-unit", intervention_time=20,
            )

    def test_intervention_outside_panel_raises(self, medium_panel):
        with pytest.raises(MlsynthDataError, match="intervention_time"):
            prepare_musc_inputs(
                medium_panel, unitid="unit", time="time", outcome="y",
                treated_unit="u00", intervention_time=9999,
            )

    def test_too_few_donors_raises(self):
        rng = np.random.default_rng(0)
        Y = _factor_panel(rng, N=3, T_pre=10, T_post=2)
        df = _long_panel(Y, treated_idx=0, T0=10)
        with pytest.raises(MlsynthDataError, match="at least 3 donor units"):
            prepare_musc_inputs(
                df, unitid="unit", time="time", outcome="y",
                treated_unit="u00", intervention_time=10,
            )


# ---------------------------------------------------------------------------
# Layer 3 — estimator integration
# ---------------------------------------------------------------------------

class TestEstimatorPipeline:
    def test_fit_runs_end_to_end(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": False,
        }).fit()
        assert isinstance(res, MUSCResults)
        assert SC_NAME in res.fits and MUSC_NAME in res.fits

    def test_fit_with_inference_returns_ci(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": True,
        }).fit()
        # Variance is finite; CI bounds are finite.
        assert np.isfinite(res.inference.variance)
        assert np.isfinite(res.inference.ci_normal[0])
        assert np.isfinite(res.inference.ci_normal[1])

    def test_skipping_inference_leaves_nan_ci(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": False,
        }).fit()
        assert np.isnan(res.inference.variance)
        assert np.isnan(res.inference.ci_normal[0])

    def test_donor_weights_are_canonical_sign(self, medium_panel):
        """The reported donor weights must be non-negative and bounded
        by 1 (the canonical SC sign), even though the internal matrix
        ``M`` uses the paper's negative parametrisation.
        """
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": False,
        }).fit()
        weights = np.array(list(res.donor_weights.values()))
        assert (weights >= -1e-7).all()
        assert (weights <= 1.0 + 1e-7).all()
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_rejects_panel_without_treatment(self):
        rng = np.random.default_rng(0)
        Y = _factor_panel(rng, N=8, T_pre=10, T_post=2)
        df = _long_panel(Y, treated_idx=0, T0=10)
        df["treat"] = 0
        with pytest.raises(MlsynthDataError, match="No rows with"):
            MUSC({
                "df": df, "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "time",
                "display_graphs": False, "run_inference": False,
            }).fit()

    def test_rejects_multiple_treated_units(self, medium_panel):
        df = medium_panel.copy()
        df.loc[(df["unit"] == "u01") & (df["time"] >= 20), "treat"] = 1
        with pytest.raises(MlsynthDataError, match="one treated unit"):
            MUSC({
                "df": df, "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "time",
                "display_graphs": False, "run_inference": False,
            }).fit()


# ---------------------------------------------------------------------------
# Layer 4 — config validation and public-API contracts
# ---------------------------------------------------------------------------

class TestMUSCConfig:
    @pytest.mark.parametrize("bad_alpha", [-0.01, 0.0, 1.0, 1.5])
    def test_alpha_out_of_range_rejected(self, medium_panel, bad_alpha):
        with pytest.raises(pydantic.ValidationError):
            MUSCConfig(
                df=medium_panel, outcome="y", unitid="unit",
                time="time", treat="treat", alpha=bad_alpha,
            )

    def test_defaults_match_documented_values(self, medium_panel):
        cfg = MUSCConfig(
            df=medium_panel, outcome="y", unitid="unit",
            time="time", treat="treat",
        )
        assert cfg.alpha == 0.05
        assert cfg.run_inference is True


class TestResultContract:
    def test_aliases_forward_to_musc_variant(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": False,
        }).fit()
        assert res.att == res.fits[MUSC_NAME].att
        np.testing.assert_array_equal(
            res.counterfactual, res.fits[MUSC_NAME].counterfactual,
        )
        assert res.donor_weights == res.fits[MUSC_NAME].donor_weights
        assert res.pre_rmse == res.fits[MUSC_NAME].pre_rmse


# ---------------------------------------------------------------------------
# Replication — Lemma 1 (unbiasedness under random unit assignment)
# ---------------------------------------------------------------------------

class TestLemma1Replication:
    """Bottmer et al. (2024) Lemma 1 says that under random assignment
    of which unit is treated, MUSC's ATT estimator is exactly
    unbiased while standard SC carries a bias.
    """

    def test_musc_bias_is_exactly_zero(self):
        rng = np.random.default_rng(0)
        Y = _factor_panel(rng, N=10, T_pre=20, T_post=3)
        Y_pre = Y[:20, :]
        M_musc, _ = solve_musc_qp(Y_pre, column_balance=True)
        M_sc, _ = solve_musc_qp(Y_pre, column_balance=False)

        # Per-unit ATTs (the exact expectation over random assignment).
        musc_atts = np.array([att_for_unit(M_musc, Y, i, 20)[2]
                              for i in range(Y.shape[1])])
        sc_atts = np.array([att_for_unit(M_sc, Y, i, 20)[2]
                            for i in range(Y.shape[1])])

        # Lemma 1: MUSC's average over assignments is identically zero.
        assert np.abs(musc_atts.mean()) < 1e-9
        # SC's average over assignments is *not* zero on the same panel.
        # We can't pin it to a specific number across DGPs, but it
        # should be visibly distinguishable from MUSC's exact zero.
        assert np.abs(sc_atts.mean()) > 1e-3

    def test_musc_unbiased_across_many_dgps(self):
        """Averaging MUSC's per-assignment expectation over 50 DGP
        draws should remain at machine zero (the column-sum
        constraint mathematically annihilates the bias formula 3.2,
        irrespective of the panel).
        """
        bias_estimates = []
        for rep in range(50):
            rng = np.random.default_rng(rep + 1)
            Y = _factor_panel(rng, N=10, T_pre=20, T_post=3)
            Y_pre = Y[:20, :]
            M, _ = solve_musc_qp(Y_pre, column_balance=True)
            atts = np.array([att_for_unit(M, Y, i, 20)[2]
                             for i in range(Y.shape[1])])
            bias_estimates.append(atts.mean())
        bias_estimates = np.array(bias_estimates)
        # Allow 1e-7 for QP solver tolerance × 50 panels.
        assert np.abs(bias_estimates).max() < 1e-6


class TestProposition1Replication:
    """Bottmer et al. (2024) Proposition 1: the closed-form V̂ is an
    unbiased estimator of Var_U[τ̂] under random unit assignment.
    Across many DGP draws ``E_Y[V̂]`` should match ``E_Y[Var_U[τ̂]]``.
    """

    def test_variance_matches_empirical_variance_on_average(self):
        v_hats, v_emps = [], []
        for rep in range(50):
            rng = np.random.default_rng(rep)
            Y = _factor_panel(rng, N=10, T_pre=20, T_post=3)
            Y_pre = Y[:20, :]
            M, _ = solve_musc_qp(Y_pre, column_balance=True)

            # Empirical Var_U[τ̂] at the first post-treatment period.
            taus = np.array([att_for_unit(M, Y, i, 20)[2]
                             for i in range(Y.shape[1])])
            # Single-period ATT collapses to τ̂_i = gap[i, T0]:
            taus_first = np.array(
                [(Y[20, i] - att_for_unit(M, Y, i, 20)[0][20])
                 for i in range(Y.shape[1])]
            )
            v_emps.append(taus_first.var(ddof=0))
            v_hats.append(unbiased_variance(M, Y[20, :]))

        v_hats = np.array(v_hats); v_emps = np.array(v_emps)
        # E[V̂] / E[Var_U] should be within ~10% over 50 panels.
        ratio = v_hats.mean() / v_emps.mean()
        assert 0.85 <= ratio <= 1.15

    def test_variance_is_non_negative_in_practice(self):
        """The Prop 1 estimator can be negative in finite samples in
        principle, but on well-behaved factor panels it stays
        non-negative.
        """
        for rep in range(20):
            rng = np.random.default_rng(rep)
            Y = _factor_panel(rng, N=10, T_pre=20, T_post=3)
            M, _ = solve_musc_qp(Y[:20, :], column_balance=True)
            v = unbiased_variance(M, Y[20, :])
            assert v >= -1e-6                       # tolerate solver fuzz


# ---------------------------------------------------------------------------
# Randomization CI smoke
# ---------------------------------------------------------------------------

class TestRandomizationCI:
    def test_ci_contains_finite_bounds(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": True,
        }).fit()
        lo, hi = res.att_ci
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi

    def test_placebo_count_equals_n_donors(self, medium_panel):
        res = MUSC({
            "df": medium_panel, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "display_graphs": False, "run_inference": True,
        }).fit()
        # One placebo per non-treated unit (after solver-failure filter).
        assert res.inference.placebo_atts.size <= res.inputs.n_donors
        assert res.inference.placebo_atts.size >= res.inputs.n_donors - 2
