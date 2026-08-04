"""Tests for the GSYNTH estimator (Xu 2017, generalized synthetic control).

Layered along agents_tests.md:

* Layer 1 (numerical helpers): the Bai-normalized principal components, the
  interactive fixed effects ALS, Steps 2-3, and Algorithm 1.
* Layer 2 (data utilities): ``prepare_gsynth_inputs`` ingestion and validation.
* Layer 3 (estimator integration): recovery on a factor DGP, and the Xu (2017)
  Table 2 columns (3) and (4) reproduction on the author's EDR panel.
* Layer 4 (public API contracts): import, frozen result, two-family surface.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import GSYNTH
from mlsynth.config_models import GSYNTHConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.gsynth_helpers.ife import (
    ControlFit,
    interactive_fixed_effects,
    panel_factor,
)
from mlsynth.utils.gsynth_helpers.pipeline import (
    cross_validate_rank,
    fit_control_model,
    gsc_fit,
)
from mlsynth.utils.gsynth_helpers.setup import prepare_gsynth_inputs
from mlsynth.utils.gsynth_helpers.structures import (
    GSYNTHDesign,
    GSYNTHInputs,
    GSYNTHResults,
)
from mlsynth.utils.gsynth_helpers.inference import parametric_bootstrap

BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
TURNOUT = BASEDATA / "xu_edr_turnout.parquet"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _staggered_panel(
    n_control: int = 25, n_treated: int = 4, T: int = 40, r_true: int = 2,
    effect: float = 4.0, noise: float = 0.5, seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Interactive-fixed-effects DGP with staggered, absorbing adoption.

    ``Y_it(0) = mu + alpha_i + xi_t + lambda_i' f_t + eps_it`` and a constant
    additive effect on treated post-periods, so the true ATT is ``effect``.

    The noise level is deliberately not tiny. Algorithm 1 scores a rank by how
    well it predicts held-out treated pre-periods, and on a nearly noiseless
    panel the components past the true rank are near-zero directions that
    neither help nor hurt, so the criterion flattens and the selected rank is
    arbitrary above ``r_true``. At a realistic signal-to-noise the spurious
    components mispredict and the criterion separates.
    """
    rng = np.random.default_rng(seed)
    N = n_control + n_treated
    F = rng.standard_normal((T, r_true))
    lam = rng.standard_normal((N, r_true))
    alpha = rng.standard_normal(N)
    xi = rng.standard_normal(T)
    Y = 5.0 + alpha[None, :] + xi[:, None] + F @ lam.T
    Y = Y + noise * rng.standard_normal((T, N))

    D = np.zeros((T, N), dtype=int)
    adopt = np.linspace(T - 12, T - 4, n_treated).astype(int)
    for k in range(n_treated):
        D[adopt[k]:, n_control + k] = 1
    Y = Y + effect * D

    units = [f"u{i:02d}" for i in range(N)]
    rows = {
        "unit": np.repeat(units, T),
        "time": np.tile(np.arange(T), N),
        "y": Y.T.ravel(),
        "d": D.T.ravel(),
    }
    return pd.DataFrame(rows), effect


@pytest.fixture(scope="module")
def staggered():
    return _staggered_panel()


@pytest.fixture(scope="module")
def turnout() -> pd.DataFrame:
    if not TURNOUT.exists():  # pragma: no cover - vendored with the repo
        pytest.skip("xu_edr_turnout.parquet not present")
    return pd.read_parquet(TURNOUT)


def _cfg(df: pd.DataFrame, **kw) -> dict:
    base = dict(df=df, unitid="unit", time="time", outcome="y", treat="d",
                display_graphs=False, inference=False)
    base.update(kw)
    return base


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestPanelFactor:
    def test_bai_normalization(self):
        """F'F / T = I_r is the normalization Step 1 is defined under."""
        rng = np.random.default_rng(1)
        E = rng.standard_normal((20, 40))
        F, lam = panel_factor(E, 3)
        assert F.shape == (20, 3) and lam.shape == (40, 3)
        np.testing.assert_allclose(F.T @ F / 20, np.eye(3), atol=1e-9)

    def test_tall_panel_normalization(self):
        """The N < T branch normalizes the loadings instead, Lambda'Lambda/N = I."""
        rng = np.random.default_rng(2)
        E = rng.standard_normal((40, 12))
        F, lam = panel_factor(E, 3)
        assert F.shape == (40, 3) and lam.shape == (12, 3)
        np.testing.assert_allclose(lam.T @ lam / 12, np.eye(3), atol=1e-9)

    def test_rank_r_matrix_is_reproduced(self):
        """A rank-r matrix is spanned exactly by its first r components."""
        rng = np.random.default_rng(3)
        F0, L0 = rng.standard_normal((25, 2)), rng.standard_normal((30, 2))
        E = F0 @ L0.T
        F, lam = panel_factor(E, 2)
        np.testing.assert_allclose(F @ lam.T, E, atol=1e-8)

    def test_zero_factors_returns_empty(self):
        F, lam = panel_factor(np.zeros((6, 8)), 0)
        assert F.shape == (6, 0) and lam.shape == (8, 0)

    def test_fit_reports_its_rank(self):
        rng = np.random.default_rng(9)
        fit = interactive_fixed_effects(rng.standard_normal((12, 20)),
                                        np.empty((12, 20, 0)), 3)
        assert fit.r == 3


class TestInteractiveFixedEffects:
    def test_recovers_additive_effects_without_factors(self):
        """With r = 0 the fit is the two-way within estimator."""
        rng = np.random.default_rng(4)
        T, N = 15, 12
        alpha, xi = rng.standard_normal(N), rng.standard_normal(T)
        Y = 3.0 + alpha[None, :] + xi[:, None]
        fit = interactive_fixed_effects(Y, np.empty((T, N, 0)), 0)
        recon = fit.mu + fit.alpha[None, :] + fit.xi[:, None]
        np.testing.assert_allclose(recon, Y, atol=1e-9)

    def test_recovers_beta_on_a_noiseless_panel(self):
        rng = np.random.default_rng(5)
        T, N = 20, 15
        X = rng.standard_normal((T, N, 2))
        F, lam = rng.standard_normal((T, 2)), rng.standard_normal((N, 2))
        beta_true = np.array([1.5, -0.75])
        Y = (2.0 + rng.standard_normal(N)[None, :] + rng.standard_normal(T)[:, None]
             + np.einsum("tnk,k->tn", X, beta_true) + F @ lam.T)
        fit = interactive_fixed_effects(Y, X, 2)
        np.testing.assert_allclose(fit.beta, beta_true, atol=1e-6)

    def test_factor_space_is_recovered(self):
        """The estimated factors span the true factor space."""
        rng = np.random.default_rng(6)
        T, N = 30, 40
        F, lam = rng.standard_normal((T, 2)), rng.standard_normal((N, 2))
        Y = F @ lam.T
        fit = interactive_fixed_effects(Y, np.empty((T, N, 0)), 2, two_way=False)
        # Projection onto the estimated space leaves the true factors alone.
        P = fit.factor @ np.linalg.pinv(fit.factor)
        Fc = F - F.mean(axis=1, keepdims=True) * 0  # F itself, kept explicit
        np.testing.assert_allclose(P @ Fc, Fc, atol=1e-7)

    def test_time_invariant_covariate_is_dropped(self):
        """A covariate with no within variation cannot be identified and is
        dropped with a zero coefficient instead of making X'X singular."""
        rng = np.random.default_rng(7)
        T, N = 12, 10
        X = np.empty((T, N, 2))
        X[:, :, 0] = rng.standard_normal((T, N))
        X[:, :, 1] = 1.0                                # absorbed by the intercept
        Y = np.einsum("tnk,k->tn", X, np.array([2.0, 0.0]))
        fit = interactive_fixed_effects(Y, X, 0)
        assert fit.dropped == (1,)
        assert fit.beta[1] == 0.0
        assert abs(fit.beta[0] - 2.0) < 1e-8

    def test_iteration_is_bounded(self):
        rng = np.random.default_rng(8)
        T, N = 10, 8
        fit = interactive_fixed_effects(
            rng.standard_normal((T, N)), rng.standard_normal((T, N, 1)), 1,
            max_iter=3)
        assert fit.niter <= 3


class TestGscFit:
    def test_recovers_a_constant_effect(self, staggered):
        df, effect = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=2)
        assert abs(out.att - effect) < 0.2

    def test_effect_is_averaged_over_treated_post_cells(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=2)
        mask = inputs.D[:, inputs.treated_index] > 0
        assert out.n_post_cells == int(mask.sum())
        np.testing.assert_allclose(out.att, out.effect[mask].mean(), rtol=1e-12)

    def test_loadings_fit_only_pre_periods(self, staggered):
        """Changing a treated unit's post-period outcomes moves its effect but
        not its loading -- Step 2 sees pre-treatment data only."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        base = gsc_fit(inputs, r=2)

        bumped = df.copy()
        hit = (bumped.unit == "u25") & (bumped.d == 1)
        bumped.loc[hit, "y"] += 10.0
        moved = gsc_fit(prepare_gsynth_inputs(bumped, "y", "d", "unit", "time"), r=2)
        np.testing.assert_allclose(moved.loadings, base.loadings, atol=1e-10)
        assert moved.att > base.att

    def test_two_way_adds_an_intercept_column(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=2)
        assert out.factors_used.shape[1] == 3          # r factors + the intercept
        np.testing.assert_allclose(out.factors_used[:, -1], 1.0)
        assert out.unit_effects.shape == (inputs.n_treated,)

    def test_zero_rank_is_two_way_did(self, staggered):
        """r = 0 reduces to the two-way fixed effects counterfactual."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=0)
        assert out.factors_used.shape[1] == 1
        assert np.isfinite(out.att)

    def test_counterfactual_matches_observed_minus_effect(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=2)
        obs = inputs.Y[:, inputs.treated_index]
        np.testing.assert_allclose(out.counterfactual, obs - out.effect, atol=1e-9)

    def test_rank_beyond_pre_periods_raises(self, staggered):
        """A unit cannot identify more loadings than it has pre-periods."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        with pytest.raises(MlsynthEstimationError, match="pre-treatment"):
            gsc_fit(inputs, r=inputs.T)

    def test_one_way_omits_the_intercept_column(self, staggered):
        """Without additive unit effects Step 2 projects onto the factors
        alone, and the treated units carry no separate level."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        out = gsc_fit(inputs, r=2, two_way=False)
        assert out.factors_used.shape[1] == 2
        np.testing.assert_allclose(out.unit_effects, 0.0)

    def test_singular_loading_regression_is_translated(self, staggered):
        """A degenerate factor matrix surfaces as MlsynthEstimationError, not a
        raw LinAlgError."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        fit = fit_control_model(inputs, 2)
        rank_deficient = ControlFit(
            mu=fit.mu, alpha=fit.alpha, xi=fit.xi, beta=fit.beta,
            factor=np.zeros_like(fit.factor), lam=fit.lam)
        with pytest.raises(MlsynthEstimationError, match="singular"):
            gsc_fit(inputs, r=2, control_fit=rank_deficient)


class TestCrossValidateRank:
    def test_selects_the_generating_rank(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        cv = cross_validate_rank(inputs, r_max=4)
        assert cv.r_selected == 2
        assert set(cv.mspe) == {0, 1, 2, 3, 4}
        assert cv.mspe[2] == min(cv.mspe.values())

    def test_held_out_cell_count_does_not_depend_on_rank(self, staggered):
        """Algorithm 1's MSPE is comparable across r only if the same cells are
        scored at every r."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        cv = cross_validate_rank(inputs, r_max=4)
        assert len(set(cv.n_scored.values())) == 1

    def test_is_deterministic(self, staggered):
        """Leave-one-period-out draws no random numbers, so the selected rank
        is a property of the data."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        a = cross_validate_rank(inputs, r_max=4)
        b = cross_validate_rank(inputs, r_max=4)
        assert a.r_selected == b.r_selected and a.mspe == b.mspe

    def test_rank_ceiling_is_respected(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        cv = cross_validate_rank(inputs, r_max=99)
        assert max(cv.mspe) <= inputs.min_pre_periods - 2

    def test_one_way_ceiling_is_one_higher(self, staggered):
        """Without an intercept there is one fewer regressor to identify."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        cv = cross_validate_rank(inputs, r_max=99, two_way=False)
        assert max(cv.mspe) == inputs.min_pre_periods - 1

    def test_no_scorable_rank_raises(self, staggered):
        """A treated unit with a pre-period history too short to hold a period
        back from leaves nothing to cross-validate on."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        short = GSYNTHInputs(
            Y=inputs.Y, D=inputs.D, X=inputs.X,
            treated_index=inputs.treated_index,
            control_index=inputs.control_index,
            adoption_index=np.ones_like(inputs.adoption_index),
            unit_labels=inputs.unit_labels, time_labels=inputs.time_labels)
        with pytest.raises(MlsynthEstimationError, match="could not score"):
            cross_validate_rank(short, r_max=3)


# ----------------------------------------------------------------------
# Layer 2: ingestion
# ----------------------------------------------------------------------

class TestSetup:
    def test_shapes_and_ordering(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        assert inputs.Y.shape == (inputs.T, inputs.N)
        assert inputs.D.shape == inputs.Y.shape
        assert inputs.X.shape == (inputs.T, inputs.N, 0)
        assert inputs.n_treated == 4 and inputs.n_control == 25
        assert list(inputs.time_labels) == sorted(df.time.unique())

    def test_covariates_are_aligned_to_the_outcome_panel(self, turnout):
        inputs = prepare_gsynth_inputs(
            turnout, "turnout", "policy_edr", "abb", "year",
            covariates=["policy_mail_in", "policy_motor"])
        assert inputs.X.shape == (24, 47, 2)
        assert inputs.covariate_names == ("policy_mail_in", "policy_motor")
        j = list(inputs.unit_labels).index("CA")
        ref = (turnout[turnout.abb == "CA"].sort_values("year")
               .policy_motor.to_numpy(dtype=float))
        np.testing.assert_allclose(inputs.X[:, j, 1], ref)

    def test_control_index_is_never_treated_only(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        assert inputs.D[:, inputs.control_index].sum() == 0
        assert set(inputs.control_index) | set(inputs.treated_index) == set(
            range(inputs.N))

    def test_no_never_treated_raises(self, staggered):
        df, _ = staggered
        df = df.copy()
        df.loc[df.time >= 20, "d"] = 1
        with pytest.raises(MlsynthDataError, match="never-treated"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_no_treated_raises(self, staggered):
        df, _ = staggered
        df = df.assign(d=0)
        with pytest.raises(MlsynthDataError, match="treated"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_treatment_reversal_raises(self, staggered):
        """GSC's Step 2 needs one absorbing adoption date per treated unit."""
        df, _ = staggered
        df = df.copy()
        adopt = int(df[(df.unit == "u25") & (df.d == 1)].time.min())
        df.loc[(df.unit == "u25") & (df.time == adopt + 1), "d"] = 0
        with pytest.raises(MlsynthDataError, match="sustained|absorbing"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_treatment_from_the_first_period_raises(self, staggered):
        df, _ = staggered
        df = df.copy()
        df.loc[df.unit == "u25", "d"] = 1
        with pytest.raises(MlsynthDataError, match="pre-treatment"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_missing_covariate_column_raises(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthDataError, match="nope"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time", covariates=["nope"])

    def test_non_binary_treatment_raises(self, staggered):
        df, _ = staggered
        df = df.copy()
        df.loc[df.d == 1, "d"] = 2
        with pytest.raises(MlsynthDataError, match="0/1"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_missing_treatment_cell_raises(self, staggered):
        df, _ = staggered
        df = df.copy()
        df.loc[df.index[0], "d"] = np.nan
        with pytest.raises(MlsynthDataError, match="missing"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time")

    def test_missing_covariate_cell_raises(self, staggered):
        df, _ = staggered
        df = df.assign(z=1.0)
        df.loc[df.index[3], "z"] = np.nan
        with pytest.raises(MlsynthDataError, match="fully observed"):
            prepare_gsynth_inputs(df, "y", "d", "unit", "time", covariates=["z"])

    def test_unbalanced_panel_raises(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthDataError):
            prepare_gsynth_inputs(df.iloc[1:], "y", "d", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_smoke(self, staggered):
        df, effect = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert isinstance(res, GSYNTHResults)
        assert np.isfinite(res.effects.att)
        assert abs(res.effects.att - effect) < 0.2

    def test_accepts_a_config_object(self, staggered):
        df, _ = staggered
        res = GSYNTH(GSYNTHConfig(**_cfg(df, r=2))).fit()
        assert np.isfinite(res.effects.att)

    def test_cross_validation_is_the_default(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df)).fit()
        assert res.design.r_selected == 2
        assert res.design.r_source == "cv"

    def test_fixed_rank_reports_its_source(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=1)).fit()
        assert res.design.r_selected == 1 and res.design.r_source == "user"
        assert res.design.cv is None

    def test_event_study_horizons(self, staggered):
        """Horizon 0 is the first treated period, matching STACKEDSC."""
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        es = res.event_study
        assert es.horizons[0] < 0 <= es.horizons[-1]
        assert 0 in set(es.horizons.tolist())
        assert len(es.horizons) == len(es.tau) == len(es.n_units)
        post = np.asarray(es.horizons) >= 0
        assert np.all(np.asarray(es.n_units)[post] >= 1)
        # Horizon 0 is the first treated period for every unit, so all of them
        # contribute to it.
        at_zero = int(np.flatnonzero(np.asarray(es.horizons) == 0)[0])
        assert es.n_units[at_zero] == res.inputs.n_treated

    def test_result_is_frozen(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        with pytest.raises(Exception):
            res.effects = None

    def test_time_series_contract(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        ts = res.time_series
        assert ts.observed_outcome.shape == ts.counterfactual_outcome.shape
        np.testing.assert_allclose(
            ts.estimated_gap,
            ts.observed_outcome - ts.counterfactual_outcome, atol=1e-9)

    def test_pre_period_fit_is_reported(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert res.fit_diagnostics.rmse_pre is not None
        assert res.fit_diagnostics.rmse_pre >= 0.0

    def test_per_unit_fits(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert set(res.per_unit) == {"u25", "u26", "u27", "u28"}
        for fit in res.per_unit.values():
            assert np.isfinite(fit.att)
            assert fit.n_post >= 1

    def test_inference_populates_a_standard_error(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, inference=True, r=2, n_bootstrap=30, seed=0)).fit()
        se = res.inference.standard_error
        assert se is not None and np.isfinite(se) and se > 0
        assert res.inference.ci_lower < res.effects.att < res.inference.ci_upper
        assert res.inference.method == "parametric bootstrap"

    def test_inference_is_seed_reproducible(self, staggered):
        df, _ = staggered
        kw = dict(inference=True, r=2, n_bootstrap=20, seed=7)
        a = GSYNTH(_cfg(df, **kw)).fit()
        b = GSYNTH(_cfg(df, **kw)).fit()
        assert a.inference.standard_error == b.inference.standard_error

    def test_inference_off_leaves_no_standard_error(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert res.inference.standard_error is None

    def test_inference_carries_a_horizon_band(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, inference=True, r=2, n_bootstrap=30, seed=1)).fit()
        inf = res.inference_detail
        assert inf.tau_lower.shape == inf.tau.shape == res.event_study.tau.shape
        assert np.all(inf.tau_lower <= inf.tau_upper)
        assert np.all(np.isfinite(inf.tau_se))

    def test_inference_with_covariates(self, turnout):
        """The bootstrap rebuilds the panel from the fitted surface, which
        includes the covariate contribution."""
        res = GSYNTH(dict(df=turnout, unitid="abb", time="year",
                          outcome="turnout", treat="policy_edr", r=2,
                          covariates=["policy_mail_in", "policy_motor"],
                          inference=True, n_bootstrap=25, seed=3,
                          display_graphs=False)).fit()
        assert np.isfinite(res.inference.standard_error)

    def test_too_few_surviving_draws_raises(self, staggered):
        """A bootstrap that keeps almost nothing reports that instead of
        returning a variance estimated from a handful of draws."""
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        fit = gsc_fit(inputs, r=2)
        with pytest.raises(MlsynthEstimationError, match="too few"):
            parametric_bootstrap(inputs, fit, r=2, n_bootstrap=1, seed=0)

    def test_failed_refits_are_discarded_and_counted(self, staggered, monkeypatch):
        """A draw whose refit fails is dropped, not silently treated as an
        estimate of zero."""
        from mlsynth.utils.gsynth_helpers import inference as inf_mod

        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        fit = gsc_fit(inputs, r=2)

        real, calls = inf_mod.gsc_fit, {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise MlsynthEstimationError("simulated refit failure")
            return real(*a, **k)

        monkeypatch.setattr(inf_mod, "gsc_fit", flaky)
        out = parametric_bootstrap(inputs, fit, r=2, n_bootstrap=40, seed=0)
        assert out.n_failed > 0
        assert np.isfinite(out.se)

    def test_a_pool_that_never_fits_raises(self, staggered, monkeypatch):
        from mlsynth.utils.gsynth_helpers import inference as inf_mod

        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        fit = gsc_fit(inputs, r=2)

        def always_fails(*a, **k):
            raise np.linalg.LinAlgError("simulated")

        monkeypatch.setattr(inf_mod, "gsc_fit", always_fails)
        with pytest.raises(MlsynthEstimationError, match="could not simulate"):
            parametric_bootstrap(inputs, fit, r=2, n_bootstrap=10, seed=0)

    def test_covariates_change_the_estimate(self, turnout):
        bare = GSYNTH(dict(df=turnout, unitid="abb", time="year", outcome="turnout",
                           treat="policy_edr", r=2, inference=False,
                           display_graphs=False)).fit()
        withx = GSYNTH(dict(df=turnout, unitid="abb", time="year", outcome="turnout",
                            treat="policy_edr", r=2, inference=False,
                            covariates=["policy_mail_in", "policy_motor"],
                            display_graphs=False)).fit()
        assert withx.design.beta.shape == (2,)
        assert bare.effects.att != withx.effects.att

    def test_plotting_smoke(self, staggered, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2, display_graphs=True)).fit()
        assert np.isfinite(res.effects.att)
        plt.close("all")

    def test_plotting_with_the_inference_band(self, staggered, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2, display_graphs=True, inference=True,
                          n_bootstrap=15, seed=2)).fit()
        assert np.isfinite(res.inference.standard_error)
        plt.close("all")


class TestConfigValidation:
    def test_extra_field_forbidden(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError):
            GSYNTH(_cfg(df, not_a_field=1))

    def test_negative_rank_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError):
            GSYNTH(_cfg(df, r=-1))

    def test_rank_above_ceiling_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError, match="r_max"):
            GSYNTH(_cfg(df, r=5, r_max=3))

    def test_bad_alpha_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError):
            GSYNTH(_cfg(df, alpha=1.5))

    def test_zero_bootstraps_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError):
            GSYNTH(_cfg(df, inference=True, n_bootstrap=0))

    def test_empty_covariate_list_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError, match="at least one"):
            GSYNTH(_cfg(df, covariates=[]))

    def test_repeated_covariate_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError, match="repeats"):
            GSYNTH(_cfg(df, covariates=["z", "z"]))

    def test_covariate_naming_a_reserved_column_rejected(self, staggered):
        df, _ = staggered
        with pytest.raises(MlsynthConfigError, match="may not name"):
            GSYNTH(_cfg(df, covariates=["y"]))

    def test_data_error_is_translated(self, staggered):
        """An ingestion failure surfaces as MlsynthDataError, not a raw
        pandas/NumPy error."""
        df, _ = staggered
        with pytest.raises(MlsynthDataError):
            GSYNTH(_cfg(df.assign(d=0))).fit()

    def test_plotting_failure_is_translated(self, staggered, monkeypatch):
        from mlsynth.estimators import gsynth as mod
        from mlsynth.exceptions import MlsynthPlottingError

        def boom(_results):
            raise RuntimeError("no display")

        monkeypatch.setattr(mod, "plot_gsynth", boom)
        df, _ = staggered
        with pytest.raises(MlsynthPlottingError, match="plotting failed"):
            GSYNTH(_cfg(df, r=2, display_graphs=True)).fit()

    def test_unexpected_failure_is_translated(self, staggered, monkeypatch):
        """Anything the pipeline raises that is not already an mlsynth error
        comes back as MlsynthEstimationError, never as a raw NumPy error."""
        from mlsynth.estimators import gsynth as mod

        def boom(*a, **k):
            raise RuntimeError("solver exploded")

        monkeypatch.setattr(mod, "gsc_fit", boom)
        df, _ = staggered
        with pytest.raises(MlsynthEstimationError, match="estimation failed"):
            GSYNTH(_cfg(df, r=2)).fit()


@pytest.mark.skipif(not TURNOUT.exists(), reason="turnout panel not vendored")
class TestXu2017Table2:
    """Xu (2017) Table 2, columns (3) and (4): the ATT of Election Day
    Registration on voter turnout, at two unobserved factors.

    The table prints two decimals, so a printed figure pins the true one only
    to within 0.005, and that is the tolerance every assertion here uses.
    """

    def _fit(self, turnout, covariates, **kw):
        cfg = dict(df=turnout, unitid="abb", time="year", outcome="turnout",
                   treat="policy_edr", display_graphs=False, r=2)
        cfg.setdefault("inference", False)
        cfg.update(kw)
        if covariates:
            cfg["covariates"] = list(covariates)
        return GSYNTH(cfg).fit()

    def test_panel_matches_the_table_header(self, turnout):
        inputs = prepare_gsynth_inputs(
            turnout, "turnout", "policy_edr", "abb", "year")
        assert inputs.T * inputs.N == 1128
        assert inputs.n_treated == 9 and inputs.n_control == 38

    def test_column_three_no_covariates(self, turnout):
        res = self._fit(turnout, None)
        assert abs(res.effects.att - 5.13) < 0.005

    def test_column_four_with_covariates(self, turnout):
        res = self._fit(turnout, ("policy_mail_in", "policy_motor"))
        assert abs(res.effects.att - 4.90) < 0.005

    def test_column_four_covariate_coefficients(self, turnout):
        """Table 2 column (4) prints 0.15 for mail-in and -1.05 for motor
        voter."""
        res = self._fit(turnout, ("policy_mail_in", "policy_motor"))
        np.testing.assert_allclose(res.design.beta, [0.15, -1.05], atol=0.005)

    def test_cross_validation_finds_two_factors(self, turnout):
        """The paper reports two unobserved factors under both
        specifications."""
        for covs in (None, ("policy_mail_in", "policy_motor")):
            cfg = dict(df=turnout, unitid="abb", time="year", outcome="turnout",
                       treat="policy_edr", display_graphs=False, inference=False,
                       r_max=5)
            if covs:
                cfg["covariates"] = list(covs)
            assert GSYNTH(cfg).fit().design.r_selected == 2

    def test_standard_error_is_near_the_published_2_27(self, turnout):
        """Column (3)'s error comes from a parametric bootstrap, so matching to
        the digit would mean matching an RNG stream. Ten percent catches a
        broken inference routine."""
        res = self._fit(turnout, None, inference=True, n_bootstrap=200, seed=2139)
        se = res.inference.standard_error
        assert abs(se - 2.27) / 2.27 < 0.10


# ----------------------------------------------------------------------
# Layer 4: public API
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_exported(self):
        import mlsynth
        assert "GSYNTH" in mlsynth.__all__
        assert mlsynth.GSYNTH is GSYNTH

    def test_config_reexported(self):
        from mlsynth.config_models import GSYNTHConfig as A
        from mlsynth.utils.gsynth_helpers.config import GSYNTHConfig as B
        assert A is B

    def test_effect_result_family(self, staggered):
        from mlsynth.config_models import BaseEstimatorResults

        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert isinstance(res, BaseEstimatorResults)
        for slot in ("effects", "time_series", "weights", "fit_diagnostics",
                     "inference", "method_details"):
            assert getattr(res, slot) is not None

    def test_structures_are_frozen(self, staggered):
        df, _ = staggered
        inputs = prepare_gsynth_inputs(df, "y", "d", "unit", "time")
        assert isinstance(inputs, GSYNTHInputs)
        with pytest.raises(Exception):
            inputs.T = 3
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert isinstance(res.design, GSYNTHDesign)
        with pytest.raises(Exception):
            res.design.r_selected = 9

    def test_method_name(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert res.method_details.method_name == "GSYNTH"

    def test_shorthand_accessors(self, staggered):
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        assert res.r == res.design.r_selected == 2
        assert res.att == res.effects.att

    def test_prepopulated_contract_is_left_alone(self, staggered):
        """Rebuilding a result whose standardized slots are already filled does
        not overwrite them."""
        df, _ = staggered
        res = GSYNTH(_cfg(df, r=2)).fit()
        rebuilt = GSYNTHResults(
            inputs=res.inputs, design=res.design, aggregate=res.aggregate,
            event_study=res.event_study,
            inference_detail=res.inference_detail, per_unit=res.per_unit,
            metadata=res.metadata, effects=res.effects,
            time_series=res.time_series, weights=res.weights,
            fit_diagnostics=res.fit_diagnostics, inference=res.inference,
            method_details=res.method_details)
        assert rebuilt.effects is res.effects
