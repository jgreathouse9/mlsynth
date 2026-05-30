"""Tests for the MASC (Kellogg, Mogstad, Pouliot & Torgovitsky 2021) estimator.

Layer 1 -- helper primitives: nearest-neighbour selection, the simplex SC
solver, the analytic-phi closed form, the per-fold covariate aggregator.
Layer 2 -- data setup validation: prepare_masc_inputs identifies the
treated unit, enforces a balanced panel, and assembles the covariate
panels with the expected shapes.
Layer 3 -- estimator integration: MASC.fit() runs end-to-end on tiny
synthetic panels, both with and without covariates, returning a
MASCResults object whose contents respect the documented invariants.
Layer 4 -- public API contracts: import paths, dict-vs-config equivalence,
exception translation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MASC
from mlsynth.config_models import MASCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
)
from mlsynth.utils.masc_helpers import (
    MASCFit,
    MASCInputs,
    MASCResults,
    analytic_phi,
    cross_validate,
    masc_combine,
    nearest_neighbor_weights,
    prepare_masc_inputs,
    run_masc,
    sc_simplex_weights,
)
from mlsynth.utils.masc_helpers.crossval import _aggregate_covariates


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _toy_panel(
    n_donors: int = 5, T: int = 15, T0: int = 10,
    seed: int = 0, effect: float = -2.0,
):
    """Tiny factor-model panel with one treated unit (``"u0"``).

    The treated unit is a noisy linear combination of donors ``u1`` and
    ``u2``; the others (``u3``..) are uncorrelated. Both matching and SC
    should therefore concentrate weight on ``u1`` and ``u2``.
    """
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.normal(size=T))
    rows = []
    units = ["u0"] + [f"u{j}" for j in range(1, n_donors + 1)]
    for u_idx, u in enumerate(units):
        for t in range(T):
            if u == "u0":
                y = 0.6 * factor[t] + rng.normal(scale=0.1)
            elif u in ("u1", "u2"):
                y = factor[t] + rng.normal(scale=0.1)
            else:
                y = rng.normal()
            if u == "u0" and t >= T0:
                y += effect
            rows.append({"unit": u, "t": t, "y": y, "treat": int(u == "u0" and t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _toy_panel()


@pytest.fixture
def panel_with_cov():
    df = _toy_panel()
    rng = np.random.default_rng(1)
    df["x1"] = rng.uniform(0, 1, size=len(df))
    df["x2"] = rng.uniform(0, 1, size=len(df))
    return df


def _cfg(df, **kw):
    base = dict(
        df=df, outcome="y", treat="treat", unitid="unit", time="t",
        display_graphs=False,
    )
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Layer 1 -- helper primitives
# --------------------------------------------------------------------------- #
class TestNearestNeighbors:
    def test_returns_simplex_weights_summing_to_one(self):
        Y0 = np.array([1.0, 2.0, 3.0])
        YJ = np.array([[1.1, 5.0, -1.0], [2.0, 4.5, -0.5], [3.0, 4.0, 0.0]])
        w = nearest_neighbor_weights(Y0, YJ, m=1)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= 0.0)
        # closest donor (column 0) gets all mass.
        assert w[0] == pytest.approx(1.0)

    def test_m_equals_J_gives_uniform_weights(self):
        Y0 = np.array([0.0, 0.0])
        YJ = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w = nearest_neighbor_weights(Y0, YJ, m=3)
        assert np.allclose(w, 1.0 / 3.0)

    def test_raises_when_m_exceeds_pool(self):
        Y0 = np.zeros(3)
        YJ = np.ones((3, 2))
        with pytest.raises(ValueError):
            nearest_neighbor_weights(Y0, YJ, m=3)


class TestSCSimplexSolver:
    def test_outcomes_only_recovers_the_obvious_match(self):
        # Treated = donor 0 plus tiny noise; SC should put all weight on donor 0.
        rng = np.random.default_rng(0)
        Y_d_pre = rng.normal(size=(8, 3))
        Y_t_pre = Y_d_pre[:, 0] + rng.normal(scale=1e-4, size=8)
        w = sc_simplex_weights(Y_t_pre, Y_d_pre)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= 0.0)
        assert w[0] > 0.95

    def test_covariate_branch_respects_simplex(self):
        rng = np.random.default_rng(1)
        Y_d_pre = rng.normal(size=(8, 4))
        Y_t_pre = Y_d_pre @ np.array([0.5, 0.3, 0.2, 0.0])
        X_treated = np.array([1.0, 0.5, 2.0])
        X_donors = rng.normal(size=(3, 4))
        w = sc_simplex_weights(
            Y_t_pre, Y_d_pre, X_treated=X_treated, X_donors=X_donors,
        )
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= -1e-9)


class TestAnalyticPhi:
    def test_phi_clipped_to_unit_interval(self):
        # Y_match coincides with Y_treated => phi tends to 1.
        Y_treated = np.array([1.0, 2.0, 3.0])
        Y_match = np.array([1.0, 2.0, 3.0])
        Y_sc = np.array([5.0, 5.0, 5.0])
        w = np.full(3, 1.0 / 3.0)
        assert analytic_phi(Y_treated, Y_match, Y_sc, w) == pytest.approx(1.0)

    def test_phi_zero_when_sc_perfect(self):
        Y_treated = np.array([1.0, 2.0, 3.0])
        Y_match = np.array([5.0, 5.0, 5.0])
        Y_sc = np.array([1.0, 2.0, 3.0])  # exactly tracks treated
        w = np.full(3, 1.0 / 3.0)
        # |num| = 0, phi -> 0.
        assert analytic_phi(Y_treated, Y_match, Y_sc, w) == pytest.approx(0.0)

    def test_phi_zero_when_match_equals_sc(self):
        # denominator vanishes; helper should fall back to 0.
        Y_treated = np.array([1.0, 2.0])
        Y_match = np.array([3.0, 3.0])
        Y_sc = np.array([3.0, 3.0])
        w = np.array([0.5, 0.5])
        assert analytic_phi(Y_treated, Y_match, Y_sc, w) == 0.0


class TestMascCombine:
    def test_convex_combination(self):
        wm = np.array([1.0, 0.0])
        ws = np.array([0.0, 1.0])
        wc = masc_combine(wm, ws, phi=0.3)
        assert wc[0] == pytest.approx(0.3)
        assert wc[1] == pytest.approx(0.7)
        assert wc.sum() == pytest.approx(1.0)


class TestAggregateCovariates:
    def test_returns_none_when_no_panels(self):
        out = _aggregate_covariates(None, None, (), None, pre_end_period=5)
        assert out == (None, None)

    def test_default_window_uses_pre_end_period(self):
        T, J, P = 6, 3, 2
        # treated panel: covariate-by-time block
        cov_treated = np.arange(T * P).reshape(T, P).astype(float)
        cov_donors = np.ones((T, J, P))
        time_index = np.arange(T)
        Xt, Xd = _aggregate_covariates(
            cov_treated, cov_donors, ("a", "b"), time_index, pre_end_period=3,
        )
        # average of rows 0..2 of cov_treated.
        assert Xt.shape == (P,)
        assert Xd.shape == (P, J)
        assert np.allclose(Xt, cov_treated[:3].mean(axis=0))


# --------------------------------------------------------------------------- #
# Layer 2 -- data setup validation
# --------------------------------------------------------------------------- #
class TestPrepareInputs:
    def test_identifies_treated_unit_and_intervention(self, panel):
        inputs = prepare_masc_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="t",
        )
        assert inputs.treated_label == "u0"
        assert inputs.intervention_time == 10
        assert inputs.T0 == 10
        assert inputs.T1 == 5

    def test_donor_pool_excludes_treated_unit(self, panel):
        inputs = prepare_masc_inputs(
            panel, outcome="y", treat="treat", unitid="unit", time="t",
        )
        assert "u0" not in inputs.donor_labels
        assert len(inputs.donor_labels) == 5

    def test_rejects_zero_treated_rows(self, panel):
        panel = panel.copy()
        panel["treat"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_masc_inputs(
                panel, outcome="y", treat="treat", unitid="unit", time="t",
            )

    def test_rejects_multiple_treated_units(self, panel):
        df = panel.copy()
        # Mark a second unit as treated.
        df.loc[(df["unit"] == "u1") & (df["t"] >= 10), "treat"] = 1
        with pytest.raises(MlsynthDataError):
            prepare_masc_inputs(
                df, outcome="y", treat="treat", unitid="unit", time="t",
            )

    def test_covariates_attach_panels_with_expected_shapes(self, panel_with_cov):
        inputs = prepare_masc_inputs(
            panel_with_cov, outcome="y", treat="treat",
            unitid="unit", time="t",
            covariates=["x1", "x2"],
        )
        assert inputs.has_covariates
        assert inputs.cov_treated_panel.shape == (inputs.T, 2)
        assert inputs.cov_donors_panel.shape == (inputs.T, inputs.J, 2)
        assert inputs.covariate_names == ("x1", "x2")


# --------------------------------------------------------------------------- #
# Layer 3 -- estimator integration
# --------------------------------------------------------------------------- #
class TestEstimatorPipeline:
    def test_fit_returns_results_object(self, panel):
        res = MASC(_cfg(panel)).fit()
        assert isinstance(res, MASCResults)
        assert isinstance(res.fit, MASCFit)
        assert isinstance(res.inputs, MASCInputs)

    def test_weights_form_simplex(self, panel):
        res = MASC(_cfg(panel)).fit()
        assert np.all(res.weights >= -1e-9)
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_cv_grid_has_expected_shape(self, panel):
        res = MASC(_cfg(panel, m_grid=[1, 2, 3], min_preperiods=5)).fit()
        assert res.fit.cv_grid.shape == (3, 3)
        # m_hat is one of the candidates.
        assert res.m_hat in {1, 2, 3}
        # phi_hat is in [0, 1].
        assert 0.0 <= res.phi_hat <= 1.0

    def test_phi_zero_recovers_pure_sc(self, panel):
        res = MASC(_cfg(panel, m_grid=[1], min_preperiods=5)).fit()
        if res.phi_hat == 0.0:
            # MASC weights collapse to the SC weights.
            assert np.allclose(res.weights, res.fit.weights_sc)

    def test_phi_one_recovers_pure_match(self, panel):
        res = MASC(_cfg(panel, m_grid=[1], min_preperiods=5)).fit()
        if res.phi_hat == 1.0:
            assert np.allclose(res.weights, res.fit.weights_match)

    def test_counterfactual_has_correct_length(self, panel):
        res = MASC(_cfg(panel)).fit()
        assert res.counterfactual.shape == (res.inputs.T,)
        assert res.gap.shape == (res.inputs.T,)

    def test_att_sign_matches_synthetic_effect(self, panel):
        # The synthetic panel has effect = -2; the estimated ATT should be negative.
        res = MASC(_cfg(panel)).fit()
        assert res.att < 0.0

    def test_pipeline_runs_with_covariates(self, panel_with_cov):
        res = MASC(_cfg(panel_with_cov, covariates=["x1", "x2"])).fit()
        assert isinstance(res, MASCResults)
        assert np.isclose(res.weights.sum(), 1.0)


# --------------------------------------------------------------------------- #
# Layer 4 -- public API contracts
# --------------------------------------------------------------------------- #
class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import MASC as _MASC
        from mlsynth.config_models import MASCConfig as _Cfg
        assert _MASC is MASC
        assert _Cfg is MASCConfig

    def test_dict_config_equivalent_to_typed_config(self, panel):
        cfg_obj = MASCConfig(**_cfg(panel))
        r1 = MASC(cfg_obj).fit()
        r2 = MASC(_cfg(panel)).fit()
        assert r1.m_hat == r2.m_hat
        assert r1.phi_hat == pytest.approx(r2.phi_hat)
        assert r1.att == pytest.approx(r2.att)

    def test_results_donor_weights_dict_aligned(self, panel):
        res = MASC(_cfg(panel)).fit()
        # Donor-weights dict keys match the donor labels.
        assert set(res.donor_weights.keys()) == set(res.inputs.donor_labels)
        # Values sum to 1.
        assert sum(res.donor_weights.values()) == pytest.approx(1.0)

    def test_invalid_config_translates_to_mlsynth_error(self, panel):
        # `outcome` does not exist in the panel; setup should raise
        # MlsynthDataError after Pydantic validation passes.
        with pytest.raises((MlsynthConfigError, MlsynthDataError)):
            MASC(_cfg(panel, outcome="not_a_column")).fit()
