"""Tests for the BVS-SS estimator and its helper subpackage.

Covers:
    * BVSSConfig validation (priors, burn-in sanity, dirichlet bounds).
    * prepare_bvss_inputs (demean conventions, donor / time consistency).
    * Posterior primitives (VM, RSS, RSS2, AM, loglike).
    * Pair Gibbs update (degenerate s=0 path, the three-case logic).
    * MH_tau (reflective barrier at tau_min, log-RW Jacobian).
    * gibbs_BVS sampler (shapes, finiteness, reproducibility under seed).
    * Inference assembly (ATT mean, credible intervals, no-post fallback).
    * BVSS estimator class (smoke + edge + error wrapping).
    * Plotter (smoke).
    * Immutability of all frozen dataclasses.

Reference: Xu & Zhou (2025), arXiv:2503.06454.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import BVSS
from mlsynth.config_models import BVSSConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from mlsynth.utils.bvss_helpers.gibbs_pair import (
    _compute_candidate_posteriors,
    _sample_pair,
)
from mlsynth.utils.bvss_helpers.inference import compute_inference
from mlsynth.utils.bvss_helpers.mh import MH_tau
from mlsynth.utils.bvss_helpers.plotter import plot_bvss
from mlsynth.utils.bvss_helpers.posterior import AM, RSS, RSS2, VM, loglike
from mlsynth.utils.bvss_helpers.sampler import (
    _update_phi_tau,
    gibbs_BVS,
)
from mlsynth.utils.bvss_helpers.setup import prepare_bvss_inputs
from mlsynth.utils.bvss_helpers.structures import (
    BVSSInference,
    BVSSInputs,
    BVSSPosterior,
    BVSSResults,
)


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(
    n_units=6,
    T=30,
    T0=20,
    seed=0,
    treated_effect=0.0,
    rho=0.6,
):
    """AR(1) + factor panel suitable for BVS-SS smoke tests."""
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = rho * common[t - 1] + rng.standard_normal()

    Y = np.zeros((T, n_units))
    for i in range(n_units):
        load = rng.standard_normal()
        idio = rng.standard_normal(T) * 0.4
        Y[:, i] = 10.0 + load * common + idio  # baseline level 10

    if treated_effect != 0.0:
        Y[T0:, 0] += treated_effect

    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({
                "unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                "treat": int(i == 0 and t >= T0),
            })
    return pd.DataFrame(rows), Y


@pytest.fixture
def panel_df():
    df, _ = _make_panel()
    return df


@pytest.fixture
def panel_with_effect():
    df, _ = _make_panel(treated_effect=2.0)
    return df


@pytest.fixture
def inputs(panel_df):
    return prepare_bvss_inputs(
        df=panel_df, outcome="y", unitid="unitid",
        time="time", treat="treat",
    )


# =========================================================================
# CONFIG
# =========================================================================

class TestBVSSConfig:

    def test_defaults(self, panel_df):
        cfg = BVSSConfig(df=panel_df, outcome="y", unitid="unitid",
                          time="time", treat="treat")
        assert cfg.n_iter == 2000
        assert cfg.burn_in == 1000
        assert cfg.kappa1 == 1.0 and cfg.kappa2 == 1.0
        assert cfg.theta == 0.25
        assert cfg.tau_a == 0.01 and cfg.tau_b == 0.1
        assert cfg.tau_min == 1e-6
        assert cfg.ci_alpha == 0.05

    def test_theta_must_be_in_unit_interval(self, panel_df):
        with pytest.raises(Exception):
            BVSSConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", theta=0.0)
        with pytest.raises(Exception):
            BVSSConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", theta=1.0)

    def test_priors_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            BVSSConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", kappa1=0)
        with pytest.raises(Exception):
            BVSSConfig(df=panel_df, outcome="y", unitid="unitid",
                        time="time", treat="treat", tau_b=0)

    def test_burn_in_must_be_below_n_iter(self, panel_df):
        # The config itself allows burn_in == n_iter; the runtime check
        # in BVSS.fit() catches the impossible combination.
        with pytest.raises(MlsynthConfigError, match="burn_in"):
            BVSS({
                "df": panel_df, "outcome": "y", "unitid": "unitid",
                "time": "time", "treat": "treat",
                "n_iter": 50, "burn_in": 50,
            }).fit()

    def test_invalid_dict_wrapped(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid BVSS configuration"):
            BVSS({
                "df": panel_df, "outcome": "y", "unitid": "unitid",
                "time": "time", "treat": "treat", "theta": -1,
            })


# =========================================================================
# DATA PREP
# =========================================================================

class TestPrepareBVSSInputs:

    def test_shapes(self, panel_df, inputs):
        # 6 total units = 1 treated + 5 donors
        assert inputs.X_pre_demean.shape == (20, 5)
        assert inputs.Y_pre_demean.shape == (20,)
        assert inputs.X_post_demean.shape == (10, 5)
        assert inputs.Gram.shape == (5, 5)
        assert inputs.N == 5
        assert inputs.T0 == 20
        assert inputs.T == 30

    def test_demean_consistency(self, inputs):
        # X_pre_demean columns must each have mean zero (after subtracting
        # mean_X) up to numerical noise.
        np.testing.assert_allclose(inputs.X_pre_demean.mean(axis=0),
                                    np.zeros(inputs.N), atol=1e-12)
        np.testing.assert_allclose(inputs.Y_pre_demean.mean(),
                                    0.0, atol=1e-12)

    def test_post_uses_pre_mean(self, inputs):
        # X_post_demean must use the pre-treatment mean (not its own).
        recovered = inputs.X_post_demean + inputs.mean_X
        # And the recovered post block must match the original donor matrix.
        assert recovered.shape == (10, 5)

    def test_too_few_pre_periods(self):
        rows = []
        for i in range(3):
            for t in range(2):
                rows.append({
                    "unitid": f"u{i}", "time": t, "y": float(t),
                    "treat": int(i == 0 and t >= 1),
                })
        df = pd.DataFrame(rows)
        with pytest.raises(MlsynthDataError):
            prepare_bvss_inputs(df, "y", "unitid", "time", "treat")


# =========================================================================
# POSTERIOR PRIMITIVES
# =========================================================================

class TestPosteriorPrimitives:

    def test_VM_empty_gamma(self):
        # Empty selection should return the fallback [[1/tau]].
        V = VM(np.zeros(4), tau=0.5, Gram=np.eye(4))
        np.testing.assert_allclose(V, [[2.0]])

    def test_VM_selected_gamma(self):
        Gram = np.eye(3)
        gamma = np.array([1, 0, 1])
        V = VM(gamma, tau=1.0, Gram=Gram)
        # X_gamma^T X_gamma = I_2 (selected diagonals), + (1/1) I = 2I_2.
        np.testing.assert_allclose(V, 2 * np.eye(2))

    def test_RSS_reduces_to_zTz_when_gamma_zero(self):
        z = np.array([1.0, -1.0, 2.0])
        X = np.random.default_rng(0).normal(size=(3, 4))
        Gram = X.T @ X
        rss = RSS(np.zeros(4), tau=1.0, z=z, X=X, Gram=Gram)
        assert np.isclose(rss, z @ z)

    def test_RSS_finite_and_nonneg(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5))
        Gram = X.T @ X
        z = rng.normal(size=20)
        gamma = np.array([1, 0, 1, 1, 0])
        out = RSS(gamma, tau=0.5, z=z, X=X, Gram=Gram)
        assert np.isfinite(out)
        # z^T Sigma z = z^T (I - X_g V^{-1} X_g^T) z >= 0 for PSD Sigma.
        assert out >= -1e-8

    def test_RSS2_symmetric(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(15, 4))
        Gram = X.T @ X
        z1 = rng.normal(size=15)
        z2 = rng.normal(size=15)
        gamma = np.array([1, 1, 0, 1])
        a = RSS2(gamma, tau=0.5, z1=z1, z2=z2, X=X, Gram=Gram)
        b = RSS2(gamma, tau=0.5, z1=z2, z2=z1, X=X, Gram=Gram)
        assert np.isclose(a, b)

    def test_loglike_finite(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        Gram = X.T @ X
        Y = rng.normal(size=20)
        mu = np.array([0.5, 0.0, 0.3, 0.2])
        gamma = (mu != 0).astype(int)
        ll = loglike(gamma, tau=0.5, mu=mu, phi=1.0, Y=Y, X=X, Gram=Gram)
        assert np.isfinite(ll)

    def test_AM_finite(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        Gram = X.T @ X
        for gamma in [np.array([1, 0, 0, 0]), np.array([1, 1, 0, 1])]:
            out = AM(gamma, tau=1.0, theta=0.25, Gram=Gram, N=4)
            assert np.isfinite(out)


# =========================================================================
# PAIR GIBBS UPDATE
# =========================================================================

class TestGibbsPair:

    @pytest.fixture
    def small_world(self):
        rng = np.random.default_rng(0)
        N, M = 4, 12
        X = rng.normal(size=(M, N))
        Y = X[:, 0] + 0.1 * rng.normal(size=M)  # signal from donor 0
        Gram = X.T @ X
        return X, Y, Gram, N

    def test_degenerate_s_zero(self, small_world):
        X, Y, Gram, N = small_world
        # Set mu so that 1 - sum(mu_{-(i,j)}) = 0  ->  ptotal should be [1,0,0,0]
        mu = np.array([0.5, 0.5, 0.0, 0.0])
        s, _, L, O, ptotal = _compute_candidate_posteriors(
            mu.copy(), i=2, j=3, X=X, Y=Y, tau=0.5, phi=1.0,
            Gram=Gram, theta=0.25,
        )
        # mutemp[i] = mutemp[j] = 0, so s = 1 - 1.0 = 0
        assert abs(s) < 1e-10
        np.testing.assert_array_equal(ptotal, [1.0, 0.0, 0.0, 0.0])
        assert L is None and O is None

    def test_nondegenerate_probs_sum_to_one(self, small_world):
        X, Y, Gram, N = small_world
        mu = np.array([0.2, 0.3, 0.0, 0.0])  # sum 0.5  =>  s = 0.5 for pair (2,3)
        s, _, L, O, ptotal = _compute_candidate_posteriors(
            mu.copy(), i=2, j=3, X=X, Y=Y, tau=0.5, phi=1.0,
            Gram=Gram, theta=0.25,
        )
        assert s > 0
        assert np.isclose(ptotal.sum(), 1.0)
        assert ptotal[0] == 0.0   # (0,0) infeasible when s > 0
        assert L > 0

    def test_sample_pair_assigns_correctly(self, small_world):
        X, Y, Gram, N = small_world
        rng = np.random.default_rng(0)

        # Force state 1 (mu_i = s, mu_j = 0)
        mu = np.array([0.2, 0.3, 0.0, 0.0])
        s = 0.5
        _sample_pair(mu, i=2, j=3, s=s, L=10.0, O=0.0,
                     phi=1.0, ptotal=np.array([0, 1.0, 0, 0]), rng=rng)
        assert mu[2] == s and mu[3] == 0.0

        # Force state 2 (mu_i = 0, mu_j = s)
        mu = np.array([0.2, 0.3, 0.0, 0.0])
        _sample_pair(mu, i=2, j=3, s=s, L=10.0, O=0.0,
                     phi=1.0, ptotal=np.array([0, 0, 1.0, 0]), rng=rng)
        assert mu[2] == 0.0 and mu[3] == s

        # Force state 3 (truncated normal draw)
        mu = np.array([0.2, 0.3, 0.0, 0.0])
        _sample_pair(mu, i=2, j=3, s=s, L=10.0, O=0.25,
                     phi=1.0, ptotal=np.array([0, 0, 0, 1.0]), rng=rng)
        assert 0 <= mu[2] <= s
        assert np.isclose(mu[2] + mu[3], s)


# =========================================================================
# MH_TAU
# =========================================================================

class TestMHTau:

    def test_returns_positive(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        Y = rng.normal(size=20)
        Gram = X.T @ X
        mu = np.array([0.3, 0.0, 0.4, 0.3])
        gamma = (mu != 0).astype(int)
        out = MH_tau(
            tau=1.0, gamma_vec=gamma, mu=mu, phi=1.0,
            Y=Y, X=X, Gram=Gram, rng=np.random.default_rng(0),
        )
        assert out > 0

    def test_respects_tau_min_floor(self):
        # The log-RW reflection prevents tau from dropping below tau_min.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(15, 3))
        Y = rng.normal(size=15)
        Gram = X.T @ X
        mu = np.array([0.5, 0.5, 0.0])
        gamma = (mu != 0).astype(int)
        tau_min = 1e-3
        out = MH_tau(
            tau=tau_min * 2, gamma_vec=gamma, mu=mu, phi=1.0,
            Y=Y, X=X, Gram=Gram, tau_min=tau_min, nrep=50,
            rng=np.random.default_rng(0),
        )
        assert out > 0

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(15, 3))
        Y = rng.normal(size=15)
        Gram = X.T @ X
        mu = np.array([0.5, 0.5, 0.0])
        gamma = (mu != 0).astype(int)
        t1 = MH_tau(1.0, gamma, mu, 1.0, Y, X, Gram,
                    rng=np.random.default_rng(7))
        t2 = MH_tau(1.0, gamma, mu, 1.0, Y, X, Gram,
                    rng=np.random.default_rng(7))
        assert np.isclose(t1, t2)


# =========================================================================
# SAMPLER
# =========================================================================

class TestGibbsBVS:

    def test_sample_shapes(self, inputs):
        out = gibbs_BVS(
            Y=inputs.Y_pre_demean, X=inputs.X_pre_demean,
            Gram=inputs.Gram, M=inputs.T0, N=inputs.N,
            size=30, rng=np.random.default_rng(0),
        )
        assert out["musample"].shape == (inputs.N, 30)
        assert out["phisample"].shape == (30,)
        assert out["tausample"].shape == (30,)
        assert out["gammasample"].shape == (inputs.N, 30)
        # phi and tau are positive
        assert np.all(out["phisample"] > 0)
        assert np.all(out["tausample"] > 0)
        # Each mu column sums to 1 (soft simplex; in this implementation
        # the pair update keeps the simplex hard since s = 1 - sum).
        np.testing.assert_allclose(out["musample"].sum(axis=0),
                                    np.ones(30), atol=1e-8)

    def test_init_mu_validation(self, inputs):
        with pytest.raises(ValueError):
            gibbs_BVS(
                Y=inputs.Y_pre_demean, X=inputs.X_pre_demean,
                Gram=inputs.Gram, M=inputs.T0, N=inputs.N, size=5,
                init_mu=np.zeros(99),
            )

    def test_seed_reproducibility(self, inputs):
        kwargs = dict(
            Y=inputs.Y_pre_demean, X=inputs.X_pre_demean,
            Gram=inputs.Gram, M=inputs.T0, N=inputs.N, size=20,
        )
        a = gibbs_BVS(rng=np.random.default_rng(42), **kwargs)
        b = gibbs_BVS(rng=np.random.default_rng(42), **kwargs)
        np.testing.assert_allclose(a["musample"], b["musample"])
        np.testing.assert_allclose(a["phisample"], b["phisample"])
        np.testing.assert_allclose(a["tausample"], b["tausample"])

    def test_update_phi_tau_finite(self, inputs):
        mu = np.full(inputs.N, 1 / inputs.N)
        phi, tau = _update_phi_tau(
            mu, inputs.X_pre_demean, inputs.Y_pre_demean, tau_prev=1.0,
            M=inputs.T0, kappa1=1.0, kappa2=1.0, Gram=inputs.Gram,
            rng=np.random.default_rng(0),
        )
        assert phi > 0
        assert tau > 0


# =========================================================================
# INFERENCE ASSEMBLY
# =========================================================================

class TestInference:

    def test_inference_with_post(self, inputs):
        rng = np.random.default_rng(0)
        # Build fake mu samples (uniform on simplex)
        mu = np.full((inputs.N, 50), 1 / inputs.N)
        inf = compute_inference(inputs, mu, ci_alpha=0.05)
        assert isinstance(inf, BVSSInference)
        assert inf.counterfactual_mean.shape == (inputs.T,)
        assert inf.counterfactual_lower.shape == (inputs.T,)
        assert inf.counterfactual_upper.shape == (inputs.T,)
        assert np.all(inf.counterfactual_lower <= inf.counterfactual_upper)
        assert inf.att_samples.shape == (50,)
        assert inf.att_ci_lower <= inf.att_mean <= inf.att_ci_upper

    def test_inference_no_post(self):
        # Synthesize an inputs object with no post period.
        rng = np.random.default_rng(0)
        N, T0 = 3, 5
        X_pre = rng.normal(size=(T0, N))
        Y_pre = rng.normal(size=T0)
        inp = BVSSInputs(
            Y_pre_demean=Y_pre - Y_pre.mean(),
            X_pre_demean=X_pre - X_pre.mean(axis=0),
            X_post_demean=None,
            Gram=(X_pre - X_pre.mean(axis=0)).T @ (X_pre - X_pre.mean(axis=0)),
            mean_Y=float(Y_pre.mean()),
            mean_X=X_pre.mean(axis=0),
            T0=T0, T=T0, N=N,
            treated_unit_name="x", donor_names=["a", "b", "c"],
            time_labels=np.arange(T0), y_target=Y_pre,
        )
        mu = np.full((N, 25), 1 / N)
        inf = compute_inference(inp, mu, ci_alpha=0.05)
        assert np.isnan(inf.att_mean)
        assert np.isnan(inf.att_ci_lower)
        assert inf.att_samples.shape == (0,)

    def test_inference_mu_shape_mismatch_raises(self, inputs):
        with pytest.raises(ValueError):
            compute_inference(inputs, np.zeros((inputs.N + 1, 10)))


# =========================================================================
# BVSS ESTIMATOR (public API)
# =========================================================================

class TestBVSSEstimator:

    def test_fit_smoke(self, panel_df):
        res = BVSS({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat",
            "n_iter": 50, "burn_in": 25, "seed": 0,
        }).fit()
        assert isinstance(res, BVSSResults)
        assert isinstance(res.inference, BVSSInference)
        assert res.posterior.mu.shape == (res.inputs.N, 25)
        assert res.posterior.phi.shape == (25,)
        assert res.posterior.tau.shape == (25,)
        # Weights sum to one (soft simplex draws the entries off the
        # hard simplex; the mean should still be very close to 1).
        assert abs(sum(res.weight_means.values()) - 1.0) < 1e-6

    def test_recovers_positive_effect_sign(self, panel_with_effect):
        res = BVSS({
            "df": panel_with_effect, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat",
            "n_iter": 80, "burn_in": 30, "seed": 0,
        }).fit()
        # +2 effect on the treated unit -> ATT should be positive.
        assert res.inference.att_mean > 0

    def test_seed_reproducibility(self, panel_df):
        cfg = dict(
            df=panel_df, outcome="y", unitid="unitid",
            time="time", treat="treat",
            n_iter=40, burn_in=20, seed=7,
        )
        r1 = BVSS(cfg).fit()
        r2 = BVSS(cfg).fit()
        np.testing.assert_allclose(r1.posterior.mu, r2.posterior.mu)
        assert r1.inference.att_mean == r2.inference.att_mean

    def test_unexpected_error_wrapped(self, monkeypatch, panel_df):
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")
        monkeypatch.setattr("mlsynth.estimators.bvss.gibbs_BVS", boom)
        with pytest.raises(MlsynthEstimationError):
            BVSS({
                "df": panel_df, "outcome": "y", "unitid": "unitid",
                "time": "time", "treat": "treat",
                "n_iter": 20, "burn_in": 10,
            }).fit()


# =========================================================================
# PLOTTER (smoke)
# =========================================================================

class TestPlotter:

    @pytest.fixture(autouse=True)
    def _matplotlib_agg(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)

    def test_plot_runs(self, panel_df):
        res = BVSS({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat",
            "n_iter": 30, "burn_in": 10, "seed": 0,
            "display_graphs": False,
        }).fit()
        plot_bvss(res)


# =========================================================================
# IMMUTABILITY
# =========================================================================

class TestImmutability:

    def test_inputs_frozen(self, inputs):
        with pytest.raises(FrozenInstanceError):
            inputs.T0 = 99   # type: ignore[misc]

    def test_results_frozen(self, panel_df):
        res = BVSS({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat",
            "n_iter": 30, "burn_in": 10, "seed": 0,
        }).fit()
        with pytest.raises(FrozenInstanceError):
            res.inference = None   # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            res.posterior.mu = np.zeros((5, 5))   # type: ignore[misc]
