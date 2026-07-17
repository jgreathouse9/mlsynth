"""Tests for the CSCIPCA estimator (Wang 2024, Instrumented PCA).

Layered per ``agents/agents_tests.md``:

* numerical core -- the vectorized ALS matches a readable Kronecker-loop
  oracle bit-for-bit, and the normalization leaves the counterfactual invariant;
* setup -- the covariate cube is built in dataprep's unit order, with the
  documented error paths;
* config -- validators reject blank/duplicate covariates and bad params;
* estimator -- recovery of a known effect, the paper's alpha-monotone bias
  finding, the conformal band, and the standardized result contract;
* smoke -- plotting.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import CSCIPCA
from mlsynth.config_models import CSCIPCAConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.cscipca_helpers.als import (
    als_estimate,
    counterfactual,
    normalize,
    solve_factors,
    solve_gamma,
)
from mlsynth.utils.cscipca_helpers.setup import prepare_cscipca_inputs
from mlsynth.utils.cscipca_helpers.structures import (
    CSCIPCADesign,
    CSCIPCAInference,
    CSCIPCAInputs,
    CSCIPCAResults,
)


# ----------------------------------------------------------------------
# Instrumented-PCA panel fixture (unit 0 treated; loadings = X_it Gamma)
# ----------------------------------------------------------------------
def _ipca_panel(
    N_co=20, T0=40, T1=8, L=4, K=2, tau=2.0, noise=0.1, alpha_obs=1.0, seed=0,
):
    """Single-treated CSC-IPCA DGP: Y_it = (X_it Gamma) F_t + noise.

    Returns ``(df, covariate_names, true_post_att)``. Only the first
    ``alpha_obs`` share of the ``L`` covariates are exposed as columns.
    """
    rng = np.random.default_rng(seed)
    N, T = N_co + 1, T0 + T1
    F = rng.standard_normal((K, T))
    X = rng.standard_normal((N, T, L))
    Gamma = rng.uniform(-0.3, 0.3, (L, K))
    Y = np.einsum("itl,lk,kt->it", X, Gamma, F) + noise * rng.standard_normal((N, T))
    Y[0, T0:] += tau                                    # unit 0 treated
    n_obs = max(1, int(round(alpha_obs * L)))
    covs = [f"x{l}" for l in range(n_obs)]
    rows = []
    for i in range(N):
        for t in range(T):
            row = {"unit": i, "time": t, "y": float(Y[i, t]),
                   "D": int(i == 0 and t >= T0)}
            for l in range(n_obs):
                row[f"x{l}"] = float(X[i, t, l])
            rows.append(row)
    return pd.DataFrame(rows), covs, float(tau)


def _cfg(df, covs, **kw):
    base = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                covariates=covs, n_factors=2, inference=False)
    base.update(kw)
    return base


@pytest.fixture
def panel():
    return _ipca_panel()


# ======================================================================
# Layer 1: numerical ALS core
# ======================================================================
def _naive_gamma(Y, X, F, K):
    """Readable Kronecker-loop oracle for the Gamma subproblem (reference form)."""
    N, T, L = X.shape
    numer = np.zeros(L * K)
    denom = np.zeros((L * K, L * K))
    for t in range(T):
        for i in range(N):
            kp = np.kron(X[i, t, :], F[:, t])
            numer += kp * Y[i, t]
            denom += np.outer(kp, kp)
    return np.linalg.solve(denom, numer).reshape(L, K)


def _naive_factors(Y, X, gamma):
    N, T, L = X.shape
    K = gamma.shape[1]
    Fo = np.zeros((K, T))
    for t in range(T):
        G = gamma.T @ X[:, t, :].T @ X[:, t, :] @ gamma
        b = gamma.T @ X[:, t, :].T @ Y[:, t]
        Fo[:, t] = np.linalg.solve(G, b)
    return Fo


class TestALSCore:
    def _data(self, seed=0):
        rng = np.random.default_rng(seed)
        N, T, L, K = 8, 15, 3, 2
        X = rng.standard_normal((N, T, L))
        F = rng.standard_normal((K, T))
        Gamma = rng.uniform(-0.5, 0.5, (L, K))
        Y = np.einsum("itl,lk,kt->it", X, Gamma, F)
        return Y, X, F, K

    def test_solve_gamma_matches_loop_oracle(self):
        Y, X, F, K = self._data()
        assert np.allclose(solve_gamma(Y, X, F, K), _naive_gamma(Y, X, F, K), atol=1e-10)

    def test_solve_factors_matches_loop_oracle(self):
        Y, X, F, K = self._data(1)
        rng = np.random.default_rng(2)
        gamma = rng.standard_normal((X.shape[2], K))
        assert np.allclose(solve_factors(Y, X, gamma), _naive_factors(Y, X, gamma), atol=1e-10)

    def test_counterfactual_matches_explicit(self):
        Y, X, F, K = self._data(3)
        rng = np.random.default_rng(4)
        gamma = rng.standard_normal((X.shape[2], K))
        yhat = counterfactual(X, gamma, F)
        expect = np.array([[X[i, t] @ gamma @ F[:, t] for t in range(X.shape[1])]
                           for i in range(X.shape[0])])
        assert np.allclose(yhat, expect, atol=1e-10)

    def test_als_recovers_noiseless_fit(self):
        Y, X, F, K = self._data(5)
        F_hat, gamma_hat, n_iter, converged = als_estimate(Y, X, K, max_iter=200, tol=1e-10)
        assert converged and n_iter <= 200
        # The fit (not the rotation) is recovered: residual ~ 0.
        assert np.max(np.abs(counterfactual(X, gamma_hat, F_hat) - Y)) < 1e-6

    def test_normalize_leaves_counterfactual_invariant(self):
        Y, X, F, K = self._data(6)
        F_hat, gamma_hat, *_ = als_estimate(Y, X, K, max_iter=200, tol=1e-12)
        gn, Fn = normalize(gamma_hat, F_hat)
        assert np.allclose(gn.T @ gn, np.eye(K), atol=1e-8)           # Gamma'Gamma = I
        assert np.allclose(counterfactual(X, gn, Fn),
                           counterfactual(X, gamma_hat, F_hat), atol=1e-8)


# ======================================================================
# Layer 2: setup / ingestion
# ======================================================================
class TestSetup:
    def test_cube_shape_and_order(self, panel):
        df, covs, _ = panel
        inp = prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs, 2)
        assert inp.control_covariates.shape == (20, 48, len(covs))
        assert inp.treated_covariates.shape == (48, len(covs))
        assert inp.T0 == 40 and inp.T == 48 and inp.L == len(covs)
        # cube row for a donor equals that unit's raw covariate series
        d0 = inp.donor_names[0]
        raw = (df[df["unit"] == d0].sort_values("time")["x0"].to_numpy())
        assert np.allclose(inp.control_covariates[0, :, 0], raw)

    def test_missing_covariate_raises(self, panel):
        df, covs, _ = panel
        with pytest.raises(MlsynthDataError, match="not found"):
            prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs + ["nope"], 2)

    def test_underidentified_pre_period_raises(self):
        # T0 < L*K : treated mapping not identified.
        df, covs, _ = _ipca_panel(T0=5, T1=4, L=4, K=2)
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs, 2)

    def test_multiple_cohorts_raises(self, panel):
        df, covs, _ = panel
        # promote a second unit to a (differently-timed) treated cohort
        df = df.copy()
        df.loc[(df["unit"] == 1) & (df["time"] >= 44), "D"] = 1
        with pytest.raises(MlsynthDataError, match="single treated unit"):
            prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs, 2)

    def test_missing_outcome_raises(self, panel):
        df, covs, _ = panel
        df = df.copy()
        df.loc[df.index[0], "y"] = np.nan
        with pytest.raises(MlsynthDataError, match="missing"):
            prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs, 2)


# ======================================================================
# Layer 3: config validation
# ======================================================================
class TestConfig:
    def test_empty_covariates_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises((MlsynthConfigError, ValueError)):
            CSCIPCAConfig(**_cfg(df, []))

    def test_blank_covariate_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError, match="non-empty"):
            CSCIPCAConfig(**_cfg(df, ["x0", "  "]))

    def test_duplicate_covariate_rejected(self, panel):
        df, _, _ = panel
        with pytest.raises(MlsynthConfigError, match="duplicate"):
            CSCIPCAConfig(**_cfg(df, ["x0", "x0"]))

    def test_outcome_as_covariate_rejected(self, panel):
        df, covs, _ = panel
        with pytest.raises(MlsynthConfigError, match="outcome"):
            CSCIPCAConfig(**_cfg(df, covs + ["y"]))

    def test_bad_n_factors_rejected(self, panel):
        df, covs, _ = panel
        with pytest.raises((MlsynthConfigError, ValueError)):
            CSCIPCAConfig(**_cfg(df, covs, n_factors=0))

    def test_fewer_covariates_than_factors_rejected(self, panel):
        df, covs, _ = panel        # 4 covariates
        with pytest.raises(MlsynthConfigError, match="identify the loadings"):
            CSCIPCAConfig(**_cfg(df, covs[:2], n_factors=3))

    def test_extra_field_forbidden(self, panel):
        df, covs, _ = panel
        with pytest.raises((MlsynthConfigError, ValueError)):
            CSCIPCAConfig(**_cfg(df, covs, bogus=1))


# ======================================================================
# Layer 4: estimator behavior
# ======================================================================
class TestEstimator:
    def test_recovers_known_att(self, panel):
        df, covs, tau = panel
        res = CSCIPCA(_cfg(df, covs)).fit()
        assert isinstance(res, CSCIPCAResults)
        assert abs(res.effects.att - tau) < 0.25
        assert res.fit_diagnostics.rmse_pre < 0.3
        assert res.metadata["converged"]

    def test_alpha_monotone_bias(self):
        """Paper headline: bias shrinks as the observed-covariate share rises."""
        biases = {}
        for a in (1 / 3, 1.0):
            errs = []
            for s in range(6):
                df, covs, tau = _ipca_panel(L=6, K=2, T0=40, alpha_obs=a, seed=100 + s)
                att = CSCIPCA(_cfg(df, covs)).fit().effects.att
                errs.append(att - tau)
            biases[a] = abs(float(np.mean(errs)))
        assert biases[1.0] < biases[1 / 3]        # more covariates -> less bias
        assert biases[1.0] < 0.2                    # near-unbiased when all observed

    def test_conformal_band_covers_truth_excludes_zero(self):
        df, covs, tau = _ipca_panel(N_co=15, T0=30, T1=6, L=3, noise=0.1, seed=7)
        res = CSCIPCA(_cfg(df, covs, alpha=0.1, inference=True, n_nulls=61)).fit()
        inf = res.inference_detail
        assert inf.att_lower <= tau <= inf.att_upper       # band covers truth
        assert not (inf.att_lower <= 0.0 <= inf.att_upper)  # excludes no-effect
        assert inf.ci_lower_t.shape == res.design.tau.shape
        assert np.all(inf.ci_lower_t <= inf.tau + 1e-9)
        assert np.all(inf.tau <= inf.ci_upper_t + 1e-9)

    def test_inference_toggle_off_is_empty(self, panel):
        df, covs, _ = panel
        res = CSCIPCA(_cfg(df, covs, inference=False)).fit()
        assert res.inference_detail.ci_lower_t.size == 0
        assert res.inference.ci_lower is None

    def test_dict_and_config_equivalent(self, panel):
        df, covs, _ = panel
        a = CSCIPCA(_cfg(df, covs)).fit().effects.att
        b = CSCIPCA(CSCIPCAConfig(**_cfg(df, covs))).fit().effects.att
        assert a == pytest.approx(b)

    def test_invalid_config_dict_raises(self, panel):
        df, covs, _ = panel
        with pytest.raises(MlsynthConfigError):
            CSCIPCA(_cfg(df, covs, n_factors=-1))

    def test_short_pre_period_raises_in_setup(self):
        # L >= K but T0 < L*K (8 covariates x 4 factors = 32 > T0 = 20):
        # the treated mapping is under-identified -> setup error.
        df, covs, _ = _ipca_panel(N_co=20, T0=20, T1=5, L=8, K=4)
        with pytest.raises(MlsynthDataError, match="pre-treatment periods"):
            CSCIPCA(_cfg(df, covs, n_factors=4)).fit()

    def test_n_factors_over_control_rank_raises_in_pipeline(self):
        # L >= K and T0 >= L*K, but K > min(N_control, T): unestimable count.
        df, covs, _ = _ipca_panel(N_co=3, T0=26, T1=3, L=5, K=5)
        with pytest.raises(MlsynthEstimationError, match="exceeds"):
            CSCIPCA(_cfg(df, covs, n_factors=5)).fit()

    def test_missing_covariate_value_raises(self, panel):
        df, covs, _ = panel
        df = df.copy()
        df.loc[df.index[3], "x0"] = np.nan
        with pytest.raises(MlsynthDataError, match="missing covariate"):
            CSCIPCA(_cfg(df, covs)).fit()

    def test_plotting_failure_translated(self, panel, monkeypatch):
        df, covs, _ = panel
        import mlsynth.estimators.cscipca as mod

        def _boom(_results):
            raise ValueError("backend down")

        monkeypatch.setattr(mod, "plot_cscipca", _boom)
        with pytest.raises(Exception, match="plotting failed"):
            CSCIPCA(_cfg(df, covs, display_graphs=True)).fit()

    def test_unexpected_error_translated(self, panel, monkeypatch):
        df, covs, _ = panel
        import mlsynth.estimators.cscipca as mod

        def _boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mod, "fit_cscipca", _boom)
        with pytest.raises(MlsynthEstimationError, match="estimation failed"):
            CSCIPCA(_cfg(df, covs)).fit()


# ======================================================================
# Layer 5: standardized contract + plotting smoke
# ======================================================================
class TestContract:
    def test_flat_accessors_resolve(self, panel):
        df, covs, _ = panel
        res = CSCIPCA(_cfg(df, covs)).fit()
        assert np.isfinite(res.att)
        cf = res.time_series.counterfactual_outcome
        obs = res.time_series.observed_outcome
        assert cf.shape == obs.shape == (48,)
        # gap = observed - counterfactual on the post-period equals tau
        gap_post = (obs - cf)[40:]
        assert np.allclose(gap_post, res.design.tau, atol=1e-8)
        assert res.weights.donor_weights == {}         # factor model: no weights
        assert res.n_factors == res.design.n_factors == 2

    def test_input_properties(self, panel):
        df, covs, _ = panel
        inp = prepare_cscipca_inputs(df, "y", "D", "unit", "time", covs, 2)
        assert inp.N_co == 20
        assert inp.n_post == 8
        assert inp.L == len(covs)

    def test_plotting_smoke(self, panel):
        import matplotlib
        matplotlib.use("Agg")
        df, covs, _ = panel
        res = CSCIPCA(_cfg(df, covs, display_graphs=True, inference=True, n_nulls=25,
                           alpha=0.1)).fit()
        assert res is not None


# ======================================================================
# Layer 6: Path-A empirical reproduction (Wang 2024 Brexit -> UK FDI)
# ======================================================================
_BASEDATA = pathlib.Path(__file__).resolve().parents[2] / "basedata"
_FDI = _BASEDATA / "fdi_oecd_brexit.csv"
_FDI_COVS = ["log_gdp", "log_gdp_percap", "import_to_gdp", "export_to_gdp",
             "inflation_gdp_deflator", "gross_capital_forma_gdp", "unemployment",
             "employment_15", "log_population"]


@pytest.mark.skipif(not _FDI.exists(), reason="FDI panel not vendored")
class TestBrexitReproduction:
    def test_reported_att_path(self):
        """CSC-IPCA matches Wang (2024): 2017 -7.8, 2018 -12.9, 2019 -18.3."""
        df = pd.read_csv(_FDI)
        res = CSCIPCA({
            "df": df, "outcome": "fdi", "treat": "treated", "unitid": "country",
            "time": "year", "covariates": _FDI_COVS, "n_factors": 2,
            "inference": False,
        }).fit()
        years = res.time_series.time_periods
        gap = res.time_series.estimated_gap
        reported = {2017: -7.8, 2018: -12.9, 2019: -18.3}
        for y, target in reported.items():
            assert abs(float(gap[years == y][0]) - target) < 0.3
        assert res.metadata["converged"]
        assert df["country"].nunique() == 30
