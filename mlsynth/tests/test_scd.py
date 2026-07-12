"""Tests for SCD (Synthetic Control with Differencing, Rincon & Song 2026).

Layered per ``agents/agents_tests.md``:
  Layer 1 (mechanisms): weighted group means; lambda differencing schemes;
    simplex weight solve; RC pointwise variance; in_C confidence-set membership.
  Layer 2 (setup): grouped-microdata + survey-weight ingestion; treated-unit
    identification; balance / pre-post guards.
  Layer 3 (integration): SCD.fit on the Arizona LAWA CPS extract -- reproduces
    the R reference value-for-value (weights, effect path, ATT, RC SE, confidence
    set), results contract, plotting smoke.
  Layer 4 (config): every validator raises the translated error.

The pinned numbers are the base-R reference captured under
``benchmarks/reference/scd_cps/`` (SCD reproduced from scratch on public CPS
microdata; the upstream package is GPL and is cross-validated, not vendored).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlsynth import SCD  # noqa: E402
from mlsynth.config_models import BaseEstimatorResults, SCDConfig  # noqa: E402
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError  # noqa: E402
from mlsynth.utils.scd_helpers.weights import (  # noqa: E402
    solve_scd_weights,
    lambda_vector,
    in_C,
    confidence_set,
)
from mlsynth.utils.scd_helpers.setup import prepare_scd_inputs  # noqa: E402

_CPS = Path(__file__).resolve().parents[2] / "basedata" / "cps_lawa_arizona.parquet"

# --- pinned reference values (benchmarks/reference/scd_cps/reference.json) ---
_DID_ATT = 0.11595232
_DID_THETA_POST1 = -0.34194581
_DID_SE_POST1 = 0.34410021
_UNIFORM_ATT = 0.11731723
_SC_ATT = 0.11578535
_DID_TOP_WEIGHTS = {
    "Ohio": 0.22354985, "Missouri": 0.16933478, "Connecticut": 0.14574073,
    "Arkansas": 0.13733699, "Wyoming": 0.11938771, "Colorado": 0.08144521,
    "West Virginia": 0.08096848,
}
_HATV_TRACE = 9235.07463421


@pytest.fixture(scope="module")
def cps() -> pd.DataFrame:
    df = pd.read_parquet(_CPS)
    df["treat"] = ((df["state_name"] == "Arizona") & (df["D"] == 1)).astype(int)
    return df


def _cfg(df, **over):
    base = dict(df=df, outcome="wklyearn", treat="treat", unitid="state_name",
                time="period", weight_col="weight", display_graphs=False,
                compute_inference=False)
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Layer 1: mechanisms
# ---------------------------------------------------------------------------
class TestMechanisms:
    def test_lambda_vectors(self):
        did = lambda_vector("did", 5)
        assert np.allclose(did, [0, 0, 0, 0, 1])
        unif = lambda_vector("uniform", 5)
        assert np.allclose(unif, np.full(5, 0.2))
        sc = lambda_vector("sc", 5)
        assert np.allclose(sc, np.zeros(5))

    def test_simplex_solve_is_on_simplex(self):
        rng = np.random.default_rng(0)
        G = rng.standard_normal((20, 6))
        g = rng.standard_normal(20)
        w = solve_scd_weights(G, g)
        assert w.shape == (6,)
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= -1e-9).all()

    def test_single_donor_weight_is_one(self):
        G = np.random.default_rng(1).standard_normal((10, 1))
        g = np.random.default_rng(2).standard_normal(10)
        w = solve_scd_weights(G, g)
        assert np.allclose(w, [1.0])


# ---------------------------------------------------------------------------
# Layer 2: setup / ingestion
# ---------------------------------------------------------------------------
class TestSetup:
    def test_ingest_shapes(self, cps):
        inp = prepare_scd_inputs(cps, "wklyearn", "treat", "state_name", "period", "weight")
        assert inp.treated_name == "Arizona"
        assert inp.K == 46
        assert inp.T0 == 54
        assert inp.Tstar == 55
        assert inp.group_means.shape == (47, inp.Ttot)

    def test_missing_weight_column_raises(self, cps):
        with pytest.raises(MlsynthDataError):
            prepare_scd_inputs(cps, "wklyearn", "treat", "state_name", "period", "nope")

    def test_unweighted_when_no_weight_col(self, cps):
        inp = prepare_scd_inputs(cps, "wklyearn", "treat", "state_name", "period", None)
        # unweighted group mean equals simple average within a cell
        g = cps[(cps.state_name == "Arizona") & (cps.period == 1)]
        assert abs(inp.group_means[0, 0] - g["wklyearn"].mean()) < 1e-9


# ---------------------------------------------------------------------------
# Layer 3: integration -- reproduce the R reference
# ---------------------------------------------------------------------------
class TestReproduction:
    def test_did_weights_match_reference(self, cps):
        res = SCD(_cfg(cps, differencing="did")).fit()
        dw = res.weights.donor_weights
        for state, w in _DID_TOP_WEIGHTS.items():
            assert abs(dw[state] - w) < 1e-3, f"{state}: {dw[state]} vs {w}"
        assert abs(sum(dw.values()) - 1.0) < 1e-6

    def test_did_att_and_path(self, cps):
        res = SCD(_cfg(cps, differencing="did")).fit()
        assert abs(res.att - _DID_ATT) < 1e-3
        # first post-period effect is the theta at Tstar (index T0 = 54)
        post_first = res.time_series.estimated_gap[54]
        assert abs(post_first - _DID_THETA_POST1) < 1e-3

    def test_uniform_and_sc_att(self, cps):
        assert abs(SCD(_cfg(cps, differencing="uniform")).fit().att - _UNIFORM_ATT) < 1e-3
        assert abs(SCD(_cfg(cps, differencing="sc")).fit().att - _SC_ATT) < 1e-3

    def test_gap_equals_observed_minus_counterfactual(self, cps):
        res = SCD(_cfg(cps, differencing="did")).fit()
        ts = res.time_series
        assert np.allclose(ts.estimated_gap, ts.observed_outcome - ts.counterfactual_outcome, atol=1e-8)

    def test_results_contract(self, cps):
        res = SCD(_cfg(cps, differencing="did")).fit()
        assert isinstance(res, BaseEstimatorResults)
        assert res.donor_weights is not None
        assert res.counterfactual is not None
        assert res.method_details.method_name == "SCD"

    def test_inference_rc_se_and_bands(self, cps):
        res = SCD(_cfg(cps, differencing="did", compute_inference=True)).fit()
        det = res.inference.details
        # per-period SE at first post period matches reference
        se = np.asarray(det["se"])
        assert abs(se[54] - _DID_SE_POST1) < 1e-3
        assert abs(det["hatV_trace"] - _HATV_TRACE) < 1e-1
        # bands bracket the point path
        lo = np.asarray(det["lower"]); hi = np.asarray(det["upper"])
        gap = res.time_series.estimated_gap
        assert (lo <= gap + 1e-9).all() and (hi >= gap - 1e-9).all()
        assert res.att_ci is not None and res.att_ci[0] <= res.att <= res.att_ci[1]

    def test_confidence_set_membership_matches_reference(self, cps):
        inp = prepare_scd_inputs(cps, "wklyearn", "treat", "state_name", "period", "weight")
        from mlsynth.utils.scd_helpers.inference import build_inference_operators
        ops = build_inference_operators(inp, differencing="did")
        K = inp.K
        u = np.full(K, 1.0 / K)
        v1 = np.zeros(K); v1[0] = 1.0
        ed = np.zeros(K); ed[0] = 0.5; ed[4] = 0.5
        assert in_C(ops.hat_w, ops, kappa=0.05, tol=1e-6)          # optimum -> in
        assert in_C(u, ops, kappa=0.05, tol=1e-6)                  # interior -> in
        assert not in_C(v1, ops, kappa=0.05, tol=1e-6)             # boundary -> out
        assert not in_C(ed, ops, kappa=0.05, tol=1e-6)             # boundary -> out

    def test_confidence_set_sweep_accepts_optimum(self, cps):
        inp = prepare_scd_inputs(cps, "wklyearn", "treat", "state_name", "period", "weight")
        from mlsynth.utils.scd_helpers.inference import build_inference_operators
        ops = build_inference_operators(inp, differencing="did")
        cs = confidence_set(ops, kappa=0.05, tol=1e-6, n_grid=300, radius=0.05, random_state=0)
        assert cs.shape[1] == inp.K
        assert cs.shape[0] >= 1

    def test_plot_smoke(self, cps):
        res = SCD(_cfg(cps, differencing="did", compute_inference=True, display_graphs=True)).fit()
        ax = res.plot()
        assert ax is not None


# ---------------------------------------------------------------------------
# Layer 4: config validation
# ---------------------------------------------------------------------------
class TestConfig:
    def test_bad_differencing_raises(self, cps):
        with pytest.raises(MlsynthConfigError):
            SCD(_cfg(cps, differencing="bogus"))

    def test_kappa_ge_alpha_raises(self, cps):
        with pytest.raises(MlsynthConfigError):
            SCD(_cfg(cps, compute_inference=True, alpha=0.05, kappa=0.05))

    def test_alpha_out_of_range_raises(self, cps):
        with pytest.raises(MlsynthConfigError):
            SCD(_cfg(cps, alpha=1.5))

    def test_missing_weight_column_raises_on_fit(self, cps):
        with pytest.raises(MlsynthDataError):
            SCD(_cfg(cps, weight_col="does_not_exist")).fit()

    def test_no_treated_unit_raises(self, cps):
        df = cps.copy(); df["treat"] = 0
        with pytest.raises(MlsynthDataError):
            SCD(_cfg(df, differencing="did")).fit()


# ---------------------------------------------------------------------------
# Layer 2b: setup failure paths (reported, not swallowed)
# ---------------------------------------------------------------------------
class TestFailures:
    def _prep(self, df, **over):
        kw = dict(outcome="wklyearn", treat="treat", unitid="state_name",
                  time="period", weight_col="weight")
        kw.update(over)
        return prepare_scd_inputs(df, **kw)

    def test_missing_outcome_column(self, cps):
        with pytest.raises(MlsynthDataError):
            self._prep(cps.drop(columns=["wklyearn"]))

    def test_nan_outcome(self, cps):
        df = cps.copy(); df.loc[df.index[0], "wklyearn"] = np.nan
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_multiple_treated_units(self, cps):
        df = cps.copy()
        df.loc[(df.state_name == "Ohio") & (df.period >= 55), "treat"] = 1
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_treated_at_first_period(self, cps):
        df = cps.copy()
        df.loc[df.state_name == "Arizona", "treat"] = 1
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_negative_weights(self, cps):
        df = cps.copy(); df.loc[df.index[0], "weight"] = -1.0
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_empty_cell(self, cps):
        # drop every Ohio row at period 3 -> an empty (group, period) cell
        df = cps[~((cps.state_name == "Ohio") & (cps.period == 3))].copy()
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_single_unit_no_donors(self, cps):
        df = cps[cps.state_name == "Arizona"].copy()
        with pytest.raises(MlsynthDataError):
            self._prep(df)

    def test_estimation_error_is_wrapped(self, cps, monkeypatch):
        import mlsynth.estimators.scd as scd_mod

        def boom(*a, **k):
            raise ValueError("synthetic failure")

        monkeypatch.setattr(scd_mod, "run_scd", boom)
        from mlsynth.exceptions import MlsynthEstimationError
        with pytest.raises(MlsynthEstimationError):
            SCD(_cfg(cps, differencing="did")).fit()

    def test_plot_save(self, cps, tmp_path):
        res = SCD(_cfg(cps, differencing="did", compute_inference=True)).fit()
        out = tmp_path / "scd.png"
        from mlsynth.utils.scd_helpers.plotter import plot_scd
        plot_scd(res, save=str(out))
        assert out.exists()
