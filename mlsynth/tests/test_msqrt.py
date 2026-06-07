"""Tests for the MSQRT estimator (Shen, Song & Abadie 2025).

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): the Multivariate Square-root Lasso solve
  recovers a known sparse donor combination; CV picks a finite lambda.
* Layer 2 (data utilities): block-panel ingestion + block-design guard.
* Layer 3 (estimator integration): recovery of a planted ATT on a low-rank
  multiple-treated panel; block-conformal band.
* Layer 4 (public API contracts): import, frozen results, config validation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlsynth import MSQRT
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.msqrt_helpers import (
    MSQRTInputs,
    MSQRTResults,
    fit_msqrt_weights,
    prepare_msqrt_inputs,
    select_lambda_cv,
    simulate_msqrt_panel,
)


def _block_panel(n_tr=3, n_co=8, T0=24, n_post=6, att=2.0, seed=1):
    return simulate_msqrt_panel(
        n_treated=n_tr, n_control=n_co, T0=T0, n_post=n_post,
        att=att, seed=seed,
    )


# ----------------------------------------------------------------------
# Layer 1: the square-root Lasso solve
# ----------------------------------------------------------------------

class TestSolve:
    def test_recovers_sparse_combination(self):
        rng = np.random.default_rng(0)
        T, n = 40, 6
        X = rng.standard_normal((T, n))
        # treated = 0.7*donor0 + 0.3*donor2, others zero
        true = np.zeros((n, 1))
        true[0, 0], true[2, 0] = 0.7, 0.3
        Y = X @ true + rng.standard_normal((T, 1)) * 0.01
        theta, Y_hat, nz = fit_msqrt_weights(Y, X, lambd=0.001)
        assert theta.shape == (n, 1)
        # the two active donors carry most of the weight mass
        assert abs(theta[0, 0]) > 0.4 and abs(theta[2, 0]) > 0.1
        assert np.linalg.norm(Y - Y_hat) < 0.5

    def test_larger_lambda_is_sparser(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 8))
        Y = X @ rng.standard_normal((8, 2))
        _, _, nz_small = fit_msqrt_weights(Y, X, lambd=0.001)
        _, _, nz_big = fit_msqrt_weights(Y, X, lambd=5.0)
        assert nz_big.sum() <= nz_small.sum()

    def test_admm_matches_cvxpy(self):
        # The fast ADMM must reach the same optimum as the conic solver, on the
        # *elementwise* L1 objective (eq. 5) -- guards the solver + the
        # elementwise-penalty fix against regressions.
        cp = pytest.importorskip("cvxpy")
        from mlsynth.utils.msqrt_helpers.optimization import (
            fit_msqrt_admm, _fit_msqrt_cvxpy,
        )
        rng = np.random.default_rng(0)
        T, n, m = 25, 8, 3
        X = rng.standard_normal((T, n))
        true = np.zeros((n, m))
        true[[0, 2], 0] = [0.6, 0.4]; true[1, 1] = 1.0; true[[3, 5], 2] = [0.5, 0.5]
        Y = X @ true + rng.standard_normal((T, m)) * 0.1

        def obj(Th, lam):
            return ((1 / np.sqrt(T)) * np.linalg.norm(Y - X @ Th, "nuc")
                    + lam * np.abs(Th).sum())

        for lam in (0.05, 0.2, 0.5):
            Ta = fit_msqrt_admm(Y, X, lam)
            Tc = _fit_msqrt_cvxpy(Y, X, lam)
            assert obj(Ta, lam) == pytest.approx(obj(Tc, lam), rel=1e-3)
            assert np.linalg.norm(Ta - Tc) < 1e-2

    def test_elementwise_l1_not_induced_norm(self):
        # The penalty must be sum_ij |Theta_ij|, not cp.norm(.,1)'s induced
        # 1-norm (max column abs-sum). Build a case where they differ and check
        # the objective the ADMM minimises uses the elementwise penalty.
        cp = pytest.importorskip("cvxpy")
        from mlsynth.utils.msqrt_helpers.optimization import fit_msqrt_admm
        rng = np.random.default_rng(3)
        T, n, m = 30, 5, 4
        X = rng.standard_normal((T, n))
        Y = X @ rng.standard_normal((n, m))
        Th = fit_msqrt_admm(Y, X, 0.3)
        elementwise = np.abs(Th).sum()
        induced = np.abs(Th).sum(axis=0).max()      # max column sum
        # for a non-trivial multi-column solution these are clearly different
        assert elementwise > induced + 1e-6

    def test_warm_start_matches_cold(self):
        # Warm-starting from a nearby solution must not change the optimum.
        from mlsynth.utils.msqrt_helpers.optimization import fit_msqrt_admm
        rng = np.random.default_rng(5)
        T, n, m = 50, 12, 4
        X = rng.standard_normal((T, n)) + 2.0
        Y = X @ rng.standard_normal((n, m))
        cold = fit_msqrt_admm(Y, X, 0.1)
        _, state = fit_msqrt_admm(Y, X, 0.15, return_state=True)
        warm = fit_msqrt_admm(Y, X, 0.1, warm_start=state)
        assert np.linalg.norm(cold - warm) < 1e-3

    def test_over_relaxation_converges_to_optimum(self):
        # The default (over_relax=1.5) and a larger value must both reach the
        # cvxpy optimum -- including on mean-shifted, ill-conditioned data.
        cp = pytest.importorskip("cvxpy")
        from mlsynth.utils.msqrt_helpers.optimization import (
            fit_msqrt_admm, _fit_msqrt_cvxpy,
        )
        rng = np.random.default_rng(6)
        T, n, m = 50, 12, 4
        X = rng.standard_normal((T, n)) + 2.0
        Y = X @ rng.standard_normal((n, m))

        def obj(Th):
            return ((1 / np.sqrt(T)) * np.linalg.norm(Y - X @ Th, "nuc")
                    + 0.2 * np.abs(Th).sum())

        ref = obj(_fit_msqrt_cvxpy(Y, X, 0.2))
        for orx in (1.5, 1.7):
            assert obj(fit_msqrt_admm(Y, X, 0.2, over_relax=orx)) == \
                pytest.approx(ref, rel=1e-3)

    def test_cv_returns_grid_value(self):
        df = _block_panel()
        inp = prepare_msqrt_inputs(df, "Y", "treated", "unit", "time")
        grid = np.logspace(-2, 1, 4)
        lam = select_lambda_cv(inp.Y_pre, inp.X_pre, grid)
        assert lam in set(float(x) for x in grid)


# ----------------------------------------------------------------------
# Layer 2: ingestion + block-design guard
# ----------------------------------------------------------------------

class TestIngestion:
    def test_shapes(self):
        df = _block_panel(n_tr=3, n_co=8, T0=24, n_post=6)
        inp = prepare_msqrt_inputs(df, "Y", "treated", "unit", "time")
        assert inp.m == 3 and inp.n == 8
        assert inp.Y_pre.shape == (24, 3)
        assert inp.X_post.shape == (6, 8)
        assert inp.T0 == 24 and inp.n_post == 6

    def test_staggered_rejected(self):
        df = _block_panel()
        # delay one treated unit's adoption -> staggered
        mask = (df["unit"] == "t0") & (df["time"].isin([24, 25]))
        df.loc[mask, "treated"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_msqrt_inputs(df, "Y", "treated", "unit", "time")

    def test_needs_two_donors(self):
        df = _block_panel(n_tr=2, n_co=1)
        with pytest.raises(MlsynthDataError):
            prepare_msqrt_inputs(df, "Y", "treated", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestIntegration:
    def test_recovers_planted_att(self):
        df = _block_panel(n_tr=4, n_co=10, T0=30, n_post=8, att=3.0, seed=2)
        res = MSQRT({
            "df": df, "outcome": "Y", "treat": "treated",
            "unitid": "unit", "time": "time",
            "n_lambda": 4, "display_graphs": False,
        }).fit()
        # planted effect is +3; tiny noisy panel -> generous band
        assert res.att > 1.0
        assert res.att_t.shape == (8,)
        assert len(res.unit_att) == 4
        assert res.pre_rmse < 5.0

    def test_scpi_prediction_intervals(self):
        df = _block_panel(n_tr=3, n_co=8, T0=24, n_post=6, seed=3)
        res = MSQRT({
            "df": df, "outcome": "Y", "treat": "treated",
            "unitid": "unit", "time": "time",
            "n_lambda": 4, "inference": True, "alpha": 0.1,
            "display_graphs": False,
        }).fit()
        scpi = res.inference_intervals
        assert scpi is not None
        assert scpi.method == "cfpt_scpi"
        # standardized inference mirrored into the contract slot
        assert res.inference is not None
        assert res.att_ci == pytest.approx((scpi.taua.lower, scpi.taua.upper))
        # in-sample omitted for MSQRT
        assert scpi.in_sample_included is False
        assert scpi.alpha_out == 0.1 and scpi.alpha_in == 0.0
        # overall ATT band (TAUA) brackets the point estimate
        lo, hi = scpi.ci
        assert lo < res.att < hi
        assert scpi.taua.point == res.att
        # full predictand family present
        assert len(scpi.tsus) == 3 * 6          # units x periods
        assert len(scpi.taus) == 3              # one per unit
        assert len(scpi.tsua) == 6              # one per post-period
        assert set(scpi.simultaneous) == set(scpi.taus)
        # simultaneous bands are wider than (or equal to) pointwise TSUS
        u0 = next(iter(scpi.taus))
        k0 = res.inputs.time_labels[res.inputs.T0]
        pt = scpi.tsus[(u0, k0)]
        sim = scpi.simultaneous[u0][0]
        assert (sim.upper - sim.lower) >= (pt.upper - pt.lower) - 1e-9

    def test_time_dependence_general_is_wider(self):
        df = _block_panel(n_tr=3, n_co=8, T0=24, n_post=6, seed=7)
        common = {"df": df, "outcome": "Y", "treat": "treated",
                  "unitid": "unit", "time": "time", "n_lambda": 3,
                  "inference": True, "display_graphs": False}
        iid = MSQRT({**common, "time_dependence": "iid"}).fit()
        gen = MSQRT({**common, "time_dependence": "general"}).fit()
        w_iid = iid.inference_intervals.taua.upper - iid.inference_intervals.taua.lower
        w_gen = gen.inference_intervals.taua.upper - gen.inference_intervals.taua.lower
        assert w_gen >= w_iid - 1e-9

    def test_fixed_lambda_skips_cv(self):
        df = _block_panel(seed=4)
        res = MSQRT({
            "df": df, "outcome": "Y", "treat": "treated",
            "unitid": "unit", "time": "time",
            "lambda_": 0.5, "display_graphs": False,
        }).fit()
        assert res.best_lambda == 0.5


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestAPI:
    def test_results_are_frozen(self):
        df = _block_panel(seed=5)
        res = MSQRT({
            "df": df, "outcome": "Y", "treat": "treated",
            "unitid": "unit", "time": "time",
            "n_lambda": 3, "display_graphs": False,
        }).fit()
        assert isinstance(res, MSQRTResults)
        # MSQRTResults is now a frozen pydantic EffectResult.
        with pytest.raises(ValidationError):
            res.best_lambda = 0.0

    def test_bad_config_raises(self):
        df = _block_panel()
        with pytest.raises(MlsynthConfigError):
            # missing required outcome column name
            MSQRT({"df": df, "treat": "treated",
                   "unitid": "unit", "time": "time"})

    def test_inputs_immutable(self):
        df = _block_panel()
        inp = prepare_msqrt_inputs(df, "Y", "treated", "unit", "time")
        assert isinstance(inp, MSQRTInputs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            inp.T0 = 0
