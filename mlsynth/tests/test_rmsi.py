"""Tests for the RMSI estimator (Agarwal, Choi & Yuan 2026).

Layered per agents/agents_tests.md:

* Layer 1 (numerical core): Algorithm 1 recovers a planted four-component
  matrix to the noise floor and beats no-side-info SVT; Algorithm 3 imputes a
  block and recovers a planted ATT.
* Layer 2 (data utilities): side-information ingestion + block-design guard.
* Layer 3 (estimator integration): ATT recovery on the side-info DGP; the
  Path-A (Prop 99) and Path-B (synthetic) replications.
* Layer 4 (public API contracts): import, frozen results, config validation.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import pytest

from mlsynth import RMSI
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.rmsi_helpers import (
    RMSIInputs,
    RMSIResults,
    algorithm1,
    algorithm3,
    prepare_rmsi_inputs,
    simulate_rmsi_panel,
)


# ----------------------------------------------------------------------
# Layer 1: numerical core
# ----------------------------------------------------------------------

class TestCore:
    def _dgp(self, N=60, T=50, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, (N, 2)); Z = rng.uniform(-1, 1, (T, 2))
        G1 = np.column_stack([np.sin(X[:, 0]), X[:, 1] ** 2])
        Q1 = np.column_stack([Z[:, 0], np.cos(Z[:, 1])])
        W = rng.normal(size=(N, 2)); V = rng.normal(size=(T, 2))
        M = G1 @ Q1.T + 0.7 * (W @ V.T)
        return M, X, Z, rng

    def test_algorithm1_recovers_and_beats_no_side_info(self):
        M, X, Z, rng = self._dgp()
        Y = M + rng.normal(scale=0.3, size=M.shape)
        Mhat, comps = algorithm1(Y, X, Z, J=3)
        rel = np.linalg.norm(Mhat - M) / np.linalg.norm(M)
        base, _ = algorithm1(Y, np.zeros((M.shape[0], 0)),
                             np.zeros((M.shape[1], 0)), J=3)
        rel_base = np.linalg.norm(base - M) / np.linalg.norm(M)
        assert set(comps) == {"M1", "M2", "M3", "M4"}
        assert rel < rel_base                      # side info helps
        assert rel < 0.30                          # near the noise floor

    def test_algorithm3_imputes_block_and_recovers_att(self):
        M, X, Z, rng = self._dgp(N=50, T=40)
        N, T = M.shape; N0, T0, att = 40, 28, 3.0
        Y = M + rng.normal(scale=0.3, size=(N, T))
        Y[N0:, T0:] += att                         # treated block
        Mhat, k = algorithm3(Y, X, Z, control_idx=np.arange(N0), T0=T0, J=3)
        att_hat = np.mean(Y[N0:, T0:] - Mhat[N0:, T0:])
        assert k >= 1
        assert abs(att_hat - att) < 0.5            # ATT recovered


# ----------------------------------------------------------------------
# Layer 2: ingestion
# ----------------------------------------------------------------------

class TestIngestion:
    def test_side_info_shapes(self):
        df = simulate_rmsi_panel(n_units=30, n_treated=6, T0=18, n_post=8,
                                 d_unit=2, d_time=2, seed=0)
        inp = prepare_rmsi_inputs(df, "Y", "treated", "unit", "time",
                                  unit_covariates=["x0", "x1"],
                                  time_covariates=["z0", "z1"])
        assert inp.N == 30 and inp.T0 == 18
        assert inp.X.shape == (30, 2) and inp.Z.shape == (inp.T, 2)
        assert inp.treated_idx.size == 6

    def test_staggered_rejected(self):
        df = simulate_rmsi_panel(n_units=12, n_treated=4, T0=20, n_post=6, seed=2)
        # delay one treated unit's adoption -> staggered
        u = df[df["treated"] == 1]["unit"].iloc[0]
        df.loc[(df["unit"] == u) & (df["time"].isin([20, 21])), "treated"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_rmsi_inputs(df, "Y", "treated", "unit", "time")

    def test_covariates_optional(self):
        df = simulate_rmsi_panel(n_units=20, n_treated=4, T0=15, n_post=6, seed=3)
        inp = prepare_rmsi_inputs(df, "Y", "treated", "unit", "time")
        assert inp.X.shape[1] == 0 and inp.Z.shape[1] == 0   # no covariates


# ----------------------------------------------------------------------
# Layer 3: estimator integration + replications
# ----------------------------------------------------------------------

class TestIntegration:
    def test_recovers_planted_att(self):
        df = simulate_rmsi_panel(n_units=40, n_treated=8, T0=20, n_post=11,
                                 d_unit=2, d_time=2, att=5.0, seed=0)
        res = RMSI({"df": df, "outcome": "Y", "treat": "treated",
                    "unitid": "unit", "time": "time",
                    "unit_covariates": ["x0", "x1"],
                    "time_covariates": ["z0", "z1"],
                    "display_graphs": False}).fit()
        assert isinstance(res, RMSIResults)
        assert abs(res.att - 5.0) < 1.0
        assert res.counterfactual.shape == (40, 31)
        assert res.rank >= 1

    def test_prop99_path_a(self):
        # Path A: California Proposition 99 effect from basedata.
        base = pathlib.Path(__file__).resolve().parents[2] / "basedata"
        p99 = base / "P99data.csv"
        if not p99.exists():
            pytest.skip("P99data not present")
        from mlsynth.utils.rmsi_helpers.replication import replicate_prop99
        res = replicate_prop99(str(p99), rank=3, verbose=False)
        assert -28.0 < res.att < -12.0            # ADH ~ -19 to -20

    def test_synthetic_path_b(self):
        # Path B: the paper's synthetic MNAR Monte Carlo runs and returns AMSE.
        from mlsynth.utils.rmsi_helpers.replication import (
            run_rmsi_simulation, RMSISimConfig,
        )
        out = run_rmsi_simulation(
            RMSISimConfig(N=60, T=60, N0=30, T0=30, J=5, n_reps=3),
            seed=0, verbose=False)
        assert np.isfinite(out["rmsi"]) and np.isfinite(out["no_side_info"])


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestAPI:
    def test_results_frozen(self):
        df = simulate_rmsi_panel(n_units=20, n_treated=4, T0=15, n_post=6, seed=5)
        res = RMSI({"df": df, "outcome": "Y", "treat": "treated",
                    "unitid": "unit", "time": "time",
                    "unit_covariates": ["x0", "x1"], "display_graphs": False}).fit()
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.att = 0.0

    def test_bad_config_raises(self):
        df = simulate_rmsi_panel(n_units=12)
        with pytest.raises(MlsynthConfigError):
            RMSI({"df": df, "treat": "treated", "unitid": "unit", "time": "time"})

    def test_inputs_immutable(self):
        df = simulate_rmsi_panel(n_units=12, n_treated=3, T0=15, n_post=6, seed=6)
        inp = prepare_rmsi_inputs(df, "Y", "treated", "unit", "time")
        assert isinstance(inp, RMSIInputs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            inp.T0 = 0
