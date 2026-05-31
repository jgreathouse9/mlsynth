"""Tests for the Spatial Synthetic Difference-in-Differences (SpSyDiD) estimator.

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): spatial-matrix utilities, SDID weight QPs.
* Layer 2 (data utilities): prepare_spsydid_inputs partition logic.
* Layer 3 (estimator integration): SpSyDiD.fit on synthetic spatial DGPs.
* Layer 4 (public API contracts): top-level import, frozen dataclasses,
  W-shape validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SpSyDiD
from mlsynth.config_models import SpSyDiDConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.spsydid_helpers import (
    SpSyDiDInputs,
    SpSyDiDResults,
    contiguity_weights,
    inverse_distance_weights,
    knn_weights,
    prepare_spsydid_inputs,
    row_standardize,
    validate_spatial_matrix,
)


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------

def _grid_panel(
    grid_x: int = 8,
    grid_y: int = 8,
    T_pre: int = 16,
    T_post: int = 8,
    tau_true: float = 2.0,
    tau_s_true: float = 1.0,
    treated_units=(0, 7, 24, 39, 56, 63),
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(grid_x), np.arange(grid_y))
    coords = np.column_stack([xs.flatten(), ys.flatten()])
    N = coords.shape[0]
    T = T_pre + T_post
    W = knn_weights(coords, k=4, row_standardized=True)

    unit_fe = rng.standard_normal(N) * 0.5
    time_fe = np.linspace(0.0, 1.0, T)
    Y0 = (
        unit_fe[:, None]
        + time_fe[None, :]
        + rng.standard_normal((N, T)) * noise
    )
    D = np.zeros((N, T), dtype=float)
    for u in treated_units:
        D[u, T_pre:] = 1.0
    Y = Y0 + tau_true * D + tau_s_true * (W @ D)

    rows = [
        {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows), W, tau_true, tau_s_true


@pytest.fixture
def grid_panel():
    return _grid_panel()


# ----------------------------------------------------------------------
# Layer 1: spatial matrix utilities
# ----------------------------------------------------------------------

class TestSpatial:
    def test_validate_rejects_wrong_shape(self):
        W = np.zeros((4, 4))
        with pytest.raises(MlsynthDataError):
            validate_spatial_matrix(W, n_units=5)

    def test_validate_rejects_negative(self):
        W = -np.ones((3, 3))
        np.fill_diagonal(W, 0.0)
        with pytest.raises(MlsynthDataError):
            validate_spatial_matrix(W, n_units=3)

    def test_validate_rejects_nonzero_diagonal(self):
        W = np.eye(3)
        with pytest.raises(MlsynthDataError):
            validate_spatial_matrix(W, n_units=3)

    def test_row_standardize(self):
        W = np.array([[0.0, 1.0, 1.0], [2.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        S = row_standardize(W)
        np.testing.assert_allclose(S.sum(axis=1), [1.0, 1.0, 0.0])

    def test_knn_weights_row_sums_to_one(self):
        coords = np.random.default_rng(0).standard_normal((10, 2))
        W = knn_weights(coords, k=3)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        # No self-loops.
        np.testing.assert_array_equal(np.diag(W), np.zeros(10))

    def test_inverse_distance_cutoff_removes_far_pairs(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        W = inverse_distance_weights(coords, cutoff=2.0)
        # Unit 2 is >> 2 away from units 0 and 1 -- so unit 2's row is zero
        # and units 0 / 1 have no edge to unit 2.
        assert W[0, 2] == 0.0 and W[1, 2] == 0.0
        assert W[2, :].sum() == 0.0

    def test_contiguity_weights_from_adjacency(self):
        adj = {0: [1], 1: [0, 2], 2: [1]}
        W = contiguity_weights(adj, unit_order=[0, 1, 2])
        np.testing.assert_allclose(W[0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(W[1], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(W[2], [0.0, 1.0, 0.0])


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs_partition(self, grid_panel):
        df, W, _, _ = grid_panel
        inp = prepare_spsydid_inputs(
            df=df, outcome="y", treat="D", unitid="unit", time="time",
            spatial_matrix=W,
        )
        assert isinstance(inp, SpSyDiDInputs)
        # 6 directly treated corners + their k=4 neighbours = spillover set.
        # Pure controls = the rest of the 64-unit grid.
        assert inp.N_direct == 6
        assert inp.N_spillover > 0
        assert inp.N_pure > 0
        assert inp.N_direct + inp.N_spillover + inp.N_pure == 64
        # T0 detected correctly.
        assert inp.T0 == 16
        assert inp.T == 24

    def test_missing_pure_controls_rejected(self):
        # Construct a degenerate panel where every donor has a treated
        # neighbour (small grid + every unit has full connectivity).
        rng = np.random.default_rng(0)
        N, T = 6, 8
        T_pre = 5
        # Fully-connected W (excluding diagonal).
        W = np.ones((N, N)) - np.eye(N)
        W = row_standardize(W)
        D = np.zeros((N, T))
        D[0, T_pre:] = 1.0  # only one treated unit, but everyone is a neighbour
        Y = rng.standard_normal((N, T))
        rows = [
            {"unit": i, "time": t, "y": float(Y[i, t]), "D": float(D[i, t])}
            for i in range(N) for t in range(T)
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(MlsynthDataError, match="No pure controls"):
            prepare_spsydid_inputs(
                df=df, outcome="y", treat="D", unitid="unit", time="time",
                spatial_matrix=W,
            )


# ----------------------------------------------------------------------
# Layer 3: estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_recovers_direct_and_spillover(self, grid_panel):
        df, W, tau_true, tau_s_true = grid_panel
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "spatial_matrix": W,
            "display_graphs": False,
        }).fit()
        assert isinstance(res, SpSyDiDResults)
        # ATT should recover the planted direct effect within tolerance.
        assert abs(res.att - tau_true) < 0.2
        # Spillover coefficient should be in the right direction.
        assert abs(res.aite - tau_s_true) < 0.25

    def test_no_spillover_when_W_is_zero(self, grid_panel):
        df, _, tau_true, _ = grid_panel
        N = 64
        W_zero = np.zeros((N, N))
        # No spillover means everyone non-treated is a pure control.
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "spatial_matrix": W_zero,
            "display_graphs": False,
        }).fit()
        # With W=0, (WD)=0 so tau_s is identified at zero loading; the
        # regression should still recover tau.
        assert abs(res.att - tau_true) < 0.2
        assert res.inputs.N_spillover == 0
        assert res.inputs.N_pure == 64 - 6

    def test_donor_weights_sum_correctly(self, grid_panel):
        df, W, _, _ = grid_panel
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "spatial_matrix": W,
            "display_graphs": False,
        }).fit()
        # Pure-control weights sum to ~1 (SDID simplex constraint).
        pure_idx = res.inputs.pure_control_indices
        pure_total = sum(
            res.unit_weights[res.inputs.unit_names[i]]
            for i in pure_idx
        )
        assert abs(pure_total - 1.0) < 1e-3
        # Directly-treated units have uniform 1/N_tr weight.
        for i in res.inputs.direct_indices:
            assert abs(
                res.unit_weights[res.inputs.unit_names[i]]
                - 1.0 / res.inputs.N_direct
            ) < 1e-9


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import SpSyDiD as _SS
        assert _SS is SpSyDiD

    def test_results_dataclass_frozen(self, grid_panel):
        df, W, _, _ = grid_panel
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "spatial_matrix": W,
            "display_graphs": False,
        }).fit()
        with pytest.raises(Exception):
            res.att = 0.0
        with pytest.raises(Exception):
            res.inputs.T0 = 99

    def test_tau_alias_matches_att(self, grid_panel):
        df, W, _, _ = grid_panel
        res = SpSyDiD({
            "df": df, "outcome": "y", "treat": "D",
            "unitid": "unit", "time": "time",
            "spatial_matrix": W,
            "display_graphs": False,
        }).fit()
        assert res.tau == res.att
        assert res.tau_s == res.aite

    def test_wrong_W_shape_rejected(self, grid_panel):
        df, _, _, _ = grid_panel
        bad_W = np.zeros((5, 5))
        with pytest.raises(MlsynthDataError):
            SpSyDiD({
                "df": df, "outcome": "y", "treat": "D",
                "unitid": "unit", "time": "time",
                "spatial_matrix": bad_W,
            "display_graphs": False,
        }).fit()


def test_weights_results_exposed(grid_panel):
    """SpSyDiD exposes pure-control SDID weights + time weights via WeightsResults."""
    from mlsynth.config_models import WeightsResults
    df, W, _, _ = grid_panel
    res = SpSyDiD({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "spatial_matrix": W,
                   "display_graphs": False}).fit()
    assert isinstance(res.weights, WeightsResults)
    assert "time_weights" in res.weights.summary_stats


# ----------------------------------------------------------------------
# Path-B replication: pin Serenini & Masek (2024) headline Monte Carlo
# numbers as a permanent regression check. Drives through the public
# ``SpSyDiD(config).fit()`` API.
# ----------------------------------------------------------------------
class TestPathBReplication:
    """Pin the headline findings of the authors' simulation studies."""

    @pytest.fixture(scope="class")
    def state_level_result(self):
        try:
            import libpysal                              # noqa: F401
        except ImportError:
            pytest.skip("libpysal not installed; state-level MC needs it.")
        base = Path(__file__).resolve().parents[2] / "basedata"
        if not (base / "state_unemployment.csv").exists():
            pytest.skip("state_unemployment.csv not present.")
        if not (base / "US_no_islands_matrix.gal").exists():
            pytest.skip("US_no_islands_matrix.gal not present.")
        from examples.spsydid.replicate_state_level_mc import run_state_level_mc
        return run_state_level_mc(reps=40, rho=0.8, treated_fips=5, seed=0)

    def test_state_level_mean_att_bias_near_zero(self, state_level_result):
        # Authors' reference (their algorithm, same DGP): mean ATT bias = +0.0187.
        # mlsynth (public API, same DGP) should agree within MC noise.
        assert state_level_result["n_reps"] == 40
        assert state_level_result["att_bias_mean"] == pytest.approx(0.019, abs=0.01)
        # SD is comparable to the authors' 0.32 +/- a bit.
        assert state_level_result["att_bias_sd"] == pytest.approx(0.33, abs=0.05)

    def test_state_level_rho_bias_bounded(self, state_level_result):
        # The rho bias is noisier (heavy-tailed), but should stay sub-unity in mean.
        assert abs(state_level_result["rho_bias_mean"]) < 0.3

    @pytest.fixture(scope="class")
    def county_level_result(self):
        base = Path(__file__).resolve().parents[2] / "basedata"
        for f in ("spsydid_bls_county_subset.csv", "spsydid_county_matrices.pkl"):
            if not (base / f).exists():
                pytest.skip(f"{f} not present.")
        from examples.spsydid.replicate_county_level_mc import run_county_level_mc
        # 50 reps is plenty to see the headline (bias mean ~0); 200 in docs.
        return run_county_level_mc(reps=50, rho=0.5, seed=123)

    def test_county_level_all_states_unbiased(self, county_level_result):
        results, _ = county_level_result
        # Headline: across every state, |ATT bias mean| < 0.1 against an
        # ATT magnitude of ~1.5 pp. The notebook uses 1000 reps; we use 50
        # so tolerances are loose but the qualitative finding holds.
        assert set(results.keys()) >= {"WY", "OR", "PA", "AL"}
        for state, r in results.items():
            assert abs(r["att_bias_mean"]) < 0.1, (
                f"{state}: ATT bias mean = {r['att_bias_mean']:+.4f} "
                f"(magnitude expected near zero)"
            )

    def test_county_level_aite_bias_small(self, county_level_result):
        results, _ = county_level_result
        for state, r in results.items():
            assert abs(r["aite_bias_mean"]) < 0.15, (
                f"{state}: AITE bias mean = {r['aite_bias_mean']:+.4f}"
            )
