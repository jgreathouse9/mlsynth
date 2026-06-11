"""Coverage + correctness tests for the SAR spillover DGP simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.spillsynth_helpers.sar.simulation import (
    SARSimSample,
    rook_adjacency,
    simulate_sar_panel,
)


class TestRookAdjacency:
    def test_shape_and_symmetry(self):
        W = rook_adjacency(4, 4)
        assert W.shape == (16, 16)
        assert np.array_equal(W, W.T)
        assert np.all(np.diag(W) == 0)

    def test_corner_has_two_neighbours_center_has_four(self):
        W = rook_adjacency(3, 3)
        assert W[0].sum() == 2          # corner (0,0)
        assert W[4].sum() == 4          # centre (1,1)


class TestSimulateSarPanel:
    def test_shapes_and_fields(self):
        s = simulate_sar_panel(rho=0.6, seed=0)
        assert isinstance(s, SARSimSample)
        assert s.n_controls == 16
        # long panel: (N controls + 1 treated) * T rows
        assert isinstance(s.df, pd.DataFrame)
        assert len(s.df) == 17 * 30
        assert set(s.df["unit"]) == {"T", *[f"c{i}" for i in range(16)]}
        assert s.spatial_W.shape == (16, 16)
        assert s.spatial_w.shape == (16,)
        assert s.rho == 0.6

    def test_treatment_is_absorbing_on_treated_only(self):
        s = simulate_sar_panel(seed=1)
        tr = s.df[s.df["unit"] == "T"].sort_values("time")
        assert tr["d"].tolist() == [0] * 20 + [1] * 10
        assert (s.df[s.df["unit"] != "T"]["d"] == 0).all()

    def test_spatial_weights_row_normalised(self):
        s = simulate_sar_panel(seed=2)
        rowsums = s.spatial_W.to_numpy().sum(axis=1)
        np.testing.assert_allclose(rowsums, 1.0)
        np.testing.assert_allclose(s.spatial_w.to_numpy().sum(), 1.0)

    def test_rho_zero_differs_from_rho_positive(self):
        a = simulate_sar_panel(rho=0.0, seed=3).df["y"].to_numpy()
        b = simulate_sar_panel(rho=0.6, seed=3).df["y"].to_numpy()
        assert not np.allclose(a, b)

    def test_seed_is_deterministic(self):
        a = simulate_sar_panel(seed=7).df["y"].to_numpy()
        b = simulate_sar_panel(seed=7).df["y"].to_numpy()
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(5)
        s = simulate_sar_panel(rng=rng, seed=999)
        assert s.n_controls == 16

    def test_custom_grid_and_periods(self):
        s = simulate_sar_panel(n_rows=2, n_cols=3, T=20, T0=12, seed=0)
        assert s.n_controls == 6
        assert len(s.df) == 7 * 20
        tr = s.df[s.df["unit"] == "T"].sort_values("time")
        assert tr["d"].tolist() == [0] * 12 + [1] * 8

    def test_single_unit_grid(self):
        # exercises the N == 1 guard on alpha[1]
        s = simulate_sar_panel(n_rows=1, n_cols=1, T=10, T0=6, seed=0)
        assert s.n_controls == 1

    @pytest.mark.parametrize("kwargs", [
        {"T": 10, "T0": 0}, {"T": 10, "T0": 10}, {"n_rows": 0}, {"n_cols": 0},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate_sar_panel(seed=0, **kwargs)
