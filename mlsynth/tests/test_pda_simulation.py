"""Coverage + correctness tests for the Shi & Huang (2023) PDA Table-1 DGP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.pda_helpers.simulation import (
    _factors,
    _shock,
    simulate_pda_panel,
)


class TestSimulatePdaPanel:
    def test_default_iid_shapes_and_labels(self):
        s = simulate_pda_panel(N=20, T1=15, seed=0)
        assert s.T1 == 15 and s.T2 == 15
        assert s.Y_treated.shape == (30,)
        assert s.Y_controls.shape == (20, 30)
        assert s.relevant_donors == ["c000", "c001", "c002", "c003"]
        assert s.shock == "D1" and s.is_null is True
        # long frame: (N + 1 treated) * T rows; one treated unit
        assert isinstance(s.df, pd.DataFrame)
        assert len(s.df) == 21 * 30
        assert set(s.df["unit"]) == {"treated", *[f"c{i:03d}" for i in range(20)]}
        # treatment is absorbing on the treated unit only
        tr = s.df[s.df["unit"] == "treated"].sort_values("time")
        assert tr["treat"].tolist() == [0] * 15 + [1] * 15
        assert (s.df[s.df["unit"] != "treated"]["treat"] == 0).all()

    def test_seed_is_deterministic(self):
        a = simulate_pda_panel(N=10, T1=12, seed=7).Y_treated
        b = simulate_pda_panel(N=10, T1=12, seed=7).Y_treated
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(3)
        s = simulate_pda_panel(N=8, T1=10, rng=rng, seed=999)
        assert s.Y_controls.shape == (8, 20)

    def test_dynamic_factors_differ_from_iid(self):
        kw = dict(N=12, T1=40, seed=1)
        iid = simulate_pda_panel(dynamic_factors=False, **kw).Y_treated
        dyn = simulate_pda_panel(dynamic_factors=True, **kw).Y_treated
        assert not np.allclose(iid, dyn)

    def test_effect_override_sets_custom_and_shifts_post(self):
        s0 = simulate_pda_panel(N=10, T1=10, effect=0.0, seed=2)
        assert s0.shock == "custom" and s0.is_null is True
        s5 = simulate_pda_panel(N=10, T1=10, effect=5.0, seed=2)
        assert s5.shock == "custom" and s5.is_null is False
        # the only difference is a +5 shift on the treated post-period
        np.testing.assert_allclose(s5.Y_treated[:10], s0.Y_treated[:10])
        np.testing.assert_allclose(s5.Y_treated[10:], s0.Y_treated[10:] + 5.0)

    @pytest.mark.parametrize("shock,is_null", [
        ("D1", True), ("D2", True), ("D3", True),
        ("D4", False), ("D5", False), ("D6", False), ("D7", False),
    ])
    def test_all_shocks_run_and_flag_null(self, shock, is_null):
        s = simulate_pda_panel(N=10, T1=12, shock=shock, seed=4)
        assert s.shock == shock
        assert s.is_null is is_null

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            simulate_pda_panel(N=4, T1=10)           # N must exceed 4 relevant
        with pytest.raises(ValueError):
            simulate_pda_panel(N=10, T1=0)            # T1 positive
        with pytest.raises(ValueError):
            simulate_pda_panel(N=10, T1=10, T2=0)     # T2 positive


class TestHelpers:
    def test_factors_iid_shape(self):
        f = _factors(50, np.random.default_rng(0), dynamic=False)
        assert f.shape == (50, 4)

    def test_factors_dynamic_shape_after_burn(self):
        f = _factors(50, np.random.default_rng(0), dynamic=True)
        assert f.shape == (50, 4)

    def test_shock_null_processes_are_mean_small(self):
        rng = np.random.default_rng(0)
        assert np.allclose(_shock("D1", 30, rng), 0.0)
        # D4/D5 carry positive mean shifts
        assert _shock("D5", 2000, np.random.default_rng(1)).mean() > 0.5

    def test_shock_unknown_raises(self):
        with pytest.raises(ValueError):
            _shock("D9", 10, np.random.default_rng(0))
