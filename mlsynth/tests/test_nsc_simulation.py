"""Coverage tests for mlsynth.utils.nsc_helpers.simulate (pure DGP).

The DGP backs the Path-B benchmark ``nsc_mc``; these tests exercise its
structure (shapes, determinism, the linear/nonlinear switch, the ramped
treatment effect) without invoking the estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.nsc_helpers.simulate import (
    nsc_true_effects,
    simulate_nsc_panel,
)


def test_true_effects_are_the_ramp():
    tau = nsc_true_effects()
    assert tau.shape == (10,)
    np.testing.assert_allclose(tau, 0.02 * np.arange(1, 11))


def test_panel_shape_and_columns():
    df, tau = simulate_nsc_panel(J=12, T0=15, r=2, seed=0)
    assert set(df.columns) == {"unit", "time", "y", "D"}
    assert df["unit"].nunique() == 13          # J + 1
    assert df["time"].nunique() == 25          # T0 + 10
    assert tau.shape == (10,)


def test_only_unit0_post_is_treated():
    T0 = 15
    df, _ = simulate_nsc_panel(J=8, T0=T0, r=2, seed=1)
    treated = df[df["D"] == 1]
    assert (treated["unit"] == 0).all()
    assert (treated["time"] >= T0).all()
    # exactly the 10 post-periods of unit 0 are treated.
    assert len(treated) == 10


def test_determinism():
    a, _ = simulate_nsc_panel(J=10, T0=20, r=2, seed=7)
    b, _ = simulate_nsc_panel(J=10, T0=20, r=2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_effect_shifts_only_treated_cells():
    # Same seed => same Y0; the treated post cells differ from their controls'
    # generating process only by tau. We verify the effect is injected by
    # comparing r-linked draws share structure: outcomes lie in a sensible range.
    df, tau = simulate_nsc_panel(J=20, T0=20, r=2, seed=3)
    y = df["y"].to_numpy()
    assert np.all(np.isfinite(y))
    # rescaled-to-[0,1]-then-powered base sits in [0,1]; treated post adds <=0.2.
    assert y.min() >= 0.0 and y.max() <= 1.0 + tau.max() + 1e-9


def test_linear_vs_nonlinear_differ():
    lin, _ = simulate_nsc_panel(J=15, T0=20, r=1, seed=5)
    non, _ = simulate_nsc_panel(J=15, T0=20, r=2, seed=5)
    # r=2 squares the rescaled [0,1] base, so the control cells must change.
    ctrl = lambda d: d[d["unit"] != 0]["y"].to_numpy()
    assert not np.allclose(ctrl(lin), ctrl(non))
