"""Coverage tests for mlsynth.utils.seq_sdid_helpers.simulate (pure DGP).

The DGP backs the Path-B benchmark ``seq_sdid_mc``; these tests exercise its
structure (determinism, staggered/absorbing treatment, donor-balance cap)
without invoking the estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.seq_sdid_helpers.simulate import (
    _NEVER,
    _ar2_noise,
    CalibratedDesign,
    calibrate_staggered_ife,
    simulate_replication,
)


def test_ar2_noise_shape_burn_and_determinism():
    a = _ar2_noise(20, 5, (0.5, -0.1), 1.0, np.random.default_rng(0))
    b = _ar2_noise(20, 5, (0.5, -0.1), 1.0, np.random.default_rng(0))
    assert a.shape == (20, 5)
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)        # seeded => identical


def test_calibrate_returns_design_with_consistent_counts():
    d = calibrate_staggered_ife(seed=2024)
    assert isinstance(d, CalibratedDesign)
    assert d.base.shape == (d.T, d.S)
    assert d.n_treated + d.n_never == d.S
    assert d.n_treated > 0 and d.n_never > 0
    # a_max caps below the latest adopting cohort so donors remain.
    treated_dates = sorted({int(a) for a in d.adopt if a != _NEVER})
    assert d.a_max <= treated_dates[-1]
    assert d.n_cohorts == len(treated_dates)


def test_calibrate_is_deterministic():
    d1 = calibrate_staggered_ife(seed=7)
    d2 = calibrate_staggered_ife(seed=7)
    np.testing.assert_array_equal(d1.base, d2.base)
    np.testing.assert_array_equal(d1.adopt, d2.adopt)
    assert d1.a_max == d2.a_max


def test_calibrate_few_cohorts_falls_back_to_first():
    # donor_tail >= number of cohorts exercises the ``else treated_dates[0]`` arm.
    d = calibrate_staggered_ife(seed=1, n_cohorts_span=3, donor_tail=50)
    treated_dates = sorted({int(a) for a in d.adopt if a != _NEVER})
    assert d.a_max == treated_dates[0]


def test_simulate_replication_shape_and_absorbing_treatment():
    d = calibrate_staggered_ife(seed=2024)
    df = simulate_replication(d, np.random.default_rng(0), tau=1.0, n_copies=3)
    assert set(df.columns) == {"unit", "year", "y", "treat"}
    assert df["unit"].nunique() == d.S * 3
    assert df["year"].nunique() == d.T
    # treatment is staggered and absorbing within every unit.
    for _, g in df.sort_values("year").groupby("unit"):
        tr = g["treat"].to_numpy()
        assert tr.tolist() == sorted(tr.tolist())   # non-decreasing 0..1
    # at least one never-treated unit (all-zero treat) exists.
    per_unit_max = df.groupby("unit")["treat"].max()
    assert (per_unit_max == 0).any() and (per_unit_max == 1).any()


def test_simulate_replication_effect_shifts_treated_cells():
    d = calibrate_staggered_ife(seed=2024)
    rng_seed = 123
    base = simulate_replication(d, np.random.default_rng(rng_seed), tau=0.0)
    bumped = simulate_replication(d, np.random.default_rng(rng_seed), tau=5.0)
    # Same noise draw (same seed) => treated cells differ by exactly tau, others equal.
    treated = bumped["treat"] == 1
    np.testing.assert_allclose(
        (bumped.loc[treated, "y"] - base.loc[treated, "y"]).to_numpy(), 5.0)
    np.testing.assert_allclose(
        bumped.loc[~treated, "y"].to_numpy(), base.loc[~treated, "y"].to_numpy())
