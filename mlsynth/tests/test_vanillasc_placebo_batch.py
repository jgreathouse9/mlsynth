"""VanillaSC's in-space placebo, solved as one leave-one-out family.

Test-first (per ``agents/agents_tests.md``).

The outcome-only placebo refits every donor against the others, which is one
family over the donor matrix and not ``J`` unrelated problems. The p-value it
produces is a *rank* statistic over those fits, so the bar is not that the
batched path be optimal -- it is that it reproduce the ranks the one-at-a-time
path produced, exactly. These tests hold it to that, and check that the fast path
is skipped wherever a refit is not simply that one simplex QP.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import mlsynth.utils.bilevel.minnorm as minnorm
from mlsynth import VanillaSC


def _panel(seed=0, n_units=12, T=20, t0=14, effect=-4.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        level = 20.0 + 3.0 * u
        for t in range(T):
            y = level + 0.8 * t + rng.normal(scale=0.6)
            treated = int(u == 0 and t >= t0)
            if treated:
                y += effect
            rows.append({"unit": u, "time": t, "y": y, "treat": treated})
    return pd.DataFrame(rows)


def _fit(df, **kw):
    cfg = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
               display_graphs=False, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def _without_batching(df, **kw):
    """The same fit with the family solver disabled, so the loop runs."""
    real = minnorm.solve_simplex_loo_exact

    def boom(*a, **k):
        raise RuntimeError("batched path disabled for this test")

    try:
        minnorm.solve_simplex_loo_exact = boom
        return _fit(df, **kw)
    finally:
        minnorm.solve_simplex_loo_exact = real


# --------------------------------------------------------------------------- #
# 1. The batched family reproduces the loop, on everything the p-value uses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_placebo_inference_is_identical_to_the_loop(seed):
    df = _panel(seed=seed)
    got, want = _fit(df), _without_batching(df)
    assert got.inference.p_value == want.inference.p_value
    assert got.inference.details["rank"] == want.inference.details["rank"]
    assert got.inference.details["n_placebos"] == want.inference.details["n_placebos"]
    assert np.isclose(got.inference.details["treated_rmspe_ratio"],
                      want.inference.details["treated_rmspe_ratio"], rtol=0, atol=1e-12)
    assert np.isclose(got.effects.att, want.effects.att, rtol=0, atol=1e-12)


def test_more_donors_than_pre_periods_still_matches():
    """The geometry a design-shape guard would have refused, and the one the
    ordinary synthetic control has."""
    df = _panel(seed=7, n_units=30, T=12, t0=8)
    got, want = _fit(df), _without_batching(df)
    assert got.inference.p_value == want.inference.p_value
    assert got.inference.details["rank"] == want.inference.details["rank"]


# --------------------------------------------------------------------------- #
# 2. The fast path is skipped where a refit is not one plain simplex QP
# --------------------------------------------------------------------------- #
def test_ridge_augmented_fits_keep_the_loop():
    """Augmented SCM puts a ridge layer over the base, so a placebo refit is not
    the base solve alone. Disabling the family solver must change nothing."""
    df = _panel(seed=2)
    got, want = _fit(df, augment="ridge"), _without_batching(df, augment="ridge")
    assert got.inference.p_value == want.inference.p_value
    assert np.isclose(got.effects.att, want.effects.att, rtol=0, atol=1e-12)


def test_covariate_fits_keep_the_loop():
    df = _panel(seed=3)
    df["x"] = df.groupby("unit")["y"].transform("mean") + 0.1
    got = _fit(df, covariates=["x"], backend="mscmt")
    want = _without_batching(df, covariates=["x"], backend="mscmt")
    assert got.inference.p_value == want.inference.p_value


def test_penalized_backend_keeps_the_loop():
    df = _panel(seed=4)
    got = _fit(df, backend="penalized")
    want = _without_batching(df, backend="penalized")
    assert got.inference.p_value == want.inference.p_value


# --------------------------------------------------------------------------- #
# 3. The family solver is actually being used on the default path
# --------------------------------------------------------------------------- #
def test_the_default_outcome_only_path_uses_the_family_solver():
    """A fast path nothing reaches is the failure mode this guards."""
    calls = []
    real = minnorm.solve_simplex_loo_exact

    def spy(M, **kw):
        calls.append(M.shape)
        return real(M, **kw)

    try:
        minnorm.solve_simplex_loo_exact = spy
        _fit(_panel(seed=5))
    finally:
        minnorm.solve_simplex_loo_exact = real
    assert len(calls) == 1, "the placebo family should be solved in one call"


def test_a_two_donor_panel_still_works():
    df = _panel(seed=6, n_units=3, T=16, t0=11)
    got, want = _fit(df), _without_batching(df)
    assert got.inference.p_value == want.inference.p_value
