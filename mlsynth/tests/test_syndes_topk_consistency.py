"""A SYNDES pool's rank-1 design is the single best design: ``top_K`` only enlarges the menu.

Reported: on the GeoLift market panel (40 markets, 90 days) with ``K=2``, ``top_K=1``
selected {cleveland, new york} and ``top_K=5`` selected {milwaukee, portland}. Every
computation was right. Pricing all 780 two-market designs independently reproduced
both: the in-sample top five, whose first entry is the ``top_K=1`` design, and the top
five on the leading 63 of 90 days, ranked correctly by held-out error. The cause was a
default. Commit 2bbfe352 made any ``top_K >= 2`` set ``holdout_frac=0.3`` when neither
``selection`` nor ``holdout_frac`` was given, so asking for a menu silently changed
the selection rule, the slice the designs were fitted on, and the default penalty
``lam`` (estimated on that slice). The ``top_K``, ``selection`` and ``holdout_frac``
descriptions and the docs page still said the default was in-sample.

Holdout and IC selection stay available on request (``holdout_frac`` or
``selection``); without one, the pool is ranked by the in-sample objective and its
first entry is the design ``top_K=1`` returns.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth import SYNDES
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.config import SYNDESConfig


def _panel(seed=0, N=8, T=30):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2)).cumsum(axis=0)
    L = rng.normal(size=(N, 2))
    Y = 100.0 + F @ L.T + rng.normal(scale=0.5, size=(T, N))
    return pd.DataFrame([{"unit": f"u{j}", "time": t, "y": float(Y[t, j])}
                         for j in range(N) for t in range(T)])


def _cfg(df, **kw):
    return dict(df=df, outcome="y", unitid="unit", time="time", K=2, **kw)


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SYNDES(_cfg(df, **kw)).fit()


def _labels(design):
    return sorted(np.asarray(design.selected_unit_labels).tolist())


# --------------------------------------------------------------------------- smoke
def test_a_pool_selects_the_single_best_design():
    df = _panel()
    one, five = _fit(df, top_K=1), _fit(df, top_K=5)
    assert _labels(five.design) == _labels(one.design)
    assert five.design.objective_value == pytest.approx(one.design.objective_value, rel=1e-12)


# --------------------------------------------------------------------------- unit
def test_the_default_selection_of_a_pool_is_in_sample():
    cfg = SYNDESConfig(**_cfg(_panel(), top_K=5))
    assert cfg.holdout_frac is None
    assert cfg.selection in (None, "in_sample")


def test_the_pool_is_ranked_by_the_in_sample_objective():
    res = _fit(_panel(1), top_K=5)
    obj = [e["objective"] for e in res.pool]
    assert obj == sorted(obj)
    assert all(e.get("oos_rmse") is None for e in res.pool)


def test_holdout_is_still_available_on_request():
    res = _fit(_panel(2), top_K=4, holdout_frac=0.3)
    assert all(np.isfinite(e["oos_rmse"]) for e in res.pool)
    oos = [e["oos_rmse"] for e in res.pool]
    assert oos == sorted(oos)


# --------------------------------------------------------------------------- edge
def test_a_pool_larger_than_the_design_space_still_starts_at_the_best():
    df = _panel(3, N=4)                       # 6 two-unit sets, some mirrored
    one, many = _fit(df, top_K=1), _fit(df, top_K=10)
    assert _labels(many.design) == _labels(one.design)
    assert 1 <= len(many.pool) <= 6


# --------------------------------------------------------------------------- failure
def test_holdout_without_a_fraction_is_refused():
    with pytest.raises(MlsynthConfigError, match="holdout_frac"):
        SYNDES(_cfg(_panel(), top_K=4, selection="holdout"))


def test_holdout_with_a_single_design_is_refused():
    with pytest.raises(MlsynthConfigError, match="top_K >= 2"):
        SYNDES(_cfg(_panel(), top_K=1, holdout_frac=0.3))


# --------------------------------------------------------------------------- property
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), N=st.integers(4, 9), top_K=st.integers(2, 6))
@example(seed=0, N=8, top_K=5)
def test_property_top_K_never_changes_the_selected_design(seed, N, top_K):
    df = _panel(seed, N=N)
    one, pool = _fit(df, top_K=1), _fit(df, top_K=top_K)
    assert _labels(pool.design) == _labels(one.design)
    assert _labels(pool.design) == sorted(pool.pool[0]["markets"])
    obj = [e["objective"] for e in pool.pool]
    assert obj == sorted(obj)
