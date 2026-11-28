"""Phase 4a: the exact path is PANGEO's default design; ``fast`` is deprecated.

The clustering heuristic (``fast_partition``) measured 6.4 to 26.8 times the optimum's cost on seasonal arms of 20 to
80 markets (11.3 times the best design found at 120) and 1.7 to 2.1 times on grouped ones, while the exact path proves
the optimum in seconds to minutes up to about 80 markets at Q = 2. So the default design is the exact path with small
node budgets (search / POPMUSIC / certify = 5M / 1M / 0.5M), ``fast`` is accepted for one release with a
DeprecationWarning and no longer changes the path, and the heuristic runs only where the exact path cannot score every
group (past ``columns.MAX_COLUMNS``), with ``PangeoHeuristicFallbackWarning``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, config as pconfig, fast_partition as fp_mod, pipeline
from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel
from mlsynth.utils.pangeo_helpers.config import PANGEOConfig
from mlsynth.utils.pangeo_helpers.structures import PangeoHeuristicFallbackWarning

BASE = dict(outcome="sales", arm="arm", unitid="unit", time="time")


def _df(n=8, T=24, seed=0):
    return make_seasonal_sales_panel(units_per_arm=n, arms=("A",), T=T, seed=seed)


def _fit(df, **kw):
    return PANGEO(dict(df=df, **BASE, **kw)).fit()


def _groups(res, arm="A"):
    return sorted((tuple(sorted(map(str, p.treatment + p.control))), round(float(p.gap_variance), 9))
                  for p in res.arm_designs[arm].pairs)


@pytest.fixture
def no_heuristic(monkeypatch):
    """The heuristic must not run on the exact path."""
    def boom(*a, **kw):
        raise AssertionError("fast_partition was called on the exact path")
    monkeypatch.setattr(pipeline, "fast_partition", boom)


# --- smoke ---------------------------------------------------------------------------------------------------------

def test_smoke_the_default_design_is_the_exact_path(no_heuristic):
    res = _fit(_df(), max_supergeo_size=2)
    assert res.arm_designs["A"].pairs
    assert "heuristic" not in res.metadata["solver"]
    assert res.metadata["solver_diagnostics"]["A"].get("path") != "fast"


# --- unit: the deprecated flag ---------------------------------------------------------------------------------------

def test_fast_defaults_to_none_and_warns_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = PANGEOConfig(df=_df(), **BASE)
    assert cfg.fast is None


@pytest.mark.parametrize("value", [True, False])
def test_passing_fast_warns_it_is_deprecated(value):
    with pytest.warns(DeprecationWarning, match="fast.*deprecated"):
        PANGEOConfig(df=_df(), **BASE, fast=value)


@pytest.mark.parametrize("value", [True, False])
def test_fast_no_longer_changes_the_design(value, no_heuristic):
    df = _df()
    base = _fit(df, max_supergeo_size=2)
    with pytest.warns(DeprecationWarning):
        other = _fit(df, max_supergeo_size=2, fast=value)
    assert _groups(other) == _groups(base)


def test_run_pangeo_fast_true_warns_and_runs_the_exact_path(no_heuristic):
    from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs
    inp = prepare_pangeo_inputs(df=_df(), **BASE)
    with pytest.warns(DeprecationWarning, match="fast.*deprecated"):
        res = pipeline.run_pangeo(inp, max_supergeo_size=2, fast=True, compute_power=False)
    assert "heuristic" not in res.metadata["solver"]


def test_supergeo_total_with_fast_true_runs_instead_of_raising():
    with pytest.warns(DeprecationWarning):
        res = _fit(_df(), max_supergeo_size=1, objective="supergeo_total", fast=True)
    assert res.arm_designs["A"].pairs


def test_fast_rejects_a_non_boolean():
    with pytest.raises(Exception, match="fast"):
        PANGEOConfig(df=_df(), **BASE, fast="yes")


# --- unit: the default budgets -------------------------------------------------------------------------------------

def test_the_default_budgets_are_the_small_ones():
    cfg = PANGEOConfig(df=_df(), **BASE)
    assert (cfg.search_node_limit, cfg.popmusic_node_limit, cfg.certify_node_limit) == (5_000_000, 1_000_000, 500_000)
    assert (pconfig.DESIGN_SEARCH_NODE_LIMIT, pconfig.DESIGN_POPMUSIC_NODE_LIMIT,
            pconfig.DESIGN_CERTIFY_NODE_LIMIT) == (5_000_000, 1_000_000, 500_000)


def test_the_estimator_passes_all_three_budgets_to_the_exact_path(monkeypatch):
    seen, real = [], columns.solve_exact

    def spy(*a, **kw):
        seen.append((kw.get("search_node_limit"), kw.get("popmusic_node_limit"), kw.get("certify_node_limit")))
        return real(*a, **kw)
    monkeypatch.setattr(pipeline, "solve_exact", spy)
    _fit(_df(), max_supergeo_size=2)
    _fit(_df(), max_supergeo_size=2, search_node_limit=777, popmusic_node_limit=55, certify_node_limit=33)
    assert seen[0] == (5_000_000, 1_000_000, 500_000)
    assert seen[-1] == (777, 55, 33)


@pytest.mark.parametrize("bad", [-1, 2.5, True, "10"])
def test_an_invalid_search_budget_is_refused_by_name(bad):
    with pytest.raises(Exception, match="search_node_limit"):
        PANGEOConfig(df=_df(), **BASE, search_node_limit=bad)


def test_the_q_sweep_passes_the_search_budget(monkeypatch):
    seen, real = [], columns.solve_exact

    def spy(*a, **kw):
        seen.append(kw.get("search_node_limit"))
        return real(*a, **kw)
    monkeypatch.setattr(pipeline, "solve_exact", spy)
    _fit(_df(n=8), search_node_limit=4321)            # max_supergeo_size None: the automatic Q sweep
    assert seen and set(seen) == {4321}


# --- edge: the fallback past MAX_COLUMNS ----------------------------------------------------------------------------

def test_past_max_columns_the_heuristic_runs_with_a_warning(monkeypatch):
    n, Q = 10, 2
    monkeypatch.setattr(columns, "MAX_COLUMNS", columns.count_columns(n, Q) - 1)
    with pytest.warns(PangeoHeuristicFallbackWarning, match="MAX_COLUMNS|groups"):
        res = _fit(_df(n=n), max_supergeo_size=Q)
    assert "heuristic" in res.metadata["solver"]
    assert res.metadata["solver_diagnostics"]["A"]["path"] == "fast"


def test_at_exactly_max_columns_the_exact_path_runs(monkeypatch, no_heuristic):
    n, Q = 10, 2
    monkeypatch.setattr(columns, "MAX_COLUMNS", columns.count_columns(n, Q))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PangeoHeuristicFallbackWarning)
        res = _fit(_df(n=n), max_supergeo_size=Q)
    assert "heuristic" not in res.metadata["solver"]


def test_pairs_are_never_routed_to_the_heuristic(monkeypatch, no_heuristic):
    """Pairs reach MAX_COLUMNS only past about 6,300 markets (C(n, 2) > 2e7); if they ever did, the pairs path's
    own size check raises, and the heuristic is not used in its place."""
    monkeypatch.setattr(columns, "MAX_COLUMNS", 1)
    with pytest.raises(Exception, match="candidate groups"):
        _fit(_df(n=8), max_supergeo_size=1)


# --- failure --------------------------------------------------------------------------------------------------------

def test_supergeo_total_still_refuses_groups_larger_than_pairs():
    with pytest.raises(Exception, match="pairs"):
        PANGEOConfig(df=_df(), **BASE, objective="supergeo_total", max_supergeo_size=2)


# --- property: the default design never costs more than the heuristic -----------------------------------------------

def _cost(Yw, names, groups):
    """A design's PANGEO cost on a window: the centred sum of squares of its halves' mean-path difference."""
    pos = {nm: i for i, nm in enumerate(names)}
    tot = 0.0
    for a, b in groups:
        d = Yw[[pos[u] for u in a]].mean(axis=0) - Yw[[pos[u] for u in b]].mean(axis=0)
        d = d - d.mean()
        tot += float(d @ d)
    return tot


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(4, 9), Q=st.integers(1, 2))
def test_property_the_exact_default_costs_at_most_the_heuristic(seed, n, Q):
    from hypothesis import assume
    assume(not (Q == 1 and n % 2))                  # an odd arm's pairs leave a market out; the heuristic does not
    rng = np.random.default_rng(seed)
    T = 12
    Y = rng.standard_normal((n, T)).cumsum(axis=1) + rng.normal(0, 3, (n, 1))
    names = [f"u{i}" for i in range(n)]
    df = pd.DataFrame([(names[i], t, float(Y[i, t]), "A") for i in range(n) for t in range(T)],
                      columns=["unit", "time", "sales", "arm"])
    res = _fit(df, max_supergeo_size=Q, compute_power=False)
    n_E = int(round(0.7 * T))                       # the default estimation window (frac_E = 0.7)
    Yw = Y[:, :n_E]
    exact = _cost(Yw, names, [(list(map(str, p.treatment)), list(map(str, p.control)))
                              for p in res.arm_designs["A"].pairs])
    heur = fp_mod.fast_partition(np.arange(n), Yw, Q)
    hcost = _cost(Yw, names, [([names[u] for u in g["side_a"]], [names[u] for u in g["side_b"]]) for g in heur])
    assert exact <= hcost * (1 + 1e-9) + 1e-9


# --- option (a): the automatic Q sweep stops at the largest Q the exact path scores ---------------------------------

def _inputs(sizes, T=20, seed=0):
    from mlsynth.utils.pangeo_helpers import prepare_pangeo_inputs
    rng = np.random.default_rng(seed)
    rows = []
    for a, m in zip("ABC", sizes):
        Y = rng.standard_normal((m, T)).cumsum(axis=1)
        rows += [(f"{a}{i}", t, float(Y[i, t]), a) for i in range(m) for t in range(T)]
    df = pd.DataFrame(rows, columns=["unit", "time", "sales", "arm"])
    return prepare_pangeo_inputs(df=df, **BASE)


def test_the_sweep_cap_is_the_largest_q_the_largest_arm_can_score(monkeypatch):
    inp = _inputs([10, 14])
    monkeypatch.setattr(columns, "MAX_COLUMNS", columns.count_columns(14, 2))
    assert pipeline._q_upper(inp) == 2                   # Q = 3 at 14 markets has more groups; Q = 2 fits exactly
    monkeypatch.setattr(columns, "MAX_COLUMNS", columns.count_columns(14, 2) - 1)
    assert pipeline._q_upper(inp) == 1


def test_the_sweep_designs_no_q_past_the_cap_and_never_falls_back(monkeypatch):
    n = 12
    monkeypatch.setattr(columns, "MAX_COLUMNS", columns.count_columns(n, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PangeoHeuristicFallbackWarning)
        res = _fit(_df(n=n))                             # max_supergeo_size None: the automatic sweep
    assert max(row["q"] for row in res.metadata["q_sweep"]) == 2
    assert not res.metadata["heuristic_arms"]


def test_the_sweep_keeps_q_1_when_even_pairs_exceed_the_limit(monkeypatch):
    monkeypatch.setattr(columns, "MAX_COLUMNS", 0)
    assert pipeline._q_upper(_inputs([8])) == 1


@settings(max_examples=60, deadline=None)
@given(sizes=st.lists(st.integers(2, 40), min_size=1, max_size=3), limit=st.integers(0, 3_000_000))
def test_property_the_sweep_cap_respects_every_bound(sizes, limit):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(columns, "MAX_COLUMNS", limit)
        q = pipeline._q_upper(_inputs(sizes, T=6))
    big, small = max(sizes), min(sizes)
    assert 1 <= q <= max(1, min((small + 1) // 2, pipeline._AUTO_Q_CAP))
    if q > 1:
        assert columns.count_columns(big, q) <= limit           # every arm can be scored at the cap
    if q < min((small + 1) // 2, pipeline._AUTO_Q_CAP):
        assert columns.count_columns(big, q + 1) > limit        # and the next Q could not be
