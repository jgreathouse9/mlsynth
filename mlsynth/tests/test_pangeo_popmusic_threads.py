"""POPMUSIC on threads (:func:`popmusic.improve` with ``n_jobs``).

A pass's clusters of two or more groups run in waves of ``popmusic.WAVE`` in cluster order. Every tree of a wave gets
the lesser of ``sub_limit`` and the budget left divided by the wave's trees, and runs on its own thread; each cluster's
group floor is set from the design at the start of its wave, and the covers are committed in cluster order, each only
when its rigorous upper bound is below the cluster's and the design keeps at least k groups. The waves, limits, floors
and commits do not depend on the number of threads, so neither does the design: ``n_jobs`` changes the time taken and
nothing else.
"""
from __future__ import annotations

import os
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover, popmusic, vtree


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _arm(seed, n, Q, min_pairs=1):
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    return Y, idx, pool, k, columns.root_lp(pool, k)


def _paths(Y):
    P = np.asarray(Y, dtype=float)
    return P - P.mean(axis=1, keepdims=True)


def _seed_design(n):
    return np.array([columns.column_index(g, n) for g in columns.seed_partition(n)], dtype=np.int64)


def _upper(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _units(pool, j):
    return [int(u) for u in pool.members[j] if u >= 0]


def _is_cover(pool, rows, n, k):
    return sorted(u for j in rows for u in _units(pool, j)) == list(range(n)) and len(rows) >= k


def _same(a, b):
    assert a["rows"].tolist() == b["rows"].tolist()
    assert a["upper"] == b["upper"] and a["nodes"] == b["nodes"] and a["trace"] == b["trace"]
    assert (a["passes"], a["subproblems"], a["improved"]) == (b["passes"], b["subproblems"], b["improved"])


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_improve_on_threads_returns_a_cover_no_worse_than_its_start():
    n, Q = 16, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    rows = _seed_design(n)
    out = popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=4, delta=2, n_jobs=3)
    assert _is_cover(pool, out["rows"], n, k) and out["upper"] <= _upper(pool, rows) and out["improved"] >= 1


# --- unit: independence from the thread count ----------------------------------------------------------------------

@pytest.mark.parametrize("seed,n,Q", [(3, 16, 2), (6, 18, 2), (2, 14, 3)])
def test_the_design_and_its_counts_do_not_depend_on_the_threads(seed, n, Q):
    Y, _, pool, k, root = _arm(seed, n, Q)
    rows = _seed_design(n)
    runs = [popmusic.improve(pool, root, k, Q, rows, _paths(Y), 10**6, alpha=4, delta=2, n_jobs=j)
            for j in (1, 2, 4)]
    _same(runs[0], runs[1])
    _same(runs[0], runs[2])


def test_threads_are_used_with_n_jobs_workers_and_none_means_every_core(monkeypatch):
    """More than one thread runs the trees on an executor of n_jobs workers; None takes every core; 1 runs inline."""
    n, Q = 16, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    seen, real = [], popmusic.ThreadPoolExecutor

    def ex(max_workers=None, **kw):
        seen.append(max_workers)
        return real(max_workers=max_workers, **kw)
    monkeypatch.setattr(popmusic, "ThreadPoolExecutor", ex)
    popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), 10**5, alpha=4, delta=2, n_jobs=3)
    assert seen and set(seen) == {3}
    seen.clear()
    popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), 10**5, alpha=4, delta=2, n_jobs=None)
    assert seen and set(seen) == {os.cpu_count() or 1}
    seen.clear()
    popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), 10**5, alpha=4, delta=2, n_jobs=1)
    assert seen == []


# --- unit: waves, limits, floors, commits --------------------------------------------------------------------------

def test_each_wave_splits_what_is_left_of_the_budget_evenly_up_to_the_cap(monkeypatch):
    """With one thread the trees run in cluster order; within each wave of m trees every limit is the lesser of
    sub_limit and the budget left at the wave's start divided by m, and the budget left drops by the wave's nodes."""
    n, Q = 20, 2
    Y, _, pool, k, root = _arm(6, n, Q)
    calls, real = [], vtree.search

    def tree(sub, r, keep, kk, best_up, node_limit, **kw):
        res = real(sub, r, keep, kk, best_up, node_limit, **kw)
        calls.append((int(node_limit), int(res.nodes)))
        return res
    monkeypatch.setattr(vtree, "search", tree)
    monkeypatch.setattr(popmusic, "WAVE", 3)
    budget, cap = 60, 25
    out = popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), budget, alpha=4, delta=2, sub_limit=cap,
                           n_jobs=1)
    left, i = budget, 0
    for _, subs, _, _ in out["trace"]:
        done = 0
        while done < subs:
            m = min(3, subs - done)
            want = min(cap, left // m)
            if want == 0:
                m, want = min(m, left), 1
            wave = calls[i:i + m]
            assert all(lim == want for lim, _ in wave), (wave, want, left, m)
            left -= sum(u for _, u in wave)
            i += m
            done += m
    assert i == len(calls) and out["nodes"] == sum(u for _, u in calls) <= budget


def test_the_floor_comes_from_the_design_at_the_start_of_the_wave(monkeypatch):
    """A cluster's floor is k less the groups outside it when its wave starts: with k forcing every group to stay a
    pair, each tree is asked for exactly its cluster's groups."""
    n, Q = 12, 2
    Y, _, pool, k, root = _arm(4, n, Q, min_pairs=6)
    ks, real = [], vtree.search
    monkeypatch.setattr(vtree, "search", lambda sub, r, keep, kk, *a, **kw: ks.append((len(sub.unit_indices), kk))
                        or real(sub, r, keep, kk, *a, **kw))
    out = popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), 10**6, alpha=4, delta=2, n_jobs=2)
    assert ks and all(kk == m // 2 for m, kk in ks)
    assert len(out["rows"]) == 6


def test_commit_takes_covers_in_order_and_refuses_one_that_breaks_the_floor():
    """The commit step: covers in cluster order, each taken when cheaper and when the design keeps k groups."""
    take = popmusic._commit
    # three clusters of 2 groups each, 6 groups in all, k = 5: the first merge to one group is taken (5 left), the
    # second would leave 4 and is refused, the third is not cheaper and is refused
    old = [[1, 2], [3, 4], [5, 6]]
    new = [[7], [8], [9]]
    cheaper = [True, True, False]
    out, taken = take(old, new, cheaper, total=6, k=5)
    assert out == [[7], [3, 4], [5, 6]] and taken == 1
    # with k = 4 both merges fit
    out, taken = take(old, new, cheaper, total=6, k=4)
    assert out == [[7], [8], [5, 6]] and taken == 2
    # a cover with more groups than the cluster always fits the floor
    out, taken = take([[1]], [[2, 3]], [True], total=3, k=3)
    assert out == [[2, 3]] and taken == 1


def test_threads_report_to_the_progress_total_in_full(monkeypatch):
    """The trees' nodes reach the progress array's total, so search_nodes still ends equal to it."""
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    monkeypatch.setattr(popmusic, "ALPHA", 4)
    monkeypatch.setattr(popmusic, "DELTA", 2)
    prog = cover.new_progress()
    _, d = columns.solve_exact(idx, Y, Q, progress=prog, n_jobs=3)
    assert d["popmusic_nodes"] > 0 and d["search_nodes"] == int(prog[cover.P_TOTAL])


def test_solve_exact_passes_n_jobs_to_popmusic(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    seen, real = [], popmusic.improve
    monkeypatch.setattr(popmusic, "improve", lambda *a, **kw: seen.append(kw.get("n_jobs")) or real(*a, **kw))
    columns.solve_exact(idx, Y, Q, n_jobs=2)
    columns.solve_exact(idx, Y, Q)
    assert seen == [2, None]


# --- failures -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "2"])
def test_an_invalid_n_jobs_is_refused_by_name(bad):
    n, Q = 10, 2
    Y, _, pool, k, root = _arm(3, n, Q)
    with pytest.raises(MlsynthConfigError, match="n_jobs"):
        popmusic.improve(pool, root, k, Q, _seed_design(n), _paths(Y), 100, alpha=4, delta=2, n_jobs=bad)


@pytest.mark.parametrize("bad", [0, -2, True])
def test_solve_exact_refuses_an_invalid_n_jobs(bad):
    n, Q = 12, 2
    Y, idx, *_ = _arm(3, n, Q)
    with pytest.raises(MlsynthConfigError, match="n_jobs"):
        columns.solve_exact(idx, Y, Q, n_jobs=bad)


def test_the_config_holds_n_jobs_and_the_estimator_passes_it(monkeypatch):
    from mlsynth import PANGEO
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, pipeline
    from mlsynth.utils.pangeo_helpers.config import PANGEOConfig
    df = make_seasonal_sales_panel(units_per_arm=8, arms=("A",), T=24, seed=0)
    base = dict(df=df, outcome="sales", arm="arm", unitid="unit", time="time")
    assert PANGEOConfig(**base).n_jobs is None
    with pytest.raises(Exception, match="n_jobs"):
        PANGEOConfig(**base, n_jobs=0)
    seen, real = [], columns.solve_exact
    monkeypatch.setattr(pipeline, "solve_exact", lambda *a, **kw: seen.append(kw.get("n_jobs")) or real(*a, **kw))
    PANGEO(dict(base, fast=False, max_supergeo_size=2, n_jobs=2)).fit()
    assert seen and set(seen) == {2}


# --- properties ---------------------------------------------------------------------------------------------------

@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 16), Q=st.integers(1, 3), budget=st.integers(0, 4000),
       alpha=st.integers(2, 8), delta=st.integers(1, 4), repeat=st.integers(1, 3), sub=st.integers(1, 600),
       pairs=st.integers(1, 8), wave=st.integers(1, 5), jobs=st.integers(2, 4))
def test_property_threads_change_nothing_and_the_floor_and_budget_hold(seed, n, Q, budget, alpha, delta, repeat, sub,
                                                                      pairs, wave, jobs):
    """Property: over arms, floors, settings and wave sizes, the design on several threads is the design on one;
    it is an exact cover with at least k groups, no worse than its start, within the node budget."""
    if not columns.feasible(n, Q, pairs):
        return
    Y, _, pool, k, root = _arm(seed, n, Q, min_pairs=pairs)
    start = _seed_design(n)
    if not _is_cover(pool, start, n, k):
        return
    saved = popmusic.WAVE
    popmusic.WAVE = wave
    try:
        kw = dict(alpha=alpha, delta=delta, repeat=repeat, sub_limit=sub)
        one = popmusic.improve(pool, root, k, Q, start, _paths(Y), budget, n_jobs=1, **kw)
        many = popmusic.improve(pool, root, k, Q, start, _paths(Y), budget, n_jobs=jobs, **kw)
    finally:
        popmusic.WAVE = saved
    _same(one, many)
    assert _is_cover(pool, one["rows"], n, k) and one["upper"] <= _upper(pool, start) and one["nodes"] <= budget
