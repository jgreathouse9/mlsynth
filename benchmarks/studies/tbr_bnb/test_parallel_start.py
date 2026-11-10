"""The start's restarts on the search's pool.

The start is the library's greedy design, descended by its local search from the
greedy design and from ``restarts`` random designs drawn from one seeded generator,
the best descent winning and an earlier one keeping a tie. The draws depend only on
the generator and each descent only on its own draw, so with a pool the start draws
them first, descends them on the pool, and takes the best in draw order: the same
design as local_search's, in a quarter of the time on four processes. The tests pin
that equality, on the pool and in this process, across panels and seeds.
"""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design import local as lib_local
from mlsynth.utils.tbr_helpers.design.local import local_search
from mlsynth.utils.tbr_helpers.design.search import greedy_search

from . import treatment
from .brute import geo_panel

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _roles(p):
    return [frozenset({"treatment", "control", "unassigned"})] * p


def _library(M, k, restarts):
    """The library's start: greedy, then local_search with all its restarts, as one call."""
    out = greedy_search(M, _roles(M.shape[1]), max_treatment_size=k, n_test=N_TEST, objective="reference")
    best = local_search(M, _roles(M.shape[1]), out[-1], n_test=N_TEST, restarts=restarts, seed=0)
    return sorted(best.treatment), sorted(best.control)


@pytest.fixture(scope="module")
def pool():
    with treatment._pool(2) as p:
        yield p


# ---------------------------------------------------------------- smoke


def test_smoke_the_pooled_start_is_a_design_of_size_k(pool):
    M = geo_panel(5)
    T, C = treatment._start(M, 3, N_TEST, 6, pool)
    assert len(T) == 3 and T == sorted(T) and C == sorted(C) and C
    assert not set(T) & set(C) and set(T) | set(C) <= set(range(M.shape[1]))


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed,k", [(5, 2), (5, 3), (7, 3), (3, 4)])
def test_the_pooled_start_is_local_searchs(pool, seed, k):
    M = geo_panel(seed)
    assert treatment._start(M, k, N_TEST, 8, pool) == _library(M, k, 8)


def test_without_a_pool_the_start_is_local_searchs():
    M = geo_panel(5)
    assert treatment._start(M, 3, N_TEST, 8, None) == _library(M, 3, 8)


def test_the_search_hands_the_start_its_pool(monkeypatch):
    """certify_treatment on two jobs gives _start the pool it made; on one job, none."""
    seen = []
    real = treatment._start

    def spy(M, k, n_test, restarts, pool=None):
        seen.append(pool)
        return real(M, k, n_test, restarts, pool)

    monkeypatch.setattr(treatment, "_start", spy)
    M = geo_panel(5)
    treatment.certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=2, gap=Q(0), n_jobs=2)
    treatment.certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=2, gap=Q(0), n_jobs=1)
    assert seen[0] is not None and seen[1] is None


def test_the_descents_run_on_the_pool(monkeypatch, pool):
    """Every restart's descent goes through the pool, not this process."""
    calls = []
    real = treatment._map

    def spy(p, fn, argss):
        argss = list(argss)
        calls.append((p, fn, len(argss)))
        return real(p, fn, argss)

    monkeypatch.setattr(treatment, "_map", spy)
    treatment._start(geo_panel(5), 3, N_TEST, 6, pool)
    assert calls == [(pool, treatment._descend, 6)]


def test_a_tie_keeps_the_earlier_descent(monkeypatch):
    """Descents that tie the best keep the first, as local_search's strict > does: the
    greedy's descent over every restart that only matches it, and the first restart
    over a later one with the same key."""
    M = geo_panel(5)
    real = treatment._descend
    base = real(M, _roles(M.shape[1]), *_library(M, 3, 0), N_TEST)

    def tied(M, roles, T, C, n_test):
        return base[0], [0, 1, 2], [3, 4]

    monkeypatch.setattr(treatment, "_map", lambda p, fn, argss: [fn(*a) for a in argss])  # in this process
    monkeypatch.setattr(treatment, "_descend", tied)
    T, C = treatment._start(M, 3, N_TEST, 4, object())
    assert (T, C) == _library(M, 3, 0)          # the greedy's descent keeps the tie

    order = iter(range(100))

    def ranked(M, roles, T, C, n_test):
        i = next(order)
        return base[0] + (1,), [i, 50, 51], [52]  # every restart beats the greedy, all tie

    monkeypatch.setattr(treatment, "_descend", ranked)
    T, C = treatment._start(M, 3, N_TEST, 4, object())
    assert T == [0, 50, 51]                      # the first restart keeps the tie


# ---------------------------------------------------------------- edge


def test_no_restarts_is_the_greedys_descent(monkeypatch, pool):
    calls = []
    monkeypatch.setattr(treatment, "_map", lambda p, fn, a: calls.append(1) or [])
    M = geo_panel(5)
    assert treatment._start(M, 3, N_TEST, 0, pool) == _library(M, 3, 0)
    assert not calls


def test_a_draw_that_is_no_design_is_skipped_as_local_search_skips_it(monkeypatch, pool):
    real = lib_local._random_design
    n = iter(range(1000))

    def sometimes(rng, roles, k):
        d = real(rng, roles, k)
        return None if next(n) % 3 == 1 else d

    M = geo_panel(5)
    monkeypatch.setattr(lib_local, "_random_design", sometimes)
    pooled = treatment._start(M, 3, N_TEST, 9, pool)
    n = iter(range(1000))
    assert pooled == _library(M, 3, 9)


def test_a_single_control_panel(pool):
    M = geo_panel(5)[:, :3]
    assert treatment._start(M, 2, N_TEST, 4, pool) == _library(M, 2, 4)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("use_pool", [False, True])
def test_negative_restarts_are_refused(pool, use_pool):
    with pytest.raises(MlsynthDataError, match="restarts"):
        treatment._start(geo_panel(5), 3, N_TEST, -1, pool if use_pool else None)


def test_a_worker_error_reaches_the_caller(pool):
    """A panel the descent cannot score raises from the pool, not a design."""
    M = geo_panel(5).copy()
    with pytest.raises(IndexError):
        treatment._descend(M, _roles(M.shape[1]), [0, 1], [2, 3, 99], N_TEST)
    with pytest.raises(IndexError):
        treatment._map(pool, treatment._descend, [(M, _roles(M.shape[1]), [0, 1], [2, 3, 99], N_TEST)])


# ---------------------------------------------------------------- property


@SETTINGS
@given(seed=st.integers(0, 40), p=st.integers(5, 9), k=st.integers(1, 3), restarts=st.integers(0, 7))
def test_property_the_pooled_start_is_local_searchs(seed, p, k, restarts):
    M = geo_panel(seed, n=24, p=p)
    with treatment._pool(2) as pool:
        assert treatment._start(M, k, N_TEST, restarts, pool) == _library(M, k, restarts)
