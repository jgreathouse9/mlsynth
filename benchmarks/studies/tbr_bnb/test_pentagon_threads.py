"""Pentagonal separation on threads (#965): the ladder of the RCA that found it slow.

Incident. On the hotel_med search (27 markets, 4 treated, n_jobs=4) a pentagonal separation
took 0.125 s a call on main and 0.224 s with a solver that releases the interpreter lock,
against 0.066 s for one call alone. Four threads calling separate_pentagons at once took
13.7x to 14.8x the time of one thread's calls, 3.4x to 3.7x what running them one after
another costs.

Causes, each the bottom of its own branch:

1. The local search's starts were rebuilt on every call: 3000 five-sets and sign vectors
   drawn one start at a time from a numpy Generator, 46 to 52 ms of a 71 to 77 ms call. They
   depend only on (n, restarts, seed), so every call rebuilt the same arrays. Each draw
   releases and retakes the interpreter lock, about 6000 handoffs a call, so on 4 threads
   the draws took 18.8x to 21.2x one thread's time, at every switch interval tried (0.5,
   5 and 50 ms); the same loop drawn from the random module took 1.2x to 2.0x.
2. The violated rows were collected by a Python loop over every start (12 to 15 ms at a
   point violating many rows), which holds the lock, so threads ran it one at a time.

The compiled search itself took 1.6x on 4 threads (it releases the lock), and the sign
matrix 0.03 ms. The parts reproduce the whole: 974 + 14 + 58 = 1045 ms against 1052 ms
measured, in units of one thread's calls.

Rungs: the starts are built once per (n, restarts, seed) and cannot be written (causes 1, 4);
the rows are those of the per-start loop, at any point, seed, k and tolerance (cause 2, a
property against that loop kept here as the reference); two threads separating at once
cost about what one does (the symptom)."""
import threading
import time

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbr_helpers.design.exact import fastpent, slopefree
from mlsynth.utils.tbr_helpers.design.exact.slopefree import PENTAGON, _pentagon_search, _pentagon_starts, _sign_matrix, separate_pentagons

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _reference_starts(n, restarts, seed):
    """The starts as main drew them before the cache: one Generator, per start five indices
    without replacement and then five signs."""
    rng = np.random.default_rng(seed)
    S0 = np.empty((restarts, 5), dtype=np.int64); B0 = np.empty((restarts, 5), dtype=np.int64)
    for i in range(restarts):
        S0[i] = rng.choice(n, 5, replace=False)
        B0[i] = [int(v) for v in rng.choice([-1, 1], 5)]
    return S0, B0


def _reference(u, U, k=400, restarts=None, tol=1e-6, seed=0):
    """separate_pentagons as main wrote it before the fix: the same search, then a dict
    keyed by the sorted, sign-normalised five-set (the last start's violation kept),
    sorted by violation and then key."""
    restarts = slopefree.PENTAGON_RESTARTS if restarts is None else restarts
    n = np.asarray(u).size + 1
    if n < 5:
        return []
    Y, d = _sign_matrix(u, U)
    S0, B0 = _reference_starts(n, restarts, seed)
    run = fastpent.search if (slopefree.PENTAGON_KERNEL and fastpent.AVAILABLE) else _pentagon_search
    S, B, val = run(Y, d, S0, B0)
    found = {}
    for i in range(len(val)):
        if (1 - val[i]) / 2 > tol:
            key = sorted(zip((int(v) for v in S[i]), (int(s_) for s_ in B[i])))
            if key[0][1] < 0:
                key = [(v, -s_) for v, s_ in key]
            found[(tuple(v for v, _ in key), tuple(s_ for _, s_ in key))] = float((1 - val[i]) / 2)
    best = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(viol, PENTAGON, key) for key, viol in best]


def _point(Y):
    """(u, U) of the point whose sign matrix is Y (Y = T Z T', _sign_matrix)."""
    n = Y.shape[0]
    T = np.zeros((n, n)); T[0, 0] = 1.0; T[1:, 0] = -1.0; T[1:, 1:] = 2.0 * np.eye(n - 1)
    Ti = np.linalg.inv(T)
    Z = Ti @ Y @ Ti.T
    return Z[0, 1:], Z[1:, 1:]


def _correlation(n, r, seed):
    """A random correlation matrix of rank r: unit-diagonal and positive semidefinite."""
    G = np.random.default_rng(seed).standard_normal((n, r))
    Y = G @ G.T
    dg = np.sqrt(np.diag(Y))
    return Y / np.outer(dg, dg)


def _equal_minus_quarter(n):
    """Y_ij = -1/4 off the diagonal: every equal-sign five-set is violated by exactly 1/2,
    so the violations tie and the order among them is the key's."""
    Y = np.full((n, n), -0.25); np.fill_diagonal(Y, 1.0)
    return Y


# ---------------------------------------------------------------- smoke


def test_smoke_the_rows_at_a_random_point_are_the_per_start_loops():
    u, U = _point(_correlation(24, 6, 0))
    out = separate_pentagons(u, U, k=400)
    assert out and len(out) <= 400
    assert out == _reference(u, U, k=400)


# ---------------------------------------------------------------- cause 1: the starts


def test_the_starts_are_built_once_per_size_restarts_and_seed(monkeypatch):
    built = []
    real = np.random.default_rng

    def counting(seed=None):
        built.append(seed)
        return real(seed)

    monkeypatch.setattr(slopefree.np.random, "default_rng", counting)
    u, U = _point(_correlation(11, 4, 1))
    for _ in range(3):
        separate_pentagons(u, U, restarts=4321, seed=17)
    assert built.count(17) == 1


def test_the_same_starts_come_back_and_differ_by_seed_and_size():
    a = _pentagon_starts(12, 50, 3)
    assert _pentagon_starts(12, 50, 3) is a
    assert not np.array_equal(_pentagon_starts(12, 50, 4)[0], a[0])
    assert _pentagon_starts(13, 50, 3)[0].shape == (50, 5) and _pentagon_starts(12, 51, 3)[0].shape == (51, 5)


def test_the_starts_are_the_generators_draws():
    S0, B0 = _pentagon_starts(9, 40, 5)
    R0, C0 = _reference_starts(9, 40, 5)
    assert np.array_equal(S0, R0) and np.array_equal(B0, C0)
    assert S0.dtype == np.int64 and B0.dtype == np.int64


# ---------------------------------------------------------------- cause 2: the rows


def test_tied_violations_come_in_key_order():
    u, U = _point(_equal_minus_quarter(9))
    out = separate_pentagons(u, U, k=400, restarts=300)
    assert len(out) > 3 and {v for v, _, _ in out} == {0.5}
    assert [key for _, _, key in out] == sorted(key for _, _, key in out)
    assert out == _reference(u, U, k=400, restarts=300)


def test_a_row_found_from_many_starts_is_reported_once():
    u, U = _point(_equal_minus_quarter(6))
    out = separate_pentagons(u, U, restarts=500)
    keys = [key for _, _, key in out]
    assert len(keys) == len(set(keys)) and out == _reference(u, U, restarts=500)


def test_every_row_has_its_signs_normalised_and_its_indices_sorted():
    u, U = _point(_correlation(16, 3, 2))
    out = separate_pentagons(u, U, k=400, restarts=400)
    assert out
    for _, kind, (S, b) in out:
        assert kind == PENTAGON and list(S) == sorted(set(S)) and b[0] == 1
        assert all(type(x) is int for x in S + b)


def test_k_keeps_the_most_violated_rows():
    u, U = _point(_correlation(16, 3, 2))
    every = separate_pentagons(u, U, k=400, restarts=400)
    assert len(every) > 5
    assert separate_pentagons(u, U, k=5, restarts=400) == every[:5]


def test_the_violation_is_half_the_rows_excess():
    u, U = _point(_equal_minus_quarter(5))
    (viol, _, _), = separate_pentagons(u, U, restarts=50)
    assert viol == 0.5 and type(viol) is float


# ---------------------------------------------------------------- edges and the contract


def test_a_point_violating_nothing_has_no_rows():
    u, U = _point(np.eye(7))           # Y = I: b'Yb = 5 at every five-set
    assert separate_pentagons(u, U, restarts=50) == [] == _reference(u, U, restarts=50)


def test_one_start_gives_at_most_one_row():
    u, U = _point(_equal_minus_quarter(8))
    out = separate_pentagons(u, U, restarts=1)
    assert len(out) <= 1 and out == _reference(u, U, restarts=1)


def test_five_indices_is_enough():
    u, U = _point(_equal_minus_quarter(5))
    assert separate_pentagons(u, U, restarts=20) == _reference(u, U, restarts=20) != []


def test_the_cached_starts_cannot_be_written():
    S0, B0 = _pentagon_starts(10, 30, 0)
    with pytest.raises(ValueError):
        S0[0, 0] = 9
    with pytest.raises(ValueError):
        B0[0, 0] = -B0[0, 0]


def test_both_searches_take_the_read_only_starts_and_leave_them():
    u, U = _point(_correlation(10, 4, 3))
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(10, 60, 0)
    before = (S0.copy(), B0.copy())
    py = _pentagon_search(Y, d, S0, B0)
    if fastpent.AVAILABLE:
        fast = fastpent.search(Y, d, S0, B0)
        assert np.array_equal(fast[0], py[0]) and np.array_equal(fast[1], py[1])
    assert np.array_equal(S0, before[0]) and np.array_equal(B0, before[1])


@pytest.mark.parametrize("kw", [dict(restarts=[3]), dict(seed=1.5), dict(k=None)])
def test_bad_settings_are_refused_before_the_cache(kw):
    from mlsynth.exceptions import MlsynthConfigError
    u, U = _point(_correlation(8, 3, 0))
    with pytest.raises(MlsynthConfigError):
        separate_pentagons(u, U, **kw)


# ---------------------------------------------------------------- the symptom


@pytest.mark.skipif(not fastpent.AVAILABLE, reason="the compiled search is what runs on threads")
def test_two_threads_separating_at_once_cost_about_what_one_does():
    """Two threads, five calls each, against one thread's five calls: before the fix 4.3x to
    4.6x (the starts' draws handed the lock back and forth), after it about 1x. The bound,
    3x, is above what a machine with one core would take running them one after another."""
    u, U = _point(_correlation(24, 6, 0))
    separate_pentagons(u, U)

    def five():
        for _ in range(5):
            separate_pentagons(u, U)

    best = np.inf
    for _ in range(2):
        t = time.perf_counter(); five(); one = time.perf_counter() - t
        ts = [threading.Thread(target=five) for _ in range(2)]
        t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]
        best = min(best, (time.perf_counter() - t) / one)
    assert best < 3.0


# ---------------------------------------------------------------- properties


@SETTINGS
@given(st.integers(5, 14), st.integers(1, 6), st.integers(0, 10_000), st.integers(1, 120),
       st.integers(0, 50), st.integers(1, 60), st.sampled_from([1e-6, 1e-3, 0.05, 0.2]))
def test_property_the_rows_are_the_per_start_loops(n, r, pseed, restarts, seed, k, tol):
    u, U = _point(_correlation(n, r, pseed))
    assert separate_pentagons(u, U, k=k, restarts=restarts, tol=tol, seed=seed) == \
        _reference(u, U, k=k, restarts=restarts, tol=tol, seed=seed)


@SETTINGS
@given(st.integers(5, 12), st.integers(1, 200), st.integers(0, 50), st.integers(1, 30))
def test_property_ties_and_duplicates_follow_the_per_start_loop(n, restarts, seed, k):
    u, U = _point(_equal_minus_quarter(n))
    assert separate_pentagons(u, U, k=k, restarts=restarts, seed=seed) == \
        _reference(u, U, k=k, restarts=restarts, seed=seed)


@SETTINGS
@given(st.integers(5, 40), st.integers(1, 80), st.integers(0, 2**32 - 1))
def test_property_the_starts_are_the_draws_built_once_and_read_only(n, restarts, seed):
    S0, B0 = _pentagon_starts(n, restarts, seed)
    R0, C0 = _reference_starts(n, restarts, seed)
    assert np.array_equal(S0, R0) and np.array_equal(B0, C0)
    assert _pentagon_starts(n, restarts, seed) is _pentagon_starts(n, restarts, seed)
    assert not S0.flags.writeable and not B0.flags.writeable
