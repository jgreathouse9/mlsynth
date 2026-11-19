"""The exact search's RSS descent ends, including where designs fit exactly (#965).

``exact.search.incumbent`` descends from random control groups by add, drop and
swap moves, taking a move while it lowers the RSS. It used to take a move when the
new RSS was below ``cur * (1 - 1e-12)``, a test that assumes a positive RSS. Where
a design fits exactly the RSS is zero up to rounding and can be computed as a tiny
negative number; then ``cur * (1 - 1e-12)`` is above ``cur``, a move that changes
nothing passes the test, and the descent cycles between two exact fits forever. On
a panel whose geos are multiples of one series it never returned, and TBR's
search="exact" with it.

The descent now takes a move only when it lowers the RSS by more than 1e-12 of
y'y, the scale the RSS is computed at. The invariant: the descent ends, at a
design the band and the fixings allow, from which no allowed move lowers the RSS
by more than rounding.
"""
from __future__ import annotations

import signal
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbr_helpers.design.exact.hull import positive_rss
from mlsynth.utils.tbr_helpers.design.exact.search import incumbent
from mlsynth.utils.tbr_helpers.design.exact.treatment import certify_treatment
from mlsynth.utils.tbr_helpers.design.search import ROLES

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


class _Hung(Exception):
    pass


@contextmanager
def _within(seconds):
    """Fail, not hang, when the body runs past ``seconds``."""
    def _raise(*_):
        raise _Hung(f"did not return within {seconds} s")
    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _multiples(scales, n=30, seed=0):
    """Geos that are each a multiple of one series, centred: every design fits
    the treated geo exactly."""
    base = 100 + np.random.default_rng(seed).normal(size=n).cumsum()
    Y = np.outer(base, scales)
    return Y - Y.mean(axis=0)


def _local_optimum(y, X, C, *, fixed=frozenset(), band=None):
    """True when no add, drop or swap the band and the fixings allow lowers the
    positive-slope RSS of C by more than 1e-9 of y'y."""
    p = X.shape[1]
    lo, hi = (1, p) if band is None else band
    tol = 1e-9 * float(y @ y)
    cur = positive_rss(y, X, sorted(C))
    moves = [C | {j} for j in range(p) if j not in C] + \
            [C - {j} for j in C if j not in fixed] + \
            [(C - {a}) | {b} for a in C if a not in fixed for b in range(p) if b not in C]
    for S in moves:
        if lo <= len(S) <= hi and fixed <= S and positive_rss(y, X, sorted(S)) < cur - tol:
            return False
    return True


# --------------------------------------------------------------------------- smoke
def test_smoke_the_descent_ends_where_every_design_fits_exactly():
    Yc = _multiples([1.0, 2.0, 3.0, 0.5])
    y, X = Yc[:, 0], Yc[:, 1:]
    with _within(30):
        rss, C = incumbent(y, X, restarts=50, seed=0)
    assert C and abs(rss) <= 1e-9 * float(y @ y)


# --------------------------------------------------------------------------- unit
def test_every_start_ends_at_a_local_optimum_where_designs_fit_exactly():
    """Restarts enough to draw every start of three controls, among them the pair
    the old test cycled on."""
    Yc = _multiples([1.0, 2.0, 3.0, 0.5])
    y, X = Yc[:, 0], Yc[:, 1:]
    for seed in range(20):
        with _within(30):
            _, C = incumbent(y, X, restarts=20, seed=seed)
        assert _local_optimum(y, X, C)


def test_on_a_panel_with_noise_the_descent_reaches_the_enumerated_optimum():
    """The tolerance does not stop the descent short on an ordinary panel: with
    enough restarts it finds the best positive-slope control group."""
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(40, 6)).cumsum(axis=0) + 50
    Yc = Y - Y.mean(axis=0)
    y, X = Yc[:, 0], Yc[:, 1:]
    best = min(positive_rss(y, X, list(C)) for r in range(1, 6)
               for C in combinations(range(5), r))
    with _within(30):
        rss, _ = incumbent(y, X, restarts=200, seed=0)
    assert rss == pytest.approx(best, rel=1e-12, abs=1e-9 * float(y @ y))


@given(seed=st.integers(0, 10_000), p=st.integers(2, 6), exact=st.integers(0, 6),
       lo=st.integers(1, 3), width=st.integers(0, 5), fixed=st.sets(st.integers(0, 5), max_size=2))
@SETTINGS
def test_property_the_descent_ends_at_an_allowed_local_optimum(seed, p, exact, lo, width, fixed):
    """Over panels where some controls are exact multiples of the treated series and
    the rest carry noise, under random bands and fixings: the descent ends, keeps to
    both, and no allowed move improves on its design by more than rounding."""
    rng = np.random.default_rng(seed)
    n = 30
    base = 100 + rng.normal(size=n).cumsum()
    cols = [base * rng.uniform(0.5, 3.0) if j < exact else
            base * rng.uniform(0.5, 3.0) + rng.normal(scale=rng.choice([1e-9, 1.0, 10.0]), size=n)
            for j in range(p)]
    Y = np.column_stack([base] + cols)
    Yc = Y - Y.mean(axis=0)
    y, X = Yc[:, 0], Yc[:, 1:]
    fixed = frozenset(j for j in fixed if j < p)
    lo = min(lo, p)
    hi = min(p, max(lo, len(fixed)) + width)
    lo = min(lo, hi)
    with _within(30):
        _, C = incumbent(y, X, restarts=5, seed=seed, fixed=fixed, band=(lo, hi))
    assert fixed <= C and lo <= len(C) <= hi
    assert _local_optimum(y, X, C, fixed=fixed, band=(lo, hi))


# --------------------------------------------------------------------------- edge
def test_a_treated_series_of_zeros_ends_at_once():
    Yc = _multiples([1.0, 2.0, 3.0])
    y, X = np.zeros(Yc.shape[0]), Yc[:, 1:]
    with _within(30):
        rss, C = incumbent(y, X, restarts=10, seed=0)
    assert rss == 0.0 and C


def test_a_single_control_column():
    Yc = _multiples([1.0, 2.0])
    with _within(30):
        rss, C = incumbent(Yc[:, 0], Yc[:, 1:], restarts=5, seed=0)
    assert C == {0}


def test_the_certified_search_ends_where_every_design_fits_exactly():
    """End to end: with every design an exact fit there is no design to rank, and
    the search says so and ends, within its time limit, not past it."""
    base = 100 + np.random.default_rng(0).normal(size=30).cumsum()
    M = np.outer(base, [1.0, 2.0, 3.0, 0.5])
    with _within(60):
        out = certify_treatment(M, 1, n_test=5, eligibility=[frozenset(ROLES)] * 4,
                                n_jobs=1, time_limit=30.0)
    assert out.treatment is None and out.certified
