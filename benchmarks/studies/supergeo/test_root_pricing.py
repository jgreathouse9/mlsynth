"""The root's pricing by the safe window, beside the brute-force scan.

Column generation at the root prices every group of lo..hi markets against the duals. The brute-force scan
(sg._scan, or parity.scan for an odd count) walks all of them, 1.26e8 at 235 markets, whatever the duals. Once the
duals are small, few groups can have a negative reduced cost, and the safe window (certify_exact.safe_window, a
superset of every group with reduced cost at or below a threshold) holds few rows: on a simulated 235-market panel
3,539 rows at the last iteration against 126,018,750 groups. certify_exact._price_window scores the window's rows with
the scan's own arithmetic and keeps them in the scan's order, so it returns the scan's arrays exactly, and
certify_exact._root_price chooses the window when its row count, counted with an early stop
(certify_exact._window_rows), is at most the number of groups over WINDOW_RATIO.
"""
import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import certify_exact as ce, parity, sg, test_bcp

LIMITS = [(2, 4), (2, 3), (3, 4), (2, 2), (4, 4)]


def _scan(z, pi, tau, lo, hi, thresh, odd):
    if odd:
        return parity.scan(z, pi, tau, lo, hi, thresh, 5_000_000)[:3]
    return sg._scan(z, pi, lo, hi, thresh, 5_000_000)[:3]


def _instance(seed, n, kind, scale):
    rng = np.random.default_rng(seed)
    z = test_bcp.gen(rng, n, kind)
    pi = np.abs(rng.normal(0, scale, n))
    return z, pi


# --- the window route returns the scan's arrays --------------------------------------------------------------------------

@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 14), kind=st.integers(0, 3), limits=st.sampled_from(LIMITS),
       scale=st.sampled_from([1e-3, 0.1, 1.0, 30.0]), tau=st.floats(0, 3), thresh=st.sampled_from([0.0, -1e-9, 0.5]))
def test_the_window_route_returns_the_scans_arrays(seed, n, kind, limits, scale, tau, thresh):
    """Property, over the four generators, every group-size limit, duals from tiny to large, any parity credit and
    threshold, odd and even counts: the window route's rows, scores and reduced costs equal the scan's, element for
    element and in the same order, so the root's choice of the 2,000 cheapest columns (an argsort) is the same."""
    lo, hi = limits
    z, pi = _instance(seed, n, kind, scale)
    odd = n % 2 == 1
    tau = tau if odd else 0.0
    want = _scan(z, pi, tau, lo, hi, thresh, odd)
    got = ce._price_window(z, pi, tau, lo, hi, thresh, odd)
    for a, b in zip(want, got):
        assert a.shape[0] == b.shape[0]
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_window_with_no_negative_group_returns_nothing():
    """Edge: with zero duals no group has a negative reduced cost; the window route returns empty arrays of the
    scan's shapes."""
    z = np.arange(1.0, 10.0)
    rows, sc, rc = ce._price_window(z, np.zeros(9), 0.0, 2, 4, -1e-9, True)
    assert rows.shape == (0, 4) and sc.shape == (0,) and rc.shape == (0,)


def test_equal_totals_tie_everywhere():
    """Edge: eight equal totals, every group of even size costs 0 and every triple a total squared; the window route
    and the scan agree under ties."""
    z = np.full(8, 3.0)
    pi = np.full(8, 0.25)
    want = _scan(z, pi, 0.0, 2, 4, 0.0, False)
    got = ce._price_window(z, pi, 0.0, 2, 4, 0.0, False)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(want, got))


# --- the capped count ------------------------------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 14), kind=st.integers(0, 3), limits=st.sampled_from(LIMITS),
       scale=st.sampled_from([1e-3, 1.0, 30.0]), limit=st.integers(0, 3000))
def test_the_capped_count_is_exact_below_its_limit(seed, n, kind, limits, scale, limit):
    """Property: _window_rows returns the window's exact row count (the sum of its per-size counts) when that count
    is at most the limit, and some number above the limit otherwise: never below the true count, so the switch
    never chooses the window for a window larger than it allows."""
    lo, hi = limits
    z, pi = _instance(seed, n, kind, scale)
    odd = n % 2 == 1
    tau = 0.5 if odd else 0.0
    full = ce._window_rows(z, pi, tau, lo, hi, 0.0, odd, limit=None)
    got = ce._window_rows(z, pi, tau, lo, hi, 0.0, odd, limit=limit)
    if full <= limit:
        assert got == full
    else:
        assert limit < got <= full


# --- the switch ------------------------------------------------------------------------------------------------------

def test_the_switch_takes_the_window_when_it_is_small(monkeypatch):
    """Unit: _root_price uses the window when its rows number at most groups / WINDOW_RATIO, and the scan otherwise;
    the count is asked to stop at that limit."""
    seen = {}
    def rows(z, pi, tau, lo, hi, thresh, odd, limit=None):
        seen["limit"] = limit
        return seen["rows"]
    monkeypatch.setattr(ce, "_window_rows", rows)
    monkeypatch.setattr(ce, "_price_window", lambda *a: "window")
    monkeypatch.setattr(ce, "_price_scan", lambda *a: "scan")
    z = np.arange(1.0, 21.0); pi = np.zeros(20)
    groups = sum(math.comb(20, k) for k in (2, 3, 4))
    limit = groups // ce.WINDOW_RATIO
    seen["rows"] = limit
    assert ce._root_price(z, pi, 0.0, 2, 4, 0.0, False) == "window" and seen["limit"] == limit
    seen["rows"] = limit + 1
    assert ce._root_price(z, pi, 0.0, 2, 4, 0.0, False) == "scan"


def test_the_root_is_the_same_by_either_route(monkeypatch):
    """Unit: the root's rescaled totals, scale and duals are the same with the window always taken as with the scan
    always taken, on an odd and an even instance."""
    for n in (13, 14):
        z = test_bcp.gen(np.random.default_rng(n), n, 2)
        monkeypatch.setattr(ce, "WINDOW_RATIO", 10 ** 12)                    # the window never small enough
        a = ce._root(z, 2, 4)
        monkeypatch.setattr(ce, "WINDOW_RATIO", 1)
        monkeypatch.setattr(ce, "_window_rows", lambda *a, **k: 0)            # the window always taken
        b = ce._root(z, 2, 4)
        monkeypatch.undo()
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        assert np.array_equal(a[2], b[2]) and a[3] == b[3]
