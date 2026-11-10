"""The floor kernel's NNLS, lean: the panel's Gram read through the controls' index
(no per-group copy), the gradient summed over the support only, and the Cholesky
factor of the support extended by a row when a column joins (refactored only when
one leaves). The point only decides how tight a floor is, since the certificate holds
at any point of the box, so the tests pin the lean solve to the active set it
replaces (kept below as the reference): the same objective to rounding, the KKT
conditions, and the same floors.
"""
import math
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from . import fastfloor
from .brute import geo_panel
from .treatment import gram_floor, rss_floor

needs_numba = pytest.mark.skipif(not fastfloor.AVAILABLE, reason="numba is not installed")
SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

if fastfloor.AVAILABLE:
    from numba import njit

    @njit(cache=False)
    def _ref_chol_solve(A, b, idx, k, L, z, x):
        for i in range(k):
            for j in range(i + 1):
                s = A[idx[i], idx[j]]
                for q in range(j):
                    s -= L[i, q] * L[j, q]
                if i == j:
                    if s <= 0.0:
                        return False
                    L[i, i] = math.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
        for i in range(k):
            s = b[idx[i]]
            for q in range(i):
                s -= L[i, q] * z[q]
            z[i] = s / L[i, i]
        for i in range(k - 1, -1, -1):
            s = z[i]
            for q in range(i + 1, k):
                s -= L[q, i] * x[q]
            x[i] = s / L[i, i]
        return True

    @njit(cache=False)
    def _ref_nnls(G, c, passive, w, maxit):
        """The kernel's active set before this change: full gradient, a fresh Cholesky
        for every solve, on a copied Gram."""
        m = c.shape[0]
        idx = np.empty(m, np.int64); L = np.empty((m, m)); z = np.empty(m); x = np.empty(m)
        cmax = 1.0
        for j in range(m):
            if abs(c[j]) > cmax:
                cmax = abs(c[j])
        tol = 1e-12 * cmax
        for j in range(m):
            w[j] = 0.0
        while True:
            k = 0
            for j in range(m):
                if passive[j]:
                    idx[k] = j
                    k += 1
            if k == 0:
                break
            ok = _ref_chol_solve(G, c, idx, k, L, z, x)
            bad = False
            for q in range(k):
                if (not ok) or not (x[q] > 0.0):
                    passive[idx[q]] = False
                    bad = True
            if not bad:
                for q in range(k):
                    w[idx[q]] = x[q]
                break
        for _ in range(maxit):
            best = tol
            jbest = -1
            for j in range(m):
                if not passive[j]:
                    g = c[j]
                    for q in range(m):
                        g -= G[j, q] * w[q]
                    if g > best:
                        best = g
                        jbest = j
            if jbest < 0:
                return
            passive[jbest] = True
            while True:
                k = 0
                for j in range(m):
                    if passive[j]:
                        idx[k] = j
                        k += 1
                ok = _ref_chol_solve(G, c, idx, k, L, z, x)
                if not ok:
                    passive[jbest] = False
                    return
                allpos = True
                for q in range(k):
                    if not (x[q] > 0.0):
                        allpos = False
                if allpos:
                    for j in range(m):
                        w[j] = 0.0
                    for q in range(k):
                        w[idx[q]] = x[q]
                    break
                alpha = 1.0
                for q in range(k):
                    if not (x[q] > 0.0):
                        wq = w[idx[q]]
                        a = wq / (wq - x[q]) if wq - x[q] != 0.0 else 0.0
                        if a < alpha:
                            alpha = a
                for q in range(k):
                    j = idx[q]
                    w[j] = w[j] + alpha * (x[q] - w[j])
                    if w[j] <= 1e-15 * max(1.0, abs(x[q])):
                        w[j] = 0.0
                        passive[j] = False


def _problem(seed, n=40, p=12, k=2, scale=1.0):
    """A group's Gram and cross products from a panel: G over the controls, c = X'y."""
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 10.0, size=(n, 1))
    M = (base + rng.normal(0.0, 5.0, size=(n, p)).cumsum(axis=0) / 4) * scale
    T = list(range(k))
    rest = np.array([j for j in range(p) if j not in T], np.int64)
    Gf = np.ascontiguousarray(M.T @ M)
    c = Gf[np.ix_(rest, T)].sum(axis=1)
    return Gf, rest, np.ascontiguousarray(c)


def _lean(Gf, rest, c, passive, maxit=500):
    m = rest.shape[0]
    w = np.zeros(m)
    idx = np.empty(m, np.int64); L = np.empty((m, m)); z = np.empty(m); x = np.empty(m)
    fastfloor._nnls_gram(Gf, rest, c, passive, w, idx, L, z, x, maxit)
    return w


def _ref(Gf, rest, c, passive, maxit=500):
    G = np.ascontiguousarray(Gf[np.ix_(rest, rest)])
    w = np.zeros(rest.shape[0])
    _ref_nnls(G, c, passive, w, maxit)
    return w


def _objective(Gf, rest, c, w):
    G = Gf[np.ix_(rest, rest)]
    return float(w @ G @ w - 2.0 * c @ w)


def _scale(Gf, rest, c):
    return max(1.0, float(np.abs(c).max()) ** 2 / max(float(np.diag(Gf)[rest].min()), 1e-300))


# ---------------------------------------------------------------- smoke


@needs_numba
def test_smoke_a_feasible_point_at_the_references_objective():
    Gf, rest, c = _problem(0)
    w = _lean(Gf, rest, c, np.zeros(rest.shape[0], np.bool_))
    assert w.shape == c.shape and np.all(np.isfinite(w)) and np.all(w >= 0)
    ref = _ref(Gf, rest, c, np.zeros(rest.shape[0], np.bool_))
    assert _objective(Gf, rest, c, w) == pytest.approx(_objective(Gf, rest, c, ref), rel=1e-10, abs=1e-9)


# ---------------------------------------------------------------- unit


@needs_numba
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [1, 2, 4])
def test_the_point_is_the_references_point(seed, k):
    Gf, rest, c = _problem(seed, k=k)
    m = rest.shape[0]
    w = _lean(Gf, rest, c, np.zeros(m, np.bool_))
    ref = _ref(Gf, rest, c, np.zeros(m, np.bool_))
    np.testing.assert_allclose(w, ref, rtol=1e-7, atol=1e-9 * max(1.0, float(np.abs(ref).max())))


@needs_numba
@pytest.mark.parametrize("seed", range(6))
def test_the_kkt_conditions_hold(seed):
    """w >= 0; the gradient c - Gw is at most the tolerance off the support and zero on
    it, to rounding."""
    Gf, rest, c = _problem(seed, p=16, k=3)
    w = _lean(Gf, rest, c, np.zeros(rest.shape[0], np.bool_))
    G = Gf[np.ix_(rest, rest)]
    g = c - G @ w
    scale = float(np.abs(c).max())
    assert np.all(w >= 0)
    assert np.all(g[w == 0] <= 1e-9 * scale)
    assert np.all(np.abs(g[w > 0]) <= 1e-7 * scale)


@needs_numba
def test_the_support_is_left_in_passive():
    """The next group's warm start reads ``passive``: it is the support of the point."""
    Gf, rest, c = _problem(3, p=16, k=2)
    passive = np.zeros(rest.shape[0], np.bool_)
    w = _lean(Gf, rest, c, passive)
    assert np.array_equal(passive, w > 0)


@needs_numba
def test_a_warm_start_ends_where_a_cold_start_does():
    Gf, rest, c = _problem(4, p=16, k=2)
    m = rest.shape[0]
    cold = _lean(Gf, rest, c, np.zeros(m, np.bool_))
    for start in (np.ones(m, np.bool_), cold > 0, np.arange(m) % 2 == 0):
        warm = _lean(Gf, rest, c, start.copy())
        assert _objective(Gf, rest, c, warm) == pytest.approx(_objective(Gf, rest, c, cold), rel=1e-10, abs=1e-9)


@needs_numba
def test_the_kernels_floors_are_rss_floors_on_a_wider_panel():
    """Through the whole kernel, on a panel wider than the existing suite's: 3 of 18 geos."""
    M = geo_panel(7, n=60, p=18)
    lam = gram_floor(M)
    groups = list(combinations(range(18), 3))[::7]
    fl, lo, hi = fastfloor.floors(np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M),
                                  np.array(groups, np.int64), float(lam))
    for f, T in zip(fl, groups):
        old = float(rss_floor(M, T, lam))
        assert abs(max(f, 0.0) - old) <= 1e-9 * max(1.0, abs(old))


# ---------------------------------------------------------------- edge


@needs_numba
def test_a_single_control():
    Gf, rest, c = _problem(5, p=3, k=2)
    w = _lean(Gf, rest, c, np.zeros(1, np.bool_))
    assert w[0] == pytest.approx(max(0.0, c[0] / Gf[rest[0], rest[0]]), rel=1e-12)


@needs_numba
def test_no_positive_gradient_leaves_zero():
    Gf, rest, c = _problem(6)
    w = _lean(Gf, rest, -np.abs(c), np.zeros(rest.shape[0], np.bool_))
    assert np.all(w == 0)


@needs_numba
def test_a_duplicated_column_stops_at_a_feasible_point_as_the_reference_does():
    """A column in the span of the support fails its Cholesky row: the solve stops at
    the last feasible point, as before."""
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, 6)); X = np.c_[X, X[:, 0]]
    y = X[:, :3].sum(axis=1) + 0.1 * rng.normal(size=30)
    Gf = np.ascontiguousarray(X.T @ X); rest = np.arange(7, dtype=np.int64); c = X.T @ y
    w = _lean(Gf, rest, c, np.zeros(7, np.bool_))
    ref = _ref(Gf, rest, c, np.zeros(7, np.bool_))
    assert np.all(w >= 0)
    assert _objective(Gf, rest, c, w) == pytest.approx(_objective(Gf, rest, c, ref), rel=1e-8, abs=1e-8)


@needs_numba
def test_a_warm_start_with_columns_that_must_leave():
    """Every column passive at the start: those the solve drops are refactored out."""
    Gf, rest, c = _problem(9, p=20, k=4)
    m = rest.shape[0]
    w = _lean(Gf, rest, c, np.ones(m, np.bool_))
    ref = _ref(Gf, rest, c, np.zeros(m, np.bool_))
    assert (w == 0).any()
    assert _objective(Gf, rest, c, w) == pytest.approx(_objective(Gf, rest, c, ref), rel=1e-10, abs=1e-9)


@needs_numba
def test_the_iteration_cap_is_kept():
    Gf, rest, c = _problem(10, p=20, k=2)
    m = rest.shape[0]
    w = _lean(Gf, rest, c, np.zeros(m, np.bool_), maxit=2)
    assert np.count_nonzero(w) <= 2 and np.all(w >= 0)


# ---------------------------------------------------------------- property


@needs_numba
@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(4, 18), k=st.integers(1, 3),
       scale_e=st.integers(-6, 6), warm=st.integers(0, 2 ** 20))
@example(seed=0, p=4, k=3, scale_e=0, warm=0)
def test_property_the_lean_solve_reaches_the_references_objective(seed, p, k, scale_e, warm):
    """Over panels, sizes, scales and warm starts: feasible, its support left in
    ``passive``, and the reference's objective to rounding."""
    Gf, rest, c = _problem(seed, n=30, p=p, k=k, scale=2.0 ** scale_e)
    m = rest.shape[0]
    start = np.array([(warm >> j) & 1 for j in range(m)], np.bool_)
    passive = start.copy()
    w = _lean(Gf, rest, c, passive)
    ref = _ref(Gf, rest, c, np.zeros(m, np.bool_))
    assert np.all(w >= 0) and np.array_equal(passive, w > 0)
    fo, fr = _objective(Gf, rest, c, w), _objective(Gf, rest, c, ref)
    assert fo <= fr + 1e-9 * max(1.0, abs(fr))
    assert fr <= fo + 1e-9 * max(1.0, abs(fo))
