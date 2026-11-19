"""The pentagonal separation's local search in one compiled loop (fastpent), and
separate_pentagons' use of it when numba is installed (#965).

separate_pentagons draws its starts (_pentagon_starts) and then runs a local search
from each: replace one member at a time by the index and sign that lower b'Y_SS b
most, positions in order, until a pass moves nothing. The search was a Python loop
of small numpy calls, about 1 s a call at any panel size and holding the interpreter
lock, so sizes on threads waited on each other. fastpent.search is the same loop
compiled, nogil. At a point whose entries are dyadic with small denominators every
sum the search takes is exact in any order, so the compiled and Python searches
take the same steps and report the same rows; elsewhere they may differ in the last
bit of a value, and each still stops every start at a local minimum. The tests check
both, the dispatch, the refusals, and the triangle separation's cached index.
"""
import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from mlsynth.utils.tbr_helpers.design.exact import fastpent, slopefree
from mlsynth.utils.tbr_helpers.design.exact.slopefree import _pentagon_search, _pentagon_starts, _sign_matrix, separate_pentagons, separate_triangles
from .test_pentagons import _design, _many_pentagons, _pentagon_point

needs_numba = pytest.mark.skipif(not fastpent.AVAILABLE, reason="numba is not installed")

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _python(monkeypatch):
    """separate_pentagons on the Python search."""
    monkeypatch.setattr(slopefree, "PENTAGON_KERNEL", False)


def _dyadic_point(seed, p, den=16):
    """u and U with entries in multiples of 1/den, U symmetric with u on its diagonal:
    every product and sum the search forms is exact in doubles."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, den + 1, p) / den
    U = rng.integers(0, den + 1, (p, p)) / den
    U = np.triu(U, 1); U = U + U.T; U[np.diag_indices(p)] = u
    return u, U


def _local_minimum(Y, d, S, b, val):
    """Whether no member of (S, b) can be replaced, at its best sign, to lower b'Y_SS b
    by more than 1e-12, and val is b'Y_SS b."""
    S, b = list(S), np.array(b, dtype=float)
    if not np.isclose(val, b @ Y[np.ix_(S, S)] @ b, rtol=0, atol=1e-9):
        return False
    for t in range(5):
        rest = [q for q in range(5) if q != t]
        for v in range(Y.shape[0]):
            if v in [S[q] for q in rest]:
                continue
            for s in (-1, 1):
                T = [S[q] for q in rest] + [v]; c = np.r_[b[rest], s]
                if c @ Y[np.ix_(T, T)] @ c < val - 1e-12:
                    return False
    return True


# ---------------------------------------------------------------- smoke


@needs_numba
def test_smoke_the_compiled_search_reports_the_python_searchs_rows(monkeypatch):
    Z = _many_pentagons()
    fast = separate_pentagons(Z[0, 1:], Z[1:, 1:], k=400)
    _python(monkeypatch)
    slow = separate_pentagons(Z[0, 1:], Z[1:, 1:], k=400)
    assert fast and fast == slow


# ---------------------------------------------------------------- unit


@needs_numba
def test_the_compiled_search_stops_every_start_at_a_local_minimum():
    """On a grid of 1/128 a step can gain as little as 1/128^2, so a search that stops on
    small gains stops short here."""
    u, U = _dyadic_point(1, 11, den=128)
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 60, 0)
    S, B, val = fastpent.search(Y, d, S0, B0)
    assert S.shape == (60, 5) and B.shape == (60, 5) and val.shape == (60,)
    for i in range(60):
        assert len(set(S[i])) == 5 and S[i].min() >= 0 and S[i].max() < Y.shape[0]
        assert set(B[i]) <= {-1, 1}
        assert _local_minimum(Y, d, S[i], B[i], val[i])


@needs_numba
@pytest.mark.parametrize("den", [16, 128])
def test_the_compiled_and_python_searches_take_the_same_steps(den):
    u, U = _dyadic_point(2, 13, den)
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 80, 4)
    a = fastpent.search(Y, d, S0, B0)
    b = _pentagon_search(Y, d, S0, B0)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@needs_numba
def test_a_move_that_gains_little_is_taken():
    """Y = I with Y_05 = Y_50 = eps / 2: from the start (0, 1, 2, 3, 4), all signs +,
    replacing member 1 by index 5 at sign -1 lowers b'Y_SS b from 5 to 5 - eps, and no
    other move from the start does. eps = 2^-12 is dyadic, so the values are exact, and
    small, so a search that needs a larger gain to move stays at 5."""
    eps = 2.0 ** -12
    Y = np.eye(6); Y[0, 5] = Y[5, 0] = eps / 2
    d = np.diag(Y).copy()
    S0 = np.array([[0, 1, 2, 3, 4]], dtype=np.int64); B0 = np.ones((1, 5), dtype=np.int64)
    S, B, val = fastpent.search(Y, d, S0, B0)
    assert 5 in S[0] and val[0] <= 5 - eps
    for x, y in zip((S, B, val), _pentagon_search(Y, d, S0, B0)):
        assert np.array_equal(x, y)


@needs_numba
def test_the_search_leaves_its_starts_untouched():
    u, U = _dyadic_point(3, 9)
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 20, 1)
    keep = S0.copy(), B0.copy()
    fastpent.search(Y, d, S0, B0)
    assert np.array_equal(S0, keep[0]) and np.array_equal(B0, keep[1])


@needs_numba
def test_separate_pentagons_runs_the_kernel_when_numba_is_installed(monkeypatch):
    calls, real = [], fastpent.search

    def spy(*a):
        calls.append(1)
        return real(*a)
    monkeypatch.setattr(fastpent, "search", spy)
    Z = _pentagon_point()
    separate_pentagons(Z[0, 1:], Z[1:, 1:], restarts=10)
    assert calls == [1]


@pytest.mark.parametrize("switch", ["kernel-off", "numba-missing"])
def test_separate_pentagons_runs_the_python_search_otherwise(monkeypatch, switch):
    if switch == "kernel-off":
        monkeypatch.setattr(slopefree, "PENTAGON_KERNEL", False)
    else:
        monkeypatch.setattr(fastpent, "AVAILABLE", False)

    def boom(*a):
        raise AssertionError("the kernel ran")
    monkeypatch.setattr(fastpent, "search", boom)
    Z = _pentagon_point()
    out = separate_pentagons(Z[0, 1:], Z[1:, 1:])
    assert out and out[0][2] == ((0, 1, 2, 3, 4), (1, 1, 1, 1, 1))


@needs_numba
def test_the_kernel_releases_the_interpreter_lock():
    assert fastpent._search.targetoptions.get("nogil") is True


def test_the_starts_are_the_generators_draws_in_order():
    """Each start's five indices without replacement, then its five signs, from one
    generator seeded by ``seed``: the draws the search made inside its loop before."""
    S0, B0 = _pentagon_starts(9, 4, 7)
    rng = np.random.default_rng(7)
    for i in range(4):
        assert list(S0[i]) == list(rng.choice(9, 5, replace=False))
        assert list(B0[i]) == [int(v) for v in rng.choice([-1, 1], 5)]


def test_the_triangle_index_is_built_once_per_size_and_read_only():
    a = slopefree._triples(9)
    assert a is slopefree._triples(9)
    assert a.shape == (84, 3) and not a.flags.writeable
    assert [tuple(r) for r in a] == list(itertools.combinations(range(9), 3))


def _triangles_reference(u, U, k=400, tol=1e-6):
    """separate_triangles as it was, building its index on every call."""
    u = np.asarray(u, float); U = np.asarray(U, float); p = u.size
    if p < 3:
        return []
    t = np.array(list(itertools.combinations(range(p), 3)))
    a, b, d = t.T
    out = []
    for i, j, kk in ((a, b, d), (b, a, d), (d, a, b)):
        v = U[i, j] + U[i, kk] - U[j, kk] - u[i]
        out += [(float(v[q]), 0, (int(i[q]), int(j[q]), int(kk[q]))) for q in np.flatnonzero(v > tol)]
    v = u[a] + u[b] + u[d] - U[a, b] - U[a, d] - U[b, d] - 1
    out += [(float(v[q]), 1, (int(a[q]), int(b[q]), int(d[q]))) for q in np.flatnonzero(v > tol)]
    out.sort(key=lambda r: -r[0])
    return out[:k]


@pytest.mark.parametrize("seed", range(4))
def test_the_cached_index_finds_the_same_triangles(seed):
    u, U = _dyadic_point(seed, 12, den=8)
    assert separate_triangles(u, U, k=60) == _triangles_reference(u, U, k=60)


# ---------------------------------------------------------------- edges


@needs_numba
def test_one_start(monkeypatch):
    Z = _many_pentagons()
    fast = separate_pentagons(Z[0, 1:], Z[1:, 1:], restarts=1)
    _python(monkeypatch)
    assert fast == separate_pentagons(Z[0, 1:], Z[1:, 1:], restarts=1)


@needs_numba
def test_exactly_five_indices(monkeypatch):
    u, U = _dyadic_point(5, 4)
    fast = separate_pentagons(u, U, restarts=30)
    _python(monkeypatch)
    assert fast == separate_pentagons(u, U, restarts=30)


@needs_numba
def test_ties_go_to_the_first_index(monkeypatch):
    """Y_ij = -1/4 off the diagonal: every candidate of a step ties with others."""
    Z = _many_pentagons()
    Y, d = _sign_matrix(Z[0, 1:], Z[1:, 1:])
    S0, B0 = _pentagon_starts(Y.shape[0], 50, 2)
    for x, y in zip(fastpent.search(Y, d, S0, B0), _pentagon_search(Y, d, S0, B0)):
        assert np.array_equal(x, y)


@needs_numba
def test_a_design_violates_no_pentagon_on_the_kernel():
    Z = np.asarray(_design(9, (0, 3, 4, 8)), dtype=float)
    assert separate_pentagons(Z[0, 1:], Z[1:, 1:]) == []


@pytest.mark.parametrize("p", [0, 1, 2])
def test_fewer_than_three_controls_have_no_triangle(p):
    assert separate_triangles(np.zeros(p), np.zeros((p, p))) == []


# ---------------------------------------------------------------- failures


@needs_numba
@pytest.mark.parametrize("bad", ["shape", "index", "repeat", "sign"])
def test_the_kernel_refuses_starts_it_cannot_search(bad):
    u, U = _dyadic_point(6, 8)
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 5, 0)
    if bad == "shape":
        S0 = S0[:, :4]
    elif bad == "index":
        S0 = S0.copy(); S0[2, 1] = Y.shape[0]
    elif bad == "repeat":
        S0 = S0.copy(); S0[1, 1] = S0[1, 0]
    else:
        B0 = B0.copy(); B0[0, 0] = 0
    with pytest.raises(MlsynthConfigError, match="start"):
        fastpent.search(Y, d, S0, B0)


@needs_numba
def test_the_kernel_refuses_a_sign_matrix_that_is_not_square():
    u, U = _dyadic_point(7, 8)
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 3, 0)
    with pytest.raises(MlsynthConfigError, match="sign matrix"):
        fastpent.search(Y[:, :-1], d, S0, B0)


# ---------------------------------------------------------------- properties


@needs_numba
@SETTINGS
@given(st.integers(0, 10_000), st.integers(4, 14), st.sampled_from([2, 4, 16, 128]), st.integers(1, 40), st.integers(1, 30))
def test_property_at_dyadic_points_the_kernel_reports_the_python_searchs_rows(seed, p, den, restarts, k):
    u, U = _dyadic_point(seed, p, den)
    fast = separate_pentagons(u, U, k=k, restarts=restarts, seed=seed % 7)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(slopefree, "PENTAGON_KERNEL", False)
        slow = separate_pentagons(u, U, k=k, restarts=restarts, seed=seed % 7)
    assert fast == slow


@needs_numba
@SETTINGS
@given(st.integers(0, 10_000), st.integers(4, 12))
def test_property_every_start_ends_at_a_local_minimum(seed, p):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, p); U = rng.uniform(0, 1, (p, p)); U = (U + U.T) / 2; U[np.diag_indices(p)] = u
    Y, d = _sign_matrix(u, U)
    S0, B0 = _pentagon_starts(Y.shape[0], 12, seed % 5)
    S, B, val = fastpent.search(Y, d, S0, B0)
    assert all(_local_minimum(Y, d, S[i], B[i], val[i]) for i in range(12))
