"""The start group certified in its turn, and the exact data built from integers (#965).

The start group's local design is the first incumbent, without a certificate: a
design is an incumbent whatever proves it, and the certificate that matters is the
group's own, which it then gets in its turn, against the incumbent of that turn,
like every other group. On a panel where the pass finds a better design, the start
group is certified against that design's ceiling, which closes it sooner than its
own optimum does.

ExactData's Gram, cross products and y'y are integers over one power of two: a
finite double is an integer over a power of two, so the products of the doubles'
integers, over the square of the largest denominator, are the exact rationals that
summing Fractions gives."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import treatment
from .brute import geo_panel, mirror_panel, raw_panel, relative_optimum
from .certificate import ExactData, qarr
from .treatment import _split, certify_treatment

N_TEST = 8
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
REL = 1e-9


def _worst_group(M, k=2):
    """The treatment group whose own best key is lowest."""
    p = M.shape[1]
    from mlsynth.utils.tbr_helpers.design.objective import score_split
    own = {}
    for T in combinations(range(p), k):
        y = M[:, list(T)].sum(axis=1)
        rest = [j for j in range(p) if j not in T]
        own[T] = max(score_split(y, M[:, list(C)].sum(axis=1), objective="relative", n_test=N_TEST).key[0]
                     for r in range(1, len(rest) + 1) for C in combinations(rest, r))
    return list(min(own, key=own.get))


def _spy_ceilings(monkeypatch):
    real = treatment._optimum
    seen = []

    def spy(Mc, M, T, n_test, restarts, ceiling, floors, starts, until=None, cols=None):
        seen.append(ceiling)
        return real(Mc, M, T, n_test, restarts, ceiling, floors, starts, until=until, cols=cols)

    monkeypatch.setattr(treatment, "_optimum", spy)
    return seen


# ---------------------------------------------------------------- smoke


def test_smoke_a_poor_start_still_certifies_the_optimum():
    M = raw_panel(5)
    best, T, _ = relative_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst_group(M), restarts=5, blank_periods=0)
    assert out.certified and set(out.treatment) == T
    assert out.key == pytest.approx(best, rel=REL)


# ---------------------------------------------------------------- the start group's turn


def test_every_certification_has_an_incumbent(monkeypatch):
    """The first incumbent is the library search's own design, so every group is
    certified against a ceiling, the start group included."""
    M = raw_panel(5)
    seen = _spy_ceilings(monkeypatch)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=0, blank_periods=0)
    assert out.certified and seen and all(c is not None for c in seen)


def test_the_first_incumbent_is_the_library_design():
    """With no pass, the certified design is at least the library search's: the
    incumbent only improves."""
    M = raw_panel(5)
    T0, C = treatment._start(M, 2, N_TEST, 3)
    from mlsynth.utils.tbr_helpers.design.objective import score_split
    key0 = score_split(M[:, T0].sum(axis=1), M[:, C].sum(axis=1), objective="relative", n_test=N_TEST).key[0]
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=0, blank_periods=0)
    assert out.key >= key0 * (1 - REL)


def test_the_start_group_is_certified_against_a_better_incumbent():
    """The pass finds the optimum from the worst start, so the start group, if it is
    not screened out, is certified against that design's ceiling."""
    M = raw_panel(5)
    start = _worst_group(M)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10, blank_periods=0)
    mine = [s for s in out.survivors if s.get("start")]
    theirs = [s for s in out.survivors if s["treatment"] == sorted(start)]
    assert mine == theirs and len(mine) <= 1
    assert all(s["ceiling"] is not None for s in mine)
    assert set(out.treatment) != set(start)


def test_the_start_group_is_searched_once(monkeypatch):
    M = raw_panel(5)
    start = _worst_group(M)
    real, calls = treatment._local, []
    monkeypatch.setattr(treatment, "_local",
                        lambda Mc, M_, T, *a: calls.append(tuple(sorted(T))) or real(Mc, M_, T, *a))
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10, blank_periods=0)
    assert calls.count(tuple(sorted(start))) == 1
    assert out.searched == len(calls) - 1                 # the start's search is not the pass's


def test_the_optimum_group_as_the_start_is_searched_once_and_marked(monkeypatch):
    M = raw_panel(5)
    _, T, _ = relative_optimum(M, 2, N_TEST)
    real, calls = treatment._local, []
    monkeypatch.setattr(treatment, "_local",
                        lambda Mc, M_, T_, *a: calls.append(tuple(sorted(T_))) or real(Mc, M_, T_, *a))
    out = certify_treatment(M, 2, n_test=N_TEST, start=sorted(T), restarts=3, pass_restarts=10, blank_periods=0)
    assert calls.count(tuple(sorted(T))) == 1
    mine = [s for s in out.survivors if s.get("start")]
    assert len(mine) == 1 and set(mine[0]["treatment"]) == T
    assert out.certified and set(out.treatment) == T


def test_the_start_groups_seeds_are_its_local_designs(monkeypatch):
    """The start group's certification starts each sign from the design the descent
    found for that sign before the pass."""
    M = raw_panel(5)
    start = tuple(_worst_group(M))
    got = []
    real = treatment._optimum

    def spy(Mc, M_, T, n_test, restarts, ceiling, floors, starts, until=None, cols=None):
        if tuple(T) == start:
            got.append(starts)
        return real(Mc, M_, T, n_test, restarts, ceiling, floors, starts, until=until, cols=cols)

    monkeypatch.setattr(treatment, "_optimum", spy)
    Mc = M - M.mean(axis=0)
    want = treatment._local(Mc, M, start, N_TEST, 3)[0]
    out = certify_treatment(M, 2, n_test=N_TEST, start=list(start), restarts=3, pass_restarts=0, blank_periods=0)
    assert out.certified and got in ([], [want])          # screened out, or seeded with both signs
    assert got or not any(s.get("start") for s in out.survivors)


@SETTINGS
@given(seed=st.integers(0, 300), pick=st.integers(0, 20), mirror=st.booleans())
def test_property_any_start_certifies_the_optimum(seed, pick, mirror):
    M = mirror_panel(seed, p=6) if mirror else raw_panel(seed, p=6)
    groups = list(combinations(range(6), 2))
    start = list(groups[pick % len(groups)])
    best, _, _ = relative_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=3, blank_periods=0)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)
    assert out.screened_out + len(out.survivors) == out.groups


# ---------------------------------------------------------------- edges


def test_a_start_whose_designs_all_fit_exactly_leaves_no_incumbent(monkeypatch):
    """When the start group has no design the library scores, the first
    certification has no ceiling, and the search still certifies the optimum."""
    M = raw_panel(5)
    start = _worst_group(M)
    real = treatment._local

    def none_for_start(Mc, M_, T, *a):
        starts, d = real(Mc, M_, T, *a)
        return starts, (None if sorted(T) == sorted(start) else d)

    monkeypatch.setattr(treatment, "_local", none_for_start)
    seen = _spy_ceilings(monkeypatch)
    best, T, _ = relative_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0, blank_periods=0)
    assert seen[0] is None
    assert out.certified and set(out.treatment) == T


# ---------------------------------------------------------------- the exact data in integers


def _reference(y, X):
    yq, Xq = qarr(y), qarr(X)
    return Xq.T @ Xq, Xq.T @ yq, yq @ yq


def _check(y, X):
    D = ExactData(y, X)
    G, c, yy = _reference(y, X)
    assert D.yy == yy and isinstance(D.yy, Q)
    assert all(a == b for a, b in zip(D.c.tolist(), c.tolist()))
    assert all(a == b for a, b in zip(D.G.ravel().tolist(), G.ravel().tolist()))
    assert D.G.dtype == object and all(isinstance(v, Q) for v in D.G.ravel())


def test_the_exact_data_on_a_geo_panel():
    M = geo_panel(2, p=7)
    y, X, _ = _split(M, (1, 4))
    _check(y, X)


@SETTINGS
@given(seed=st.integers(0, 10_000), n=st.integers(1, 12), p=st.integers(1, 6), e=st.integers(-40, 40))
def test_property_the_exact_data_equal_the_fractions(seed, n, p, e):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * 2.0 ** e * rng.choice([1e-12, 1.0, 3e9], size=(n, p))
    X[rng.random((n, p)) < 0.2] = 0.0
    y = rng.normal(size=n) * 2.0 ** (-e)
    _check(y, X)


def test_the_exact_data_keep_the_rationals_for_rss():
    M = geo_panel(3, p=6)
    y, X, _ = _split(M, (0, 2))
    D = ExactData(y, X)
    yq, Xq = qarr(y), qarr(X)
    x = Xq[:, [0, 2]].sum(axis=1)
    r = yq - Q(0.75) * x
    assert D.rss({0, 2}, 0.75) == r @ r


def test_all_zero_data():
    _check(np.zeros(4), np.zeros((4, 2)))


def test_no_periods():
    D = ExactData(np.zeros(0), np.zeros((0, 3)))
    assert D.yy == 0 and D.G.shape == (3, 3) and D.c.shape == (3,)
    assert all(v == 0 and isinstance(v, Q) for v in list(D.G.ravel()) + list(D.c))


def test_one_control():
    M = geo_panel(4, p=3)
    y, X, _ = _split(M, (0, 1))
    _check(y, X)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_non_finite_data_are_refused(bad):
    X = np.ones((5, 3)); X[2, 1] = bad
    with pytest.raises(MlsynthDataError, match="finite"):
        ExactData(np.ones(5), X)
    with pytest.raises(MlsynthDataError, match="finite"):
        ExactData(np.array([1.0, bad, 0.0]), np.ones((3, 2)))
