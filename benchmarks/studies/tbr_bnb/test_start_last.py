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
from .brute import geo_panel, treatment_optimum
from .certificate import ExactData, qarr
from .gated import certify_gated
from .treatment import _split, certify_treatment

N_TEST = 8
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _worst_passing(M, k=2):
    own = {}
    for t in combinations(range(M.shape[1]), k):
        res = certify_gated(*_split(M, t)[:2], n_test=N_TEST, restarts=3)
        if res.passing:
            own[t] = res.key
    return list(min(own, key=own.get))


def _spy_ceilings(monkeypatch):
    real = treatment.certify_gated
    seen = []

    def spy(y, X, *a, ceiling=None, **kw):
        seen.append(ceiling)
        return real(y, X, *a, ceiling=ceiling, **kw)

    monkeypatch.setattr(treatment, "certify_gated", spy)
    return seen


# ---------------------------------------------------------------- smoke


def test_smoke_a_poor_start_still_certifies_the_key_optimum():
    M = geo_panel(5)
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst_passing(M), restarts=5)
    assert out.certified and set(out.treatment) == T
    assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)


# ---------------------------------------------------------------- the start group's turn


def test_every_certification_has_an_incumbent_when_the_start_passes(monkeypatch):
    """Without a given start, the first incumbent is the library search's own design,
    which passes on this panel, so every group is certified against a ceiling, the
    start group included."""
    M = geo_panel(5)
    T0, C = treatment._start(M, 2, N_TEST, 3)
    assert treatment._design(M, T0, C, N_TEST)[1] is not None
    seen = _spy_ceilings(monkeypatch)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=0)
    assert out.certified and seen and all(c is not None for c in seen)


def test_the_first_incumbent_is_the_library_design():
    """With no pass and no group that beats it, the certified design is at least the
    library search's: the incumbent only improves."""
    M = geo_panel(5)
    T0, C = treatment._start(M, 2, N_TEST, 3)
    _, key0, _, _ = treatment._design(M, T0, C, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=0)
    assert out.key >= key0


def test_the_start_group_is_certified_against_a_better_incumbent():
    """The pass finds the key's optimum from the worst passing start, so the start
    group, if it is not screened out, is certified against that design's ceiling."""
    M = geo_panel(5)
    start = _worst_passing(M)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10)
    mine = [s for s in out.survivors if s.get("start")]
    theirs = [s for s in out.survivors if s["treatment"] == sorted(start)]
    assert mine == theirs and len(mine) <= 1               # certified once, and marked, or screened out
    assert all(s["ceiling"] is not None for s in mine)
    assert set(out.treatment) != set(start)


def test_the_start_group_is_searched_once(monkeypatch):
    M = geo_panel(5)
    start = _worst_passing(M)
    real, calls = treatment._local, []
    monkeypatch.setattr(treatment, "_local", lambda M_, T, *a: calls.append(tuple(sorted(T))) or real(M_, T, *a))
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10)
    assert calls.count(tuple(sorted(start))) == 1
    assert out.searched == len(calls) - 1                 # the start's search is not the pass's


def test_the_start_groups_seed_is_its_local_design(monkeypatch):
    """The start group's certification starts from the design found for it before
    the pass, as every other group's starts from its pass design."""
    real = treatment.certify_gated
    seeds = []
    M = geo_panel(5)
    start = _worst_passing(M)
    y0 = M[:, start].sum(axis=1)

    def spy(y, X, *a, start=None, **kw):
        if np.array_equal(y, y0):
            seeds.append(start)
        return real(y, X, *a, start=start, **kw)

    monkeypatch.setattr(treatment, "certify_gated", spy)
    C0 = treatment._local(M, start, N_TEST, 3)[0]
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert out.certified and seeds in ([], [set(C0)])     # screened out, or seeded
    assert seeds or not any(s.get("start") for s in out.survivors)


@SETTINGS
@given(seed=st.integers(0, 300), pick=st.integers(0, 20))
def test_property_any_start_certifies_the_key_optimum(seed, pick):
    M = geo_panel(seed, p=6)
    groups = list(combinations(range(6), 2))
    start = list(groups[pick % len(groups)])
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=3)
    assert out.certified
    if key is None:
        assert not out.passing
    else:
        assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)
    assert out.screened_out + len(out.survivors) == out.groups


# ---------------------------------------------------------------- edges


def test_a_start_with_no_passing_design_leaves_no_incumbent(monkeypatch):
    """When the start group's local design fails the gates, the first certification
    has no incumbent, as before the pass, and the search still certifies the
    optimum."""
    M = geo_panel(5)
    start = _worst_passing(M)
    real = treatment._local

    def failing(M_, T, *a):
        C, key, rss, rest = real(M_, T, *a)
        return C, (None if sorted(T) == sorted(start) else key), rss, rest

    monkeypatch.setattr(treatment, "_local", failing)
    seen = _spy_ceilings(monkeypatch)
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert seen[0] is None
    assert out.certified and set(out.treatment) == T


def test_a_library_design_that_fails_the_gates_is_not_the_incumbent(monkeypatch):
    """The library's design is the first incumbent only when it passes: one that fails
    a gate leaves the first certification without a ceiling."""
    from .brute import treatment_optimum as _opt
    from mlsynth.utils.tbr_helpers.design.objective import score_split

    M = geo_panel(5)
    failing = None
    for T in combinations(range(M.shape[1]), 2):
        rest = [j for j in range(M.shape[1]) if j not in T]
        for r in range(1, len(rest) + 1):
            for C in combinations(rest, r):
                sc = score_split(M[:, list(T)].sum(axis=1), M[:, list(C)].sum(axis=1), objective="reference",
                                 n_test=N_TEST)
                if not all(sc.key[:4]):
                    failing = (list(T), list(C))
                    break
            if failing:
                break
        if failing:
            break
    assert failing is not None
    assert treatment._design(M, *failing, N_TEST)[1] is None
    monkeypatch.setattr(treatment, "_start", lambda M_, k, n_test, restarts: failing)
    seen = _spy_ceilings(monkeypatch)
    key, T, _ = _opt(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=0)
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
