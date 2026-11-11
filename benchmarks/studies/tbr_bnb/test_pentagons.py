"""Pentagonal rows in the slope-free relaxation (#965).

With s_0 = 1 and s_v = 2 u_v - 1 every design is a vector of signs, and for any five
of its indices S and signs b the sum of the five odd terms b_x s_x is odd, so its
square is at least 1: sum_{x<z} b_x b_z s_x s_z >= -2, the pentagonal (hypermetric)
inequality of the cut polytope. In the lift it is a linear row in Z, valid at every
design, so a size's relaxation may carry it like a triangle and its multiplier enters
the exact certificate like any other row's. separate_pentagons finds the rows a
relaxed point violates; size_certificate adds them from round PENTAGONS_FROM.

The checks here are of the row (valid at every design, tight at some), of its place
among Rows' rows (the matrices are the table's), of the separation (it reports what
the point violates and nothing at a mixture of designs), and of the switches."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import progress, slopefree
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import (PENTAGON, Rows, _ratio, certify_slope_free, pentagon_row, separate_pentagons,
                        size_certificate)

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _design(p, S):
    """Z = zz' of the design S (control indices), z = (1, u)."""
    z = np.zeros(p + 1, dtype=object); z[:] = 0; z[0] = 1
    for j in S:
        z[j + 1] = 1
    return np.outer(z, z)


def _value(coef, Z):
    return sum((v * Z[a, b] for (a, b), v in coef.items()), Q(0))


def _pentagon_point():
    """A point of four controls whose sign matrix Y has Y_ii = 1 and Y_ij = -1/4: it is
    positive semidefinite and meets every triangle row (three entries sum to -3/4 >= -1),
    and with S all five indices and b all ones b'Yb = 5 - 5 = 0 < 1, so the pentagonal
    row is violated by (1 - 0) / 2 = 1/2."""
    n = 5
    Y = np.full((n, n), -0.25); np.fill_diagonal(Y, 1.0)
    Z = np.empty((n, n))
    Z[0, 0] = 1.0
    for v in range(1, n):
        Z[0, v] = Z[v, 0] = (Y[0, v] + 1) / 2
    for v in range(1, n):
        for w in range(1, n):
            Z[v, w] = (Y[v, w] + Y[0, v] + Y[0, w] + 1) / 4
    return Z


def _open_size():
    """A size held open at every round: the size's own best ratio less a little, and no
    cut, so the best design is feasible and every bound stays above zero."""
    y, X = panel(3, p=8)
    D = ExactData(y, X); c, G = D.c, D.G
    m = 4
    best = max(_ratio(c, G, set(C))[1] for C in itertools.combinations(range(8), m))
    return c, G, best * Q(999, 1000), m


# ---------------------------------------------------------------- smoke


@pytest.mark.parametrize("solver", ["clarabel", "workspace"])
def test_smoke_a_search_with_pentagons_returns_the_enumerated_optimum(solver):
    y, X = panel(1, p=8)
    best, C = positive_optimum(y, X)
    out = certify_slope_free(y, X, restarts=1, deep_restarts=200, solver=solver, pentagons=True)
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


# ---------------------------------------------------------------- the row


def test_a_pentagonal_row_holds_at_every_design_and_is_tight_at_one():
    p = 6
    for S in [(0, 1, 2, 3, 4), (1, 2, 4, 5, 6), (0, 2, 3, 5, 6)]:
        for b in itertools.product((1, -1), repeat=5):
            coef, rhs = pentagon_row(S, b)
            values = [_value(coef, _design(p, D)) for r in range(p + 1)
                      for D in itertools.combinations(range(p), r)]
            assert max(values) <= rhs and max(values) == rhs


def test_the_row_and_its_negated_signs_are_the_same_row():
    S, b = (0, 2, 3, 5, 7), (1, -1, 1, 1, -1)
    assert pentagon_row(S, b) == pentagon_row(S, tuple(-v for v in b))


def test_the_row_is_the_sign_form_written_on_z():
    """-sum b_x b_z Y_xz <= 2 with Y_0t = 2 Z_0t - 1 and Y_vw = 4 Z_vw - 2 Z_0v - 2 Z_0w + 1."""
    S, b = (0, 1, 2, 3, 4), (1, 1, 1, 1, 1)
    coef, rhs = pentagon_row(S, b)
    assert coef[(1, 2)] == -4 and coef[(0, 1)] == -2 + 2 * 3 and rhs == 2 - (4 - 6)


def test_a_pentagonal_row_sits_among_the_triangles_and_the_matrices_are_the_tables():
    rng = np.random.default_rng(1)
    p, m = 8, 3
    c = [int(v) for v in rng.integers(-5, 6, p)]
    pent = ((0, 1, 3, 5, 7), (1, -1, 1, 1, -1))
    tri = [(0, (0, 1, 2)), (PENTAGON, pent), (1, (3, 4, 5))]
    R = Rows(c, m, triangles=tri, box=True)
    lo, hi = R.triangle_rows
    assert hi - lo == 3 and R.ineq[lo + 1] == pentagon_row(*pent)[0] and R.bin[lo + 1] == pentagon_row(*pent)[1]
    n = R.n
    _, Mi = R.matrices(1.0)
    T = np.zeros((R.n_in, n * n))
    for i, r in enumerate(R.ineq):
        for (a, bb), v in r.items():
            T[i, a + n * bb] += float(v)
    assert np.array_equal(Mi.toarray(), T)


# ---------------------------------------------------------------- separation


def test_the_separation_finds_the_violated_pentagon():
    Z = _pentagon_point()
    out = separate_pentagons(Z[0, 1:], Z[1:, 1:])
    assert out
    viol, kind, (S, b) = out[0]
    assert kind == PENTAGON and S == (0, 1, 2, 3, 4) and b == (1, 1, 1, 1, 1)
    assert viol == pytest.approx(0.5)


def test_every_reported_row_is_violated_by_what_it_reports_and_comes_in_order():
    Z = _many_pentagons()
    out = separate_pentagons(Z[0, 1:], Z[1:, 1:], k=50)
    assert out
    viols = [v for v, _, _ in out]
    assert viols == sorted(viols, reverse=True) and len(out) <= 50
    for v, kind, (S, b) in out:
        coef, rhs = pentagon_row(S, b)
        lhs = sum(val * Z[a, bb] for (a, bb), val in coef.items())
        assert kind == PENTAGON and lhs - rhs == pytest.approx(v, abs=1e-9) and v > 0
        assert list(S) == sorted(set(S)) and len(S) == 5 and b[0] == 1
    assert len({key for _, _, key in out}) == len(out)


def test_the_separation_is_deterministic():
    Z = _pentagon_point()
    assert separate_pentagons(Z[0, 1:], Z[1:, 1:], seed=3) == separate_pentagons(Z[0, 1:], Z[1:, 1:], seed=3)


def test_a_design_violates_no_pentagon():
    Z = np.asarray(_design(7, (0, 2, 5)), dtype=float)
    assert separate_pentagons(Z[0, 1:], Z[1:, 1:]) == []


def _many_pentagons():
    """Eight geos whose sign matrix has Y_ij = -1/4 off the diagonal: every five-set with
    equal signs is violated by 1/2, and each is where the local search stops, so the
    point violates C(9, 5) = 126 pentagonal rows and the search finds dozens."""
    W = np.full((9, 9), -0.25); np.fill_diagonal(W, 1.0)
    Z = np.empty_like(W); Z[0, 0] = 1.0
    Z[0, 1:] = Z[1:, 0] = (W[0, 1:] + 1) / 2
    Z[1:, 1:] = (W[1:, 1:] + W[0, 1:][:, None] + W[0, 1:][None, :] + 1) / 4
    return Z


def test_the_separation_keeps_the_k_most_violated_rows():
    Z = _many_pentagons()
    every = separate_pentagons(Z[0, 1:], Z[1:, 1:], k=400)
    three = separate_pentagons(Z[0, 1:], Z[1:, 1:], k=3)
    assert len(every) > 3 and three == every[:3]


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_fewer_than_five_indices_have_no_pentagon(p):
    Z = np.eye(p + 1)
    assert separate_pentagons(Z[0, 1:], Z[1:, 1:]) == []


# ---------------------------------------------------------------- in the size's rounds


@pytest.fixture
def stubs(monkeypatch):
    """Separations that always find one new row, so every round of an open size runs:
    a fresh triangle and a fresh pentagon each call (every one valid at every design)."""
    calls = {"pentagons": 0}
    tri = iter(itertools.combinations(range(8), 3))
    five = iter(itertools.combinations(range(9), 5))

    def triangles(u, U, k=400, tol=1e-6):
        return [(0.1, 1, next(tri))]

    def pentagons(u, U, k=400, **kw):
        calls["pentagons"] += 1
        return [(0.5, PENTAGON, (next(five), (1, 1, 1, 1, 1)))]

    monkeypatch.setattr(slopefree, "separate_triangles", triangles)
    monkeypatch.setattr(slopefree, "separate_pentagons", pentagons)
    return calls


@pytest.fixture
def rounds_seen(monkeypatch):
    """The pentagonal rows each Rows of a size's relaxation rounds carries."""
    seen = []
    real = slopefree.Rows

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(sum(1 for kind, _ in self.triangles if kind == PENTAGON))

    monkeypatch.setattr(slopefree, "Rows", Spy)
    return seen


def test_pentagons_enter_from_round_pentagons_from(rounds_seen, stubs):
    c, G, lam, m = _open_size()
    rounds_seen.clear()
    res = size_certificate(c, G, lam, m, schedules=(), rounds=3, solver="clarabel", pentagons=True)
    assert not res.certified
    # the first-order tier's rows, then rounds 0 to 3; separated after round 1 and round 2
    assert rounds_seen == [0, 0, 0, 1, 2] and stubs["pentagons"] == 2


def test_the_round_pentagons_start_from_is_read_at_the_call(rounds_seen, stubs, monkeypatch):
    monkeypatch.setattr(slopefree, "PENTAGONS_FROM", 0)
    c, G, lam, m = _open_size()
    rounds_seen.clear()
    size_certificate(c, G, lam, m, schedules=(), rounds=3, solver="clarabel", pentagons=True)
    assert rounds_seen == [0, 0, 1, 2, 3] and stubs["pentagons"] == 3


def test_without_pentagons_no_pentagonal_row_is_built(rounds_seen, stubs):
    c, G, lam, m = _open_size()
    rounds_seen.clear()
    size_certificate(c, G, lam, m, schedules=(), rounds=3, solver="clarabel", pentagons=False)
    assert stubs["pentagons"] == 0 and rounds_seen == [0] * 5


def test_pentagons_default_to_the_module_setting_read_at_the_call(stubs, monkeypatch):
    c, G, lam, m = _open_size()
    monkeypatch.setattr(slopefree, "PENTAGONS", False)
    size_certificate(c, G, lam, m, schedules=(), rounds=2, solver="clarabel")
    assert stubs["pentagons"] == 0
    monkeypatch.setattr(slopefree, "PENTAGONS", True)
    size_certificate(c, G, lam, m, schedules=(), rounds=2, solver="clarabel")
    assert stubs["pentagons"] == 1


def test_a_round_reports_its_pentagons(stubs):
    c, G, lam, m = _open_size()
    seen = []
    with progress.reporting(progress=seen.append):
        size_certificate(c, G, lam, m, schedules=(), rounds=3, solver="clarabel", pentagons=True)
    ev = [e for e in seen if e.kind == "pentagons"]
    assert [e.fields["round"] for e in ev] == [1, 2]
    assert all(e.fields["m"] == m and e.fields["pentagons"] == 1 and e.fields["seconds"] >= 0 for e in ev)


def test_the_search_passes_its_switch_to_every_size(monkeypatch):
    seen = []
    real = slopefree.size_certificate

    def spy(*a, **k):
        seen.append(k.get("pentagons"))
        return real(*a, **k)

    monkeypatch.setattr(slopefree, "size_certificate", spy)
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)           # bound every node: none is priced
    y, X = panel(1, p=8)
    certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False, pentagons=False)
    assert seen and set(seen) == {False}
    seen.clear()
    certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False, pentagons=True)
    assert seen and set(seen) == {True}


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("S", [(0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 1, 2, 3, -1), (0, 1, 2, 3, 4.0), "01234"])
def test_a_row_needs_five_distinct_indices(S):
    with pytest.raises(MlsynthConfigError, match="five"):
        pentagon_row(S, (1, 1, 1, 1, 1))


@pytest.mark.parametrize("b", [(1, 1, 1, 1), (1, 1, 1, 1, 0), (1, 1, 1, 1, 2), (1, 1, 1, 1, True)])
def test_a_row_needs_five_signs(b):
    with pytest.raises(MlsynthConfigError, match="signs"):
        pentagon_row((0, 1, 2, 3, 4), b)


def test_rows_refuse_a_malformed_pentagon():
    with pytest.raises(MlsynthConfigError, match="five"):
        Rows([1, 2, 3, 4, 5], 2, triangles=[(PENTAGON, ((0, 1, 2), (1, 1, 1)))])


@pytest.mark.parametrize("idx", [((0, 1, 2, 3, 4),), [(0, 1, 2, 3, 4), (1, 1, 1, 1, 1)], (0, 1, 2, 3, 4)])
def test_rows_refuse_a_pentagon_index_that_is_not_a_pair(idx):
    with pytest.raises(MlsynthConfigError, match="five indices, five signs"):
        Rows([1, 2, 3, 4, 5], 2, triangles=[(PENTAGON, idx)])


def test_rows_refuse_a_pentagon_past_the_controls():
    with pytest.raises(MlsynthConfigError, match="index"):
        Rows([1, 2, 3, 4, 5], 2, triangles=[(PENTAGON, ((0, 1, 2, 3, 6), (1, 1, 1, 1, 1)))])


@pytest.mark.parametrize("kw", [dict(k=0), dict(k=1.5), dict(restarts=0), dict(restarts="9"), dict(seed=-1)])
def test_the_separation_refuses_bad_settings(kw):
    Z = _pentagon_point()
    with pytest.raises(MlsynthConfigError):
        separate_pentagons(Z[0, 1:], Z[1:, 1:], **kw)


@pytest.mark.parametrize("bad", [1, "yes", 0.0])
def test_the_switch_is_a_bool(bad):
    y, X = panel(1, p=8)
    with pytest.raises(MlsynthConfigError, match="pentagons"):
        certify_slope_free(y, X, restarts=1, pentagons=bad)
    c, G, lam, m = _open_size()
    with pytest.raises(MlsynthConfigError, match="pentagons"):
        size_certificate(c, G, lam, m, pentagons=bad)


def test_a_bad_module_start_round_is_refused(monkeypatch):
    monkeypatch.setattr(slopefree, "PENTAGONS_FROM", -1)
    c, G, lam, m = _open_size()
    with pytest.raises(MlsynthConfigError, match="PENTAGONS_FROM"):
        size_certificate(c, G, lam, m, schedules=(), rounds=2, solver="clarabel", pentagons=True)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(st.integers(5, 9).flatmap(lambda n: st.tuples(
    st.just(n), st.lists(st.integers(0, n - 1), min_size=5, max_size=5, unique=True),
    st.lists(st.sampled_from([1, -1]), min_size=5, max_size=5),
    st.lists(st.booleans(), min_size=n - 1, max_size=n - 1))))
def test_property_every_design_meets_every_pentagonal_row(case):
    n, S, b, u = case
    coef, rhs = pentagon_row(tuple(S), tuple(b))
    Z = _design(n - 1, [j for j, x in enumerate(u) if x])
    assert _value(coef, Z) <= rhs


@SETTINGS
@given(st.integers(5, 8).flatmap(lambda p: st.tuples(
    st.just(p), st.lists(st.lists(st.booleans(), min_size=p, max_size=p), min_size=1, max_size=6),
    st.lists(st.floats(0.05, 1.0), min_size=6, max_size=6))))
def test_property_a_mixture_of_designs_violates_no_pentagon(case):
    p, designs, w = case
    w = np.array(w[:len(designs)]); w = w / w.sum()
    Z = sum(wi * np.asarray(_design(p, [j for j, x in enumerate(d) if x]), dtype=float) for wi, d in zip(w, designs))
    assert separate_pentagons(Z[0, 1:], Z[1:, 1:], restarts=200) == []


@SETTINGS
@given(st.integers(6, 9).flatmap(lambda p: st.lists(st.tuples(
    st.lists(st.integers(0, p), min_size=5, max_size=5, unique=True),
    st.lists(st.sampled_from([1, -1]), min_size=5, max_size=5)), min_size=1, max_size=4).map(lambda L: (p, L))))
def test_property_the_matrices_are_the_tables_with_pentagons(case):
    p, pents = case
    tri = [(PENTAGON, (tuple(sorted(S)), tuple(b))) for S, b in pents]
    R = Rows(list(range(1, p + 1)), 3, triangles=tri)
    n = R.n
    _, Mi = R.matrices(1.0)
    T = np.zeros((R.n_in, n * n))
    for i, r in enumerate(R.ineq):
        for (a, bb), v in r.items():
            T[i, a + n * bb] += float(v)
    assert np.array_equal(Mi.toarray(), T)
