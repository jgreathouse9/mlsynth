"""The slope-free relaxation of the shared-slope control-group problem (#965).

Every claim is checked against enumeration on panels small enough for it: the
constraint table holds at every design, the certificate is at least every design's
value at any sign-correct multipliers, and certify_slope_free reaches the
enumerated positive-slope optimum."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import (Rows, bound, certify_slope_free, design_value, dyadic_below, objective,
                        separate_triangles, size_certificate, solve)

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _designs(p, m):
    for C in combinations(range(p), m):
        u = np.zeros(p, dtype=int); u[list(C)] = 1
        yield set(C), u


def _lifted(u):
    z = np.concatenate([[1], u]).astype(object)
    return np.array([[Q(int(a * b)) for b in z] for a in z], dtype=object)


def _exact(y, X):
    D = ExactData(y, X)
    return D, D.c, D.G


def _ratio(c, G, C):
    u = np.zeros(len(c), dtype=object); u[:] = Q(0); u[list(C)] = Q(1)
    return (c @ u) ** 2 / (u @ G @ u)


# ---------------------------------------------------------------- smoke


def test_smoke_certify_reaches_the_enumerated_optimum():
    y, X = panel(0, p=6)
    out = certify_slope_free(y, X, restarts=10)
    best, C = positive_optimum(y, X)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert set(out.controls) == C


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", range(3))
def test_every_design_satisfies_the_constraint_table(seed):
    """Z = (1, u)(1, u)' meets every row exactly at every design of size m with
    c'u >= 0, triangles included: the relaxation holds the problem."""
    y, X = panel(seed, p=5)
    _, c, G = _exact(y, X)
    for m in range(1, 6):
        tri = [(0, (0, 1, 2)), (1, (1, 3, 4)), (0, (4, 2, 0))]
        R = Rows(c, m, triangles=tri)
        for C, u in _designs(5, m):
            if c @ u.astype(object) < 0:
                continue
            assert R.holds(_lifted(u)), (m, C)


def test_the_cut_excludes_the_incumbent_and_no_other_design():
    y, X = panel(1, p=5)
    _, c, G = _exact(y, X)
    R = Rows(c, 2, cut={0, 3})
    for C, u in _designs(5, 2):
        if c @ u.astype(object) >= 0:
            assert R.holds(_lifted(u)) == (C != {0, 3})


@pytest.mark.parametrize("seed", range(3))
def test_the_bound_is_valid_and_tight_at_the_solvers_multipliers(seed):
    """At the solver's multipliers the certificate is at least every design's value
    and within 1e-6 of the solver's own value, relative to the larger of the two
    and y'y (the objective cc' - lam G is on the scale of (c'u)^2, not of y'y)."""
    y, X = panel(seed, p=6)
    D, c, G = _exact(y, X)
    lam = dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    for m in (2, 3):
        R = Rows(c, m)
        sol = solve(c, G, lam, R)
        b = bound(objective(c, G, lam), R, sol.y_eq, sol.y_in, m)
        A = objective(c, G, lam)
        best = max(design_value(A, u) for C, u in _designs(6, m) if c @ u.astype(object) >= 0)
        assert b >= best
        assert abs(float(b) - sol.value * sol.scale) <= 1e-6 * max(abs(float(b)), float(D.yy))


def test_the_trace_bound_is_one_plus_the_size():
    """tr Z = 1 + sum(diag U) = 1 + m on the relaxation's feasible set, so a matrix
    with one positive eigenvalue is charged (1 + m) times it."""
    y, X = panel(2, p=4)
    _, c, G = _exact(y, X)
    R = Rows(c, 2)
    A = np.full((4, 4), Q(0), dtype=object); A[0, 0] = Q(1)   # S = e_1 e_1' at zero multipliers
    b = bound(A, R, np.zeros(R.n_eq), np.zeros(R.n_in), 2)
    assert Q(3) <= b <= Q(3) * (1 + Q(1, 10 ** 9))   # the eigenvalue bound is rigorous, so a hair above


@pytest.mark.parametrize("seed", range(3))
def test_the_relaxation_certifies_the_optimum_on_small_panels(seed):
    """At the optimum's ratio, with the optimum cut off at its size, every size's
    certificate is at most zero, with triangles where the root needs them."""
    y, X = panel(seed, p=6)
    D, c, G = _exact(y, X)
    _, C = positive_optimum(y, X)
    lam = _ratio(c, G, C)
    for m in range(1, 7):
        st_ = size_certificate(c, G, lam, m, cut=C if len(C) == m else None)
        assert st_.certified, (seed, m, st_)


def test_triangle_separation_finds_a_violated_triangle():
    u = np.array([0.5, 0.5, 0.5]); U = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5]])
    tri = separate_triangles(u, U, k=5)
    assert tri and all(amount > 0 for amount, _, _ in tri)
    kind, (i, j, k) = tri[0][1], tri[0][2]
    # U_ij + U_ik - U_jk <= u_i is violated at i = 1: 0.5 + 0.5 - 0 > 0.5
    assert (kind, i) == (0, 1)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 5), size=st.floats(0.0, 2.0))
def test_property_the_bound_holds_at_any_multipliers(seed, m, size):
    """Any multipliers, inequality ones clipped at zero, give a certificate at least
    every design's value (weak duality with the exact trace bound)."""
    y, X = panel(seed % 40, p=5)
    _, c, G = _exact(y, X)
    lam = dyadic_below(Q(seed % 7, 10))
    R = Rows(c, m, triangles=[(0, (0, 1, 2)), (1, (2, 3, 4))])
    rng = np.random.default_rng(seed)
    y_eq = rng.normal(size=R.n_eq) * size; y_in = rng.normal(size=R.n_in) * size
    A = objective(c, G, lam)
    b = bound(A, R, y_eq, y_in, m)
    best = max((design_value(A, u) for C, u in _designs(5, m) if c @ u.astype(object) >= 0), default=None)
    if best is not None:
        assert b >= best


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_property_certify_matches_enumeration(seed):
    y, X = panel(seed % 60, p=5)
    best, C = positive_optimum(y, X)
    if C is None:
        with pytest.raises(ValueError, match="positive slope"):
            certify_slope_free(y, X, restarts=5)
        return
    out = certify_slope_free(y, X, restarts=5)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(3, 6))
def test_property_triangles_hold_at_every_binary_point(seed, k):
    """Both triangle families hold at U = uu' for every binary u (Padberg 1989)."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=k).astype(float); U = np.outer(u, u)
    assert separate_triangles(u, U, k=50) == []


# ---------------------------------------------------------------- edges and failure


def test_one_candidate_control():
    y, X = panel(3, p=1)
    best, C = positive_optimum(y, X)
    if C is None:
        pytest.skip("this panel's only control has a negative slope")
    out = certify_slope_free(y, X, restarts=3)
    assert out.certified and out.controls == [0]


def test_every_control_in_the_design():
    """m = p has one design; the size is certified at once."""
    y, X = panel(4, p=4)
    _, c, G = _exact(y, X)
    lam = _ratio(c, G, {0, 1, 2, 3}) * Q(2)
    assert size_certificate(c, G, lam, 4).certified


def test_a_negative_inequality_multiplier_is_read_as_zero():
    y, X = panel(5, p=4)
    _, c, G = _exact(y, X)
    R = Rows(c, 2); A = objective(c, G, Q(1, 2))   # 1/2 is dyadic
    y_in = np.ones(R.n_in); neg = y_in.copy(); neg[::3] = -5.0
    zero = y_in.copy(); zero[::3] = 0.0
    assert bound(A, R, np.zeros(R.n_eq), neg, 2) == bound(A, R, np.zeros(R.n_eq), zero, 2)


def test_no_positive_slope_is_refused():
    """Controls that share a factor the treatment moves against: every design's
    slope is negative, so there is no incumbent to certify."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=40)
    X = f[:, None] * 5.0 + rng.normal(size=(40, 4)); y = -5.0 * f
    X, y = X - X.mean(0), y - y.mean()
    assert positive_optimum(y, X)[1] is None
    with pytest.raises(ValueError, match="positive slope"):
        certify_slope_free(y, X, restarts=3)


def test_a_non_dyadic_matrix_is_refused():
    """The eigenvalue bound reads dyadic rationals only; a rational lam such as 1/3
    would be read as a different matrix, so the certificate refuses it."""
    y, X = panel(1, p=4)
    _, c, G = _exact(y, X)
    R = Rows(c, 2)
    with pytest.raises(ValueError, match="dyadic"):
        bound(objective(c, G, Q(1, 3)), R, np.zeros(R.n_eq), np.zeros(R.n_in), 2)


@pytest.mark.parametrize("q", [Q(1, 3), Q(2, 3), Q(1, 10), Q(7, 3), Q(5, 7), Q(3, 1)])
def test_lam_is_rounded_down_to_a_double(q):
    """Below q, dyadic, and within one ulp: rounding to nearest goes above q for
    some of these (1/10, 2/3), which would certify a weaker claim than stated."""
    d = dyadic_below(q)
    assert d <= q and d.denominator & (d.denominator - 1) == 0
    assert q - d <= Q(abs(float(q))) * Q(2) ** -51


def test_some_ratios_round_up_to_nearest():
    assert any(Q(float(q)) > q for q in (Q(1, 3), Q(2, 3), Q(1, 10), Q(7, 3), Q(5, 7)))


def test_a_cut_of_the_wrong_size_is_refused():
    y, X = panel(1, p=4)
    _, c, G = _exact(y, X)
    with pytest.raises(ValueError, match="size"):
        Rows(c, 2, cut={0, 1, 2})


# ---------------------------------------------------------------- branching


def _worse_incumbent(monkeypatch, y, X):
    """Hand the search the worst single-control design with a positive slope."""
    from . import search
    D = ExactData(y, X)
    singles = [j for j in range(X.shape[1]) if D.c[j] > 0]
    worst = min(singles, key=lambda j: D.c[j] ** 2 / D.G[j, j])
    monkeypatch.setattr(search, "incumbent", lambda y, X, **k: (None, {worst}))
    return worst


@pytest.mark.parametrize("seed", range(3))
def test_a_poor_incumbent_is_replaced_by_branching(monkeypatch, seed):
    """At a poor incumbent's ratio the relaxation cannot certify the optimum's size;
    the search branches, finds a better design, and certifies the optimum."""
    y, X = panel(seed, p=5)
    best, C = positive_optimum(y, X)
    worst = _worse_incumbent(monkeypatch, y, X)
    assert {worst} != C
    out = certify_slope_free(y, X, hull=False)
    assert out.certified and set(out.controls) == C and float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert out.nodes > len(out.sizes)                     # some size branched


def test_the_depth_limit_leaves_nodes_open(monkeypatch):
    y, X = panel(0, p=5)
    _worse_incumbent(monkeypatch, y, X)
    out = certify_slope_free(y, X, hull=False, max_depth=0)
    assert not out.certified and out.open


def test_the_hull_stage_rules_out_sizes():
    """On a panel with a strong group the hull stage leaves only a few sizes for
    the relaxation, and the answer is the same as without it."""
    y, X = panel(1, p=6)
    a = certify_slope_free(y, X, restarts=10)
    b = certify_slope_free(y, X, restarts=10, hull=False)
    assert a.certified and b.certified and a.controls == b.controls
    assert len(a.sizes_after_hull) < 6 and b.sizes_after_hull == list(range(1, 7))


def test_triangles_need_three_geos():
    assert separate_triangles(np.array([0.5, 0.5]), np.full((2, 2), 0.25)) == []


def test_a_round_adds_the_separated_triangles(monkeypatch):
    """A size whose root bound is above zero gets the separated triangles in the next
    round's constraints, and is certified when that round's bound is at most zero."""
    from . import slopefree as sf
    y, X = panel(0, p=5)
    _, c, G = _exact(y, X)
    seen, bounds = [], iter([Q(1), Q(-1)])
    real_rows = sf.Rows

    def rows(*a, **k):
        seen.append(list(k.get("triangles", ())))
        return real_rows(*a, **k)

    monkeypatch.setattr(sf, "Rows", rows)
    monkeypatch.setattr(sf, "bound", lambda *a, **k: next(bounds))
    monkeypatch.setattr(sf, "separate_triangles", lambda u, U, k=400: [(0.1, 0, (0, 1, 2))])
    res = sf.size_certificate(c, G, Q(1, 2), 2)
    assert res.certified and res.rounds == 1 and res.triangles == 1
    assert seen == [[], [(0, (0, 1, 2))]]


def test_rounds_stop_at_the_limit(monkeypatch):
    from . import slopefree as sf
    y, X = panel(0, p=5)
    _, c, G = _exact(y, X)
    monkeypatch.setattr(sf, "bound", lambda *a, **k: Q(1))
    calls = iter(range(100))
    monkeypatch.setattr(sf, "separate_triangles", lambda u, U, k=400: [(0.1, 0, (0, 1, next(calls) % 3 + 2))])
    res = sf.size_certificate(c, G, Q(1, 2), 2, rounds=2)
    assert not res.certified and res.rounds == 2 and res.triangles == 2


def test_the_hull_stage_keeps_the_optimums_size_against_a_poor_incumbent(monkeypatch):
    """The hull stage is handed the incumbent's exact RSS as its upper bound; a
    smaller one would let it discard the size that holds the optimum."""
    y, X = panel(2, p=6)
    best, C = positive_optimum(y, X)
    _worse_incumbent(monkeypatch, y, X)
    out = certify_slope_free(y, X)
    assert len(C) in out.sizes_after_hull
    assert out.certified and set(out.controls) == C
