"""The search over treatment groups under the relative key (#965, step 9).

Every treatment group of the size gets an exact floor on the RSS of its designs, one
per slope sign; a group whose floors are both at or above its ceiling, the RSS below
which a design of the group could beat the incumbent's relative key, is ruled out,
and every other group has each sign that can compete certified against the ceiling
(slopefree.certify_slope_free). Every claim is checked against enumeration of all
treatment and control groups with the library's own key."""
from fractions import Fraction as Q
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import score_split

from . import slopefree, treatment
from .brute import geo_panel, mirror_panel, raw_panel, relative_optimum, wide_panel
from .certificate import ExactData
from .treatment import (_split, certify_treatment, design_moments, geo_sums, gram_floor, group_level,
                        relative_bar, relative_ceiling, rss_floor, square_sum)

SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8
REL = 1e-9


def _designs(M, T, sign):
    """The exact RSS of every control group of T whose slope has the sign ``sign``."""
    y = M[:, sorted(T)].sum(axis=1)
    rest = [j for j in range(M.shape[1]) if j not in T]
    D = ExactData(y, M[:, rest])
    for r in range(1, len(rest) + 1):
        for C in combinations(range(len(rest)), r):
            u = np.zeros(len(rest), dtype=object); u[list(C)] = 1
            cu = D.c @ u
            if sign * cu > 0:
                yield D.yy - cu * cu / (u @ D.G @ u)


def _own(M, k=2):
    """Each treatment group's own best relative key, by enumeration."""
    p = M.shape[1]
    own = {}
    for T in combinations(range(p), k):
        y = M[:, list(T)].sum(axis=1)
        rest = [j for j in range(p) if j not in T]
        own[T] = max(score_split(y, M[:, list(C)].sum(axis=1), objective="relative", n_test=N_TEST).key[0]
                     for r in range(1, len(rest) + 1) for C in combinations(rest, r))
    return own


def _worst(M, k=2):
    own = _own(M, k)
    return list(min(own, key=own.get))


def _centred(M):
    return M - M.mean(axis=0)


# ---------------------------------------------------------------- smoke


def test_smoke_the_certified_design_is_the_optimum():
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, blank_periods=0)
    best, T, C = relative_optimum(M, 2, N_TEST)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)
    assert set(out.treatment) == T and set(out.controls) == C


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed,k", [(0, 1), (0, 2), (1, 3), (4, 2), (5, 1), (5, 3)])
def test_the_optimum_on_small_panels(seed, k):
    M = raw_panel(seed)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=5, blank_periods=0)
    best, _, _ = relative_optimum(M, k, N_TEST)
    assert out.certified and out.key == pytest.approx(best, rel=REL)


def test_every_group_is_screened_or_certified():
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, blank_periods=0)
    assert out.groups == comb(M.shape[1], 2)
    assert out.screened_out + len(out.survivors) == out.groups
    assert all(s["certified"] for s in out.survivors)


def test_the_screen_rules_out_groups():
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, blank_periods=0)
    assert out.screened_out > 0


def test_a_better_group_replaces_a_poor_start():
    """Started from the group whose own optimum has the smallest key, with no pass,
    the search ends at the optimum: a survivor that beats the incumbent replaces it."""
    M = raw_panel(5)
    best, T, _ = relative_optimum(M, 2, N_TEST)
    worst = _worst(M)
    assert set(worst) != T
    out = certify_treatment(M, 2, n_test=N_TEST, start=worst, restarts=5, pass_restarts=0, blank_periods=0)
    assert out.certified and set(out.treatment) == T
    assert out.key == pytest.approx(best, rel=REL)
    assert out.replaced >= 1


@given(seed=st.integers(0, 10_000), k=st.integers(1, 3), t=st.integers(0, 20), sign=st.sampled_from([1, -1]),
       mirror=st.booleans())
@SETTINGS
def test_property_each_signs_floor_is_below_its_designs(seed, k, t, sign, mirror):
    M = _centred(mirror_panel(seed % 300, p=6) if mirror else raw_panel(seed % 300, p=6))
    groups = list(combinations(range(6), k))
    T = groups[t % len(groups)]
    floor = rss_floor(M, T, gram_floor(M), sign=sign)
    assert floor >= 0
    assert all(floor <= rss for rss in _designs(M, T, sign))


@given(seed=st.integers(0, 10_000), wide=st.booleans())
@SETTINGS
def test_property_the_gram_floor_holds_for_every_treatment_group(seed, wide):
    """One floor serves every group: it is at most x'x = u'Gu for every control
    group of every treatment group, on centred panels and on positive panels with
    more geos than periods."""
    M = wide_panel(seed % 300, n=5, p=7) if wide else geo_panel(seed % 300, p=6)
    p = M.shape[1]
    lam = gram_floor(M)
    assert lam is not None and lam > 0
    for T in combinations(range(p), 2):
        G = ExactData(M[:, 0], np.delete(M, list(T), axis=1)).G
        for r in range(1, p - 1):
            for C in combinations(range(p - 2), r):
                assert lam <= sum(G[i, j] for i in C for j in C)


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2))
@SETTINGS
def test_property_a_group_ruled_out_has_no_better_design(seed, k):
    """Against the certified optimum, enumeration finds no design with a larger key:
    no group the screen ruled out held one."""
    M = raw_panel(seed % 200, p=6)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=3, blank_periods=0)
    best, _, _ = relative_optimum(M, k, N_TEST)
    assert out.certified and out.key == pytest.approx(best, rel=REL)


# ---------------------------------------------------------------- edge


def test_one_control_left():
    M = raw_panel(2, p=4)
    out = certify_treatment(M, 3, n_test=N_TEST, restarts=3, blank_periods=0)
    best, T, C = relative_optimum(M, 3, N_TEST)
    assert out.certified and set(out.treatment) == T and len(out.controls) == 1
    assert out.key == pytest.approx(best, rel=REL)


# ---------------------------------------------------------------- y'y in integers


def _fraction_sum(v):
    return sum(Q(x) ** 2 for x in v)


def test_smoke_square_sum_is_exact_on_mixed_exponents():
    v = [0.1, -3.5, 1e300, 5e-324, 2.0 ** -1074, -1e-300, 7.0]
    s = square_sum(np.array(v))
    assert isinstance(s, Q) and s == _fraction_sum(v)


def test_square_sum_of_a_panel_column():
    M = geo_panel(3, p=6)
    y, _, _ = _split(M, (0, 2))
    assert square_sum(y) == _fraction_sum(y.tolist())


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True), max_size=40))
@settings(max_examples=300, deadline=None)
def test_property_square_sum_is_the_exact_sum_of_squares(v):
    assert square_sum(np.array(v, dtype=float)) == _fraction_sum(v)


def test_square_sum_of_nothing_is_zero():
    assert square_sum(np.array([], dtype=float)) == 0
    assert square_sum(np.zeros(5)) == 0


def test_square_sum_of_integers():
    assert square_sum(np.array([3.0, -4.0])) == 25


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_square_sum_of_a_non_finite_value_raises(bad):
    with pytest.raises(MlsynthDataError, match="finite"):
        square_sum(np.array([1.0, bad]))


def test_the_floor_reads_a_given_yy():
    M = geo_panel(5, p=6); lam = gram_floor(M); T = (1, 4)
    y, _, _ = _split(M, T)
    for sign in (1, -1):
        assert rss_floor(M, T, lam, yy=square_sum(y), sign=sign) == rss_floor(M, T, lam, sign=sign)


def test_a_given_yy_is_not_recomputed(monkeypatch):
    """The floor's cap reads the y'y it is given (the bracket's top, from the screen), so the
    exact sum of squares is never formed for it."""
    M = geo_panel(5, p=6); lam = gram_floor(M); T = (1, 4)
    want = rss_floor(M, T, lam, sign=-1)
    top = treatment.yy_bracket(_split(M, T)[0])[1]
    monkeypatch.setattr(treatment, "square_sum", lambda v: pytest.fail("y'y recomputed"))
    for sign in (1, -1):
        assert rss_floor(M, T, lam, yy=top, sign=sign) >= 0
    assert float(rss_floor(M, T, lam, yy=top, sign=-1)) == pytest.approx(float(want), rel=1e-9)


# ---------------------------------------------------------------- one design's moments in integers


def _fraction_moments(y, X, C):
    D = ExactData(y, X)
    u = np.zeros(X.shape[1], dtype=object); u[:] = Q(0); u[sorted(C)] = Q(1)
    return D.c @ u, u @ D.G @ u, D.yy


def test_smoke_design_moments_are_exact_on_a_panel():
    M = geo_panel(4, p=7)
    y, X, _ = _split(M, (0, 3))
    for C in ({0}, {1, 4}, {0, 2, 3}, set(range(5))):
        got = design_moments(y, X, C)
        assert all(isinstance(v, Q) for v in got)
        assert got == _fraction_moments(y, X, C)


@given(seed=st.integers(0, 10_000), scale=st.sampled_from([1e-300, 1e-6, 1.0, 1e6, 1e150]),
       cols=st.sets(st.integers(0, 4), min_size=1))
@settings(max_examples=150, deadline=None)
def test_property_design_moments_match_the_fraction_arithmetic(seed, scale, cols):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=9) * scale
    X = rng.normal(size=(9, 5)) * scale
    assert design_moments(y, X, cols) == _fraction_moments(y, X, cols)


def test_design_moments_of_a_negative_slope():
    """c'u < 0 is returned as it is: the caller decides what a negative slope means."""
    y = np.array([1.0, -2.0, 0.5]); X = -np.column_stack([y, y * 0.5])
    cu, uGu, yy = design_moments(y, X, {0, 1})
    assert cu < 0 and uGu > 0 and yy == square_sum(y)


def test_design_moments_of_an_empty_design_raise():
    y = np.ones(3); X = np.ones((3, 2))
    with pytest.raises(MlsynthDataError, match="no control"):
        design_moments(y, X, set())


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_design_moments_of_a_non_finite_column_raise(bad):
    y = np.ones(3); X = np.ones((3, 2)); X[1, 1] = bad
    with pytest.raises(MlsynthDataError, match="finite"):
        design_moments(y, X, {1})


def test_the_pass_scores_its_designs_without_the_exact_gram(monkeypatch):
    """The pass needs one design's c'u, u'Gu and y'y per sign, not the group's whole
    exact Gram, which took 0.17 s of its 0.5 s a group on GeoLift."""
    from . import certificate

    M = raw_panel(2, p=6)
    T = (0, 1)
    want = treatment._local(_centred(M), M, T, N_TEST, 3)
    monkeypatch.setattr(certificate, "ExactData", lambda *a, **k: pytest.fail("the pass built the exact Gram"))
    assert treatment._local(_centred(M), M, T, N_TEST, 3) == want


def test_the_pass_reports_the_exact_rss_and_sign_of_its_design():
    M = raw_panel(3, p=6)
    T = (2, 5)
    starts, d = treatment._local(_centred(M), M, T, N_TEST, 3)
    y, X, rest = _split(_centred(M), T)
    cu, uGu, yy = _fraction_moments(y, X, [rest.index(j) for j in d.controls])
    assert d.rss == yy - cu * cu / uGu
    assert d.slope == (1 if cu >= 0 else -1)
    assert set(starts) == {1, -1}


def test_the_pass_keeps_the_better_sign(monkeypatch):
    """The descent on -y can end at the better design: the pass reports the lower RSS
    of the two signs' designs."""
    M = mirror_panel(104, p=7)
    best, T, C = relative_optimum(M, 2, N_TEST)
    starts, d = treatment._local(_centred(M), M, tuple(sorted(T)), N_TEST, 5)
    assert d.slope == -1 and set(d.controls) == C


def test_a_singular_gram_with_a_negative_entry_gives_no_floor_and_rules_nothing_out():
    M = geo_panel(1, p=5)
    M = np.column_stack([M, -M[:, 0]])         # a geo and its negative
    assert gram_floor(M) is None
    assert rss_floor(M, (1,), None) == 0 and rss_floor(M, (1,), None, sign=-1) == 0


def test_a_singular_positive_gram_gives_the_diagonal_floor():
    """Twelve geos over ten periods: M'M is singular, so no eigenvalue bound exists,
    but every entry is positive and the floor is the smallest x_j'x_j."""
    M = wide_panel(0)
    G = ExactData(M[:, 0], M).G
    assert gram_floor(M) == min(G[j, j] for j in range(M.shape[1]))


def test_more_geos_than_periods_still_certifies():
    M = wide_panel(0)
    out = certify_treatment(M, 1, n_test=4, restarts=5, blank_periods=0)
    best, T, C = relative_optimum(M, 1, 4)
    assert out.certified
    assert out.key == pytest.approx(best, rel=REL)


def test_each_survivor_is_certified_against_the_incumbents_ceiling(monkeypatch):
    """Every sign certified is asked only whether the group has a design below its
    ceiling, RSS* (L_T / level*)^2, and only a sign whose floor is below it is run."""
    real = slopefree.certify_slope_free
    seen = []

    def spy(y, X, *a, ceiling=None, **kw):
        seen.append((square_sum(y), ceiling))
        return real(y, X, *a, ceiling=ceiling, **kw)

    monkeypatch.setattr(slopefree, "certify_slope_free", spy)
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, start=_worst(M), pass_restarts=0, blank_periods=0)
    assert seen and len(seen) >= len(out.survivors)
    assert all(c is not None and c > 0 for _, c in seen)
    for s in out.survivors:
        assert s["signs"] and set(s["signs"]) <= {1, -1}


def _flaky(monkeypatch, fail_calls):
    """_optimum, cut short on the given calls (1-based), as if a group's certificate
    had not closed in time; returns the ceilings each call saw."""
    real = treatment._optimum
    seen = []

    def wrapped(Mc, M, T, n_test, restarts, ceiling, floors, starts):
        res = real(Mc, M, T, n_test, restarts, ceiling, floors, starts)
        seen.append(ceiling)
        if len(seen) in fail_calls:
            res = treatment._GroupOutcome(res.design, False, res.signs)
        return res

    monkeypatch.setattr(treatment, "_optimum", wrapped)
    return seen


def _retried_turn(M, plain):
    """The first certification (1-based) whose ceiling at its turn is above the final
    incumbent's for its group: cut short there, the group is retried."""
    sums, n = geo_sums(M), M.shape[0]
    bar = relative_bar(plain.rss, plain.level, plain.gap)
    for i, s in enumerate(plain.survivors):
        c = relative_ceiling(group_level(sums, s["treatment"], n), bar)
        if s["ceiling"] is None or c < s["ceiling"]:
            return i + 1
    raise AssertionError("no certification of this run would be retried")


def _improving(monkeypatch):
    """A run whose incumbent improves several times, with the winner certified after
    other groups: the queue read worst bound first (the search reads it best first,
    where the screen leaves these panels one survivor), from a poor start and no pass.
    The retries are about what happens after a group is left open, not the order."""
    real = treatment._bound_order
    monkeypatch.setattr(treatment, "_bound_order", lambda r, ell: -real(r, ell))
    M = raw_panel(0, p=7)
    plain = certify_treatment(M, 2, n_test=N_TEST, start=[0, 1], restarts=3, pass_restarts=0, blank_periods=0)
    sv = [s["treatment"] for s in plain.survivors]
    assert plain.replaced >= 2 and sv.index(sorted(plain.treatment)) > 0
    return M, [0, 1], plain


def test_a_group_left_open_is_retried_against_the_final_ceiling(monkeypatch):
    """A survivor whose certificate does not close against the ceiling of its turn
    is certified again at the end against the final incumbent's, which is lower
    when the incumbent improved since: the search is certified once it closes."""
    M, start, plain = _improving(monkeypatch)
    turn = _retried_turn(M, plain)
    seen = _flaky(monkeypatch, {turn})
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0, blank_periods=0)
    assert out.certified and out.retried == 1
    assert seen[-1] is not None and (seen[turn - 1] is None or seen[-1] < seen[turn - 1])
    assert out.key == plain.key


def test_a_retried_group_can_replace_the_incumbent(monkeypatch):
    """The winning group, first in the queue, cut short on its turn before its design
    is found: the groups after it are certified against the poor start and improve
    the incumbent, which lowers the winner's ceiling, so it is retried, and its
    optimum, found on the retry, replaces the incumbent."""
    M = raw_panel(0, p=7)
    plain = certify_treatment(M, 2, n_test=N_TEST, start=[0, 1], restarts=3, pass_restarts=0, blank_periods=0)
    turn = 1 + [s["treatment"] for s in plain.survivors].index(sorted(plain.treatment))
    real = treatment._optimum
    calls = []

    def wrapped(*a):
        res = real(*a)
        calls.append(1)
        if len(calls) == turn:
            res = treatment._GroupOutcome(None, False, res.signs)
        return res

    monkeypatch.setattr(treatment, "_optimum", wrapped)
    out = certify_treatment(M, 2, n_test=N_TEST, start=[0, 1], restarts=3, pass_restarts=0, blank_periods=0)
    assert out.certified and out.retried == 1 and len(out.survivors) > len(plain.survivors)
    assert out.key == plain.key and sorted(out.treatment) == sorted(plain.treatment)


def test_a_group_still_open_after_its_retry_leaves_the_search_uncertified(monkeypatch):
    M, start, plain = _improving(monkeypatch)
    n = len(plain.survivors)
    _flaky(monkeypatch, {_retried_turn(M, plain), n + 1})     # open on its turn and on its retry
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0, blank_periods=0)
    assert not out.certified and out.retried == 1


# ---------------------------------------------------------------- the pass and the queue


def test_the_pass_needs_no_more_certifications_than_without_it():
    M = raw_panel(5, p=7)
    start = _worst(M)
    best, T, _ = relative_optimum(M, 2, N_TEST)
    off = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0, blank_periods=0)
    on = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10, blank_periods=0)
    for out in (off, on):
        assert out.certified and set(out.treatment) == T
        assert out.key == pytest.approx(best, rel=REL)
    assert off.searched == 0 and on.searched >= 1
    assert len(on.survivors) <= len(off.survivors)
    assert on.replaced <= off.replaced


def test_the_pass_adopts_its_best_design_as_the_incumbent():
    M = raw_panel(5)
    best, T, _ = relative_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst(M), restarts=3, pass_restarts=10, blank_periods=0)
    assert out.pass_key == pytest.approx(best, rel=REL)
    assert out.replaced == 0


def test_groups_are_certified_best_pass_design_first(monkeypatch):
    M = raw_panel(5, p=7)
    order = []
    real = treatment._optimum

    def spy(Mc, M_, T, *a):
        order.append(tuple(T))
        return real(Mc, M_, T, *a)

    monkeypatch.setattr(treatment, "_optimum", spy)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst(M), restarts=3, pass_restarts=10, blank_periods=0)
    assert out.certified and len(order) == len(out.survivors)
    sums, n = geo_sums(M), M.shape[0]
    vals = []
    for T in order:
        _, d = treatment._local(_centred(M), M, T, N_TEST, 10)
        vals.append(None if d is None else group_level(sums, T, n) ** 2 / d.rss)
    found = [v for v in vals if v is not None]
    assert found == sorted(found, reverse=True)


def test_each_certification_starts_from_its_groups_pass_designs(monkeypatch):
    seen = []
    real = treatment._optimum

    def spy(Mc, M_, T, n_test, restarts, ceiling, floors, starts):
        seen.append(starts)
        return real(Mc, M_, T, n_test, restarts, ceiling, floors, starts)

    monkeypatch.setattr(treatment, "_optimum", spy)
    M = raw_panel(5, p=7)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst(M), restarts=3, pass_restarts=10, blank_periods=0)
    assert len(seen) == len(out.survivors) >= 1
    assert all(set(s) == {1, -1} for s in seen)


def test_a_negative_pass_is_refused():
    with pytest.raises(MlsynthDataError, match="pass_restarts"):
        certify_treatment(raw_panel(0), 2, n_test=N_TEST, pass_restarts=-1, blank_periods=0)


def test_a_refused_certificate_is_reported(monkeypatch):
    """A sign whose certification is cut short leaves the search uncertified."""
    real = slopefree.certify_slope_free

    def short(*a, **kw):
        res = real(*a, **kw)
        res.certified = False
        return res

    monkeypatch.setattr(slopefree, "certify_slope_free", short)
    out = certify_treatment(raw_panel(5), 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert not out.certified


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("k", [0, 6, -1])
def test_a_size_without_a_control_raises(k):
    with pytest.raises(MlsynthDataError, match="treatment"):
        certify_treatment(raw_panel(0), k, n_test=N_TEST, blank_periods=0)


def test_a_start_of_the_wrong_size_raises():
    with pytest.raises(MlsynthDataError, match="start"):
        certify_treatment(raw_panel(0), 2, n_test=N_TEST, start=[0], blank_periods=0)
