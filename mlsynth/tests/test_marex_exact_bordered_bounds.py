"""MAREX's exact backend: the treated bound priced from a prefix, and made safe.

The search prices every candidate treated set ``S`` by the closed-form bound
``min { w'G_S w + a'w : 1'w = 1 } = (2 + t)^2 / (4 s) - u / 4``, with
``s = 1'G_S^-1 1``, ``t = 1'G_S^-1 a`` and ``u = a'G_S^-1 a``. Candidates
arrive in lexicographic order, so consecutive ones share all but their last
market. Writing ``S = P + {j}``, the Banachiewicz inversion formula (Zhang
2005, (0.7.2)) extends ``s``, ``t`` and ``u`` from the prefix ``P`` in
``O(m^2)``, and the prefix's inverse is formed once for its whole run.

An update is not guaranteed to be as accurate as a fresh solve, so the bound no
longer rests on the solve. For any vector ``v``,

    g(S) >= -v'G_S v + min over i of (2 (G_S v)_i + a_i),

with equality at the affine minimiser, where it is the closed form
(Neumaier and Shcherbina 2004, sec. 3). The search evaluates this at the
minimiser bordering returns and subtracts an a priori bound on the rounding of
that evaluation (Higham 2002, (3.5) and (3.13)), so the bound holds however
accurate the update was.

The tests check bordering against direct solves, the safe form against the fit
at arbitrary points, the bound against the closed form it replaces, and the
cases where a solve breaks down.

Levels: smoke, unit, edge, failure, property.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.exact import (
    search_cluster,
    side_fit,
    treated_fit_bound,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import _problem


# ------------------------------------------------------------------ helpers
def _subsets(n, m):
    return np.array(list(itertools.combinations(range(n), m)), dtype=np.int64)


def _fit(B, A, S, a=None):
    S = list(S)
    return side_fit(B, A, S, None if a is None else a[S])[0]


def _penalty(seed, n, scale=1.0):
    return scale * np.abs(np.random.default_rng(seed).normal(size=n))


def _direct(B, A, S, a=None):
    """``s, t, u, G_S^-1 1, G_S^-1 a`` by a fresh solve, and ``cond(G_S)``."""
    Bc = B[:, list(S)] - A[:, None]
    G = Bc.T @ Bc
    cond = np.linalg.cond(G)
    if not cond < 1e12:
        return np.nan, np.nan, np.nan, None, None, np.inf
    one = np.ones(len(S))
    x = np.linalg.solve(G, one)
    if a is None:
        return x.sum(), None, None, x, None, cond
    aS = a[list(S)]
    y = np.linalg.solve(G, aS)
    return x.sum(), one @ y, aS @ y, x, y, cond


def _closed_form(B, A, S, a=None):
    s, t, u, _, _, cond = _direct(B, A, S, a)
    val = 1.0 / s if a is None else (2.0 + t) ** 2 / (4.0 * s) - u / 4.0
    return max(val, 0.0), cond


def _sales(seed, n=8, T=30, level=5e4):
    """Columns at the scale of weekly sales: a large level, a factor, noise."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0) * 1e3
    B = level + f @ rng.uniform(0.5, 1.5, size=(2, n)) + rng.normal(size=(T, n)) * 50.0
    return B, B.mean(axis=1)


def _gram(B, A):
    return exact_mod.centred_gram(B, A)


def _near_duplicates(seed, eps, pair=(4, 5), n=8, T=12):
    """Sales-sized columns with one market a near-copy of another.

    The copy differs by a relative ``eps`` per period, so any set holding both
    has a Gram matrix with a condition number near ``1 / eps^2``.
    """
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = 1e3 * (20 + f @ rng.uniform(0.5, 1.5, size=(2, n))
               + rng.normal(size=(T, n)) * 0.3)
    B[:, pair[1]] = B[:, pair[0]] * (1 + eps * rng.normal(size=T))
    return B, B.mean(axis=1)


def _unguarded(G, subs):
    """The closed form ``1/s`` from bordering alone, with no guard."""
    s = exact_mod.bordered_terms(G, subs)[0]
    return np.where(np.isfinite(s) & (s > 0), 1.0 / np.where(s > 0, s, 1.0), 0.0)


def _exact_form(B, A, S, v, a=None):
    """The safe form at ``v`` in exact rational arithmetic on the float data.

    Every float is a rational, so centring, the Gram matrix and the form are
    computed with no rounding at all: the value the floating-point evaluation
    approximates.
    """
    S, T = list(S), B.shape[0]
    cols = [[Fraction(float(B[t, j])) - Fraction(float(A[t])) for t in range(T)]
            for j in S]
    m = len(S)
    G = [[sum(p * q for p, q in zip(cols[i], cols[k])) for k in range(m)]
         for i in range(m)]
    w = [Fraction(float(x)) for x in v]
    Gv = [sum(G[i][k] * w[k] for k in range(m)) for i in range(m)]
    lin = [Fraction(0)] * m if a is None else [Fraction(float(a[j])) for j in S]
    return -sum(w[i] * Gv[i] for i in range(m)) + min(
        2 * Gv[i] + lin[i] for i in range(m))


# --------------------------------------------------------------------- smoke
@pytest.mark.parametrize("linear", [False, True])
def test_the_bound_is_finite_and_non_negative_for_every_size(linear):
    B, A = _problem(60, n=9)
    a = _penalty(61, 9) if linear else None
    for m in range(1, 9):
        lb = treated_fit_bound(B, A, _subsets(9, m), linear=a)
        assert lb.shape == (len(_subsets(9, m)),)
        assert np.all(np.isfinite(lb)) and np.all(lb >= 0.0)


# ---------------------------------------------------------- unit: bordering
@pytest.mark.parametrize("linear", [False, True])
def test_bordering_reproduces_the_direct_solve(linear):
    B, A = _problem(62, n=9, T=12)
    a = _penalty(63, 9) if linear else None
    G, _ = _gram(B, A)
    for m in (2, 3, 4, 5):
        subs = _subsets(9, m)
        s, t, x, y = exact_mod.bordered_terms(G, subs, a)
        for r, S in enumerate(subs):
            ds, dt, _, dx, dy, cond = _direct(B, A, S, a)
            assert cond < 1e6
            assert s[r] == pytest.approx(ds, rel=1e-9)
            np.testing.assert_allclose(x[r], dx, rtol=1e-8, atol=1e-12)
            if linear:
                assert t[r] == pytest.approx(dt, rel=1e-8, abs=1e-12)
                np.testing.assert_allclose(y[r], dy, rtol=1e-8, atol=1e-12)
            else:
                assert t is None and y is None


def test_adding_a_market_never_lowers_s():
    """``s_S - s_P = (1 - g'G_P^-1 1)^2 / sigma`` with ``sigma > 0``."""
    B, A = _problem(64, n=9, T=12)
    G, _ = _gram(B, A)
    for m in (2, 3, 4):
        subs = _subsets(9, m)
        s = exact_mod.bordered_terms(G, subs)[0]
        sP = exact_mod.bordered_terms(G, subs[:, :-1])[0] if m > 2 else \
            1.0 / G[subs[:, 0], subs[:, 0]]
        assert np.all(s >= sP * (1.0 - 1e-12))


def test_prefix_runs_follow_contiguous_rows():
    subs = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 4], [1, 2, 3]])
    prefixes, run = exact_mod.prefix_runs(subs)
    np.testing.assert_array_equal(prefixes, [[0, 1], [0, 2], [0, 1], [1, 2]])
    np.testing.assert_array_equal(run, [0, 0, 1, 2, 3])


def test_the_order_of_the_candidates_does_not_change_their_bounds():
    B, A = _problem(65, n=9)
    a = _penalty(66, 9)
    subs = _subsets(9, 4)
    perm = np.random.default_rng(67).permutation(len(subs))
    np.testing.assert_allclose(treated_fit_bound(B, A, subs[perm], linear=a),
                               treated_fit_bound(B, A, subs, linear=a)[perm],
                               rtol=1e-12, atol=0.0)


def test_a_prefix_split_across_chunks(monkeypatch):
    B, A = _problem(68, n=9)
    a = _penalty(69, 9)
    subs = _subsets(9, 4)
    whole = treated_fit_bound(B, A, subs, linear=a)
    monkeypatch.setattr(exact_mod, "CHUNK", 7)
    np.testing.assert_allclose(treated_fit_bound(B, A, subs, linear=a), whole,
                               rtol=1e-12, atol=0.0)


# --------------------------------------------------------- unit: the bound
@pytest.mark.parametrize("linear", [False, True])
def test_the_bound_matches_the_closed_form_it_replaces(linear):
    B, A = _problem(70, n=9, T=12)
    a = _penalty(71, 9, 0.5) if linear else None
    for m in (1, 2, 3, 4, 5):
        subs = _subsets(9, m)
        lb = treated_fit_bound(B, A, subs, linear=a)
        for r, S in enumerate(subs):
            cf, cond = _closed_form(B, A, S, a)
            assert cond < 1e6
            assert lb[r] == pytest.approx(cf, rel=1e-8, abs=1e-10)


def test_the_safe_form_is_the_closed_form_at_the_affine_minimiser():
    B, A = _problem(72, n=8, T=12)
    a = _penalty(73, 8)
    G, d = _gram(B, A)
    subs = _subsets(8, 3)
    s, t, x, y = exact_mod.bordered_terms(G, subs, a)
    v = exact_mod.affine_minimiser(s, t, x, y)
    np.testing.assert_allclose(v.sum(axis=1), 1.0, rtol=1e-10)
    safe = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v, a)
    u = np.einsum("li,li->l", a[subs], y)
    closed = (2.0 + t) ** 2 / (4.0 * s) - u / 4.0
    np.testing.assert_allclose(safe, closed, rtol=1e-8, atol=1e-10)


def test_the_safe_form_is_a_lower_bound_at_any_point():
    B, A = _problem(74, n=8, T=10)
    a = _penalty(75, 8, 2.0)
    G, d = _gram(B, A)
    subs = _subsets(8, 3)
    fit = np.array([_fit(B, A, S, a) for S in subs])
    rng = np.random.default_rng(76)
    for scale in (1e-3, 1.0, 1e3):
        v = rng.normal(size=subs.shape) * scale
        lb = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v, a)
        assert np.all(lb <= fit + 1e-9 * (1.0 + fit))


@pytest.mark.parametrize("linear", [False, True])
def test_the_safe_bound_never_exceeds_the_exact_value_of_the_form(linear):
    """Less its rounding term, the computed form is at most its exact value.

    The form is evaluated at the affine minimiser and at random points, on
    columns at the level of weekly sales, where the rounding of forming ``G``
    is largest next to the fit. In about half the rows the floating-point value
    lands above the exact one; the rounding term is what keeps the bound under
    it.
    """
    B, A = _sales(90, n=7, T=20)
    a = _penalty(91, 7, 1e6) if linear else None
    G, d = _gram(B, A)
    subs = _subsets(7, 3)
    rng = np.random.default_rng(92)
    points = [exact_mod.affine_minimiser(*exact_mod.bordered_terms(G, subs, a))]
    points += [rng.normal(size=subs.shape) for _ in range(3)]
    above = 0
    for v in points:
        safe = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v, a)
        term = exact_mod.safe_rounding(d, B.shape[0], subs, v, a)
        for r, S in enumerate(subs):
            exact = _exact_form(B, A, S, v[r], a)
            assert Fraction(float(safe[r])) <= exact
            above += Fraction(float(safe[r] + term[r])) > exact
    assert above > 0


@pytest.mark.parametrize("scale", [1e-155, 1e-160, 1e-165])
def test_the_safe_bound_holds_where_products_underflow(scale):
    """Below about 1e-153 the products that form ``G`` are subnormal.

    There a rounding error is absolute, up to half the smallest subnormal
    ``eta / 2``, and no multiple of the unit roundoff covers it: before the
    ``eta`` term the computed form exceeded its exact value in 12 to 83 per
    cent of rows at these scales.
    """
    exceeded = 0
    for seed in range(10):
        B, A = _problem(seed, n=5, T=8)
        B, A = B * scale, A * scale
        G, d = _gram(B, A)
        subs = _subsets(5, 3)
        v = np.random.default_rng(seed).normal(size=subs.shape)
        safe = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v)
        term = exact_mod.safe_rounding(d, B.shape[0], subs, v)
        for r, S in enumerate(subs):
            exact = _exact_form(B, A, S, v[r])
            assert Fraction(float(safe[r])) <= exact
            exceeded += Fraction(float(safe[r] + term[r])) > exact
    assert exceeded > 0


def test_the_rounding_term_sits_far_under_the_margin_on_sales_sized_data():
    """The rounding the bound gives up is small next to what the search keeps.

    The term is subtracted from every bound, so it must not eat the bound's
    power to discard: on columns at the level of weekly sales it stays under a
    hundredth of the search's margin.
    """
    B, A = _sales(77)
    G, d = _gram(B, A)
    subs = _subsets(8, 3)
    v = exact_mod.affine_minimiser(*exact_mod.bordered_terms(G, subs))
    term = exact_mod.safe_rounding(d, B.shape[0], subs, v)
    lb = treated_fit_bound(B, A, subs)
    assert np.all(term > 0.0)
    assert np.all(term < 1e-2 * exact_mod.PRUNE_MARGIN * (1.0 + lb))


# -------------------------------------------------------- the fast path
@pytest.mark.parametrize("linear", [False, True])
def test_the_fast_path_matches_the_safe_bound_where_the_guard_trusts_it(linear):
    B, A = _problem(93, n=9, T=12)
    a = _penalty(94, 9) if linear else None
    for m in (1, 2, 3, 4, 5):
        subs = _subsets(9, m)
        fast = treated_fit_bound(B, A, subs, linear=a, rigorous=False)
        safe = treated_fit_bound(B, A, subs, linear=a, rigorous=True)
        np.testing.assert_allclose(fast, safe, rtol=1e-8, atol=1e-10)


def test_the_fast_path_is_the_closed_form_with_a_penalty():
    B, A = _sales(95, n=8, T=20)
    a = _penalty(96, 8, 1e6)
    subs = _subsets(8, 3)
    fast = treated_fit_bound(B, A, subs, linear=a, rigorous=False)
    for r, S in enumerate(subs):
        cf, cond = _closed_form(B, A, S, a)
        assert cond < 1e6
        assert fast[r] == pytest.approx(cf, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("seed, eps", [(0, 1e-11), (1, 1e-12), (2, 1e-12),
                                       (3, 1e-12)])
def test_the_guard_keeps_the_fast_path_under_the_fit_where_bordering_fails(
        seed, eps):
    """Two near-copies in a prefix leave its inverse inaccurate.

    Bordering alone then puts some of these sets' bounds above their fits, by
    up to 16 times in a wider sweep; the guard sends those sets to the safe
    form.
    """
    B, A = _near_duplicates(seed, eps)
    G, _ = _gram(B, A)
    subs = _subsets(8, 3)
    fit = np.array([_fit(B, A, S) for S in subs])
    assert np.any(_unguarded(G, subs) > 1.01 * fit)
    fast = treated_fit_bound(B, A, subs, rigorous=False)
    assert np.all(fast <= fit + 1e-8 * (1.0 + fit))


def test_a_candidate_the_guard_rejects_is_priced_by_the_safe_form():
    """Near-copies in the prefix, or split between it and the new market.

    Either makes the set's Gram nearly singular: the first through the prefix's
    condition estimate, the second through ``G_jj / sigma``. At a gap of
    ``1e-6`` both are near ``1e9``, past the guard, while ``sigma`` stays
    positive, so the update still returns a closed form to refuse. Both kinds
    of row get exactly the safe bound.
    """
    B, A = _near_duplicates(1, 1e-6)
    G, _ = _gram(B, A)
    subs = _subsets(8, 3)
    in_prefix = np.array([4 in S[:-1] and 5 in S[:-1] for S in subs])
    split = np.array([S[-1] == 5 and 4 in S[:-1] for S in subs])
    fast = treated_fit_bound(B, A, subs, rigorous=False)
    safe = treated_fit_bound(B, A, subs, rigorous=True)
    unguarded = _unguarded(G, subs)
    for rows in (in_prefix, split):
        assert rows.any()
        assert np.any(unguarded[rows] != safe[rows])
        np.testing.assert_array_equal(fast[rows], safe[rows])


def test_the_search_prices_by_the_number_of_candidates(monkeypatch):
    """At or under the limit every bound is safe; above it, the fast path.

    A two-candidate buffer forces the second pass, which prices too.
    """
    monkeypatch.setattr(exact_mod, "BUFFER", 2)
    B, A = _problem(97, n=8)
    seen = []
    original = exact_mod.treated_fit_bound

    def spy(*args, rigorous=True, **kwargs):
        seen.append(rigorous)
        return original(*args, rigorous=rigorous, **kwargs)

    monkeypatch.setattr(exact_mod, "treated_fit_bound", spy)
    total = len(_subsets(8, 3))
    for limit, expected in ((total, True), (total - 1, False)):
        seen.clear()
        monkeypatch.setattr(exact_mod, "RIGOROUS_PRICING_LIMIT", limit)
        search_cluster(B, A, sizes=[3], filt=SubsetFilter())
        assert seen and set(seen) == {expected}


@pytest.mark.parametrize("seed, eps, linear", [(98, 1e-12, False),
                                               (99, 1e-6, False),
                                               (100, 1.0, True)])
def test_both_paths_find_the_same_design(monkeypatch, seed, eps, linear):
    """The near-copies carry no penalty: there the simplex fit itself can fail."""
    B, A = _near_duplicates(seed, eps)
    a = _penalty(seed, 8, 1e6) if linear else None
    designs = []
    for limit in (10 ** 9, 0):
        monkeypatch.setattr(exact_mod, "RIGOROUS_PRICING_LIMIT", limit)
        designs.append(search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                                      linear_treated=a, top_k=3))
    assert [d.treated for d in designs[0]] == [d.treated for d in designs[1]]
    np.testing.assert_allclose([d.value for d in designs[0]],
                               [d.value for d in designs[1]], rtol=1e-12)


# ----------------------------------------------------------- edge cases
def test_one_treated_market_is_bounded_exactly():
    B, A = _problem(78, n=8)
    a = _penalty(79, 8)
    subs = _subsets(8, 1)
    lb = treated_fit_bound(B, A, subs, linear=a)
    fit = np.array([_fit(B, A, S, a) for S in subs])
    np.testing.assert_allclose(lb, fit, rtol=1e-9)


@pytest.mark.parametrize("seed", [80, 81, 82, 83])
def test_a_target_the_treated_set_reproduces_gets_no_positive_bound(seed):
    """Here ``g(S) = 0``, the closed form is 0, and rounding is all that is left.

    The target is an interior convex combination of three markets at the level
    of weekly sales. Evaluated in floating point the safe form lands within its
    rounding of zero, on either side; less the rounding term it is at most zero.
    """
    B, _ = _sales(seed)
    w = np.array([0.5, 0.3, 0.2])
    A = B[:, [1, 4, 6]] @ w
    subs = np.array([[1, 4, 6]])
    assert _fit(B, A, [1, 4, 6]) < 1e-12 * float(A @ A)
    assert treated_fit_bound(B, A, subs)[0] == 0.0


def test_duplicate_markets_get_a_valid_finite_bound():
    B, A = _problem(84, n=8)
    B[:, 3] = B[:, 2]
    a = _penalty(85, 8)
    subs = _subsets(8, 3)
    lb = treated_fit_bound(B, A, subs, linear=a)
    fit = np.array([_fit(B, A, S, a) for S in subs])
    assert np.all(np.isfinite(lb))
    assert np.all(lb <= fit + 1e-9 * (1.0 + fit))


def test_a_market_at_the_target_bounds_its_sets_by_zero():
    B, A = _problem(86, n=8)
    B[:, 5] = A
    subs = np.array([S for S in _subsets(8, 3) if 5 in S])
    lb = treated_fit_bound(B, A, subs)
    assert np.all(np.isfinite(lb))
    np.testing.assert_array_equal(lb, 0.0)


def test_more_treated_markets_than_periods():
    B, A = _problem(87, n=8, T=3)
    a = _penalty(88, 8)
    subs = _subsets(8, 5)
    lb = treated_fit_bound(B, A, subs, linear=a)
    fit = np.array([_fit(B, A, S, a) for S in subs])
    assert np.all(np.isfinite(lb))
    assert np.all(lb <= fit + 1e-9 * (1.0 + fit))


# ---------------------------------------------------------------- failure
def test_a_breakdown_in_the_update_is_reported_as_no_bound_not_as_nan():
    """A prefix whose Gram is singular has no inverse to extend.

    Bordering reports it as non-finite terms; the bound for its candidates is
    then zero, which discards nothing, and never NaN, which the search would
    sort and compare without error.
    """
    B, A = _problem(89, n=8)
    B[:, 1] = B[:, 0]
    G, _ = _gram(B, A)
    subs = np.array([[0, 1, j] for j in range(2, 8)])
    s = exact_mod.bordered_terms(G, subs)[0]
    assert not np.any(np.isfinite(s))
    lb = treated_fit_bound(B, A, subs)
    assert np.all(np.isfinite(lb))
    fit = np.array([_fit(B, A, S) for S in subs])
    assert np.all(lb <= fit + 1e-9 * (1.0 + fit))


# ------------------------------------------------------------ properties
@given(seed=st.integers(0, 10_000), n=st.integers(2, 8), m=st.integers(1, 5),
       scale=st.sampled_from([1e-3, 1.0, 1e3, 1e6]), linear=st.booleans())
@settings(max_examples=80, deadline=None)
def test_the_safe_form_is_a_lower_bound_at_any_point_over_the_domain(
        seed, n, m, scale, linear):
    m = min(m, n)
    B, A = _problem(seed, n=n, T=6)
    a = _penalty(seed + 1, n, 3.0) if linear else None
    G, d = _gram(B, A)
    subs = _subsets(n, m)
    v = np.random.default_rng(seed + 2).normal(size=subs.shape) * scale
    lb = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v, a)
    fit = np.array([_fit(B, A, S, a) for S in subs])
    assert np.all(lb <= fit + 1e-9 * (1.0 + fit))


@given(seed=st.integers(0, 10_000), n=st.integers(2, 6), m=st.integers(1, 4),
       scale=st.sampled_from([1e-3, 1.0, 1e3]), sales=st.booleans(),
       linear=st.booleans(), tiny=st.sampled_from([1.0, 1e-150, 1e-160]))
@settings(max_examples=40, deadline=None)
def test_the_safe_bound_never_exceeds_the_exact_form_over_the_domain(
        seed, n, m, scale, sales, linear, tiny):
    m = min(m, n)
    B, A = _sales(seed, n=n, T=8) if sales else _problem(seed, n=n, T=8)
    B, A = B * tiny, A * tiny
    a = (_penalty(seed + 1, n, 1e6 if sales else 1.0) * tiny ** 2
         if linear else None)
    G, d = _gram(B, A)
    subs = _subsets(n, m)
    v = np.random.default_rng(seed + 2).normal(size=subs.shape) * scale
    safe = exact_mod.safe_treated_bound(G, d, B.shape[0], subs, v, a)
    for r, S in enumerate(subs):
        assert Fraction(float(safe[r])) <= _exact_form(B, A, S, v[r], a)


@given(seed=st.integers(0, 10_000), n=st.integers(2, 9), m=st.integers(1, 6),
       linear=st.booleans(), sales=st.booleans())
@settings(max_examples=80, deadline=None)
def test_the_bound_is_the_closed_form_and_under_the_fit_over_the_domain(
        seed, n, m, linear, sales):
    m = min(m, n)
    B, A = _sales(seed, n=n, T=7) if sales else _problem(seed, n=n, T=7)
    a = _penalty(seed + 1, n) * (float(np.abs(B - A[:, None]).mean()) ** 2
                                 if sales else 1.0) if linear else None
    subs = _subsets(n, m)
    lb = treated_fit_bound(B, A, subs, linear=a)
    assert np.all(np.isfinite(lb)) and np.all(lb >= 0.0)
    for r, S in enumerate(subs):
        f = _fit(B, A, S, a)
        assert lb[r] <= f + 1e-9 * (1.0 + f)
        cf, cond = _closed_form(B, A, S, a)
        if cond < 1e6:
            assert lb[r] == pytest.approx(cf, rel=1e-7, abs=1e-9 * (1.0 + f))


@given(seed=st.integers(0, 10_000), n=st.integers(3, 9), m=st.integers(2, 5))
@settings(max_examples=40, deadline=None)
def test_the_order_of_the_candidates_never_matters(seed, n, m):
    m = min(m, n)
    B, A = _problem(seed, n=n, T=8)
    a = _penalty(seed + 1, n)
    subs = _subsets(n, m)
    perm = np.random.default_rng(seed + 2).permutation(len(subs))
    np.testing.assert_allclose(treated_fit_bound(B, A, subs[perm], linear=a),
                               treated_fit_bound(B, A, subs, linear=a)[perm],
                               rtol=1e-12, atol=0.0)


@given(seed=st.integers(0, 10_000), n=st.integers(3, 9), m=st.integers(2, 6))
@settings(max_examples=40, deadline=None)
def test_adding_a_market_never_lowers_s_over_the_domain(seed, n, m):
    m = min(m, n)
    B, A = _problem(seed, n=n, T=12)
    G, _ = _gram(B, A)
    subs = _subsets(n, m)
    s = exact_mod.bordered_terms(G, subs)[0]
    sP = (exact_mod.bordered_terms(G, subs[:, :-1])[0] if m > 2
          else 1.0 / G[subs[:, 0], subs[:, 0]])
    ok = np.isfinite(s) & np.isfinite(sP)
    assert np.all(s[ok] >= sP[ok] * (1.0 - 1e-9))


@given(seed=st.integers(0, 10_000), m=st.integers(1, 5),
       eps=st.sampled_from([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3]),
       first=st.integers(0, 6), linear=st.booleans())
@settings(max_examples=60, deadline=None)
def test_the_fast_path_stays_under_the_fit_over_the_domain(
        seed, m, eps, first, linear):
    B, A = _near_duplicates(seed, eps, pair=(first, first + 1))
    # Penalised simplex fits on near-copies closer than 1e-8 can fail outright,
    # so the reference needs a gap of 1e-6 or wider; that still puts kappa
    # near 1e9, past the guard.
    a = _penalty(seed + 1, 8, 1e6) if linear and eps >= 1e-6 else None
    subs = _subsets(8, m)
    fast = treated_fit_bound(B, A, subs, linear=a, rigorous=False)
    assert np.all(np.isfinite(fast)) and np.all(fast >= 0.0)
    for r, S in enumerate(subs):
        f = _fit(B, A, S, a)
        assert fast[r] <= f + 1e-8 * (1.0 + f)


@given(seed=st.integers(0, 10_000), n=st.integers(3, 8), m=st.integers(1, 4),
       eps=st.sampled_from([1e-12, 1e-6, 1.0]), linear=st.booleans())
@settings(max_examples=30, deadline=None)
def test_both_paths_find_the_same_design_over_the_domain(seed, n, m, eps, linear):
    m = min(m, n - 1)
    B, A = _near_duplicates(seed, eps, pair=(0, n - 1), n=n, T=8)
    a = _penalty(seed + 1, n, 1e6) if linear and eps == 1.0 else None
    values = []
    for rigorous_limit in (10 ** 9, 0):
        exact_mod.RIGOROUS_PRICING_LIMIT, saved = (rigorous_limit,
                                                   exact_mod.RIGOROUS_PRICING_LIMIT)
        try:
            values.append(search_cluster(B, A, sizes=[m], filt=SubsetFilter(),
                                         linear_treated=a)[0].value)
        finally:
            exact_mod.RIGOROUS_PRICING_LIMIT = saved
    assert values[0] == pytest.approx(values[1], rel=1e-12)
