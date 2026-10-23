"""MAREX's exact backend with a certified optimality gap (``certify=True``).

The backend prices candidates and fits them in floating point, so its answer is
optimal up to rounding nobody bounded. With ``certify=True`` it also returns a
certificate: a lower bound on the optimal objective and an upper bound on the
returned design's, both exact rationals for the stored data, rounded outward.
The design is optimal to within their difference.

The pieces, tested below:

- ``Dyadic``: exact arithmetic on floating-point matrices (every double is an
  integer over a power of two);
- ``ExactData.interval``: a certified interval for one simplex fit, the Wolfe
  dual below and the exact objective at the fitted weights above;
- ``ExactData.treated_bounds``: a certified lower bound on every candidate's
  treated fit, the library's closed form made rigorous;
- ``certify_cluster``: the certificate of one cluster's search;
- ``MAREXConfig(certify=True)`` end to end.

Levels: unit, property, edge, failure, smoke, end to end.
"""
from __future__ import annotations

import dataclasses
import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.tests.test_marex_exact_backend import _cfg, _panel
from mlsynth.utils.marex_helpers.certify import (ClusterCertificate, Dyadic,
                                                ExactData, certify_cluster,
                                                round_down, round_up)
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import (ClusterDesign, search_cluster,
                                              side_fit, treated_fit_bound)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter


def _data(seed, T, n, offset=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    B = (rng.normal(size=(T, n)) + rng.normal(size=(T, 1))) * scale + offset
    A = B.mean(axis=1)
    return B, A


def _exact_g(data, cols, linear=None):
    """The fit's certified upper bound, the reference every lower bound must respect."""
    g, w = side_fit(data.B, data.A, list(cols), None if linear is None else linear[list(cols)])
    return data.interval(list(cols), w, linear)[1]


# ------------------------------------------------------------------ Dyadic
def test_dyadic_products_and_differences_are_exact():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 3)), rng.normal(size=(3, 2))
    D = Dyadic.of(a) @ Dyadic.of(b)
    for i, j in itertools.product(range(4), range(2)):
        assert D.frac(i, j) == sum(Fraction(a[i, k]) * Fraction(b[k, j]) for k in range(3))
    x, y = np.array([1e16, 1.0]), np.array([1.0, 1e-16])
    E = Dyadic.of(x) - Dyadic.of(y)
    assert E.frac(0) == Fraction(1e16) - 1 and E.frac(1) == 1 - Fraction(1e-16)


@given(st.fractions(min_value=-10**6, max_value=10**6))
def test_outward_rounding_brackets_every_rational(q):
    assert Fraction(round_down(q)) <= q <= Fraction(round_up(q))


def test_outward_rounding_past_the_largest_double():
    big = Fraction(10) ** 400
    assert round_down(big) == np.finfo(float).max and round_up(big) == np.inf
    assert round_down(-big) == -np.inf and round_up(-big) == -np.finfo(float).max


def test_dyadic_refuses_a_non_finite_entry():
    with pytest.raises(MlsynthDataError, match="finite"):
        Dyadic.of(np.array([1.0, np.nan]))


# ------------------------------------------------------------------ intervals
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(3, 25), n=st.integers(3, 8),
       k=st.integers(1, 3), penal=st.booleans())
def test_an_interval_brackets_the_fit(seed, T, n, k, penal):
    B, A = _data(seed, T, n)
    data = ExactData(B, A)
    lin = np.random.default_rng(seed).uniform(0, 2, n) if penal else None
    cols = list(range(k))
    g, w = side_fit(B, A, cols, None if lin is None else lin[cols])
    lo, hi = data.interval(cols, w, lin)
    assert lo <= hi
    assert float(lo) <= g * (1 + 1e-9) + 1e-12 and g <= float(hi) * (1 + 1e-9) + 1e-12
    assert float(hi - lo) <= 1e-9 * max(1.0, g)


def test_an_interval_at_poor_weights_still_brackets_the_fit():
    B, A = _data(1, 12, 5)
    data = ExactData(B, A)
    g, _ = side_fit(B, A, [0, 1, 2])
    lo, hi = data.interval([0, 1, 2], np.array([1.0, 0.0, 0.0]))
    assert lo <= Fraction(g) * (1 + Fraction(1, 10**9)) and Fraction(g) <= hi * (1 + Fraction(1, 10**9))


def test_an_interval_with_no_positive_weight_uses_the_uniform_point():
    B, A = _data(2, 10, 4)
    data = ExactData(B, A)
    lo, hi = data.interval([0, 1], np.array([0.0, -1.0]))
    g, _ = side_fit(B, A, [0, 1])
    assert lo <= Fraction(g) * (1 + Fraction(1, 10**9)) <= hi * (1 + Fraction(1, 10**6))


# ------------------------------------------------------------------ treated bounds
@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(3, 30), n=st.integers(4, 8),
       m=st.integers(1, 3), penal=st.booleans(),
       offset=st.sampled_from([0.0, 1e3, 1e8]))
def test_treated_bounds_never_exceed_the_fit(seed, T, n, m, penal, offset):
    B, A = _data(seed, T, n, offset=offset)
    data = ExactData(B, A)
    lin = np.random.default_rng(seed).uniform(0, 2, n) if penal else None
    sets = np.array(list(itertools.combinations(range(n), m)))
    for S, lb in zip(sets, data.treated_bounds(sets, lin)):
        assert Fraction(lb) <= _exact_g(data, S, lin), S


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(5, 30), n=st.integers(5, 8),
       m=st.integers(2, 3), penal=st.booleans())
def test_treated_bounds_are_as_tight_as_the_closed_form(seed, T, n, m, penal):
    B, A = _data(seed, T, n)
    data = ExactData(B, A)
    lin = np.random.default_rng(seed).uniform(0, 2, n) if penal else None
    sets = np.array(list(itertools.combinations(range(n), m)))
    ours = data.treated_bounds(sets, lin)
    lib = treated_fit_bound(B, A, sets, lin)
    scale = float(np.max(np.abs(data.Bf))) ** 2
    assert np.all(ours >= lib - 1e-9 * (scale + np.abs(lib)))


def test_treated_bounds_hold_on_a_hostile_collinear_panel():
    # Three nearly collinear markets of norm 1e6 and one far one, T = 200: the
    # Gram's own rounding exceeds every other error term, so a certificate that
    # trusts the floating-point Gram rises above the fit of the pair (0, 1).
    rng = np.random.default_rng(22)
    T, L = 200, 1e6
    v, u = rng.normal(size=T), rng.normal(size=T)
    B = np.column_stack([L * v + u, -L * v + u * 1.0000001,
                         0.5 * L * v + u * 0.9999999 + 1e-3 * rng.normal(size=T),
                         rng.normal(size=T) * L])
    data = ExactData(B, np.zeros(T))
    for m in (2, 3):
        sets = np.array(list(itertools.combinations(range(4), m)))
        for S, lb in zip(sets, data.treated_bounds(sets)):
            assert Fraction(lb) <= _exact_g(data, S), S


@pytest.mark.parametrize("T, x", [(200, 0.1), (5000, 1 / 3)])
def test_treated_bounds_hold_where_the_inner_products_drift(T, x):
    # Constant columns make every product in an inner product the same, so the
    # rounding of the sum drifts one way instead of averaging out: the
    # floating-point Gram is off by up to about a hundred units in the last
    # place. A bound that trusts it rises above the fit, on the pair (0, 1),
    # whose fit nearly cancels, by seventeen orders of magnitude.
    B = np.column_stack([np.full(T, x), np.full(T, -x * 1.0000001),
                         np.full(T, 2 * x + 0.3)])
    data = ExactData(B, np.zeros(T))
    for m in (1, 2):
        sets = np.array(list(itertools.combinations(range(3), m)))
        for S, lb in zip(sets, data.treated_bounds(sets)):
            assert Fraction(lb) <= _exact_g(data, S), S


def test_treated_bounds_respect_the_rounding_of_the_target_difference():
    # A target far from the markets: B - A in floating point drops the markets'
    # low-order bits, so the exact problem (on B and A as given) differs from
    # the one the floating-point Gram describes.
    rng = np.random.default_rng(5)
    T = 40
    B = rng.normal(size=(T, 5)) * 1e-3 + 1.0
    A = np.full(T, 3.0e8) + rng.normal(size=T)
    data = ExactData(B, A)
    sets = np.array(list(itertools.combinations(range(5), 2)))
    for S, lb in zip(sets, data.treated_bounds(sets)):
        assert Fraction(lb) <= _exact_g(data, S), S


def test_a_singular_block_gets_a_zero_bound():
    B = np.ones((6, 3))
    data = ExactData(B, np.zeros(6))
    assert data.treated_bounds(np.array([[0, 1]]))[0] == 0.0


def test_no_sets_no_bounds():
    data = ExactData(*_data(0, 5, 4))
    assert data.treated_bounds(np.zeros((0, 2), dtype=np.int64)).shape == (0,)


# ------------------------------------------------------------------ certify_cluster
def _within(cert, value, rel=1e-12):
    """``value``, a floating-point objective, lies in the certified bracket up
    to its own rounding."""
    return (cert.lower <= value * (1 + rel) + 1e-300
            and value <= cert.upper * (1 + rel) + 1e-300)


def _search(B, A, m, **kw):
    return search_cluster(B, A, [m], filt=kw.pop("filt", SubsetFilter()), **kw)[0]


@pytest.mark.parametrize("penal", [False, True])
def test_a_cluster_certificate_brackets_the_returned_design(penal):
    B, A = _data(3, 14, 9)
    lin_t = np.linspace(0.1, 1.0, 9) if penal else None
    lin_c = np.linspace(1.0, 0.1, 9) if penal else None
    best = _search(B, A, 3, linear_treated=lin_t, linear_control=lin_c)
    cert = certify_cluster(B, A, [3], SubsetFilter(), best,
                           linear_treated=lin_t, linear_control=lin_c)
    assert isinstance(cert, ClusterCertificate)
    assert _within(cert, best.value)
    assert cert.gap <= 1e-9 * best.value
    assert cert.unresolved == ()


def test_a_suboptimal_incumbent_shows_in_the_gap():
    B, A = _data(4, 14, 8)
    best = _search(B, A, 2)
    designs = search_cluster(B, A, [2], filt=SubsetFilter(), top_k=4)
    worse = designs[3]
    cert = certify_cluster(B, A, [2], SubsetFilter(), worse)
    assert cert.gap >= (worse.value - best.value) * (1 - 1e-9)
    assert best.treated in cert.unresolved
    # the optimum is among the unresolved sets, so the lower bound finds it
    assert best.value * (1 - 1e-9) <= cert.lower <= best.value * (1 + 1e-12)


def test_a_poorly_fitted_design_shows_in_the_gap():
    # The right treated set with uniform weights: the certificate brackets
    # what was returned, so the gap measures the weights' shortfall too.
    B, A = _data(11, 14, 8)
    best = _search(B, A, 3)
    k = len(best.treated)
    w = np.full(k, 1.0 / k)
    poor = dataclasses.replace(best, w=w, value=float("nan"))
    g_poor = float(np.sum((A - B[:, list(best.treated)] @ w) ** 2))
    g_best, _ = side_fit(B, A, list(best.treated))
    cert = certify_cluster(B, A, [3], SubsetFilter(), poor)
    assert cert.lower <= best.value * (1 + 1e-12)
    assert cert.gap >= (g_poor - g_best) * (1 - 1e-9)


def test_the_certificate_reads_the_weights_not_the_reported_value():
    B, A = _data(12, 14, 8)
    best = _search(B, A, 2)
    told = dataclasses.replace(best, value=0.0)
    ref = certify_cluster(B, A, [2], SubsetFilter(), best)
    cert = certify_cluster(B, A, [2], SubsetFilter(), told)
    assert (cert.lower, cert.upper) == (ref.lower, ref.upper)


def test_an_exact_tie_is_reported_unresolved():
    B, A = _data(6, 14, 8)
    B[:, 7] = B[:, 0]
    A = B.mean(axis=1)
    best = _search(B, A, 2)
    twin = tuple(sorted(7 if j == 0 else 0 if j == 7 else j for j in best.treated))
    cert = certify_cluster(B, A, [2], SubsetFilter(), best)
    if twin != best.treated and (0 in best.treated or 7 in best.treated):
        assert twin in cert.unresolved
    assert _within(cert, best.value)


def test_a_certificate_honours_the_restrictions():
    B, A = _data(7, 12, 8)
    filt = SubsetFilter(forced_in=frozenset([5]), forbidden=frozenset([0, 1]))
    best = _search(B, A, 3, filt=filt)
    cert = certify_cluster(B, A, [3], filt, best)
    assert _within(cert, best.value) and cert.gap <= 1e-9 * best.value


def test_a_certificate_with_donor_exclusion_prices_the_right_controls():
    B, A = _data(8, 12, 7)
    donors = [(0, 1), (2, 3)]
    best = _search(B, A, 2, donor_exclusion=donors)
    cert = certify_cluster(B, A, [2], SubsetFilter(), best, donor_exclusion=donors)
    assert _within(cert, best.value) and cert.gap <= 1e-9 * best.value


def test_a_candidate_with_no_control_is_not_a_design():
    # Markets 0 and 1 straddle the target, so (0, 1) fits it exactly and is
    # priced under every design; treating them excludes the other two, so it
    # leaves no control and is no design at all.
    rng = np.random.default_rng(10)
    A = rng.normal(size=10)
    e = rng.normal(size=10)
    B = np.column_stack([A + e, A - e, A + 3 * rng.normal(size=10),
                         A + 3 * rng.normal(size=10)])
    donors = [(0, 2), (1, 3)]
    best = _search(B, A, 2, donor_exclusion=donors)
    assert best.treated != (0, 1)
    cert = certify_cluster(B, A, [2], SubsetFilter(), best, donor_exclusion=donors)
    assert (0, 1) not in cert.unresolved
    assert _within(cert, best.value) and cert.gap <= 1e-9 * best.value


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(3, 16), n=st.integers(4, 7),
       m=st.integers(1, 3), penal=st.booleans(), donors=st.booleans())
def test_a_certificate_brackets_the_brute_force_optimum(seed, T, n, m, penal, donors):
    # Every treated set fitted: the optimum lies in the bracket, and every set
    # that can match the design is listed as unresolved, and no other.
    rng = np.random.default_rng(seed)
    B, A = _data(seed, T, n)
    lin_t = rng.uniform(0, 1, n) if penal else None
    lin_c = rng.uniform(0, 1, n) if penal else None
    excl = [(0, n - 1), (1, n - 2)] if donors else []
    best = _search(B, A, m, linear_treated=lin_t, linear_control=lin_c,
                   donor_exclusion=excl)
    cert = certify_cluster(B, A, [m], SubsetFilter(), best, linear_treated=lin_t,
                           linear_control=lin_c, donor_exclusion=excl)
    values = {}
    for S in itertools.combinations(range(n), m):
        drop = set(S) | {j for i, j in excl if i in S}
        control = [j for j in range(n) if j not in drop]
        if not control:
            continue
        g, _ = side_fit(B, A, list(S), None if lin_t is None else lin_t[list(S)])
        h, _ = side_fit(B, A, control, None if lin_c is None else lin_c[control])
        values[S] = g + h
    fstar = min(values.values())
    assert cert.lower <= fstar * (1 + 1e-12) + 1e-300
    assert _within(cert, best.value)
    for S, f in values.items():
        if S == best.treated:
            continue
        if f <= cert.upper * (1 - 1e-9):
            assert S in cert.unresolved, S
        if S in cert.unresolved:
            assert f <= cert.upper * (1 + 1e-9), S


def test_a_design_with_the_wrong_control_pool_is_refused():
    # The certificate bounds the design's own set through its control fit, so
    # that fit has to be over the pool the treated set leaves.
    B, A = _data(13, 12, 6)
    best = _search(B, A, 2)
    short = dataclasses.replace(best, control=best.control[1:], v=best.v[1:])
    with pytest.raises(MlsynthEstimationError, match="control"):
        certify_cluster(B, A, [2], SubsetFilter(), short)


def test_a_better_control_pool_is_not_discarded_on_the_design_s():
    # Markets 0 and 1 straddle the target, so treating them fits it exactly,
    # but their donor exclusions leave only poor controls. Treating 2 and 3
    # fits worse and leaves 0 and 1 as controls, which fit exactly. Given the
    # first design, the certificate has to see that the second is better: its
    # control floor comes from the whole cluster, not from the design's pool.
    rng = np.random.default_rng(14)
    T = 10
    A = rng.normal(size=T)
    e = rng.normal(size=T)
    B = np.column_stack([A + e, A - e,
                         A + 0.3 * rng.normal(size=T), A + 0.3 * rng.normal(size=T),
                         A + 5 * rng.normal(size=T), A + 5 * rng.normal(size=T),
                         A + 5 * rng.normal(size=T)])
    donors = [(0, 2), (0, 3), (1, 2), (1, 3)]
    designs = search_cluster(B, A, [2], filt=SubsetFilter(), donor_exclusion=donors,
                             top_k=21)
    pair = next(d for d in designs if d.treated == (0, 1))
    best = designs[0]
    assert best.value < pair.value and best.treated != (0, 1)
    cert = certify_cluster(B, A, [2], SubsetFilter(), pair, donor_exclusion=donors)
    assert best.treated in cert.unresolved
    assert cert.lower <= best.value * (1 + 1e-12)


def test_a_degenerate_cluster_is_still_certified():
    # Every market identical: every closed form is singular, every bound zero,
    # so every candidate is fitted exactly and all of them tie.
    B = np.tile(np.linspace(0, 1, 6)[:, None], (1, 5)) + 0.0
    A = B.mean(axis=1)
    best = _search(B, A, 2)
    cert = certify_cluster(B, A, [2], SubsetFilter(), best)
    assert _within(cert, best.value)
    assert len(cert.unresolved) == 9


def test_a_certificate_refuses_the_coupled_design():
    B, A = _data(9, 10, 6)
    best = _search(B, A, 2)
    with pytest.raises(MlsynthConfigError, match="weakly"):
        certify_cluster(B, A, [2], SubsetFilter(), best, beta=0.5)


# ------------------------------------------------------------------ end to end
def test_marex_returns_a_certificate_when_asked():
    df, T0 = _panel(seed=1)
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, certify=True))).fit()
    c = res.certificate
    assert c is not None
    assert c.lower <= c.upper and c.gap == pytest.approx(c.upper - c.lower)
    assert c.gap <= 1e-9 * max(1.0, abs(c.upper))
    assert set(c.clusters) == set(res.clusters)
    assert _within(c, res.globres.objective)


def test_marex_certificates_add_over_clusters():
    df, T0 = _panel(J=10, seed=2, region_sizes=[5, 5])
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, certify=True,
                                   cluster="region"))).fit()
    c = res.certificate
    assert len(c.clusters) == 2
    assert c.lower <= sum(k.upper for k in c.clusters.values())
    assert c.lower == pytest.approx(sum(k.lower for k in c.clusters.values()), rel=1e-12)
    assert c.upper == pytest.approx(sum(k.upper for k in c.clusters.values()), rel=1e-12)


def test_marex_reports_ties_by_unit_label():
    # u7 copies u0 inside the second region, so any design treating one of the
    # two ties with the design treating the other; the certificate names the
    # tied sets by unit label.
    df, T0 = _panel(J=10, seed=4, region_sizes=[3, 7])
    y = df.set_index(["unit", "time"])["y"]
    df.loc[df.unit == "u7", "y"] = y.loc["u3"].to_numpy()
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=1, certify=True,
                                   cluster="region"))).fit()
    c = res.certificate
    labels = set(df.unit)
    for k in c.clusters.values():
        for S in k.unresolved:
            assert set(S) <= labels
    treated = set(map(str, res.assignment["treated"]))
    if treated & {"u3", "u7"}:
        twin = ({"u3", "u7"} - treated).pop()
        assert any(twin in S for k in c.clusters.values() for S in k.unresolved)
    assert _within(c, res.globres.objective)


def test_marex_without_certify_has_no_certificate():
    df, T0 = _panel(seed=1)
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=2))).fit()
    assert res.certificate is None


def test_certify_needs_the_exact_backend():
    df, T0 = _panel(seed=1)
    with pytest.raises(MlsynthConfigError, match="certify"):
        MAREXConfig(**_cfg(df, T0, "mip", m_eq=2, certify=True))


def test_certify_refuses_the_weakly_targeted_design():
    df, T0 = _panel(seed=1)
    with pytest.raises(MlsynthConfigError, match="certify"):
        MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, certify=True,
                           design="weakly_targeted"))
