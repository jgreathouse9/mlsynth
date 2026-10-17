"""MAREX's exact backend: bounds that see the penalties.

The search discards a candidate treated set when a lower bound on its
objective clears the best design found so far. For the ``penalized`` and
``unit_penalized`` designs the only bound used to be the closed form on the
treated fit, ``min { w'G w + a'w : 1'w = 1 }``, which lets the weights go
negative: under a large penalty it buys a low ``a'w`` with negative weight on
the costly markets and falls to zero. Two bounds that see the penalties sit
beside it.

* The treated side: ``1/s(S) + min over S of a``, since the minimum of a sum
  is at least the sum of the minimums. The search takes the larger of this and
  the closed form.
* The control side: a floor under every candidate's control fit. A candidate's
  controls are a subset of the cluster, and a simplex fit only falls as markets
  are added, so the penalised fit over the whole cluster sits under all of
  them. The floor is that fit's certified lower bound, its value less its
  Frank-Wolfe gap.

The tests check that each is a lower bound, that each is as strong as it
claims, that the search still reaches the enumerated optimum under heavy
penalties, and that the bounds are what let it fit fewer candidates.

Levels: smoke, differential, unit, edge, failure, property.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.exact import (
    search_cluster,
    side_fit,
    treated_fit_bound,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import (
    _assert_same_design,
    _both,
    _fit as _marex_fit,
    _panel,
    _problem,
    _treated,
)


# ------------------------------------------------------------------ helpers
def _penalty(seed, n, scale):
    return scale * np.abs(np.random.default_rng(seed).normal(size=n))


def _subsets(n, m):
    return np.array(list(itertools.combinations(range(n), m)), dtype=np.int64)


def _fit(B, A, cols, linear):
    cols = list(cols)
    return side_fit(B, A, cols, None if linear is None else linear[cols])[0]


def _floor(B, A, linear):
    return exact_mod.control_fit_floor(B, A, linear)


def _closed_form(B, A, S, a):
    """The shipped bound for one candidate: ``(1/s, closed form)``, or ``None``.

    ``None`` where ``G_S`` is too ill-conditioned for the reference solve to
    be trusted; the comparisons below skip those rows.
    """
    Bc = B[:, list(S)] - A[:, None]
    G = Bc.T @ Bc
    if np.linalg.cond(G) > 1e8:
        return None
    one = np.ones(len(S))
    s = one @ np.linalg.solve(G, one)
    if s <= 0:
        return None
    if a is None:
        return 1.0 / s, 1.0 / s
    xa = np.linalg.solve(G, a)
    t, u = one @ xa, a @ xa
    return 1.0 / s, (2.0 + t) ** 2 / (4.0 * s) - u / 4.0


def _shipped_bound(B, A, subsets, linear=None):
    """The closed form alone, clipped at zero: the bound before the penalties."""
    out = np.zeros(len(subsets))
    for r, S in enumerate(subsets):
        cf = _closed_form(B, A, S, None if linear is None else linear[list(S)])
        out[r] = 0.0 if cf is None else max(cf[1], 0.0)
    return out


def _enumerate(B, A, m, lin_t=None, lin_c=None, donors=()):
    """Every treated set of size ``m`` with its objective, best first."""
    n = B.shape[1]
    rows = []
    for S in itertools.combinations(range(n), m):
        excluded = {j for i, j in donors if i in S}
        C = [j for j in range(n) if j not in S and j not in excluded]
        if not C:
            continue
        rows.append((_fit(B, A, S, lin_t) + _fit(B, A, C, lin_c), S))
    return sorted(rows)


def _count_fits(monkeypatch):
    """Wrap the exact fit so a test can read the calls the search makes, in order."""
    calls = []
    original = exact_mod.side_fit

    def counted(B, A, cols, linear=None):
        out = original(B, A, cols, linear)
        calls.append((tuple(cols), out[0]))
        return out

    monkeypatch.setattr(exact_mod, "side_fit", counted)
    return calls


def _heavy(seed, n=9, T=8, scale=5.0):
    B, A = _problem(seed, n=n, T=T)
    return B, A, _penalty(seed + 1, n, scale), _penalty(seed + 2, n, scale)


# --------------------------------------------------------------------- smoke
def test_a_heavily_penalised_search_returns_the_enumerated_optimum():
    B, A, lin_t, lin_c = _heavy(30)
    got = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c)[0]
    value, S = _enumerate(B, A, 3, lin_t, lin_c)[0]
    assert got.treated == S
    assert got.value == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_the_floor_is_a_finite_non_negative_number():
    B, A, _, lin_c = _heavy(31)
    floor = _floor(B, A, lin_c)
    assert isinstance(floor, float)
    assert np.isfinite(floor) and floor >= 0.0


# -------------------------------------------- differential: exact == MIQP
# Comparing treated sets is only valid where the optimum is unique, and two
# backends may return different members of a tie (#779). lambda1 == lambda2 is
# avoided: the penalised objective is then the same function on both sides, so
# a design and its mirror tie exactly, and CI's SCIP returned the mirror of the
# exact backend's design at 1.0 / 1.0. _assert_unique_optimum checks the
# premise on each panel before the sets are compared.
UNIQUE_GAP = 1e-4


def _assert_unique_optimum(df, T0, exact, **over):
    """No other treated support comes within UNIQUE_GAP of the optimum.

    Any other optimal support either omits a market of this one or contains
    all of them; forbidding each market in turn finds the first kind, the
    mirror included, since a mirror's treated markets are this design's
    controls. The second kind, a strict superset at an equal objective, is
    not checked.
    """
    best = exact.globres.objective
    for unit in sorted(_treated(exact)):
        other = _marex_fit(df, T0, "exact", not_to_be_treated=[unit], **over)
        gap = (other.globres.objective - best) / abs(best)
        assert gap > UNIQUE_GAP, (
            f"forbidding {unit} costs {gap:.2e}: the optimum is tied with "
            f"{sorted(_treated(other))}, so the treated sets are not comparable"
        )


@pytest.mark.parametrize("design, extra", [
    ("penalized", {"lambda1": 1.0, "lambda2": 0.7}),
    ("penalized", {"lambda1": 0.0, "lambda2": 1.0}),
    ("penalized", {"lambda1": 1.0, "lambda2": 0.0}),
    ("unit_penalized", {"xi": 1.0, "lambda1_unit": 0.1}),
])
def test_exact_matches_the_mip_under_heavy_penalties(design, extra):
    df, T0 = _panel(seed=40)
    exact, mip = _both(df, T0, m_eq=3, design=design, **extra)
    _assert_unique_optimum(df, T0, exact, m_eq=3, design=design, **extra)
    _assert_same_design(exact, mip)


def test_exact_matches_the_mip_under_heavy_penalties_with_two_clusters():
    df, T0 = _panel(J=12, seed=41, region_sizes=(6, 6))
    over = dict(m_eq=2, cluster="region", design="penalized",
                lambda1=1.0, lambda2=0.7)
    exact, mip = _both(df, T0, **over)
    _assert_unique_optimum(df, T0, exact, **over)
    _assert_same_design(exact, mip)


def test_the_uniqueness_check_reports_the_mirror_tie():
    df, T0 = _panel(seed=40)
    over = dict(m_eq=3, design="penalized", lambda1=1.0, lambda2=1.0)
    exact = _marex_fit(df, T0, "exact", **over)
    with pytest.raises(AssertionError, match="tied with"):
        _assert_unique_optimum(df, T0, exact, **over)


# ------------------------------------------------- unit: the treated bound
def test_the_treated_bound_is_at_least_the_sum_of_the_minimums():
    B, A, lin_t, _ = _heavy(32)
    subs = _subsets(B.shape[1], 3)
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    plain = treated_fit_bound(B, A, subs)
    mins = lin_t[subs].min(axis=1)
    assert np.all(lb >= plain + mins - 1e-9 * (1.0 + lb))


def test_the_treated_bound_is_at_least_the_shipped_closed_form():
    B, A, lin_t, _ = _heavy(33, scale=0.5)
    subs = _subsets(B.shape[1], 3)
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    shipped = _shipped_bound(B, A, subs, lin_t)
    assert np.all(lb >= shipped - 1e-9 * (1.0 + shipped))
    assert np.any(lb > 0.0)


def test_a_large_penalty_no_longer_drives_the_treated_bound_to_zero():
    """The closed form spends negative weight on the costly markets."""
    B, A = _problem(34)
    n = B.shape[1]
    lin_t = 50.0 + _penalty(35, n, 50.0)
    subs = _subsets(n, 3)
    shipped = _shipped_bound(B, A, subs, lin_t)
    assert np.any(shipped < lin_t[subs].min(axis=1))
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    assert np.all(lb >= lin_t[subs].min(axis=1))


def test_the_treated_bound_never_exceeds_the_treated_fit_under_heavy_penalties():
    B, A = _problem(36)
    n = B.shape[1]
    for scale in (0.01, 1.0, 100.0):
        lin_t = _penalty(37, n, scale)
        for m in (1, 2, 3, 4):
            subs = _subsets(n, m)
            lb = treated_fit_bound(B, A, subs, linear=lin_t)
            fit = np.array([_fit(B, A, S, lin_t) for S in subs])
            assert np.all(lb <= fit + 1e-8 * (1.0 + fit))


def test_equal_penalties_shift_the_bound_by_the_penalty():
    """With ``a = c 1`` the two forms coincide at ``1/s + c``."""
    B, A = _problem(38)
    n = B.shape[1]
    subs = _subsets(n, 3)
    c = 2.5
    lb = treated_fit_bound(B, A, subs, linear=np.full(n, c))
    plain = treated_fit_bound(B, A, subs)
    np.testing.assert_allclose(lb, plain + c, rtol=1e-9)


def test_one_treated_market_is_bounded_exactly():
    B, A, lin_t, _ = _heavy(39)
    subs = _subsets(B.shape[1], 1)
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    fit = np.array([_fit(B, A, S, lin_t) for S in subs])
    np.testing.assert_allclose(lb, fit, rtol=1e-9)


# --------------------------------------------------- unit: the control floor
def test_the_floor_sits_under_every_control_pool():
    B, A, _, lin_c = _heavy(42, n=7)
    floor = _floor(B, A, lin_c)
    for size in range(1, 8):
        for C in itertools.combinations(range(7), size):
            h = _fit(B, A, C, lin_c)
            assert floor <= h + 1e-9 * (1.0 + h)


@pytest.mark.parametrize("twin", [0, 4, 8])
def test_the_floor_counts_every_market_of_the_cluster(twin):
    """A market that carries the control fit on its own must be in the floor.

    One market is the target's twin and cheap, so the fit over the whole
    cluster rests on it; a floor fitted without it sits above that fit.
    """
    B, A = _problem(43, n=9)
    B[:, twin] = A + 1e-3
    lin_c = 1.0 + _penalty(44, 9, 1.0)
    lin_c[twin] = 1e-3
    h_all = _fit(B, A, range(9), lin_c)
    assert h_all < 0.5 * _fit(B, A, [j for j in range(9) if j != twin], lin_c)
    assert _floor(B, A, lin_c) <= h_all + 1e-12


def test_the_floor_is_the_fit_over_the_whole_cluster():
    B, A, _, lin_c = _heavy(45)
    h_all = _fit(B, A, range(B.shape[1]), lin_c)
    assert h_all > 0.1
    assert _floor(B, A, lin_c) == pytest.approx(h_all, rel=1e-9)


def test_the_floor_holds_when_the_solver_stops_short(monkeypatch):
    """The floor is the value less the gap, so it holds at any point.

    The solver is made to stop at the uniform weights, a feasible point well
    above the minimum. The fit's value there exceeds the true minimum; its
    value less its gap does not.
    """
    B, A, _, lin_c = _heavy(46)
    n = B.shape[1]
    h_all = _fit(B, A, range(n), lin_c)
    uniform = np.full(n, 1.0 / n)
    r = A - B @ uniform
    assert r @ r + lin_c @ uniform > 1.5 * h_all
    monkeypatch.setattr(exact_mod, "solve_simplex_qp",
                        lambda X, y, linear=None: np.full(X.shape[1],
                                                          1.0 / X.shape[1]))
    floor = _floor(B, A, lin_c)
    assert floor <= h_all + 1e-12


# ------------------------------------------------------- unit: the search
def test_the_floor_spares_fits_the_treated_bound_cannot(monkeypatch):
    B, A, lin_t, lin_c = _heavy(47)
    calls = _count_fits(monkeypatch)
    with_floor = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                                linear_treated=lin_t, linear_control=lin_c)[0]
    fits_with = len(calls)
    calls.clear()
    monkeypatch.setattr(exact_mod, "control_fit_floor",
                        lambda B, A, linear=None: 0.0)
    without = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                             linear_treated=lin_t, linear_control=lin_c)[0]
    assert with_floor.treated == without.treated
    assert with_floor.value == pytest.approx(without.value, rel=1e-12)
    assert fits_with < len(calls)


def test_no_candidate_gets_a_control_fit_its_treated_fit_and_the_floor_rule_out(
        monkeypatch):
    """The floor applies to a fitted treated side too, not only to its bound.

    A candidate whose bound is under the incumbent still has its treated side
    fitted; if that fit plus the floor clears the incumbent, the control fit
    is not paid for. The calls are read in order: a fit over ``m`` markets is
    a treated side, over ``n - m`` a control pool, over ``n`` the floor.
    """
    n, m = 10, 3
    B, A, lin_t, lin_c = _heavy(42, n=n, scale=10.0)
    floor = _floor(B, A, lin_c)
    calls = _count_fits(monkeypatch)
    search_cluster(B, A, sizes=[m], filt=SubsetFilter(),
                   linear_treated=lin_t, linear_control=lin_c)
    best, g, spared = np.inf, None, 0
    for cols, value in calls + [((None,) * m, None)]:
        if len(cols) == m:
            if g is not None and g < best:
                spared += 1
            g = value
        elif len(cols) == n - m:
            cut = best + exact_mod.PRUNE_MARGIN * (1.0 + abs(best))
            assert g + floor < cut
            best = min(best, g + value)
            g = None
    assert spared > 0


@pytest.mark.parametrize("buffer", [exact_mod.BUFFER, 2])
def test_no_candidate_is_fitted_whose_bound_and_the_floor_clear_the_incumbent(
        monkeypatch, buffer):
    """Both passes add the floor to every bound before comparing it.

    Each treated fit the search pays for is checked against the incumbent at
    the time of the call. A two-candidate buffer forces the second pass.
    """
    monkeypatch.setattr(exact_mod, "BUFFER", buffer)
    n, m = 10, 3
    B, A, lin_t, lin_c = _heavy(42, n=n, scale=10.0)
    floor = _floor(B, A, lin_c)
    calls = _count_fits(monkeypatch)
    search_cluster(B, A, sizes=[m], filt=SubsetFilter(),
                   linear_treated=lin_t, linear_control=lin_c)
    best, g = np.inf, None
    for cols, value in calls:
        if len(cols) == m:
            lb = treated_fit_bound(B, A, np.array([cols]), linear=lin_t)[0]
            assert lb + floor < best + exact_mod.PRUNE_MARGIN * (1.0 + abs(best))
            g = value
        elif len(cols) == n - m:
            best = min(best, g + value)
    lb_all = treated_fit_bound(B, A, _subsets(n, m), linear=lin_t)
    cut = best + exact_mod.PRUNE_MARGIN * (1.0 + abs(best))
    assert np.any((lb_all < best) & (lb_all + floor >= cut))


def test_the_penalty_bounds_spare_fits_the_shipped_bound_pays_for(monkeypatch):
    """The search with both bounds fits fewer candidates than without them."""
    B, A, lin_t, lin_c = _heavy(49, n=10)
    calls = _count_fits(monkeypatch)
    new = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c)[0]
    fits_new = len(calls)
    calls.clear()
    monkeypatch.setattr(exact_mod, "treated_fit_bound", _shipped_bound)
    monkeypatch.setattr(exact_mod, "control_fit_floor",
                        lambda B, A, linear=None: 0.0)
    old = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c)[0]
    assert new.treated == old.treated
    assert fits_new < len(calls)


# ----------------------------------------------------------- edge cases
def test_duplicate_markets_still_get_the_smallest_penalty():
    """A singular Gram leaves no closed form; the penalty term still bounds."""
    B, A = _problem(50)
    n = B.shape[1]
    B[:, 1] = B[:, 0]
    lin_t = 1.0 + _penalty(51, n, 2.0)
    subs = np.array([S for S in _subsets(n, 3) if 0 in S and 1 in S])
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    fit = np.array([_fit(B, A, S, lin_t) for S in subs])
    assert np.all(lb >= lin_t[subs].min(axis=1))
    assert np.all(lb <= fit + 1e-8 * (1.0 + fit))


def test_a_single_control_market_under_heavy_penalties():
    B, A, lin_t, lin_c = _heavy(52, n=6)
    got = search_cluster(B, A, sizes=[5], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c)[0]
    value, S = _enumerate(B, A, 5, lin_t, lin_c)[0]
    assert got.treated == S
    assert got.value == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_a_zero_penalty_changes_nothing():
    B, A = _problem(53)
    n = B.shape[1]
    subs = _subsets(n, 3)
    np.testing.assert_allclose(
        treated_fit_bound(B, A, subs, linear=np.zeros(n)),
        treated_fit_bound(B, A, subs), rtol=1e-9, atol=1e-12)
    assert _floor(B, A, np.zeros(n)) == pytest.approx(0.0, abs=1e-10)


def test_without_a_control_penalty_the_floor_is_zero():
    """The target is the cluster mean, which the whole cluster reproduces."""
    B, A = _problem(54)
    assert _floor(B, A, None) == pytest.approx(0.0, abs=1e-10)


def test_donor_exclusions_under_heavy_penalties():
    B, A, lin_t, lin_c = _heavy(55, n=7)
    donors = ((0, 6), (1, 5), (2, 4))
    got = search_cluster(B, A, sizes=[2], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c,
                         donor_exclusion=donors)[0]
    value, S = _enumerate(B, A, 2, lin_t, lin_c, donors)[0]
    assert got.treated == S
    assert got.value == pytest.approx(value, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------- failure
@pytest.mark.parametrize("which", ["linear_treated", "linear_control"])
@pytest.mark.parametrize("bad", [-1e-3, np.nan, np.inf])
def test_a_penalty_that_is_negative_or_not_finite_is_refused(
        monkeypatch, which, bad):
    """The bounds take both fits to be non-negative, so the search refuses first."""
    B, A = _problem(56)
    vec = np.ones(B.shape[1])
    vec[3] = bad

    def never(*args, **kwargs):
        raise AssertionError("a fit ran before the penalty was checked")

    monkeypatch.setattr(exact_mod, "side_fit", never)
    with pytest.raises(MlsynthConfigError, match=which):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(), **{which: vec})


# ------------------------------------------------------------ properties
_scale = st.sampled_from([1e-2, 1e-1, 1.0, 10.0, 100.0])


@given(seed=st.integers(0, 10_000), n=st.integers(3, 8), m=st.integers(1, 4),
       scale=_scale, donor=st.booleans())
@settings(max_examples=60, deadline=None)
def test_the_combined_bound_never_exceeds_a_candidates_objective(
        seed, n, m, scale, donor):
    m = min(m, n - 1)
    B, A = _problem(seed, n=n, T=7)
    lin_t, lin_c = _penalty(seed + 1, n, scale), _penalty(seed + 2, n, scale)
    donors = ((0, n - 1), (1, n - 2)) if donor and n > 3 else ()
    subs = _subsets(n, m)
    lb = treated_fit_bound(B, A, subs, linear=lin_t) + _floor(B, A, lin_c)
    for r, S in enumerate(subs):
        excluded = {j for i, j in donors if i in S}
        C = [j for j in range(n) if j not in S and j not in excluded]
        if not C:
            continue
        f = _fit(B, A, S, lin_t) + _fit(B, A, C, lin_c)
        assert lb[r] <= f + 1e-8 * (1.0 + abs(f))


@given(seed=st.integers(0, 10_000), n=st.integers(2, 8), m=st.integers(1, 5),
       scale=_scale)
@settings(max_examples=60, deadline=None)
def test_the_treated_bound_is_at_least_each_of_its_parts(seed, n, m, scale):
    m = min(m, n)
    B, A = _problem(seed, n=n, T=7)
    lin_t = _penalty(seed + 1, n, scale)
    subs = _subsets(n, m)
    lb = treated_fit_bound(B, A, subs, linear=lin_t)
    mins = lin_t[subs].min(axis=1)
    assert np.all(lb >= mins)
    for r, S in enumerate(subs):
        cf = _closed_form(B, A, S, lin_t[list(S)])
        if cf is None:
            continue
        tol = 1e-8 * (1.0 + abs(cf[0]) + mins[r])
        assert lb[r] >= cf[0] + mins[r] - tol
        assert lb[r] >= cf[1] - 1e-8 * (1.0 + abs(cf[1]))


@given(seed=st.integers(0, 10_000), n=st.integers(2, 7), scale=_scale,
       target=st.sampled_from(["mean", "off"]))
@settings(max_examples=60, deadline=None)
def test_the_floor_sits_under_every_control_pool_over_the_domain(
        seed, n, scale, target):
    B, A = _problem(seed, n=n, T=6)
    if target == "off":
        A = A + np.random.default_rng(seed + 3).normal(size=A.shape) * 2.0
    lin_c = _penalty(seed + 1, n, scale)
    floor = _floor(B, A, lin_c)
    assert floor >= 0.0
    for size in range(1, n + 1):
        for C in itertools.combinations(range(n), size):
            h = _fit(B, A, C, lin_c)
            assert floor <= h + 1e-9 * (1.0 + h)


@given(seed=st.integers(0, 10_000), n=st.integers(3, 8), m=st.integers(1, 4),
       scale=_scale, top_k=st.integers(1, 3), donor=st.booleans())
@settings(max_examples=60, deadline=None)
def test_the_penalised_search_is_exact_over_the_domain(
        seed, n, m, scale, top_k, donor):
    m = min(m, n - 1)
    B, A = _problem(seed, n=n, T=7)
    lin_t, lin_c = _penalty(seed + 1, n, scale), _penalty(seed + 2, n, scale)
    donors = ((0, n - 1), (1, n - 2)) if donor and n > 3 else ()
    want = _enumerate(B, A, m, lin_t, lin_c, donors)[:top_k]
    got = search_cluster(B, A, sizes=[m], filt=SubsetFilter(),
                         linear_treated=lin_t, linear_control=lin_c,
                         donor_exclusion=donors, top_k=top_k)
    np.testing.assert_allclose([d.value for d in got], [v for v, _ in want],
                               rtol=1e-9, atol=1e-12)
