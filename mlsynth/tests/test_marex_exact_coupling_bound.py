"""The coupling bound: a per-candidate lower bound that sees the weakly-targeted
coupling (issue #770).

The weakly-targeted objective, with Abadie and L'Hour's control penalty ``lam``
(zero without it), is

    f(S) = min ||A - B_S w||^2 + beta ||B_S w - B_C v||^2
                               + beta lam sum_j v_j ||B_S w - b_j||^2.

The penalty is at least ``beta lam ||B_S w - B_C v||^2`` by convexity of the
square, and ``||B_S w - B_C v||`` is at least the distance between the two
hulls, so

    f(S) >= g(S) + beta (1 + lam) dist(conv B_S, conv B_C)^2,

with ``g`` the treated fit. Wolfe's (1976) nearest-point method finds a
direction ``z``; his Theorem 2.2 turns any direction into a lower bound on the
distance, ``(min_{i in S} z'b_i - max_{j in C} z'b_j) / |z|``. The bound is
certified from ``z`` and the data alone, with the rounding of those products
bounded a priori, so neither an inaccurate iteration nor floating point can
make it exceed the distance.

References: a conic solve of the distance (Clarabel), the coupled fit for the
objective, exact rational arithmetic for the certificate, and plain
enumeration for the search. Levels: smoke, unit, edge, failure, property.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.exact import (
    CouplingBound,
    coupled_fit,
    hull_separation,
    search_cluster,
    side_fit,
    treated_fit_bound,
)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter
from mlsynth.tests.test_marex_exact_backend import _problem


# ------------------------------------------------------------------ helpers
def _distance2(B, S, C):
    """The squared distance between the two hulls, by a conic solve."""
    Bs, Bc = B[:, list(S)], B[:, list(C)]
    w = cp.Variable(len(S), nonneg=True)
    v = cp.Variable(len(C), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Bs @ w - Bc @ v)),
                      [cp.sum(w) == 1, cp.sum(v) == 1])
    prob.solve(solver=cp.CLARABEL)
    return max(float(prob.value), 0.0)


def _under(d, ref2):
    """``d`` is under the distance whose square is ``ref2``, up to the conic
    solve's own accuracy; squares, since near zero a root amplifies it."""
    return d * d <= ref2 * (1 + 1e-6) + 1e-7


def _controls(n, S, excluded=()):
    drop = set(S)
    for i, j in excluded:
        if i in S:
            drop.add(j)
    return [j for j in range(n) if j not in drop]


def _subsets(n, m):
    return np.array(list(itertools.combinations(range(n), m)), dtype=np.int64)


def _masks(n, S, C):
    tm = np.zeros((1, n), dtype=bool)
    cm = np.zeros((1, n), dtype=bool)
    tm[0, list(S)] = True
    cm[0, list(C)] = True
    return tm, cm


def _separated(seed, n=8, T=6, gap=3.0):
    """Two clouds of markets ``gap`` apart along the first axis."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, n))
    B[0, : n // 2] += gap
    return B, B.mean(axis=1)


def _exact_theorem(B, z, S, C):
    """Wolfe's Theorem 2.2 for direction ``z``, in exact rational arithmetic:
    the numerator ``min_S z'b - max_C z'b`` and ``|z|^2``."""
    zf = [Fraction(float(x)) for x in z]
    p = {j: sum(zf[t] * Fraction(float(B[t, j])) for t in range(B.shape[0]))
         for j in set(S) | set(C)}
    num = min(p[j] for j in S) - max(p[j] for j in C)
    return num, sum(x * x for x in zf)


def _assert_under_theorem(d, B, z, S, C):
    num, zz = _exact_theorem(B, z, S, C)
    if num <= 0 or zz == 0:
        assert d == 0.0
    else:
        assert Fraction(float(d)) ** 2 * zz <= num ** 2


def _coupled(B, A, S, C, beta, lam=0.0):
    return coupled_fit(B, A, list(S), list(C), beta, control_penalty=lam)[0]


def _brute_force(B, A, m, beta, lam=0.0, excluded=(), k=1):
    n = B.shape[1]
    vals = []
    for S in itertools.combinations(range(n), m):
        C = _controls(n, S, excluded)
        if C:
            vals.append((_coupled(B, A, S, C, beta, lam), S))
    vals.sort()
    return vals[:k]


# --------------------------------------------------------------------- smoke
def test_the_bound_prices_a_batch():
    B, A = _problem(0, n=8)
    sets = _subsets(8, 3)
    cb = CouplingBound(B, A, beta=1.0)
    d = cb.distance(sets)
    assert d.shape == (len(sets),)
    assert np.isfinite(d).all() and (d >= 0).all()
    lb = treated_fit_bound(B, A, sets)
    bound = cb.bound(sets, lb)
    assert bound.shape == (len(sets),) and (bound >= lb).all()


def test_the_screened_search_matches_plain_enumeration_at_large_beta():
    B, A = _problem(8, n=12)
    d = search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=100.0)[0]
    (value, S), = _brute_force(B, A, 3, 100.0)
    assert d.treated == S
    assert d.value == pytest.approx(value, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------- unit
@pytest.mark.parametrize("seed", range(4))
def test_the_bound_is_under_the_hull_distance(seed):
    B, A = _problem(seed, n=8)
    cb = CouplingBound(B, A, beta=1.0)
    sets = _subsets(8, 3)[::5]
    d = cb.distance(sets)
    for S, di in zip(sets, d):
        assert _under(di, _distance2(B, S, _controls(8, S)))


@pytest.mark.parametrize("seed", range(4))
def test_run_to_the_end_the_bound_is_the_hull_distance(seed):
    """Wolfe's method converges to the nearest point, where his bound is the
    distance itself; a weak bound would pass the validity test above."""
    B, A = _separated(seed)
    cb = CouplingBound(B, A, beta=1.0)
    sets = _subsets(8, 2)
    d = cb.distance(sets)
    for S, di in zip(sets, d):
        ref2 = _distance2(B, S, _controls(8, S))
        assert di * di == pytest.approx(ref2, rel=1e-5, abs=1e-7)


def test_the_bound_stops_once_it_reaches_what_was_asked():
    B, A = _separated(1)
    cb = CouplingBound(B, A, beta=1.0)
    sets = _subsets(8, 2)
    full = cb.distance(sets)
    need = (0.5 * full) ** 2
    d = cb.distance(sets, need=need)
    reached = full > 0.1
    assert (d[reached] ** 2 >= need[reached] * (1 - 1e-9)).all()
    assert (d <= full * (1 + 1e-9)).all()
    best, status, _ = exact_mod._wolfe_lockstep(
        cb.gram, cb.pair_d2, sets, cb.control_mask(sets), need,
        exact_mod.COUPLING_MAX_ITER, exact_mod.COUPLING_CORRAL)
    assert (status[reached] == exact_mod.WOLFE_CLEARED).all()


def test_the_bound_aims_wolfe_at_the_cutoff(monkeypatch):
    """A candidate stops as soon as its bound clears what the cutoff needs,
    ``(cutoff - lb) / (beta (1 + lam))``; running every one to convergence
    gives the same answers at several times the cost."""
    asked = []
    real = exact_mod._wolfe_lockstep

    def spy(G, pair_d2, sets, cmask, need, max_iter, corral):
        asked.append(np.array(need, dtype=float))
        return real(G, pair_d2, sets, cmask, need, max_iter, corral)

    monkeypatch.setattr(exact_mod, "_wolfe_lockstep", spy)
    B, A = _problem(23, n=8)
    sets = _subsets(8, 3)
    lb = treated_fit_bound(B, A, sets)
    cut = float(np.median(lb)) * 3.0
    CouplingBound(B, A, beta=2.0, control_penalty=0.1).bound(sets, lb, cut)
    np.testing.assert_allclose(np.concatenate(asked),
                               np.maximum((cut - lb) / 2.2, 0.0), rtol=1e-12)


def test_wolfe_reports_why_each_candidate_stopped():
    B, A = _separated(2, n=10, T=8)
    cb = CouplingBound(B, A, beta=1.0)
    sets = _subsets(10, 3)[:40]
    mask = cb.control_mask(sets)
    inf = np.full(len(sets), np.inf)
    run = lambda need, it, k: exact_mod._wolfe_lockstep(
        cb.gram, cb.pair_d2, sets, mask, need, it, k)[1]
    assert (run(np.zeros(len(sets)) + 1e-12, 200, 24)
            == exact_mod.WOLFE_CLEARED).any()
    assert (run(inf, 200, 24) == exact_mod.WOLFE_CONVERGED).all()
    assert (run(inf, 1, 24) == exact_mod.WOLFE_OUT_OF_ITERATIONS).all()
    assert (run(inf, 200, 1) == exact_mod.WOLFE_CORRAL_FULL).any()


def test_the_bound_sees_the_donors_a_treated_market_excludes():
    """Market 1 is the control nearest market 0, 1 away along the second axis;
    the rest of the pool is 5 away along the first. A bound read against the
    whole complement stays under 1 when market 0 excludes market 1,
    and an iteration that chases market 1 settles on a direction along which
    the pool is not separated at all."""
    B = np.array([[0.0, 0.0, 5.0, 5.0, 6.0],
                  [0.0, 1.0, 1.0, -1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])
    A = B.mean(axis=1)
    S = np.array([[0]])
    plain = CouplingBound(B, A, beta=1.0).distance(S)[0]
    cb = CouplingBound(B, A, beta=1.0, donor_exclusion=[(0, 1)])
    assert not cb.control_mask(S)[0, 1]
    assert plain ** 2 == pytest.approx(_distance2(B, [0], [1, 2, 3, 4]),
                                       rel=1e-6)
    assert plain <= 1.0
    assert cb.distance(S)[0] == pytest.approx(5.0, rel=1e-9)


@pytest.mark.parametrize("beta, lam", [(0.01, 0.0), (1.0, 0.0), (100.0, 0.0),
                                       (1.0, 0.5), (0.5, 1.0), (100.0, 0.0099)])
def test_the_combined_bound_is_under_the_coupled_objective(beta, lam):
    B, A = _problem(5, n=8)
    sets = _subsets(8, 3)
    cb = CouplingBound(B, A, beta=beta, control_penalty=lam)
    bound = cb.bound(sets, treated_fit_bound(B, A, sets))
    for S, b in zip(sets, bound):
        f = _coupled(B, A, S, _controls(8, S), beta, lam)
        assert b <= f * (1 + 1e-9) + 1e-12


def test_the_control_penalty_strengthens_the_bound():
    """The penalty is at least beta lam times the squared coupling, so the
    bound scales by (1 + lam); the plain coupling bound is weaker."""
    B, A = _separated(4)
    sets = _subsets(8, 2)
    lb = treated_fit_bound(B, A, sets)
    plain = CouplingBound(B, A, beta=0.5).bound(sets, lb)
    pen = CouplingBound(B, A, beta=0.5, control_penalty=1.0).bound(sets, lb)
    d = CouplingBound(B, A, beta=0.5).distance(sets)
    np.testing.assert_allclose(pen - lb, 2.0 * (plain - lb), rtol=1e-12)
    assert (pen[d > 0] > plain[d > 0]).all()


def test_the_bound_needs_no_treated_fit():
    """It uses the closed-form treated bound it is given, so pricing a
    candidate takes no simplex fit."""
    B, A = _problem(6, n=8)
    sets = _subsets(8, 3)
    lb = treated_fit_bound(B, A, sets)
    cb = CouplingBound(B, A, beta=1.0)
    np.testing.assert_allclose(cb.bound(sets, lb) - lb,
                               cb.distance(sets) ** 2, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("seed", range(3))
def test_the_bound_clears_most_candidates_at_large_beta(seed):
    """What the issue asked of a bound: at beta = 100 it leaves few
    candidates to fit, where the treated-side bound leaves almost all."""
    B, A = _problem(seed, n=12)
    sets = _subsets(12, 3)
    lb = treated_fit_bound(B, A, sets)
    (opt, _), = _brute_force(B, A, 3, 100.0)
    cut = opt * (1 + 1e-6)
    cb = CouplingBound(B, A, beta=100.0)
    left = cb.survivors(sets, lb, cut)
    assert (lb < cut).sum() > 0.8 * len(sets)
    assert left.sum() < 0.1 * len(sets)


# ------------------------------------------------- the certificate, rounded
def _near_touching(seed, scale=1e8, T=6, n=5):
    """Markets at a large offset, with a control moved onto the treated
    minimum along ``z``: the exact numerator sits at the size of the rounding
    in ``z'b``, so a certificate that ignores the rounding overshoots."""
    rng = np.random.default_rng(seed)
    B = scale + rng.normal(size=(T, n))
    z = rng.normal(size=T)
    S, C = [0, 1], list(range(2, n))
    p = z @ B
    j = C[int(np.argmax(p[C]))]
    B[:, j] += (p[S].min() - p[j]) / (z @ z) * z
    return B, z, S, C


@pytest.mark.parametrize("seed", range(60))
def test_the_certificate_is_under_wolfes_bound_in_exact_arithmetic(seed):
    B, z, S, C = _near_touching(seed)
    tm, cm = _masks(B.shape[1], S, C)
    d = hull_separation(B, z[None, :], tm, cm)[0]
    _assert_under_theorem(d, B, z, S, C)


@pytest.mark.parametrize("seed", range(20))
def test_the_certificate_survives_gradual_underflow(seed):
    """At the bottom of the range a product carries an absolute error that no
    relative bound covers."""
    B, z, S, C = _near_touching(seed, scale=1.0)
    B, z = B * 2.0 ** -520, z * 2.0 ** -530
    tm, cm = _masks(B.shape[1], S, C)
    d = hull_separation(B, z[None, :], tm, cm)[0]
    _assert_under_theorem(d, B, z, S, C)


@pytest.mark.parametrize("seed", range(10))
def test_the_certificate_costs_little_on_well_separated_hulls(seed):
    B, A = _separated(seed, gap=5.0)
    S, C = [0, 1, 2], [4, 5, 6, 7]
    z = B[:, S].mean(axis=1) - B[:, C].mean(axis=1)
    tm, cm = _masks(8, S, C)
    d = hull_separation(B, z[None, :], tm, cm)[0]
    num, zz = _exact_theorem(B, z, S, C)
    exact = float(num) / np.sqrt(float(zz))
    assert exact > 0
    assert d == pytest.approx(exact, rel=1e-12)


def test_a_lying_iteration_cannot_make_the_bound_invalid(monkeypatch):
    """Validity rests on the certificate alone: an iteration that reports
    every candidate cleared, at an absurd bound, with arbitrary directions,
    still yields bounds under the distance."""
    rng = np.random.default_rng(0)

    def liar(G, pair_d2, sets, cmask, need, max_iter, corral):
        b = len(sets)
        return (np.full(b, 1e300), np.full(b, exact_mod.WOLFE_CLEARED),
                rng.normal(size=(b, G.shape[0])))

    monkeypatch.setattr(exact_mod, "_wolfe_lockstep", liar)
    B, A = _problem(2, n=8)
    sets = _subsets(8, 2)
    d = CouplingBound(B, A, beta=1.0).distance(sets)
    for S, di in zip(sets, d):
        assert _under(di, _distance2(B, S, _controls(8, S)))


# ------------------------------------------------------------- the search
def _count_fits(monkeypatch):
    calls = []
    real = exact_mod.coupled_fit

    def spy(B, A, treated, control, beta, **kw):
        calls.append(tuple(treated))
        return real(B, A, treated, control, beta, **kw)

    monkeypatch.setattr(exact_mod, "coupled_fit", spy)
    return calls


def _no_screen(monkeypatch):
    monkeypatch.setattr(CouplingBound, "survivors",
                        lambda self, sets, lb, cutoff: np.ones(len(sets), bool))


def test_the_screen_cuts_the_coupled_fits(monkeypatch):
    B, A = _problem(3, n=12)
    calls = _count_fits(monkeypatch)
    with_screen = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                                 beta=100.0)[0]
    screened = len(calls)
    calls.clear()
    _no_screen(monkeypatch)
    without = search_cluster(B, A, sizes=[3], filt=SubsetFilter(),
                             beta=100.0)[0]
    assert with_screen.treated == without.treated
    assert with_screen.value == pytest.approx(without.value, rel=1e-12)
    assert screened < 0.25 * len(calls)


def _record_screens(monkeypatch):
    seen = []
    real = CouplingBound.survivors

    def spy(self, sets, lb, cutoff):
        keep = real(self, sets, lb, cutoff)
        seen.append(([tuple(int(i) for i in s) for s in sets], cutoff, keep))
        return keep

    monkeypatch.setattr(CouplingBound, "survivors", spy)
    return seen


@pytest.mark.parametrize("buffer", [512, 2])
def test_every_fit_after_the_first_incumbent_passed_a_screen(monkeypatch,
                                                             buffer):
    """In both passes: a two-candidate buffer forces the second."""
    monkeypatch.setattr(exact_mod, "BUFFER", buffer)
    B, A = _problem(9, n=11)
    screens = _record_screens(monkeypatch)
    calls = _count_fits(monkeypatch)
    search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=100.0)
    kept = {s for sets, _, keep in screens for s, k in zip(sets, keep) if k}
    first = exact_mod.SCREEN_SEED if buffer > exact_mod.SCREEN_SEED else buffer
    assert screens
    assert all(S in kept for S in calls[first:])


def test_the_screen_aims_at_the_incumbent(monkeypatch):
    B, A = _problem(10, n=11)
    screens = _record_screens(monkeypatch)
    best = search_cluster(B, A, sizes=[3], filt=SubsetFilter(), beta=100.0)[0]
    cuts = [c for _, c, _ in screens]
    assert cuts and all(np.isfinite(c) for c in cuts)
    assert all(c >= best.value for c in cuts)
    assert all(a >= b for a, b in zip(cuts, cuts[1:]))     # it only improves


def _record_builds(monkeypatch):
    built = []
    real = CouplingBound.__init__

    def spy(self, *args, **kwargs):
        built.append(args)
        real(self, *args, **kwargs)

    monkeypatch.setattr(CouplingBound, "__init__", spy)
    return built


def test_beta_zero_takes_no_screen(monkeypatch):
    """At beta = 0 the coupling bound is the treated-side bound; building it
    would batch the walk and price every batch for nothing."""
    built = _record_builds(monkeypatch)
    B, A = _problem(11, n=9)
    d = search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=0.0)[0]
    assert not built
    assert d.value == pytest.approx(min(side_fit(B, A, list(S))[0]
                                        for S in _subsets(9, 2)), rel=1e-9)


def test_the_separable_designs_take_no_screen(monkeypatch):
    built = _record_builds(monkeypatch)
    B, A = _problem(12, n=9)
    search_cluster(B, A, sizes=[2], filt=SubsetFilter())
    assert not built


def test_a_positive_beta_takes_one_screen(monkeypatch):
    built = _record_builds(monkeypatch)
    B, A = _problem(12, n=9)
    search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=1e-6)
    assert len(built) == 1


def test_the_screened_search_keeps_its_top_k(monkeypatch):
    """Until k designs exist there is no cutoff to aim at, so the first
    candidates are fitted unscreened, however many k asks for."""
    B, A = _problem(13, n=10)
    k = exact_mod.SCREEN_SEED + 3
    designs = search_cluster(B, A, sizes=[2], filt=SubsetFilter(),
                             beta=100.0, top_k=k)
    ref = _brute_force(B, A, 2, 100.0, k=k)
    assert [d.treated for d in designs] == [S for _, S in ref]


# ---------------------------------------------------------------------- edge
def test_one_treated_and_one_control_market():
    B, A = _problem(14, n=2)
    d = CouplingBound(B, A, beta=1.0).distance(np.array([[0]]))[0]
    assert d == pytest.approx(np.linalg.norm(B[:, 0] - B[:, 1]), rel=1e-9)


def test_a_treated_market_inside_the_control_hull_has_no_coupling_bound():
    B, A = _problem(15, n=5)
    B = np.column_stack([B, B[:, 1:5].mean(axis=1)])
    d = CouplingBound(B, A, beta=100.0).distance(np.array([[5]]))[0]
    assert d == 0.0


def test_a_duplicated_market_across_the_sides_gives_zero_not_nan():
    B, A = _problem(16, n=5)
    B = np.column_stack([B, B[:, 0]])
    d = CouplingBound(B, A, beta=1.0).distance(np.array([[0], [0]]))
    assert np.array_equal(d, [0.0, 0.0])


def test_a_candidate_with_no_control_market_gets_zero():
    B, A = _problem(17, n=4)
    cb = CouplingBound(B, A, beta=1.0,
                       donor_exclusion=[(0, 1), (0, 2), (0, 3)])
    sets = np.array([[0], [1]])
    assert not cb.control_mask(sets)[0].any()
    d = cb.distance(sets)
    assert d[0] == 0.0 and d[1] > 0.0


def test_a_single_period():
    B = np.array([[0.0, 1.0, 5.0, 7.0]])
    A = np.array([2.0])
    d = CouplingBound(B, A, beta=1.0).distance(np.array([[0, 1]]))[0]
    assert d == pytest.approx(4.0, rel=1e-12)


def test_markets_at_a_large_offset():
    """The certificate reads the data uncentred; the offset adds rounding to
    every product, and the bound must stay under the distance and near it."""
    B, A = _separated(18)
    B, A = B + 1e7, A + 1e7
    cb = CouplingBound(B, A, beta=1.0)
    sets = _subsets(8, 2)
    d = cb.distance(sets)
    for S, di in zip(sets, d):
        ref2 = _distance2(B - 1e7, S, _controls(8, S))
        assert _under(di, ref2)
        assert di * di >= ref2 * (1 - 1e-4) - 1e-6


def test_the_screened_search_with_exclusions_and_a_penalty():
    B, A = _problem(19, n=10)
    excl = ((0, 1), (2, 3), (4, 5))
    lam = 0.0099
    d = search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=100.0,
                       control_penalty=lam, donor_exclusion=excl)[0]
    (value, S), = _brute_force(B, A, 2, 100.0, lam, excl)
    assert d.treated == S
    assert d.value == pytest.approx(value, rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------- failure
def test_non_finite_data_gives_a_bound_of_zero_not_nan():
    B, A = _problem(20, n=5)
    B[0, 3] = np.nan
    d = CouplingBound(B, A, beta=1.0).distance(_subsets(5, 2))
    assert np.isfinite(d).all() and (d >= 0).all()
    tm, cm = _masks(5, [0], [3])
    assert hull_separation(B, np.ones((1, B.shape[0])), tm, cm)[0] == 0.0


@pytest.mark.parametrize("beta, lam, fragment", [
    (-1.0, 0.0, "beta"),
    (1.0, -0.5, "control_penalty"),
    (np.inf, 0.0, "beta"),
])
def test_the_bound_refuses_what_has_no_minimum(beta, lam, fragment):
    B, A = _problem(21, n=5)
    with pytest.raises(MlsynthConfigError, match=fragment):
        CouplingBound(B, A, beta=beta, control_penalty=lam)


def test_an_uncertified_fit_still_raises_through_the_screened_search(
        monkeypatch):
    monkeypatch.setattr(exact_mod, "COUPLED_MAX_ITER", 0)
    B, A = _problem(22, n=9)
    with pytest.raises(exact_mod.CoupledFitNotCertified):
        search_cluster(B, A, sizes=[2], filt=SubsetFilter(), beta=100.0)


# ------------------------------------------------------------------ property
@given(seed=st.integers(0, 10_000), n=st.integers(3, 8), m=st.integers(1, 3),
       beta=st.sampled_from([0.01, 1.0, 100.0]),
       frac=st.sampled_from([0.0, 0.5, 0.99]),
       excl=st.booleans())
@settings(max_examples=40, deadline=None)
def test_the_bound_never_exceeds_a_candidates_objective(seed, n, m, beta, frac,
                                                       excl):
    """The validity the issue asks for: never above the candidate's objective."""
    m = min(m, n - 1)
    B, A = _problem(seed, n=n, T=6)
    lam = frac * (np.sqrt(1.0 + 4.0 / beta) - 1.0) / 2.0     # within convexity
    pairs = ((0, n - 1), (1, n - 2)) if excl else ()
    sets = _subsets(n, m)
    cb = CouplingBound(B, A, beta=beta, control_penalty=lam,
                       donor_exclusion=pairs)
    bound = cb.bound(sets, treated_fit_bound(B, A, sets))
    for S, b in zip(sets, bound):
        C = _controls(n, S, pairs)
        if C:
            f = _coupled(B, A, S, C, beta, lam)
            assert b <= f * (1 + 1e-9) + 1e-12


@given(seed=st.integers(0, 10_000), n=st.integers(3, 7), m=st.integers(1, 3),
       T=st.integers(1, 6), offset=st.sampled_from([0.0, 1e4, 1e8]))
@example(seed=0, n=3, m=1, T=1, offset=0.0)
@settings(max_examples=30, deadline=None)
def test_the_distance_bound_is_under_the_distance_over_the_domain(
        seed, n, m, T, offset):
    m = min(m, n - 1)
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, n)) + offset
    A = B.mean(axis=1)
    sets = _subsets(n, m)
    d = CouplingBound(B, A, beta=1.0).distance(sets)
    for S, di in zip(sets, d):
        assert _under(di, _distance2(B - offset, S, _controls(n, S)))


@given(seed=st.integers(0, 10_000), n=st.integers(3, 7), m=st.integers(1, 3),
       beta=st.sampled_from([0.01, 1.0, 100.0]), k=st.integers(1, 3),
       excl=st.booleans(), pen=st.booleans())
@settings(max_examples=30, deadline=None)
def test_the_screened_search_is_exact_over_the_domain(seed, n, m, beta, k,
                                                      excl, pen):
    m = min(m, n - 1)
    B, A = _problem(seed, n=n, T=6)
    lam = 0.5 * (np.sqrt(1.0 + 4.0 / beta) - 1.0) / 2.0 if pen else 0.0
    pairs = ((0, n - 1),) if excl else ()
    designs = search_cluster(B, A, sizes=[m], filt=SubsetFilter(), beta=beta,
                             control_penalty=lam, donor_exclusion=pairs,
                             top_k=k)
    ref = _brute_force(B, A, m, beta, lam, pairs, k=k)
    np.testing.assert_allclose([d.value for d in designs],
                               [v for v, _ in ref], rtol=1e-8, atol=1e-10)


@given(seed=st.integers(0, 10_000), T=st.integers(1, 8),
       scale=st.sampled_from([1.0, 1e8, 2.0 ** -500, 2.0 ** -530]))
@settings(max_examples=60, deadline=None)
def test_the_certificate_is_under_wolfes_bound_over_the_domain(seed, T, scale):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    B = scale * (1.0 + rng.normal(size=(T, n)))
    z = rng.normal(size=T) * scale
    S = [0]
    C = list(range(1, n))
    tm, cm = _masks(n, S, C)
    d = hull_separation(B, z[None, :], tm, cm)[0]
    _assert_under_theorem(d, B, z, S, C)
