"""The numba simplex master against HiGHS (bcp._master), on synthetic set-partitioning masters with cuts."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from . import bcp, bnp, dive, lpsx, rank1, sg


def instance(seed, n, ncol, ncut, kind="sr3"):
    rng = np.random.default_rng(seed)
    z = rng.lognormal(0, 1, n) * 10; z /= np.median(np.diff(np.sort(z)))
    pool = set(sg.initial_pool(z, 2, 4))
    from math import comb
    ncol = min(ncol, comb(n, 2) + comb(n, 3) + comb(n, 4))    # there are no more groups than that
    while len(pool) < ncol:
        k = int(rng.integers(2, 5)); pool.add(tuple(sorted(rng.choice(n, k, replace=False).tolist())))
    cols = sorted(pool); costs = np.array([dive._sc(z, c) for c in cols])
    if kind == "sr3":
        cuts = sorted({tuple(sorted(rng.choice(n, 3, replace=False).tolist())) for _ in range(ncut)})
    else:
        cuts = []
        for _ in range(ncut):
            size, num, den = rank1.FAMILIES[int(rng.integers(len(rank1.FAMILIES)))]
            if size > n: continue
            cuts.append(rank1.Cut(tuple(sorted(rng.choice(n, size, replace=False).tolist())), tuple(rng.permutation(num)), den))
    return cols, costs, n % 2 == 1, cuts


def test_duplicate_pool_cuts_share_one_dual():
    cols, costs, odd, cuts = instance(0, 4, 24, 30, "rank1")
    r = lpsx.Master(cols, costs, 4, odd).solve(np.ones(len(cols), bool), cuts)
    rhs = np.array([lpsx.cut_rhs(c) for c in cuts])
    assert abs(r.pi.sum() + r.tau + r.sigma @ rhs - r.lp) <= 1e-7 * max(1.0, r.lp)


def highs(cols, costs, n, odd, cuts, kind="sr3"):
    fn = bcp._master if kind == "sr3" else bcp._master_r1
    return bnp.solve_feasible(fn, cols, costs, n, odd, cuts)


def pair_mask(cols, i, j, together):
    return np.array([not ((i in c) != (j in c) if together else (i in c and j in c)) for c in cols])


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 31), kind=st.sampled_from(["sr3", "rank1"]),
       ncut=st.integers(0, 40))
def test_cold_solve_matches_highs(seed, n, kind, ncut):
    cols, costs, odd, cuts = instance(seed, n, 6 * n, ncut, kind)
    m = lpsx.Master(cols, costs, n, odd)
    r = m.solve(np.ones(len(cols), bool), cuts)
    h = highs(cols, costs, n, odd, cuts, kind)
    assert (r.art > 1e-6) == (h[-1] > 1e-6)
    if h[-1] <= 1e-6:
        assert abs(r.lp - h[0]) <= 1e-7 * max(1.0, abs(h[0]))


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(6, 31), kind=st.sampled_from(["sr3", "rank1"]))
def test_children_warm_from_the_parent_match_highs(seed, n, kind):
    """A chain of pair decisions, each child warm-started from its parent's basis and active cut rows."""
    rng = np.random.default_rng(seed)
    cols, costs, odd, cuts = instance(seed, n, 6 * n, 25, kind)
    m = lpsx.Master(cols, costs, n, odd)
    allowed = np.ones(len(cols), bool)
    r = m.solve(allowed, cuts)
    for _ in range(6):
        i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        allowed = allowed & pair_mask(cols, i, j, bool(rng.random() < 0.5))
        r = m.solve(allowed, cuts, warm=r.warm)
        keep = [c for c, a in zip(cols, allowed) if a]
        h = highs(keep, costs[allowed], n, odd, cuts, kind)
        assert (r.art > 1e-6) == (h[-1] > 1e-6)
        if h[-1] <= 1e-6:                       # with an artificial in use the value depends on its price
            assert abs(r.lp - h[0]) <= 1e-7 * max(1.0, abs(h[0])), (r.lp, h[0])
        assert np.all(r.x[~allowed] == 0.0)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 25), kind=st.sampled_from(["sr3", "rank1"]))
def test_the_solution_and_its_duals_certify_each_other(seed, n, kind):
    """Primal feasible over every pool cut (active or not), dual feasible with sigma <= 0 and tau >= 0, and the
    two objectives equal: the duals give the node bound the search needs."""
    cols, costs, odd, cuts = instance(seed, n, 6 * n, 30, kind)
    m = lpsx.Master(cols, costs, n, odd)
    r = m.solve(np.ones(len(cols), bool), cuts)
    coef = np.array([[float(lpsx.cut_coef(cut, c)) for c in cols] for cut in cuts]).reshape(len(cuts), len(cols))
    cover = np.zeros(n)
    for c, v in zip(cols, r.x):
        for g in c: cover[g] += v
    cover += r.x_art
    assert np.abs(cover - 1).max() <= 1e-7
    if cuts:
        rhs = np.array([lpsx.cut_rhs(c) for c in cuts])
        assert np.all(coef @ r.x <= rhs + 1e-7)
    assert np.all(r.sigma <= 0) and r.tau >= 0
    rc = costs - np.array([r.pi[list(c)].sum() + (r.tau if odd and len(c) % 2 else 0.0) for c in cols]) - (r.sigma @ coef if cuts else 0.0)
    assert rc.min() >= -1e-7 * max(1.0, np.abs(costs).max())
    dual = r.pi.sum() + r.tau + (r.sigma @ np.array([lpsx.cut_rhs(c) for c in cuts]) if cuts else 0.0)
    if r.art <= 1e-9:
        assert abs(dual - r.lp) <= 1e-7 * max(1.0, abs(r.lp))


def test_only_the_cut_rows_that_bind_are_carried():
    cols, costs, odd, cuts = instance(3, 30, 400, 400)
    m = lpsx.Master(cols, costs, 30, odd)
    r = m.solve(np.ones(len(cols), bool), cuts)
    assert r.active_rows < len(cuts) // 4
    assert abs(r.lp - highs(cols, costs, 30, odd, cuts)[0]) <= 1e-7 * max(1.0, r.lp)


def test_a_fully_degenerate_master_terminates():
    """Every column costs the same: every basis is optimal along a long degenerate face."""
    n = 12
    import itertools
    cols = [c for k in (2, 3) for c in itertools.combinations(range(n), k)]
    costs = np.ones(len(cols))
    m = lpsx.Master(cols, costs, n, False)
    r = m.solve(np.ones(len(cols), bool), [])
    assert abs(r.lp - highs(cols, costs, n, False, [])[0]) <= 1e-9


def test_a_child_with_no_cover_uses_its_artificials():
    cols = [(0, 1), (2, 3), (0, 2), (1, 3)]; costs = np.array([1.0, 1.0, 2.0, 2.0])
    m = lpsx.Master(cols, costs, 4, False)
    r = m.solve(np.ones(4, bool), [])
    r2 = m.solve(np.array([True, False, True, False]), [], warm=r.warm)
    assert r2.art > 1e-6


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        lpsx.Master([(0, 1)], np.array([1.0, 2.0]), 2, False)
    with pytest.raises(ValueError):
        lpsx.Master([(0, 1)], np.array([np.nan]), 2, False)
    m = lpsx.Master([(0, 1)], np.array([1.0]), 2, False)
    with pytest.raises(ValueError):
        m.solve(np.ones(3, bool), [])


def test_artificials_priced_too_low_are_raised_until_a_feasible_master_drops_them():
    """Artificials at 1e-9 of the dearest column: the phase-1 LP finds the master feasible, and it is solved
    again at a higher price until no artificial is in use (as bnp.solve_feasible does for HiGHS)."""
    cols = [(0, 1), (2, 3), (0, 2), (1, 3)]; costs = np.array([1e12, 1e12, 2e12, 2e12])
    r = lpsx.Master(cols, costs, 4, False, factor=1e-9).solve(np.ones(4, bool), [])
    assert r.art <= 1e-9 and abs(r.lp - 2e12) <= 1e-9 * 2e12
