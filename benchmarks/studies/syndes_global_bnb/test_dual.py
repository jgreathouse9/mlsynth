"""The lift's bound by its regularised dual: valid for any multipliers,
minimised by a quasi-Newton method with triangle inequalities as cuts, no
interior-point solve."""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem

from .bounds import lift_upper, split, trs_upper
from .dual import Cuts, dual_upper, restrict, separate, theta
from .test_bounds import _engine, _node_problem, _nodes


def _best(R, I, O, F, K):
    """The subtree's maximum of ``sigma' R sigma``, by brute force."""
    N = R.shape[0]
    best = -np.inf
    for comb in combinations(F, K - len(I)):
        sig = -np.ones(N); sig[list(I) + list(comb)] = 1.0
        best = max(best, float(sig @ R @ sig))
    return best


def _random_cuts(rng, n, count):
    triples = np.array(list(combinations(range(n + 1), 3)))
    pick = triples[rng.choice(len(triples), min(count, len(triples)), replace=False)]
    return Cuts(pick[:, 0], pick[:, 1], pick[:, 2], rng.integers(1, 5, len(pick)))


def _triangle(X, i, j, k, t):
    sign = {1: (1, 1, 1), 2: (1, -1, -1), 3: (-1, 1, -1), 4: (-1, -1, 1)}[t]
    return 1.0 + sign[0] * X[i, j] + sign[1] * X[i, k] + sign[2] * X[j, k]


# ----------------------------------------------------------------------
# the function: valid for any multipliers, and its gradient
# ----------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(_node_problem(), st.integers(0, 2**31 - 1), st.sampled_from([1e-4, 1e-2, 1.0]))
def test_any_multipliers_bound_the_subtree(problem, seed, alpha):
    """Weak duality: any equality multipliers, any triangle inequalities and
    any nonnegative multipliers on them give a bound; the regularised value
    lies above the eigenvalue one."""
    R, I, O, F, K = problem
    r = K - len(I)
    if r < 0 or r > len(F) or len(F) < 2:
        return
    c, b, A = split(R, I, O, F)
    n, s = len(F), 2 * r - len(F)
    rng = np.random.default_rng(seed)
    cuts = _random_cuts(rng, n, 12)
    y = rng.standard_normal(2 * n + 2)
    v = np.abs(rng.standard_normal(len(cuts)))
    exact, smooth, _, _, _ = theta(c, b, A, s, y, cuts, v, alpha)
    best = _best(R, I, O, F, K)
    tol = 1e-9 * max(1.0, abs(best), float(np.abs(R).sum()), float(np.abs(y).sum() + v.sum()))
    assert exact >= best - tol
    assert smooth >= exact - tol


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("alpha", [0.01, 0.3])
def test_the_gradient_is_the_regularised_values(seed, alpha):
    rng = np.random.default_rng(seed)
    eng = _engine(9, 12, seed)
    I, O, F, K = [0], [1, 2], list(range(3, 9)), 4
    c, b, A = split(eng.R, I, O, F)
    unit = np.abs(A).max()
    c, b, A = c / unit, b / unit, A / unit
    n, s = len(F), 2 * (K - len(I)) - len(F)
    cuts = _random_cuts(rng, n, 10)
    y = rng.standard_normal(2 * n + 2)
    v = np.abs(rng.standard_normal(len(cuts)))
    _, _, gy, gv, _ = theta(c, b, A, s, y, cuts, v, alpha)
    h = 1e-6
    for k in range(len(y)):
        e = np.zeros_like(y); e[k] = h
        fd = (theta(c, b, A, s, y + e, cuts, v, alpha)[1]
              - theta(c, b, A, s, y - e, cuts, v, alpha)[1]) / (2 * h)
        assert gy[k] == pytest.approx(fd, rel=1e-4, abs=1e-6)
    for k in range(len(v)):
        e = np.zeros_like(v); e[k] = h
        fd = (theta(c, b, A, s, y, cuts, v + e, alpha)[1]
              - theta(c, b, A, s, y, cuts, v - e, alpha)[1]) / (2 * h)
        assert gv[k] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_zero_multipliers_leave_the_bordered_eigenvalue():
    """``y = 0`` and no cuts leave only the objective: ``c + (n + 1)
    lambda_max`` of ``[[0, b'], [b, A]]``."""
    rng = np.random.default_rng(3)
    n = 5
    A = rng.standard_normal((n, n)); A = A + A.T
    b = rng.standard_normal(n)
    M = np.block([[np.zeros((1, 1)), b[None, :]], [b[:, None], A]])
    exact = theta(2.0, b, A, 1, np.zeros(2 * n + 2), Cuts.empty(), np.zeros(0), 0.1)[0]
    assert exact == pytest.approx(2.0 + (n + 1) * np.linalg.eigvalsh(M)[-1], rel=1e-12)


@settings(max_examples=100, deadline=None)
@given(st.integers(2, 7), st.integers(0, 2**31 - 1), st.sampled_from([0.0, 0.05]),
       st.booleans())
def test_separation_returns_the_violated_triangles_most_violated_first(n, seed, gap, products):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n + 1, n + 2))
    X = G @ G.T
    d = 1.0 / np.sqrt(np.diag(X))
    X = X * d[:, None] * d[None, :]
    expected = sorted(
        ((_triangle(X, i, j, k, t), (i, j, k, t))
         for i, j, k in combinations(range(n + 1), 3) if not products or i == 0
         for t in (1, 2, 3, 4) if _triangle(X, i, j, k, t) < -gap),
        key=lambda e: e[0])
    got = separate(X, gap=gap, limit=10_000, products=products)
    assert [(int(a), int(b_), int(c_), int(t)) for a, b_, c_, t in zip(got.i, got.j, got.k, got.t)] \
        == [key for _, key in expected]
    few = separate(X, gap=gap, limit=3, products=products)
    assert len(few) == min(3, len(expected))


# ----------------------------------------------------------------------
# the minimised bound
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("N,K", [(8, 3), (9, 4)])
@pytest.mark.parametrize("cuts", ["products", "triangles"])
def test_the_dual_lies_above_every_completion_and_under_the_trust_region(seed, N, K, cuts):
    eng = _engine(N, 12, seed)
    rng = np.random.default_rng(seed)
    for I, O, F in list(_nodes(N, K, rng))[:10]:
        r = K - len(I)
        if r < 0 or r > len(F):
            continue
        best = _best(eng.R, I, O, F, K)
        ud = dual_upper(eng.R, I, O, F, K, cuts=cuts)
        ut, _ = trs_upper(eng.R, I, O, F, K)
        scale = max(abs(best), 1.0)
        assert ud >= best - 1e-7 * scale
        assert ud <= ut + 1e-9 * scale


@pytest.mark.parametrize("seed", range(4))
def test_with_the_products_alone_the_dual_reaches_the_lift(seed):
    """The products are the triangle inequalities through the homogenising
    index, so with them alone the dual bounds the lift's relaxation."""
    eng = _engine(16, 12, seed)
    rng = np.random.default_rng(seed)
    checked = 0
    for I, O, F in list(_nodes(16, 6, rng))[:12]:
        r = 6 - len(I)
        if r < 1 or r >= len(F) or len(F) < 6:
            continue
        ul = lift_upper(eng.R, I, O, F, 6)
        ud = dual_upper(eng.R, I, O, F, 6, cuts="products")
        assert ud <= ul + 2e-3 * abs(ul)
        checked += 1
    assert checked >= 3


def test_triangles_among_the_free_units_tighten_the_bound():
    """Every triangle inequality holds at a design, so adding those among free
    units never loosens the bound, and on some nodes it tightens it."""
    tighter = 0
    for seed in range(4):
        eng = _engine(14, 12, seed)
        rng = np.random.default_rng(seed)
        for I, O, F in list(_nodes(14, 5, rng))[:8]:
            r = 5 - len(I)
            if r < 1 or r >= len(F) or len(F) < 5:
                continue
            up = dual_upper(eng.R, I, O, F, 5, cuts="products")
            ua = dual_upper(eng.R, I, O, F, 5, cuts="triangles")
            assert ua >= _best(eng.R, I, O, F, 5) - 1e-7 * abs(ua)
            assert ua <= up + 2e-3 * abs(up)
            tighter += ua < up - 1e-3 * abs(up)
    assert tighter > 0


def test_a_target_stops_the_search_once_reached():
    """A node prunes once its bound reaches the incumbent's; the method stops
    there, with fewer evaluations, and what it returns is still a bound."""
    eng = _engine(14, 12, 1)
    I, O, F, K = [], [0, 1], list(range(2, 14)), 5
    full, info = dual_upper(eng.R, I, O, F, K, info=True)
    target = full * 1.02
    early, info_early = dual_upper(eng.R, I, O, F, K, target=target, info=True)
    assert early <= target
    assert early >= _best(eng.R, I, O, F, K) - 1e-9 * abs(full)
    assert info_early.evaluations < info.evaluations


def test_a_target_out_of_reach_changes_nothing():
    eng = _engine(12, 12, 1)
    I, O, F, K = [], [0], list(range(1, 12)), 4
    full, info = dual_upper(eng.R, I, O, F, K, info=True)
    same, info_t = dual_upper(eng.R, I, O, F, K, target=0.5 * _best(eng.R, I, O, F, K), info=True)
    assert same == full
    assert info_t.evaluations == info.evaluations


def test_a_child_starts_from_its_parents_multipliers():
    """Fixing a unit keeps the other multipliers and cuts meaningful: read on
    the child they are a valid start, and over the children of four panels a
    better one than zero. Node by node it can cost more; the gain is in
    aggregate."""
    warm_evals = cold_evals = 0
    for seed in range(4):
        eng = _engine(14, 12, seed)
        I, O, F, K = [], [0], list(range(1, 14)), 5
        _, parent = dual_upper(eng.R, I, O, F, K, info=True)
        for cI, cO, cF in ((I + [F[0]], O, F[1:]), (I, O + [F[0]], F[1:])):
            start = restrict(parent.state, F, cF)
            uw, iw = dual_upper(eng.R, cI, cO, cF, K, warm=start, info=True)
            uc, ic = dual_upper(eng.R, cI, cO, cF, K, info=True)
            assert uw >= _best(eng.R, cI, cO, cF, K) - 1e-9 * abs(uw)
            assert uw == pytest.approx(uc, rel=2e-3)
            warm_evals += iw.evaluations
            cold_evals += ic.evaluations
    assert warm_evals < cold_evals


@settings(max_examples=100, deadline=None)
@given(st.integers(0, 2**31 - 1), st.booleans())
def test_a_parents_multipliers_bound_each_child(seed, treated):
    """Dual fixing: the parent's multipliers, read on a child, give the child a
    bound at the price of one eigendecomposition."""
    rng = np.random.default_rng(seed)
    N, K = 9, 4
    G = rng.standard_normal((N, N)); R = G + G.T
    I, O, F = [], [0], list(range(1, N))
    c, b, A = split(R, I, O, F)
    n, s = len(F), 2 * K - len(F)
    cuts = _random_cuts(rng, n, 20)
    state = (rng.standard_normal(2 * n + 2), cuts, np.abs(rng.standard_normal(len(cuts))))
    u = int(rng.choice(F))
    cI, cO = (I + [u], O) if treated else (I, O + [u])
    cF = [f for f in F if f != u]
    y, cc, cv = restrict(state, F, cF)
    cc_, cb, cA = split(R, cI, cO, cF)
    bound = theta(cc_, cb, cA, 2 * (K - len(cI)) - len(cF), y, cc, cv, 1.0)[0]
    assert bound >= _best(R, cI, cO, cF, K) - 1e-9 * max(1.0, float(np.abs(R).sum()) + float(np.abs(y).sum()))


def test_restricting_keeps_each_cuts_orientation():
    """A child whose free units come in another order reads each cut on the
    same three units, with the same positive edge; cuts through a unit the
    child fixed are dropped."""
    parent = [10, 11, 12, 13]
    y = np.arange(2 * 4 + 2, dtype=float)
    # (0, 11, 13) with its positive edge (11, 13); (11, 12, 13) all positive;
    # (0, 10, 12) runs through the fixed unit 10.
    cuts = Cuts(np.array([0, 2, 0]), np.array([2, 3, 1]), np.array([4, 4, 3]),
                np.array([4, 1, 2]))
    state = (y, cuts, np.array([5.0, 6.0, 7.0]))
    cy, cc, cv = restrict(state, parent, [13, 12, 11])
    # child index: 13 -> 1, 12 -> 2, 11 -> 3
    got = sorted(zip(cc.i.tolist(), cc.j.tolist(), cc.k.tolist(), cc.t.tolist(), cv.tolist()))
    # (0, 11, 13) -> (0, 1, 3), positive edge (1, 3), the last pair -> type 4
    # (11, 12, 13) -> (1, 2, 3), all positive -> type 1
    assert got == [(0, 1, 3, 4, 5.0), (1, 2, 3, 1, 6.0)]
    # diag multipliers: index 0 kept, then units 13, 12, 11; the sum's kept;
    # the row multipliers follow the units.
    assert cy[:4].tolist() == [y[0], y[4], y[3], y[2]]
    assert cy[4] == y[5]
    assert cy[5:].tolist() == [y[6 + 3], y[6 + 2], y[6 + 1]]


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_the_dual_holds_whatever_the_outcomes_units(scale):
    rng = np.random.default_rng(0)
    Y = scale * (rng.standard_normal((12, 3)) @ rng.standard_normal((3, 10))
                 + 0.3 * rng.standard_normal((12, 10)))
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    best = _best(eng.R, [], [], list(range(10)), 4)
    ud = dual_upper(eng.R, [], [], list(range(10)), 4)
    ut = trs_upper(eng.R, [], [], list(range(10)), 4)[0]
    assert ud >= best * (1 - 1e-7)
    assert ud < ut - 0.2 * (ut - best)


def test_edge_nodes():
    eng = _engine(6, 12, 0)
    c = split(eng.R, [0, 1], [2, 3, 4, 5], [])[0]
    assert dual_upper(eng.R, [0, 1], [2, 3, 4, 5], [], 2) == c
    one = dual_upper(eng.R, [0], [2, 3, 4, 5], [1], 2)
    assert one == pytest.approx(trs_upper(eng.R, [0], [2, 3, 4, 5], [1], 2)[0], rel=1e-12)
    assert dual_upper(np.zeros((6, 6)), [], [0], [1, 2, 3, 4, 5], 2) == 0.0
