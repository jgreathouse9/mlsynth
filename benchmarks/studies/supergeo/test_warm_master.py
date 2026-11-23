"""The root's master LP kept across column generation (certify_exact._Master).

The root re-solved its master from scratch at every iteration: parity.master and sg.master_lp build a new LP through
scipy's linprog, about 0.7 to 0.9 s each at 20,000 columns on a simulated 235-market panel, 13.5 s of a 42.7 s root.
_Master keeps one HiGHS model, appends each iteration's columns, replaces the costs when the root rescales, and
re-solves by the primal simplex from the last basis: appended columns leave the basis primal feasible, so the primal
simplex resumes where the dual simplex (HiGHS's default) would first repair dual feasibility (5,659 iterations
against 1,751 after 2,000 new columns). On that panel the master took 3.2 s.
"""
import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import certify_exact as ce, dive, parity, sg, test_bcp


def _all_groups(n, lo=2, hi=4):
    return [c for k in range(lo, hi + 1) for c in itertools.combinations(range(n), k)]


def _cold(cols, costs, n, odd):
    if odd:
        lp, pi, tau, x = parity.master(cols, costs, n)
        return lp, pi, max(float(tau), 0.0), x
    lp, pi, x = sg.master_lp(cols, costs, n)
    return lp, pi, 0.0, x


def _check_optimal(cols, costs, n, odd, lp, pi, tau, x):
    """Primal feasibility, dual feasibility and equal objectives: the duals certify the LP value."""
    A = np.zeros((n, len(cols)))
    for j, c in enumerate(cols):
        A[list(c), j] = 1.0
    assert np.allclose(A @ x, 1.0, atol=1e-7) and (x >= -1e-9).all()
    oddcol = np.array([len(c) % 2 == 1 for c in cols])
    if odd:
        assert x[oddcol].sum() >= 1 - 1e-7
    scale = max(1.0, float(np.abs(costs).max()))
    rc = costs - A.T @ pi - (tau * oddcol if odd else 0.0)
    assert rc.min() >= -1e-9 * scale
    assert tau >= 0.0
    assert float(pi.sum() + tau) == pytest.approx(lp, rel=1e-9, abs=1e-9 * scale)
    assert float(costs @ x) == pytest.approx(lp, rel=1e-9, abs=1e-9 * scale)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 10), kind=st.integers(0, 3), batches=st.integers(1, 4),
       rescale_at=st.integers(-1, 3), k=st.integers(0, 6))
def test_the_kept_master_matches_the_cold_one_at_every_solve(seed, n, kind, batches, rescale_at, k):
    """Property, over odd and even counts and the four generators: the columns of lo..hi groups are appended in
    batches, with the costs multiplied by 4^k at one solve (the root's exact rescale); at every solve the kept master's
    LP value equals the cold master's on the same columns, and its primal and dual solutions certify it."""
    rng = np.random.default_rng(seed)
    z = test_bcp.gen(rng, n, kind)
    odd = n % 2 == 1
    pool = [tuple(c) for c in sg.initial_pool(z, 2, 4)]
    rest = [c for c in _all_groups(n) if c not in set(pool)]
    rng.shuffle(rest)
    cuts = np.sort(rng.choice(len(rest) + 1, size=batches, replace=True))
    master = ce._Master(n, odd)
    cols, f = list(pool), 1.0
    for b in range(batches + 1):
        if b:
            cols += rest[cuts[b - 1]:cuts[b] if b < batches else len(rest)]
        if b == rescale_at:
            f *= 4.0 ** k
        costs = np.array([dive._sc(z, c) for c in cols]) * f
        lp, pi, tau, x = master.solve(cols, costs)
        want = _cold(cols, costs, n, odd)[0]
        assert lp == pytest.approx(want, rel=1e-9, abs=1e-9 * max(1.0, float(costs.max())))
        _check_optimal(cols, costs, n, odd, lp, pi, tau, x)


def test_a_solve_with_no_new_columns_returns_the_same_answer():
    """Edge: solving again with the same columns and costs returns the same LP value and needs no pivot."""
    z = test_bcp.gen(np.random.default_rng(1), 9, 2)
    cols = list(sg.initial_pool(z, 2, 4)); costs = np.array([dive._sc(z, c) for c in cols])
    m = ce._Master(9, True)
    a = m.solve(cols, costs); b = m.solve(cols, costs)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_an_infeasible_master_raises():
    """Failure: an odd count with no odd-sized column cannot meet the parity row; the master raises RuntimeError
    naming the solver's status instead of returning duals."""
    z = np.arange(1.0, 6.0)
    cols = [(0, 1), (2, 3), (1, 4), (0, 4), (2, 4), (3, 4), (0, 2), (1, 3)]
    m = ce._Master(5, True)
    with pytest.raises(RuntimeError, match="master"):
        m.solve(cols, np.array([dive._sc(z, c) for c in cols]))


def test_the_root_builds_one_master(monkeypatch):
    """Unit: the root constructs one _Master and solves it at every iteration."""
    made = []
    real = ce._Master
    class Counting(real):
        def __init__(self, *a):
            made.append(1); super().__init__(*a)
    monkeypatch.setattr(ce, "_Master", Counting)
    ce._root(test_bcp.gen(np.random.default_rng(2), 13, 2), 2, 4)
    assert len(made) == 1


@pytest.mark.parametrize("n, seed", [(13, 3), (14, 4), (21, 5), (22, 6)])
def test_the_root_bound_is_the_cold_roots(monkeypatch, n, seed):
    """Unit: the root's bound, sum pi + tau over the scale (the original units), is the same with the kept master
    as with the cold masters, on odd and even counts; the duals may differ, the LP optimum may not."""
    z = test_bcp.gen(np.random.default_rng(seed), n, 2)
    zr, scale, pi, tau = ce._root(z, 2, 4)
    class Cold:
        def __init__(self, n, odd):
            self.n, self.odd = n, odd
        def solve(self, cols, costs):
            return _cold(cols, costs, self.n, self.odd)
    monkeypatch.setattr(ce, "_Master", Cold)
    zc, sc, pc, tc = ce._root(z, 2, 4)
    assert (pi.sum() + tau) / scale == pytest.approx((pc.sum() + tc) / sc, rel=1e-9, abs=1e-12)
