"""The node bounds hold: below every completion's closed form, at every node."""
from itertools import combinations

import numpy as np
import pytest

from mlsynth.utils.syndes_helpers.gram import BoundEngine, TwoWayProblem, signs_from_subsets

from .bounds import node_lower, node_upper, spectral_upper, split, trs_upper


def _engine(N, T, seed, lam=None):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((T, 3)) @ rng.standard_normal((3, N)) + 0.3 * rng.standard_normal((T, N))
    return BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam))


def _nodes(N, K, rng, count=30):
    for _ in range(count):
        units = rng.permutation(N)
        i = int(rng.integers(0, K + 1))
        o = int(rng.integers(0, N - K + 1))
        yield sorted(units[:i]), sorted(units[i:i + o]), sorted(units[i + o:])


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("N,K", [(8, 3), (9, 4), (10, 2)])
def test_both_bounds_lie_above_every_completion(seed, N, K):
    eng = _engine(N, 12, seed)
    rng = np.random.default_rng(seed)
    for I, O, F in _nodes(N, K, rng):
        r = K - len(I)
        if r < 0 or r > len(F):
            continue
        sets = [tuple(sorted(I + list(c))) for c in combinations(F, r)]
        cut = eng.cut_value(signs_from_subsets(N, np.array(sets)))
        best = float(cut.max())
        ut, _ = trs_upper(eng.R, I, O, F, K)
        us = spectral_upper(eng.R, I, O, F, K)
        assert ut >= best * (1 - 1e-10)
        assert us >= best * (1 - 1e-10)
        u = node_upper(eng.R, I, O, F, K)
        assert u == min(ut, us)
        lb = eng.bound(signs_from_subsets(N, np.array(sets))).lower
        assert node_lower(eng.alpha, u) <= lb.min() * (1 + 1e-10)


def test_neither_bound_dominates():
    """The spectral bound takes the linear term over the box (``|b|_1``), the
    trust-region bound over the sphere (up to ``sqrt(n) |b|_2``), so each can
    be the smaller; the node takes the minimum."""
    trs_wins = spectral_wins = 0
    for N, K, seed in [(8, 3, s) for s in range(6)] + [(9, 4, s) for s in range(6)]:
        eng = _engine(N, 12, seed)
        rng = np.random.default_rng(seed)
        for I, O, F in _nodes(N, K, rng):
            r = K - len(I)
            if r < 0 or r > len(F) or len(F) < 2:
                continue
            ut, _ = trs_upper(eng.R, I, O, F, K)
            us = spectral_upper(eng.R, I, O, F, K)
            trs_wins += ut < us * (1 - 1e-9)
            spectral_wins += us < ut * (1 - 1e-9)
    assert trs_wins > 0 and spectral_wins > 0


def test_at_the_root_trs_is_the_rayleigh_bound():
    eng = _engine(10, 12, 0)
    ut, _ = trs_upper(eng.R, [], [], list(range(10)), 4)
    assert node_lower(eng.alpha, ut) == pytest.approx(eng.global_lower_bound(4), rel=1e-9)


def test_a_leaf_is_its_own_closed_form():
    eng = _engine(8, 12, 1)
    I, O = [0, 3, 5], [1, 2, 4, 6, 7]
    ut, _ = trs_upper(eng.R, I, O, [], 3)
    lb = eng.bound(signs_from_subsets(8, np.array([I]))).lower[0]
    assert node_lower(eng.alpha, ut) == pytest.approx(lb, rel=1e-10)


# ----------------------------------------------------------------------
# validity over arbitrary symmetric matrices, degenerate ones included
# ----------------------------------------------------------------------

from hypothesis import given, settings, strategies as st


@st.composite
def _node_problem(draw):
    N = draw(st.integers(3, 8))
    K = draw(st.integers(1, N - 1))
    kind = draw(st.sampled_from(["random", "low_rank", "zero", "negative", "repeated"]))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    if kind == "random":
        M = rng.standard_normal((N, N)); R = M + M.T
    elif kind == "low_rank":
        v = rng.standard_normal((N, 1)); R = v @ v.T
    elif kind == "zero":
        R = np.zeros((N, N))
    elif kind == "negative":
        M = rng.standard_normal((N, N)); R = -(M @ M.T)
    else:                                        # repeated eigenvalues
        Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
        R = Q @ np.diag(np.repeat([2.0, 0.0], [N // 2, N - N // 2])) @ Q.T
    units = rng.permutation(N)
    i = int(rng.integers(0, K + 1))
    o = int(rng.integers(0, N - K + 1))
    I, O, F = sorted(units[:i]), sorted(units[i:i + o]), sorted(units[i + o:])
    return R, I, O, F, K


@settings(max_examples=400, deadline=None)
@given(_node_problem())
def test_the_upper_bounds_hold_on_any_symmetric_matrix(problem):
    R, I, O, F, K = problem
    r = K - len(I)
    if r < 0 or r > len(F):
        return
    N = R.shape[0]
    best = -np.inf
    for comb in combinations(F, r):
        sig = -np.ones(N); sig[list(I) + list(comb)] = 1.0
        best = max(best, float(sig @ R @ sig))
    tol = 1e-9 * max(1.0, abs(best), float(np.abs(R).sum()))
    assert trs_upper(R, I, O, F, K)[0] >= best - tol
    assert spectral_upper(R, I, O, F, K) >= best - tol


@pytest.mark.parametrize("kind", ["random", "negative"])
def test_with_two_free_units_the_trust_region_bound_is_exact(kind):
    """With ``n = 2`` and ``r = 1`` the sphere cut by the plane is exactly the two
    sign completions, so the relaxation loses nothing, whatever the sign of
    ``A`` on the plane's direction."""
    rng = np.random.default_rng(7)
    M = rng.standard_normal((6, 6))
    R = M + M.T if kind == "random" else -(M @ M.T)
    I, O, F, K = [0], [1, 2, 3], [4, 5], 2
    best = -np.inf
    for j in F:
        sig = -np.ones(6); sig[[0, j]] = 1.0
        best = max(best, float(sig @ R @ sig))
    assert trs_upper(R, I, O, F, K)[0] == pytest.approx(best, rel=1e-9, abs=1e-9)


# ----------------------------------------------------------------------
# the lifted bound: Shor's relaxation with the four product inequalities
# ----------------------------------------------------------------------

from .bounds import lift_upper


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("N,K", [(8, 3), (9, 4)])
def test_the_lift_lies_above_every_completion_and_under_the_trust_region(seed, N, K):
    eng = _engine(N, 12, seed)
    rng = np.random.default_rng(seed)
    for I, O, F in list(_nodes(N, K, rng))[:10]:
        r = K - len(I)
        if r < 0 or r > len(F):
            continue
        sets = [tuple(sorted(I + list(c))) for c in combinations(F, r)]
        best = float(eng.cut_value(signs_from_subsets(N, np.array(sets))).max())
        ul = lift_upper(eng.R, I, O, F, K)
        ut, _ = trs_upper(eng.R, I, O, F, K)
        scale = max(abs(best), 1.0)
        assert ul >= best - 1e-7 * scale
        assert ul <= ut + 1e-6 * scale


def test_the_lift_is_exact_where_the_trust_region_is_not():
    """On the node of ``[8-3-2]`` the trust-region bound is 7% above the best
    completion; the product inequalities close most of that."""
    eng = _engine(8, 12, 2)
    I, O, F, K = [0], [1, 2, 6, 7], [3, 4, 5], 3
    sets = [tuple(sorted(I + list(c))) for c in combinations(F, 2)]
    best = float(eng.cut_value(signs_from_subsets(8, np.array(sets))).max())
    ut, _ = trs_upper(eng.R, I, O, F, K)
    ul = lift_upper(eng.R, I, O, F, K)
    assert ut > 1.05 * best
    assert ul < best + 0.25 * (ut - best)


def _bisection_reference(R, I, O, F, K):
    """The dual minimised by 400 bisection steps: the tightness reference."""
    from .bounds import split
    c, b, A = split(R, I, O, F)
    n = len(F); s = 2 * (K - len(I)) - n
    x0 = np.full(n, s / n); rho2 = n - s * s / n
    const = c + 2 * b @ x0 + x0 @ A @ x0
    P = np.linalg.svd(np.ones((1, n)))[2][1:].T
    lam, Q = np.linalg.eigh(P.T @ A @ P)
    gt = Q.T @ (P.T @ (b + A @ x0))
    d = lambda mu: const + mu * rho2 + np.sum(gt ** 2 / (mu - lam))
    slope = lambda mu: rho2 - np.sum(gt ** 2 / (mu - lam) ** 2)
    lo, hi = lam[-1] + 1e-13 * max(1, abs(lam).max()), lam[-1] + 1e6
    if slope(lo) >= 0:
        return d(lo)
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slope(mid) < 0 else (lo, mid)
    return d(hi)


@pytest.mark.parametrize("seed", range(4))
def test_the_trust_region_dual_is_minimised(seed):
    eng = _engine(14, 10, seed)
    rng = np.random.default_rng(seed)
    for I, O, F in list(_nodes(14, 5, rng))[:15]:
        r = 5 - len(I)
        if r < 1 or r >= len(F):
            continue
        ref = _bisection_reference(eng.R, I, O, F, 5)
        got = trs_upper(eng.R, I, O, F, 5)[0]
        assert got == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_the_lift_holds_whatever_the_outcomes_units(scale):
    """``R`` scales as the outcome's units to the power -4, so a panel in the
    hundreds of thousands puts its entries near 1e-11, under any solver's
    absolute tolerances. Unscaled, the solver reports optimal at a value 12%
    under the true maximum on GeoLift, a bound above the optimum."""
    rng = np.random.default_rng(0)
    Y = scale * (rng.standard_normal((12, 3)) @ rng.standard_normal((3, 10))
                 + 0.3 * rng.standard_normal((12, 10)))
    eng = BoundEngine.from_problem(TwoWayProblem.from_panel(Y))
    K = 4
    sets = np.array(list(combinations(range(10), K)))
    best = float(eng.cut_value(signs_from_subsets(10, sets)).max())
    ul = lift_upper(eng.R, [], [], list(range(10)), K)
    ut = trs_upper(eng.R, [], [], list(range(10)), K)[0]
    assert ul >= best * (1 - 1e-7)
    assert ul < ut - 0.2 * (ut - best)          # the lift, not the fallback


def test_the_certified_lift_is_the_solvers_value_to_its_accuracy():
    """The bound is rebuilt from the solver's multipliers, so it is valid for
    any of them; with accurate ones it is the relaxation's value."""
    import cvxpy as cp
    from .bounds import _lift_problem, split
    eng = _engine(8, 12, 2)
    I, O, F, K = [0], [1, 2, 6, 7], [3, 4, 5], 3
    certified = lift_upper(eng.R, I, O, F, K)
    c, b, A = split(eng.R, I, O, F)
    unit = max(abs(c), np.abs(b).max(), np.abs(A).max())
    prob = _lift_problem(c / unit, b / unit, A / unit, 2 * (K - len(I)) - len(F))[0]
    prob.solve(solver="CLARABEL")
    assert prob.status == "optimal"
    assert certified == pytest.approx(unit * prob.value, rel=1e-4)


class _Dual:
    def __init__(self, value):
        self.dual_value = value


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 2**31 - 1))
def test_the_lift_certificate_holds_for_any_multipliers(seed):
    """Weak duality: any equality multipliers and any product multipliers,
    negative entries included (the certificate clips them), give a bound."""
    from .bounds import _certify_lift
    rng = np.random.default_rng(seed)
    N, K = 7, 3
    M = rng.standard_normal((N, N)); R = M + M.T
    units = rng.permutation(N)
    I, O, F = sorted(units[:1]), sorted(units[1:3]), sorted(units[3:])
    c, b, A = split(R, I, O, F)
    n, s = len(F), 2 * (K - len(I)) - len(F)
    eq = [_Dual(rng.standard_normal()), _Dual(rng.standard_normal(n)),
          _Dual(rng.standard_normal()), _Dual(rng.standard_normal(n))]
    rlt = [_Dual(rng.standard_normal((n, n))) for _ in range(4)]
    best = max(float(sig @ R @ sig) for sig in (
        np.where(np.isin(np.arange(N), list(I) + list(comb)), 1.0, -1.0)
        for comb in combinations(F, K - len(I))))
    assert _certify_lift(c, b, A, s, eq, rlt) >= best - 1e-9 * max(1.0, abs(best))


def test_the_lift_of_a_zero_matrix_is_its_constant():
    assert lift_upper(np.zeros((6, 6)), [], [0], [1, 2, 3, 4, 5], 2) == 0.0
