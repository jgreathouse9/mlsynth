"""MAREX's exact backend above the candidate limit: a certified branch and bound.

Below ``CANDIDATE_LIMIT`` treated sets the exact backend prices every set. Above
it, the standard, penalized and unit-penalized designs are searched by a branch
and bound (:mod:`mlsynth.utils.marex_helpers.branch`): a node fixes some markets
into the treated set and some out, and is either bounded by the persp+N
relaxation, certified in exact arithmetic, or priced set by set when that is
cheaper. The pieces, tested below:

- ``lambda_min_lower``: an exact lower bound on a symmetric matrix's smallest
  eigenvalue, which the node certificate rests on;
- ``_psd2``: a solver's 2x2 multiplier made positive semidefinite;
- ``node_bound``: the certified persp+N bound of one node;
- ``branch_and_bound``: the search, against brute force over random panels,
  restrictions and penalties, with its certificate;
- MAREX end to end, with the limit lowered so the branch and bound runs.

Levels: unit, property, edge, failure, smoke, end to end.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsynth import MAREX
from mlsynth.exceptions import (MlsynthConfigError, MlsynthDataError,
                                MlsynthEstimationError)
from mlsynth.tests.test_marex_exact_backend import _cfg, _panel
from mlsynth.utils.marex_helpers import branch as branch_mod
from mlsynth.utils.marex_helpers import exact as exact_mod
from mlsynth.utils.marex_helpers.branch import (BranchResult, NodeBound,
                                               _psd2, branch_and_bound,
                                               lambda_min_lower, node_bound)
from mlsynth.utils.marex_helpers.certify import Dyadic, ExactData
from mlsynth.utils.marex_helpers.config import MAREXConfig
from mlsynth.utils.marex_helpers.exact import (ClusterDesign, search_cluster,
                                              side_fit)
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

#: Bound every node holding two or more sets: the lifts decide the search.
ALWAYS_LIFT = dict(price_seconds=1.0, lift_seconds=lambda k: 0.0)
#: Never bound: the root is priced set by set.
NEVER_LIFT = dict(price_seconds=0.0, lift_seconds=lambda k: 1.0)


# ------------------------------------------------------------------ fixtures
def _problem(seed, n, T=12):
    """A random factor panel's markets and their mean, the search's input."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, (2, n)) + rng.normal(size=(T, n))
    return B, B.mean(axis=1)


def _pool(n, S, donors):
    drop = set(S)
    for i, j in donors:
        if i in S:
            drop.add(j)
    return [j for j in range(n) if j not in drop]


def _objective(B, A, S, lin_t=None, lin_c=None, donors=()):
    """``g(S) + h(C(S))``, or ``None`` when ``S`` leaves no control."""
    n = B.shape[1]
    C = _pool(n, S, donors)
    if not C:
        return None
    S = list(S)
    g = side_fit(B, A, S, None if lin_t is None else lin_t[S])[0]
    h = side_fit(B, A, C, None if lin_c is None else lin_c[C])[0]
    return g + h


def _brute(B, A, sizes, filt=SubsetFilter(), lin_t=None, lin_c=None, donors=()):
    """Every admissible design's objective, best first."""
    n = B.shape[1]
    out = []
    for m in sizes:
        for S in itertools.combinations(range(n), m):
            if not filt.accepts(S):
                continue
            f = _objective(B, A, S, lin_t, lin_c, donors)
            if f is not None:
                out.append((f, S))
    return sorted(out)


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * (1.0 + abs(b))


def _exactly_psd(M):
    """Whether the rational symmetric matrix ``M`` is positive semidefinite:
    every principal minor is non-negative."""
    k = len(M)

    def det(rows):
        a = [list(r) for r in rows]
        d = Fraction(1)
        size = len(a)
        for c in range(size):
            p = next((r for r in range(c, size) if a[r][c] != 0), None)
            if p is None:
                return Fraction(0)
            if p != c:
                a[c], a[p] = a[p], a[c]
                d = -d
            d *= a[c][c]
            for r in range(c + 1, size):
                f = a[r][c] / a[c][c]
                for j in range(c, size):
                    a[r][j] -= f * a[c][j]
        return d

    for size in range(1, k + 1):
        for idx in itertools.combinations(range(k), size):
            if det([[M[i][j] for j in idx] for i in idx]) < 0:
                return False
    return True


def _rational(D):
    k = D.N.shape[0]
    return [[D.frac(i, j) for j in range(k)] for i in range(k)]


# ------------------------------------------------------- lambda_min_lower
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), k=st.integers(1, 5),
       scale=st.sampled_from([1e-6, 1.0, 1e6]), rank=st.integers(0, 5))
def test_lambda_min_lower_is_an_exact_lower_bound(seed, k, scale, rank):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, min(rank, k) or 1)) * scale
    S = X @ X.T - rng.uniform(0, 1) * scale ** 2 * np.eye(k)
    S = (S + S.T) / 2
    D = Dyadic.of(S)
    lam = lambda_min_lower(D)
    M = _rational(D)
    shifted = [[M[i][j] - (lam if i == j else 0) for j in range(k)] for i in range(k)]
    assert _exactly_psd(shifted)
    true = float(np.linalg.eigvalsh(S)[0])
    assert float(lam) >= true - 1e-8 * (1.0 + np.abs(S).sum())


def test_lambda_min_lower_of_a_singular_matrix_is_at_most_zero():
    x = np.array([[1.0], [2.0], [3.0]])
    lam = lambda_min_lower(Dyadic.of(x @ x.T))
    assert lam <= 0 and float(lam) > -1e-9


def test_lambda_min_lower_of_a_matrix_past_the_largest_double_is_unknown():
    D = Dyadic(np.array([[10 ** 400, 0], [0, 1]], dtype=object), 0)
    assert lambda_min_lower(D) is None


def test_the_exact_gram_and_transpose():
    B, A = _problem(21, 4, T=5)
    data = ExactData(B, A)
    for i, j in itertools.product(range(4), range(4)):
        want = sum((Fraction(B[t, i]) - Fraction(A[t])) * (Fraction(B[t, j]) - Fraction(A[t]))
                   for t in range(5))
        assert data.gram.frac(i, j) == want
    D = Dyadic.of(np.arange(6.0).reshape(2, 3))
    assert D.T.N.shape == (3, 2) and D.T.frac(2, 1) == D.frac(1, 2)


def test_lambda_min_lower_holds_whatever_factor_it_is_handed(monkeypatch):
    # An eigenvalue estimate one too high, and a "factor" of the matrix shifted
    # by 2I: the bound must come from the exact residual, not from the estimate.
    real_eig, real_chol = np.linalg.eigvalsh, np.linalg.cholesky
    monkeypatch.setattr(branch_mod.np.linalg, "eigvalsh", lambda S: real_eig(S) + 1.0)
    monkeypatch.setattr(branch_mod.np.linalg, "cholesky",
                        lambda M: real_chol(M + 2.0 * np.eye(len(M))))
    S = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 3.0]])
    D = Dyadic.of(S)
    lam = lambda_min_lower(D)
    M = _rational(D)
    assert _exactly_psd([[M[i][j] - (lam if i == j else 0) for j in range(3)]
                         for i in range(3)])


def test_lambda_min_lower_reports_failure_instead_of_guessing(monkeypatch):
    def refuse(*args, **kwargs):
        raise np.linalg.LinAlgError("refused")
    monkeypatch.setattr(branch_mod.np.linalg, "cholesky", refuse)
    assert lambda_min_lower(Dyadic.of(np.eye(3))) is None


# ------------------------------------------------------------------ _psd2
@given(a=st.floats(-2, 2), b=st.floats(-2, 2), c=st.floats(-2, 2))
def test_a_repaired_multiplier_is_positive_semidefinite(a, b, c):
    a2, b2, c2 = _psd2(np.array([[a, b], [b, c]]))
    assert a2 >= 0 and c2 >= 0
    assert Fraction(b2) ** 2 <= Fraction(a2) * Fraction(c2)


def test_a_positive_semidefinite_multiplier_is_left_alone():
    assert _psd2(np.array([[2.0, 1.0], [1.0, 3.0]])) == (2.0, 1.0, 3.0)


def test_a_multiplier_whose_diagonal_product_underflows_is_repaired_at_once():
    # a * c is subnormal, so sqrt(a * c) can overshoot sqrt(a) sqrt(c) by
    # 1.3e15 units in the last place; walking that back one unit at a time
    # does not finish.
    import signal

    def stop(signum, frame):
        raise TimeoutError("the repair did not finish")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.alarm(5)
    try:
        a, c = 2.0446911808705326e-163, 1.2089356071576097e-161
        a2, b2, c2 = _psd2(np.array([[a, 1.0], [1.0, c]]))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert Fraction(b2) ** 2 <= Fraction(a2) * Fraction(c2)
    assert b2 > 0.5 * np.sqrt(a) * np.sqrt(c)


# -------------------------------------------------------------- node_bound
def _subtree_min(B, A, m, I, O, lin_t=None, lin_c=None, donors=()):
    filt = SubsetFilter(forced_in=frozenset(I), forbidden=frozenset(O))
    found = _brute(B, A, [m], filt, lin_t, lin_c, donors)
    return found[0][0] if found else None


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(4, 8), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans(), donor=st.booleans())
def test_the_node_bound_never_exceeds_the_subtree_minimum(seed, n, m, fix, linear,
                                                           donor):
    m = min(m, n - 2)
    B, A = _problem(seed, n)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n).tolist()
    I = sorted(order[:min(fix, m - 1)])
    O = sorted(order[len(I):len(I) + fix])
    F = sorted(set(range(n)) - set(I) - set(O))
    lin_t = rng.uniform(0, 3, n) if linear else None
    lin_c = rng.uniform(0, 3, n) if linear else None
    donors = ((order[0], order[-1]),) if donor else ()
    best = _subtree_min(B, A, m, I, O, lin_t, lin_c, donors)
    if best is None:
        return
    nb = node_bound(ExactData(B, A), m, I, O, F, lin_t, lin_c)
    assert isinstance(nb, NodeBound)
    if nb.lower is not None:
        assert nb.lower <= Fraction(best) * (1 + Fraction(1, 10**12)) + Fraction(1, 10**300)


def _exact_objective(G, S, w, C, v, lin_t, lin_c):
    """``w'G_S w + a'w + v'G_C v + b'v`` in exact arithmetic."""
    out = Fraction(0)
    for cols, x, lin in ((S, w, lin_t), (C, v, lin_c)):
        for a, i in enumerate(cols):
            for b, j in enumerate(cols):
                out += G.frac(i, j) * x[a] * x[b]
            if lin is not None:
                out += Fraction(float(lin[i])) * x[a]
    return out


def _simplex(rng, k):
    raw = [Fraction(int(v)) for v in rng.integers(0, 50, k)]
    raw[int(rng.integers(k))] += 1
    total = sum(raw)
    return [q / total for q in raw]


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(3, 7), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans(),
       size=st.sampled_from([1e-3, 1.0, 1e3]))
def test_the_certificate_holds_for_any_multipliers(seed, n, m, fix, linear, size):
    # The certificate's claim: whatever multipliers it is handed, of any size
    # and sign, the bound it certifies is below every design in the subtree, at
    # every feasible choice of weights. The solver's multipliers are one choice.
    m = min(m, n - 1)
    B, A = _problem(seed, n, T=6)
    data = ExactData(B, A)
    rng = np.random.default_rng(seed + 3)
    order = rng.permutation(n).tolist()
    I = sorted(order[:min(fix, m - 1)])
    O = sorted(order[len(I):len(I) + fix])
    F = sorted(set(range(n)) - set(I) - set(O))
    r = m - len(I)
    if r > len(F) or len(O) + len(F) - r < 1:
        return
    S = sorted(I + rng.choice(F, r, replace=False).tolist())
    C = sorted(set(O + F) - set(S))
    lin_t = rng.uniform(0, 3, n) if linear else None
    lin_c = rng.uniform(0, 3, n) if linear else None
    G = data.gram
    gmax = max(G.frac(i, i) for i in range(n))
    scale = Fraction(2) ** round(math.log2(gmax))
    pos = {j: i for i, j in enumerate(F)}
    sides = []
    for U, control, lin in ((sorted(I + F), False, lin_t), (sorted(O + F), True, lin_c)):
        k = len(U)
        free = [(i, pos[j]) for i, j in enumerate(U) if j in pos]
        noise = lambda *shape: rng.normal(scale=size, size=shape)  # noqa: E731
        sides.append(branch_mod._SideDuals(
            U=U, free=free, control=control, lin=lin, y0=float(noise()),
            alpha=float(noise()), beta=noise(k), H=noise(k, k),
            mu=[float(noise()) for _ in free], Lam=[noise(2, 2) for _ in free]))
    bound = branch_mod._certificate(G, scale, r, len(F), sides)
    if bound is None:
        return
    w, v = _simplex(rng, len(S)), _simplex(rng, len(C))
    assert bound <= _exact_objective(G, S, w, C, v, lin_t, lin_c)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_node_bound_is_as_tight_as_its_relaxation(seed):
    B, A = _problem(seed, 9, T=20)
    nb = node_bound(ExactData(B, A), 3, [], [], list(range(9)))
    assert nb.lower is not None and nb.relaxation is not None
    assert float(nb.lower) <= nb.relaxation * (1 + 1e-6)
    assert float(nb.lower) >= nb.relaxation - 1e-4 * abs(nb.relaxation)


def test_the_node_bound_is_positive_where_the_natural_relaxation_is_zero():
    # The natural relaxation scores exactly 0 here (requirement 1 of the study);
    # persp+N sees the second moments.
    B, A = _problem(4, 9, T=20)
    nb = node_bound(ExactData(B, A), 3, [], [], list(range(9)))
    assert nb.lower > 0


def test_the_node_bound_reports_the_indicators():
    B, A = _problem(7, 8)
    nb = node_bound(ExactData(B, A), 2, [], [], list(range(8)))
    assert set(nb.z) == set(range(8))
    assert abs(sum(nb.z.values()) - 2) < 1e-5


def test_a_node_whose_treated_set_is_complete():
    # r = 0: every free market is a control, and the bound stays under the one
    # design left.
    B, A = _problem(5, 7)
    I, O, F = [0, 1], [2], [3, 4, 5, 6]
    nb = node_bound(ExactData(B, A), 2, I, O, F)
    assert nb.lower <= Fraction(_objective(B, A, (0, 1))) * (1 + Fraction(1, 10**12))


def test_a_degenerate_panel_gets_a_bound_of_zero_or_less():
    B = np.ones((6, 5))
    nb = node_bound(ExactData(B, B.mean(axis=1)), 2, [], [], list(range(5)))
    assert nb.lower is None or nb.lower <= 0


def test_a_failed_solve_leaves_the_node_unbounded(monkeypatch):
    import cvxpy as cp

    def fail(self, *args, **kwargs):
        raise cp.error.SolverError("refused")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    B, A = _problem(3, 6)
    nb = node_bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None and nb.z is None and nb.relaxation is None


def test_a_solve_without_multipliers_leaves_the_node_unbounded(monkeypatch):
    import cvxpy as cp

    def no_duals(self, *args, **kwargs):
        return None
    monkeypatch.setattr(cp.Problem, "solve", no_duals)
    B, A = _problem(3, 6)
    nb = node_bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None


def test_multipliers_that_are_missing_or_not_finite_are_refused():
    class Con:
        def __init__(self, d):
            self.dual_value = d

    class Side:
        def __init__(self, d):
            self.c00 = self.cnn = self.csum = self.crow = Con(np.ones(2))
            self.clin, self.cper = [Con(d)], []

    assert branch_mod._duals_finite([Side(1.0)])
    assert not branch_mod._duals_finite([Side(None)])
    assert not branch_mod._duals_finite([Side(np.nan)])


def test_an_eigenvalue_bound_that_fails_leaves_the_node_unbounded(monkeypatch):
    monkeypatch.setattr(branch_mod, "lambda_min_lower", lambda D: None)
    B, A = _problem(3, 6)
    nb = node_bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None


# ------------------------------------------------------------- the search
def _search(B, A, sizes, **kw):
    kw.setdefault("filt", SubsetFilter())
    return branch_and_bound(B, A, sizes, **kw)


def test_the_search_returns_a_certified_design():
    B, A = _problem(0, 8)
    res = _search(B, A, [3], **ALWAYS_LIFT)
    assert isinstance(res, BranchResult)
    d = res.designs[0]
    assert isinstance(d, ClusterDesign) and len(d.treated) == 3
    assert d.w.sum() == pytest.approx(1.0) and d.v.sum() == pytest.approx(1.0)
    best = _brute(B, A, [3])[0][0]
    assert res.certificate.lower <= best * (1 + 1e-12)
    assert d.value <= res.certificate.upper * (1 + 1e-12)
    assert res.certificate.gap <= 1e-9 * (1 + best)
    assert res.lifts > 0


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(5, 9), m=st.integers(1, 3),
       two_sizes=st.booleans(), linear=st.booleans(), donor=st.booleans(),
       conflict=st.booleans(), forced=st.booleans(), forbidden=st.booleans(),
       quota=st.booleans(), lift=st.booleans())
def test_the_search_is_exact_over_the_domain(seed, n, m, two_sizes, linear, donor,
                                             conflict, forced, forbidden, quota,
                                             lift):
    m = min(m, n - 2)
    sizes = [m, m + 1] if two_sizes else [m]
    B, A = _problem(seed, n)
    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(n).tolist()
    filt = SubsetFilter(
        forced_in=frozenset(order[:1]) if forced else frozenset(),
        forbidden=frozenset(order[1:2]) if forbidden else frozenset(),
        conflict_pairs=((order[2], order[3]),) if conflict else (),
        strata=(((order[3], order[4]), None, 1),) if quota else ())
    lin_t = rng.uniform(0, 3, n) if linear else None
    lin_c = rng.uniform(0, 3, n) if linear else None
    donors = ((order[2], order[-1]),) if donor else ()
    truth = _brute(B, A, sizes, filt, lin_t, lin_c, donors)
    costs = ALWAYS_LIFT if lift else NEVER_LIFT
    if not truth:
        with pytest.raises(MlsynthEstimationError):
            _search(B, A, sizes, filt=filt, linear_treated=lin_t,
                    linear_control=lin_c, donor_exclusion=donors, **costs)
        return
    res = _search(B, A, sizes, filt=filt, linear_treated=lin_t,
                  linear_control=lin_c, donor_exclusion=donors, **costs)
    d = res.designs[0]
    assert _close(d.value, truth[0][0])
    assert filt.accepts(d.treated)
    assert _close(_objective(B, A, d.treated, lin_t, lin_c, donors), d.value)
    assert list(d.control) == _pool(n, d.treated, donors)
    assert res.certificate.lower <= truth[0][0] * (1 + 1e-12) + 1e-300
    assert truth[0][0] <= res.certificate.upper * (1 + 1e-12)


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(6, 9), k=st.integers(2, 4))
def test_the_search_keeps_the_top_k_designs(seed, n, k):
    B, A = _problem(seed, n)
    truth = _brute(B, A, [2])[:k]
    res = _search(B, A, [2], top_k=k, **ALWAYS_LIFT)
    assert len(res.designs) == len(truth)
    for d, (f, _) in zip(res.designs, truth):
        assert _close(d.value, f)


@pytest.mark.parametrize("seed", [11, 12])
def test_the_search_agrees_with_the_enumeration(seed):
    B, A = _problem(seed, 10, T=16)
    lin = np.random.default_rng(seed).uniform(0, 2, 10)
    filt = SubsetFilter(conflict_pairs=((0, 1),))
    kw = dict(filt=filt, linear_treated=lin, linear_control=lin, top_k=2)
    enum = search_cluster(B, A, [3], **kw)
    for costs in (ALWAYS_LIFT, NEVER_LIFT):
        got = branch_and_bound(B, A, [3], **kw, **costs).designs
        assert [d.treated for d in got] == [d.treated for d in enum]
        for a, b in zip(got, enum):
            assert _close(a.value, b.value, 1e-10)


def test_without_lifts_the_search_prices_every_set():
    B, A = _problem(2, 8)
    res = _search(B, A, [3], **NEVER_LIFT)
    assert res.lifts == 0
    assert _close(res.designs[0].value, _brute(B, A, [3])[0][0])


def test_the_default_costs_bound_a_large_root():
    # 40 markets, 8 treated: 77 million sets, which the default costs price
    # far above one persp+N bound, so the root is bounded; ten thousand sets
    # are priced.
    assert branch_mod.PRICE_SECONDS * 76_904_685 > branch_mod.lift_seconds(40)
    assert branch_mod.PRICE_SECONDS * 10_000 < branch_mod.lift_seconds(10)


# ------------------------------------------------------------- edge cases
def test_an_exact_tie_is_reported_unresolved():
    B, A = _problem(6, 7)
    B = np.column_stack([B, B[:, 2]])          # market 7 duplicates market 2
    A = B.mean(axis=1)
    truth = _brute(B, A, [1])
    res = _search(B, A, [1], **ALWAYS_LIFT)
    if truth[0][1] in ((2,), (7,)):
        other = (7,) if res.designs[0].treated == (2,) else (2,)
        assert other in res.certificate.unresolved


def test_a_cluster_one_larger_than_the_treated_set():
    B, A = _problem(8, 4)
    res = _search(B, A, [3], **ALWAYS_LIFT)
    assert _close(res.designs[0].value, _brute(B, A, [3])[0][0])


def test_every_treated_market_forced():
    B, A = _problem(9, 6)
    filt = SubsetFilter(forced_in=frozenset({1, 4}))
    res = _search(B, A, [2], filt=filt, **ALWAYS_LIFT)
    assert res.designs[0].treated == (1, 4)


def test_one_free_choice_left():
    B, A = _problem(10, 7)
    filt = SubsetFilter(forced_in=frozenset({0, 3}))
    res = _search(B, A, [3], filt=filt, **ALWAYS_LIFT)
    assert _close(res.designs[0].value, _brute(B, A, [3], filt)[0][0])


def test_a_conflict_on_the_branching_market_is_carried_to_the_child():
    B, A = _problem(13, 8)
    pairs = tuple((i, j) for i in range(8) for j in range(i + 1, 8) if (i + j) % 3 == 0)
    filt = SubsetFilter(conflict_pairs=pairs)
    res = _search(B, A, [3], filt=filt, **ALWAYS_LIFT)
    assert filt.accepts(res.designs[0].treated)
    assert _close(res.designs[0].value, _brute(B, A, [3], filt)[0][0])


def test_a_collinear_panel_is_still_searched_exactly():
    B, A = _problem(14, 5, T=4)
    B = np.column_stack([B, B[:, :3] @ np.array([0.2, 0.3, 0.5])])
    A = B.mean(axis=1)
    res = _search(B, A, [2], **ALWAYS_LIFT)
    assert _close(res.designs[0].value, _brute(B, A, [2])[0][0])


def test_the_search_still_holds_when_every_solve_fails(monkeypatch):
    import cvxpy as cp

    def fail(self, *args, **kwargs):
        raise cp.error.SolverError("refused")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    B, A = _problem(15, 8)
    res = _search(B, A, [3], **ALWAYS_LIFT)
    assert _close(res.designs[0].value, _brute(B, A, [3])[0][0])


# ------------------------------------------------------------- failures
def test_no_admissible_design_is_reported():
    B, A = _problem(16, 5)
    filt = SubsetFilter(forced_in=frozenset({0}), forbidden=frozenset({0}))
    with pytest.raises(MlsynthEstimationError, match="admissible"):
        _search(B, A, [2], filt=filt, **ALWAYS_LIFT)


def test_a_set_whose_exclusions_leave_no_control_is_not_a_design():
    B, A = _problem(19, 4)
    donors = ((0, 1), (0, 2), (0, 3))
    res = _search(B, A, [1], donor_exclusion=donors, **ALWAYS_LIFT)
    assert res.designs[0].treated != (0,)
    assert _close(res.designs[0].value, _brute(B, A, [1], donors=donors)[0][0])


def test_no_design_with_a_control_is_reported():
    B, A = _problem(20, 2)
    with pytest.raises(MlsynthEstimationError, match="no control"):
        _search(B, A, [1], donor_exclusion=((0, 1), (1, 0)), **ALWAYS_LIFT)
    with pytest.raises(MlsynthEstimationError, match="no control"):
        _search(*_problem(20, 4), [4], **ALWAYS_LIFT)


def test_a_quota_the_treated_count_cannot_meet_admits_no_design():
    B, A = _problem(23, 6)
    filt = SubsetFilter(strata=(((0, 1), 2, None),))
    with pytest.raises(MlsynthEstimationError, match="admissible"):
        _search(B, A, [1], filt=filt, **ALWAYS_LIFT)


def test_forced_units_that_conflict_admit_no_design():
    B, A = _problem(22, 6)
    filt = SubsetFilter(forced_in=frozenset({0, 1}), conflict_pairs=((0, 1),))
    with pytest.raises(MlsynthEstimationError, match="admissible"):
        _search(B, A, [2], filt=filt, **ALWAYS_LIFT)


def test_a_search_that_runs_out_of_lifts_reports_its_gap():
    B, A = _problem(17, 9)
    with pytest.raises(MlsynthEstimationError, match="gap") as err:
        _search(B, A, [3], max_lifts=1, **ALWAYS_LIFT)
    assert "backend='mip'" in str(err.value)


def test_non_finite_data_is_refused():
    B, A = _problem(18, 6)
    B[0, 0] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        _search(B, A, [2], **ALWAYS_LIFT)


# ------------------------------------------------------------- end to end
def _low_limit(monkeypatch, limit=10):
    monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", limit)


@pytest.mark.parametrize("over", [
    dict(),
    dict(design="penalized", lambda1=0.5, lambda2=0.5),
    dict(design="unit_penalized", xi=0.3),
])
def test_marex_above_the_limit_returns_the_enumeration_s_design(monkeypatch, over):
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, **over))).fit()
    _low_limit(monkeypatch)
    got = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, **over))).fit()
    assert set(map(str, got.assignment["treated"])) == set(map(str, want.assignment["treated"]))
    assert got.globres.objective == pytest.approx(want.globres.objective, rel=1e-9)


def test_marex_above_the_limit_certifies_its_design(monkeypatch):
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, certify=True))).fit()
    cert = res.certificate
    assert cert.lower <= res.globres.objective * (1 + 1e-12)
    assert res.globres.objective <= cert.upper * (1 + 1e-12)


def test_marex_above_the_limit_keeps_a_pool(monkeypatch):
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, top_K=3))).fit()
    _low_limit(monkeypatch)
    got = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=2, top_K=3))).fit()
    assert [frozenset(map(str, p["markets"])) for p in got.pool] == \
           [frozenset(map(str, p["markets"])) for p in want.pool]
    assert [p["mip_objective"] for p in got.pool] == pytest.approx(
        [p["mip_objective"] for p in want.pool], rel=1e-9)


def test_marex_reports_a_search_that_did_not_close_as_such(monkeypatch):
    def stop(*args, **kwargs):
        raise exact_mod.SearchNotClosed("gap left open; use backend='mip'")
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    monkeypatch.setattr(branch_mod, "branch_and_bound", stop)
    with pytest.raises(exact_mod.SearchNotClosed) as err:
        MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert "infeasible" not in str(err.value)


def test_marex_above_the_limit_still_refuses_the_coupled_design(monkeypatch):
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    with pytest.raises(MlsynthConfigError, match="backend='mip'"):
        MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3,
                                 design="weakly_targeted", beta=1.0))).fit()
