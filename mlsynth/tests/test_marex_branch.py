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
from mlsynth.utils.marex_helpers.diagnostics import MAREXSearchWarning
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
    """Every admissible design's objective, best first. A set is admissible
    when the filter accepts it and its treated fit weights every market in it
    (jgreathouse9/mlsynth#950)."""
    n = B.shape[1]
    out = []
    for m in sizes:
        for S in itertools.combinations(range(n), m):
            if not filt.accepts(S):
                continue
            w = side_fit(B, A, list(S), None if lin_t is None else lin_t[list(S)])[1]
            if np.any(w <= 1e-8):
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


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(4, 8), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans(), donor=st.booleans(),
       control_block=st.booleans())
def test_the_node_bound_never_exceeds_the_subtree_minimum(seed, n, m, fix, linear,
                                                           donor, control_block):
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
    nb = node_bound(ExactData(B, A), m, I, O, F, lin_t, lin_c, control_block=control_block)
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


def _solved_duals(monkeypatch, data, m, I, O, F, lin_t=None, lin_c=None):
    """The multipliers a node's solve hands its certificate, and what it scales by."""
    seen = {}
    real = branch_mod._certificate

    def grab(G, scale, r, n_free, sides):
        seen.update(G=G, scale=scale, r=r, n_free=n_free, sides=sides)
        return real(G, scale, r, n_free, sides)
    monkeypatch.setattr(branch_mod, "_certificate", grab)
    node_bound(data, m, I, O, F, lin_t, lin_c)
    monkeypatch.setattr(branch_mod, "_certificate", real)
    return seen


@pytest.mark.parametrize("seed", [0, 2, 3])
@pytest.mark.parametrize("shift, ratio", [(0.01, 0.5), (0.01, 2.0), (0.03, 0.5), (1.0, 1.0)])
def test_the_certificate_holds_when_its_equality_multipliers_shift(monkeypatch, seed,
                                                                   shift, ratio):
    # Multipliers on M_00 = 1 and 1'x = 1 multiply terms that vanish at every
    # design, so shifting them leaves the Lagrangian of every design unchanged
    # while bending the certificate's matrix along the designs themselves: the
    # bound must pay that curvature in full, twice its smallest eigenvalue.
    B, A = _problem(seed, 6, T=8)
    data = ExactData(B, A)
    I, O, F = [], [0], [1, 2, 3, 4, 5]
    seen = _solved_duals(monkeypatch, data, 2, I, O, F)
    sides = [dataclasses.replace(s, y0=s.y0 + shift * ratio, alpha=s.alpha + shift)
             for s in seen["sides"]]
    bound = branch_mod._certificate(seen["G"], seen["scale"], seen["r"],
                                    seen["n_free"], sides)
    assert bound <= Fraction(_subtree_min(B, A, 2, I, O)) * (1 + Fraction(1, 10**12))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_certificate_holds_with_multipliers_from_a_heavier_penalty(monkeypatch, seed):
    # Multipliers solved for the penalty times the certificate's own scale are
    # still multipliers: the certificate must bound the true penalty's subtree
    # with them. A certificate that left the penalty unscaled would be exact
    # for that heavier problem, and report its value.
    B, A = _problem(seed, 6, T=8)
    data = ExactData(B, A)
    lin = np.random.default_rng(seed).uniform(0, 3, 6)
    G = data.gram
    scale = float(Fraction(2) ** round(math.log2(max(G.frac(i, i) for i in range(6)))))
    I, O, F = [], [0], [1, 2, 3, 4, 5]
    seen = _solved_duals(monkeypatch, data, 2, I, O, F, lin * scale, lin * scale)
    sides = [dataclasses.replace(s, lin=lin) for s in seen["sides"]]
    bound = branch_mod._certificate(seen["G"], seen["scale"], seen["r"],
                                    seen["n_free"], sides)
    truth = _subtree_min(B, A, 2, I, O, lin, lin)
    assert bound <= Fraction(truth) * (1 + Fraction(1, 10**12))


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


# --------------------------------------- the bound's control block, kept or not
# The node bound solves a persp+N block per side. When the two blocks will not fit
# in memory, the search bounds the node on the treated block alone plus the
# certified control floor over the markets the node leaves the controls: every
# design's objective is g(S) + h(C(S)) with C(S) inside O + F, so its minimum
# over the node is at least the treated part's bound plus that floor. The answer
# is the same either way; the one block is weaker, by one to two points at the
# root, and needs about half the memory.

def _control_floor(data, cols, lin):
    """The certified lower bound on every control fit inside ``cols``, computed
    here independently of the search: the fit's exact lower end at its weights."""
    cols = list(cols)
    _, v = side_fit(data.B, data.A, cols, None if lin is None else np.asarray(lin)[cols])
    return data.interval(cols, v, lin)[0]


def _certificates_of(fn):
    """Run ``fn`` and return every side list and value ``_certificate`` saw."""
    seen = []
    real = branch_mod._certificate

    def grab(G, scale, r, n_free, sides):
        out = real(G, scale, r, n_free, sides)
        seen.append((sides, out))
        return out
    branch_mod._certificate = grab
    try:
        fn()
    finally:
        branch_mod._certificate = real
    return seen


def test_the_node_bound_solves_both_blocks_by_default():
    B, A = _problem(3, 8)
    seen = _certificates_of(lambda: node_bound(ExactData(B, A), 2, [], [1],
                                               [0, 2, 3, 4, 5, 6, 7]))
    (sides, _), = seen
    assert [s.control for s in sides] == [False, True]


def test_without_the_control_block_the_bound_solves_the_treated_block_alone():
    B, A = _problem(3, 8)
    I, O, F = [], [1], [0, 2, 3, 4, 5, 6, 7]
    seen = _certificates_of(lambda: node_bound(ExactData(B, A), 2, I, O, F,
                                               control_block=False))
    (sides, _), = seen
    assert len(sides) == 1, "the control side's block is still solved"
    assert not sides[0].control and sides[0].U == sorted(I + F)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(4, 8), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans())
def test_one_block_is_the_treated_certificate_plus_the_control_floor(seed, n, m, fix,
                                                                     linear):
    m = min(m, n - 2)
    B, A = _problem(seed, n)
    data = ExactData(B, A)
    rng = np.random.default_rng(seed + 5)
    order = rng.permutation(n).tolist()
    I = sorted(order[:min(fix, m - 1)])
    O = sorted(order[len(I):len(I) + fix])
    F = sorted(set(range(n)) - set(I) - set(O))
    lin_t = rng.uniform(0, 3, n) if linear else None
    lin_c = rng.uniform(0, 3, n) if linear else None
    box = {}
    seen = _certificates_of(lambda: box.update(
        nb=node_bound(data, m, I, O, F, lin_t, lin_c, control_block=False)))
    nb = box["nb"]
    if nb.lower is None:
        return
    (_, treated), = seen
    assert nb.lower == treated + _control_floor(data, sorted(O + F), lin_c)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("control_block", [True, False])
def test_at_a_node_holding_one_set_the_bound_is_its_objective(seed, linear,
                                                              control_block):
    # A convex fit's semidefinite relaxation is exact, and with every market
    # decided the floor is the control fit itself: either form's bound is the
    # design's value, to the certificate's rounding.
    B, A = _problem(seed, 7)
    rng = np.random.default_rng(seed)
    lin_t = rng.uniform(0, 3, 7) if linear else None
    lin_c = rng.uniform(0, 3, 7) if linear else None
    I, O = [1, 4], [0, 2, 3, 5, 6]
    nb = node_bound(ExactData(B, A), 2, I, O, [], lin_t, lin_c, control_block=control_block)
    value = _objective(B, A, (1, 4), lin_t, lin_c)
    assert float(nb.lower) <= value * (1 + 1e-12)
    assert float(nb.lower) >= value - 1e-6 * max(1.0, abs(value))


@pytest.mark.parametrize("control_block", [True, False])
def test_a_node_that_leaves_no_control_market_is_unbounded(control_block):
    B, A = _problem(2, 5)
    nb = node_bound(ExactData(B, A), 5, list(range(5)), [], [], control_block=control_block)
    assert nb.lower is None and nb.relaxation is None


def _memory_between_the_forms(k_treated, k_control):
    """Memory in which one block of ``k_treated`` markets fits and the two
    blocks with ``k_control`` on the control side do not."""
    one = branch_mod.lift_bytes(k_treated)
    two = branch_mod.lift_bytes(k_treated, k_control)
    return (one + two) / 2 / branch_mod.MEMORY_SHARE


def _blocks_of_every_bound(fn):
    seen = _certificates_of(fn)
    return [len(sides) for sides, _ in seen]


def test_a_node_whose_two_blocks_fit_is_bounded_on_both(monkeypatch):
    monkeypatch.setattr(branch_mod, "available_memory", lambda: 64 * 2 ** 30)
    B, A = _problem(4, 9)
    blocks = _blocks_of_every_bound(lambda: _search(B, A, [3], **ALWAYS_LIFT))
    assert blocks and set(blocks) == {2}


@pytest.mark.parametrize("seed", [0, 4, 7])
def test_a_node_whose_two_blocks_do_not_fit_is_bounded_on_one(monkeypatch, seed):
    monkeypatch.setattr(branch_mod, "available_memory",
                        lambda: _memory_between_the_forms(9, 9))
    B, A = _problem(seed, 9)
    box = {}
    blocks = _blocks_of_every_bound(
        lambda: box.update(res=_search(B, A, [3], **ALWAYS_LIFT)))
    assert blocks and blocks[0] == 1, "the root was bounded on both blocks"
    res = box["res"]
    assert res.closed
    assert res.designs[0].treated == _brute(B, A, [3])[0][1]


def test_a_node_whose_one_block_does_not_fit_is_refused(monkeypatch):
    monkeypatch.setattr(branch_mod, "available_memory",
                        lambda: branch_mod.lift_bytes(9) / 2 / branch_mod.MEMORY_SHARE)
    B, A = _problem(4, 9)
    with pytest.raises(MlsynthConfigError, match="9 markets"):
        _search(B, A, [3], **ALWAYS_LIFT)


@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 10**6), share=st.floats(0.0, 1.0))
def test_whichever_form_memory_allows_the_search_returns_the_optimum(seed, share):
    """Memory anywhere from the one block's need to both blocks' with room to
    spare: the design is the brute-force optimum."""
    B, A = _problem(seed, 8)
    one, two = branch_mod.lift_bytes(8), branch_mod.lift_bytes(8, 8)
    memory = (one + share * 2 * (two - one)) / branch_mod.MEMORY_SHARE
    real = branch_mod.available_memory
    branch_mod.available_memory = lambda: memory
    try:
        res = _search(B, A, [3], **ALWAYS_LIFT)
    finally:
        branch_mod.available_memory = real
    assert res.closed and res.designs[0].treated == _brute(B, A, [3])[0][1]


@pytest.mark.parametrize("memory, charged", [("ample", 9), ("tight", 7)])
def test_the_cost_model_charges_the_block_solved(monkeypatch, memory, charged):
    # Two markets forbidden of nine: the treated side can use 7 markets and the
    # controls 9. Both blocks are charged by the larger, one block by its own.
    df, T0 = _panel(J=9)
    units = sorted(df["unit"].unique(), key=str)
    monkeypatch.setattr(exact_mod, "CANDIDATE_LIMIT", 10)
    avail = 64 * 2 ** 30 if memory == "ample" else _memory_between_the_forms(7, 9)
    monkeypatch.setattr(branch_mod, "available_memory", lambda: avail)
    seen = []
    real = branch_mod.branch_and_bound

    def spy(*args, **kwargs):
        def cost(k):
            seen.append(k)
            return 0.0
        return real(*args, price_seconds=1.0, lift_seconds=cost, **kwargs)
    monkeypatch.setattr(branch_mod, "branch_and_bound", spy)
    MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, not_to_be_treated=units[:2]))).fit()
    assert seen and seen[0] == charged


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


# ------------------------------------------------------------- the work limit
#: A set priced at a second and a bound at a second, so a limit of a few
#: seconds stops the search after a bound or two.
UNIT_COSTS = dict(price_seconds=1.0, lift_seconds=lambda k: 1.0)


def _charges(monkeypatch):
    """Record the ``k`` of every bound the search computes: the markets on the
    larger side when both blocks are solved, the treated side's when one is."""
    ks = []
    real = branch_mod.node_bound

    def counted(data, m, I, O, F, lin_t=None, lin_c=None, control_block=True):
        ks.append(max(len(I) + len(F), len(O) + len(F)) if control_block
                  else len(I) + len(F))
        return real(data, m, I, O, F, lin_t, lin_c, control_block=control_block)

    monkeypatch.setattr(branch_mod, "node_bound", counted)
    return ks


def test_a_closed_search_reports_its_work_and_no_open_node():
    B, A = _problem(0, 8)
    res = _search(B, A, [3], **ALWAYS_LIFT)
    assert res.open == 0 and res.closed
    assert res.work > 0


def test_a_search_at_its_limit_returns_its_best_design_and_bracket():
    B, A = _problem(17, 9)
    truth = _brute(B, A, [3])[0][0]
    res = _search(B, A, [3], work_limit=2.5, **UNIT_COSTS)
    assert not res.closed and res.open > 0
    assert res.work <= 2.5
    d, c = res.designs[0], res.certificate
    assert _close(_objective(B, A, d.treated), d.value)
    assert c.lower <= truth * (1 + 1e-12)
    assert truth <= d.value * (1 + 1e-12) and d.value <= c.upper * (1 + 1e-12)
    assert c.lower < c.upper


def test_pricing_is_charged_per_set():
    B, A = _problem(2, 8)
    res = _search(B, A, [3], price_seconds=1e-3, lift_seconds=lambda k: 1.0)
    assert res.lifts == 0
    assert res.work == pytest.approx(res.priced * 1e-3, rel=1e-12)


def test_a_bound_is_charged_at_its_cost(monkeypatch):
    ks = _charges(monkeypatch)
    B, A = _problem(4, 9)
    res = _search(B, A, [3], price_seconds=1.0, lift_seconds=lambda k: 0.5 + k)
    assert res.lifts == len(ks) > 0
    assert res.work == pytest.approx(res.priced * 1.0 + sum(0.5 + k for k in ks),
                                     rel=1e-12)


def test_no_charge_passes_the_limit(monkeypatch):
    """The search does not start a bound or a pricing that would take its
    work past the limit: the work is at most the limit."""
    ks = _charges(monkeypatch)
    B, A = _problem(4, 9)
    full = _search(B, A, [3], price_seconds=1.0, lift_seconds=lambda k: 0.5 + k)
    limit = full.work / 2
    ks.clear()
    res = _search(B, A, [3], work_limit=limit, price_seconds=1.0,
                  lift_seconds=lambda k: 0.5 + k)
    assert res.work <= limit and not res.closed
    assert res.work == pytest.approx(res.priced * 1.0 + sum(0.5 + k for k in ks),
                                     rel=1e-12)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(6, 9), m=st.integers(2, 4),
       limit=st.floats(0.5, 40.0), linear=st.booleans())
def test_property_the_bracket_holds_at_any_limit(seed, n, m, limit, linear):
    """At any limit the search either reports that it found no design, or
    returns a design whose certificate brackets the optimum, with its work
    within the limit; a search with no open node returns the optimum."""
    m = min(m, n - 2)
    B, A = _problem(seed, n)
    rng = np.random.default_rng(seed + 3)
    lin = rng.uniform(0, 3, n) if linear else None
    truth = _brute(B, A, [m], lin_t=lin, lin_c=lin)[0][0]
    try:
        res = _search(B, A, [m], linear_treated=lin, linear_control=lin,
                      work_limit=limit, **UNIT_COSTS)
    except exact_mod.SearchNotClosed:
        return
    d, c = res.designs[0], res.certificate
    assert res.work <= limit
    assert c.lower <= truth * (1 + 1e-12) + 1e-300
    assert truth <= d.value * (1 + 1e-9)
    assert d.value <= c.upper * (1 + 1e-12)
    if res.closed:
        assert _close(d.value, truth)


def test_the_default_limit_closes_the_documented_searches():
    """Walmart with eight treated, the largest search the docs report, is
    modelled at about 150 seconds: 22 bounds at 45 markets and 33 million sets
    priced. The default limit leaves it room."""
    walmart8 = 22 * branch_mod.lift_seconds(45) + 33_257_562 * branch_mod.PRICE_SECONDS
    assert walmart8 < branch_mod.WORK_LIMIT


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


@pytest.mark.parametrize("limit", [0, -1.0, float("nan"), float("inf"), True, "60"])
def test_an_invalid_work_limit_is_refused(limit):
    B, A = _problem(17, 9)
    with pytest.raises(MlsynthConfigError, match="work_limit"):
        _search(B, A, [3], work_limit=limit)


def test_a_limit_reached_before_any_design_is_reported():
    """No design to return: the search says so, names the limit, and does
    not point to a backend that is gone."""
    B, A = _problem(17, 9)
    with pytest.raises(exact_mod.SearchNotClosed, match="work_limit") as err:
        _search(B, A, [3], work_limit=0.5, **UNIT_COSTS)
    assert "mip" not in str(err.value)


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


def test_marex_reports_a_search_that_found_no_design_as_such(monkeypatch):
    def stop(*args, **kwargs):
        raise exact_mod.SearchNotClosed("no design within the limit; raise work_limit")
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    monkeypatch.setattr(branch_mod, "branch_and_bound", stop)
    with pytest.raises(exact_mod.SearchNotClosed) as err:
        MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    assert "infeasible" not in str(err.value)


def test_marex_warns_when_a_search_reaches_its_work_limit(monkeypatch):
    real = branch_mod.branch_and_bound

    def limited(*args, **kwargs):
        return real(*args, work_limit=3.0, price_seconds=1.0,
                    lift_seconds=lambda k: 1.0, **kwargs)
    df, T0 = _panel(J=9)
    want = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3))).fit()
    _low_limit(monkeypatch)
    monkeypatch.setattr(branch_mod, "branch_and_bound", limited)
    with pytest.warns(MAREXSearchWarning, match="work limit"):
        res = MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3, certify=True))).fit()
    cert = res.certificate
    assert cert.lower <= want.globres.objective * (1 + 1e-9)
    assert want.globres.objective <= res.globres.objective * (1 + 1e-9)
    assert res.globres.objective <= cert.upper * (1 + 1e-12)
    counts = res.search.counts
    assert counts.bnb_open > 0 and 0 < counts.bnb_work <= 3.0


def test_marex_above_the_limit_still_refuses_the_coupled_design(monkeypatch):
    df, T0 = _panel(J=9)
    _low_limit(monkeypatch)
    with pytest.raises(MlsynthConfigError, match="backend='mip'"):
        MAREX(MAREXConfig(**_cfg(df, T0, "exact", m_eq=3,
                                 design="weakly_targeted", beta=1.0))).fit()
