"""The persp+N bound of one simplex fit under a cardinality, and its certificate.

:mod:`mlsynth.utils.persp_bound` bounds ``min w'G w + a'w`` over the weights on
``m`` units with the fixed-in units among them, by the face form of persp+N, and
certifies the bound in exact arithmetic from whatever multipliers the solver
returns. MAREX's node bound is two of these sides sharing their indicators;
LEXSCM's is one, with its options: a ridge on the weights, a floor on every
chosen weight, a budget on the chosen units, and pairs never chosen together.

Levels: unit (the certificate's pieces), property (the bound is a bound, over
the domain and for any multipliers), edge (nothing free, one free unit, a
degenerate panel, a stopped solve), failure (a solve or an eigenvalue bound that
fails leaves the node unbounded).
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

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils import persp_bound as pb
from mlsynth.utils.fast_scm_helpers.unit_level import solve_penalised_weights
from mlsynth.utils.marex_helpers.certify import ExactData


# ------------------------------------------------------------------ fixtures
def _problem(seed, n, T=12):
    """A random factor panel's markets and their mean."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, (2, n)) + rng.normal(size=(T, n))
    return B, B.mean(axis=1)


def _fit(data, S, lin=None, floor=0.0, ridge=0.0):
    """``min w'(G_S + ridge I) w + a'w`` over the simplex on ``S`` with every
    weight at least ``floor``, by the library's floored solver."""
    S = list(S)
    Q = data.Gf[np.ix_(S, S)] + ridge * np.eye(len(S))
    a = np.zeros(len(S)) if lin is None else np.asarray(lin, float)[S]
    w, _ = solve_penalised_weights(Q, a, min_weight=floor)
    return float(w @ Q @ w + a @ w)


def _designs(n, m, I, O, costs=None, budget=None, conflicts=()):
    for S in itertools.combinations(range(n), m):
        if not set(I) <= set(S) or set(O) & set(S):
            continue
        if budget is not None and sum(costs[j] for j in S) > budget + 1e-12:
            continue
        if any(i in S and j in S for i, j in conflicts):
            continue
        yield S


def _subtree_min(data, m, I, O, lin=None, floor=0.0, costs=None, budget=None, conflicts=(),
                 ridge=0.0):
    vals = [_fit(data, S, lin, floor, ridge)
            for S in _designs(data.n, m, I, O, costs, budget, conflicts)]
    return min(vals) if vals else None


def _node(seed, n, m, fix):
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n).tolist()
    I = sorted(order[:min(fix, m - 1)])
    O = sorted(order[len(I):len(I) + fix])
    F = sorted(set(range(n)) - set(I) - set(O))
    return rng, I, O, F


def _below(lower, value):
    return lower <= Fraction(value) * (1 + Fraction(1, 10**9)) + Fraction(1, 10**12)


# ------------------------------------------------------------- the bound
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(4, 8), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans(), floor=st.booleans(),
       budget=st.booleans(), conflicts=st.booleans(), ridge=st.sampled_from([0.0, 0.3, 30.0]))
def test_the_bound_never_exceeds_the_subtree_minimum(seed, n, m, fix, linear, floor,
                                                     budget, conflicts, ridge):
    m = min(m, n - 1)
    B, A = _problem(seed, n)
    data = ExactData(B, A)
    rng, I, O, F = _node(seed, n, m, fix)
    opt = dict(linear=rng.uniform(0, 3, n) if linear else None,
               floor=0.5 / m if floor and m > 1 else 0.0, ridge=ridge)
    if budget:
        costs = rng.uniform(1.0, 3.0, n)
        opt.update(costs=costs, budget=float(np.sort(costs)[:m].sum() + 0.5 * m))
    if conflicts:
        opt["conflicts"] = tuple(tuple(sorted(int(v) for v in rng.choice(n, 2, replace=False)))
                                 for _ in range(2))
    best = _subtree_min(data, m, I, O, opt["linear"], opt["floor"], opt.get("costs"),
                        opt.get("budget"), opt.get("conflicts", ()), ridge)
    nb = pb.bound(data, m, I, O, F, **opt)
    assert isinstance(nb, pb.NodeBound)
    if best is None or nb.lower is None:
        return
    assert _below(nb.lower, best)
    assert nb.relaxation <= best + 1e-6 * (1.0 + abs(best))       # the program itself


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_certificate_is_as_tight_as_the_relaxation(seed):
    B, A = _problem(seed, 9, T=20)
    nb = pb.bound(ExactData(B, A), 3, [], [], list(range(9)))
    assert float(nb.lower) <= nb.relaxation * (1 + 1e-9)
    assert float(nb.lower) >= nb.relaxation - 1e-5 * abs(nb.relaxation)
    assert nb.status == "optimal"


@pytest.mark.parametrize("option", ["floor", "budget", "conflicts", "ridge"])
def test_an_option_in_the_bound_never_lowers_it(option):
    B, A = _problem(5, 9, T=20)
    data = ExactData(B, A)
    plain = pb.bound(data, 3, [], [], list(range(9)))
    costs = np.linspace(1.0, 3.0, 9)
    extra = {"floor": dict(floor=0.2),
             "budget": dict(costs=costs, budget=5.0),
             "conflicts": dict(conflicts=((0, 1), (2, 3), (4, 5), (6, 7))),
             "ridge": dict(ridge=0.5 * float(np.mean(np.diag(data.Gf))))}[option]
    tighter = pb.bound(data, 3, [], [], list(range(9)), **extra)
    assert tighter.relaxation >= plain.relaxation - 1e-7 * abs(plain.relaxation)
    assert _below(tighter.lower, _subtree_min(data, 3, [], [], floor=extra.get("floor", 0.0),
                                              costs=extra.get("costs"), budget=extra.get("budget"),
                                              conflicts=extra.get("conflicts", ()),
                                              ridge=extra.get("ridge", 0.0)))


def test_the_bound_reports_the_indicators():
    B, A = _problem(7, 8)
    nb = pb.bound(ExactData(B, A), 2, [], [], list(range(8)))
    assert set(nb.z) == set(range(8))
    assert abs(sum(nb.z.values()) - 2) < 1e-5


# ---------------------------------------------- the certificate, for any multipliers
def _random_duals(rng, size, side, n_rows):
    noise = lambda *shape: rng.normal(scale=size, size=shape)  # noqa: E731
    k = len(side.U)
    return dataclasses.replace(
        side, alpha=float(noise()), H=noise(k, k),
        mu=tuple(float(noise()) for _ in side.free),
        Lam=tuple(noise(2, 2) for _ in side.free),
        nu=tuple(float(noise()) for _ in side.floored),
        rho=tuple(float(noise()) for _ in side.floored),
        tau=tuple(float(noise()) for _ in side.apart)), [float(noise()) for _ in range(n_rows)]


def _exact_value(G, S, w, lin):
    out = Fraction(0)
    for a, i in enumerate(S):
        for b, j in enumerate(S):
            out += G.frac(i, j) * w[a] * w[b]
        if lin is not None:
            out += Fraction(float(lin[i])) * w[a]
    return out


def _floored_simplex(rng, k, floor):
    raw = [Fraction(int(v)) for v in rng.integers(0, 50, k)]
    raw[int(rng.integers(k))] += 1
    total = sum(raw)
    c = Fraction(floor)
    return [c + (1 - k * c) * q / total for q in raw]


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10**6), n=st.integers(3, 7), m=st.integers(1, 3),
       fix=st.integers(0, 2), linear=st.booleans(), floor=st.booleans(),
       budget=st.booleans(), conflicts=st.booleans(), ridge=st.sampled_from([0.0, 0.5]),
       size=st.sampled_from([1e-3, 1.0, 1e3]))
def test_the_certificate_holds_for_any_multipliers(seed, n, m, fix, linear, floor, budget,
                                                   conflicts, ridge, size):
    # Whatever multipliers the certificate is handed, of any size and sign, the
    # bound it certifies is below every design of the node at every feasible
    # choice of weights.
    m = min(m, n)
    B, A = _problem(seed, n, T=6)
    data = ExactData(B, A)
    rng, I, O, F = _node(seed, n, m, fix)
    r = m - len(I)
    if r > len(F):
        return
    lin = rng.uniform(0, 3, n) if linear else None
    c = 0.5 / m if floor and m > 1 else 0.0
    costs = rng.uniform(1.0, 3.0, n) if budget else None
    cap = float(np.sort(costs)[:m].sum() + 0.5 * m) if budget else None
    pairs = tuple(tuple(sorted(int(v) for v in rng.choice(n, 2, replace=False)))
                  for _ in range(2)) if conflicts and n > 1 else ()
    found = list(_designs(n, m, I, O, costs, cap, pairs))
    if not found:
        return
    S = list(found[int(rng.integers(len(found)))])
    plan = pb.plan(data, m, I, O, F, linear=lin, floor=c, costs=costs, budget=cap,
                   conflicts=pairs, ridge=ridge)
    side, rows = _random_duals(rng, size, plan.sides[0], len(plan.rows))
    bound = pb.certificate(plan, [side], rows)
    if bound is None:
        return
    w = _floored_simplex(rng, len(S), c)
    assert bound <= _exact_value(data.gram, S, w, lin) + Fraction(ridge) * sum(v * v for v in w)


@pytest.mark.parametrize("seed", [0, 2, 3])
@pytest.mark.parametrize("shift", [-1.0, 0.1, 0.5])
@pytest.mark.parametrize("ridge", [0.0, 0.5])
def test_a_shifted_alpha_is_paid_for_by_the_eigenvalue_term(seed, shift, ridge):
    # 1'X1 = 1 makes a shift in alpha cancel at every design, but the
    # certificate sees only tr X <= 1: the shift moves the constant by delta
    # and S by -delta 1 1', and the eigenvalue term has to take it back. A
    # shift of a tenth of the relaxation, in the certificate's scaled units,
    # leaves anything added to S besides (G + ridge I)/scale room to hide the
    # payment and lift the bound above the subtree.
    B, A = _problem(seed, 6, T=8)
    data = ExactData(B, A)
    r = ridge * float(np.mean(np.diag(data.Gf)))
    plan = pb.plan(data, 2, [], [0], [1, 2, 3, 4, 5], ridge=r)
    assert plan.scale >= 2                      # the scale has to matter for this to test it
    nb, duals, rows = pb.solve(plan)
    delta = shift * nb.relaxation / float(plan.scale)
    moved = [dataclasses.replace(d, alpha=d.alpha + delta) for d in duals]
    bound = pb.certificate(plan, moved, rows)
    assert _below(bound, _subtree_min(data, 2, [], [0], ridge=r))


def test_a_solve_stopped_early_still_certifies_a_bound():
    B, A = _problem(4, 10, T=16)
    data = ExactData(B, A)
    full = pb.bound(data, 3, [], [], list(range(10)))
    for cap in (3, 6, 10):
        early = pb.bound(data, 3, [], [], list(range(10)), max_iter=cap)
        assert early.status != "optimal"
        if early.lower is not None:
            assert early.lower <= full.lower * (1 + Fraction(1, 10**9))
            assert _below(early.lower, _subtree_min(data, 3, [], []))


# ------------------------------------------------------------------ edges
def test_a_node_with_nothing_left_to_choose():
    B, A = _problem(5, 7)
    data = ExactData(B, A)
    nb = pb.bound(data, 2, [0, 1], [2], [3, 4, 5, 6])
    assert _below(nb.lower, _fit(data, (0, 1)))


def test_a_node_with_one_free_unit():
    B, A = _problem(6, 6)
    data = ExactData(B, A)
    nb = pb.bound(data, 3, [0, 1], [2, 3, 4], [5])
    assert _below(nb.lower, _fit(data, (0, 1, 5)))


def test_a_floor_on_fixed_in_units_holds_their_weights():
    B, A = _problem(8, 7, T=14)
    data = ExactData(B, A)
    nb = pb.bound(data, 3, [0, 1], [], [2, 3, 4, 5, 6], floor=0.3)
    assert _below(nb.lower, _subtree_min(data, 3, [0, 1], [], floor=0.3))


def test_a_degenerate_panel_gets_a_bound_of_zero_or_less():
    B = np.ones((6, 5))
    nb = pb.bound(ExactData(B, B.mean(axis=1)), 2, [], [], list(range(5)))
    assert nb.lower is None or nb.lower <= 0


def test_a_budget_no_completion_meets_leaves_the_node_unbounded():
    B, A = _problem(9, 6)
    nb = pb.bound(ExactData(B, A), 3, [], [], list(range(6)), costs=np.ones(6), budget=2.0)
    assert nb.lower is None


# ---------------------------------------------------------------- failures
def test_a_failed_solve_leaves_the_node_unbounded(monkeypatch):
    import cvxpy as cp

    def fail(self, *args, **kwargs):
        raise cp.error.SolverError("refused")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    B, A = _problem(3, 6)
    nb = pb.bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None and nb.z is None and nb.relaxation is None


def test_a_solve_without_multipliers_leaves_the_node_unbounded(monkeypatch):
    import cvxpy as cp

    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    B, A = _problem(3, 6)
    nb = pb.bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None


def test_an_eigenvalue_bound_that_fails_leaves_the_node_unbounded(monkeypatch):
    monkeypatch.setattr(pb, "lambda_min_lower", lambda D: None)
    B, A = _problem(3, 6)
    nb = pb.bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None


def test_a_floor_that_leaves_no_weights_is_refused():
    B, A = _problem(3, 6)
    with pytest.raises(MlsynthConfigError, match="floor"):
        pb.bound(ExactData(B, A), 3, [], [], list(range(6)), floor=0.34)


def test_a_budget_without_costs_is_refused():
    B, A = _problem(3, 6)
    with pytest.raises(MlsynthConfigError, match="cost"):
        pb.bound(ExactData(B, A), 3, [], [], list(range(6)), budget=5.0)


def test_a_negative_ridge_is_refused():
    B, A = _problem(3, 6)
    with pytest.raises(MlsynthConfigError, match="ridge"):
        pb.bound(ExactData(B, A), 3, [], [], list(range(6)), ridge=-1.0)


def test_a_node_holding_a_conflicting_pair_has_no_design():
    B, A = _problem(3, 6)
    nb = pb.bound(ExactData(B, A), 3, [0, 1], [], [2, 3, 4, 5], conflicts=((0, 1),))
    assert nb.lower is None


def test_a_block_that_was_never_solved_has_no_multipliers():
    import cvxpy as cp

    B, A = _problem(3, 6)
    plan = pb.plan(ExactData(B, A), 2, [], [], list(range(6)))
    block = pb._Block(cp, plan, plan.sides[0], cp.Variable(6))
    assert block.duals() is None


def test_a_solve_whose_block_lost_its_multipliers_leaves_the_node_unbounded(monkeypatch):
    monkeypatch.setattr(pb._Block, "duals", lambda self: None)
    B, A = _problem(3, 6)
    nb = pb.bound(ExactData(B, A), 2, [], [], list(range(6)))
    assert nb.lower is None


@pytest.mark.parametrize("budget, holds", [(10.0, True), (2.5, False)])
def test_a_budget_on_a_node_with_nothing_free(budget, holds):
    # Every unit decided: the budget either holds for the one design left or
    # rules it out, and no row in z is left to write.
    B, A = _problem(10, 5)
    data = ExactData(B, A)
    nb = pb.bound(data, 3, [0, 1, 2], [3, 4], [], costs=np.ones(5), budget=budget)
    if holds:
        assert _below(nb.lower, _fit(data, (0, 1, 2)))
    else:
        assert nb.lower is None
