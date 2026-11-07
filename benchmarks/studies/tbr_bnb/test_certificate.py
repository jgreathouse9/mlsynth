"""Every exact bound lies below the enumerated optimum of what it bounds, for the
solver's multipliers and for any others; and the slope cap lies above every
positive best-fit slope."""
from itertools import combinations

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower

from .brute import panel, piece_optimum, wide_panel
from .certificate import ExactData, gram_lower, hull_bound, joint_bound, shor_bound
from .hull import ControlBound
from .lifts import Lift

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _solve(lift, lo, hi, m, extra=()):
    lift.lo.value, lift.hi.value, lift.m.value = lo, hi, m
    P = cp.Problem(lift.prob.objective, lift.prob.constraints + list(extra))
    P.solve(solver=cp.CLARABEL)
    return P


def _duals(lift):
    return [c.dual_value for c in lift.prob.constraints[1:]]


# ---------------------------------------------------------------- smoke
def test_the_joint_bound_is_a_finite_rational_below_the_piece_optimum():
    y, X = panel(0)
    L = Lift(y, X, "full")
    _solve(L, 0.4, 0.9, 3)
    b, S, k = joint_bound(L, y, X, 0.4, 0.9, 3, duals=_duals(L))
    assert b is not None and np.isfinite(float(b))
    assert float(b) <= piece_optimum(y, X, 0.4, 0.9, 3)


# ---------------------------------------------------------------- unit
def test_the_exact_lagrangian_is_cvxpys_in_exact_arithmetic():
    y, X = panel(1)
    L = Lift(y, X, "full")
    _solve(L, 0.3, 0.8, 2)
    Z = L.prob.variables()[0]
    n = Z.shape[0]
    Lag = L.prob.objective.expr
    for c in L.prob.constraints[1:]:
        d = c.dual_value if type(c).__name__ == "Equality" else np.maximum(c.dual_value, 0)
        Lag = Lag + cp.sum(cp.multiply(d, c.expr))
    Z.value = np.zeros((n, n))
    G = Lag.grad[Z].toarray().reshape(n, n, order="F")
    Sc = (G + G.T) / 2 * L.scale ** 2
    _, Se, k = joint_bound(L, y, X, 0.3, 0.8, 2, duals=_duals(L))
    Sf = np.array(Se.tolist(), dtype=float)
    assert np.abs(Sf - Sc).max() <= 1e-12 * np.abs(Sc).max()
    assert float(k) == pytest.approx(float(Lag.value) * L.scale ** 2, rel=1e-12)


def test_the_certificate_is_within_rounding_of_the_solver_where_the_lift_is_tight():
    y, X = panel(2)
    L = Lift(y, X, "full")
    P = _solve(L, 0.4, 0.9, 3)
    b, _, _ = joint_bound(L, y, X, 0.4, 0.9, 3, duals=_duals(L))
    assert abs(float(b) - P.value * L.scale ** 2) <= 1e-4 * abs(float(b))


def test_the_no_good_cut_lifts_the_bound_past_the_best_and_not_past_the_second():
    y, X = panel(3)
    lo, hi, m = 0.3, 0.9, 3
    vals = {C: piece_optimum(y, X, lo, hi, m, include=C, exclude=set(range(6)) - set(C))
            for C in combinations(range(6), m)}
    best = min(vals, key=vals.get)
    second = min(v for C, v in vals.items() if C != best)
    L = Lift(y, X, "full")
    u = L.prob.variables()[0][1:7, 0]
    sign = np.array([1.0 if j in best else -1.0 for j in range(6)])
    ng = sign @ u <= m - 1
    _solve(L, lo, hi, m, [ng])
    b, _, _ = joint_bound(L, y, X, lo, hi, m, duals=_duals(L),
                          nogood=(set(best), float(np.ravel(ng.dual_value)[0])))
    assert float(b) <= second * (1 + 1e-9)
    assert float(b) > vals[best]


def test_the_shor_bound_is_below_the_piece_optimum():
    y, X = panel(4)
    D = ExactData(y, X)
    L = Lift(y, X, "shor")
    L.piece(0.4, 0.9, 3)
    b, _, _ = shor_bound(L, D, 0.4, 0.9, 3, L.scale)
    assert float(b) <= piece_optimum(y, X, 0.4, 0.9, 3)


def test_the_slope_cap_lies_above_every_positive_best_fit_slope():
    y, X = panel(5)
    cap = float(ExactData(y, X).slope_cap())
    for r in range(1, 7):
        for C in combinations(range(6), r):
            x = X[:, list(C)].sum(axis=1)
            assert (y @ x) / (x @ x) <= cap


def test_the_incumbents_exact_rss_is_its_float_rss_to_rounding():
    y, X = panel(6)
    C = {0, 2}
    x = X[:, sorted(C)].sum(axis=1)
    b = (y @ x) / (x @ x)
    assert float(ExactData(y, X).rss(C, b)) == pytest.approx(float(((y - b * x) ** 2).sum()), rel=1e-12)


# ---------------------------------------------------------------- properties
@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6), corrupt=st.booleans())
def test_the_joint_bound_never_exceeds_the_piece_optimum(seed, lo, width, m, corrupt):
    """Any multipliers certify: the solver's, and the same ones scrambled."""
    y, X = panel(seed)
    hi = lo * width if lo > 0 else width - 1.0
    L = Lift(y, X, "full")
    _solve(L, lo, hi, m)
    duals = _duals(L)
    if corrupt:
        rng = np.random.default_rng(seed)
        duals = [np.asarray(d) * rng.uniform(0, 3, np.shape(d)) + rng.normal(0, 0.01, np.shape(d))
                 for d in duals]
    b, _, _ = joint_bound(L, y, X, lo, hi, m, duals=duals)
    if b is not None:
        assert float(b) <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-12)


@SETTINGS
@given(seed=st.integers(0, 10_000), j=st.integers(0, 5), val=st.integers(0, 1), m=st.integers(1, 5))
def test_a_fixed_node_is_bounded_by_its_own_optimum(seed, j, val, m):
    y, X = panel(seed)
    lo, hi = 0.3, 0.9
    L = Lift(y, X, "full")
    uj = L.prob.variables()[0][1 + j, 0]
    fx = uj == val
    P = _solve(L, lo, hi, m, [fx])
    if P.status not in ("optimal", "optimal_inaccurate"):
        return
    b, _, _ = joint_bound(L, y, X, lo, hi, m, duals=_duals(L),
                          fixes=[(j, val, float(np.ravel(fx.dual_value)[0]))])
    truth = piece_optimum(y, X, lo, hi, m, include={j} if val else None, exclude=None if val else {j})
    if b is not None and np.isfinite(truth):
        assert float(b) <= truth * (1 + 1e-12)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0),
       m=st.integers(1, 6), shift=st.floats(-1.0, 1.0))
def test_the_hull_bound_holds_at_any_point(seed, lo, width, m, shift):
    """The first-order bound is valid whatever point it is read at."""
    y, X = panel(seed)
    hi = lo * width if lo > 0 else width - 1.0
    H = ControlBound(y, X)
    H.piece(lo, hi, m)
    w = H.point + shift * np.random.default_rng(seed).normal(size=6)
    assert float(hull_bound(ExactData(y, X), w, lo, hi, m)) <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-12)


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_the_slope_cap_holds_over_panels(seed):
    y, X = panel(seed, p=5)
    cap = ExactData(y, X).slope_cap()
    for r in range(1, 6):
        for C in combinations(range(5), r):
            x = X[:, list(C)].sum(axis=1)
            assert (y @ x) / (x @ x) <= float(cap)


# ---------------------------------------------------------------- edge and failure
def test_a_singular_gram_has_no_slope_cap():
    y, X = panel(7)
    X = np.column_stack([X, X[:, 0]])                       # a duplicated geo
    assert ExactData(y, X).slope_cap() is None


def test_one_candidate_control_is_bounded_by_its_own_fit():
    y, X = panel(8, p=1)
    L = Lift(y, X, "full")
    _solve(L, 0.0, 3.0, 1)
    b, _, _ = joint_bound(L, y, X, 0.0, 3.0, 1, duals=_duals(L))
    assert float(b) <= piece_optimum(y, X, 0.0, 3.0, 1) * (1 + 1e-12)


def test_a_hull_bound_on_an_empty_band_is_infinite_or_above():
    """m lo above p hi is impossible for m <= p, so the knapsack always fills."""
    y, X = panel(9)
    b = hull_bound(ExactData(y, X), np.zeros(6), 0.5, 0.6, 6)
    assert np.isfinite(float(b))


def test_no_eigenvalue_bound_means_no_certificate(monkeypatch):
    """When lambda_min cannot be bounded the certificates report None, never a number."""
    from . import certificate
    y, X = panel(10)
    monkeypatch.setattr(certificate, "lambda_min_lower", lambda D: None)
    L = Lift(y, X, "full"); _solve(L, 0.4, 0.9, 3)
    assert joint_bound(L, y, X, 0.4, 0.9, 3)[0] is None        # multipliers read from the lift
    S = Lift(y, X, "shor"); S.piece(0.4, 0.9, 3)
    assert shor_bound(S, ExactData(y, X), 0.4, 0.9, 3, S.scale)[0] is None


# ---------------------------------------------------------------- the trace bounds and the multipliers' signs
from .certificate import joint_trace, shor_trace  # noqa: E402


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0), m=st.integers(1, 6))
def test_the_trace_bounds_hold_at_every_design_and_are_attained(seed, lo, width, m):
    """A design's lifted matrix has trace at most the bound, with equality at b = hi."""
    from fractions import Fraction
    hi = lo * width if lo > 0 else width - 1.0
    rng = np.random.default_rng(seed)
    u = np.zeros(6); u[rng.choice(6, m, replace=False)] = 1
    for b in (rng.uniform(lo, hi), hi):
        zj = np.r_[1, u, b * u]; zs = np.r_[1, b * u]
        assert zj @ zj <= float(joint_trace(m, Fraction(hi))) * (1 + 1e-12)
        assert zs @ zs <= float(shor_trace(m, Fraction(hi))) * (1 + 1e-12)
    assert zj @ zj == pytest.approx(float(joint_trace(m, Fraction(hi))), rel=1e-12)
    assert zs @ zs == pytest.approx(float(shor_trace(m, Fraction(hi))), rel=1e-12)


def test_a_negative_inequality_multiplier_is_ignored():
    """Weak duality admits only nonnegative multipliers on inequalities: a negative
    one is clipped to zero, so the certificate is the one with that multiplier at zero."""
    y, X = panel(11)
    L = Lift(y, X, "full"); _solve(L, 0.4, 0.9, 3)
    duals = _duals(L)
    zero = [np.array(d, dtype=float) for d in duals]
    zero[8][1, 2] = 0.0                                      # a product family's multiplier
    bad = [np.array(d, dtype=float) for d in zero]
    bad[8][1, 2] = -50.0
    assert joint_bound(L, y, X, 0.4, 0.9, 3, duals=bad)[0] == joint_bound(L, y, X, 0.4, 0.9, 3, duals=zero)[0]
    S = Lift(y, X, "shor"); S.piece(0.4, 0.9, 3)
    D = ExactData(y, X); sd = [c.dual_value for c in S.prob.constraints[1:]]
    szero = [np.array(d, dtype=float) for d in sd]
    szero[-1][0, 1] = 0.0                                    # W >= 0
    sbad = [np.array(d, dtype=float) for d in szero]
    sbad[-1][0, 1] = -50.0
    assert shor_bound(S, D, 0.4, 0.9, 3, S.scale, duals=sbad)[0] == shor_bound(S, D, 0.4, 0.9, 3, S.scale, duals=szero)[0]


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), width=st.floats(1.05, 4.0), m=st.integers(1, 6))
def test_the_shor_bound_never_exceeds_the_piece_optimum_for_any_multipliers(seed, lo, width, m):
    y, X = panel(seed)
    hi = lo * width if lo > 0 else width - 1.0
    S = Lift(y, X, "shor"); S.piece(lo, hi, m)
    rng = np.random.default_rng(seed)
    duals = [np.asarray(c.dual_value) * rng.uniform(0, 3, np.shape(c.dual_value))
             + rng.normal(0, 0.01, np.shape(c.dual_value)) for c in S.prob.constraints[1:]]
    b = shor_bound(S, ExactData(y, X), lo, hi, m, S.scale, duals=duals)[0]
    if b is not None:
        assert float(b) <= piece_optimum(y, X, lo, hi, m) * (1 + 1e-12)


def test_a_binding_fixing_is_certified_to_the_solvers_value():
    """Fixing in a geo outside the optimum binds; the certificate then matches the
    solver's value to its accuracy, which a wrong constant on the fixing would not."""
    y, X = panel(12)
    lo, hi, m = 0.3, 0.9, 3
    vals = {j: piece_optimum(y, X, lo, hi, m, include={j}) for j in range(6)}
    j = max(vals, key=vals.get)                              # the most expensive geo to force in
    L = Lift(y, X, "full")
    fx = L.prob.variables()[0][1 + j, 0] == 1
    P = _solve(L, lo, hi, m, [fx])
    d = float(np.ravel(fx.dual_value)[0])
    b, _, _ = joint_bound(L, y, X, lo, hi, m, duals=_duals(L), fixes=[(j, 1, d)])
    assert abs(d) * L.scale ** 2 > 1e-3 * abs(float(b))      # the fixing binds
    assert float(b) == pytest.approx(P.value * L.scale ** 2, rel=1e-5)


# ---------------------------------------------------------------- the Gram floor


def _subsets_min(G):
    """min u'Gu over every nonzero 0/1 vector u, by enumeration (exact)."""
    p = G.shape[0]
    return min(sum(G[i, j] for i in C for j in C)
               for r in range(1, p + 1) for C in combinations(range(p), r))


def _wide(seed, n=5, p=7):
    """A treatment series and p positive candidate controls over n < p periods."""
    M = wide_panel(seed, n=n, p=p + 1)
    return M[:, 0], M[:, 1:]


def test_smoke_gram_lower_of_a_nonnegative_gram_is_its_smallest_diagonal():
    y, X = _wide(0)
    G = ExactData(y, X).G
    assert all(g >= 0 for g in G.ravel())
    assert gram_lower(G) == min(G[j, j] for j in range(G.shape[0]))


def test_a_singular_nonnegative_gram_has_a_slope_cap_above_every_slope():
    """Seven controls over five periods: lambda_min(G) is 0, and the cap comes from
    the diagonal instead, above the best-fit slope of every control group."""
    y, X = _wide(1)
    D = ExactData(y, X)
    lam = lambda_min_lower(_from_fractions(D.G.tolist()))
    assert lam is None or lam <= 0                          # no eigenvalue cap
    cap = D.slope_cap()
    assert cap is not None
    for r in range(1, 8):
        for C in combinations(range(7), r):
            x = X[:, list(C)].sum(axis=1)
            assert (y @ x) / (x @ x) <= float(cap)


def test_a_gram_with_a_negative_entry_takes_the_eigenvalue_bound():
    y, X = panel(3)
    G = ExactData(y, X).G
    assert any(g < 0 for g in G.ravel())
    assert gram_lower(G) == lambda_min_lower(_from_fractions(G.tolist()))


def test_a_negative_entry_can_put_a_design_below_the_smallest_diagonal():
    """Two nearly opposite controls sum to a short series, so u'Gu over the pair is
    far below either diagonal entry; the floor stays below it."""
    rng = np.random.default_rng(4)
    a, b, c = rng.normal(size=(3, 6))
    X = np.column_stack([a, -a + 0.01 * b, c])
    G = ExactData(rng.normal(size=6), X).G
    assert _subsets_min(G) < min(G[j, j] for j in range(3)) / 100
    floor = gram_lower(G)
    assert floor is not None and 0 < floor <= _subsets_min(G)


def test_a_singular_gram_with_a_negative_entry_has_no_floor():
    y, X = panel(5)
    X = np.column_stack([X, -X[:, 0]])                       # a geo and its negative
    assert gram_lower(ExactData(y, X).G) is None
    assert ExactData(y, X).slope_cap() is None


def test_an_empty_gram_has_no_floor():
    assert gram_lower(np.zeros((0, 0), dtype=object)) is None


def test_a_zero_control_has_no_floor():
    """A control that is zero in every period makes u'Gu zero at that control."""
    y, X = _wide(2)
    X = X.copy(); X[:, 3] = 0.0
    assert gram_lower(ExactData(y, X).G) is None


@SETTINGS
@given(seed=st.integers(0, 10_000), wide=st.booleans(), n=st.integers(3, 8), p=st.integers(1, 6))
def test_property_the_gram_floor_is_below_every_control_group(seed, wide, n, p):
    """Over mixed-sign and positive panels, with fewer or more controls than periods,
    the floor is at most u'Gu for every control group, and never below the eigenvalue
    bound when that is positive."""
    y, X = _wide(seed, n=n, p=p) if wide else panel(seed, n=n, p=p)
    G = ExactData(y, X).G
    floor = gram_lower(G)
    if floor is not None:
        assert 0 < floor <= _subsets_min(G)
    lam = lambda_min_lower(_from_fractions(G.tolist()))
    if lam is not None and lam > 0:
        assert floor is not None and floor >= lam
