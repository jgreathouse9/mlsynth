"""The dual oracle: its maps are each other's adjoints, it is the exact certificate
in doubles, and its supergradient is one."""
import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .brute import panel
from .certificate import joint_bound
from .lifts import Lift
from .oracle import Oracle

SETTINGS = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _layout_ok(O, p):
    assert O.dim == 1 + 3 * p + 1 + p + p * p + 16 * p * p + (1 if O.C is not None else 0) + len(O.fixes)


def test_smoke_theta_is_finite_and_its_supergradient_has_the_layout():
    y, X = panel(0)
    O = Oracle(y, X, 0.4, 0.9, 3, C={0, 1}, fixes=[(2, 0)])
    _layout_ok(O, 6)
    th, g, lam, Q = O.theta(np.zeros(O.dim))
    assert np.isfinite(th) and g.shape == (O.dim,)


def test_the_oracle_is_the_exact_certificate_in_doubles():
    y, X = panel(1)
    lo, hi, m, C = 0.4, 0.9, 3, {0, 1, 2}
    L = Lift(y, X, "full")
    L.lo.value, L.hi.value, L.m.value = lo, hi, m
    u = L.prob.variables()[0][1:7, 0]
    sign = np.array([1.0 if j in C else -1.0 for j in range(6)])
    ng = sign @ u <= len(C) - 1
    fx = u[4] == 0
    cp.Problem(L.prob.objective, L.prob.constraints + [ng, fx]).solve(solver=cp.CLARABEL)
    O = Oracle(y, X, lo, hi, m, C=C, fixes=[(4, 0)])
    yv = O.duals_from_cvxpy(L.prob.constraints[1:], ng, [fx])
    yv[O.ineq] = np.maximum(yv[O.ineq], 0)
    th = O.theta(yv)[0] * O.scale ** 2
    b, _, _ = joint_bound(L, y, X, lo, hi, m, duals=[c.dual_value for c in L.prob.constraints[1:]],
                          fixes=[(4, 0, float(np.ravel(fx.dual_value)[0]))],
                          nogood=(C, float(np.ravel(ng.dual_value)[0])))
    assert th == pytest.approx(float(b), rel=1e-9)


@SETTINGS
@given(seed=st.integers(0, 10_000), lo=st.floats(0.0, 1.0), m=st.integers(1, 6))
def test_the_forward_map_is_the_adjoints_transpose(seed, lo, m):
    """<S(y), Z> + k(y) = obj(Z) + y'expr(Z) for every y and every symmetric Z."""
    y, X = panel(seed)
    O = Oracle(y, X, lo, lo + 0.7, m, C={0}, fixes=[(1, 1)])
    rng = np.random.default_rng(seed)
    yv = rng.normal(size=O.dim)
    A = rng.normal(size=(O.n, O.n)); Z = A + A.T
    S, k = O.S_k(yv)
    obj = O.yy - 2 * O.c @ Z[0, 7:] + np.sum(O.G * Z[7:, 7:])
    lhs, rhs = np.sum(S * Z) + k, obj + yv @ O.expr(Z)
    assert abs(lhs - rhs) <= 1e-9 * (1 + abs(lhs))


@SETTINGS
@given(seed=st.integers(0, 10_000), m=st.integers(1, 6))
def test_the_supergradient_is_one(seed, m):
    """theta(y + d) <= theta(y) + g'd for the concave theta, any sign-correct step."""
    y, X = panel(seed)
    O = Oracle(y, X, 0.3, 1.0, m)
    rng = np.random.default_rng(seed)
    yv = np.abs(rng.normal(size=O.dim)) * 1e-2
    th, g, _, _ = O.theta(yv)
    for _ in range(20):
        d = rng.normal(size=O.dim) * 1e-3
        d[O.ineq] = np.abs(d[O.ineq])
        assert O.theta(yv + d)[0] <= th + g @ d + 1e-10 * (1 + abs(th))


def test_theta_at_any_sign_correct_point_is_below_the_lifts_value():
    y, X = panel(2)
    L = Lift(y, X, "full")
    v = L.piece(0.4, 0.9, 3) / L.scale ** 2
    O = Oracle(y, X, 0.4, 0.9, 3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        yv = rng.normal(size=O.dim); yv[O.ineq] = np.abs(yv[O.ineq])
        assert O.theta(yv)[0] <= v * (1 + 1e-6)


def test_a_semidefinite_lagrangian_bounds_by_its_constant():
    """With S >= 0 the bound is k itself and Z* = 0: a heavy weight on Z_00 = 1 and
    on diag(U) = u makes S positive definite."""
    y, X = panel(3)
    O = Oracle(y, X, 0.4, 0.9, 3)
    yv = np.zeros(O.dim); yv[O.sl["z00"]] = 1e6; yv[O.sl["dU"]] = 1.0
    th, g, lam, _ = O.theta(yv)
    assert lam[0] >= 0
    S, k = O.S_k(yv)
    assert th == k and np.allclose(g, O.expr(np.zeros((O.n, O.n))))
