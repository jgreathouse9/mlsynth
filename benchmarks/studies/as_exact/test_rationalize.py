"""Predictor weights V that rationalise donor weights, tested on their own.

In the diagonal class the predictors V can weight (its support) and the inactive
donors' inequalities that hold with equality for every rationaliser (the implicit
equalities, Schrijver, Theory of Linear and Integer Programming, 1986, sec. 8.1)
decide the program the barrier method solves. ``_diag_classes`` finds both with one
LP; the oracle asks one LP per predictor and per inequality, as the authors' code
does. test_exact.py compares the resulting V with the authors' on the applications.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.optimize import linprog

from . import exact, rationalize
from .capture import random_instance
from .test_exact import CASES, _inputs

TOL = rationalize.TOL


def _one_lp_per_row(Aeq, Sm):
    """The predictors and inequalities that can be positive, one LP each, on the slice 1'd = 1."""
    K = Aeq.shape[1]
    A_eq = np.vstack([Aeq, np.ones((1, K))])
    b_eq = np.r_[np.zeros(len(Aeq)), 1.0]

    def most(c):
        res = linprog(-c, A_ub=-Sm if len(Sm) else None, b_ub=np.zeros(len(Sm)) if len(Sm) else None,
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return -res.fun if res.status == 0 else None
    support = [k for k in range(K) if (most(np.eye(K)[k]) or 0.0) > TOL]
    strict = [i for i in range(len(Sm)) if (most(Sm[i]) or 0.0) > TOL]
    return support, strict


def _rationalisable(seed, K, J):
    """Donor weights from the diagonal search on the first exterior design from `seed` on."""
    while (inst := random_instance(seed, K, J)) is None:
        seed += 1
    X1, X0, Z1, Z0 = inst
    return X1, X0, exact.solve_diag(X1, X0, Z1, Z0)[1]


def _check_start(Aeq, Sm, support, strict, x0):
    """x0 is a strictly feasible start: positive on the support and on the strict rows, and on every equality."""
    implicit = [i for i in range(len(Sm)) if i not in strict]
    d = np.zeros(Aeq.shape[1])
    d[support] = x0
    assert x0.min() > 0.0 and abs(d.sum() - 1.0) <= 1e-9
    assert np.abs(Aeq @ d).max(initial=0.0) <= 1e-7
    assert (Sm[strict] @ d).min(initial=1.0) > 0.0
    assert np.abs(Sm[implicit] @ d).max(initial=0.0) <= 1e-7


# --------------------------------------------------------------------------- unit: the applications
@pytest.mark.parametrize("name", ["prop99", "germany"])
def test_one_lp_classifies_the_application_like_one_lp_per_row(name):
    case = CASES[name]
    X1, X0, _, _ = _inputs(case)
    Aeq, Sm = rationalize._diag_system(X1, X0, np.array(case["diag"]["w"]))
    support, strict, x0 = rationalize._diag_classes(Aeq, Sm)
    assert (support, strict) == _one_lp_per_row(Aeq, Sm)
    _check_start(Aeq, Sm, support, strict, x0)


# --------------------------------------------------------------------------- property
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(3, 5), extra=st.integers(3, 9))
def test_property_one_lp_classifies_like_one_lp_per_row(seed, K, extra):
    X1, X0, w = _rationalisable(seed, K, K + extra)
    Aeq, Sm = rationalize._diag_system(X1, X0, w)
    support, strict, x0 = rationalize._diag_classes(Aeq, Sm)
    assert (support, strict) == _one_lp_per_row(Aeq, Sm)
    _check_start(Aeq, Sm, support, strict, x0)


# --------------------------------------------------------------------------- failure
def test_weights_no_diagonal_v_rationalises_are_refused():
    """Proposition 99's donor 0 alone is a vertex no positive diagonal V attains."""
    X1, X0, _, _ = _inputs(CASES["prop99"])
    w = np.zeros(X0.shape[1])
    w[0] = 1.0
    Aeq, Sm = rationalize._diag_system(X1, X0, w)
    assert _one_lp_per_row(Aeq, Sm) == ([], [])          # the oracle agrees no rationaliser exists
    with pytest.raises(ValueError, match="no diagonal V rationalises w"):
        rationalize.diag_rationalizers(X1, X0, w)


# --------------------------------------------------------------------------- Newton's method
def test_the_line_search_costs_one_function_value_per_trial():
    """Derivatives are evaluated once per iterate, the last only to see the decrement has converged;
    every trial step of the line search asks for the objective alone."""
    calls = {"derivatives": 0, "objective": 0}

    def vgh(x):
        calls["derivatives"] += 1
        return -np.log(x).sum(), -1.0 / x, np.diag(1.0 / x ** 2)

    def fun(x):
        calls["objective"] += 1
        return -np.log(x).sum() if np.all(x > 0) else np.inf

    x = rationalize._newton_eq(np.array([0.9, 0.08, 0.02]), vgh, np.ones((1, 3)), np.ones(1),
                               lambda x: bool(np.all(x > 0)), fun=fun)
    assert np.allclose(x, 1.0 / 3.0, atol=1e-8)
    assert 0 < calls["derivatives"] <= calls["objective"] + 1


def test_a_shared_null_space_gives_the_same_answer():
    def vgh(x):
        return -np.log(x).sum(), -1.0 / x, np.diag(1.0 / x ** 2)

    A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0]])
    x0 = np.array([0.4, 0.4, 0.15, 0.05])
    N = rationalize._null(A)
    a = rationalize._newton_eq(x0, vgh, A, A @ x0, lambda x: bool(np.all(x > 0)))
    b = rationalize._newton_eq(x0, vgh, A, A @ x0, lambda x: bool(np.all(x > 0)), N=N)
    assert np.allclose(a, b, atol=1e-12) and np.allclose(A @ a, A @ x0, atol=1e-12)


# --------------------------------------------------------------------------- the interior case
# Inside the hull r = 0, and the exterior conditions say nothing: V rationalises w_A when the cell
# A is its selection, <Q_j^A, V> > 0 for every donor outside A (paper, eq. 18). The analytic centre
# and the log-det maximiser are taken over that set, on the trace-one slice. The centres are checked
# by their optimality conditions and by the selection they make (test_interior._selection), and all
# four against cvxpy solving the same programs. cvxpy's exponential-cone answers meet those
# conditions only to about 1e-3 here, where Newton's method meets them to 1e-6 or better, so they
# are compared to 1e-4.
import cvxpy as cp                                                     # noqa: E402

from . import interior                                                 # noqa: E402
from .test_interior import CASES as INTERIOR, _selection               # noqa: E402

INTERIOR_DESIGNS = ["random_interior_K3_J12_seed4007", "random_interior_K4_J16_seed4024",
                    "random_interior_K5_J20_seed4085", "prop99_placebo_Illinois"]


def _interior_answer(name, solve):
    X1, X0, Z1, Z0 = _inputs(INTERIOR[name])
    w = solve(X1, X0, Z1, Z0)[1]
    C1, C0 = interior._standardised(X1, X0)
    cell = interior.Cell(np.flatnonzero(w), C1, C0)
    return X1, X0, w, cell, C1, C0


def _centre_residual_diag(R, d):
    """Spread of the centre's gradient 1/d + R'(1 / R d), which is a multiple of 1 at the centre."""
    g = 1.0 / d + R.T @ (1.0 / (R @ d))
    return float(np.ptp(g) / np.abs(g).mean())


def _centre_residual_pd(Q, V):
    """V^-1 + sum_j Q_j / <Q_j, V>, a multiple of I at the centre on the trace-one slice."""
    M = np.linalg.inv(V) + np.einsum("jkl,j->kl", Q, 1.0 / np.einsum("jkl,kl->j", Q, V))
    nu = np.trace(M @ V) / np.trace(V)
    return float(np.abs(M - nu * np.eye(len(V))).max() / abs(nu))


def _cvx_diag(R, centre):
    K = R.shape[1]
    d = cp.Variable(K)
    obj = cp.sum(cp.log(d)) + (cp.sum(cp.log(R @ d)) if centre else 0)
    cons = [cp.sum(d) == 1] + ([] if centre else [R @ d >= 0])
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return d.value


def _cvx_pd(Q, centre):
    K = Q.shape[1]
    V = cp.Variable((K, K), PSD=True)
    slacks = cp.hstack([cp.trace(Qj @ V) for Qj in Q])
    obj = cp.log_det(V) + (cp.sum(cp.log(slacks)) if centre else 0)
    cons = [cp.trace(V) == 1] + ([] if centre else [slacks >= 0])
    cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.CLARABEL)
    return V.value


@pytest.mark.parametrize("name", INTERIOR_DESIGNS)
def test_the_interior_diagonal_centre_selects_the_cell(name):
    X1, X0, w, cell, C1, C0 = _interior_answer(name, interior.solve_interior_diag)
    out = rationalize.diag_rationalizers(X1, X0, w)
    d = out["center"]
    assert d.min() > 0.0 and abs(d.sum() - 1.0) <= 1e-12 and (cell.R @ d).min() > 0.0
    assert np.allclose(_selection(C1, C0, np.diag(d)), w, atol=1e-8)
    assert _centre_residual_diag(cell.R, d) <= 1e-5
    assert np.allclose(d, _cvx_diag(cell.R, centre=True), atol=1e-4)
    dl = out["logdet"]
    assert dl.min() >= 0.0 and abs(dl.sum() - 1.0) <= 1e-9 and (cell.R @ dl).min() >= -1e-9
    assert np.allclose(dl, _cvx_diag(cell.R, centre=False), atol=1e-4)
    assert out["support"] == list(range(X0.shape[0])) and out["forced_zero"] == [] and out["n_implicit"] == 0


@pytest.mark.parametrize("name", INTERIOR_DESIGNS)
def test_the_interior_pd_centre_selects_the_cell(name):
    X1, X0, w, cell, C1, C0 = _interior_answer(name, interior.solve_interior_pd)
    Q = interior.q_matrices(cell, C0)
    out = rationalize.pd_rationalizers(X1, X0, w, ranges=True)
    Vc, Vl = out["Vcenter"], out["Vstar"]
    assert np.linalg.eigvalsh(Vc).min() > 0.0 and abs(np.trace(Vc) - 1.0) <= 1e-9
    assert np.einsum("jkl,kl->j", Q, Vc).min() > 0.0
    assert np.allclose(_selection(C1, C0, Vc), w, atol=1e-8)
    assert _centre_residual_pd(Q, Vc) <= 1e-5
    assert np.allclose(Vc, _cvx_pd(Q, centre=True), atol=1e-4)
    assert abs(np.trace(Vl) - 1.0) <= 1e-9 and np.einsum("jkl,kl->j", Q, Vl).min() >= -1e-9
    assert np.allclose(Vl, _cvx_pd(Q, centre=False), atol=1e-4)
    assert np.allclose(out["center_diag"], np.diag(Vc)) and np.all(out["lo"] <= out["center_diag"] + 1e-9)
    assert np.all(out["center_diag"] <= out["hi"] + 1e-9)


def test_interior_ranges_match_one_program_per_bound():
    X1, X0, w, cell, C1, C0 = _interior_answer("random_interior_K3_J12_seed4007", interior.solve_interior_pd)
    Q = interior.q_matrices(cell, C0)
    out = rationalize.pd_rationalizers(X1, X0, w, ranges=True)
    K = X0.shape[0]
    for k in range(K):
        for sense, got in ((cp.Minimize, out["lo"][k]), (cp.Maximize, out["hi"][k])):
            V = cp.Variable((K, K), PSD=True)
            prob = cp.Problem(sense(V[k, k]), [cp.trace(V) == 1] + [cp.trace(Qj @ V) >= 0 for Qj in Q])
            prob.solve(solver=cp.CLARABEL)
            assert abs(prob.value - got) <= 1e-4


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), K=st.integers(2, 4), extra=st.integers(1, 9))
def test_property_interior_centres_select_the_answer(seed, K, extra):
    from .test_interior import _design
    X1, X0, Z1, Z0 = _design(seed, K, K + 1 + extra)
    C1, C0 = interior._standardised(X1, X0)
    w = interior.solve_interior_diag(X1, X0, Z1, Z0)[1]
    assert np.allclose(_selection(C1, C0, np.diag(rationalize.diag_rationalizers(X1, X0, w)["center"])), w, atol=1e-8)
    w = interior.solve_interior_pd(X1, X0, Z1, Z0)[1]
    assert np.allclose(_selection(C1, C0, rationalize.pd_rationalizers(X1, X0, w)["Vcenter"]), w, atol=1e-8)


def test_interior_weights_no_diagonal_v_selects_are_refused():
    """The PD answer on this design is a cell no positive diagonal V selects (its loss is below the
    diagonal optimum), so no diagonal rationaliser exists."""
    case = INTERIOR["random_interior_K3_J12_seed4007"]
    assert case["pd"]["loss"] < case["diag"]["loss"]
    X1, X0, w, _, _, _ = _interior_answer("random_interior_K3_J12_seed4007", interior.solve_interior_pd)
    with pytest.raises(ValueError, match="no diagonal V selects"):
        rationalize.diag_rationalizers(X1, X0, w)


def test_interior_weights_that_are_not_a_cell_are_refused():
    """Inside the hull a rationalised w is the barycentric weights of K + 1 donors."""
    X1, X0, Z1, Z0 = _inputs(INTERIOR["random_interior_K3_J12_seed4007"])
    w = np.full(X0.shape[1], 1.0 / X0.shape[1])
    for fn in (rationalize.diag_rationalizers, rationalize.pd_rationalizers):
        with pytest.raises(ValueError, match="exact match on K \\+ 1 donors"):
            fn(X1, X0, w)


def test_interior_weights_no_positive_definite_v_selects_are_refused():
    """On this design the best exact match is not PD-attainable: its cell is not the selection under
    any positive definite V, so it has no rationaliser."""
    case = INTERIOR["random_interior_K2_J8_seed4003"]
    X1, X0, Z1, Z0 = _inputs(case)
    C1, C0 = interior._standardised(X1, X0)
    cells = interior.walk_admissible(C1, C0, {})
    loss = {A: float(np.sum((Z1 - Z0[:, list(A)] @ w) ** 2)) for A, w in cells.items()}
    A = min(loss, key=loss.get)
    assert loss[A] < case["pd"]["loss"]
    w = np.zeros(X0.shape[1])
    w[list(A)] = cells[A]
    with pytest.raises(ValueError, match="no positive definite V selects"):
        rationalize.pd_rationalizers(X1, X0, w)
