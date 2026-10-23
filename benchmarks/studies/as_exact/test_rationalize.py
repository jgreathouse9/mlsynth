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
