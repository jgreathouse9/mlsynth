# ============================================================
# tests/test_opt_helpers.py
# ============================================================

import pytest
import numpy as np
import cvxpy as cp
from mlsynth.utils.opthelpers import OptHelpers  # Adjust import as needed

@pytest.fixture
def small_vars():
    np.random.seed(0)
    J = 5
    w = cp.Variable(J)
    b0 = cp.Variable()
    y = np.random.randn(10)
    X = np.random.randn(10, J)
    return w, b0, X, y

@pytest.fixture
def small_problem():
    np.random.seed(1)
    J = 4
    T = 5
    w = cp.Variable(J)
    b0 = cp.Variable()
    X = np.random.randn(T, J)
    w0 = np.ones(J) / J
    y = X @ w0 + 0.01 * np.random.randn(T)
    return w, b0, X, y

# ======================
# Loss helper tests
# ======================

def test_squared_loss_basic(small_vars):
    w, b0, X, y = small_vars
    loss = OptHelpers.squared_loss(y, X, w)
    assert isinstance(loss, cp.Expression)
    # Scaled vs unscaled
    loss_unscaled = OptHelpers.squared_loss(y, X, w, scale=False)
    assert isinstance(loss_unscaled, cp.Expression)

def test_squared_loss_with_intercept(small_vars):
    w, b0, X, y = small_vars
    loss = OptHelpers.squared_loss(y, X, w, b0=b0)
    assert isinstance(loss, cp.Expression)

def test_squared_loss_numeric(small_problem):
    w, b0, X, y = small_problem
    w.value = np.zeros(X.shape[1])
    b0.value = 0.0
    loss_expr = OptHelpers.squared_loss(y, X, w, b0=b0)
    val = loss_expr.value
    assert np.isfinite(val)
    # Check scaled vs unscaled relation
    T = y.shape[0]
    scaled = OptHelpers.squared_loss(y, X, w, b0=b0, scale=True).value
    unscaled = OptHelpers.squared_loss(y, X, w, b0=b0, scale=False).value
    assert np.isclose(unscaled, T * scaled)

# ======================
# Penalty helpers
# ======================

@pytest.mark.parametrize("alpha, second_norm", [
    (0.5, "l2"), (0.5, "inf"), (1.0, "l2"), (0.0, "l2")
])
def test_elastic_net_penalty_types(small_vars, alpha, second_norm):
    w, _, _, _ = small_vars
    pen = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=alpha, second_norm=second_norm)
    assert isinstance(pen, cp.Expression)
    pen_zero = OptHelpers.elastic_net_penalty(w, lam=0.0)
    assert pen_zero == 0.0

def test_elastic_net_penalty_invalid_norm(small_vars):
    w, _, _, _ = small_vars
    with pytest.raises(ValueError):
        OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="foo")

def test_l2_only_penalty(small_vars):
    w, _, _, _ = small_vars
    pen = OptHelpers.l2_only_penalty(w)
    assert isinstance(pen, cp.Expression)

def test_entropy_penalty(small_vars):
    w, _, _, _ = small_vars
    pen = OptHelpers.entropy_penalty(w)
    assert isinstance(pen, cp.Expression)

def test_el_penalty(small_vars):
    w, _, _, _ = small_vars
    pen = OptHelpers.el_penalty(w)
    assert isinstance(pen, cp.Expression)

def test_elastic_net_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0])
    pen = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="l2")
    assert np.isfinite(pen.value) and pen.value > 0
    pen_inf = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="inf")
    assert np.isfinite(pen_inf.value) and pen_inf.value > 0

def test_entropy_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0]) / w.shape[0]  # valid simplex
    pen = OptHelpers.entropy_penalty(w)
    val = pen.value
    assert np.isfinite(val)
    assert val <= 0  # negative entropy

def test_entropy_penalty_near_boundary(small_problem):
    w, _, _, _ = small_problem
    w.value = np.array([0.99, 0.0025, 0.0025, 0.005])  # Sum=1, small positives
    pen = OptHelpers.entropy_penalty(w)
    assert np.isfinite(pen.value) and pen.value <= 0

def test_el_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0]) / w.shape[0]  # positive weights
    pen = OptHelpers.el_penalty(w)
    val = pen.value
    assert np.isfinite(val) and val >= 0

def test_el_penalty_near_boundary(small_problem):
    w, _, _, _ = small_problem
    w.value = np.array([0.99, 0.0025, 0.0025, 0.005])  # Sum=1, small positives
    pen = OptHelpers.el_penalty(w)
    assert np.isfinite(pen.value) and pen.value >= 0

# ======================
# Constraint helpers
# ======================

def test_simplex_constraints(small_vars):
    w, _, _, _ = small_vars
    cons = OptHelpers.simplex_constraints(w)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_affine_constraints(small_vars):
    w, _, _, _ = small_vars
    cons = OptHelpers.affine_constraints(w)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_nonneg_constraints(small_vars):
    w, _, _, _ = small_vars
    cons = OptHelpers.nonneg_constraints(w)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_unit_constraints(small_vars):
    w, _, _, _ = small_vars
    cons = OptHelpers.unit_constraints(w)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

# ======================
# Relaxed balance constraints
# ======================

@pytest.mark.parametrize("obj_type", ["l2", "entropy", "el"])
def test_relaxed_balance_constraints(small_vars, obj_type):
    w, b0, X, y = small_vars
    residual = y - X @ w
    cons, gam = OptHelpers.relaxed_balance_constraints(X, residual, tau=0.1, objective_type=obj_type)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)
    assert isinstance(gam, cp.Variable)

def test_relaxed_balance_constraints_invalid_type(small_vars):
    w, _, X, y = small_vars
    residual = y - X @ w
    with pytest.raises(ValueError):
        OptHelpers.relaxed_balance_constraints(X, residual, tau=0.1, objective_type="foo")

# ======================
# Full constraint assembly
# ======================

@pytest.mark.parametrize("ctype", ["unconstrained", "simplex", "affine", "nonneg", "unit"])
def test_build_constraints_basic(small_vars, ctype):
    w, b0, X, y = small_vars
    cons = OptHelpers.build_constraints(w, constraint_type=ctype, X=X, y=y, b0=b0, tau=0.1)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

@pytest.mark.parametrize("ctype", ["unconstrained", "simplex", "affine", "nonneg", "unit"])
@pytest.mark.parametrize("tau", [None, 0.1])
def test_build_constraints_with_tau(small_vars, ctype, tau):
    w, b0, X, y = small_vars
    cons = OptHelpers.build_constraints(w, constraint_type=ctype, X=X, y=y, b0=b0, tau=tau)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)
    # If tau is None, no relaxed constraints added
    if tau is None:
        assert len(cons) <= 2  # Depending on ctype

def test_build_constraints_invalid_type(small_vars):
    w, b0, X, y = small_vars
    with pytest.raises(ValueError):
        OptHelpers.build_constraints(w, constraint_type="invalid", X=X, y=y)

# =========================
# Constraint evaluation helpers
# =========================

@pytest.mark.parametrize(
    "constraint_func, weight_generator, expect_violation, tau",
    [
        # Simplex: valid weights
        (OptHelpers.simplex_constraints, lambda n: np.ones(n)/n, False, None),
        # Affine: valid weights
        (OptHelpers.affine_constraints, lambda n: np.ones(n)/n, False, None),
        # Nonneg: valid weights
        (OptHelpers.nonneg_constraints, lambda n: np.ones(n), False, None),
        # Unit: valid weights
        (OptHelpers.unit_constraints, lambda n: np.linspace(0, 1, n), False, None),
        # Unit: invalid weights
        (OptHelpers.unit_constraints, lambda n: np.array([0.5, 1.5, -0.1, 0.8]), True, None),
        # Relaxed balance: valid weights, large tau
        (OptHelpers.relaxed_balance_constraints, lambda n: np.ones(n)/n, False, 10.0),
        # Relaxed balance: valid weights, small tau
        (OptHelpers.relaxed_balance_constraints, lambda n: np.ones(n)/n, False, 0.1),
    ]
)
def test_constraints_numeric_full(small_problem, constraint_func, weight_generator, expect_violation, tau):
    """
    Unified test for all numeric constraint types.
    Handles:
        - simplex, affine, nonneg, unit constraints
        - relaxed_balance_constraints (with residual + tau)
        - valid and intentionally invalid weights
    """
    w, b0, X, y = small_problem
    n = w.shape[0]
    w.value = weight_generator(n)
    
    # Special handling for relaxed_balance_constraints
    if constraint_func == OptHelpers.relaxed_balance_constraints:
        b0.value = 0
        residual = y - X @ w.value - b0.value
        cons, gam = constraint_func(X, residual, tau=tau)
        gam.value = 0.0
    else:
        cons = constraint_func(w)

    # Evaluate constraints
    for c in cons:
        if c.value is None:
            continue
        v = c.violation()
        if isinstance(v, np.ndarray):
            if expect_violation:
                with pytest.raises(AssertionError):
                    assert np.all(v <= 0)
            else:
                assert np.all(v <= 1e-8)
        else:
            if expect_violation:
                with pytest.raises(AssertionError):
                    assert v <= 0
            else:
                assert v <= 1e-8


# =========================
# Integration: Full optimization tests
# =========================

@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("constraint_type", ["simplex", "affine", "nonneg", "unit"])
def test_full_optimization_loss_constraints(small_problem, intercept, constraint_type):
    w, b0, X, y = small_problem
    b0_use = b0 if intercept else None
    loss = OptHelpers.squared_loss(y, X, w, b0=b0_use)
    cons = OptHelpers.build_constraints(w, constraint_type=constraint_type)
    prob = cp.Problem(cp.Minimize(loss), cons)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    if constraint_type in ["simplex", "affine"]:
        assert np.isclose(np.sum(w.value), 1, atol=1e-6)
    if constraint_type in ["simplex", "nonneg", "unit"]:
        assert np.all(w.value >= -1e-8)
    if constraint_type == "unit":
        assert np.all(w.value <= 1 + 1e-8)
    if intercept:
        assert np.isfinite(b0.value)

@pytest.mark.parametrize("penalty_type", ["elastic_l2", "elastic_inf", "l2_only", "entropy", "el"])
def test_full_optimization_with_penalty(small_problem, penalty_type):
    w, _, X, y = small_problem
    loss = OptHelpers.squared_loss(y, X, w)
    if penalty_type == "elastic_l2":
        pen = OptHelpers.elastic_net_penalty(w, lam=0.1, alpha=0.5, second_norm="l2")
    elif penalty_type == "elastic_inf":
        pen = OptHelpers.elastic_net_penalty(w, lam=0.1, alpha=0.5, second_norm="inf")
    elif penalty_type == "l2_only":
        pen = OptHelpers.l2_only_penalty(w)
    elif penalty_type == "entropy":
        pen = OptHelpers.entropy_penalty(w)
    elif penalty_type == "el":
        pen = OptHelpers.el_penalty(w)
    obj = loss + pen
    cons = OptHelpers.simplex_constraints(w)  # Suitable for all, esp. entropy/el
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(np.sum(w.value), 1, atol=1e-6)
    assert np.all(w.value >= -1e-8)

@pytest.mark.parametrize("objective_type", ["l2", "entropy", "el"])
def test_full_optimization_with_relaxed_balance(small_problem, objective_type):
    w, b0, X, y = small_problem
    loss = OptHelpers.squared_loss(y, X, w, b0=b0)
    cons = OptHelpers.build_constraints(
        w, constraint_type="simplex", X=X, y=y, b0=b0, tau=0.1, objective_type=objective_type
    )
    prob = cp.Problem(cp.Minimize(loss), cons)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(np.sum(w.value), 1, atol=1e-6)
    assert np.all(w.value >= -1e-8)
