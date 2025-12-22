# ============================================================
# tests/test_opt_helpers.py
# ============================================================

import pytest
import numpy as np
import cvxpy as cp
from mlsynth.utils.opthelpers import OptHelpers


@pytest.fixture
def small_vars():
    np.random.seed(0)
    J = 5
    w = cp.Variable(J)
    b0 = cp.Variable()
    y = np.random.randn(10)
    X = np.random.randn(10, J)
    return w, b0, X, y


# ======================
# Loss helper tests
# ======================

def test_squared_loss_basic(small_vars):
    w, b0, X, y = small_vars
    loss = OptHelpers.squared_loss(y, X, w)
    assert isinstance(loss, cp.Expression)
    # scaled loss
    loss_unscaled = OptHelpers.squared_loss(y, X, w, scale=False)
    assert isinstance(loss_unscaled, cp.Expression)

def test_squared_loss_with_intercept(small_vars):
    w, b0, X, y = small_vars
    loss = OptHelpers.squared_loss(y, X, w, b0=b0)
    assert isinstance(loss, cp.Expression)


# ======================
# Penalty helpers
# ======================

def test_elastic_net_penalty_types(small_vars):
    w, _, _, _ = small_vars
    pen = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="l2")
    assert isinstance(pen, cp.Expression)
    pen_inf = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="linf")
    assert isinstance(pen_inf, cp.Expression)
    pen_l1_only = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=1.0)
    assert isinstance(pen_l1_only, cp.Expression)
    pen_zero = OptHelpers.elastic_net_penalty(w, lam=0.0)
    assert pen_zero == 0.0

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


# ======================
# Relaxed balance constraints
# ======================

def test_relaxed_balance_constraints(small_vars):
    w, b0, X, y = small_vars
    residual = y - X @ w
    cons, gam = OptHelpers.relaxed_balance_constraints(X, residual, tau=0.1)
    assert isinstance(cons, list)
    assert isinstance(gam, cp.Variable)


# ======================
# Full constraint assembly
# ======================

@pytest.mark.parametrize("ctype", ["unconstrained", "simplex", "affine", "nonneg"])
def test_build_constraints_basic(small_vars, ctype):
    w, b0, X, y = small_vars
    cons = OptHelpers.build_constraints(w, constraint_type=ctype, X=X, y=y, b0=b0, tau=0.1)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_build_constraints_invalid_type(small_vars):
    w, b0, X, y = small_vars
    with pytest.raises(ValueError):
        OptHelpers.build_constraints(w, constraint_type="invalid", X=X, y=y)





@pytest.fixture
def small_problem():
    np.random.seed(1)
    J = 4
    w = cp.Variable(J)
    b0 = cp.Variable()
    y = np.random.randn(5)
    X = np.random.randn(5, J)
    return w, b0, X, y

# =========================
# Numeric evaluation
# =========================

def test_squared_loss_numeric(small_problem):
    w, b0, X, y = small_problem
    w.value = np.zeros(X.shape[1])
    b0.value = 0.0
    loss_expr = OptHelpers.squared_loss(y, X, w, b0=b0)
    val = loss_expr.value
    assert np.isfinite(val)

def test_elastic_net_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0])
    pen = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="l2")
    assert np.isfinite(pen.value) and pen.value > 0
    pen_linf = OptHelpers.elastic_net_penalty(w, lam=1.0, alpha=0.5, second_norm="linf")
    assert np.isfinite(pen_linf.value) and pen_linf.value > 0

def test_entropy_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0]) / w.shape[0]  # valid simplex
    pen = OptHelpers.entropy_penalty(w)
    val = pen.value
    assert np.isfinite(val)
    # negative entropy is <= 0
    assert val <= 0

def test_el_penalty_numeric(small_problem):
    w, _, _, _ = small_problem
    w.value = np.ones(w.shape[0]) / w.shape[0]  # positive weights
    pen = OptHelpers.el_penalty(w)
    val = pen.value
    assert np.isfinite(val) and val >= 0

# =========================
# Constraint evaluation
# =========================

def check_constraints(cons):
    for c in cons:
        if c.value is not None:
            v = c.violation()
            if isinstance(v, np.ndarray):
                assert np.all(np.isclose(v, 0, atol=1e-8))
            else:
                assert np.isclose(v, 0, atol=1e-8)

def test_simplex_constraints_numeric(small_problem):
    w, _, _, _ = small_problem
    cons = OptHelpers.simplex_constraints(w)
    w.value = np.ones(w.shape[0]) / w.shape[0]
    check_constraints(cons)

def test_affine_constraints_numeric(small_problem):
    w, _, _, _ = small_problem
    cons = OptHelpers.affine_constraints(w)
    w.value = np.ones(w.shape[0]) / w.shape[0]
    check_constraints(cons)

def test_nonneg_constraints_numeric(small_problem):
    w, _, _, _ = small_problem
    cons = OptHelpers.nonneg_constraints(w)
    w.value = np.ones(w.shape[0])
    check_constraints(cons)



# =========================
# Unit constraints
# =========================

def test_unit_constraints_structure(small_problem):
    w, _, _, _ = small_problem
    cons = OptHelpers.unit_constraints(w)
    assert isinstance(cons, list)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_build_constraints_unit_type(small_problem):
    w, b0, X, y = small_problem
    cons = OptHelpers.build_constraints(w, constraint_type="unit")
    assert isinstance(cons, list)
    # All constraints should be CVXPY constraints
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)

def test_unit_constraints_numeric(small_problem):
    w, _, _, _ = small_problem
    cons = OptHelpers.unit_constraints(w)
    # Assign valid values within [0,1]
    w.value = np.linspace(0, 1, w.shape[0])
    # Check numeric violations
    for c in cons:
        v = c.violation()
        if isinstance(v, np.ndarray):
            assert np.all(np.isclose(v, 0, atol=1e-8))
        else:
            assert np.isclose(v, 0, atol=1e-8)

def test_build_constraints_unit_numeric(small_problem):
    w, b0, X, y = small_problem
    # Build only structural unit constraints (exclude tau for numeric check)
    cons = OptHelpers.build_constraints(w, constraint_type="unit")
    w.value = np.linspace(0, 1, w.shape[0])
    for c in cons:
        v = c.violation()
        if isinstance(v, np.ndarray):
            assert np.all(np.isclose(v, 0, atol=1e-8))
        else:
            assert np.isclose(v, 0, atol=1e-8)










# =========================
# Relaxed balance numeric
# =========================

def test_relaxed_balance_constraints_numeric(small_problem):
    w, b0, X, y = small_problem
    w.value = np.ones(X.shape[1])
    b0.value = 0
    residual = y - X @ w
    cons, gam = OptHelpers.relaxed_balance_constraints(X, residual, tau=0.1)
    gam.value = 0.0
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)
    assert isinstance(gam, cp.Variable)

# =========================
# Full constraint assembly numeric
# =========================

def test_build_constraints_numeric(small_problem):
    w, b0, X, y = small_problem
    w.value = np.ones(X.shape[1])
    b0.value = 0.0
    cons = OptHelpers.build_constraints(w, constraint_type="simplex", X=X, y=y, b0=b0, tau=0.1)
    assert all(isinstance(c, cp.constraints.constraint.Constraint) for c in cons)
