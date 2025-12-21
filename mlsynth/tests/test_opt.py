# ============================================================
# tests/opttest_legacy.py
# ============================================================

import numpy as np
import cvxpy as cp
import pytest
from mlsynth.utils.optutils import Opt2   # new implementation
from mlsynth.utils.estutils import Opt    # old implementation
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlsynth.utils.datautils import dataprep
from mlsynth.utils.crossval import RelaxationCV
from mlsynth.utils.resultutils import effects

@pytest.fixture
def small_data():
    np.random.seed(0)
    T, J = 5, 3
    X = np.random.randn(T, J)
    y = np.random.randn(T)
    return y, X


def make_scm_synth(T=40, T0=25, J=6, seed=123):
    """Generate synthetic SCM-style data for testing."""
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, J))
    w_true = rng.normal(size=J)
    y = Y0 @ w_true
    donor_names = [f"donor_{j}" for j in range(J)]
    time = np.arange(T)
    return {
        "y": y,
        "donor_matrix": Y0,
        "pre_periods": T0,
        "donor_names": donor_names,
        "time_labels": time,
        "w_true": w_true,
    }


def test_ols_equivalence():
    """
    Ground-floor equivalence test: pure OLS on synthetic data.
    Compares Opt2 (new), Opt (old), and closed-form OLS.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=1)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]
    donor_names = data["donor_names"]

    # ---------- Closed-form OLS ----------
    w_cf = np.linalg.lstsq(Y0[:T0], y[:T0], rcond=None)[0]

    # ---------- NEW CODE ----------
    sc_results_new = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=False,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,
        alpha=0.0,
    )
    w_new = sc_results_new["weights"]["w"]

    # ---------- OLD CODE ----------
    sc_results_old = Opt.SCopt(
        num_control_units=Y0.shape[1],
        target_outcomes_pre_treatment=y[:T0],
        donor_outcomes_pre_treatment=Y0,
        num_pre_treatment_periods=T0,
        scm_model_type="OLS",
        donor_names=donor_names,
        lambda_penalty=0,
        p=2,
        q=2,
    )
    w_old = sc_results_old.solution.primal_vars[next(iter(sc_results_old.solution.primal_vars))]

    # ---------- Assertions ----------
    np.testing.assert_allclose(w_new, w_old, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(w_new, w_cf, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------
# Unit test: OLS equivalence with intercept
# ------------------------------------------------------------
def test_ols_equivalence_with_intercept():
    """
    Ground-floor equivalence test: OLS with intercept on synthetic data.
    Compares Opt2 (new), Opt (old), and closed-form OLS.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=2)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]
    donor_names = data["donor_names"]

    # ---------- Closed-form OLS with intercept ----------
    Y0_aug = np.hstack([Y0[:T0], np.ones((T0, 1))])
    w_aug = np.linalg.lstsq(Y0_aug, y[:T0], rcond=None)[0]
    w_cf = w_aug[:-1]
    b0_cf = w_aug[-1]

    # ---------- NEW CODE (Opt2) ----------
    sc_results_new = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,
        alpha=0.0,
    )
    w_new = sc_results_new["weights"]["w"]
    b0_new = sc_results_new["weights"]["b0"]

    # ---------- OLD CODE (Opt) ----------
    sc_results_old = Opt.SCopt(
        num_control_units=Y0.shape[1],
        target_outcomes_pre_treatment=y[:T0],
        donor_outcomes_pre_treatment=Y0,
        num_pre_treatment_periods=T0,
        scm_model_type="OLS",
        donor_names=donor_names,
        lambda_penalty=0,
        p=2,
        q=2,
        fit_intercept=True
    )

    # Robust extraction of weight vector and intercept
    old_vars = list(sc_results_old.solution.primal_vars.values())
    w_old = None
    b0_old = 0.0
    for v in old_vars:
        arr = np.atleast_1d(v)  # convert scalar to 1D array if needed
        if arr.size == Y0.shape[1]:
            w_old = arr
        elif arr.size == 1:
            b0_old = arr.item()  # scalar intercept

    assert w_old is not None, "Could not find weight vector in old solver output"

    # ---------- Assertions ----------
    np.testing.assert_allclose(w_new, w_old, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(w_new, w_cf, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(b0_new, b0_old, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(b0_new, b0_cf, atol=1e-6, rtol=1e-6)



# ============================================================
# Ridge equivalence test: Opt2 vs closed-form Ridge
# ============================================================

@pytest.mark.parametrize("seed", [3, 7, 42])
def test_ridge_equivalence(seed):
    """
    Ground-floor equivalence test: Ridge regression (L2) with synthetic data.
    Compares Opt2.SCopt (new) against closed-form Ridge solution.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]

    # ---------- Sanity assertions ----------
    assert Y0.shape[0] >= T0
    assert y.shape[0] == Y0.shape[0]
    assert T0 > 0

    # ---------- Ridge penalty ----------
    lam = 1.5

    # ---------- Closed-form Ridge ----------
    X_pre = Y0[:T0]
    ridge_matrix = X_pre.T @ X_pre + lam * np.eye(Y0.shape[1])
    w_ridge = np.linalg.solve(ridge_matrix, X_pre.T @ y[:T0])
    y_ridge_pred = Y0 @ w_ridge

    # ---------- SCopt Ridge (pure L2) ----------
    sc_results_new = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=False,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=0.0,
    )
    w_new = sc_results_new["weights"]["w"]
    y_new_pred = sc_results_new["predictions"]

    # ---------- Shape assertions ----------
    assert w_new.shape == w_ridge.shape
    assert y_new_pred.shape == y_ridge_pred.shape

    # ---------- Hybrid value assertions ----------
    # L2 norm checks (tight)
    np.testing.assert_allclose(np.linalg.norm(w_new), np.linalg.norm(w_ridge), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y_new_pred), np.linalg.norm(y_ridge_pred), rtol=1e-4, atol=1e-6)
    # Relaxed max elementwise differences
    assert np.max(np.abs(w_new - w_ridge)) < 1e-3
    assert np.max(np.abs(y_new_pred - y_ridge_pred)) < 2.5e-3


@pytest.mark.parametrize("seed", [3, 7, 42])
def test_ridge_equivalence_with_intercept(seed):
    """
    Ridge regression with intercept: compare Opt2.SCopt vs. closed-form solution
    using hybrid L2 + max elementwise checks for robust equivalence.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]

    # ---------- Sanity checks ----------
    assert Y0.shape[0] >= T0
    assert y.shape[0] == Y0.shape[0]
    assert T0 > 0

    # ---------- Ridge penalty ----------
    lam = 1.5

    # ---------- Closed-form ridge with intercept ----------
    X_pre = Y0[:T0]
    X_aug = np.hstack([X_pre, np.ones((T0, 1))])
    ridge_matrix = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
    w_aug = np.linalg.solve(ridge_matrix, X_aug.T @ y[:T0])
    w_ridge = w_aug[:-1]
    b0_ridge = w_aug[-1]
    y_ridge_pred = Y0 @ w_ridge + b0_ridge

    # ---------- SCopt ridge with intercept ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=0.0,
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    w_new = sc_results["weights"]["w"]
    b0_new = sc_results["weights"]["b0"]
    y_new_pred = sc_results["predictions"]

    # ---------- Shape assertions ----------
    assert w_new.shape == w_ridge.shape
    assert y_new_pred.shape == y_ridge_pred.shape

    # ---------- Hybrid value assertions ----------
    # L2 norm checks (tight)
    np.testing.assert_allclose(np.linalg.norm(w_new), np.linalg.norm(w_ridge), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y_new_pred), np.linalg.norm(y_ridge_pred), rtol=1e-4, atol=1e-6)
    # Relaxed max elementwise differences
    assert np.max(np.abs(w_new - w_ridge)) < 1e-3
    assert np.max(np.abs(y_new_pred - y_ridge_pred)) < 2.5e-3
    assert np.abs(b0_new - b0_ridge) < 2e-3




@pytest.mark.parametrize("seed", [3, 7, 42])
def test_lasso_equivalence_old_vs_new(seed):
    """
    Compare LASSO weights and predictions from new Opt2.SCopt vs old Opt.SCopt.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]
    donor_names = [f"donor_{i}" for i in range(Y0.shape[1])]
    y_pre = y[:T0]

    # ---------- Old SCopt ----------
    scm_old = Opt.SCopt(
        num_control_units=Y0.shape[1],
        target_outcomes_pre_treatment=y_pre,
        num_pre_treatment_periods=T0,
        donor_outcomes_pre_treatment=Y0,
        scm_model_type="OLS",
        donor_names=donor_names,
        lambda_penalty=0.0,
        p=1,  # L1 norm
        q=1,
    )
    w_old = scm_old.solution.primal_vars[next(iter(scm_old.solution.primal_vars))]
    y_old_pred = Y0 @ w_old

    # ---------- New SCopt (Opt2) ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=False,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,
        alpha=1.0,  # pure L1
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )
    w_new = sc_results["weights"]["w"]
    y_new_pred = sc_results["predictions"]

    # ---------- Shape assertions ----------
    assert w_new.shape == w_old.shape, f"Weight shapes mismatch: {w_new.shape} != {w_old.shape}"
    assert y_new_pred.shape == y_old_pred.shape, f"Prediction shapes mismatch: {y_new_pred.shape} != {y_old_pred.shape}"

    # ---------- Hybrid value assertions ----------
    # L2 norm checks
    np.testing.assert_allclose(np.linalg.norm(w_new), np.linalg.norm(w_old), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y_new_pred), np.linalg.norm(y_old_pred), rtol=1e-3, atol=1e-6)
    # Max-elementwise differences (relaxed for solver variability)
    assert np.max(np.abs(w_new - w_old)) < 5e-3, f"Max weight diff too large: {np.max(np.abs(w_new - w_old))}"
    assert np.max(np.abs(y_new_pred - y_old_pred)) < 5e-3, f"Max pred diff too large: {np.max(np.abs(y_new_pred - y_old_pred))}"



@pytest.mark.parametrize("seed", [3, 7, 42])
def test_lasso_equivalence_with_intercept(seed):
    """
    Compare LASSO weights and predictions with intercept from new Opt2.SCopt vs old Opt.SCopt.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]
    donor_names = [f"donor_{i}" for i in range(Y0.shape[1])]
    y_pre = y[:T0]

    # ---------- Old SCopt (augment with intercept) ----------
    Y0_aug = np.hstack([Y0, np.ones((Y0.shape[0], 1))])
    scm_old = Opt.SCopt(
        num_control_units=Y0_aug.shape[1],
        target_outcomes_pre_treatment=y_pre,
        num_pre_treatment_periods=T0,
        donor_outcomes_pre_treatment=Y0_aug,
        scm_model_type="OLS",
        donor_names=donor_names + ["intercept"],
        lambda_penalty=0.0,
        p=1,
        q=1,
    )
    w_old_aug = scm_old.solution.primal_vars[next(iter(scm_old.solution.primal_vars))]
    w_old = w_old_aug[:-1]
    b0_old = w_old_aug[-1]
    y_old_pred = Y0 @ w_old + b0_old

    # ---------- New SCopt ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,
        alpha=1.0,
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )
    w_new = sc_results["weights"]["w"]
    b0_new = sc_results["weights"]["b0"]
    y_new_pred = sc_results["predictions"]

    # ---------- Shape assertions ----------
    assert w_new.shape == w_old.shape
    assert y_new_pred.shape == y_old_pred.shape

    # ---------- Hybrid value assertions ----------
    np.testing.assert_allclose(np.linalg.norm(w_new), np.linalg.norm(w_old), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y_new_pred), np.linalg.norm(y_old_pred), rtol=1e-3, atol=1e-6)
    assert np.max(np.abs(w_new - w_old)) < 5e-3
    assert np.max(np.abs(y_new_pred - y_old_pred)) < 5e-3
    assert np.abs(b0_new - b0_old) < 5e-4



@pytest.mark.parametrize("seed", [3, 7, 42])
def test_elastic_net(seed):
    """
    Elastic Net synthetic control: lambda=0.5, alpha=0.2
    Checks basic shapes and internal consistency (predictions match weights).
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]

    # ---------- Penalty parameters ----------
    lam = 0.5
    alpha = 0.2  # Elastic Net: mix of L1 and L2

    # ---------- Fit Elastic Net SCopt ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=alpha,
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    w = sc_results["weights"]["w"]
    b0 = sc_results["weights"]["b0"].item()
    y_pred = sc_results["predictions"]

    # ---------- Shape checks ----------
    assert w.shape[0] == Y0.shape[1], f"Weight vector shape mismatch: {w.shape[0]} != {Y0.shape[1]}"
    assert np.isscalar(b0), "Intercept should be a scalar"
    np.testing.assert_allclose(y_pred, Y0 @ w + b0, rtol=1e-8, atol=1e-8)

    # ---------- Prediction consistency ----------
    np.testing.assert_allclose(y_pred, Y0 @ w + b0, rtol=1e-8, atol=1e-8)



@pytest.mark.parametrize("seed", [3, 7, 42])
def test_linf_elastic_net_reduces_to_ols(seed):
    """
    Test that when lam=0 and second_norm='linf', the elastic-net-style SCopt
    converges to the OLS solution.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]

    # ---------- Fit OLS directly ----------
    X_pre = Y0[:T0]
    y_pre = y[:T0]
    X_aug = np.hstack([X_pre, np.ones((T0, 1))])
    ols_coef = np.linalg.lstsq(X_aug, y_pre, rcond=None)[0]
    w_ols = ols_coef[:-1]
    b0_ols = ols_coef[-1]
    y_ols_pred = Y0 @ w_ols + b0_ols

    # ---------- Fit SCopt with lam=0, second_norm='linf' ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,  # No penalty
        alpha=0.5,  # Doesn't matter, penalty is 0
        second_norm="linf",
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    w_new = sc_results["weights"]["w"]
    b0_new = sc_results["weights"]["b0"].item()
    y_new_pred = sc_results["predictions"]

    # ---------- Assertions ----------
    # Shape checks
    assert w_new.shape == w_ols.shape
    assert np.isscalar(b0_new)

    # Value checks: predictions and weights should match OLS
    np.testing.assert_allclose(w_new, w_ols, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b0_new, b0_ols, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(y_new_pred, y_ols_pred, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [3, 7, 42])
def test_l1linf_elastic_net_with_linf_zero_equals_lasso(seed):
    """
    Test that the L1-Linf elastic net with alpha=1 (max norm weight 0)
    produces the same solution as the LASSO SCopt.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]

    lam = 0.5  # example penalty
    alpha = 1.0  # full L1, no Linf contribution

    # ---------- Fit standard LASSO SCopt ----------
    lasso_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=1.0,  # only L1
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    # ---------- Fit L1-Linf elastic net with alpha=1 ----------
    l1linf_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=1.0,  # max norm weight = 0
        second_norm="linf",  # doesn't matter, weight=0
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    # ---------- Compare solutions ----------
    w_lasso = lasso_results["weights"]["w"]
    b0_lasso = lasso_results["weights"]["b0"]
    y_lasso_pred = lasso_results["predictions"]

    w_l1linf = l1linf_results["weights"]["w"]
    b0_l1linf = l1linf_results["weights"]["b0"]
    y_l1linf_pred = l1linf_results["predictions"]

    # ---------- Assertions ----------
    np.testing.assert_allclose(w_l1linf, w_lasso, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b0_l1linf, b0_lasso, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(y_l1linf_pred, y_lasso_pred, rtol=1e-6, atol=1e-8)



@pytest.mark.parametrize("seed", [3, 7, 42])
def test_simplex_ols_equivalence(seed):
    """
    Pure simplex-constrained OLS:
    Compare old Opt.SCopt vs. new Opt2.SCopt with lambda=0.
    """
    # ---------- Setup ----------
    data = make_scm_synth(T=40, T0=25, J=6, seed=seed)
    y = data["y"]
    Y0 = data["donor_matrix"]
    T0 = data["pre_periods"]
    donor_names = [f"J{i}" for i in range(Y0.shape[1])]
    y_pre = y[:T0]

    # ---------- Old Opt ----------
    scm_old = Opt.SCopt(
        num_control_units=Y0.shape[1],
        target_outcomes_pre_treatment=y_pre,
        num_pre_treatment_periods=T0,
        donor_outcomes_pre_treatment=Y0,
        scm_model_type="SIMPLEX",
        donor_names=donor_names,
        lambda_penalty=0.0,
        p=1,  # L1 norm (irrelevant with lambda=0)
        q=1,  # L∞ norm (irrelevant with lambda=0)
    )
    w_old = scm_old.solution.primal_vars[next(iter(scm_old.solution.primal_vars))]

    # ---------- New Opt2 ----------
    sc_results = Opt2.SCopt(
        y=y,
        X=Y0,
        T0=T0,
        fit_intercept=False,
        constraint_type="simplex",
        objective_type="penalized",
        lam=0.0,
        alpha=100000.0,  # irrelevant
        second_norm="linf",
        solver="CVXOPT",
        tol_abs=1e-8,
        tol_rel=1e-8
    )
    w_new = sc_results["weights"]["w"]

    # ---------- Assertions ----------
    assert w_old.shape == w_new.shape, "Weight vector shape mismatch"

    diff_norm = np.linalg.norm(w_old - w_new, ord=2)
    assert diff_norm < 2e-5, f"L2 norm of weight difference too large: {diff_norm:.2e}"



def test_unpenalized_scm_equivalence():
    """
    Canonical SCM should be equivalent whether specified via
    symbolic flags or a custom objective.
    """

    rng = np.random.default_rng(123)

    T, J = 20, 5
    X = rng.normal(size=(T, J))
    true_w = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    y = X @ true_w + 0.01 * rng.normal(size=T)

    # -----------------------------
    # A. Built-in symbolic SCM
    # -----------------------------
    res_builtin = Opt2.SCopt(
        y=y,
        X=X,
        T0=T,
        fit_intercept=False,
        constraint_type="simplex",
        objective_type="penalized",
        lam=0.0,
        solver="CLARABEL",
    )

    w_builtin = res_builtin["weights"]["w"]

    # -----------------------------
    # B. Custom objective SCM
    # -----------------------------
    def custom_objective(y, X, w, b0):
        return cp.Minimize(cp.sum_squares(y - X @ w))

    res_custom = Opt2.SCopt(
        y=y,
        X=X,
        T0=T,
        fit_intercept=False,
        constraint_type="simplex",
        custom_objective_callable=custom_objective,
        solver="CLARABEL",
    )

    w_custom = res_custom["weights"]["w"]

    # -----------------------------
    # Assertions
    # -----------------------------
    np.testing.assert_allclose(
        w_builtin,
        w_custom,
        atol=1e-6,
        err_msg="Built-in SCM and custom-objective SCM differ!"
    )

    # Optional: also check feasibility
    assert np.all(w_builtin >= -1e-8)
    assert abs(np.sum(w_builtin) - 1.0) < 1e-6



# =========================
# CVXPY atom inspection helper
# =========================

def atom_names(expr):
    """
    Robustly extract atom names from a CVXPY expression.

    Handles:
    - atom instances (atom.name())
    - atom classes (atom.__name__)
    - mixed canonicalized expressions
    """
    names = set()
    for atom in expr.atoms():
        # Instance case
        if hasattr(atom, "name") and callable(getattr(atom, "name", None)):
            try:
                names.add(atom.name())
                continue
            except TypeError:
                pass

        # Class or fallback case
        if hasattr(atom, "__name__"):
            names.add(atom.__name__)
        elif hasattr(atom, "NAME"):
            names.add(atom.NAME)
        else:
            names.add(type(atom).__name__)

    return names


# =========================
# Tests
# =========================

def test_entropy_objective_used(small_data):
    y, X = small_data

    res = Opt2.SCopt(
        y=y,
        X=X,
        objective_type="relaxed",
        relaxation_type="entropy",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atom_set = atom_names(res["problem"].objective)

    assert "entr" in atom_set
    assert "norm2" not in atom_set


def test_l2_relaxed_objective_has_no_entropy(small_data):
    y, X = small_data

    res = Opt2.SCopt(
        y=y,
        X=X,
        objective_type="relaxed",
        relaxation_type="l2",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atom_set = atom_names(res["problem"].objective)

    assert "entr" not in atom_set
    assert "Pnorm" in atom_set


def test_penalized_objective_no_entropy(small_data):
    y, X = small_data

    res = Opt2.SCopt(
        y=y,
        X=X,
        objective_type="penalized",
        lam=1.0,
        alpha=0.5,
        solve=False,
    )

    atom_set = atom_names(res["problem"].objective)

    assert "entr" not in atom_set
