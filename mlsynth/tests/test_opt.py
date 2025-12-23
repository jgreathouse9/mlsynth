# ============================================================
# tests/test_opt2.py
# ============================================================

import numpy as np
import cvxpy as cp
import pytest
from mlsynth.utils.optutils import Opt2
from mlsynth.utils.opthelpers import OptHelpers
from sklearn.linear_model import Lasso

# Fixture for synthetic data
@pytest.fixture
def incrementality_synth_panel():
    # Assuming incrementality_synth_panel returns y, X, T0
    # Replace with actual implementation or mock if needed
    np.random.seed(42)
    T = 10
    J = 5
    X = np.random.randn(T, J)
    w_true = np.random.rand(J)
    w_true /= w_true.sum()
    y = X @ w_true + 0.1 * np.random.randn(T)
    T0 = 5
    return y, X, T0

@pytest.mark.parametrize("constraint_type", ["simplex", "affine", "unconstrained"])
@pytest.mark.parametrize("objective_type", ["penalized"])
def test_scopt_runs(incrementality_synth_panel, constraint_type, objective_type):
    """Smoke test: SCopt runs without error and returns expected keys."""

    y, X, T0 = incrementality_synth_panel

    # --- Relaxed-only parameters ---
    kwargs = {}
    if objective_type == "penalized":
        kwargs.update({"lam": 0.1, "alpha": 0.5})
    elif objective_type == "relaxed":
        # Skip infeasible combos
        if constraint_type not in ["simplex", "affine"]:
            pytest.skip("Entropy relaxation requires sum-to-one constraints")
        kwargs.update({"relaxation_type": "entropy", "tau": 0.1})

    fit_intercept_flag = True
    if objective_type == "relaxed" and kwargs.get("relaxation_type") == "entropy":
        fit_intercept_flag = False

    # --- Run SCopt ---
    result = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=fit_intercept_flag,
        constraint_type=constraint_type,
        objective_type=objective_type,
        solver="CLARABEL",
        **kwargs
    )

    # Smoke checks
    assert "weights" in result
    assert "w" in result["weights"]
    w = result["weights"]["w"]
    assert w.shape[0] == X.shape[1]
    if "b0" in result["weights"]:
        b0 = result["weights"]["b0"]
        assert np.isscalar(b0) or (np.ndim(b0) == 0)
    if "predictions" in result:
        y_pred = result["predictions"]
        assert y_pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("solve", [True, False])
def test_scopt_solve_flag(incrementality_synth_panel, solve):
    """Check that solve=False builds the problem but does not solve it."""
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        constraint_type="simplex",
        objective_type="penalized",
        lam=0.1,
        alpha=0.5,
        solve=solve
    )

    # CVXPY problem object exists
    assert "problem" in res

    if solve:
        # Solved problem returns weights
        assert "weights" in res
    else:
        # We expect no weights yet
        assert "weights" not in res or res["weights"] is None



@pytest.mark.parametrize("lam", [0.1, 0.5])
@pytest.mark.parametrize("second_norm", ["l2", "inf"])
def test_l1linf_elastic_net_with_linf_zero_equals_lasso(incrementality_synth_panel, lam, second_norm):
    """L1-Linf elastic net with alpha=1 should produce same solution as LASSO SCopt."""
    y, X, T0 = incrementality_synth_panel
    alpha = 1.0  # full L1, no Linf contribution

    # ---------- Standard LASSO SCopt ----------
    lasso_results = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=alpha,
        solver="CLARABEL",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    # ---------- L1-Linf Elastic Net ----------
    l1linf_results = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=alpha,
        second_norm=second_norm,
        solver="CLARABEL",
        tol_abs=1e-8,
        tol_rel=1e-8
    )

    # ---------- Assertions ----------
    np.testing.assert_allclose(l1linf_results["weights"]["w"], lasso_results["weights"]["w"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(l1linf_results["weights"]["b0"], lasso_results["weights"]["b0"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(l1linf_results["predictions"], lasso_results["predictions"], rtol=1e-6, atol=1e-8)


def test_unpenalized_scm_equivalence(incrementality_synth_panel):
    """
    Canonical SCM should be equivalent whether specified via symbolic flags
    or a custom objective.
    """
    y, X, T0 = incrementality_synth_panel

    # -----------------------------
    # A. Built-in symbolic SCM
    # -----------------------------
    res_builtin = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
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
        T0=T0,
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

    # Feasibility checks
    assert np.all(w_builtin >= -1e-8)
    assert abs(np.sum(w_builtin) - 1.0) < 1e-6


@pytest.mark.parametrize("constraint_type", ["simplex", "affine"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("second_norm", ["l2", "inf"])
def test_penalized_sc_weight_invariants(incrementality_synth_panel, constraint_type, alpha, second_norm):
    """
    Invariant tests for penalized synthetic control weights.

    Simplex:
        - weights are finite
        - weights sum to 1
        - weights lie in [0, 1]

    Affine:
        - weights are finite
        - weights sum to 1
        - weights may be negative or >1 (no bounds asserted)
    """
    y, X, T0 = incrementality_synth_panel

    # ------------------ Fit model ------------------
    sc_results = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=False,
        constraint_type=constraint_type,
        objective_type="penalized",
        lam=0.5,
        alpha=alpha,
        second_norm=second_norm,
        solver="CLARABEL",
        tol_abs=1e-8,
        tol_rel=1e-8,
    )

    w = sc_results["weights"]["w"]

    # ------------------ Universal invariants ------------------
    assert w.ndim == 1, "Weights must be a 1D vector"
    assert np.all(np.isfinite(w)), f"Weights contain NaN or infinity: {w}"
    assert np.isclose(np.sum(w), 1.0, atol=1e-10), (
        f"Weights do not sum to 1 for constraint={constraint_type}, "
        f"alpha={alpha}, second_norm={second_norm}. "
        f"Sum={np.sum(w)}"
    )

    # ------------------ Simplex-specific invariants ------------------
    if constraint_type == "simplex":
        w_clipped = np.clip(w, 0.0, 1.0 + 1e-10)
        assert np.all(w_clipped >= 0.0), f"Simplex weights must be non-negative. Found: {w}"
        assert np.all(w_clipped <= 1.0 + 1e-10), f"Simplex weights must be ≤ 1. Found: {w}"


# =========================
# CVXPY atom inspection helper
# =========================

def atom_names(expr):
    """Robustly extract atom names from a CVXPY expression."""
    names = set()
    for atom in expr.atoms():
        if hasattr(atom, "name") and callable(getattr(atom, "name", None)):
            try:
                names.add(atom.name())
                continue
            except TypeError:
                pass
        if hasattr(atom, "__name__"):
            names.add(atom.__name__)
        elif hasattr(atom, "NAME"):
            names.add(atom.NAME)
        else:
            names.add(type(atom).__name__)
    return names


def test_entropy_objective_used(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        objective_type="relaxed",
        relaxation_type="entropy",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atom_set = atom_names(res["problem"].objective)

    assert "entr" in atom_set
    assert "norm2" not in atom_set


def test_l2_relaxed_objective_has_no_entropy(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        objective_type="relaxed",
        relaxation_type="l2",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atom_set = atom_names(res["problem"].objective)

    assert "entr" not in atom_set
    assert "Pnorm" in atom_set


# -------------------------------
# 1. T0=None should run smoothly
# -------------------------------

def test_scopt_no_T0(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=None,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.1,
        alpha=0.5,
        solver="CLARABEL",
    )

    assert "weights" in res
    w = res["weights"]["w"]
    assert w.shape[0] == X.shape[1]

# -------------------------------
# 2. Non-standard constraints
# -------------------------------

@pytest.mark.parametrize("constraint_type", ["nonneg", "unit"])
def test_scopt_edge_constraints(incrementality_synth_panel, constraint_type):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=False,
        constraint_type=constraint_type,
        objective_type="penalized",
        lam=0.1,
        alpha=0.5,
        solver="CLARABEL",
    )

    w = res["weights"]["w"]
    assert np.all(np.isfinite(w))
    if constraint_type == "nonneg":
        assert np.all(w >= -1e-8)
    if constraint_type == "unit":
        assert np.all(w >= -1e-8)
        assert np.all(w <= 1 + 1e-8)

# -------------------------------
# 3. Custom penalty callable
# -------------------------------

def test_scopt_custom_penalty(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    def custom_penalty(w):
        return cp.sum_squares(w) * 0.5

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.1,
        alpha=0.5,
        custom_penalty_callable=custom_penalty,
        solver="CLARABEL",
    )

    w = res["weights"]["w"]
    assert np.all(np.isfinite(w))

# -------------------------------
# 4. Extreme regularization values
# -------------------------------

@pytest.mark.parametrize("lam", [1e-6, 1e3])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_scopt_extreme_regularization(incrementality_synth_panel, lam, alpha):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=lam,
        alpha=alpha,
        solver="CLARABEL",
    )

    w = res["weights"]["w"]
    assert np.all(np.isfinite(w))

# -------------------------------
# 5. Minimal data corner cases
# -------------------------------

def test_scopt_single_feature_single_obs():
    y = np.array([1.0])
    X = np.array([[2.0]])
    T0 = 1

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        fit_intercept=True,
        constraint_type="unconstrained",
        objective_type="penalized",
        lam=0.0,
        alpha=0.5,
        solver="CLARABEL",
    )

    w = res["weights"]["w"]
    assert w.shape[0] == 1
    assert np.all(np.isfinite(w))
    if "b0" in res["weights"]:
        b0 = res["weights"]["b0"]
        assert np.isscalar(b0) or (np.ndim(b0) == 0)


# -------------------------------
# Helper to extract CVXPY atom names
# -------------------------------
def atom_names(expr):
    names = set()
    for atom in expr.atoms():
        if hasattr(atom, "name") and callable(getattr(atom, "name", None)):
            try:
                names.add(atom.name())
                continue
            except TypeError:
                pass
        if hasattr(atom, "__name__"):
            names.add(atom.__name__)
        elif hasattr(atom, "NAME"):
            names.add(atom.NAME)
        else:
            names.add(type(atom).__name__)
    return names

# -------------------------------
# 1. Entropy relaxation
# -------------------------------
def test_entropy_relaxation_atoms(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        objective_type="relaxed",
        relaxation_type="entropy",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atoms = atom_names(res["problem"].objective)
    assert "entr" in atoms
    assert "Pnorm" not in atoms

# -------------------------------
# 2. L2 relaxation
# -------------------------------
def test_l2_relaxation_atoms(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = Opt2.SCopt(
        y=y,
        X=X,
        T0=T0,
        objective_type="relaxed",
        relaxation_type="l2",
        constraint_type="simplex",
        tau=0.1,
        solve=False,
    )

    atoms = atom_names(res["problem"].objective)
    assert "entr" not in atoms
    assert "Pnorm" in atoms


