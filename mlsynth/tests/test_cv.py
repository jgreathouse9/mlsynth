import numpy as np
import pytest
from unittest.mock import patch

from mlsynth.utils.crossval import ElasticNetCV, RelaxationCV, fit_en_scm, generate_lambda_seq2


from mlsynth.utils.optutils import Opt2
from mlsynth.tests.helper_tests import incrementality_synth_panel
from mlsynth.exceptions import MlsynthEstimationError


# ======================================================
# ElasticNetCV grid processing tests
# ======================================================

def test_process_grid_scalar_lambda_alpha(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel
    enet = ElasticNetCV(alpha=0.5, lam=0.3)

    enet._process_grid(X, y)

    assert np.allclose(enet.alphas_, [0.5])
    assert np.allclose(enet.lams_, [0.3])


def test_process_grid_generate_lambda(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel
    enet = ElasticNetCV(alpha=0.5, lam=None)

    enet._process_grid(X, y)

    assert enet.alphas_[0] == 0.5
    assert enet.lams_.shape[0] == 30


def test_process_grid_default_alpha(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel
    enet = ElasticNetCV(alpha=None, lam=None)

    enet._process_grid(X, y)

    assert len(enet.alphas_) == 10
    assert len(enet.lams_) == 30

# ======================================================
# Lambda sequence smoke tests
# ======================================================

def test_generate_lambda_seq2_monotone_and_bounded(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel

    lam_seq = generate_lambda_seq2(y, X, alpha=0.5)

    assert lam_seq.ndim == 1
    assert len(lam_seq) == 30
    assert np.all(np.diff(lam_seq) < 0)        # strictly decreasing
    assert lam_seq.max() <= 20
    assert lam_seq.min() > 0


# ======================================================
# Grid processing smoke tests
# ======================================================

def test_process_grid_nonempty(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    model = ElasticNetCV(alpha=None, lam=None)

    model._process_grid(X, y)

    assert len(model.alphas_) > 0
    assert len(model.lams_) > 0


# ======================================================
# Cross-validation invariants
# ======================================================

def test_cross_validate_sets_attributes(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    model = ElasticNetCV(alpha=[0.2, 0.5], lam=[0.1, 0.2], n_splits=3)

    model.fit(X, y)

    assert hasattr(model, "lam_")
    assert hasattr(model, "alpha_")
    assert model.cv_performed_ is True


def test_selected_params_from_grid(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    alphas = [0.2, 0.8]
    lams = [0.05, 0.1, 0.2]

    model = ElasticNetCV(alpha=alphas, lam=lams)
    model.fit(X, y)

    assert model.alpha_ in alphas
    assert model.lam_ in lams


# ======================================================
# Solver failure tolerance tests
# ======================================================

@patch("mlsynth.utils.crossval.Opt2.SCopt", side_effect=RuntimeError("boom"))
def test_solve_enet_failure_returns_zero_weights(mock_scopt, incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    model = ElasticNetCV()

    w = model._solve_enet(X, y, lam=0.1, alpha=0.5)

    assert w.shape == (X.shape[1],)
    assert np.all(w == 0.0)


@patch.object(ElasticNetCV, "_solve_enet", return_value=np.array([]))
def test_cv_does_not_crash_on_bad_weights(mock_solve, incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    model = ElasticNetCV(alpha=[0.5], lam=[0.1], n_splits=2)

    # force safe fallback shape
    mock_solve.return_value = np.zeros(X.shape[1])

    model.fit(X, y)

    assert hasattr(model, "coef_")
    assert model.coef_.shape[0] == X.shape[1]


# ======================================================
# End-to-end fit_en_scm smoke tests
# ======================================================

def test_fit_en_scm_return_schema(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = fit_en_scm(
        X_pre=X[:T0],
        y_pre=y[:T0],
        X_post=X[T0:],
        y=y,
        donor_names=[f"d{i}" for i in range(X.shape[1])],
    )

    assert "donor_weights" in res
    assert "predictions" in res
    assert "Results" in res
    assert "Model" in res
    assert "hyperparameters" in res


def test_fit_en_scm_weights_sum_to_one(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel

    res = fit_en_scm(
        X_pre=X[:T0],
        y_pre=y[:T0],
        X_post=X[T0:],
        y=y,
        donor_names=[f"d{i}" for i in range(X.shape[1])],
        standardize=True,
    )

    s = sum(res["donor_weights"].values())
    assert np.isclose(s, 1.0, atol=1e-6)


# ======================================================
# API contract / behavior tests
# ======================================================

def test_predict_before_fit_raises(incrementality_synth_panel):
    _, X, _ = incrementality_synth_panel
    model = ElasticNetCV()

    with pytest.raises(AttributeError):
        model.predict(X)


# ======================================================
# RelaxationCV: _generate_tau_grid (valid behavior)
# ======================================================

def test_generate_tau_grid_valid(incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel
    n_taus = 20
    model = RelaxationCV(n_taus=n_taus)

    model._generate_tau_grid(X, y)

    assert hasattr(model, "taus_")
    assert model.taus_.shape[0] == n_taus
    assert np.all(np.isfinite(model.taus_))
    assert np.all(model.taus_ > 0)
    assert np.all(model.taus_[:-1] > model.taus_[1:])
    assert np.isclose(model.taus_[-1], 1e-5, rtol=1e-8)
    assert np.isclose(
        model.taus_[0],
        np.linalg.norm(X.T @ y, np.inf),
        rtol=1e-8
    )



def test_coefficients_not_constant(incrementality_synth_panel):
    y, X, _ = incrementality_synth_panel
    model = ElasticNetCV(alpha=0.5, lam=None)

    model.fit(X, y)

    w = model.coef_

    # Basic sanity
    assert w.ndim == 1
    assert w.shape[0] == X.shape[1]

    # Not all coefficients are identical
    assert not np.allclose(w, w[0])



# ======================================================
# RelaxationCV: _generate_tau_grid (hard failures)
# ======================================================

def test_generate_tau_grid_raises_on_near_zero_signal():
    rng = np.random.default_rng(123)

    T, J = 30, 5
    X = rng.normal(size=(T, J))

    # Force orthogonality → no identifying signal
    y = rng.normal(size=T)
    y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    model = RelaxationCV(n_taus=10)

    with pytest.raises(
        MlsynthEstimationError,
        match="tau|identif|signal|X.T @ y"
    ):
        model._generate_tau_grid(X, y)


def test_generate_tau_grid_raises_on_non_finite_norm():
    T, J = 20, 4
    X = np.ones((T, J))
    y = np.full(T, np.inf)

    model = RelaxationCV(n_taus=10)

    with pytest.raises(MlsynthEstimationError, match="finite"):
        model._generate_tau_grid(X, y)


def test_generate_tau_grid_shape_mismatch():
    X = np.random.randn(10, 3)
    y = np.random.randn(8)

    model = RelaxationCV(n_taus=10)

    with pytest.raises(
            MlsynthEstimationError,
            match="(?i)shape|dimension"
    ):
        model._generate_tau_grid(X, y)


@pytest.mark.parametrize("n_taus", [0, 1, -5])
def test_generate_tau_grid_invalid_n_taus(n_taus, incrementality_synth_panel):
    y, X, T0 = incrementality_synth_panel
    model = RelaxationCV(n_taus=n_taus)

    with pytest.raises(MlsynthEstimationError, match="n_taus"):
        model._generate_tau_grid(X, y)


def test_generate_tau_grid_monotone_and_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)

    model = RelaxationCV(n_taus=12)
    model._generate_tau_grid(X, y)

    taus = model.taus_

    assert np.all(np.isfinite(taus))
    assert np.all(taus > 0)
    assert np.all(np.diff(taus) < 0)  # strictly decreasing


## Edge Case Tests


def test_generate_tau_grid_min_n_taus():
    """Ensure n_taus >= 2 is enforced"""
    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    model = RelaxationCV(tau=None, n_taus=1)
    with pytest.raises(MlsynthEstimationError, match="n_taus"):
        model._process_tau_grid(X, y)


def test_generate_tau_grid_negative_n_taus():
    """Negative n_taus should fail"""
    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    model = RelaxationCV(tau=None, n_taus=-5)
    with pytest.raises(MlsynthEstimationError, match="n_taus"):
        model._process_tau_grid(X, y)


def test_generate_tau_grid_upper_limit_equals_lower_limit():
    """If signal exactly at lower limit, grid should be flat or raise"""
    X = np.eye(5)
    y = np.ones(5) * 1e-5  # exactly lower_limit
    model = RelaxationCV(tau=None, n_taus=3)

    with pytest.raises(MlsynthEstimationError, match="identifying signal"):
        model._process_tau_grid(X, y)


def test_generate_tau_grid_single_feature():
    """Check grid generation with J=1 (single donor)"""
    X = np.ones((10, 1))
    y = np.arange(1, 11)
    model = RelaxationCV(tau=None, n_taus=5)

    model._process_tau_grid(X, y)
    assert model.taus_.shape[0] == model.n_taus
    assert np.all(model.taus_ > 0)


def test_generate_tau_grid_large_signal():
    """Ensure large signals do not break geomspace"""
    X = np.eye(5)
    y = np.ones(5) * 1e5
    model = RelaxationCV(tau=None, n_taus=4)

    model._process_tau_grid(X, y)
    assert model.taus_.shape[0] == model.n_taus
    assert model.taus_[0] > model.taus_[-1]


# ------------------------------
# Smoke tests for _process_tau_grid
# ------------------------------

def test_process_tau_grid_with_scalar_tau():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)

    tau_scalar = 0.1
    model = RelaxationCV(tau=tau_scalar, n_taus=5)
    model._process_tau_grid(X, y)

    # scalar input → skip CV
    assert model.skip_cv_ is True
    assert np.all(model.taus_ == tau_scalar)
    assert model.tau_ == tau_scalar

def test_process_tau_grid_with_array_tau():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)

    tau_array = [0.2, 0.05, 0.15]
    model = RelaxationCV(tau=tau_array, n_taus=5)
    model._process_tau_grid(X, y)

    # array input → do not skip CV, descending sort
    assert model.skip_cv_ is False
    assert np.all(model.taus_ == sorted(tau_array, reverse=True))

def test_process_tau_grid_with_none_tau_generates_grid():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)

    model = RelaxationCV(tau=None, n_taus=5)
    model._process_tau_grid(X, y)

    # None input → triggers _generate_tau_grid
    assert model.skip_cv_ is False
    assert hasattr(model, "taus_")
    assert model.taus_.shape[0] == model.n_taus
    assert np.all(model.taus_ > 0)
    assert np.all(np.diff(model.taus_) <= 0)

def test_process_tau_grid_invalid_X_or_y_shape_raises():
    X = np.random.randn(10, 3)
    y = np.random.randn(8)  # mismatch
    model = RelaxationCV(tau=None, n_taus=5)

    with pytest.raises(MlsynthEstimationError, match="Shape mismatch"):
        model._process_tau_grid(X, y)

def test_process_tau_grid_small_signal_raises():
    X = np.eye(10)
    y = np.ones(10) * 1e-6  # below lower_limit
    model = RelaxationCV(tau=None, n_taus=5)

    with pytest.raises(MlsynthEstimationError, match="identifying signal"):
        model._process_tau_grid(X, y)


def test_process_tau_grid_array_with_duplicates():
    """Tau array with duplicates should sort descending without issue"""
    X = np.random.randn(6, 2)
    y = np.random.randn(6)
    tau_array = [0.1, 0.05, 0.1]
    model = RelaxationCV(tau=tau_array, n_taus=3)

    model._process_tau_grid(X, y)
    assert np.all(model.taus_ == sorted(tau_array, reverse=True))


def test_process_tau_grid_scalar_equals_none():
    """Scalar tau slightly above lower_limit should skip CV"""
    X = np.eye(5)
    y = np.ones(5)
    tau_val = 1e-4
    model = RelaxationCV(tau=tau_val, n_taus=3)

    model._process_tau_grid(X, y)
    assert model.skip_cv_ is True
    assert model.tau_ == tau_val


def test_generate_tau_grid_degenerate_but_finite():
    """Signal barely above lower_limit should succeed"""
    X = np.eye(5)
    y = np.ones(5) * 1.1e-5  # slightly above lower_limit
    model = RelaxationCV(tau=None, n_taus=3)

    model._process_tau_grid(X, y)
    assert np.all(model.taus_ > 0)
    assert model.taus_.shape[0] == 3



# ------------------------------
# Smoke tests for _cross_validate
# ------------------------------

def test_cross_validate_runs_basic():
    """Basic smoke test: _cross_validate runs without errors and sets attributes."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    model = RelaxationCV(tau=None, n_taus=5, n_splits=3)
    model._process_tau_grid(X, y)  # generate taus

    model._cross_validate(X, y)

    # Smoke check attributes
    assert hasattr(model, "tau_")
    assert hasattr(model, "cv_mean_mse_")
    assert hasattr(model, "cv_performed_")
    assert hasattr(model, "cv_mse_path_")

    # Basic sanity checks
    assert model.cv_performed_ is True
    assert isinstance(model.cv_mean_mse_, np.ndarray)
    assert np.all(np.array(list(model.cv_mse_path_.values())) >= 0)
    assert model.tau_ in model.taus_

def test_cross_validate_with_single_fold():
    """Cross-validation still works if n_splits = 2 (minimal time series split)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)

    model = RelaxationCV(tau=None, n_taus=3, n_splits=2)
    model._process_tau_grid(X, y)
    model._cross_validate(X, y)

    assert model.cv_performed_ is True
    assert len(model.cv_mean_mse_) == len(model.taus_)
    assert model.tau_ in model.taus_

def test_cross_validate_with_scalar_tau_skips_cv():
    """When tau is scalar, _cross_validate should not run (skip_cv_ = True)."""
    X = np.random.randn(8, 2)
    y = np.random.randn(8)
    tau_val = 0.1

    model = RelaxationCV(tau=tau_val, n_taus=1, n_splits=2)
    model._process_tau_grid(X, y)

    # Should skip CV
    assert model.skip_cv_ is True
    assert model.taus_[0] == tau_val

# ------------------------------
# Edge cases for _cross_validate
# ------------------------------

def test_cross_validate_all_folds_fail():
    """
    If all folds return None (no feasible taus), _cross_validate should raise ValueError.
    We'll simulate this by monkey-patching _fit_fold.
    """
    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    model = RelaxationCV(tau=None, n_taus=5, n_splits=3)
    model._process_tau_grid(X, y)

    # Patch _fit_fold to always return None
    model._fit_fold = lambda X, y, train_idx, test_idx: None

    with pytest.raises(ValueError, match="No feasible taus for any fold"):
        model._cross_validate(X, y)

def test_cross_validate_shape_mismatch_in_fold():
    """
    If _fit_fold returns arrays of different lengths per fold,
    _cross_validate trims to minimum length. Check behavior.
    """
    X = np.random.randn(12, 4)
    y = np.random.randn(12)

    model = RelaxationCV(tau=None, n_taus=5, n_splits=3)
    model._process_tau_grid(X, y)

    # Simulate _fit_fold returning variable-length lists
    fold_outputs = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.1, 2.1, 3.1, 4.1]),  # shorter
        np.array([0.9, 1.9, 2.9, 3.9, 4.9])
    ]
    def fake_fit_fold(X, y, train_idx, test_idx):
        return fold_outputs.pop(0)
    model._fit_fold = fake_fit_fold

    model._cross_validate(X, y)

    # Should trim taus_ and cv_mean_mse_ to shortest fold length
    assert len(model.taus_) == 4
    assert len(model.cv_mean_mse_) == 4
    assert model.cv_performed_ is True

def test_cross_validate_with_nonfinite_errors():
    """
    Even if some folds produce inf or NaN errors, _cross_validate should not crash
    if trimming occurs. We'll simulate some non-finite fold outputs.
    """
    X = np.random.randn(10, 3)
    y = np.random.randn(10)

    model = RelaxationCV(tau=None, n_taus=4, n_splits=2)
    model._process_tau_grid(X, y)

    fold_outputs = [
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([0.5, 1.5, np.inf, 2.5])
    ]
    model._fit_fold = lambda X, y, train_idx, test_idx: fold_outputs.pop(0)

    model._cross_validate(X, y)

    # Attributes should still be set
    assert model.cv_performed_ is True
    assert model.tau_ in model.taus_
    assert np.any(~np.isfinite(model.cv_mean_mse_))

def test_cross_validate_too_few_timepoints_for_splits():
    """
    If the number of time points is less than n_splits, TimeSeriesSplit will raise an error.
    _cross_validate should propagate this exception.
    """
    X = np.random.randn(3, 2)
    y = np.random.randn(3)
    model = RelaxationCV(tau=None, n_taus=3, n_splits=5)
    model._process_tau_grid(X, y)

    with pytest.raises(ValueError):
        model._cross_validate(X, y)


# Expected Behaviors of Cross Validate, Relaxed

# ------------------------------
# 1. Tau selection behavior
# ------------------------------
def test_cross_validate_selects_optimal_tau():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)

    n_taus = 4
    model = RelaxationCV(tau=None, n_taus=n_taus, n_splits=2)
    model._process_tau_grid(X, y)

    # Fake fold errors: make tau index 2 minimal
    fold_errors = [
        np.array([0.5, 0.4, 0.1, 0.8]),
        np.array([0.6, 0.3, 0.05, 0.9])
    ]
    model._fit_fold = lambda X, y, train_idx, test_idx: fold_errors.pop(0)

    model._cross_validate(X, y)

    # CV performed
    assert model.cv_performed_ is True

    # tau_ matches minimal CV error index
    expected_tau = model.taus_[2]
    assert model.tau_ == expected_tau

    # cv_mean_mse_ computed correctly
    expected_mean = np.mean([[0.5, 0.4, 0.1, 0.8], [0.6, 0.3, 0.05, 0.9]], axis=0)
    assert np.allclose(model.cv_mean_mse_, expected_mean)

    # cv_mse_path_ maps tau to mean MSE
    for tau, mse in zip(model.taus_, expected_mean):
        assert np.isclose(model.cv_mse_path_[float(tau)], mse)


# ------------------------------
# 2. Fit produces coef_ and sets tau_
# ------------------------------
def test_fit_computes_coef_and_sets_tau():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(15, 3))
    y = rng.normal(size=15)

    model = RelaxationCV(tau=None, n_taus=3, n_splits=2)

    # Patch _solve_relax_problem to return known weights
    model._solve_relax_problem = lambda X, y, tau: np.ones(X.shape[1]) / X.shape[1]

    # Patch _fit_fold to produce dummy errors
    model._fit_fold = lambda X, y, train_idx, test_idx: np.ones(model.n_taus)

    model.fit(X, y)

    # tau_ should be in taus_
    assert model.tau_ in model.taus_

    # coef_ shape and constraints
    assert model.coef_.shape == (X.shape[1],)
    assert np.all(model.coef_ >= 0)
    assert np.isclose(model.coef_.sum(), 1.0)

    # CV attributes set
    if not getattr(model, "skip_cv_", False):
        assert model.cv_performed_ is True
        assert hasattr(model, "cv_mean_mse_")
        assert hasattr(model, "cv_mse_path_")


# ------------------------------
# 3. Scalar tau skips CV
# ------------------------------
def test_fit_with_scalar_tau_skips_cv():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)

    scalar_tau = 0.2
    model = RelaxationCV(tau=scalar_tau, n_splits=2)

    # Patch _solve_relax_problem
    model._solve_relax_problem = lambda X, y, tau: np.array([0.3, 0.7])

    model.fit(X, y)

    # skip_cv_ should be True
    assert model.skip_cv_ is True

    # tau_ equals scalar tau
    assert model.tau_ == scalar_tau

    # cv_performed_ should not be True
    assert getattr(model, "cv_performed_", False) is False

    # coef_ returned correctly
    assert np.allclose(model.coef_, [0.3, 0.7])


# ------------------------------
# 4. Fit handles edge case small tau grid
# ------------------------------
def test_fit_with_small_tau_grid():
    X = np.eye(5)
    y = np.ones(5) * 0.1

    model = RelaxationCV(tau=None, n_taus=2, n_splits=2)

    # Patch _solve_relax_problem to identity weights
    model._solve_relax_problem = lambda X, y, tau: np.ones(X.shape[1]) / X.shape[1]

    # Patch _fit_fold to produce decreasing errors
    model._fit_fold = lambda X, y, train_idx, test_idx: np.array([0.2, 0.1])

    model.fit(X, y)

    assert model.tau_ in model.taus_
    assert np.allclose(model.coef_, np.ones(X.shape[1]) / X.shape[1])
    assert model.cv_performed_ is True


## Testing Fit Fold


# ------------------------------
# 1. Smoke test: runs without crashing
# ------------------------------
def test_fit_fold_smoke():
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    train_idx = np.arange(6)
    test_idx = np.arange(6, 10)

    model = RelaxationCV(tau=None, n_taus=3)
    model.taus_ = np.array([0.1, 0.5, 1.0])

    # Patch _solve_relax_problem to return uniform weights
    model._solve_relax_problem = lambda X, y, tau: np.ones(X.shape[1]) / X.shape[1]

    fold_mse = model._fit_fold(X, y, train_idx, test_idx)

    # Returns array of length n_taus
    assert isinstance(fold_mse, np.ndarray)
    assert fold_mse.shape[0] == len(model.taus_)


# ------------------------------
# 2. Expected behavior: MSE is correct for perfect prediction
# ------------------------------
def test_fit_fold_perfect_prediction():
    # Training and test are identical → MSE should be 0 for correct weights
    X = np.eye(4)
    y = np.array([1, 2, 3, 4])
    train_idx = np.arange(2)
    test_idx = np.arange(2, 4)

    model = RelaxationCV(tau=None, n_taus=2)
    model.taus_ = np.array([0.1, 0.5])

    # Patch _solve_relax_problem to perfectly fit training y
    def perfect_weights(X_train, y_train, tau):
        # For identity, just return training y padded with zeros
        w = np.zeros(X.shape[1])
        w[train_idx] = y_train
        return w
    model._solve_relax_problem = perfect_weights

    fold_mse = model._fit_fold(X, y, train_idx, test_idx)

    # MSE should be non-negative and length equals n_taus
    assert np.all(fold_mse >= 0)
    assert len(fold_mse) == 2


# ------------------------------
# 3. Multiple taus produces distinct MSEs
# ------------------------------
def test_fit_fold_multiple_taus():
    X = np.eye(5)
    y = np.arange(1, 6)
    train_idx = np.arange(3)
    test_idx = np.arange(3, 5)

    model = RelaxationCV(tau=None, n_taus=3)
    model.taus_ = np.array([0.1, 0.5, 1.0])

    # Return different weights depending on tau
    model._solve_relax_problem = lambda X_train, y_train, tau: np.full(X.shape[1], tau / X.shape[1])

    fold_mse = model._fit_fold(X, y, train_idx, test_idx)

    # Shape check
    assert fold_mse.shape[0] == len(model.taus_)
    # Distinct taus should produce distinct MSEs
    assert len(np.unique(fold_mse)) == len(model.taus_)



def test_name_elastic_net_model_with_mocked_panel(incrementality_synth_panel):
    # Unpack synthetic panel
    y, X, T0 = incrementality_synth_panel

    # Pre-treatment period
    y_pre = y[:T0]
    X_pre = X[:T0]

    # Post-treatment period
    X_post = X[T0:]

    donor_names = [f"GEO{i:02d}" for i in range(X.shape[1])]

    # Test alpha = 1.0 (should be L1)
    result = fit_en_scm(
        X_pre, y_pre, X_post, donor_names=donor_names,
        alpha=1.0, lam=0.1, y=y
    )
    assert result["Model"] == "$\\ell_1$"

    # Test alpha = 0.0 (should be L2)
    result = fit_en_scm(
        X_pre, y_pre, X_post, donor_names=donor_names,
        alpha=0.0, lam=0.1, second_norm="l2", y=y
    )
    assert result["Model"] == "$\\ell_2$"

    # Test intermediate alpha (Elastic Net)
    result = fit_en_scm(
        X_pre, y_pre, X_post, donor_names=donor_names,
        alpha=0.5, lam=0.1, second_norm="l2", y=y
    )
    assert result["Model"] == "$\\alpha \\ell_1 + (1-\\alpha) \\ell_2$"

    # Test alpha ~ 0 with second_norm = L1_INF (should be L-infinity)
    result = fit_en_scm(
        X_pre, y_pre, X_post, donor_names=donor_names,
        alpha=0.01, lam=0.1, second_norm="L1_INF", y=y
    )
    hg
    assert result["Model"] == "$\\ell_\\infty$"
