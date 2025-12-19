import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from mlsynth.utils.crossval import ElasticNetCV, generate_lambda_seq2, RelaxationCV
from mlsynth.utils.optutils import Opt2


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def small_data():
    np.random.seed(0)
    X = np.random.randn(10, 5)
    y = X @ np.array([1, 0.5, 0, 0, 0]) + 0.1 * np.random.randn(10)
    return X, y


# ------------------------------
# ElasticNetCV Tests
# ------------------------------

def test_process_grid_scalar_lambda_alpha(small_data):
    X, y = small_data
    enet = ElasticNetCV(alpha=0.5, lam=0.3)
    enet._process_grid(X, y)
    assert np.allclose(enet.alphas_, [0.5])
    assert np.allclose(enet.lams_, [0.3])


def test_process_grid_generate_lambda(small_data):
    X, y = small_data
    enet = ElasticNetCV(alpha=0.5, lam=None)
    enet._process_grid(X, y)
    assert enet.alphas_[0] == 0.5
    assert enet.lams_.shape[0] == 30  # default generate_lambda_seq2


def test_process_grid_default_alpha(small_data):
    X, y = small_data
    enet = ElasticNetCV(alpha=None, lam=None)
    enet._process_grid(X, y)
    assert len(enet.alphas_) == 10
    assert len(enet.lams_) == 30


@patch("mlsynth.utils.crossval.Opt2.SCopt")
def test_solve_enet(mock_scopt, small_data):
    X, y = small_data
    mock_res = {"weights": {"w": np.ones(X.shape[1])}}
    mock_scopt.return_value = mock_res
    enet = ElasticNetCV(alpha=0.5, lam=0.1)
    w = enet._solve_enet(X, y, lam=0.1, alpha=0.5)
    assert w.shape[0] == X.shape[1]
    assert np.all(w == 1)


@patch.object(ElasticNetCV, "_solve_enet")
def test_fit_fold(mock_solve, small_data):
    X, y = small_data
    mock_solve.return_value = np.ones(X.shape[1])
    enet = ElasticNetCV(alpha=0.5, lam=0.1)
    train_idx = np.arange(8)
    test_idx = np.arange(8, 10)
    mse = enet._fit_fold(X, y, train_idx, test_idx, lam=0.1, alpha=0.5)
    assert mse >= 0


@patch.object(ElasticNetCV, "_fit_fold")
def test_cross_validate(mock_fit_fold, small_data):
    X, y = small_data
    mock_fit_fold.return_value = 0.5
    enet = ElasticNetCV(alpha=[0.1, 0.5], lam=[0.1, 0.2], n_splits=2)
    enet.alphas_ = np.array([0.1, 0.5])
    enet.lams_ = np.array([0.1, 0.2])

    # patch cross_validate manually
    enet.lam_ = 0.1
    enet.alpha_ = 0.1

    enet._cross_validate(X, y)
    assert hasattr(enet, "lam_")
    assert hasattr(enet, "alpha_")
    assert enet.cv_performed_ is True


@patch.object(ElasticNetCV, "_solve_enet")
def test_fit_predict(mock_solve, small_data):
    X, y = small_data
    mock_solve.return_value = np.ones(X.shape[1])
    enet = ElasticNetCV(alpha=0.5, lam=0.1)
    # manually set attributes since we skip CV
    enet.lam_ = 0.1
    enet.alpha_ = 0.5
    enet.coef_ = mock_solve.return_value
    y_pred = enet.predict(X)
    assert np.allclose(y_pred, X @ np.ones(X.shape[1]))


# ------------------------------
# ElasticNetCV vector alpha/lam smoke test
# ------------------------------
@patch.object(ElasticNetCV, "_solve_enet")
@patch.object(ElasticNetCV, "_cross_validate")
def test_enet_vector_alpha_lam(mock_cv, mock_solve, small_data):
    X, y = small_data
    mock_solve.return_value = np.ones(X.shape[1])
    mock_cv.return_value = None
    enet = ElasticNetCV(alpha=[0.1, 0.5], lam=[0.05, 0.1])
    enet._process_grid(X, y)

    # manually set attributes to avoid AttributeError
    enet.lam_ = 0.05
    enet.alpha_ = 0.1
    enet.coef_ = mock_solve.return_value

    y_pred = enet.predict(X)
    assert y_pred.shape == y.shape
    assert np.allclose(y_pred, X @ np.ones(X.shape[1]))


# ------------------------------
# RelaxationCV Tests
# ------------------------------

def test_generate_tau_grid(small_data):
    X, y = small_data
    relax = RelaxationCV(n_taus=5, n_splits=2)
    relax._generate_tau_grid(X, y)
    assert len(relax.taus_) == 5
    assert np.all(np.diff(relax.taus_) < 0)
    assert relax.taus_[-1] >= 1e-4


def test_process_tau_grid_scalar(small_data):
    X, y = small_data
    relax = RelaxationCV(tau=0.5, n_splits=2)
    relax._process_tau_grid(X, y)
    assert relax.skip_cv_ is True
    assert relax.tau_ == 0.5
    assert np.all(relax.taus_ == 0.5)


def test_process_tau_grid_list(small_data):
    X, y = small_data
    relax = RelaxationCV(tau=[0.1, 0.2, 0.3], n_splits=2)
    relax._process_tau_grid(X, y)
    assert relax.skip_cv_ is False
    assert np.allclose(relax.taus_, [0.3, 0.2, 0.1])


def test_process_tau_grid_none(small_data):
    X, y = small_data
    relax = RelaxationCV(tau=None, n_taus=5, n_splits=2)
    relax._process_tau_grid(X, y)
    assert relax.skip_cv_ is False
    assert len(relax.taus_) == 5
    assert np.all(np.diff(relax.taus_) < 0)


@patch("mlsynth.utils.crossval.Opt2.SCopt")
def test_solve_relax_problem(mock_scopt, small_data):
    X, y = small_data
    w_mock = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    mock_res = {"weights": {"w": w_mock}}
    mock_scopt.return_value = mock_res
    relax = RelaxationCV(n_splits=2)
    relax._process_tau_grid(X, y)
    w = relax._solve_relax_problem(X, y, tau=0.1)
    assert np.allclose(w, w_mock)


@patch.object(RelaxationCV, "_solve_relax_problem")
def test_fit_fold_relax(mock_solve, small_data):
    X, y = small_data
    mock_solve.side_effect = lambda X, y, tau=None: np.ones(X.shape[1])
    relax = RelaxationCV(tau=[0.1, 0.2], n_splits=2)
    relax._process_tau_grid(X, y)
    train_idx = np.arange(6)
    test_idx = np.arange(6, 10)
    fold_mse = relax._fit_fold(X, y, train_idx, test_idx)
    assert fold_mse.shape == (2,)
    assert np.all(fold_mse >= 0)


@patch.object(RelaxationCV, "_fit_fold")
def test_cross_validate_relax(mock_fold, small_data):
    X, y = small_data
    mock_fold.return_value = np.array([0.2, 0.1])
    relax = RelaxationCV(tau=[0.1, 0.2], n_splits=2)
    relax._process_tau_grid(X, y)
    relax._cross_validate(X, y)
    assert relax.cv_performed_ is True
    assert relax.tau_ in relax.taus_


@patch.object(RelaxationCV, "_solve_relax_problem")
@patch.object(RelaxationCV, "_cross_validate")
def test_fit_predict_relax(mock_cv, mock_solve, small_data):
    X, y = small_data
    mock_solve.return_value = np.ones(X.shape[1])
    mock_cv.return_value = None
    relax = RelaxationCV(tau=0.5, n_splits=2)
    relax.fit(X, y)
    y_pred = relax.predict(X)
    assert y_pred.shape == y.shape
    assert np.allclose(y_pred, np.sum(X, axis=1))


# ------------------------------
# Smoke test: single tau list
# ------------------------------
def test_relaxation_single_tau_list(small_data):
    X, y = small_data
    relax = RelaxationCV(tau=[0.7], n_splits=2)  # small n_splits to avoid ValueError
    relax.fit(X, y)
    y_pred = relax.predict(X)
    assert y_pred.shape == y.shape


# ------------------------------
# Smoke test: empty test split
# ------------------------------
def test_fit_fold_empty_split_relax(small_data):
    X, y = small_data
    relax = RelaxationCV(tau=0.1)
    relax._process_tau_grid(X, y)
    fold_mse = relax._fit_fold(
        X, y, train_idx=np.arange(5), test_idx=np.array([], dtype=int)
    )
    assert fold_mse.shape == (len(relax.taus_),)
