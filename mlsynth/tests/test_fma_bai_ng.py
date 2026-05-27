"""Tests for the FMA Bai & Ng factor-selection helpers.

Covers :mod:`mlsynth.utils.fma_helpers.bai_ng` (`nbpiid`, `standardize`,
`demean_matrix`), relocated from the former shared ``denoiseutils`` module.
"""

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.fma_helpers.bai_ng import nbpiid, standardize, demean_matrix


# ---------- Tests for standardize ----------
def test_standardize_matrix():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    X_std = standardize(X)
    assert X_std.shape == X.shape
    np.testing.assert_array_almost_equal(np.mean(X_std, axis=0), np.zeros(X.shape[1]), decimal=6)
    np.testing.assert_array_almost_equal(np.std(X_std, axis=0), np.ones(X.shape[1]), decimal=6)


def test_standardize_single_value_column():
    # Column with all same values -> std=0 -> division by zero guarded
    X = np.array([[1, 2], [1, 3], [1, 4]], dtype=float)
    with pytest.raises(MlsynthDataError, match=r"Cannot standardize columns with zero standard deviation: columns \[[^]]+\]"):
        standardize(X)


def test_standardize_non_array():
    with pytest.raises(MlsynthDataError, match="must be a NumPy array"):
        standardize([[1, 2], [3, 4]])  # type: ignore


def test_standardize_1d_array():
    Y = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    Y_std = standardize(Y)
    assert Y_std.shape == Y.shape
    assert np.mean(Y_std) == pytest.approx(0.0, abs=1e-9)
    assert np.std(Y_std) == pytest.approx(1.0, abs=1e-9)


def test_standardize_1d_zero_std():
    with pytest.raises(MlsynthDataError, match="zero standard deviation"):
        standardize(np.array([3.0, 3.0, 3.0]))


# ---------- Tests for demean_matrix ----------
def test_demean_matrix_valid():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    X_demeaned = demean_matrix(X)
    assert X_demeaned.shape == X.shape
    np.testing.assert_array_almost_equal(np.mean(X_demeaned, axis=0), np.zeros(X.shape[1]), decimal=6)


def test_demean_matrix_already_demeaned():
    X = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float)
    X_demeaned = demean_matrix(X)
    np.testing.assert_array_almost_equal(X_demeaned, X, decimal=6)


def test_demean_matrix_1d():
    Y = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(demean_matrix(Y), np.array([-10.0, 0.0, 10.0]))


def test_demean_matrix_invalid_input_type():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` must be a NumPy array."):
        demean_matrix([1, 2, 3])  # type: ignore


def test_demean_matrix_scalar_input():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` must be at least 1D."):
        demean_matrix(np.array(5))


def test_demean_matrix_empty_input():
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.array([]))
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.empty((0, 5)))
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.empty((5, 0)))


def test_demean_matrix_2d_zero_cols():
    # size 0 -> caught by the "cannot be empty" check
    with pytest.raises(MlsynthDataError, match="Input `input_matrix` cannot be empty."):
        demean_matrix(np.empty((3, 0)))


# ---------- Tests for nbpiid ----------
def test_nbpiid_smoke():
    rng = np.random.default_rng(0)
    T_obs, N_series = 20, 10
    x = rng.random((T_obs, N_series))
    kmax, jj, demean_flag = 5, 1, 1

    ic1_selected, chat, Fhat = nbpiid(x, kmax, jj, demean_flag)

    assert isinstance(ic1_selected, int)
    assert 0 <= ic1_selected <= kmax
    assert chat.shape == x.shape
    assert np.all(np.isfinite(chat))
    if ic1_selected > 0:
        assert Fhat.shape == (T_obs, ic1_selected)
        assert np.all(np.isfinite(Fhat))
    else:
        assert Fhat.shape == (T_obs, 0)


@pytest.mark.parametrize("jj_val", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("demean_val", [0, 1, 2])
def test_nbpiid_various_criteria_and_preprocessing(jj_val, demean_val):
    rng = np.random.default_rng(1)
    T_obs, N_series = 30, 15
    true_factors = rng.random((T_obs, 2))
    true_loadings = rng.random((2, N_series))
    x_structured = true_factors @ true_loadings + rng.random((T_obs, N_series)) * 0.1
    kmax = 4

    ic1_selected, chat, Fhat = nbpiid(x_structured, kmax, jj_val, demean_val)

    assert isinstance(ic1_selected, int)
    assert 0 <= ic1_selected <= kmax
    assert chat.shape == x_structured.shape
    assert Fhat.shape == (T_obs, ic1_selected) if ic1_selected > 0 else Fhat.shape == (T_obs, 0)


def test_nbpiid_jj10_and_jj11():
    rng = np.random.default_rng(2)
    T_obs, N_series = 20, 10
    x = rng.random((T_obs, N_series))
    kmax, demean_flag = 3, 2

    m_N, m_T = N_series // 2, T_obs // 2
    ic1_10, _, _ = nbpiid(x, kmax, 10, demean_flag, N_series_adjustment=m_N, T_obs_adjustment=m_T)
    assert isinstance(ic1_10, int) and 0 <= ic1_10 <= kmax

    ic1_11, _, _ = nbpiid(x, kmax, 11, demean_flag)
    assert isinstance(ic1_11, int) and 0 <= ic1_11 <= kmax


def test_nbpiid_jj10_missing_adjustments():
    x = np.random.default_rng(3).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`N_series_adjustment` must be an integer if criterion_selector_code is 10."):
        nbpiid(x, 3, 10, 0)


def test_nbpiid_jj10_missing_T_adjustment():
    x = np.random.default_rng(3).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`T_obs_adjustment` must be an integer if criterion_selector_code is 10."):
        nbpiid(x, 3, 10, 0, N_series_adjustment=5)


def test_nbpiid_invalid_criterion_code():
    x = np.random.default_rng(4).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`criterion_selector_code` must be one of .* got 99."):
        nbpiid(x, 3, 99, 0)


def test_nbpiid_criterion_code_not_int():
    x = np.random.default_rng(4).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`criterion_selector_code` must be an integer."):
        nbpiid(x, 3, 1.5, 0)  # type: ignore


def test_nbpiid_kmax_zero_or_negative():
    x = np.random.default_rng(5).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` must be positive."):
        nbpiid(x, 0, 1, 1)
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` must be positive."):
        nbpiid(x, -1, 1, 1)


def test_nbpiid_kmax_not_int():
    x = np.random.default_rng(5).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`max_factors_to_test` must be an integer."):
        nbpiid(x, 2.5, 1, 1)  # type: ignore


def test_nbpiid_kmax_too_large():
    x = np.random.default_rng(6).random((5, 10))
    with pytest.raises(MlsynthConfigError, match=r"`max_factors_to_test` .* cannot exceed min\(num_time_periods, num_series\) which is 5."):
        nbpiid(x, 6, 1, 1)


def test_nbpiid_invalid_input_panel_data():
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` must be a NumPy array."):
        nbpiid([1, 2, 3], 3, 1, 1)  # type: ignore
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` must be a 2D array."):
        nbpiid(np.array([1, 2, 3]), 3, 1, 1)
    with pytest.raises(MlsynthDataError, match="Input `input_panel_data` cannot be empty."):
        nbpiid(np.empty((5, 0)), 3, 1, 1)


def test_nbpiid_invalid_preprocessing_code():
    x = np.random.default_rng(7).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`preprocessing_method_code` must be one of .* got 5."):
        nbpiid(x, 3, 1, 5)


def test_nbpiid_preprocessing_code_not_int():
    x = np.random.default_rng(7).random((20, 10))
    with pytest.raises(MlsynthConfigError, match="`preprocessing_method_code` must be an integer."):
        nbpiid(x, 3, 1, 1.0)  # type: ignore


def test_nbpiid_preprocessing_failure_is_wrapped():
    """A zero-std column under standardize (code 2) is re-raised as estimation error."""
    x = np.random.default_rng(8).random((20, 10))
    x[:, 3] = 5.0  # constant column -> standardize raises -> wrapped
    with pytest.raises(MlsynthEstimationError, match="Data preprocessing failed"):
        nbpiid(x, 3, 1, 2)


def test_nbpiid_eigen_failure_is_wrapped():
    """Non-finite entries make the eigen decomposition fail; error is wrapped."""
    x = np.random.default_rng(9).random((20, 10))
    x[0, 0] = np.nan
    with pytest.raises(MlsynthEstimationError, match="Eigen decomposition failed"):
        nbpiid(x, 3, 1, 0)


def test_nbpiid_criterion11_invalid_T():
    x = np.random.default_rng(8).random((1, 10))
    with pytest.raises(MlsynthEstimationError, match=r"T must be > 1 for criterion 11 due to log\(T\)."):
        nbpiid(x, 1, 11, 0)

    x2 = np.random.default_rng(8).random((2, 10))
    with pytest.raises(MlsynthEstimationError, match=r"log\(T\) must be > 1 for criterion 11 .*"):
        nbpiid(x2, 1, 11, 0)
