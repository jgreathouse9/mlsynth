import numpy as np
import pytest
from mlsynth.utils.bayesutils import BayesSCM
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError

def test_BayesSCM_smoke():
    """Smoke test for the BayesSCM function."""
    T0 = 10  # Number of pre-intervention periods
    N_donors = 5  # Number of donor units

    M_denoised = np.random.rand(T0, N_donors)
    Y_target_2d = np.random.rand(T0, 1)
    Y_target_1d = np.random.rand(T0)
    sigma2 = 0.1
    alpha = 1.0

    # Test with 2D Y_target
    beta_D, Sigma_D, Y_pred, Y_var = BayesSCM(M_denoised, Y_target_2d, sigma2, alpha)

    assert isinstance(beta_D, np.ndarray)
    assert beta_D.shape == (N_donors,)
    assert np.all(np.isfinite(beta_D))

    assert isinstance(Sigma_D, np.ndarray)
    assert Sigma_D.shape == (N_donors, N_donors)
    assert np.all(np.isfinite(Sigma_D))

    assert isinstance(Y_pred, np.ndarray)
    assert Y_pred.shape == (T0,)
    assert np.all(np.isfinite(Y_pred))

    assert isinstance(Y_var, np.ndarray)
    assert Y_var.shape == (T0,)
    assert np.all(np.isfinite(Y_var))
    assert np.all(Y_var > 0) # Variances should be positive

    # Test with 1D Y_target
    beta_D_1d, Sigma_D_1d, Y_pred_1d, Y_var_1d = BayesSCM(M_denoised, Y_target_1d, sigma2, alpha)

    assert isinstance(beta_D_1d, np.ndarray)
    assert beta_D_1d.shape == (N_donors,)
    assert np.all(np.isfinite(beta_D_1d))

    assert isinstance(Sigma_D_1d, np.ndarray)
    assert Sigma_D_1d.shape == (N_donors, N_donors)
    assert np.all(np.isfinite(Sigma_D_1d))

    assert isinstance(Y_pred_1d, np.ndarray)
    assert Y_pred_1d.shape == (T0,)
    assert np.all(np.isfinite(Y_pred_1d))

    assert isinstance(Y_var_1d, np.ndarray)
    assert Y_var_1d.shape == (T0,)
    assert np.all(np.isfinite(Y_var_1d))
    assert np.all(Y_var_1d > 0)

def test_BayesSCM_invalid_inputs():
    """Test BayesSCM with specific invalid parameter values."""
    T0 = 10
    N_donors = 5
    M_denoised = np.random.rand(T0, N_donors)
    Y_target = np.random.rand(T0)
    
    # observation_noise_variance == 0
    with pytest.raises(MlsynthDataError, match="observation_noise_variance must be positive."):
        BayesSCM(M_denoised, Y_target, 0.0, 1.0)
    
    # observation_noise_variance < 0
    with pytest.raises(MlsynthDataError, match="observation_noise_variance must be positive."):
        BayesSCM(M_denoised, Y_target, -0.1, 1.0)

    # weights_prior_precision < 0
    with pytest.raises(MlsynthDataError, match="weights_prior_precision must be non-negative."):
        BayesSCM(M_denoised, Y_target, 0.1, -1.0)

    # Test case for MlsynthEstimationError due to singular matrix
    # When weights_prior_precision is 0 and M.T @ M is singular.
    num_donors_for_sing_test = 2
    # M_s.T @ M_s will be [[1,1],[1,1]] (after scaling by 1/sigma2_s), which is singular.
    M_s = np.array([[1.0, 1.0]], dtype=float)  # Shape (1, 2)
    Y_s = np.array([1.0], dtype=float)      # Shape (1,)
    sigma2_s = 0.1
    alpha_s = 0.0  # weights_prior_precision = 0

    expected_error_pattern_singular = (
        r"Failed to compute posterior covariance\. Posterior precision matrix may be singular or not positive definite\. "
        r"Consider increasing weights_prior_precision or checking denoised_donor_matrix for collinearity\."
    )
    with pytest.raises(MlsynthEstimationError, match=expected_error_pattern_singular):
        BayesSCM(M_s, Y_s, sigma2_s, alpha_s)

    # Original test logic for rank deficient M, but alpha > 0 (should pass)
    # N_donors is 5 from the top of the function
    M_rank_deficient_passing = np.random.rand(N_donors - 1, N_donors) # Shape (4,5)
    Y_target_short_passing = np.random.rand(N_donors - 1)
    beta_D_passing, _, _, _ = BayesSCM(M_rank_deficient_passing, Y_target_short_passing, 0.1, 1.0) # alpha = 1.0
    assert beta_D_passing.shape == (N_donors,)


    # M_denoised with N_donors = 0 (This part is handled by new logic in BayesSCM)
    M_no_donors = np.empty((T0, 0))
    Y_target_no_donors = np.random.rand(T0)
    sigma2_val = 0.1
    alpha_val = 1.0
    
    beta_D_no_donors, Sigma_D_no_donors, Y_pred_no_donors, Y_var_no_donors = BayesSCM(
        M_no_donors, Y_target_no_donors, sigma2_val, alpha_val
    )

    assert beta_D_no_donors.shape == (0,)
    assert Sigma_D_no_donors.shape == (0, 0)
    assert Y_pred_no_donors.shape == (T0,)
    np.testing.assert_array_almost_equal(Y_pred_no_donors, np.zeros(T0))
    assert Y_var_no_donors.shape == (T0,)
    np.testing.assert_array_almost_equal(Y_var_no_donors, np.full(T0, sigma2_val))


def test_BayesSCM_detailed_invalid_inputs():
    """Test BayesSCM with various invalid input types, shapes, and content."""
    T0 = 10
    N_donors = 5
    valid_M = np.random.rand(T0, N_donors)
    valid_Y = np.random.rand(T0)
    valid_sigma2 = 0.1
    valid_alpha = 1.0

    # Invalid types
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix must be a NumPy array."):
        BayesSCM([[1,2],[3,4]], valid_Y, valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="target_outcome_pre_intervention must be a NumPy array."):
        BayesSCM(valid_M, [1,2,3], valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="observation_noise_variance must be a float or int."):
        BayesSCM(valid_M, valid_Y, "0.1", valid_alpha)
    with pytest.raises(MlsynthDataError, match="weights_prior_precision must be a float or int."):
        BayesSCM(valid_M, valid_Y, valid_sigma2, "1.0")

    # Invalid ndim for M_denoised
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix must be a 2D array."):
        BayesSCM(np.random.rand(T0), valid_Y, valid_sigma2, valid_alpha) # 1D M
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix must be a 2D array."):
        BayesSCM(np.random.rand(T0, N_donors, 1), valid_Y, valid_sigma2, valid_alpha) # 3D M

    # Invalid ndim for Y_target
    with pytest.raises(MlsynthDataError, match="target_outcome_pre_intervention must be a 1D or 2D array."):
        BayesSCM(valid_M, np.random.rand(T0, N_donors, 1), valid_sigma2, valid_alpha) # 3D Y
    with pytest.raises(MlsynthDataError, match="2D target_outcome_pre_intervention must have shape .*"):
        BayesSCM(valid_M, np.random.rand(T0, 2), valid_sigma2, valid_alpha) # Y with shape (T0, 2)

    # M_denoised with zero pre-intervention periods but donors exist
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix must have at least one pre-intervention period if donors exist."):
        BayesSCM(np.empty((0, N_donors)), np.empty(0), valid_sigma2, valid_alpha)
    
    # Test (0,0) M_denoised - should pass if Y is also (0,)
    BayesSCM(np.empty((0,0)), np.empty(0), valid_sigma2, valid_alpha)


    # Shape mismatch between M and Y
    with pytest.raises(MlsynthDataError, match="Number of pre-intervention periods in denoised_donor_matrix .* must match .*"):
        BayesSCM(valid_M, np.random.rand(T0 - 1), valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="Number of pre-intervention periods in denoised_donor_matrix .* must match .*"):
        BayesSCM(np.random.rand(T0 -1, N_donors), valid_Y, valid_sigma2, valid_alpha)

    # NaN/Inf values
    M_with_nan = valid_M.copy(); M_with_nan[0,0] = np.nan
    Y_with_nan = valid_Y.copy(); Y_with_nan[0] = np.nan
    M_with_inf = valid_M.copy(); M_with_inf[0,0] = np.inf
    Y_with_inf = valid_Y.copy(); Y_with_inf[0] = np.inf

    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix contains NaN or Inf values."):
        BayesSCM(M_with_nan, valid_Y, valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix contains NaN or Inf values."):
        BayesSCM(M_with_inf, valid_Y, valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="target_outcome_pre_intervention contains NaN or Inf values."):
        BayesSCM(valid_M, Y_with_nan, valid_sigma2, valid_alpha)
    with pytest.raises(MlsynthDataError, match="target_outcome_pre_intervention contains NaN or Inf values."):
        BayesSCM(valid_M, Y_with_inf, valid_sigma2, valid_alpha)

    # Test MlsynthEstimationError for non-finite matrix_to_invert (e.g. if M contains inf)
    # This should be caught by input validation on M_denoised first.
    M_inf_for_inversion = np.array([[1.0, 2.0], [3.0, np.inf]])
    Y_for_inf_inversion = np.array([1.0, 2.0])
    with pytest.raises(MlsynthDataError, match="denoised_donor_matrix contains NaN or Inf values."):
        BayesSCM(M_inf_for_inversion, Y_for_inf_inversion, valid_sigma2, valid_alpha)
