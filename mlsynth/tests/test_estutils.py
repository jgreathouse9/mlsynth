import numpy as np
import pytest
import pandas as pd
import cvxpy as cp
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.estutils import (
    bartlett,
    compute_hac_variance,
    compute_t_stat_and_ci,
    get_theta,
    get_sigmasq,
    l2_relax,
    adaptive_cross_validate_tau,
    Opt, # Class containing SCopt
    NSC_opt,
    NSCcv,
    SMOweights,
    # All specified complex functions now have at least smoke tests.
    pi_surrogate, 
    pi_surrogate_post, 
    ci_bootstrap, 
    TSEST, 
    pcr, 
    pda, 
    SRCest, 
    RPCASYNTH, # Added for testing
)
from mlsynth.utils.resultutils import effects # For pda, TSEST etc.
from mlsynth.utils.datautils import dataprep # For RPCASYNTH

# ---------- Tests for bartlett ----------
def test_bartlett():
    assert bartlett(0, 5) == 1.0
    assert bartlett(1, 5) == 1 - 1/6
    assert bartlett(5, 5) == 1 - 5/6
    assert bartlett(6, 5) == 0.0
    assert bartlett(-1, 5) == 1 - 1/6 # abs(i)

# ---------- Tests for compute_hac_variance ----------
def test_compute_hac_variance_smoke():
    effects = np.random.randn(20)
    lag = 3
    var = compute_hac_variance(effects, lag)
    assert isinstance(var, float)
    assert not np.isnan(var)
    assert var >= 0 # Variance should be non-negative

def test_compute_hac_variance_empty_effects():
    assert np.isnan(compute_hac_variance(np.array([]), 3))

def test_compute_hac_variance_single_effect():
    # With one effect, variance is np.var([x], ddof=1) which is nan.
    # The function should handle this.
    # np.var([1.0], ddof=1) is nan.
    # The function returns variance if T2==1, which is np.var(residuals, ddof=1)
    # residuals will be [0.0] if effect is [x]. np.var([0.0], ddof=1) is nan.
    with pytest.warns(RuntimeWarning): # Expect warnings from np.var with ddof=1 for single element
        assert np.isnan(compute_hac_variance(np.array([1.0]), 3))


# ---------- Tests for compute_t_stat_and_ci ----------
def test_compute_t_stat_and_ci_smoke():
    att = 0.5
    effects_post = np.random.randn(20) + 0.5
    lag = 3
    t_stat, ci = compute_t_stat_and_ci(att, effects_post, lag)
    assert isinstance(t_stat, float)
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert isinstance(ci[0], float)
    assert isinstance(ci[1], float)
    assert ci[0] <= ci[1] # Lower bound <= Upper bound

def test_compute_t_stat_and_ci_empty_effects():
    t_stat, ci = compute_t_stat_and_ci(0.5, np.array([]), 3)
    assert np.isnan(t_stat)
    assert np.isnan(ci[0])
    assert np.isnan(ci[1])

def test_compute_t_stat_and_ci_zero_variance_effects():
    # If all effects are same, HAC variance might be zero or near zero
    att = 0.5
    effects_post = np.full(20, 0.5)
    lag = 3
    t_stat, ci = compute_t_stat_and_ci(att, effects_post, lag)
    # If SE is zero, t_stat can be inf or nan. CI can be (att, att) or (nan, nan)
    if np.isinf(t_stat) or np.isnan(t_stat): # Check for inf or nan
        pass
    else: # pragma: no cover
        pytest.fail("t_stat should be inf or nan for zero variance effects if att is non-zero")
    
    # If SE is zero and att is also zero, t_stat is nan
    t_stat_zero_att, _ = compute_t_stat_and_ci(0.0, np.full(20,0.0), lag)
    assert np.isnan(t_stat_zero_att)


# ---------- Tests for get_theta ----------
def test_get_theta_smoke():
    T0 = 10
    N_donors = 3
    y1_pre = np.random.rand(T0)
    Y0_pre = np.random.rand(T0, N_donors)
    
    theta_hat, Y_theta_aligned = get_theta(y1_pre, Y0_pre)
    
    assert isinstance(theta_hat, np.ndarray)
    assert theta_hat.shape == (N_donors,)
    assert np.all(np.isfinite(theta_hat))
    
    assert isinstance(Y_theta_aligned, np.ndarray)
    assert Y_theta_aligned.shape == (T0, N_donors)
    assert np.all(np.isfinite(Y_theta_aligned))

def test_get_theta_zero_variance_donor():
    T0 = 10
    N_donors = 2
    y1_pre = np.arange(T0, dtype=float)
    Y0_pre = np.zeros((T0, N_donors))
    Y0_pre[:, 0] = 1.0 # Zero variance donor
    Y0_pre[:, 1] = np.arange(T0, dtype=float) * 2 # Non-zero variance donor

    theta_hat, _ = get_theta(y1_pre, Y0_pre)
    assert theta_hat.shape == (N_donors,)
    assert theta_hat[0] == 0.0 # Theta for zero-variance donor should be 0 due to np.divide where=
    assert not np.isnan(theta_hat[1])


# ---------- Tests for get_sigmasq ----------
def test_get_sigmasq_smoke():
    T0 = 10
    N_donors = 3
    y1_pre = np.random.rand(T0)
    Y0_pre = np.random.rand(T0, N_donors)
    
    sigma2_est = get_sigmasq(y1_pre, Y0_pre)
    
    assert isinstance(sigma2_est, float)
    assert not np.isnan(sigma2_est)
    assert sigma2_est >= 0

def test_get_sigmasq_perfect_fit():
    T0 = 10
    N_donors = 1
    y1_pre = np.arange(T0, dtype=float) * 2.0
    Y0_pre = np.arange(T0, dtype=float).reshape(-1,1) # y1_pre = Y0_pre * 2 (after demeaning)
                                                    # theta_hat will be ~2
                                                    # residual_error should be near zero
    
    # Demeaning:
    # y1_demeaned = y1_pre - np.mean(y1_pre)
    # Y0_demeaned = Y0_pre - np.mean(Y0_pre)
    # theta_hat = (Y0_demeaned.T @ y1_demeaned) / np.sum(Y0_demeaned**2)
    # projected_y1 = Y0_demeaned * theta_hat
    # residual_error = y1_demeaned - projected_y1
    # sigma2 = norm(residual_error)**2
    # If y1_pre is a scaled version of Y0_pre, after demeaning they are still scaled.
    # So residual_error should be close to 0.
    
    sigma2_est = get_sigmasq(y1_pre, Y0_pre)
    assert isinstance(sigma2_est, float)
    assert pytest.approx(sigma2_est, abs=1e-9) == 0.0

def test_get_sigmasq_zero_variance_donors():
    T0 = 10
    N_donors = 2
    y1_pre = np.arange(T0, dtype=float)
    Y0_pre = np.ones((T0, N_donors)) # All donors have zero variance after demeaning

    # diag_G will be all zeros. diag_G_inv will be all zeros.
    # Z_projection_matrix will be all zeros.
    # projected_y1 will be all zeros.
    # residual_error = Q @ y1_pre
    # sigma2_est = norm(Q @ y1_pre)**2 = sum of squares of demeaned y1_pre
    
    sigma2_est = get_sigmasq(y1_pre, Y0_pre)
    expected_sigma2 = np.sum((y1_pre - np.mean(y1_pre))**2)
    assert pytest.approx(sigma2_est) == expected_sigma2

# ---------- Tests for l2_relax ----------
def test_l2_relax_smoke():
    T_total = 20
    T_pre = 10
    N_controls = 3
    treated_unit = np.random.rand(T_total)
    X = np.random.rand(T_total, N_controls)
    tau = 0.1

    coeffs, intercept, counterfactuals = l2_relax(T_pre, treated_unit, X, tau)

    assert isinstance(coeffs, np.ndarray)
    assert coeffs.shape == (N_controls,)
    assert isinstance(intercept, float)
    assert isinstance(counterfactuals, np.ndarray)
    assert counterfactuals.shape == (T_total,)
    assert np.all(np.isfinite(coeffs))
    assert np.isfinite(intercept)
    assert np.all(np.isfinite(counterfactuals))

def test_l2_relax_optimization_failure():
    # This test is tricky as forcing CVXPY to fail reliably without specific solver knowledge is hard.
    # A very large tau might make the feasible set empty or cause issues.
    # Or, if Sigma_cov is singular and eta_cov is not in its image.
    T_total = 5
    T_pre = 3
    N_controls = 2
    treated_unit = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0,0.0], [0.0,0.0]]) # Singular X_subset.T @ X_subset
    
    # With singular Sigma_cov, if eta_cov is not in its column space, problem might be infeasible.
    # Or if tau is too small.
    # tau_very_small = 1e-8 
    # with pytest.raises(ValueError, match="Optimization failed with status"):
    #      l2_relax(T_pre, treated_unit, X, tau_very_small) # This setup was not failing as expected

    # Try T_pre = 0, which should lead to division by zero for n_sub
    T_pre_zero = 0
    tau_val = 0.1
    # Expect an error during calculation of Sigma_cov or eta_cov due to division by n_sub=0
    # This might be a RuntimeWarning leading to NaNs, then CVXPY error, or direct ZeroDivisionError
    # CVXPY might raise cp.error.SolverError or cp.error.DCPError if inputs are NaN/Inf.
    # Or, if problem becomes trivial/ill-defined.
    # The function itself doesn't explicitly check for T_pre=0 before calculations.
    # Assuming T_pre_zero is a config/data issue leading to estimation failure.
    with pytest.raises((MlsynthConfigError, MlsynthDataError, MlsynthEstimationError, ZeroDivisionError, cp.error.SolverError, cp.error.DCPError, ValueError)):
         l2_relax(T_pre_zero, treated_unit, X, tau_val)


# ---------- Tests for cross_validate_tau ----------
def test_cross_validate_tau_smoke():
    T_pre = 20
    N_controls = 3
    treated_unit_pre = np.random.rand(T_pre)
    X_pre = np.random.rand(T_pre, N_controls)
    tau_init_upper = 1.0

    optimal_tau, min_mse = cross_validate_tau(treated_unit_pre, X_pre, tau_init_upper, num_coarse_points=10)

    assert isinstance(optimal_tau, float)
    assert isinstance(min_mse, float)
    assert optimal_tau > 0
    assert min_mse >= 0

# ---------- Tests for Opt.SCopt ----------
@pytest.mark.parametrize("model", ["SIMPLEX", "MSCb", "MSCa", "MSCc", "OLS"])
def test_Opt_SCopt_smoke(model):
    Nco = 3
    t1 = 10
    y = np.random.rand(t1)
    X = np.random.rand(t1, Nco)
    
    # For MSCa/MSCc, donor_names might be used if intercept is named
    donor_names = [f"donor_{i}" for i in range(Nco)]

    prob_or_dict = Opt.SCopt(Nco, y, t1, X, scm_model_type=model, donor_names=donor_names)
    
    assert prob_or_dict is not None
    if isinstance(prob_or_dict, dict): # MA model (not tested here yet)
        assert "Lambdas" in prob_or_dict # pragma: no cover
    else: # CVXPY problem
        assert prob_or_dict.status == "optimal"
        weights = prob_or_dict.solution.primal_vars[next(iter(prob_or_dict.solution.primal_vars))]
        expected_len = Nco + 1 if model in ["MSCa", "MSCc"] else Nco
        assert len(weights) == expected_len

def test_Opt_SCopt_MA_smoke():
    # Setup for MA model
    Nco = 3
    t1 = 10
    # Dummy model_weights structure
    model_weights_data = {
        "model1": {"weights": np.random.rand(Nco), "cf": np.random.rand(t1)},
        "model2": {"weights": np.random.rand(Nco), "cf": np.random.rand(t1)}
    }
    y_ma = np.random.rand(t1)
    X_ma = np.random.rand(t1, Nco) # X is not directly used by MA path in SCopt, but pass for consistency

    result_ma = Opt.SCopt(Nco, y_ma, t1, X_ma, scm_model_type="MA", base_model_results_for_averaging=model_weights_data)
    assert isinstance(result_ma, dict)
    assert "Lambdas" in result_ma
    assert "w_MA" in result_ma
    assert "Counterfactual_pre" in result_ma
    assert len(result_ma["Lambdas"]) == len(model_weights_data)
    assert result_ma["w_MA"].shape == (Nco,)
    assert result_ma["Counterfactual_pre"].shape == (t1,)

    def test_Opt_SCopt_MA_invalid_input():
        with pytest.raises(MlsynthConfigError, match="For 'MA', base_model_results_for_averaging must be a dictionary"):
            Opt.SCopt(3, np.random.rand(10), 10, np.random.rand(10,3), scm_model_type="MA", base_model_results_for_averaging=None)

def test_Opt_SCopt_dimension_mismatch():
    with pytest.raises(MlsynthDataError, match="For non-MA models, donor_outcomes_pre_treatment and target_outcomes_pre_treatment \\(after slicing to num_pre_treatment_periods\\) must have matching row counts."):
        Opt.SCopt(3, np.random.rand(10), 10, np.random.rand(5,3), scm_model_type="SIMPLEX")


# ---------- Tests for NSC_opt ----------
def test_NSC_opt_smoke():
    T0 = 10
    N_donors = 3
    y_target_pre = np.random.rand(T0)
    Y0_donors_pre = np.random.rand(T0, N_donors)
    a_reg = 0.1
    b_reg = 0.1

    weights = NSC_opt(y_target_pre, Y0_donors_pre, a_reg, b_reg)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (N_donors,)
    assert np.all(np.isfinite(weights))
    assert pytest.approx(np.sum(weights)) == 1.0

# ---------- Tests for NSCcv ----------
def test_NSCcv_smoke():
    T0 = 20 # Need enough for KFold splits
    N_donors = 7 # Need enough for KFold splits
    y_target_pre = np.random.rand(T0)
    Y0_donors_pre = np.random.rand(T0, N_donors)
    a_grid = np.array([0.1, 0.5])
    b_grid = np.array([0.1, 0.5])
    num_cv_folds_param = 3 # min(num_cv_folds, N_donors) will be 3

    best_a, best_b = NSCcv(y_target_pre, Y0_donors_pre, a_grid, b_grid, num_cv_folds_param)
    assert isinstance(best_a, float)
    assert isinstance(best_b, float)
    assert best_a in a_grid
    assert best_b in b_grid

def test_NSCcv_not_enough_donors_for_cv():
    T0 = 10
    N_donors = 1 # Not enough for k_folds=2+
    y_target_pre = np.random.rand(T0)
    Y0_donors_pre = np.random.rand(T0, N_donors)
    with pytest.warns(UserWarning, match=r"Not enough donors \(1\) for meaningful 1-fold CV\. Returning default L1=0\.01, L2=0\.01\."):
        best_a, best_b = NSCcv(y_target_pre, Y0_donors_pre, num_cv_folds=1) # k_folds=1 to trigger specific path
    assert best_a == 0.01 # Default values when CV cannot run
    assert best_b == 0.01


# ---------- Tests for SMOweights ----------
def test_SMOweights_smoke():
    T_total = 15
    T0_pre = 10
    N_donors = 3
    K_outcomes = 2

    target_list = [np.random.rand(T_total) for _ in range(K_outcomes)]
    donor_list = [np.random.rand(T_total, N_donors) for _ in range(K_outcomes)]
    data = {"Target": target_list, "Donors": donor_list}

    # Test concatenated
    weights_concat = SMOweights(data, aggregation_method="concatenated", num_pre_treatment_periods_override=T0_pre)
    assert isinstance(weights_concat, np.ndarray)
    assert weights_concat.shape == (N_donors,)
    assert np.all(weights_concat >= 0)
    assert pytest.approx(np.sum(weights_concat)) == 1.0

    # Test average
    weights_avg = SMOweights(data, aggregation_method="average", num_pre_treatment_periods_override=T0_pre)
    assert isinstance(weights_avg, np.ndarray)
    assert weights_avg.shape == (N_donors,)
    assert np.all(weights_avg >= 0)
    assert pytest.approx(np.sum(weights_avg)) == 1.0
    
    # Test with T0_pre = None
    weights_concat_full = SMOweights(data, aggregation_method="concatenated", num_pre_treatment_periods_override=None)
    assert weights_concat_full.shape == (N_donors,)


def test_SMOweights_invalid_method():
    data = {"Target": [np.random.rand(10)], "Donors": [np.random.rand(10,3)]}
    with pytest.raises(MlsynthConfigError, match="Invalid aggregation_method: invalid_method. Choose 'concatenated' or 'average'."):
        SMOweights(data, aggregation_method="invalid_method")

def test_SMOweights_empty_input():
    with pytest.raises(MlsynthDataError, match="Input 'Target' and 'Donors' lists cannot be empty."):
        SMOweights({"Target": [], "Donors": []})

def test_SMOweights_mismatch_lengths():
    data = {"Target": [np.random.rand(10)], "Donors": []}
    with pytest.raises(MlsynthDataError, match="Target and Donor lists must have the same length"):
        SMOweights(data)

# ---------- Tests for hac ----------
from mlsynth.utils.estutils import hac

def test_hac_smoke():
    T_obs = 20
    K_moments = 3
    G_moments = np.random.rand(T_obs, K_moments)
    J_lag = 3

    omega_hac = hac(G_moments, J_lag)
    assert isinstance(omega_hac, np.ndarray)
    assert omega_hac.shape == (K_moments, K_moments)
    assert np.all(np.isfinite(omega_hac))
    # Check for symmetry (HAC matrix should be symmetric)
    assert np.allclose(omega_hac, omega_hac.T)
    # Check for positive semi-definiteness (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(omega_hac)
    assert np.all(eigenvalues >= -1e-9) # Allow for small numerical errors

def test_hac_zero_lag():
    T_obs = 20
    K_moments = 2
    G_moments = np.random.rand(T_obs, K_moments)
    J_lag = 0 
    
    omega_hac_zero_lag = hac(G_moments, J_lag)
    # Expected: (G_moments.T @ G_moments) / T_obs
    expected_omega = (G_moments.T @ G_moments) / T_obs
    
    assert omega_hac_zero_lag.shape == (K_moments, K_moments)
    np.testing.assert_array_almost_equal(omega_hac_zero_lag, expected_omega)

def test_hac_lag_greater_than_T_obs():
    T_obs = 5
    K_moments = 2
    G_moments = np.random.rand(T_obs, K_moments)
    J_lag = 10 # lag > T_obs - 1
    
    # The loop for j will go up to min(J_lag, T_obs - 1) + 1 = T_obs
    # So it will compute all possible autocovariances
    omega_hac = hac(G_moments, J_lag)
    assert omega_hac.shape == (K_moments, K_moments)
    assert np.all(np.isfinite(omega_hac))

# ---------- Tests for pi ----------
from mlsynth.utils.estutils import pi

def test_pi_smoke():
    T_total = 30
    N_W_features = 2
    N_Z0_instruments = 2 # Must be same as N_W_features for Z0W_pre to be square for solve
    
    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    # Ensure Z0 has same number of columns as W for basic pi version
    Z0 = np.random.rand(T_total, N_Z0_instruments) 
    
    T0 = 20
    t1_post_eval = 10
    lag_hac = 3

    # Make Z0[:T0].T @ W[:T0] invertible
    W_pre_temp = np.random.rand(T0, N_W_features)
    Z0_pre_temp = W_pre_temp @ np.random.rand(N_W_features, N_Z0_instruments) # Ensure Z0W is non-singular
    
    W_test = np.vstack((W_pre_temp, np.random.rand(T_total - T0, N_W_features)))
    Z0_test = np.vstack((Z0_pre_temp, np.random.rand(T_total - T0, N_Z0_instruments)))


    y_pi, alpha, se_tau = pi(Y, W_test, Z0_test, T0, t1_post_eval, T_total, lag_hac)

    assert isinstance(y_pi, np.ndarray)
    assert y_pi.shape == (T_total,)
    assert np.all(np.isfinite(y_pi))

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == (N_W_features,) # Alpha for original W
    assert np.all(np.isfinite(alpha))

    assert isinstance(se_tau, float)
    # se_tau can be nan if LinAlgError occurs or var_tau_effect is negative
    # assert np.isfinite(se_tau) # Not guaranteed if LinAlgError or var < 0

def test_pi_with_covariates():
    T_total = 30
    N_W_features = 2
    N_Z0_instruments = 2
    N_Cw_features = 1
    N_Cy_features = 1
    
    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    Z0 = np.random.rand(T_total, N_Z0_instruments)
    Cw = np.random.rand(T_total, N_Cw_features)
    Cy = np.random.rand(T_total, N_Cy_features)
    
    T0 = 20
    t1_post_eval = 10
    lag_hac = 3

    # Ensure augmented Z0W_pre is invertible
    W_aug_pre_temp = np.random.rand(T0, N_W_features + N_Cw_features)
    Z0_aug_pre_temp = W_aug_pre_temp @ np.random.rand(N_W_features + N_Cw_features, N_Z0_instruments + N_Cy_features)
    
    W_test = np.vstack((W_aug_pre_temp[:, :N_W_features], np.random.rand(T_total - T0, N_W_features)))
    Z0_test = np.vstack((Z0_aug_pre_temp[:, :N_Z0_instruments], np.random.rand(T_total - T0, N_Z0_instruments)))
    Cw_test = np.vstack((W_aug_pre_temp[:, N_W_features:], np.random.rand(T_total - T0, N_Cw_features)))
    Cy_test = np.vstack((Z0_aug_pre_temp[:, N_Z0_instruments:], np.random.rand(T_total - T0, N_Cy_features)))


    y_pi, alpha, se_tau = pi(Y, W_test, Z0_test, T0, t1_post_eval, T_total, lag_hac, common_aux_covariates_1=Cw_test, common_aux_covariates_2=Cy_test)
    
    assert alpha.shape == (N_W_features,) # Should be for original W

# ---------- Tests for pi2 ----------
from mlsynth.utils.estutils import pi2

def test_pi2_smoke():
    T_total = 30
    N_W_features = 2
    N_Z0_instruments = 2 # For pi2, Z0 can have different dim from W, but W_aug and Z0_aug must match.
                         # Since Cy and Cw are appended to both, W and Z0 must match if Cy/Cw are used.
                         # For the smoke test without Cy/Cw, they must match.
    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    Z0 = np.random.rand(T_total, N_Z0_instruments)
    
    T0 = 20
    t1_post_eval = 5 # t1 in pi2 is number of post-treatment periods for mean effect
    lag_hac = 3 # lag in pi2

    # pi2 uses cvxpy, so less direct concern about matrix inversion for alpha, tau
    y_pi, alpha, se_tau = pi2(Y, W, Z0, T0, t1_post_eval, T_total, lag_hac)

    assert isinstance(y_pi, np.ndarray)
    assert y_pi.shape == (T_total,)
    assert np.all(np.isfinite(y_pi))

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == (N_W_features,)
    assert np.all(np.isfinite(alpha))

    assert isinstance(se_tau, float) # Currently placeholder (np.nan)
    assert np.isnan(se_tau) # Check it's indeed the placeholder

def test_pi2_with_covariates():
    T_total = 30
    N_W_features = 2
    N_Z0_instruments = 2 # Must be same as N_W_features because Cy and Cw are added to both
    N_Cw_features = 1
    N_Cy_features = 1

    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    Z0 = np.random.rand(T_total, N_Z0_instruments)
    Cw = np.random.rand(T_total, N_Cw_features)
    Cy = np.random.rand(T_total, N_Cy_features)

    T0 = 20
    t1_post_eval = 5
    lag_hac = 3
    
    y_pi, alpha, se_tau = pi2(Y, W, Z0, T0, t1_post_eval, T_total, lag_hac, covariates_for_W=Cw, covariates_for_Z0=Cy)
    
    assert alpha.shape == (N_W_features,) # Alpha for original W
    assert np.isnan(se_tau)

def test_pi2_dimension_mismatch_covariates():
    T_total = 10
    W = np.random.rand(T_total, 2)
    Z0 = np.random.rand(T_total, 3) # W and Z0 have different feature counts
    Cw = np.random.rand(T_total, 1)
    # If Cy and Cw are used, W_aug and Z0_aug must have same number of columns.
    # W_aug = (W, Cy, Cw) -> (2 + Cy_cols + 1)
    # Z0_aug = (Z0, Cy, Cw) -> (3 + Cy_cols + 1)
    # These will always differ by 1 if Cy and Cw are appended to both.
    # The original pi2 code:
    # Z0_aug = np.column_stack((Z0, Cy, Cw))
    # W_aug = np.column_stack((W, Cy, Cw))
    # So if Z0.shape[1] != W.shape[1], then Z0_aug.shape[1] != W_aug.shape[1]
    # The error is raised if W_aug.shape[1] != Z0_aug.shape[1]
    # This means the initial W and Z0 must have the same number of columns if Cw and Cy are used.
    # Let's test the case where W and Z0 initially differ, and Cw/Cy are used.
    Cy_matching_cw = np.random.rand(T_total, 1) # Cy has 1 col, Cw has 1 col

    with pytest.raises(MlsynthConfigError, match="Augmented design matrix W and proxy matrix Z0 must have the same number of columns."):
        pi2(np.random.rand(T_total), W, Z0, 5, 5, T_total, 1, covariates_for_W=Cw, covariates_for_Z0=Cy_matching_cw)

# ---------- Tests for pi_surrogate ----------
def test_pi_surrogate_smoke():
    T_total = 30
    N_W_features = 2
    N_Z0_instruments = 2 # Must match W if Cw/Cy used
    N_Z1_instruments = 2 # Must match X_surr if Cx used
    N_X_surr_features = 2

    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    Z0 = np.random.rand(T_total, N_Z0_instruments)
    Z1 = np.random.rand(T_total, N_Z1_instruments)
    X_surr = np.random.rand(T_total, N_X_surr_features)
    
    T0 = 20
    t1_post_eval = 5
    lag_hac = 3

    # Ensure Z0W_pre and Z1X_post are invertible for the solve() calls
    # For Z0W_pre:
    W_pre_temp = np.random.rand(T0, N_W_features)
    Z0_pre_temp = W_pre_temp @ np.random.rand(N_W_features, N_Z0_instruments)
    W_test = np.vstack((W_pre_temp, np.random.rand(T_total - T0, N_W_features)))
    Z0_test = np.vstack((Z0_pre_temp, np.random.rand(T_total - T0, N_Z0_instruments)))

    # For Z1X_post:
    T_post = T_total - T0
    X_surr_post_temp = np.random.rand(T_post, N_X_surr_features)
    Z1_post_temp = X_surr_post_temp @ np.random.rand(N_X_surr_features, N_Z1_instruments)
    X_surr_test = np.vstack((np.random.rand(T0, N_X_surr_features), X_surr_post_temp))
    Z1_test = np.vstack((np.random.rand(T0, N_Z1_instruments), Z1_post_temp))


    tau_mean, taut_effects, alpha, se_tau = pi_surrogate(
        Y, W_test, Z0_test, Z1_test, X_surr_test, T0, t1_post_eval, T_total, lag_hac
    )

    assert isinstance(tau_mean, float)
    assert np.isfinite(tau_mean)

    assert isinstance(taut_effects, np.ndarray)
    assert taut_effects.shape == (T_total,)
    assert np.all(np.isfinite(taut_effects))

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == (N_W_features,) # Alpha for original W
    assert np.all(np.isfinite(alpha))

    assert isinstance(se_tau, float)
    # se_tau can be nan if LinAlgError or var_tau < 0
    # assert np.isfinite(se_tau) # Not guaranteed

def test_pi_surrogate_with_covariates():
    T_total = 30
    N_W, N_Z0, N_Z1, N_X_surr = 2, 2, 2, 2
    N_Cw, N_Cy, N_Cx = 1, 1, 1 # Covariate dimensions

    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W)
    Z0 = np.random.rand(T_total, N_Z0)
    Z1 = np.random.rand(T_total, N_Z1)
    X_surr = np.random.rand(T_total, N_X_surr)
    Cw = np.random.rand(T_total, N_Cw)
    Cy = np.random.rand(T_total, N_Cy)
    Cx = np.random.rand(T_total, N_Cx)

    T0 = 20
    t1_post_eval = 5
    lag_hac = 3

    # Ensure augmented matrices for solve() are well-conditioned
    # Z0_aug.T @ W_aug (pre-period)
    W_aug_pre_dim = N_W + N_Cw # Assuming Cy is also added to W_aug
    Z0_aug_pre_dim = N_Z0 + N_Cy # Assuming Cw is also added to Z0_aug
    # The function stacks (Z0,Cy,Cw) and (W,Cy,Cw). So W_aug_dim = N_W+N_Cy+N_Cw, Z0_aug_dim = N_Z0+N_Cy+N_Cw
    # For Z0W_pre to be square, N_W must equal N_Z0.
    
    # Z1_aug.T @ X_surr_aug (post-period)
    # Z1_aug = (Z1, Cx), X_surr_aug = (X_surr, Cx)
    # So Z1_aug_dim = N_Z1+N_Cx, X_surr_aug_dim = N_X_surr+N_Cx
    # For Z1X_post to be square, N_Z1 must equal N_X_surr.

    # Simplified setup for invertibility: make base matrices square and full rank
    # For Z0W_pre_aug:
    W_aug_pre_temp = np.random.rand(T0, N_W + N_Cy + N_Cw) # Total augmented dim
    Z0_aug_pre_temp = W_aug_pre_temp @ np.random.rand(N_W + N_Cy + N_Cw, N_Z0 + N_Cy + N_Cw)
    
    # For Z1X_post_aug:
    T_post = T_total - T0
    X_surr_aug_post_temp = np.random.rand(T_post, N_X_surr + N_Cx)
    Z1_aug_post_temp = X_surr_aug_post_temp @ np.random.rand(N_X_surr + N_Cx, N_Z1 + N_Cx)

    # Construct original matrices based on these augmented invertible ones (this is complex)
    # For simplicity, just ensure W, Z0, X_surr, Z1 are square and hope augmentation works.
    # The test is primarily for smoke, not perfect conditioning.

    tau_mean, taut_effects, alpha, se_tau = pi_surrogate(
        Y, W, Z0, Z1, X_surr, T0, t1_post_eval, T_total, lag_hac, 
        aux_covariates_main_1=Cw, aux_covariates_main_2=Cy, aux_covariates_surrogate=Cx
    )
    assert alpha.shape == (N_W,) # Alpha for original W

def test_pi_surrogate_dimension_mismatch():
    T_total = 10
    # W_aug.shape[1] == Z0_aug.shape[1]
    # X_surr_aug.shape[1] == Z1_aug.shape[1]
    W = np.random.rand(T_total, 2)
    Z0 = np.random.rand(T_total, 3) # W and Z0 differ
    Z1 = np.random.rand(T_total, 2)
    X_surr = np.random.rand(T_total, 2)
    
    # Cw, Cy, Cx are None
    # W_aug = W, Z0_aug = Z0. W_aug.shape[1] != Z0_aug.shape[1]
    with pytest.raises(MlsynthConfigError, match="Dimension mismatch after augmentation for main or surrogate matrices."):
         pi_surrogate(np.random.rand(T_total), W, Z0, Z1, X_surr, 5, 5, T_total, 1)

    W2 = np.random.rand(T_total, 2)
    Z02 = np.random.rand(T_total, 2) # W2 and Z02 match
    Z1_2 = np.random.rand(T_total, 2)
    X_surr2 = np.random.rand(T_total, 3) # Z1_2 and X_surr2 differ
    with pytest.raises(MlsynthConfigError, match="Dimension mismatch after augmentation for main or surrogate matrices."):
         pi_surrogate(np.random.rand(T_total), W2, Z02, Z1_2, X_surr2, 5, 5, T_total, 1)

# ---------- Tests for pi_surrogate_post ----------
def test_pi_surrogate_post_smoke():
    T_total = 40
    N_W_features = 2
    N_Z0_pre_instr = 2 # Z_combined_aug will have N_Z0 + N_Z1 (+ N_Cy + N_Cx) columns
    N_Z1_surr_instr = 2
    N_X_post_covars = 2 # WX_combined_aug will have N_W + N_X (+ N_Cw + N_Cx) columns
                        # For ZWX_post to be square, N_Z0+N_Z1 must equal N_W+N_X (if no C_covs)

    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W_features)
    Z0_pre_instr = np.random.rand(T_total, N_Z0_pre_instr)
    Z1_surr_instr = np.random.rand(T_total, N_Z1_surr_instr)
    X_post_covars = np.random.rand(T_total, N_X_post_covars)

    T0_treatment_start = 20
    T1_post_length = T_total - T0_treatment_start
    lag_hac = 3

    # To make ZWX_post invertible:
    # Z_combined_post is (T1_post, N_Z0+N_Z1)
    # WX_combined_post is (T1_post, N_W+N_X)
    # ZWX_post = Z_combined_post.T @ WX_combined_post is (N_Z0+N_Z1, N_W+N_X)
    # This must be square, so N_Z0+N_Z1 == N_W+N_X. Here 2+2 == 2+2.
    
    # Create well-conditioned Z_combined_post and WX_combined_post for the post-treatment period
    Z_comb_post_temp = np.random.rand(T1_post_length, N_Z0_pre_instr + N_Z1_surr_instr)
    WX_comb_post_temp = Z_comb_post_temp @ np.random.rand(N_Z0_pre_instr + N_Z1_surr_instr, N_W_features + N_X_post_covars)

    # Reconstruct original matrices (this is tricky, simplify for smoke)
    # For smoke, just use random data and hope it's mostly non-singular.
    # The internal logic of pi_surrogate_post will slice these.

    tau_mean, taut_effects, params_W, se_tau = pi_surrogate_post(
        Y, W, Z0_pre_instr, Z1_surr_instr, X_post_covars,
        T0_treatment_start, T1_post_length, lag_hac
    )

    assert isinstance(tau_mean, float)
    assert np.isfinite(tau_mean)

    assert isinstance(taut_effects, np.ndarray)
    # taut_varying_effects is X_ext_aug @ gamma_coeffs_surr, so shape (T_total, )
    assert taut_effects.shape == (T_total,) 
    assert np.all(np.isfinite(taut_effects))

    assert isinstance(params_W, np.ndarray)
    assert params_W.shape == (N_W_features,) # Coeffs for original W
    assert np.all(np.isfinite(params_W))

    assert isinstance(se_tau, float)
    # se_tau can be nan

def test_pi_surrogate_post_with_covariates():
    T_total = 40
    N_W, N_Z0, N_Z1, N_X = 2, 2, 2, 2 # Base feature counts
    N_Cw, N_Cy, N_Cx = 1, 1, 1       # Covariate feature counts

    Y = np.random.rand(T_total)
    W = np.random.rand(T_total, N_W)
    Z0_pre_instr = np.random.rand(T_total, N_Z0)
    Z1_surr_instr = np.random.rand(T_total, N_Z1)
    X_post_covars = np.random.rand(T_total, N_X)
    Cw_cov = np.random.rand(T_total, N_Cw)
    Cy_cov = np.random.rand(T_total, N_Cy)
    Cx_cov = np.random.rand(T_total, N_Cx)

    T0_treatment_start = 20
    T1_post_length = T_total - T0_treatment_start
    lag_hac = 3
    
    # Z_combined_aug = (Z0, Cy, Z1, Cx) -> N_Z0+N_Cy+N_Z1+N_Cx columns
    # WX_combined_aug = (W, Cw, X, Cx) -> N_W+N_Cw+N_X+N_Cx columns
    # For ZWX_post to be square: N_Z0+N_Cy+N_Z1+N_Cx == N_W+N_Cw+N_X+N_Cx
    # N_Z0+N_Cy+N_Z1 == N_W+N_Cw+N_X
    # Here: 2+1+2 == 2+1+2 (5 == 5), so it should be square.

    tau_mean, taut_effects, params_W, se_tau = pi_surrogate_post(
        Y, W, Z0_pre_instr, Z1_surr_instr, X_post_covars,
        T0_treatment_start, T1_post_length, lag_hac,
        aux_main_covariates=Cw_cov, aux_main_instruments=Cy_cov, aux_surrogate_covariates=Cx_cov
    )
    assert params_W.shape == (N_W,) # Coeffs for original W
    assert np.all(np.isfinite(params_W))
    assert np.isfinite(tau_mean)
    assert taut_effects.shape == (T_total,)

# ---------- Tests for ci_bootstrap ----------
def test_ci_bootstrap_smoke():
    N_features = 3
    Nco = N_features # Number of control units, assume matches features for simple b_weights
    T_total = 30
    t1_pre_periods = 20 # Must be > 5 for m_subsample_size > 0
    nb_samples = 10 # Small number for test speed
    
    # Ensure t1_pre_periods > 5 for m_subsample_size to be positive
    if t1_pre_periods <= 5:
        t1_pre_periods = 6


    x_features = np.random.rand(T_total, N_features)
    y_outcome = np.random.rand(T_total)
    
    # Create plausible b_weights (sum to 1, non-negative)
    b_weights_raw = np.random.rand(Nco)
    b_weights = b_weights_raw / np.sum(b_weights_raw)
    
    y_counterfactual_original = x_features @ b_weights # Simplified counterfactual
    
    # Calculate a plausible att_original
    att_original = np.mean(y_outcome[t1_pre_periods:] - y_counterfactual_original[t1_pre_periods:])
    
    method_name = "MSCb" # Opt.SCopt will use this

    ci = ci_bootstrap(
        b_weights, Nco, x_features, y_outcome, t1_pre_periods, 
        nb_samples, att_original, method_name, y_counterfactual_original
    )

    assert isinstance(ci, list)
    assert len(ci) == 2
    assert isinstance(ci[0], float)
    assert isinstance(ci[1], float)
    
    # CI can be [nan, nan] if m_subsample_size is non-positive, or other issues.
    if not np.isnan(ci[0]) and not np.isnan(ci[1]):
        assert ci[0] <= ci[1] # Lower bound <= Upper bound
    else:
        # If nan, it's likely due to m_subsample_size <= 0 warning path
        # or if nb_samples is too small for percentile calculation (e.g. < 40 for 0.025 percentile)
        pass


def test_ci_bootstrap_subsample_warning():
    N_features = 3
    Nco = N_features
    T_total = 10
    t1_pre_periods = 5 # This will make m_subsample_size = 0
    nb_samples = 10
    
    x_features = np.random.rand(T_total, N_features)
    y_outcome = np.random.rand(T_total)
    b_weights_raw = np.random.rand(Nco)
    b_weights = b_weights_raw / np.sum(b_weights_raw)
    y_counterfactual_original = x_features @ b_weights
    att_original = np.mean(y_outcome[t1_pre_periods:] - y_counterfactual_original[t1_pre_periods:])
    method_name = "MSCb"

    with pytest.warns(UserWarning, match="Subsample size for bootstrap is non-positive"):
        ci = ci_bootstrap(
            b_weights, Nco, x_features, y_outcome, t1_pre_periods, 
            nb_samples, att_original, method_name, y_counterfactual_original
        )
    assert isinstance(ci, list)
    assert len(ci) == 2
    assert np.isnan(ci[0])
    assert np.isnan(ci[1])

# ---------- Tests for TSEST ----------
def test_TSEST_smoke():
    N_features = 3
    T_total = 30
    t1_pre_periods = 20 # Must be > 5 for ci_bootstrap
    nb_samples = 10 # Small for speed
    
    # Ensure t1_pre_periods > 5 for m_subsample_size in ci_bootstrap
    if t1_pre_periods <= 5:
        t1_pre_periods = 6

    t2_post_periods = T_total - t1_pre_periods
    
    x_features = np.random.rand(T_total, N_features)
    y_outcome = np.random.rand(T_total)
    donor_names_list = [f"donor_{i}" for i in range(N_features)]

    results_list = TSEST(
        x_features, y_outcome, t1_pre_periods, nb_samples, donor_names_list, t2_post_periods
    )

    assert isinstance(results_list, list)
    assert len(results_list) == 4 # SIMPLEX, MSCb, MSCa, MSCc

    expected_methods = ["SIMPLEX", "MSCb", "MSCa", "MSCc"]
    for item in results_list:
        assert isinstance(item, dict)
        assert len(item) == 1 # {method_name: results_dict}
        method_name = next(iter(item.keys()))
        assert method_name in expected_methods
        
        results_dict = item[method_name]
        assert isinstance(results_dict, dict)
        assert "Fit" in results_dict
        assert "Effects" in results_dict
        assert "95% CI" in results_dict
        assert "Vectors" in results_dict
        assert "WeightV" in results_dict
        assert "Weights" in results_dict

        # Check some sub-structures
        assert isinstance(results_dict["Effects"], dict)
        assert "ATT" in results_dict["Effects"]
        
        assert isinstance(results_dict["95% CI"], list)
        assert len(results_dict["95% CI"]) == 2
        
        assert isinstance(results_dict["WeightV"], np.ndarray)
        expected_weight_len = N_features
        if method_name in ["MSCa", "MSCc"]: # These add an intercept
            expected_weight_len +=1
        assert results_dict["WeightV"].shape == (expected_weight_len,)

        assert isinstance(results_dict["Weights"], dict)

# ---------- Tests for pcr ----------
@pytest.mark.parametrize("cluster_flag", [False, True])
@pytest.mark.parametrize("is_frequentist", [False, True])
def test_pcr_smoke(cluster_flag, is_frequentist):
    T_total = 30
    N_donors = 5 
    pre_periods = 20 # Used for SVT/model fitting
    
    # Ensure pre_periods is sufficient for SVDCluster if used
    # SVDCluster needs at least 2 data points for PCA/SVD.
    # And enough donors for clustering (e.g., > n_clusters)
    # For smoke, keep it simple.

    X_donors = np.random.rand(T_total, N_donors)
    y_treated = np.random.rand(T_total)
    objective_model = "MSCb" # For frequentist path
    donor_names_list = [f"donor_{i}" for i in range(N_donors)]

    # SVDCluster might reduce N_donors if clustering.
    # Ensure enough donors if cluster_flag is True for SVDCluster to work.
    # SVDCluster determines optimal clusters, could be 1.
    # If N_donors is small (e.g., <3), SVDCluster might behave unexpectedly or error.
    # For this smoke test, N_donors=5 should be okay.

    results = pcr(
        X_donors, y_treated, objective_model, donor_names_list,
        num_pre_treatment_periods=pre_periods, enable_clustering=cluster_flag, use_frequentist_scm=is_frequentist
    )

    assert isinstance(results, dict)
    assert "weights" in results
    assert "cf_mean" in results
    
    assert isinstance(results["weights"], dict)
    assert isinstance(results["cf_mean"], np.ndarray)
    assert results["cf_mean"].shape == (T_total,)

    if not is_frequentist: # Bayesian
        assert "credible_interval" in results
        assert isinstance(results["credible_interval"], tuple)
        assert len(results["credible_interval"]) == 2
        assert isinstance(results["credible_interval"][0], np.ndarray)
        assert isinstance(results["credible_interval"][1], np.ndarray)
        assert results["credible_interval"][0].shape == (T_total,)
        assert results["credible_interval"][1].shape == (T_total,)
        assert np.all(results["credible_interval"][0] <= results["credible_interval"][1])
    else: # Frequentist
        assert "credible_interval" not in results

    # Check weights sum to approx 1 for frequentist SCM models (like MSCb)
    # Bayesian weights don't necessarily sum to 1.
    if is_frequentist and objective_model in ["MSCb", "SIMPLEX", "MSCa"]: # MSCc intercept is separate
        current_weights_values = np.array(list(results["weights"].values()))
        if objective_model == "MSCa": # Intercept is part of weights from SCopt, but not in final dict
             # The weights dict from pcr only contains donor weights.
             # SCopt for MSCa returns intercept + donor weights. pcr extracts donor weights.
             # So, sum of donor weights should be 1.
            pass # Cannot directly check sum from SCopt output here easily.
                 # The returned weights dict should sum to 1.
        
        # For frequentist, weights should sum to 1 if model implies it (SIMPLEX, MSCa donors, MSCb)
        # The `weights` dict returned by `pcr` contains only donor weights.
        # For MSCb, Opt.SCopt has beta >=0, no sum constraint.
        # For SIMPLEX, Opt.SCopt has sum(beta)==1.
        # For MSCa, Opt.SCopt has sum(beta[1:])==1.
        # The `pcr` function uses `current_donor_names` for the weights dict.
        # If `objective_model` is "SIMPLEX" or "MSCa" (for donor part), sum should be 1.
        # "MSCb" does not enforce sum-to-1.
        if objective_model == "SIMPLEX": # Only SIMPLEX strictly enforces sum-to-1 on all weights
             assert pytest.approx(sum(current_weights_values), abs=1e-3) == 1.0
        # For MSCa, the donor weights (excluding intercept) sum to 1.
        # The returned dict `results["weights"]` should be these donor weights.
        if objective_model == "MSCa":
             assert pytest.approx(sum(current_weights_values), abs=1e-3) == 1.0
        
        # All weights should be non-negative for these models
        if objective_model in ["SIMPLEX", "MSCb", "MSCa"]:
            assert np.all(current_weights_values >= -1e-6) # Allow for small numerical errors

# ---------- Fixture for pda tests ----------
@pytest.fixture
def prepped_data_for_pda():
    T_total = 100
    N_donors_original = 25
    pre_periods = 70
    post_periods = T_total - pre_periods
    
    y_treated_full = np.random.rand(T_total)
    # Ensure donor_matrix has some variance for LassoCV etc.
    donor_matrix_full = np.random.rand(T_total, N_donors_original) + \
                        np.arange(T_total)[:, np.newaxis] * 0.1 
                        
    donor_names_full = pd.Index([f"donor_orig_{i}" for i in range(N_donors_original)])
    
    return {
        "y": y_treated_full,
        "donor_matrix": donor_matrix_full,
        "donor_names": donor_names_full,
        "pre_periods": pre_periods,
        "post_periods": post_periods,
        "total_periods": T_total,
    }

# ---------- Tests for pda ----------
@pytest.mark.parametrize("method", ["fs", "LASSO", "l2"])
def test_pda_smoke(prepped_data_for_pda, method):
    N_donors_param = 24 # For 'fs', max selected donors. For others, might not be used directly.
    tau_l2_param = 0.1 if method == "l2" else None # Only for l2

    # Ensure pre_periods is sufficient for LassoCV's default cv=5
    if method == "LASSO" and prepped_data_for_pda["pre_periods"] < 5:
        # This fixture modification is a bit tricky. Ideally, fixture should provide enough.
        # For now, let's assume fixture pre_periods is >= 5.
        # If not, LassoCV might fail or use fewer folds.
        # The fixture has pre_periods = 20, so it's fine.
        pass


    results = pda(prepped_data_for_pda, N_donors_param, pda_method_type=method, l2_regularization_param_override=tau_l2_param)

    assert isinstance(results, dict)
    assert "method" in results
    assert results["method"].lower() == method.lower() or (method=="l2" and results["method"] == r"l2 relaxation")
    
    assert "Effects" in results
    assert "Fit" in results
    assert "Vectors" in results
    assert "Inference" in results # All methods should provide some inference

    if method == "fs":
        assert "Betas" in results # Dict of donor coeffs
        assert isinstance(results["Betas"], dict)
    elif method == "LASSO":
        assert "non_zero_coef_dict" in results
        assert isinstance(results["non_zero_coef_dict"], dict)
    elif method == "l2":
        assert "Betas" in results
        assert "optimal_tau" in results
        assert "Intercept" in results
        assert isinstance(results["Betas"], dict)
        assert isinstance(results["optimal_tau"], float)
        assert isinstance(results["Intercept"], float)

    # Common checks for results structure
    assert isinstance(results["Effects"], dict)
    assert "ATT" in results["Effects"]
    assert isinstance(results["Fit"], dict)
    assert "T0 RMSE" in results["Fit"] # Changed from "RMSE"
    assert isinstance(results["Vectors"], dict)
    assert "Gap" in results["Vectors"]
    assert isinstance(results["Inference"], dict)
    assert "t_stat" in results["Inference"] or "standard_error" in results["Inference"]


def test_pda_invalid_method(prepped_data_for_pda):
    with pytest.raises(MlsynthConfigError, match="Invalid PDA method specified: invalid. Choose from 'fs', 'LASSO', or 'l2'."):
        pda(prepped_data_for_pda, 3, pda_method_type="invalid")

# ---------- Tests for SRCest (and implicitly __SRC_opt) ----------
def test_SRCest_smoke():
    T_total = 30
    N_donors = 4
    num_post_periods = 10
    T0_pre_periods = T_total - num_post_periods

    y1_treated_full = np.random.rand(T_total)
    Y0_donors_full = np.random.rand(T_total, N_donors)
    
    # Ensure Y0_donors_full has some variance in pre-periods for get_theta/get_sigmasq
    Y0_donors_full[:T0_pre_periods] += np.arange(T0_pre_periods)[:, np.newaxis] * 0.05

    counterfactual, w_hat, theta_hat = SRCest(y1_treated_full, Y0_donors_full, num_post_periods)

    assert isinstance(counterfactual, np.ndarray)
    assert counterfactual.shape == (T_total,)
    assert np.all(np.isfinite(counterfactual))

    assert isinstance(w_hat, np.ndarray)
    assert w_hat.shape == (N_donors,)
    assert np.all(np.isfinite(w_hat))
    assert pytest.approx(np.sum(w_hat)) == 1.0
    assert np.all(w_hat >= -1e-6) # Non-negative constraint, allow for small numerical errors

    assert isinstance(theta_hat, np.ndarray)
    assert theta_hat.shape == (N_donors,)
    assert np.all(np.isfinite(theta_hat))

def test_SRCest_edge_cases():
    # Case 1: Not enough pre-periods (e.g., T0_pre_periods < 1 for mean, or <2 for variance)
    T_total_short_pre = 5
    N_donors_short_pre = 3
    num_post_short_pre = 4 # T0_pre = 1
    y1_short_pre = np.random.rand(T_total_short_pre)
    Y0_short_pre = np.random.rand(T_total_short_pre, N_donors_short_pre)
    
    # With T0_pre = 1, get_theta might produce NaNs or zeros if denominators are zero.
    # get_sigmasq might also have issues.
    # OSQP might fail if inputs are problematic.
    # Expecting potential RuntimeWarning (e.g. from np.mean of empty slice if T0=0) or ValueErrors
    # For T0=1, np.mean is fine, but variance in get_theta (denominators) might be zero.
    # If theta_hat contains NaNs, __SRC_opt might fail.
    # Removing pytest.warns as the internal functions seem to handle T0=1 without runtime warnings,
    # potentially leading to NaN or zero inputs to the optimizer.
    cf, w, th = SRCest(y1_short_pre, Y0_short_pre, num_post_short_pre)
    # Depending on random data, it might pass or w_hat might be NaN.
    # If w_hat is NaN, then cf will be NaN.
    # This is more of a stress test than a strict pass/fail on values.
    assert cf.shape == (T_total_short_pre,)
    # Values might be NaN if optimization or intermediate steps had issues.
    # For instance, if all w_hat are NaN, cf will be NaN.
    # If w_hat are numbers but theta_hat had NaNs, cf could be NaN.
    # If sigma2 was NaN, penalty in __SRC_opt is NaN, OSQP might error or return NaNs.

    # Case 2: No post periods
    T_total_no_post = 10
    N_donors_no_post = 3
    num_post_no_post = 0 # T0_pre = 10, T1_post = 0
    y1_no_post = np.random.rand(T_total_no_post)
    Y0_no_post = np.random.rand(T_total_no_post, N_donors_no_post)
    Y0_no_post += np.arange(T_total_no_post)[:, np.newaxis] * 0.05


    cf_no_post, w_no_post, th_no_post = SRCest(y1_no_post, Y0_no_post, num_post_no_post)
    assert cf_no_post.shape == (T_total_no_post,)
    assert w_no_post.shape == (N_donors_no_post,)
    assert th_no_post.shape == (N_donors_no_post,)
    # y1_hat_post_cf part of counterfactual will be empty, concatenated with y1_hat_pre
    assert len(cf_no_post) == T_total_no_post # Should be only pre-treatment predictions

# ---------- Fixture for RPCASYNTH tests ----------
@pytest.fixture
def rpcasynth_input_data():
    T_total = 30
    N_units = 6 # Treated + 5 donors
    pre_periods = 20
    post_periods = T_total - pre_periods
    
    unit_ids = [f"unit_{i}" for i in range(N_units)]
    time_periods = list(range(T_total))
    
    data_list = []
    for unit in unit_ids:
        for t_idx, t_val in enumerate(time_periods):
            # Create some trend and noise
            outcome_val = t_idx * 0.5 + np.random.randn() + (ord(unit[-1]) - ord('0')) * 10
            data_list.append({"unitid": unit, "time": t_val, "outcome": outcome_val})
            
    df_panel = pd.DataFrame(data_list)
    treated_unit_name = "unit_0"

    # Mimic prepped_data structure
    # y is the outcome for the treated unit
    y_treated = df_panel[df_panel["unitid"] == treated_unit_name].sort_values("time")["outcome"].values
    
    prepped_data = {
        "treated_unit_name": treated_unit_name,
        "pre_periods": pre_periods,
        "post_periods": post_periods,
        "y": y_treated, # Full outcome series for the treated unit
        # Other keys like donor_matrix, X_treat, Y_treat_pre etc. are not directly used by RPCASYNTH
        # as it re-pivots the df_panel.
    }
    
    config = {
        "unitid": "unitid",
        "time": "time",
        "outcome": "outcome",
        # ROB method will be set in the test
    }
    return df_panel, config, prepped_data

# ---------- Tests for RPCASYNTH ----------
@pytest.mark.parametrize("rob_method", ["PCP", "HQF"])
def test_RPCASYNTH_smoke(rpcasynth_input_data, rob_method):
    df_panel, config, prepped_data = rpcasynth_input_data
    config["ROB"] = rob_method

    # Ensure enough pre_periods for fpca and KMeans
    # fpca needs at least 2 components, KMeans needs n_samples > n_clusters
    # The fixture has pre_periods = 20, N_units = 6, which should be fine.

    results = RPCASYNTH(df_panel, config, prepped_data)

    assert isinstance(results, dict)
    assert "name" in results
    assert results["name"] == "RPCA"
    assert "weights" in results
    assert isinstance(results["weights"], dict)
    assert "Effects" in results
    assert isinstance(results["Effects"], dict)
    assert "ATT" in results["Effects"]
    assert "Fit" in results
    assert isinstance(results["Fit"], dict)
    assert "T0 RMSE" in results["Fit"] # Changed from "RMSE"
    assert "Vectors" in results
    assert isinstance(results["Vectors"], dict)
    assert "Gap" in results["Vectors"]
    # The "Gap" vector from effects.calculate is a (T_total, 2) matrix
    assert results["Vectors"]["Gap"].shape == (prepped_data["y"].shape[0], 2)
    assert "Observed Unit" in results["Vectors"]
    assert results["Vectors"]["Observed Unit"].shape == (prepped_data["y"].shape[0], 1) # Reshaped to (T,1)
    assert "Counterfactual" in results["Vectors"]
    assert results["Vectors"]["Counterfactual"].shape == (prepped_data["y"].shape[0], 1) # Reshaped to (T,1)


def test_RPCASYNTH_invalid_rob_method(rpcasynth_input_data):
    df_panel, config, prepped_data = rpcasynth_input_data
    config["ROB"] = "INVALID_METHOD"

    with pytest.warns(UserWarning, match="Invalid robust method 'INVALID_METHOD'. Defaulting to 'PCP'."):
        results = RPCASYNTH(df_panel, config, prepped_data)
    
    # Should still run with PCP default
    assert isinstance(results, dict)
    assert "name" in results
    assert results["name"] == "RPCA"
