import numpy as np
import pytest
import pandas as pd
import cvxpy as cp
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.estutils import (
    bartlett,
    compute_hac_variance,
    compute_t_stat_and_ci,
    l2_relax,
    adaptive_cross_validate_tau,
    Opt, # Class containing SCopt
    NSC_opt,
    NSCcv,
    SMOweights,
)

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

# ---------- Tests for cross_validate_tau ----------
def test_cross_validate_tau_smoke():
    T_pre = 20
    N_controls = 3
    treated_unit_pre = np.random.rand(T_pre)
    X_pre = np.random.rand(T_pre, N_controls)
    tau_init_upper = 1.0

    optimal_tau, min_mse = adaptive_cross_validate_tau(treated_unit_pre, X_pre, tau_init_upper, num_coarse_points=10)

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
