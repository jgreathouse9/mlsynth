import numpy as np
import pytest
from mlsynth.utils.inferutils import ag_conformal, step2
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError


# --- Tests for ag_conformal ---

def test_ag_conformal_basic(): # Existing test, slightly modified for clarity
    y_true_pre = np.array([1.0, 2.0, 3.0, 2.5]) # Added one more point for variance calc
    y_pred_pre = np.array([0.9, 2.1, 2.9, 2.6])
    y_pred_post = np.array([3.5, 4.0])
    alpha = 0.1
    pad_value = -999

    lower, upper = ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=alpha, pad_value=pad_value)

    residuals = y_true_pre - y_pred_pre
    mu_hat = np.mean(residuals)
    sigma2_hat = np.var(residuals, ddof=1) # ddof=1 for sample variance
    delta = np.sqrt(2 * sigma2_hat * np.log(2 / alpha))

    expected_lower_post = y_pred_post + mu_hat - delta
    expected_upper_post = y_pred_post + mu_hat + delta

    expected_pad_arr = np.full(len(y_true_pre), pad_value)

    expected_lower_full = np.concatenate([expected_pad_arr, expected_lower_post])
    expected_upper_full = np.concatenate([expected_pad_arr, expected_upper_post])

    np.testing.assert_allclose(lower, expected_lower_full, rtol=1e-6)
    np.testing.assert_allclose(upper, expected_upper_full, rtol=1e-6)
    assert lower.shape == (len(y_true_pre) + len(y_pred_post),)
    assert upper.shape == (len(y_true_pre) + len(y_pred_post),)

def test_ag_conformal_mismatched_lengths():
    y_true_pre = np.array([1.0, 2.0])
    y_pred_pre = np.array([0.9])
    y_pred_post = np.array([3.5])
    with pytest.raises(MlsynthDataError, match="actual_outcomes_pre_treatment and predicted_outcomes_pre_treatment must have the same length."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post)

def test_ag_conformal_empty_pre_treatment():
    y_true_pre = np.array([])
    y_pred_pre = np.array([])
    y_pred_post = np.array([3.5])
    with pytest.raises(MlsynthDataError, match="Pre-treatment arrays cannot be empty."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post)

def test_ag_conformal_invalid_miscoverage_rate():
    y_true_pre = np.array([1.0, 2.0])
    y_pred_pre = np.array([0.9, 1.9])
    y_pred_post = np.array([3.5])
    with pytest.raises(MlsynthConfigError, match="miscoverage_rate must be between 0 and 1."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=0.0)
    with pytest.raises(MlsynthConfigError, match="miscoverage_rate must be between 0 and 1."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=1.0)
    with pytest.raises(MlsynthConfigError, match="miscoverage_rate must be between 0 and 1."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=-0.1)
    with pytest.raises(MlsynthConfigError, match="miscoverage_rate must be between 0 and 1."):
        ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=1.1)

def test_ag_conformal_empty_post_treatment():
    y_true_pre = np.array([1.0, 2.0, 3.0])
    y_pred_pre = np.array([0.9, 2.1, 2.9])
    y_pred_post = np.array([]) # Empty post-treatment predictions
    alpha = 0.1
    pad_value = -999

    lower, upper = ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=alpha, pad_value=pad_value)

    expected_pad_arr = np.full(len(y_true_pre), pad_value)
    
    np.testing.assert_allclose(lower, expected_pad_arr, rtol=1e-6)
    np.testing.assert_allclose(upper, expected_pad_arr, rtol=1e-6)
    assert lower.shape == (len(y_true_pre),) # Only pad values
    assert upper.shape == (len(y_true_pre),)

def test_ag_conformal_default_pad_value_nan():
    y_true_pre = np.array([1.0, 2.0, 3.0, 2.5])
    y_pred_pre = np.array([0.9, 2.1, 2.9, 2.6])
    y_pred_post = np.array([3.5, 4.0])
    alpha = 0.1

    lower, upper = ag_conformal(y_true_pre, y_pred_pre, y_pred_post, miscoverage_rate=alpha) # Default pad_value=np.nan

    # Check that pre-treatment part is NaN
    assert np.all(np.isnan(lower[:len(y_true_pre)]))
    assert np.all(np.isnan(upper[:len(y_true_pre)]))
    # Check that post-treatment part is not NaN (assuming sigma2_hat > 0)
    assert not np.any(np.isnan(lower[len(y_true_pre):]))
    assert not np.any(np.isnan(upper[len(y_true_pre):]))


# --- Fixtures for step2 ---

@pytest.fixture
def step2_sample_data() -> dict:
    """Provides sample data for the step2 function tests."""
    np.random.seed(42)
    t1 = 20  # Number of pre-treatment periods
    n_donors = 5
    n_coeffs = n_donors + 1 # donors + intercept
    nb = 50  # Number of bootstrap replications

    # R1t: restriction for sum of weights = 1 (excluding intercept)
    R1t = np.zeros((1, n_coeffs))
    if n_coeffs > 1:
        R1t[0, 1:] = 1.0 
    q1t = np.array([1.0])

    # R2t: restriction for intercept = 0
    R2t = np.zeros((1, n_coeffs))
    if n_coeffs > 0:
        R2t[0, 0] = 1.0
    q2t = np.array([0.0])
    
    # Rt: combined restrictions
    if n_coeffs > 1:
        Rt = np.vstack([R1t, R2t])
        qt = np.concatenate([q1t, q2t])
    elif n_coeffs == 1: # Only intercept
        Rt = R2t
        qt = q2t
    else: # No coeffs
        Rt = np.array([]).reshape(0,0) # Or handle as error
        qt = np.array([])


    b_MSC_c = np.random.rand(n_coeffs) # Example coefficients
    if n_coeffs > 1:
      b_MSC_c[1:] /= np.sum(b_MSC_c[1:]) # Make weights sum to 1 (approx)
      b_MSC_c[0] = 0.05 # Small intercept

    x1 = np.random.rand(t1, n_coeffs) # Pre-treatment donor matrix (incl. intercept column if used)
    # Ensure x1's first column is ones if an intercept is modeled by b_MSC_c[0]
    if n_coeffs > 0 :
        x1[:, 0] = 1.0 
    
    y1 = x1 @ b_MSC_c + np.random.normal(0, 0.1, t1) # Pre-treatment outcome for treated

    bm_MSC_c = np.zeros((n_coeffs, nb))

    return {
        "R1t": R1t, "R2t": R2t, "Rt": Rt, "b_MSC_c": b_MSC_c,
        "q1t": q1t, "q2t": q2t, "qt": qt, "t1": t1,
        "x1": x1, "y1": y1, "nb": nb, "n": n_coeffs, "bm_MSC_c": bm_MSC_c
    }

# --- Tests for step2 ---

def test_step2_smoke(step2_sample_data):
    """Smoke test for the step2 function."""
    data = step2_sample_data
    
    # Handle cases where Rt might be empty or 1D, which affects V_hatI logic in step2
    if data["Rt"].size == 0 and data["n"] == 0: # No coeffs, no restrictions
        # This case is problematic for step2 as written.
        # For a smoke test, let's ensure it doesn't crash if n=0 leads to empty Rt.
        # The function might need more robust handling for n=0.
        # For now, we expect it to run, possibly returning a default or erroring gracefully.
        # Given the current logic, it might error if Rt is empty and V_hatI cannot be formed.
        # Let's skip this specific n=0 sub-case for smoke test if Rt is truly empty.
        if data["Rt"].shape == (0,0):
             pytest.skip("Skipping n=0 case where Rt is empty, needs robust handling in step2")

    # A simplified Rt for the n=0 case if it's not skipped
    # This part is tricky because step2's internal logic for V_hatI and J_test
    # heavily depends on Rt's shape.
    # If n=0, Rt, R1t, R2t, b_MSC_c, q1t, q2t, qt, x1, bm_MSC_c would be empty or zero-dim.
    # The current step2 implementation might not gracefully handle n=0.
    # For a smoke test, we assume n > 0 as per typical usage.
    if data["n"] == 0:
        pytest.skip("step2 smoke test requires n > 0 due to internal logic assumptions.")


    recommended_model = step2(
        restriction_matrix_h0a=data["R1t"],
        restriction_matrix_h0b=data["R2t"],
        combined_restriction_matrix_h0=data["Rt"],
        msc_c_coefficients_initial=data["b_MSC_c"],
        target_values_h0a=data["q1t"],
        target_values_h0b=data["q2t"],
        target_values_h0_combined=data["qt"],
        num_pre_treatment_periods=data["t1"],
        donor_predictors_pre_treatment=data["x1"],
        treated_outcome_pre_treatment=data["y1"],
        num_bootstrap_replications=data["nb"],
        num_model_coefficients=data["n"],
        bootstrapped_msc_c_coefficients_array=data["bm_MSC_c"]
    )
    assert recommended_model in ["MSCc", "MSCa", "MSCb", "SC"]

def test_step2_n_equals_1_intercept_only(step2_sample_data):
    """Test step2 when n=1 (only intercept)."""
    data = step2_sample_data
    n_coeffs = 1
    
    R1t = np.zeros((1, n_coeffs)) # No weights to sum
    q1t = np.array([1.0]) # Target for R1t (irrelevant here but needs shape)

    R2t = np.zeros((1, n_coeffs))
    R2t[0,0] = 1.0 # Intercept = 0 restriction
    q2t = np.array([0.0])

    Rt = R2t # Only intercept restriction matters
    qt = q2t

    b_MSC_c = np.random.rand(n_coeffs) # e.g. array([0.05])
    x1 = np.ones((data["t1"], n_coeffs)) # Column of ones for intercept
    y1 = x1 @ b_MSC_c + np.random.normal(0, 0.1, data["t1"])
    bm_MSC_c = np.zeros((n_coeffs, data["nb"]))

    recommended_model = step2(
        restriction_matrix_h0a=R1t,
        restriction_matrix_h0b=R2t,
        combined_restriction_matrix_h0=Rt,
        msc_c_coefficients_initial=b_MSC_c,
        target_values_h0a=q1t,
        target_values_h0b=q2t,
        target_values_h0_combined=qt,
        num_pre_treatment_periods=data["t1"],
        donor_predictors_pre_treatment=x1,
        treated_outcome_pre_treatment=y1,
        num_bootstrap_replications=data["nb"],
        num_model_coefficients=n_coeffs,
        bootstrapped_msc_c_coefficients_array=bm_MSC_c
    )
    assert recommended_model in ["MSCc", "MSCa", "MSCb", "SC"]
    # More specific assertions could be made if we knew the expected outcome for this setup
    # For example, if b_MSC_c[0] is close to 0, MSCb might be rejected (pJ high, p2 high) -> MSCc or MSCa
    # If b_MSC_c[0] is far from 0, MSCb might be chosen (pJ low, p2 low)

# More tests for step2 could cover:
# - Different nb values (e.g., very small nb)
# - Scenarios designed to trigger specific model recommendations
# - Behavior when V_hatI is singular (though pinv is used)
# - Impact of t1 size
