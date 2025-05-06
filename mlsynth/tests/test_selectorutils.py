import numpy as np
import pytest
from unittest.mock import patch
from mlsynth.utils.selectorsutils import normalize, SVDCluster, PDAfs

# ----- Tests for normalize -----

def test_normalize_centers_columns():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    norm_X = normalize(X)
    assert np.allclose(norm_X.mean(axis=0), [0.0, 0.0]), "Each column should be mean-centered"


def test_normalize_zero_input():
    X = np.zeros((4, 3))
    norm_X = normalize(X)
    assert np.allclose(norm_X, 0), "Normalization of zero input should remain zero"


# ----- SVD mock -----

def mock_svt(Y):
    # Simple deterministic SVD using numpy
    u, s, vh = np.linalg.svd(Y, full_matrices=False)
    return len(s), Y, u, s, vh


# ----- Tests for SVDCluster -----

@patch("mlsynth.utils.denoiseutils.svt", side_effect=mock_svt)
def test_svdcluster_predictable_clusters(mocked_svt):
    # Cluster 1 (close to treated unit)
    donor1 = [1.0, 1.0, 1.0, 1.0]
    donor2 = [1.1, 1.0, 0.9, 1.0]
    
    # Cluster 2 (far away)
    donor3 = [5.0, 5.0, 5.0, 5.0]

    # Donor matrix: shape (4, 3)
    X = np.array([donor1, donor2, donor3]).T
    
    # Treated unit: similar to donor1 and donor2
    y = np.array([1.05, 1.0, 1.0, 0.95])
    donor_names = ["A", "B", "C"]

    X_sub, selected_names, selected_indices = SVDCluster(X, y, donor_names)

    assert set(selected_names) == {"A", "B"}, "Treated unit should cluster with donors A and B"
    assert "C" not in selected_names, "Donor C should not be in the same cluster"
    assert X_sub.shape[1] == 2, "Output matrix should include only 2 donors"


@patch("mlsynth.utils.denoiseutils.svt", side_effect=mock_svt)
def test_svdcluster_output_dimensions(mocked_svt):
    X = np.random.rand(6, 4)  # 6 samples, 4 donors
    y = np.random.rand(6)     # Treated unit
    donor_names = ["Donor1", "Donor2", "Donor3", "Donor4"]

    X_sub, selected_names, selected_indices = SVDCluster(X, y, donor_names)

    assert X_sub.shape[0] == X.shape[0], "Output should have same number of rows as input"
    assert X_sub.shape[1] == len(selected_names), "Number of selected donors should match shape"
    assert all(name in donor_names for name in selected_names), "All returned names must be valid donors"
    assert all(0 <= i < X.shape[1] for i in selected_indices), "All indices should be valid donor columns"


# ----- Tests for Forward Selection -----

def test_pdafs():
    # Parameters
    t1 = 156  # pre-treatment
    t2 = 3    # post-treatment
    t = t1 + t2
    N = 50    # donors

    # True model
    intercept = 0.86
    coeffs = np.array([-0.95, 0.75])  # donors 2 and 3
    selected_indices = [2, 3]

    # Simulate donor data
    np.random.seed(42)
    donor = np.random.randn(t, N)

    # Generate treated outcome
    y = (
        intercept
        + coeffs[0] * donor[:, selected_indices[0]]
        + coeffs[1] * donor[:, selected_indices[1]]
        + np.random.normal(0, 0.1, size=t)
    )

    # Run PDAfs
    result = PDAfs(y, donor, t1=t1, t=t, N=N)

    # Assertions
    selected = result["selected_donors"]
    estimated_coeffs = result["model_coefficients"]

    assert isinstance(selected, (list, np.ndarray)), "Selected donors should be a list or array"
    assert isinstance(estimated_coeffs, np.ndarray), "Coefficients should be a numpy array"

    # Check if true donors are included
    selected_set = set(selected)
    required_set = set(selected_indices)
    assert required_set.issubset(selected_set), f"Expected donors {selected_indices}, got {selected}"

    # Check coefficient accuracy
    est_intercept = estimated_coeffs[0]
    donor_coeff_map = dict(zip(selected, estimated_coeffs[1:]))

    assert np.isclose(est_intercept, intercept, atol=0.05), \
        f"Intercept off: expected ~{intercept}, got {est_intercept}"

    for idx, true_val in zip(selected_indices, coeffs):
        est_val = donor_coeff_map.get(idx)
        assert est_val is not None, f"Donor {idx} not found in selected coefficients"
        assert np.isclose(est_val, true_val, atol=0.05), \
            f"Coefficient for donor {idx} off: expected ~{true_val}, got {est_val}"

