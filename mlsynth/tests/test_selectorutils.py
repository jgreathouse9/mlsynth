import numpy as np
import pytest
from unittest.mock import patch

from mlsynth.utils.selectorsutils import normalize, SVDCluster

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
