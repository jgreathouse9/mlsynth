import numpy as np
import pytest
from mlsynth.utils.denoiseutils import (
    universal_rank,
    spectral_rank,
    shrink,
    SVT,
    svt,
    RPCA
)

# ---------- Tests for universal_rank ----------
def test_universal_rank_basic():
    s = np.array([5, 3, 1, 0.5])
    rank = universal_rank(s, ratio=0.5)
    assert isinstance(rank, int)
    assert rank >= 1

# ---------- Tests for spectral_rank ----------
def test_spectral_rank_exact():
    s = np.array([4, 2, 1])
    t = 1.0
    rank = spectral_rank(s, t)
    assert rank == len(s)

def test_spectral_rank_partial():
    s = np.array([3, 2, 1])
    t = 0.8
    rank = spectral_rank(s, t)
    assert rank >= 1 and rank <= len(s)

# ---------- Tests for shrink ----------
def test_shrink_behavior():
    s = np.array([5.0, 2.0, 0.5])
    tau = 1.0
    shrunk = shrink(s, tau)
    expected = np.array([4.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(shrunk, expected)

# ---------- Tests for SVT (singular value thresholding) ----------
def test_SVT_identity_matrix():
    X = np.eye(3)
    tau = 0.5
    result = SVT(X, tau)
    assert result.shape == X.shape
    assert np.all(np.isfinite(result))

# ---------- Tests for svt (low-rank approximation) ----------
def test_svt_low_rank_approximation():
    X = np.random.randn(5, 4)
    Y0_rank, n2, u_rank, s_rank, v_rank = svt(X)

    assert Y0_rank.shape == X.shape
    assert u_rank.shape[0] == X.shape[0]
    assert v_rank.shape[1] == X.shape[1]
    assert len(s_rank) == u_rank.shape[1] == v_rank.shape[0]

# ---------- Tests for RPCA ----------
def test_RPCA_output_shape_and_type():
    X = np.random.randn(5, 5)
    L = RPCA(X)
    assert L.shape == X.shape
    assert np.all(np.isfinite(L))

def test_RPCA_low_rank_approximation():
    # Construct a low-rank matrix and add noise
    U = np.random.randn(5, 2)
    V = np.random.randn(2, 5)
    low_rank = U @ V
    noise = np.random.normal(scale=0.1, size=low_rank.shape)
    X = low_rank + noise
    L = RPCA(X)
    error = np.linalg.norm(L - low_rank) / np.linalg.norm(low_rank)
    assert error < 0.5  # Acceptable approximation
