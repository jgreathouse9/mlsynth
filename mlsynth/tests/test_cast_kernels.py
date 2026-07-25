"""Unit tests for the CAST numerical kernels (Xia, Yan & Wainwright 2025).

The four-block estimator, the staircase partition of a staggered design, and
the entrywise variance formula. Value-for-value fidelity to the authors'
``CAST-panel`` package on the ACA panel is pinned in the golden benchmark; here
we lock the kernels' mathematical invariants, self-contained.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.cast_helpers.four_block import (
    entrywise_variance,
    four_block_est,
    staircase_blocks,
    svd_r,
)


def _low_rank(N, T, r, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(N, r)), rng.normal(size=(T, r))
    M = U @ V.T
    return M + noise * rng.normal(size=(N, T)), M


# ------------------------------------------------------------------- svd_r
def test_svd_r_shapes_and_reconstruction():
    M, _ = _low_rank(20, 12, 3)
    u, s, vh = svd_r(M, 3)
    assert u.shape == (20, 3) and s.shape == (3,) and vh.shape == (3, 12)
    assert np.allclose((u * s) @ vh, M)          # exact for a rank-3 matrix


def test_svd_r_rejects_bad_rank():
    M, _ = _low_rank(10, 8, 2)
    with pytest.raises(MlsynthConfigError):
        svd_r(M, 0)
    with pytest.raises(MlsynthConfigError):
        svd_r(M, 9)                              # r > min(N, T)


# --------------------------------------------------------- four_block_est
def test_four_block_exact_recovery_noiseless():
    """Paper S3.1.1: with M* exactly rank r and no noise, the imputed block
    equals M*_d exactly."""
    Y, M = _low_rank(30, 20, 3, noise=0.0)
    N1, T1 = 18, 12
    Md = four_block_est(Y, N1, T1, 3)
    assert Md.shape == (30 - N1, 20 - T1)
    assert np.allclose(Md, M[N1:, T1:], atol=1e-8)


def test_four_block_noise_degrades_gracefully():
    Y, M = _low_rank(60, 40, 2, noise=0.05)
    Md = four_block_est(Y, 40, 25, 2)
    err = np.abs(Md - M[40:, 25:]).mean()
    assert err < 0.5                              # small noise -> small error
    assert np.all(np.isfinite(Md))


def test_four_block_rejects_degenerate_blocks():
    Y, _ = _low_rank(20, 15, 2)
    with pytest.raises(MlsynthConfigError):
        four_block_est(Y, 1, 10, 2)               # N1 < r -> U1'U1 singular
    with pytest.raises(MlsynthConfigError):
        four_block_est(Y, 10, 1, 2)               # T1 < r


# ------------------------------------------------------ staircase_blocks
def test_staircase_single_cohort():
    """One adoption time -> k = 2 blocks (never-treated, one cohort)."""
    A = np.ones((10, 8), dtype=int)
    A[6:, 5:] = 0                                 # 4 units treated from t=5
    order, inv, N_all, T_all, k = staircase_blocks(A)
    assert k == 2
    assert list(T_all) == [5, 8]                  # control lengths present
    assert N_all[-1] == 10                        # cumulative unit counts
    assert np.array_equal(A[order][inv], A)       # permutation round-trips


def test_staircase_orders_by_treatment_duration():
    A = np.ones((6, 10), dtype=int)
    A[0, 4:] = 0                                  # longest treated
    A[1, 7:] = 0
    A[2, 9:] = 0
    order, inv, N_all, T_all, k = staircase_blocks(A)
    dur = (A.shape[1] - A.sum(axis=1))[order]
    assert list(dur) == sorted(dur)               # ascending duration after sort
    assert k == 4                                 # 3 cohorts + never-treated


def test_staircase_rejects_no_treated():
    with pytest.raises(MlsynthDataError):
        staircase_blocks(np.ones((5, 5), dtype=int))


def test_staircase_rejects_all_treated():
    A = np.ones((5, 6), dtype=int)
    A[:, 2:] = 0                                  # every unit treated, no controls
    with pytest.raises(MlsynthDataError):
        staircase_blocks(A)


# ---------------------------------------------------- entrywise_variance
def test_entrywise_variance_nonnegative_and_finite():
    Y, _ = _low_rank(40, 30, 2, noise=0.2)
    A = np.ones((40, 30), dtype=int)
    A[25:, 20:] = 0
    var = entrywise_variance(Y, A, r=2)
    assert var.shape == Y.shape
    treated = ~A.astype(bool)
    assert np.all(var[treated] >= 0)
    assert np.all(np.isfinite(var[treated]))


def test_entrywise_variance_scales_with_noise():
    """More noise -> wider intervals (variance is monotone in sigma)."""
    A = np.ones((50, 36), dtype=int)
    A[30:, 24:] = 0
    treated = ~A.astype(bool)
    v_small = entrywise_variance(_low_rank(50, 36, 2, noise=0.05)[0], A, r=2)
    v_large = entrywise_variance(_low_rank(50, 36, 2, noise=0.50)[0], A, r=2)
    assert v_large[treated].mean() > v_small[treated].mean()


def test_entrywise_variance_uses_sorted_mask():
    """Regression guard for the row-alignment bug in the reference package.

    The residual mask (paper eq. 6a) must be applied in the sorted staircase
    frame. Shuffling the input rows is a relabelling: the per-unit variances
    must come back identical, which only holds if the mask is sorted with the
    data. The reference's ``CAST-panel`` 1.0.2 masks sorted residuals with the
    unsorted indicator and fails this.
    """
    Y, _ = _low_rank(40, 28, 2, noise=0.3)
    A = np.ones((40, 28), dtype=int)
    A[20:30, 18:] = 0
    A[30:, 22:] = 0                               # two cohorts -> non-trivial sort
    var = entrywise_variance(Y, A, r=2)

    perm = np.random.default_rng(7).permutation(40)
    var_perm = entrywise_variance(Y[perm], A[perm], r=2)
    assert np.allclose(var_perm, var[perm], atol=1e-10)
