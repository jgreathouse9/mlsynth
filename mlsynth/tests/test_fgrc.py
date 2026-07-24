"""Tests for the fGRC donor-clustering helper
(:mod:`mlsynth.utils.clustersc_helpers.rpca.fgrc`).

The numerical *correctness* of this port is pinned seam-by-seam against the
reference ``OptimGRC_C`` C routine in a separate differential suite that
requires R + the ``grc`` package. This file is the self-contained contract that
ships with the library: smoke, invariants, edge/degenerate inputs, and the
translated-exception failure modes required by ``agents_tests.md`` -- no R, no
compiled reference. The one correctness anchor kept here is a finite-difference
check of the analytic gradient (validates the core numerics standalone) plus
recovery of a well-separated ground-truth partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.clustersc_helpers.rpca import fgrc


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _separated(seed, n_per=8, k=3, nb=24, c1=2, sep=6.0, dist=8.0, noise=0.3):
    """A panel of trajectories with k well-separated groups in a c1-dim
    subspace plus a dominant shared 'disturbing' trend."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    t = np.linspace(0, 1, nb)
    dist_curve = np.sin(2 * np.pi * t)
    basis = np.vstack([np.sin(2 * np.pi * (j + 2) * t) for j in range(c1)])   # (c1, nb)
    means = rng.normal(size=(k, c1)) * sep
    X = np.zeros((n, nb)); truth = np.zeros(n, dtype=int)
    for j in range(k):
        for i in range(n_per):
            idx = j * n_per + i
            X[idx] = ((means[j] + rng.normal(0, noise, c1)) @ basis
                      + dist * rng.normal() * dist_curve + rng.normal(0, noise, nb))
            truth[idx] = j
    return X, truth


def _comembership(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


# --------------------------------------------------------------------------
# Smoke
# --------------------------------------------------------------------------
@pytest.mark.parametrize("c2", [0, 1])
def test_smoke_runs_and_returns_valid_labels(c2):
    X, _ = _separated(1)
    labels, loss = fgrc.fgrc_cluster(X, c1=2, c2=c2, k=3, n_knots=8, order=4,
                                     n_random=8, nstart=8, seed=0)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset({1, 2, 3})
    assert np.isfinite(loss)


# --------------------------------------------------------------------------
# Correctness (self-contained: recovers a well-separated partition)
# --------------------------------------------------------------------------
def test_generalized_recovers_ground_truth_under_disturbing_structure():
    """The generalized model (c2=1) recovers the true partition when a dominant
    shared trend is present -- the subspace-separation contribution. Plain
    reduced k-means (c2=0) is *not* expected to here (that is the paper's point),
    so recovery is asserted only for c2=1."""
    X, truth = _separated(2)
    labels, _ = fgrc.fgrc_cluster(X, c1=2, c2=1, k=3, n_knots=8, order=4,
                                  n_random=16, nstart=16, seed=0)
    assert np.array_equal(_comembership(labels), _comembership(truth))


def test_deterministic_same_seed():
    X, _ = _separated(3)
    a, _ = fgrc.fgrc_cluster(X, 2, 1, 3, 8, 4, n_random=8, nstart=8, seed=7)
    b, _ = fgrc.fgrc_cluster(X, 2, 1, 3, 8, 4, n_random=8, nstart=8, seed=7)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Component invariants
# --------------------------------------------------------------------------
def _rand_stiefel(seed, Ns=20, Nb=6, c1=2, c2=1, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(Ns, Nb)); X = X - X.mean(0)
    A, _ = np.linalg.qr(rng.normal(size=(Nb, c1 + c2)))
    labels = rng.integers(1, k + 1, Ns); labels[:k] = np.arange(1, k + 1)
    return X, A, fgrc.indicator(labels, k), c1, c2


def test_gradient_matches_finite_difference():
    """The analytic gradient equals the numerical gradient of the (quadratic)
    Stiefel-restricted loss it is defined against -- validates the core numerics
    with no reference implementation."""
    X, A, U, c1, c2 = _rand_stiefel(4)
    Nb, nc = A.shape
    M = U @ np.linalg.solve(U.T @ U, U.T)                       # rho1=1, rho2=0 -> M = P_U
    def L2(Amat):
        A1 = Amat[:, :c1]
        v = np.trace(X.T @ X) - np.trace(A1.T @ X.T @ M @ X @ A1)
        if nc > c1:
            A2 = Amat[:, c1:]; v -= np.trace(A2.T @ X.T @ X @ A2)
        return v
    G = fgrc.grc_grad(X, U, A, c1, 1.0, 0.0)
    h = 1e-6; Gfd = np.zeros_like(A)
    for i in range(Nb):
        for j in range(nc):
            Ap = A.copy(); Ap[i, j] += h; Am = A.copy(); Am[i, j] -= h
            Gfd[i, j] = (L2(Ap) - L2(Am)) / (2 * h)
    assert np.allclose(G, Gfd, atol=1e-5)


def test_loss_invariant_to_within_block_rotation():
    X, A, U, c1, c2 = _rand_stiefel(5)
    rng = np.random.default_rng(9)
    Q1, _ = np.linalg.qr(rng.normal(size=(c1, c1)))
    Q2, _ = np.linalg.qr(rng.normal(size=(c2, c2)))
    Arot = np.hstack([A[:, :c1] @ Q1, A[:, c1:] @ Q2])
    assert np.isclose(fgrc.grc_loss(X, U, A, c1, 1.0, 0.0),
                      fgrc.grc_loss(X, U, Arot, c1, 1.0, 0.0))


def test_gp_reduces_loss_and_returns_orthonormal():
    X, A, U, c1, c2 = _rand_stiefel(6)
    l0 = fgrc.grc_loss(X, U, A, c1, 1.0, 0.0)
    A1 = fgrc.gp_algorithm(X, U, A, c1, 1.0, 0.0)
    assert fgrc.grc_loss(X, U, A1, c1, 1.0, 0.0) <= l0 + 1e-9
    assert np.allclose(A1.T @ A1, np.eye(c1 + c2), atol=1e-8)


def test_eigen_update_orthonormal_and_reduces_loss():
    X, A, U, c1, _ = _rand_stiefel(7, c2=0)
    A2 = fgrc.eigen_update(X, U, c1, 1.0, 0.0)
    assert A2.shape[1] == c1
    assert np.allclose(A2.T @ A2, np.eye(c1), atol=1e-8)
    assert fgrc.grc_loss(X, U, A2, c1, 1.0, 0.0) <= fgrc.grc_loss(X, U, A, c1, 1.0, 0.0) + 1e-9


def test_basis_expand_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 24))
    G = fgrc.basis_expand(X, n_knots=8, order=4)
    assert G.shape == (10, 8 + 4 - 2)      # Nb = n_knots + order - 2


def test_indicator_is_one_hot():
    U = fgrc.indicator(np.array([1, 2, 2, 3]), 3)
    assert np.array_equal(U.sum(1), np.ones(4))
    assert np.array_equal(U.argmax(1) + 1, np.array([1, 2, 2, 3]))


# --------------------------------------------------------------------------
# Edge / degenerate
# --------------------------------------------------------------------------
def test_edge_k_equals_n_units():
    X, _ = _separated(8, n_per=1, k=4, nb=20)     # 4 units, k=4
    labels, loss = fgrc.fgrc_cluster(X, 2, 1, 4, n_knots=6, order=4,
                                     n_random=6, nstart=6, seed=0)
    assert sorted(np.unique(labels)) == [1, 2, 3, 4]
    assert np.isfinite(loss)


def test_edge_minimal_panel():
    X, _ = _separated(9, n_per=2, k=2, nb=12)     # 4 units, 12 periods
    labels, _ = fgrc.fgrc_cluster(X, 1, 0, 2, n_knots=5, order=4,
                                  n_random=6, nstart=6, seed=0)
    assert set(np.unique(labels)).issubset({1, 2})


def test_edge_center_false_runs():
    X, _ = _separated(10)
    labels, _ = fgrc.fgrc_cluster(X, 2, 1, 3, 8, 4, center=False,
                                  n_random=6, nstart=6, seed=0)
    assert labels.shape == (X.shape[0],)


# --------------------------------------------------------------------------
# Failure — translated exceptions, never a raw NumPy/LinAlg leak
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,exc", [
    (dict(trajectories=np.ones(5), c1=2, c2=1, k=2, n_knots=8),           MlsynthDataError),      # 1D
    (dict(trajectories=np.ones((6, 20)), c1=0, c2=1, k=2, n_knots=8),     MlsynthConfigError),    # c1<1
    (dict(trajectories=np.ones((6, 20)), c1=2, c2=-1, k=2, n_knots=8),    MlsynthConfigError),    # c2<0
    (dict(trajectories=np.ones((6, 20)), c1=2, c2=1, k=1, n_knots=8),     MlsynthConfigError),    # k<2
    (dict(trajectories=np.ones((6, 20)), c1=2, c2=1, k=99, n_knots=8),    MlsynthConfigError),    # k>n_units
    (dict(trajectories=np.ones((6, 20)), c1=2, c2=1, k=2, n_knots=8,
          rho1=0.0, rho2=0.0),                                            MlsynthConfigError),    # rho1<=rho2
    (dict(trajectories=np.random.default_rng(0).normal(size=(10, 20)),
          c1=5, c2=3, k=2, n_knots=4, order=4),                          MlsynthConfigError),    # c1+c2 > n_basis
])
def test_fgrc_cluster_invalid_input_raises_translated(kwargs, exc):
    with pytest.raises(exc):
        fgrc.fgrc_cluster(**kwargs, n_random=4, nstart=4, seed=0)


@pytest.mark.parametrize("X,nk,no,exc", [
    (np.ones(5),        8, 4, MlsynthDataError),      # 1D
    (np.ones((4, 20)),  1, 4, MlsynthConfigError),    # n_knots < 2
    (np.ones((4, 20)),  8, 1, MlsynthConfigError),    # order < 2
    (np.ones((4, 3)),   4, 4, MlsynthConfigError),    # n_time < order
])
def test_basis_expand_invalid_input_raises_translated(X, nk, no, exc):
    with pytest.raises(exc):
        fgrc.basis_expand(X, nk, no)


def test_basis_expand_singular_gram_reports_translated_error():
    """More basis functions than time points makes PhiᵀPhi singular; the inverse
    failure must surface as MlsynthEstimationError, not a raw LinAlgError."""
    X = np.random.default_rng(0).normal(size=(6, 10))
    with pytest.raises(MlsynthEstimationError, match="basis expansion failed"):
        fgrc.basis_expand(X, n_knots=10, order=4)     # Nb = 12 > n_time = 10


def test_gp_line_search_exhaustion_reverts():
    """When no backtracking step improves the loss (here forced by starting at a
    converged optimum with a single allowed halving), GP hits the revert branch
    and returns the incoming solution unchanged."""
    X, A, U, c1, c2 = _rand_stiefel(11)
    A_star = fgrc.gp_algorithm(X, U, A, c1, 1.0, 0.0)          # converge
    loss_star = fgrc.grc_loss(X, U, A_star, c1, 1.0, 0.0)
    A_again = fgrc.gp_algorithm(X, U, A_star, c1, 1.0, 0.0, n_max_alpha=1)
    assert np.allclose(A_again, A_star, atol=1e-10)            # reverted, unchanged
    assert np.isclose(fgrc.grc_loss(X, U, A_again, c1, 1.0, 0.0), loss_star, atol=1e-10)


def test_degenerate_identical_trajectories_reports_translated_error():
    """Collinear (identical) trajectories cannot form k non-empty clusters. The
    failure must surface as a translated MlsynthEstimationError with a message,
    not a raw LinAlgError leaked from the projection solve."""
    X = np.ones((6, 20))
    with pytest.raises(MlsynthEstimationError, match="non-empty clusters"):
        fgrc.fgrc_cluster(X, c1=2, c2=1, k=2, n_knots=8, order=4,
                          n_random=4, nstart=4, seed=0)
