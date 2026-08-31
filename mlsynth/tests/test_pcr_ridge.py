"""The PCR weight solver's regularisation is RSC equation (18).

Amjad, Shah & Shen (2018) define the regularised synthetic control as

    beta_hat(eta) = argmin_v ||Y_1^- - (M_hat^-)^T v||^2 + eta * sum_j |v_j|^q,

with ``q = 2`` the ridge case they analyse in Section 4 and ``q = 1`` the
LASSO. mlsynth spells the same objective ``lambda_penalty * ||w||_p ** q``,
so the two coincide when ``p == q``. The property case
``rsc_rank_condition_mc`` measures Theorems 3 and 7 across ``eta`` and its
docs page states that correspondence; these tests hold it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.clustersc_helpers.pcr.frequentist import solve_ols


@pytest.fixture
def problem():
    """A well-posed (T0, J) regression with T0 > J."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(80, 12))
    beta = rng.normal(size=12)
    x = M @ beta + rng.normal(scale=0.3, size=80)
    return M, x, beta


def _ridge_closed_form(M: np.ndarray, x: np.ndarray, eta: float) -> np.ndarray:
    """(M^T M + eta I)^{-1} M^T x -- the minimiser of RSC (18) at q = 2."""
    J = M.shape[1]
    return np.linalg.solve(M.T @ M + eta * np.eye(J), M.T @ x)


# ---------------------------------------------------------------------------
# eta = 0 is the paper's unregularised algorithm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eta", [None, 0, 0.0])
def test_no_penalty_is_the_pseudo_inverse(problem, eta):
    """Algorithm 1 Step 2 at eta = 0: beta_hat = pinv(M_hat^-) x."""
    M, x, _ = problem
    got = solve_ols(M, x, lambda_penalty=eta)
    assert got == pytest.approx(np.linalg.pinv(M) @ x)


def test_no_penalty_needs_no_norm_parameters(problem):
    """p and q are ignored on the unregularised path, which never solves."""
    M, x, _ = problem
    assert solve_ols(M, x, lambda_penalty=0.0, p=1.0, q=1.0) == pytest.approx(
        solve_ols(M, x)
    )


# ---------------------------------------------------------------------------
# q = 2: the ridge case Section 4 analyses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eta", [0.1, 1.0, 10.0, 100.0])
def test_ridge_matches_rsc_equation_18(problem, eta):
    """p = q = 2 solves ||x - Mw||^2 + eta ||w||_2^2.

    The tolerance is the conic solver's, not the formula's: a mis-scaled
    penalty (eta/2, or eta on the norm instead of its square) would move the
    solution by an order of magnitude at eta = 100, not by 1e-5.
    """
    M, x, _ = problem
    got = solve_ols(M, x, lambda_penalty=eta, p=2.0, q=2.0)
    assert got == pytest.approx(_ridge_closed_form(M, x, eta), abs=1e-4)


def test_ridge_is_the_default_norm(problem):
    """Omitting p and q gives ridge, so ``lambda_penalty`` alone is (18) at q=2."""
    M, x, _ = problem
    assert solve_ols(M, x, lambda_penalty=5.0) == pytest.approx(
        solve_ols(M, x, lambda_penalty=5.0, p=2.0, q=2.0)
    )


def test_ridge_shrinks_monotonically(problem):
    """||beta_hat(eta)|| is non-increasing in eta -- the shrinkage the
    property case reports alongside the error columns."""
    M, x, _ = problem
    norms = [np.linalg.norm(solve_ols(M, x, lambda_penalty=e, p=2.0, q=2.0))
             for e in (0.0, 1.0, 10.0, 100.0, 1000.0)]
    assert norms == sorted(norms, reverse=True)


def test_ridge_drives_the_solution_to_zero_in_the_limit(problem):
    """The eta -> infinity end of the sweep, where the retained rank stops
    mattering because the penalty dominates the fit."""
    M, x, _ = problem
    w = solve_ols(M, x, lambda_penalty=1e10, p=2.0, q=2.0)
    assert np.linalg.norm(w) < 1e-4


# ---------------------------------------------------------------------------
# q = 1: the LASSO of (18)
# ---------------------------------------------------------------------------

def test_lasso_sparsifies(problem):
    """p = q = 1 is (18) at q = 1, which zeroes coefficients as eta grows."""
    M, x, _ = problem
    counts = [int(np.sum(np.abs(solve_ols(M, x, lambda_penalty=e, p=1.0, q=1.0))
                         > 1e-6))
              for e in (1.0, 50.0, 500.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def test_rejects_a_non_matrix_donor_block():
    with pytest.raises(MlsynthEstimationError, match="2D"):
        solve_ols(np.arange(10.0), np.arange(10.0))


def test_rejects_a_length_mismatch():
    with pytest.raises(MlsynthEstimationError, match="Pre-period length"):
        solve_ols(np.ones((10, 3)), np.ones(8))


# ---------------------------------------------------------------------------
# Degenerate but well-defined
# ---------------------------------------------------------------------------

def test_a_single_donor_still_solves():
    M = np.arange(1.0, 21.0).reshape(20, 1)
    x = 2.0 * M[:, 0]
    assert solve_ols(M, x, lambda_penalty=1.0, p=2.0, q=2.0) == pytest.approx(
        _ridge_closed_form(M, x, 1.0), abs=1e-6
    )


def test_more_donors_than_periods_still_solves():
    """The over-parameterised regime, where eta is doing real work."""
    rng = np.random.default_rng(1)
    M = rng.normal(size=(10, 30))
    x = rng.normal(size=10)
    w = solve_ols(M, x, lambda_penalty=1.0, p=2.0, q=2.0)
    assert w.shape == (30,)
    assert np.all(np.isfinite(w))
