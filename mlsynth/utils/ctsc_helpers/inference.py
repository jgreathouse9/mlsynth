"""Wald / sign-flip inference for CTSC (Powell 2022, Section 4).

CTSC induces mechanical cross-unit correlation (each unit's control is
built from the others). The inference procedure forms unit-level moment
scores at the restricted (null-imposed) estimate and calibrates a Wald
statistic with a Rademacher (sign-flip) randomization distribution,
permitting arbitrary within-unit and cross-unit dependence (Canay,
Romano & Shaikh 2017; Powell 2019).

For treatment variable :math:`k`, the per-unit, per-period moment is
(paper eq. 10)

.. math::

   h_{it}^{(k)} = \\Bigl(D_{it}^{(k)} - \\sum_{j \\ne i} w_j^i D_{jt}^{(k)}\\Bigr)
     \\Bigl[ Y_{it} - D_{it}'\\alpha_i
            - \\sum_{j \\ne i} w_j^i (Y_{jt} - D_{jt}'\\alpha_j) \\Bigr],

with mean zero under the null. The unit score is the time average
:math:`s_i^{(k)} = \\tfrac{1}{T}\\sum_t h_{it}^{(k)}`.

This implementation uses the unit-level scores directly with the
sign-flip test (a valid randomization test under sign symmetry of the
unit scores); the paper's optional PCA orthogonalisation of the scores
is not applied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .structures import CTSCInference

_EPS = 1e-12


def _unit_scores(
    Y: np.ndarray, D: np.ndarray, b: np.ndarray, Phi: np.ndarray
) -> np.ndarray:
    """Per-unit moment scores ``s`` of shape ``(n, K)`` (paper eq. 10, 12)."""
    n, T, K = D.shape
    U = Y - np.einsum("itk,ik->it", D, b)            # untreated outcomes (n, T)
    resid = U - Phi @ U                               # (n, T)
    # exposure_it^k = D_it^k - sum_{j!=i} Phi_ij D_jt^k
    exposure = D - np.einsum("ij,jtk->itk", Phi, D)   # (n, T, K)
    s = np.einsum("itk,it->ik", exposure, resid) / T  # (n, K)
    return s


def _wald(scores: np.ndarray, signs: Optional[np.ndarray] = None) -> float:
    """Wald statistic from unit scores (n, K); optionally sign-flipped."""
    n, K = scores.shape
    s = scores if signs is None else scores * signs[:, None]
    mean_s = s.mean(axis=0)                            # (K,)
    centered = s - s.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(n - 1, 1) / n  # cov of the mean
    cov = cov + _EPS * np.eye(K)
    try:
        return float(mean_s @ np.linalg.solve(cov, mean_s))
    except np.linalg.LinAlgError:
        return float(mean_s @ mean_s / (_EPS + np.trace(cov) / K))


def sign_flip_wald_inference(
    Y: np.ndarray,
    D: np.ndarray,
    pi: np.ndarray,
    omega: np.ndarray,
    *,
    null_value: Optional[np.ndarray] = None,
    n_draws: int = 2000,
    random_state: int = 0,
) -> CTSCInference:
    """Run the sign-flip Wald test of ``H0: alpha^AE = null_value``.

    Re-fits CTSC under the average-effect restriction, forms the unit
    scores, and calibrates the Wald statistic by Rademacher sign flips.
    Also returns per-variable joint and marginal p-values and a
    score-spread standard error for the average effect.
    """
    from .estimate import fit_ctsc

    n, T, K = D.shape
    if null_value is None:
        null_value = np.zeros(K)
    null_value = np.asarray(null_value, dtype=float)

    restricted = fit_ctsc(
        Y, D, population_weights=pi, omega=omega, restrict_ae=null_value,
    )
    scores = _unit_scores(Y, D, restricted["alpha"], restricted["weights"])

    observed = _wald(scores)
    rng = np.random.default_rng(random_state)
    draws = rng.choice([-1.0, 1.0], size=(n_draws, n))
    flipped = np.array([_wald(scores, draws[d]) for d in range(n_draws)])
    p_joint = float((flipped >= observed - _EPS).mean())

    # Per-variable marginal p-values + score-spread SE of the average effect.
    se = np.sqrt(np.var(scores, axis=0, ddof=1) / n + _EPS)   # (K,) crude SE proxy
    wald_k = np.zeros(K)
    p_k = np.zeros(K)
    for k in range(K):
        sk = scores[:, k]
        obs_k = abs(sk.mean()) / (np.std(sk, ddof=1) / np.sqrt(n) + _EPS)
        flip_k = np.abs((draws * sk[None, :]).mean(axis=1)) / (
            np.std(sk, ddof=1) / np.sqrt(n) + _EPS)
        wald_k[k] = obs_k
        p_k[k] = float((flip_k >= obs_k - _EPS).mean())

    return CTSCInference(
        method="sign_flip_wald",
        null_value=null_value,
        wald_stat=wald_k,
        p_value=p_k,
        se=se,
        n_draws=int(n_draws),
    )
