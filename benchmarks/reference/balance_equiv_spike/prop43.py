"""The paper's estimators, in the notation of Bruns-Smith et al. (2026).

Reference: Bruns-Smith, D., Dukes, O., Feller, A., & Ogburn, E. L. (2026).
"Augmented balancing weights as linear regression." JRSS-B 88(3), 699-723.
Replication package: github.com/bruns-smith/balance-equiv-jrssb.

The source population ``p`` supplies units whose outcomes are observed; the
target ``q`` supplies the feature vector the counterfactual mean is taken at.
Rows of ``Xp`` are source units, columns are features.
"""

from __future__ import annotations

import numpy as np


def _pinv_diag(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Reciprocal of ``values`` where they clear ``eps``, zero elsewhere.

    Mirrors ``pseudo_inv_diag`` in the authors' ``hyperparam.py``; a tolerance
    based ``np.linalg.pinv`` would keep numerically-zero directions.
    """
    out = np.zeros_like(values, dtype=float)
    keep = values > eps
    out[keep] = 1.0 / values[keep]
    return out


def gamma_map(evals: np.ndarray, lam: np.ndarray, delta: float) -> np.ndarray:
    """Proposition 4.3's ``gamma_j = delta * lambda_j / (sigma_j + lambda_j + delta)``.

    ``evals`` are the eigenvalues of the source second-moment matrix. The map
    is bounded above by ``lam``, which is the paper's undersmoothing statement:
    augmenting a ridge outcome model with l2 balancing weights is a single
    ridge at a strictly smaller penalty.
    """
    return delta * lam / (evals + lam + delta)


def generalized_ridge(Xp: np.ndarray, Yp: np.ndarray, diag: np.ndarray) -> np.ndarray:
    """Coefficients of a generalized ridge of ``Yp`` on ``Xp`` with penalty ``diag``.

    ``diag`` is expressed in the eigenbasis of ``Xp.T @ Xp / m``, so a scalar
    penalty is passed as a constant vector.
    """
    m = Xp.shape[0]
    evals, V = np.linalg.eigh(Xp.T @ Xp / m)
    inv = V @ np.diag(_pinv_diag(evals + diag)) @ V.T
    return inv @ Xp.T @ Yp / m


def l2_balancing_weights(Xp: np.ndarray, phi_q: np.ndarray, delta: float) -> np.ndarray:
    """Minimum-variance l2 balancing weights at penalty ``delta`` (Section 2.3).

    Returns ``w`` on the paper's scale, where the weighted source mean is
    ``Xp.T @ w / m``. ``delta -> 0`` drives that mean onto ``phi_q`` exactly.
    """
    m = Xp.shape[0]
    evals, V = np.linalg.eigh(Xp.T @ Xp / m)
    theta = phi_q @ V @ np.diag(_pinv_diag(evals + delta)) @ V.T
    return theta @ Xp.T


def augmented_estimate(Xp, Yp, phi_q, w, beta) -> float:
    """The augmented estimator, equation (7).

    Weighting term plus the outcome model evaluated on the residual feature
    shift. With ``beta`` from any outcome model and ``w`` from any weights.
    """
    m = Xp.shape[0]
    phi_shift = Xp.T @ w / m
    return float(w @ Yp / m + (phi_q - phi_shift) @ beta)


def ols_plugin(Xp: np.ndarray, Yp: np.ndarray, phi_q: np.ndarray) -> float:
    """Unpenalized OLS plug-in, the value the augmented estimator collapses to."""
    beta, *_ = np.linalg.lstsq(Xp, Yp, rcond=None)
    return float(phi_q @ beta)
