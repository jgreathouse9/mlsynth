"""Differentially private synthetic-control mechanisms (Rho, Cummings & Misra 2023).

The base estimator is a ridge synthetic control -- pre-period ridge regression of
the treated series on the donors, extrapolated to the post-period. Privacy is
obtained through differentially private empirical risk minimization (Chaudhuri,
Monteleoni & Sarwate 2011; Kifer, Smith & Thakurta 2012):

* output perturbation (Algorithm 2): fit the ridge coefficients, then add
  high-dimensional Laplace noise calibrated to their sensitivity;
* objective perturbation (Algorithm 3): add a random linear term to the ridge
  objective and solve exactly.

A second, independent budget privatizes the post-intervention donor matrix
before the counterfactual is formed. Both mechanisms are randomized; a
``numpy.random.RandomState`` is threaded through so a fixed seed reproduces the
release exactly (and reproduces the authors' reference stream for
cross-validation).

Sensitivity is with respect to changing one *donor's* entire series -- the
transposed ("vertical") synthetic-control regression treats each time point, not
each donor, as a sample, so a donor is a whole column (Rho et al. 2023, Sec 3).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np


def _ridge_coefficients(design: np.ndarray, target: np.ndarray, ridge_alpha: float) -> np.ndarray:
    """Closed-form ridge ``argmin_w ||target - design w||^2 + ridge_alpha ||w||^2``.

    Matches ``sklearn.linear_model.Ridge(alpha=ridge_alpha, fit_intercept=False)``
    to machine precision.
    """
    n_features = design.shape[1]
    gram = design.T @ design + ridge_alpha * np.eye(n_features)
    return np.linalg.solve(gram, design.T @ target)


def _high_dim_laplace(rng: np.random.RandomState, dim: int, scale: float) -> np.ndarray:
    """High-dimensional Laplace mechanism (Chaudhuri et al. 2011, Alg 1).

    A single random vector whose magnitude is ``Lap(scale)`` and whose direction
    is uniform on the unit ``dim``-sphere. Identical to the authors'
    ``laplace_sample(dim, 1, scale)``; the RNG draw order (magnitude then
    direction) is preserved so a shared seed reproduces their stream.
    """
    magnitude = rng.laplace(loc=0.0, scale=scale, size=1)[0]
    direction = rng.normal(0.0, 1.0, size=dim)
    direction = direction / np.sqrt(np.sum(direction ** 2))
    return magnitude * direction


def output_sensitivity(num_pre: int, num_donors: int, ridge_lambda: float) -> float:
    """L2 sensitivity of the ridge coefficients to one donor's series (Rho et al. 2023).

    ``Delta = 4 T0 sqrt(8 + N0) / lambda``.
    """
    return (4.0 * num_pre * math.sqrt(8.0 + num_donors)) / ridge_lambda


def _privatize_donors(
    rng: np.random.RandomState, donor_matrix: np.ndarray, epsilon2: float
) -> Tuple[np.ndarray, float]:
    """Add the Stage-2 high-dimensional Laplace noise to the donor matrix.

    Scale ``b = 2 sqrt(R) / epsilon2`` where ``R`` is the number of released
    rows (Rho et al. 2023). Mirrors the reference implementation, which
    privatizes the whole released donor path.
    """
    num_rows, num_donors = donor_matrix.shape
    scale = 2.0 * math.sqrt(num_rows) / epsilon2
    noise = _high_dim_laplace(rng, num_rows * num_donors, scale).reshape(num_rows, num_donors)
    return donor_matrix + noise, scale


def run_output_perturbation(
    rng: np.random.RandomState,
    pre_donor: np.ndarray,
    pre_target: np.ndarray,
    donor_full: np.ndarray,
    ridge_lambda: float,
    epsilon1: float,
    epsilon2: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Output perturbation (Algorithm 2): ridge coefficients + Laplace noise.

    Returns ``(counterfactual_path, private_weights, info)`` where
    ``counterfactual_path`` is over every row of ``donor_full``.
    """
    num_pre, num_donors = pre_donor.shape
    sensitivity = output_sensitivity(num_pre, num_donors, ridge_lambda)
    coef_scale = sensitivity / epsilon1
    coefficients = _ridge_coefficients(pre_donor, pre_target, ridge_lambda / 2.0)
    weights = coefficients + _high_dim_laplace(rng, num_donors, coef_scale)
    private_donors, donor_scale = _privatize_donors(rng, donor_full, epsilon2)
    counterfactual = private_donors @ weights
    info = {"sensitivity": sensitivity, "coef_scale": coef_scale,
            "donor_scale": donor_scale}
    return counterfactual, weights, info


def run_objective_perturbation(
    rng: np.random.RandomState,
    pre_donor: np.ndarray,
    pre_target: np.ndarray,
    donor_full: np.ndarray,
    ridge_lambda: float,
    epsilon1: float,
    epsilon2: float,
    delta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Objective perturbation (Algorithm 3): random linear term in the ridge objective.

    Solves ``argmin_f ||y - X f||^2 + ((lambda + Delta)/2) ||f||^2 + b^T f`` in
    closed form (the perturbed normal equations), where ``b`` is Laplace (pure
    ``epsilon``-DP, ``delta = 0``) or Gaussian (``(epsilon, delta)``-DP). The
    curvature slack ``Delta`` guarantees privacy when the base regularization is
    too weak for the requested budget.
    """
    num_pre, num_donors = pre_donor.shape
    curvature = (1.0 + math.sqrt(16.0 * num_donors - 15.0)) * num_pre
    epsilon0 = epsilon1 - math.log(1.0 + (2.0 * curvature / ridge_lambda)
                                   + (curvature ** 2 / ridge_lambda ** 2))
    if epsilon0 > 0:
        curvature_slack = 0.0
    else:
        epsilon0 = epsilon1 / 2.0
        curvature_slack = curvature / (math.e ** (epsilon1 / 4.0) - 1.0) - ridge_lambda

    if delta > 0:
        noise_scale = (4.0 * num_pre * math.sqrt(8.0 + num_donors)
                       * math.sqrt(2.0 * math.log(2.0 / delta) + epsilon0) / epsilon0)
        linear = rng.multivariate_normal(np.zeros(num_donors),
                                         noise_scale * np.eye(num_donors))
    else:
        scale_a = 4.0 * num_pre * math.sqrt(8.0 + num_donors) / epsilon0
        scale_b = (curvature * math.sqrt(num_donors) + 4.0 * num_pre) / epsilon0
        noise_scale = min(scale_a, scale_b)
        linear = _high_dim_laplace(rng, num_donors, noise_scale)

    gram = 2.0 * pre_donor.T @ pre_donor + (ridge_lambda + curvature_slack) * np.eye(num_donors)
    weights = np.linalg.solve(gram, 2.0 * pre_donor.T @ pre_target - linear)
    private_donors, donor_scale = _privatize_donors(rng, donor_full, epsilon2)
    counterfactual = private_donors @ weights
    info = {"curvature": curvature, "epsilon0": epsilon0, "curvature_slack": curvature_slack,
            "noise_scale": noise_scale, "donor_scale": donor_scale}
    return counterfactual, weights, info


def non_private_counterfactual(
    pre_donor: np.ndarray, pre_target: np.ndarray, donor_full: np.ndarray, ridge_lambda: float
) -> Tuple[np.ndarray, np.ndarray]:
    """The non-private ridge synthetic control (the ``epsilon -> infinity`` limit)."""
    coefficients = _ridge_coefficients(pre_donor, pre_target, ridge_lambda / 2.0)
    return donor_full @ coefficients, coefficients
