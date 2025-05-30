import numpy as np
from typing import Tuple
from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError


def BayesSCM(
    denoised_donor_matrix: np.ndarray,
    target_outcome_pre_intervention: np.ndarray,
    observation_noise_variance: float,
    weights_prior_precision: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implement the fully Bayesian synthetic control method.

    This function calculates the posterior distribution of synthetic control
    weights and the predictive distribution of the counterfactual outcome
    under a Bayesian framework. It assumes a Gaussian likelihood for the
    outcome and a Gaussian prior for the weights.

    Parameters
    ----------
    denoised_donor_matrix : np.ndarray
        Denoised donor matrix, representing the pre-intervention outcomes or
        predictors for donor units. Shape (num_pre_intervention_periods, num_donors),
        where num_pre_intervention_periods is the number of pre-intervention
        time periods and num_donors is the number of donor units.
    target_outcome_pre_intervention : np.ndarray
        Pre-intervention outcome vector for the treated unit.
        Shape (num_pre_intervention_periods,) or (num_pre_intervention_periods, 1).
    observation_noise_variance : float
        Variance of the noise term in the outcome model (scalar).
        This represents the observation noise :math:`\\sigma^2`.
    weights_prior_precision : float
        Prior precision parameter for the synthetic control weights (scalar).
        This is the inverse of the prior variance for the weights, assuming
        a zero-mean Gaussian prior :math:`\\beta \\sim N(0, \\text{weights_prior_precision}^{-1}I)`.

    Returns
    -------
    weights_posterior_mean : np.ndarray
        Posterior mean of the synthetic control weights. Shape (num_donors,).
    weights_posterior_covariance : np.ndarray
        Posterior covariance matrix of the synthetic control weights.
        Shape (num_donors, num_donors).
    counterfactual_predictive_mean_pre_intervention : np.ndarray
        Predictive mean of the counterfactual outcome for the treated unit
        during the pre-intervention periods. Shape (num_pre_intervention_periods,).
    counterfactual_predictive_variance_pre_intervention : np.ndarray
        Predictive variance of the counterfactual outcome for the treated unit
        during the pre-intervention periods. Shape (num_pre_intervention_periods,).

    Raises
    ------
    MlsynthDataError
        If input data types, shapes, or values are invalid.
    MlsynthEstimationError
        If matrix inversion fails during posterior computation.
    """
    # Input validation for all parameters.
    # Check denoised_donor_matrix: must be a 2D NumPy array, non-empty if donors exist, and contain no NaN/Inf.
    if not isinstance(denoised_donor_matrix, np.ndarray):
        raise MlsynthDataError("denoised_donor_matrix must be a NumPy array.")
    if denoised_donor_matrix.ndim != 2:
        raise MlsynthDataError("denoised_donor_matrix must be a 2D array.")
    if denoised_donor_matrix.shape[0] == 0 and denoised_donor_matrix.shape[1] > 0 : # Allow (0,0) for no donors, no periods
        raise MlsynthDataError("denoised_donor_matrix must have at least one pre-intervention period if donors exist.")
    if np.any(np.isnan(denoised_donor_matrix)) or np.any(np.isinf(denoised_donor_matrix)):
        raise MlsynthDataError("denoised_donor_matrix contains NaN or Inf values.")

    if not isinstance(target_outcome_pre_intervention, np.ndarray):
        raise MlsynthDataError("target_outcome_pre_intervention must be a NumPy array.")
    if target_outcome_pre_intervention.ndim not in [1, 2]:
        raise MlsynthDataError("target_outcome_pre_intervention must be a 1D or 2D array.")
    if target_outcome_pre_intervention.ndim == 2 and target_outcome_pre_intervention.shape[1] != 1:
        raise MlsynthDataError("2D target_outcome_pre_intervention must have shape (num_periods, 1).")
    
    # Flatten target_outcome_pre_intervention after initial ndim checks
    original_target_shape_0 = target_outcome_pre_intervention.shape[0]
    if target_outcome_pre_intervention.ndim > 1:
        target_outcome_pre_intervention = target_outcome_pre_intervention.flatten()

    if original_target_shape_0 != denoised_donor_matrix.shape[0]:
        raise MlsynthDataError(
            "Number of pre-intervention periods in denoised_donor_matrix "
            f"({denoised_donor_matrix.shape[0]}) must match target_outcome_pre_intervention "
            f"({original_target_shape_0})."
        )
    if np.any(np.isnan(target_outcome_pre_intervention)) or np.any(np.isinf(target_outcome_pre_intervention)):
        raise MlsynthDataError("target_outcome_pre_intervention contains NaN or Inf values.")

    if not isinstance(observation_noise_variance, (float, int)):
        raise MlsynthDataError("observation_noise_variance must be a float or int.")
    if observation_noise_variance <= 0:
        raise MlsynthDataError("observation_noise_variance must be positive.")

    if not isinstance(weights_prior_precision, (float, int)):
        raise MlsynthDataError("weights_prior_precision must be a float or int.")
    if weights_prior_precision < 0: # Allow zero for non-informative prior if matrix is invertible
        raise MlsynthDataError("weights_prior_precision must be non-negative.")

    # Compute dimensions: number of pre-intervention periods (T0) and number of donors (N_donors).
    num_pre_intervention_periods, num_donors = denoised_donor_matrix.shape
    
    # Bayesian Model Assumptions:
    # Likelihood: Y_target_pre ~ N(X_donors * beta, sigma_obs^2 * I)
    # Prior for weights: beta ~ N(0, lambda_prior^-1 * I)
    # where Y_target_pre is target_outcome_pre_intervention, X_donors is denoised_donor_matrix,
    # beta are the synthetic control weights, sigma_obs^2 is observation_noise_variance,
    # and lambda_prior is weights_prior_precision.

    # Handle edge case: if there are no donor units.
    # In this scenario, posterior mean and covariance for weights are effectively undefined or empty.
    # The predictive mean for the counterfactual would typically be the prior mean (assumed zero here),
    # and the predictive variance would be the observation noise variance.
    if num_donors == 0:
        weights_posterior_mean = np.array([]) # No weights to estimate.
        weights_posterior_covariance = np.empty((0, 0)) # Covariance is an empty 0x0 matrix.
        counterfactual_predictive_mean_pre_intervention = np.zeros(num_pre_intervention_periods)
        counterfactual_predictive_variance_pre_intervention = np.full(
            num_pre_intervention_periods, observation_noise_variance
        )
        return (
            weights_posterior_mean,
            weights_posterior_covariance,
            counterfactual_predictive_mean_pre_intervention,
            counterfactual_predictive_variance_pre_intervention,
        )

    # Compute posterior covariance of the synthetic control weights (beta).
    # Formula: Sigma_beta_post = (lambda_prior * I + (1/sigma_obs^2) * X_donors' * X_donors)^-1
    # This is derived from standard Bayesian linear regression results where:
    # - lambda_prior * I is the prior precision matrix (inverse of prior covariance).
    # - (1/sigma_obs^2) * X_donors' * X_donors is the precision from the likelihood.
    # The posterior precision is the sum of prior precision and likelihood precision.
    # The posterior covariance is the inverse of the posterior precision.
    try:
        # Construct the matrix representing the posterior precision of beta.
        posterior_precision_matrix = (
            weights_prior_precision * np.eye(num_donors)  # Prior precision part: lambda_prior * I
            + (1 / observation_noise_variance) * denoised_donor_matrix.T @ denoised_donor_matrix # Likelihood precision part
        )
        # Check for non-finite values before inversion, which can cause linalg.inv to fail or produce NaNs.
        if not np.all(np.isfinite(posterior_precision_matrix)):
             raise MlsynthEstimationError(
                "Matrix for posterior covariance inversion (posterior precision) contains non-finite values. "
                "Check input data and parameters."
            )
        # Invert the posterior precision matrix to get the posterior covariance.
        weights_posterior_covariance: np.ndarray = np.linalg.inv(posterior_precision_matrix)
        # Check if the resulting covariance matrix contains non-finite values.
        if not np.all(np.isfinite(weights_posterior_covariance)):
            raise MlsynthEstimationError(
                "Computed posterior covariance contains non-finite values. "
                "This can happen if the posterior precision matrix was numerically singular."
            )
    except np.linalg.LinAlgError as e: # Catch linear algebra errors during inversion (e.g., singular matrix).
        raise MlsynthEstimationError(
            "Failed to compute posterior covariance. Posterior precision matrix may be singular or not positive definite. "
            "Consider increasing weights_prior_precision or checking denoised_donor_matrix for collinearity."
        ) from e

    # Compute posterior mean of the synthetic control weights (beta).
    # Formula: mu_beta_post = Sigma_beta_post * ( (1/sigma_obs^2) * X_donors' * Y_target_pre + lambda_prior * mu_prior )
    # Since mu_prior (prior mean of beta) is assumed to be 0, the term lambda_prior * mu_prior vanishes.
    # So, mu_beta_post = Sigma_beta_post * (1/sigma_obs^2) * X_donors' * Y_target_pre
    # Corrected order of operations to avoid matmul error with scalar
    term_for_mean_calc = (1 / observation_noise_variance) * (denoised_donor_matrix.T @ target_outcome_pre_intervention)
    weights_posterior_mean: np.ndarray = weights_posterior_covariance @ term_for_mean_calc


    # Predict the counterfactual outcomes for the treated unit during the pre-intervention period using the posterior mean of weights.
    # This is the expected value of the synthetic control: E[y_cf_pre] = X_donors * E[beta] = X_donors * mu_beta_post.
    # This serves as an in-sample fit of the Bayesian SCM.
    counterfactual_predictive_mean_pre_intervention: np.ndarray = denoised_donor_matrix @ weights_posterior_mean

    # Compute the predictive variance for each pre-intervention time point.
    # The predictive distribution for a new outcome y_new = X_new * beta + epsilon is Gaussian.
    # Var(y_new) = Var(X_new * beta) + Var(epsilon)
    #            = X_new * Var(beta) * X_new' + sigma_obs^2
    #            = X_new * Sigma_beta_post * X_new' + sigma_obs^2
    # Here, X_new is a row of denoised_donor_matrix (X_donors_i).
    # We need the diagonal elements of (X_donors @ Sigma_beta_post @ X_donors').
    # The term np.sum((denoised_donor_matrix @ weights_posterior_covariance) * denoised_donor_matrix, axis=1)
    # efficiently calculates these diagonal elements: diag(A @ B @ A') where A is X_donors and B is Sigma_beta_post.
    variance_from_weights_uncertainty = np.sum(
        (denoised_donor_matrix @ weights_posterior_covariance) * denoised_donor_matrix, axis=1
    )
    counterfactual_predictive_variance_pre_intervention: np.ndarray = (
        observation_noise_variance + variance_from_weights_uncertainty
    )

    return weights_posterior_mean, weights_posterior_covariance, counterfactual_predictive_mean_pre_intervention, counterfactual_predictive_variance_pre_intervention
