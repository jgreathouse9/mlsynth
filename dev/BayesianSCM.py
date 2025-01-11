# Amjad 2018 recommend a Bayesian apporach to both prediction and counterfactual estimation, bia the Bayesian Ridge.

# Perhaps, for the Cluster SC class, we may apply the Bayesian estimation to the (un)clustered version, too. But how?

# Here is the code which runs the Ridge and predicts the credible intervals.... perhaps I must add an option,

# Do X if Frequentist (the default) or do Q if the user sepcifies Bayes.

def BayesSCM(M_denoised, Y_target, sigma2, alpha):
    """
    Implements the fully Bayesian synthetic control method.
    
    Parameters:
        M_denoised (numpy.ndarray): Denoised donor matrix (T0 x (N-1)), 
                                    where T0 is the number of pre-intervention periods.
        Y_target (numpy.ndarray): Pre-intervention target vector (T0 x 1) for the treated unit.
        sigma2 (float): Noise variance.
        alpha (float): Prior precision parameter (inverse of prior variance).
        
    Returns:
        numpy.ndarray: Posterior mean of the weights (beta_D).
        numpy.ndarray: Posterior covariance of the weights (Sigma_D).
        numpy.ndarray: Predictive mean of the counterfactual (Y_pred).
        numpy.ndarray: Predictive variance of the counterfactual (Y_var).
    """
    # Compute dimensions
    T0, N_minus_1 = M_denoised.shape
    
    # Compute posterior covariance Sigma_D
    Sigma_D = np.linalg.inv(alpha * np.eye(N_minus_1) + (1 / sigma2) * M_denoised.T @ M_denoised)
    
    # Compute posterior mean beta_D
    beta_D = (1 / sigma2) * Sigma_D @ M_denoised.T @ Y_target
    
    # Predict counterfactual outcomes for the post-intervention period
    # (assuming M_post is the donor matrix in the post-intervention period)
    Y_pred = M_denoised @ beta_D  # Predict pre-intervention outcomes for validation
    
    # Predictive variance for each time point
    Y_var = sigma2 + np.sum((M_denoised @ Sigma_D) * M_denoised, axis=1)  # Element-wise sum
    
    return beta_D, Sigma_D, Y_pred, Y_var


Y0_rank, n2, u_rank, s_rank, v_rank = svt(X[:pre])
    
alpha = 1.0

beta_D, Sigma_D, Y_pred, Y_var = BayesSCM(Y0_rank, y[:pre], np.var(y[:pre], ddof=1), alpha)

num_samples = 1000  # Number of samples from the posterior predictive distribution
samples = np.random.multivariate_normal(beta_D, Sigma_D, size=num_samples)

# Compute counterfactual predictions for each sample
cf_samples = X @ samples.T  # Shape: (num_time_points, num_samples)

# Compute the 2.5th and 97.5th percentiles for a 95% credible interval
lower_bound = np.percentile(cf_samples, 2.5, axis=1)
upper_bound = np.percentile(cf_samples, 97.5, axis=1)

# Predictive mean is the median of the sampled counterfactuals
cf_mean = np.median(cf_samples, axis=1)

cf = np.dot(X,beta_D)
