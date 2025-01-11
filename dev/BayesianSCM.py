# Amjad 2018 recommend a Bayesian apporach to both prediction and counterfactual estimation, bia the Bayesian Ridge.

# Perhaps, for the Cluster SC class, we may apply the Bayesian estimation to the (un)clustered version, too. But how?

# Here is the code which runs the Ridge and predicts the credible intervals.... perhaps I must add an option,

# Do X if Frequentist (the default) or do Q if the user sepcifies Bayes.


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
