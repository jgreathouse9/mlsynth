// Multitask Gaussian Process synthetic control -- the reference Stan program.
// A verbatim transcription of the Gaussian model (code/stan/normal.stan) from the
// replication package of Ben-Michael, E., Arbour, D., Feller, A., Franks, A. &
// Raphael, S. (2023), "Estimating the effects of a California gun control program
// with multitask Gaussian processes," Annals of Applied Statistics 17(2), 985-1016.
// mlsynth's MTGP ports this model to NumPyro; this file is the ground truth the
// cross-validation benchmark (mtgp_california) runs live via rstan. Separable
// time x unit kernel: a global time-GP trend (L_global z_global), a rank-n_k_f
// intrinsic-coregionalization term (L_f z_f k_f) with a squared-exponential
// smoothness prior over time, and unit intercepts; population-scaled Gaussian
// noise; the treated post-period cells masked out of the likelihood (control_idx).
data {
  int<lower=1> N;      // number of periods
  int<lower=1> D;      // number of units
  int<lower=1> n_k_f;  // number of latent functions (coregionalization rank)
  vector[N] x;         // univariate covariate (time index)
  matrix[N, D] y;      // outcome (rate) matrix, periods x units
  matrix[N, D] inv_population;
  int num_treated;
  int control_idx[N * D - num_treated];
}
transformed data {
  // Normalize the time covariate
  real xmean = mean(x);
  real xsd = sd(x);
  real xn[N] = to_array_1d((x - xmean)/xsd);
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global;
  real<lower=0> lengthscale_f;  // lengthscale of f
  real<lower=0> sigma_f;        // scale of f
  real<lower=0> sigman;
  vector[D] state_offset;
  vector[N] z_global;
  matrix[N, n_k_f] z_f;
  matrix[n_k_f, D] k_f;
  real global_offset;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));

  // priors
  to_vector(z_f) ~ std_normal();
  to_vector(k_f) ~ std_normal();
  z_global ~ std_normal();
  lengthscale_f ~ inv_gamma(5, 5);
  lengthscale_global ~ inv_gamma(5, 5);
  sigma_f ~ normal(0, 1);
  sigma_global ~ normal(0, 1);
  sigman ~ normal(0, 1);
  state_offset ~ normal(0, 1);

  // likelihood over the CONTROL cells only (treated post-period masked)
  to_vector(y)[control_idx] ~ normal(
      to_vector(
        global_offset +
        rep_matrix(state_offset, N)' +
        rep_matrix(L_global * z_global, D) +
        L_f * z_f * k_f
      )[control_idx],
      sigman * sqrt(to_vector(inv_population))[control_idx]
  );
}
generated quantities {
  matrix[N, D] f;

  {
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    // the latent no-intervention surface: its treated column is the counterfactual
    f = global_offset + rep_matrix(state_offset, N)'
        + rep_matrix(L_global * z_global, D) + L_f * z_f * k_f;
  }
}
