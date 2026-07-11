// Bayesian Factor Synthetic Control -- the reference Stan program.
// A verbatim transcription of the appendix Stan model in Pinkney, S. (2021),
// "An Improved and Extended Bayesian Synthetic Control," arXiv:2103.16244.
// mlsynth's BFSC ports this model to NumPyro; this file is the ground truth the
// cross-validation benchmarks (bfsc_prop99, and the frozen bfsc_germany numbers)
// run live via rstan. Lower-trapezoidal factor matrix, horseshoe+ loadings,
// subtract-last-pre / divide-by-pre-sd standardization, masked-treated post.
functions {
  matrix make_F(int T, vector diagonal_loadings, vector lower_tri_loadings) {
    int L = num_elements(diagonal_loadings);
    matrix[T, L] F;
    int idx = 0;
    for (j in 1:L) {
      F[j, j] = diagonal_loadings[j];
      for (i in (j + 1):T) { idx += 1; F[i, j] = lower_tri_loadings[idx]; }
    }
    for (j in 1:(L - 1)) for (i in (j + 1):L) F[j, i] = 0;
    return F;
  }
  matrix make_beta(int J, matrix off, vector lambda, real eta, vector tau) {
    int L = cols(off);
    vector[L] cache = ( tan(0.5 * pi() * lambda) * tan(0.5 * pi() * eta) );
    vector[J] tau_ = tan(0.5 * pi() * tau);
    matrix[J, L] out;
    for (j in 1:J) out[j] = off[j] * tau_[j];
    return diag_pre_multiply(cache, out');
  }
}
data {
  int T; int J; int L; int P;
  matrix[P, J] X;
  array[J] row_vector[T] Y;
  int trt_times;
}
transformed data {
  int<lower=1> M = L * (T - L) + L * (L - 1) / 2;
  row_vector[J] j_ones = rep_row_vector(1, J);
  vector[T] t_ones = rep_vector(1.0, T);
  matrix[J, P] X_std;
  vector[J] y_mu; vector[J] y_sd;
  array[J] row_vector[T] Y_scaled;
  row_vector[T - trt_times] Y_pre_target;
  for (j in 1:J) {
    y_mu[j] = Y[j, T - trt_times];
    y_sd[j] = sd(Y[j, 1:T - trt_times]);
    Y_scaled[j] = ( Y[j] - y_mu[j] ) / y_sd[j];
  }
  for (p in 1:P) X_std[, p] = ( X[p]' - mean(X[p]) ) / sd(X[p]);
  Y_pre_target = Y_scaled[1, 1:T - trt_times];
}
parameters {
  vector[P] chi;
  vector[T] delta;
  row_vector[J] kappa;
  matrix[J, L] beta_off;
  vector<lower=0, upper=1>[L] lambda;
  real<lower=0, upper=1> eta;
  vector<lower=0, upper=1>[J] tau;
  row_vector[trt_times] Y_post_target;
  real<lower=0> sigma;
  vector<lower=0>[L] F_diag;
  vector[M] F_lower;
}
transformed parameters {
  matrix[L, J] beta = make_beta(J, beta_off, lambda, eta, tau);
}
model {
  chi ~ std_normal();
  to_vector(beta_off) ~ std_normal();
  F_diag ~ std_normal();
  F_lower ~ normal(0, 2);
  delta ~ normal(0, 2);
  kappa ~ std_normal();
  sigma ~ std_normal();
  {
    vector[J] predictors = X_std * chi;
    matrix[T, L] F = make_F(T, F_diag, F_lower);
    array[1] row_vector[T] Y_target;
    array[J] row_vector[T] Y_temp;
    Y_target[1] = append_col(Y_pre_target, Y_post_target);
    Y_temp = append_array(Y_target, Y_scaled[2:J]);
    for (j in 1:J)
      Y_temp[j]' ~ normal_id_glm(F, delta + kappa[j] + predictors[j], beta[, j], sigma);
  }
}
generated quantities {
  array[J] vector[T] synth_out;
  matrix[T, L] F_ = make_F(T, F_diag, F_lower);
  matrix[T, J] Synth_ = F_ * beta + delta * j_ones + t_ones * (kappa + (X_std * chi)');
  for (j in 1:J) synth_out[j] = Synth_[, j] * y_sd[j] + y_mu[j];
}
