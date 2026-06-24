L2relax = function(y1, X1, X2, tau, standardize = TRUE, intercept = TRUE, solver = "ECOS_BB") {
  
  suppressMessages(library(CVXR))
  
  N = ncol(X1)
  T1 = length(y1)
  
  # standardization
  
  if(intercept == TRUE) {
    mu1 = mean(y1)
    Mu1 = colMeans(X1)
  } else {
    mu1 = 0; Mu1 =0
  }
  
  if(standardize == TRUE) {
    sd1 = sd(y1)
    Sd1 = apply(X1, MARGIN = 2, FUN = sd)
  } else {
    sd1 = 1; Sd1 = 1
  }
  
  y1_tilde = (y1 - mu1) / sd1
  X1_tilde = t((t(X1) - Mu1) / Sd1)
  
  # Estimation
  
  Sigma = t(X1_tilde) %*% X1_tilde / T1
  eta = t(X1_tilde) %*% y1_tilde / T1
  
  beta = Variable(N)
  obj = sum_squares(beta)
  constr = list(eta - Sigma %*% beta <= tau,
                - eta + Sigma %*% beta <= tau)
  prob = Problem(objective = Minimize(obj), constraints = constr)
  result = solve(prob, solver = solver)
  beta_tilde = as.vector(result$getValue(beta))
  
  beta_hat = sd1 * (beta_tilde / Sd1)
  alpha_hat = mu1 - sum(Mu1 * beta_hat)
  
  # Prediction
  
  y1_hat = alpha_hat + as.vector(X1 %*% beta_hat)
  y2_hat = alpha_hat + as.vector(X2 %*% beta_hat)
  
  # output
  
  return(list(y1.pred = y1_hat, y2.pred = y2_hat,
              beta = beta_hat, alpha = alpha_hat))
}


