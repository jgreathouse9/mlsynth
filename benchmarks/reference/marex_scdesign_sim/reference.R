# SCDesign's own Section 5 simulation, run as the reference for MAREX.
#
# Three pieces of github.com/jinglongzhao2/SCDesign, reproduced verbatim:
#
#   * the model variables and `Generate_Model_Primitives` from
#     "2. Many Simulations (...)/Main_LazyRun.R" lines 48-98 and 103-263 --
#     the data-generating process behind Table 2; and
#   * `Synthetic_Control` + `Synthetic_Experiment_Cardinality_Constraint` from
#     "3. Walmart Data Simulations (...)/Walmart_LazyRun.R" -- the constrained
#     design, which runs on quadprog and needs no commercial solver.
#
# Seeding follows the authors' driver exactly. R_job.qsub submits an SGE array
# job (`#$ -t 1-1000`) and Main_LazyRun.R takes
# `repetition.RANDOM.SEED = as.numeric(Sys.getenv("SGE_TASK_ID"))`, so
# simulation i is set.seed(i) -- 1000 independent streams, not one advancing
# stream. Under this convention the DGP reproduces Table 2's tau_t panel to the
# printed precision: -13.5763 -10.9889 -8.3521 -4.9981 -2.4999 against the
# paper's -13.58 -10.99 -8.35 -5.00 -2.50.
#
# SCDesign's headline design is a non-convex Gurobi MIQP and is NOT used here.
#
# Run from the repository root:
#   Rscript benchmarks/reference/marex_scdesign_sim/reference.R
suppressMessages({library(Matrix); library(quadprog)})
N.PANELS <- 12
K.CARD <- 2

#=========================================#
#=====Below we define model variables=====#
#=========================================#

#===Basics
Weight.digits = 2
#Basics --- Scalars
N.Regions = 15
T.naught = 25
T.prime = 20
T.total = 30
r.ob.covariates.dim = 7
F.unob.covariates.dim = 11

#===Model Primitives
#Population weights --- Vectors
beta.vector = c()
#Constants --- Vectors
delta.N.vector = c()
upsilon.I.vector = c()
#Covariates --- Matrices
Z.ob.covariates.matrix = matrix(NA, nrow = r.ob.covariates.dim, ncol = N.Regions)
mu.unob.covariates.matrix = matrix(NA, nrow = F.unob.covariates.dim, ncol = N.Regions)
#Coefficients --- Matrices
theta.ob.N.matrix = matrix(NA, nrow = r.ob.covariates.dim, ncol = T.total)
gamma.ob.I.matrix = matrix(NA, nrow = r.ob.covariates.dim, ncol = T.total)
lambda.unob.N.matrix = matrix(NA, nrow = F.unob.covariates.dim, ncol = T.total)
eta.unob.I.matrix = matrix(NA, nrow = F.unob.covariates.dim, ncol = T.total)
#Noises --- Matrices
epsilon.N.matrix = matrix(NA, nrow = N.Regions, ncol = T.total)
xi.I.matrix = matrix(NA, nrow = N.Regions, ncol = T.total)

#===Potential Outcomes
#Potential Outcomes in hindsight --- Matrices
Y.N.matrix = matrix(NA, nrow = N.Regions, ncol = T.total)
Y.I.matrix = matrix(NA, nrow = N.Regions, ncol = T.total)

#===Define the range of parameters
#Constants --- Vectors
#Range --- [0,20]
range.intercept.max = 20 #ChangeThis
#Covariates --- Matrices
#Range --- [0,1]
range.covariates.max = 1 #ChangeThis
#Coefficients --- Matrices
#Range --- [0,10]
range.coefficients.max = 10 #ChangeThis
#Noises --- Matrices
#Normal Distribution ~ N(0,1)
noise.variance = 1 #ChangeThis

Generate_Model_Primitives <- function(range.intercept.max_ = range.intercept.max,
                                      range.covariates.max_ = range.covariates.max,
                                      range.coefficients.max_ = range.coefficients.max,
                                      noise.variance_ = noise.variance,
                                      random.seed_)
{
  set.seed(random.seed_)
  #===Initialize Model Primitives Again
  #Constants --- Vectors
  delta.N.vector_ = c()
  upsilon.I.vector_ = c()
  #Covariates --- Matrices
  Z.ob.covariates.matrix_ = matrix(NA, nrow = r.ob.covariates.dim, ncol = N.Regions)
  mu.unob.covariates.matrix_ = matrix(NA, nrow = F.unob.covariates.dim, ncol = N.Regions)
  #Coefficients --- Matrices
  theta.ob.N.matrix_ = matrix(NA, nrow = r.ob.covariates.dim, ncol = T.total)
  gamma.ob.I.matrix_ = matrix(NA, nrow = r.ob.covariates.dim, ncol = T.total)
  lambda.unob.N.matrix_ = matrix(NA, nrow = F.unob.covariates.dim, ncol = T.total)
  eta.unob.I.matrix_ = matrix(NA, nrow = F.unob.covariates.dim, ncol = T.total)
  #Noises --- Matrices
  epsilon.N.matrix_ = matrix(NA, nrow = N.Regions, ncol = T.total)
  xi.I.matrix_ = matrix(NA, nrow = N.Regions, ncol = T.total)
  #Potential Outcomes in hindsight --- Matrices
  Y.N.matrix_ = matrix(NA, nrow = N.Regions, ncol = T.total)
  Y.I.matrix_ = matrix(NA, nrow = N.Regions, ncol = T.total)
  
  #=====Generate Random Values=====#
  #===Basics
  #Population weights --- Vectors
  # beta.temp = runif(N.Regions)
  # beta.vector_ = beta.temp / sum(beta.temp)
  beta.vector_ = rep(1/N.Regions, N.Regions)
  #===Model Primitives
  #Constants --- Vectors
  #Range --- [0,20] #range.intercept.max_ = 20
  delta.N.vector_ = sort(c(range.intercept.max_ * runif(T.naught), range.intercept.max_ * runif(T.total - T.naught)))
  upsilon.I.vector_ = c(rep(NA, T.naught), sort(range.intercept.max_ * runif(T.total - T.naught)))
  #Covariates --- Matrices
  #Range --- [0,1] # range.covariates.max_ = 1
  for(j in 1:N.Regions)
  {
    z.temp = range.covariates.max_ * runif(r.ob.covariates.dim)
    Z.ob.covariates.matrix_[,j] = z.temp
  }
  colnames(Z.ob.covariates.matrix_) = c(1:N.Regions)
  for(j in 1:N.Regions)
  {
    mu.temp = range.covariates.max_ * runif(F.unob.covariates.dim)
    mu.unob.covariates.matrix_[,j] = mu.temp
  }
  colnames(mu.unob.covariates.matrix_) = c(1:N.Regions)
  #Coefficients --- Matrices
  #Range --- [0,10] # range.coefficients.max_ = 10
  for(t in 1:T.naught)
  {
    theta.temp = range.coefficients.max_ * runif(r.ob.covariates.dim)
    theta.ob.N.matrix_[,t] = theta.temp
  }
  for(t in (T.naught+1):T.total)
  {
    theta.temp = range.coefficients.max_ * runif(r.ob.covariates.dim)
    theta.ob.N.matrix_[,t] = theta.temp
  }
  colnames(theta.ob.N.matrix_) = c(1:T.total)
  
  for(t in 1:T.naught)
  {
    gamma.temp = rep(NA, r.ob.covariates.dim)
    gamma.ob.I.matrix_[,t] = gamma.temp
  }
  for(t in (T.naught+1):T.total)
  {
    gamma.temp = range.coefficients.max_ * runif(r.ob.covariates.dim)
    gamma.ob.I.matrix_[,t] = gamma.temp
  }
  colnames(gamma.ob.I.matrix_) = c(1:T.total)
  
  for(t in 1:T.naught)
  {
    lambda.temp = range.coefficients.max_ * runif(F.unob.covariates.dim)
    lambda.unob.N.matrix_[,t] = lambda.temp
  }
  for(t in (T.naught+1):T.total)
  {
    lambda.temp = range.coefficients.max_ * runif(F.unob.covariates.dim)
    lambda.unob.N.matrix_[,t] = lambda.temp
  }
  colnames(lambda.unob.N.matrix_) = c(1:T.total)
  
  for(t in 1:T.naught)
  {
    eta.temp = rep(NA, F.unob.covariates.dim)
    eta.unob.I.matrix_[,t] = eta.temp
  }
  for(t in (T.naught+1):T.total)
  {
    eta.temp = range.coefficients.max_ * runif(F.unob.covariates.dim)
    eta.unob.I.matrix_[,t] = eta.temp
  }
  colnames(eta.unob.I.matrix_) = c(1:T.total)
  
  #Noises --- Matrices
  #Normal Distribution ~ N(0,1) #noise.variance_ = 1
  for(t in 1:T.naught)
  {
    epsilon.temp = rnorm(N.Regions, 0, noise.variance_)
    epsilon.N.matrix_[,t] = epsilon.temp
  }
  for(t in (T.naught+1):T.total)
  {
    epsilon.temp = rnorm(N.Regions, 0, noise.variance_)
    epsilon.N.matrix_[,t] = epsilon.temp
  }
  colnames(epsilon.N.matrix_) = c(1:T.total)
  
  for(t in 1:T.naught)
  {
    xi.temp = rep(NA, N.Regions)
    xi.I.matrix_[,t] = xi.temp
  }
  for(t in (T.naught+1):T.total)
  {
    xi.temp = rnorm(N.Regions, 0, noise.variance_)
    xi.I.matrix_[,t] = xi.temp
  }
  colnames(xi.I.matrix_) = c(1:T.total)
  
  #=====Generate Potential Outcomes=====#
  #Potential Outcomes in hindsight --- Matrices
  #A Factor Model
  for(j in 1:N.Regions)
  {
    for(t in 1:T.total)
    {
      Y.N.matrix_[j,t] = delta.N.vector_[t] + theta.ob.N.matrix_[,t] %*% Z.ob.covariates.matrix_[,j] + lambda.unob.N.matrix_[,t] %*% mu.unob.covariates.matrix_[,j] + epsilon.N.matrix_[j,t]
    }
  }
  
  for(j in 1:N.Regions)
  {
    for(t in 1:T.total)
    {
      Y.I.matrix_[j,t] = upsilon.I.vector_[t] + gamma.ob.I.matrix_[,t] %*% Z.ob.covariates.matrix_[,j] + eta.unob.I.matrix_[,t] %*% mu.unob.covariates.matrix_[,j] + xi.I.matrix_[j,t]
    }
  }
  
  returned = list(beta.vector = beta.vector_, 
                  delta.N.vector = delta.N.vector_,
                  upsilon.I.vector = upsilon.I.vector_,
                  Z.ob.covariates.matrix = Z.ob.covariates.matrix_,
                  mu.unob.covariates.matrix = mu.unob.covariates.matrix_,
                  theta.ob.N.matrix = theta.ob.N.matrix_,
                  gamma.ob.I.matrix = gamma.ob.I.matrix_,
                  lambda.unob.N.matrix = lambda.unob.N.matrix_,
                  eta.unob.I.matrix = eta.unob.I.matrix_,
                  epsilon.N.matrix = epsilon.N.matrix_,
                  xi.I.matrix = xi.I.matrix_,
                  Y.N.matrix = Y.N.matrix_,
                  Y.I.matrix = Y.I.matrix_)
  return(returned)
}

# --- SCDesign Synthetic_Control (verbatim quadprog path, Walmart_LazyRun.R) ---
Synthetic_Control <- function(target.vector_, X.matrix_) {
  X.effective = X.matrix_ - matrix(data = target.vector_, nrow = nrow(X.matrix_),
                                   ncol = ncol(X.matrix_), byrow = FALSE)
  Dmat = 2 * (t(X.effective) %*% X.effective)
  pd_Dmat = as.matrix(nearPD(Dmat)$mat)
  Dmat.SizeReduced = pd_Dmat / mean(pd_Dmat)
  result = solve.QP(Dmat = Dmat.SizeReduced, dvec = rep(0, ncol(X.effective)),
                    Amat = t(rbind(matrix(1, nrow = 1, ncol = ncol(X.effective)),
                                   diag(1, ncol(X.effective)))),
                    bvec = c(1, rep(0, ncol(X.effective))), meq = 1)
  list(Weights = result$solution)
}

# --- SCDesign cardinality-constrained design (verbatim) ----------------------
Synthetic_Experiment_Cardinality_Constraint <- function(T.prime_, r.ob_, N.Regions_,
    Y.N.matrix_, Z.matrix_, f.vector_, K.cardinality_) {
  M.dim = T.prime_ + r.ob_; N.dim = N.Regions_
  X.matrix = matrix(NA, nrow = M.dim, ncol = N.dim)
  for (i in 1:M.dim) for (j in 1:N.dim) {
    if (i <= T.prime_) X.matrix[i, j] = Y.N.matrix_[j, i]
    else X.matrix[i, j] = Z.matrix_[(i - T.prime_), j]
  }
  row.means = apply(X.matrix, 1, mean); row.stdevs = apply(X.matrix, 1, sd)
  X.matrix = (X.matrix - matrix(row.means, nrow = M.dim, ncol = N.dim, byrow = FALSE)) / row.stdevs
  center.vector = X.matrix %*% f.vector_
  candidate.partition = list()
  for (pc in 1:K.cardinality_)
    candidate.partition = c(candidate.partition, combn(1:N.Regions_, pc, simplify = FALSE))
  loss = c()
  for (tc in seq_along(candidate.partition)) {
    cand = candidate.partition[[tc]]
    X.t = if (length(cand) == 1) matrix(X.matrix[, cand], ncol = 1) else X.matrix[, cand]
    X.c = X.matrix[, -cand]
    St = Synthetic_Control(center.vector, X.t); Sc = Synthetic_Control(center.vector, X.c)
    lt = (X.t %*% St$Weights - center.vector)^2; lc = (X.c %*% Sc$Weights - center.vector)^2
    loss = c(loss, mean(lt) + mean(lc))
  }
  fm = which.min(loss); cand = candidate.partition[[fm]]
  X.t = if (length(cand) == 1) matrix(X.matrix[, cand], ncol = 1) else X.matrix[, cand]
  X.c = X.matrix[, -cand]
  St = Synthetic_Control(center.vector, X.t); Sc = Synthetic_Control(center.vector, X.c)
  tw = rep(0, N.Regions_); for (i in seq_along(cand)) tw[cand[i]] = St$Weights[i]
  list(selected = sort(cand), tw = round(tw, 6), loss = min(loss))
}

f.vector <- rep(1 / N.Regions, N.Regions)
cat("== REFERENCE VALUES ==\n")
cat(sprintf("n_panels\t%.6f\n", N.PANELS))

# Table 2's first panel: the true effect path, averaged over the authors' own
# 1000 simulations (seeds 1..1000, the array-job task ids). Cheap -- no design
# is solved here, only the DGP.
TAU.SIMS <- 1000
tau.acc <- rep(0, T.total - T.naught)
for (s in 1:TAU.SIMS) {
  mp <- Generate_Model_Primitives(random.seed_ = s)
  tau.acc <- tau.acc + colMeans(mp$Y.I.matrix[, (T.naught+1):T.total] -
                                mp$Y.N.matrix[, (T.naught+1):T.total])
}
cat(sprintf("tau_sims\t%.6f\n", TAU.SIMS))
for (k in 1:(T.total - T.naught))
  cat(sprintf("tau_bar_t%d\t%.6f\n", T.naught + k, tau.acc[k] / TAU.SIMS))
cat(sprintf("k_cardinality\t%.6f\n", K.CARD))
for (s in 1:N.PANELS) {
  mp <- Generate_Model_Primitives(random.seed_ = s)   # the driver's seed = task id
  YN <- mp$Y.N.matrix; YI <- mp$Y.I.matrix; Z <- mp$Z.ob.covariates.matrix
  r <- Synthetic_Experiment_Cardinality_Constraint(T.prime, r.ob.covariates.dim,
         N.Regions, YN, Z, f.vector, K.CARD)
  for (j in 1:N.Regions) for (t in 1:T.total)
    cat(sprintf("weight\tyn_s%d_u%d_t%d\t%.10f\n", s, j, t, YN[j, t]))
  for (j in 1:N.Regions) for (t in (T.naught+1):T.total)
    cat(sprintf("weight\tyi_s%d_u%d_t%d\t%.10f\n", s, j, t, YI[j, t]))
  for (k in 1:r.ob.covariates.dim) for (j in 1:N.Regions)
    cat(sprintf("weight\tz_s%d_k%d_u%d\t%.10f\n", s, k, j, Z[k, j]))
  cat(sprintf("n_treated_s%d\t%.6f\n", s, length(r$selected)))
  cat(sprintf("loss_s%d\t%.10f\n", s, r$loss))
  for (j in r$selected) cat(sprintf("weight\tsel_s%d_u%d\t%.6f\n", s, j, r$tw[j]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
