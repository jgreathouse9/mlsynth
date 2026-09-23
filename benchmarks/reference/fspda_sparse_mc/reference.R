# Reference generator for benchmarks/cases/fspda_sparse_mc.py
#
# Shi & Huang (2023), the sparse simulation -- the second of the two studies in
# zhentaoshi/fsPDA's simulation/ directory, behind their appendix Tables B1-B3.
# Everything between the VERBATIM markers is copied unchanged from that
# repository at 5f4542c: simulation/sparse/{main_function.R, fs.R, lasso_ic.R,
# oracle.R, prd.R}.
#
# Three things about this study differ from the dense one next door, and all
# three are why it gets its own case instead of another cell in fspda_dense_mc.
#
#   The forward selection is not the same algorithm. sparse/fs.R maximises the
#   fitted sum of squares over ALL N donors at every step -- including ones
#   already picked -- scores its stopping rule on var(e), and discards its last
#   pick on exit. nonsparse/FS.R minimises mean(e^2) over the donors not yet
#   picked, scores log(mean(e^2)), and keeps everything it selects. mlsynth
#   implements the latter, which is also what the released est.fsPDA.R does.
#
#   The LASSO criterion is not the same either. lasso_ic's WIC uses var(y1-yhat)
#   where lasso.BIC uses colMeans(e.hat^2), and searches
#   seq(0.01, 0.5, length = 100) where lasso.BIC searches seq(0.01, 1, by=0.01).
#   mlsynth's lasso_criterion="mbic" reproduces lasso.BIC.
#
#   There is an oracle column, which regresses on the true support. It is a
#   floor for the other estimators, not an estimator anyone can run.
#
# So this bundle captures their outputs and lets the Python case measure how far
# mlsynth's rules land from rules that are deliberately different. Reporting
# those distances is the point; a match would be the surprise.
#
# Emitted per replication: the panel, their fs / lasso_ic / oracle coefficients,
# selected supports and post-period errors, and prd()'s Newey-West studentised
# statistic under the first of their seven shift patterns (the null).

suppressMessages(library(MASS))
suppressMessages(library(glmnet))
suppressMessages(library(sandwich))

ARGS <- commandArgs(trailingOnly = TRUE)
SRC <- if (length(ARGS) >= 1) ARGS[1] else Sys.getenv("FSPDA_SRC", "/workspace/zhentaoshi/fspda")
OUT <- if (length(ARGS) >= 2) ARGS[2] else "benchmarks/reference/fspda_sparse_mc"

# The nonsparse rules, sourced verbatim from the same repository. mlsynth
# implements these, not the sparse variants below.
source(file.path(SRC, "simulation", "nonsparse", "FS.R"))
lasso.BIC.t <- function(y, Y, T1, thresh, H = 2, grid = seq(0.01, 1, by = 0.01)) {
  N <- ncol(Y); B <- H * log(log(N)) * log(T1) / T1
  y1 <- y[1:T1]; Y1 <- Y[1:T1, ]
  f <- glmnet(x = Y1, y = y1, family = "gaussian", lambda = grid, intercept = FALSE,
              thresh = thresh, maxit = 1e7)
  IC <- log(colMeans((y1 - Y1 %*% f$beta)^2)) + B * f$df
  i <- which.min(IC)
  list(sel = which(f$beta[, i] != 0), lambda = f$lambda[i])
}

# ==================== VERBATIM: simulation/sparse/fs.R ====================
fs = function(t, t1, y, Y){

  pre.y = y[1:t1]
  post.y = y[(t1+1):t]

  PRE.Y = Y[1:t1,]

  N = dim(PRE.Y)[2]

  R2 = matrix(0, 1, N)
  for (j in 1:N){
    X = PRE.Y[,j]
    b = ginv(t(X) %*% X) %*% t(X) %*% pre.y
    SSE = (t(X %*% b)) %*% (X %*% b)

    R2[,j] = SSE
  }
  select = which.max(R2)

  X = PRE.Y[,select]
  b = ginv(t(X) %*% X) %*% t(X) %*% pre.y
  y.hat = as.vector (X %*% b)
  e.hat = pre.y - y.hat
  sigma.sq = var(e.hat)
  B = log(log(N))*log(t1)/t1
  Q.new = log(sigma.sq) + B
  Q.old = Q.new + 1

  # iteration after the first selction

  while (Q.new < Q.old ) {

    Q.old = Q.new

    R2 = matrix(0, 1, N)
    for (j in 1:N){
      X = cbind(PRE.Y [,select], PRE.Y[,j])
      b = ginv(t(X) %*% X) %*% t(X) %*% pre.y
      SSE = (t(X %*% b)) %*% (X %*% b)

      R2[,j] = SSE
    }

    index = which.max(R2)
    select = append(select, index)
    k = length(select)

    X = PRE.Y[,select]
    b = ginv(t(X) %*% X) %*% t(X) %*% pre.y
    y.hat = drop (X %*% b)
    e.hat = pre.y - y.hat
    sigma.sq = var(e.hat)
    B = log(log(N))*k*log(t1)/t1
    Q.new = log(sigma.sq) + B
  }

  select = select[1:(length(select)-1)]
  Y.SL = Y[,select]

  Y.SL = as.matrix(Y.SL)

  PRE.X = Y.SL[1:t1,]
  POST.X = Y.SL[(t1+1):t,]

  b.hat = ginv (t(PRE.X) %*% PRE.X) %*% t(PRE.X) %*% pre.y

  post.y.hat = drop(POST.X %*% b.hat)

  d = post.y - post.y.hat

  b = rep(0, N)
  b[select] = b.hat

  return( list(b = b, d = d) )
}
# ================== END VERBATIM: simulation/sparse/fs.R ==================

# ================= VERBATIM: simulation/sparse/lasso_ic.R =================
lasso_ic = function (t, t1, y, X, ic, cst){

  y1 = y[1:t1]
  y2 = y[(t1+1):t]

  X1 = X[1:t1,]
  X2 = X[(t1+1):t,]

  N = dim(X1)[2]

  lam = seq(0.01, 0.5, length = 100)

  las = glmnet(x = X1, y = y1, standardize = T, intercept = F,
                    family="gaussian",lambda = lam)

  B = as.matrix(las$beta)
  yhat = predict(las, newx = X1, type = "link")
  sigma.sq = apply(yhat, MARGIN = 2, function(x)(var(y1-x)))
  k = las$df

  if (ic == "WIC"){
    Q = log(sigma.sq) + cst*(log(log(N)))*(log(t1)/t1)*k
  } else if (ic == "BIC"){
    Q = sigma.sq + cst*log(t1)/t1*k
  } else {
    Q = sigma.sq + 2*k/t1 #AIC
  }

  b = B[,which.min(Q)]

  mse = mean( (y1 - X1 %*% b)^2 )
  y2.hat = drop (X2 %*% b)
  d = y2 - y2.hat

  return( list(b = b , d = d, mse = mse) )
}
# =============== END VERBATIM: simulation/sparse/lasso_ic.R ===============

# ================== VERBATIM: simulation/sparse/oracle.R ==================
oracle = function (t, t1, s0, y, X){

  y1 = y[1:t1]
  y2 = y[(t1+1):t]

  X1 = X[1:t1,]
  X2 = X[(t1+1):t,]

  N = dim(X1)[2]

  X10 = X1[,1:s0]
  b = solve (t(X10) %*% X10) %*% t(X10) %*% y1
  b = append (b, rep(0, N-s0))

  y2.hat = drop (X2 %*% b)
  d = y2 - y2.hat

  return( list(b = b , d = d) )
}
# ================ END VERBATIM: simulation/sparse/oracle.R ================

# ============ VERBATIM (nw branch): simulation/sparse/prd.R ===============
# prd() sweeps seven shift patterns and four variance estimators; the case
# compares the "nw" estimator under the null shift, so only that branch is kept.
prd_nw = function (d) {
  t2 = length(d)
  delta = d
  delbar = mean(delta)
  v1 = lrvar(delta, type = "Newey-West", prewhite = F, adjust = T, lag = round(t2^(1/3)))
  delsd = sqrt(v1)
  stu = delbar/delsd
  return(c(delbar, delsd, stu))
}
# ========== END VERBATIM (nw branch): simulation/sparse/prd.R =============

# ============ VERBATIM (DGPs): simulation/sparse/main_function.R ==========
set.seed(5)

G0 = matrix(rnorm(4*200, 1, 1), 4, 200)
# dgp by factor model
dgp_nnd = function (t, n){

  G = G0[,1:n]

  f = matrix(NA, t, 4)
  f[,1] = arima.sim( t, model = list(ar = 0.9), n.start = 1000)
  f[,2] = arima.sim( t, model = list(ar = 0.5, ma = 0.5), n.start = 1000)
  f[,3] = arima.sim( t, model = list(ma = c(0.8, 0.4)), n.start = 1000)
  f[,4] = rnorm(t)

  err = rnorm(t*n, 0, 0.5)
  X = f %*% G + err

  return(X)
}

# inid dgp
dgp_inid = function (t, n){

  X = matrix(NA, t, n)
  l = floor(n/4)
  for (i in 1:l){
    X[,i] = arima.sim( t, model = (list(ar = 0.9)), n.start = 1000 )
  }
  for (i in (l + 1):(2*l)){
    X[,i] = arima.sim( t, model = (list(ma = c(0.8, 0.4) ) ), n.start = 1000)
  }
  for (i in (2*l + 1):(3*l)) {
    X[,i] = arima.sim( t, model = ( list(ar = 0.5, ma = 0.5)), n.start = 1000 )
  }
  for (i in (3*l + 1):n) {
    X[,i] = arima.sim( t, model = ( list(ar = 0.5)), n.start = 1000 )
  }

  sq = seq(1,3*l+1,l)
  tr = cbind(sq, sq+1, sq+2, sq+3)
  tr = as.vector(tr)
  X = cbind(X[,tr], X[,setdiff(1:n, tr)])

  return(X)
}

# iid dgp
dgp_iid = function (t, n){
  X = matrix( rnorm(t*n), t, n)
  return(X)
}
# ========== END VERBATIM (DGPs): simulation/sparse/main_function.R ========

# Their smallest grid cell: t_list[1] = 100 with t1 = t/2, n_list[1] = 50,
# s_list[1] = 4. main_function.R runs lasso_ic with ic = "WIC" and cst = 2.
TT <- 100; T1 <- 50; N <- 50; S0 <- 4; CST <- 2
REPS <- 10

num <- function(x) sprintf("%.17g", x)
lines <- c()
emit <- function(k, v) lines <<- c(lines, paste(k, num(v), sep = "\t"))
emit_vec <- function(prefix, v) for (i in seq_along(v)) emit(sprintf("%s_k%d", prefix, i), v[i])

emit("T", TT); emit("T1", T1); emit("n_donors", N); emit("s0", S0)
emit("cst", CST); emit("n_reps", REPS)

panel.lines <- c("cell\tt\ty\tdonors")

for (dgp in c("iid", "inid", "nnd")) {
  for (rep in seq_len(REPS)) {
    # main_function.R draws in one stream from its set.seed(5); reproduce that by
    # drawing the DGPs in a fixed order rather than re-seeding per replication.
    Y <- switch(dgp,
                iid  = dgp_iid(t = TT, n = N),
                inid = dgp_inid(t = TT, n = N),
                nnd  = dgp_nnd(t = TT, n = N))
    b0 <- rep(0, N); b0[1:S0] <- 1
    e <- rnorm(TT, 0, 0.5)
    y <- drop(Y %*% b0 + e)
    tag <- sprintf("%s_r%d", dgp, rep)

    for (t in seq_len(TT)) {
      panel.lines <- c(panel.lines, paste(tag, t, num(y[t]),
                                          paste(num(Y[t, ]), collapse = " "), sep = "\t"))
    }

    # The rules mlsynth actually implements, run on the same panels. Comparing
    # against these separates a port regression from the standing difference
    # between the sparse and nonsparse rules; comparing only against the sparse
    # ones confounds the two, since a drift in mlsynth and a rule difference
    # both show up as the same lowered rate.
    rfs_ns  <- FS(y = y, Y = Y, T1 = T1, H = 1, h = 0, alpha = 0.05)
    rlas_ns <- lasso.BIC.t(y = y, Y = Y, T1 = T1, thresh = 1e-14)
    emit(sprintf("fsns_R_%s", tag), length(which(rfs_ns$beta != 0)))
    emit_vec(sprintf("fsns_sel_%s", tag), which(rfs_ns$beta != 0))
    emit(sprintf("lasns_R_%s", tag), length(rlas_ns$sel))
    emit_vec(sprintf("lasns_sel_%s", tag), rlas_ns$sel)

    rfs  <- fs(t = TT, t1 = T1, y = y, Y = Y)
    rlas <- lasso_ic(t = TT, t1 = T1, y = y, X = Y, ic = "WIC", cst = CST)
    rorc <- oracle(t = TT, t1 = T1, s0 = S0, y = y, X = Y)

    for (nm in c("fs", "las", "orc")) {
      r <- switch(nm, fs = rfs, las = rlas, orc = rorc)
      sel <- which(r$b != 0)
      emit(sprintf("%s_R_%s", nm, tag), length(sel))
      emit_vec(sprintf("%s_sel_%s", nm, tag), sel)
      emit_vec(sprintf("%s_coef_%s", nm, tag), r$b[sel])
      emit(sprintf("%s_bias_%s", nm, tag), mean(r$d))
      emit(sprintf("%s_rmse_%s", nm, tag), sqrt(mean(r$d^2)))
      p <- prd_nw(r$d)
      emit(sprintf("%s_delbar_%s", nm, tag), p[1])
      emit(sprintf("%s_delsd_%s", nm, tag), p[2])
      emit(sprintf("%s_stu_%s", nm, tag), p[3])
    }
  }
}

con <- gzfile(file.path(OUT, "panels.tsv.gz"), "wt")
writeLines(panel.lines, con); close(con)
cat(capture.output(sessionInfo()), sep = "\n")
cat("\n")
writeLines(c("== REFERENCE VALUES ==", lines))
