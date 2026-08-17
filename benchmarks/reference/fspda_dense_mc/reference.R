# Reference generator for benchmarks/cases/fspda_dense_mc.py
#
# Shi & Huang (2023), the dense ("nonsparse") Monte Carlo behind their Table 2.
# Everything between the VERBATIM markers is copied unchanged from
# zhentaoshi/fsPDA at 5f4542c -- simulation/nonsparse/{FS.R, lasso.BIC.R} and the
# DGP block of FS.simulation.dense.R. The factor loadings come from their
# committed loading.RData, so the design matrix is theirs and not a re-derivation.
#
# What this emits, per replication:
#   the panel itself (treated series + 100 donor series, T1 + T2 rows),
#   their FS() output   -- R.hat, selected donors, coefficients, ATE, Z, phi,
#   their lasso.BIC()   -- the same, plus the selected penalty,
#   a second lasso.BIC() run with glmnet converged (thresh = 1e-14).
#
# The last one earns its place. lasso.BIC calls glmnet at its default
# thresh = 1e-7, and this DGP is p = 100 against n = 50, where that default stops
# short: the returned coefficients sit about 1.2e-02 from the converged solution
# and 3.5e-05 above it in objective. The selected support and penalty are
# unaffected on most panels but not all, so the case pins mlsynth against the
# verbatim call where the two agree and against the converged call for the
# coefficients. Recording both is what lets a reader tell a port error from
# glmnet's tolerance.
#
# Their third column, scm.R, is covered too, with one substitution recorded
# below: it calls CVXR's ECOS_BB, and CVXR is not installable in this
# environment -- it now requires scs, osqp, clarabel (Rust) and S7, none of
# which are in apt and none of which build from the CRAN GitHub mirror here.
# The program is reproduced exactly and solved with quadprog instead. The
# objective is what gets compared, and it is stable to 1e-9 across ridges from
# 1e-6 to 1e-10, so the substitution does not decide the answer.

suppressMessages(library(glmnet))
suppressMessages(library(MASS))
suppressMessages(library(quadprog))

# Run either directly (Rscript reference.R [SRC] [OUT]) or through
# benchmarks/reference/generate.py, which invokes it with no arguments from the
# bundle directory and captures stdout.
ARGS <- commandArgs(trailingOnly = TRUE)
SRC <- if (length(ARGS) >= 1) ARGS[1] else Sys.getenv("FSPDA_SRC", "/workspace/zhentaoshi/fspda")
OUT <- if (length(ARGS) >= 2) ARGS[2] else "benchmarks/reference/fspda_dense_mc"

NONSPARSE <- file.path(SRC, "simulation", "nonsparse")
load(file.path(NONSPARSE, "loading.RData"))     # -> Lambda, 101 x 4

# ======================= VERBATIM: simulation/nonsparse/FS.R =======================
FS=function(y,Y,T1,H=1,h=0,R=NULL,alpha=0.05) {

  Tn=length(y)
  N=ncol(Y)

  y1=y[1:T1];y2=y[(T1+1):Tn]
  Y1=Y[1:T1,];Y2=Y[(T1+1):Tn,]

  Y.select=function(j,Y1.U) {
    X=cbind(Y1.U,Y1[,j])
    XX=t(X)%*%X
    XX.inv=try(solve(XX),silent=T)
    if ("try-error" %in% class(XX.inv)) {XX.inv=MASS::ginv(XX)}
    b.hat=XX.inv%*%t(X)%*%y1
    e.hat=y1-as.vector(X%*%b.hat)
    sigma2.hat=mean(e.hat^2)
    return(sigma2.hat)
  }

  U=NULL;Uc=1:N
  if(is.null(R)) {
    B=H*log(log(N))*log(T1)/T1
    IC=log(mean(y1^2))
    for (r in 1:T1) {
      sigma2.grid=sapply(Uc,FUN=Y.select,Y1.U=Y1[,U])
      IC.r=log(min(sigma2.grid))+B*r
      if(IC.r<=IC) {
        j.hat=Uc[which.min(sigma2.grid)]
        U=c(U,j.hat);Uc=setdiff(Uc,j.hat)
        IC=IC.r
      } else {
        break
      }
    }
  } else {
    for (r in 1:R) {
      sigma2.grid=sapply(Uc,FUN=Y.select,Y1.U=Y1[,U])
      j.hat=Uc[which.min(sigma2.grid)]
      U=c(U,j.hat);Uc=setdiff(Uc,j.hat)
    }
  }

  X=Y1[,U,drop=F]
  b.hat=as.vector(solve(t(X)%*%X)%*%t(X)%*%y1)

  R.hat=length(U)
  beta.hat=rep(0,N);beta.hat[U]=b.hat
  RSquared=var(as.vector(X%*%b.hat))/var(y1)

  y2.0.hat=as.vector(Y2[,U,drop=F]%*%b.hat)
  d.hat=y2-y2.0.hat
  ATE=mean(d.hat)

  if(h>0) {
    gamma.d=as.vector(acf(d.hat,lag.max=h,type="covariance",plot=F)$acf)
    wh=1-(1:h)/(h+1)
    lrvar.d=gamma.d[1]+2*sum(wh*gamma.d[-1])
  } else {
    lrvar.d=var(d.hat)
  }

  Z=ATE/sqrt(lrvar.d/(Tn-T1))
  phi=as.numeric(abs(Z)>qnorm(1-alpha/2))

  return(list(R=R.hat,beta=beta.hat,RSquared=RSquared,y2.0=y2.0.hat,d=d.hat,ATE=ATE,Z=Z,phi=phi))
}
# ===================== END VERBATIM: simulation/nonsparse/FS.R =====================

# =================== VERBATIM: simulation/nonsparse/lasso.BIC.R ====================
# (one addition, marked inline: a `thresh` argument forwarded to glmnet, so the
#  same function can be run at glmnet's default and at convergence.)
lasso.BIC=function(y,Y,T1,lambda.grid=seq(0.01,1,by=0.01),H=2,h=0,alpha=0.05,
                   thresh=1e-07) {                       # <- added: thresh

  suppressMessages(library(glmnet))

  Tn=length(y)
  N=ncol(Y)
  B=H*log(log(N))*log(T1)/T1

  y1=y[1:T1];y2=y[(T1+1):Tn]
  Y1=Y[1:T1,];Y2=Y[(T1+1):Tn,]

  fit.lasso=glmnet(x=Y1,y=y1,family="gaussian",lambda=lambda.grid,intercept=F,
                   thresh=thresh,maxit=1e7)              # <- added: thresh, maxit
  e.hat=y1-Y1%*%fit.lasso$beta
  IC=log(colMeans(e.hat^2))+B*fit.lasso$df

  min.id=which.min(IC)
  lambda.hat=fit.lasso$lambda[min.id]

  R.hat=fit.lasso$df[min.id]
  beta.hat=fit.lasso$beta[,min.id];names(beta.hat)=NULL
  RSquared=var(as.vector(Y1%*%beta.hat))/var(y1)

  y2.0.hat=as.vector(Y2%*%beta.hat)
  d.hat=y2-y2.0.hat
  ATE=mean(d.hat)

  if(h>0) {
    gamma.d=as.vector(acf(d.hat,lag.max=h,type="covariance",plot=F)$acf)
    wh=1-(1:h)/(h+1)
    lrvar.d=gamma.d[1]+2*sum(wh*gamma.d[-1])
  } else {
    lrvar.d=var(d.hat)
  }

  Z=ATE/sqrt(lrvar.d/(Tn-T1))
  phi=as.numeric(abs(Z)>qnorm(1-alpha/2))

  return(list(R=R.hat,lambda=lambda.hat,beta=beta.hat,RSquared=RSquared,y2.0=y2.0.hat,d=d.hat,ATE=ATE,Z=Z,phi=phi))
}
# ================= END VERBATIM: simulation/nonsparse/lasso.BIC.R ==================

# ========= VERBATIM (DGP block): simulation/nonsparse/FS.simulation.dense.R ========
# The panel-generating half of FS.simulation.dense, lifted out so the panels can
# be written to disk. Defaults are that function's own signature: sigma.eta=0.5,
# sigma.u=1, burn=1000, and the ARMA list it declares.
make.panel = function(T1, T2, Lambda, ARMA, sigma.eta=0.5, sigma.u=1, burn=1000, seed=NULL) {
  Tn=T1+T2
  N=nrow(Lambda)-1;K=ncol(Lambda)
  set.seed(seed)
  eta=matrix(rnorm(Tn*(N+1),sd=sigma.eta),nrow=Tn,ncol=N+1)
  if(is.null(ARMA)) {
    f=sapply(1:K, function(k) rnorm(Tn,sd=k*sigma.u))
  } else {
    f=cbind(rnorm(Tn,sd=sigma.u),
            sapply(ARMA, function(m) as.vector(arima.sim(model=m,n=Tn,n.start=burn,sd=sigma.u))))
  }
  Y=f%*%t(Lambda)+eta
  list(y=Y[,1], Y=Y[,-1])
}
# ======= END VERBATIM (DGP block): simulation/nonsparse/FS.simulation.dense.R ======

# ================= VERBATIM (program): simulation/nonsparse/scm.R =================
# scm.R minimises sum_squares(y_0 - y_N %*% beta) subject to sum(beta) == 1 and
# beta >= 0, via CVXR with solver "ECOS_BB". Same program, quadprog instead of
# CVXR; see the note at the top of this file. The ridge makes the Gram positive
# definite for solve.QP, which needs it at p = 100 against n = 50; it is small
# enough that the objective is unchanged in its ninth digit.
scm_quadprog = function(y, Y, T1, ridge = 1e-08) {
  y1 = y[1:T1]; Y1 = Y[1:T1, ]; N = ncol(Y1)
  D = t(Y1) %*% Y1 + diag(ridge, N)
  d = t(Y1) %*% y1
  A = cbind(rep(1, N), diag(N))
  s = solve.QP(Dmat = D, dvec = d, Amat = A, bvec = c(1, rep(0, N)), meq = 1)
  b = s$solution
  y2.0.hat = as.vector(Y[(T1+1):nrow(Y), ] %*% b)
  d.hat = y[(T1+1):length(y)] - y2.0.hat
  list(beta = b, obj = sum((y1 - Y1 %*% b)^2), d = d.hat, ATE = mean(d.hat))
}
# =============== END VERBATIM (program): simulation/nonsparse/scm.R ==============

ARMA <- list(list(ar=0.9), list(ma=c(0.8,0.4)), list(ar=0.5,ma=0.5))

# Their smallest grid cell: T.grid row 1 is T1 = T2 = 50, where h.iid[,1] and
# h.nnd[,1] both give h = 1 for the prediction long-run variance.
T1 <- 50; T2 <- 50; H.FS <- 1; H.LASSO <- 2; LAG <- 1; ALPHA <- 0.05
GRID <- seq(0.01, 1, by = 0.01)
SEEDS <- 1:4                       # 4 replications per DGP; see manifest for why

num <- function(x) sprintf("%.17g", x)
lines <- c()
emit <- function(k, v) lines <<- c(lines, paste(k, num(v), sep = "\t"))
# Per-slot keys, because the reference.out block parses one scalar per line: a
# selected set of size R becomes R keys `<prefix>_k1..kR` alongside `<prefix>_R`.
emit_vec <- function(prefix, v) for (i in seq_along(v)) emit(sprintf("%s_k%d", prefix, i), v[i])

emit("T1", T1); emit("T2", T2); emit("n_donors", nrow(Lambda) - 1)
emit("H_fs", H.FS); emit("H_lasso", H.LASSO); emit("lrvar_lag", LAG)
emit("alpha", ALPHA); emit("n_seeds", length(SEEDS))

# The panels go to their own file: they are data, not pinned values, and at
# 100 x 101 doubles per replication they would swamp the values block.
panel.lines <- c("cell\tt\ty\tdonors")

for (dgp in c("iid", "nnd")) {
  arma <- if (dgp == "iid") NULL else ARMA
  for (sd in SEEDS) {
    p <- make.panel(T1, T2, Lambda, arma, seed = sd)
    y <- p$y; Y <- p$Y
    tag <- sprintf("%s_s%d", dgp, sd)

    for (t in seq_len(T1 + T2)) {
      panel.lines <- c(panel.lines, paste(tag, t, num(y[t]),
                                          paste(num(Y[t, ]), collapse = " "),
                                          sep = "\t"))
    }

    rf <- FS(y = y, Y = Y, T1 = T1, H = H.FS, h = LAG, alpha = ALPHA)
    emit(sprintf("fs_R_%s", tag), rf$R)
    emit(sprintf("fs_ATE_%s", tag), rf$ATE)
    emit(sprintf("fs_Z_%s", tag), rf$Z)
    emit(sprintf("fs_phi_%s", tag), rf$phi)
    emit(sprintf("fs_RSquared_%s", tag), rf$RSquared)
    sel <- which(rf$beta != 0)
    emit_vec(sprintf("fs_sel_%s", tag), sel)
    emit_vec(sprintf("fs_coef_%s", tag), rf$beta[sel])

    for (mode in c("default", "converged")) {
      th <- if (mode == "default") 1e-07 else 1e-14
      rl <- lasso.BIC(y = y, Y = Y, T1 = T1, lambda.grid = GRID, H = H.LASSO,
                      h = LAG, alpha = ALPHA, thresh = th)
      pre <- sprintf("las%s", if (mode == "default") "" else "cv")
      emit(sprintf("%s_R_%s", pre, tag), rl$R)
      emit(sprintf("%s_lambda_%s", pre, tag), rl$lambda)
      emit(sprintf("%s_ATE_%s", pre, tag), rl$ATE)
      emit(sprintf("%s_Z_%s", pre, tag), rl$Z)
      emit(sprintf("%s_phi_%s", pre, tag), rl$phi)
      selL <- which(rl$beta != 0)
      emit_vec(sprintf("%s_sel_%s", pre, tag), selL)
      emit_vec(sprintf("%s_coef_%s", pre, tag), rl$beta[selL])
    }

    rs <- scm_quadprog(y, Y, T1)
    emit(sprintf("scm_obj_%s", tag), rs$obj)
    emit(sprintf("scm_ATE_%s", tag), rs$ATE)
    emit(sprintf("scm_rmse_%s", tag), sqrt(mean(rs$d^2)))
    selS <- which(rs$beta > 1e-06)
    emit(sprintf("scm_R_%s", tag), length(selS))
    emit_vec(sprintf("scm_sel_%s", tag), selS)
    emit_vec(sprintf("scm_coef_%s", tag), rs$beta[selS])
  }
}

# panels.tsv is written beside the bundle; the values block goes to stdout,
# which generate.py captures verbatim into reference.out and parses.
writeLines(panel.lines, file.path(OUT, "panels.tsv"))
cat(capture.output(sessionInfo()), sep = "\n")
cat("\n")
writeLines(c("== REFERENCE VALUES ==", lines))
