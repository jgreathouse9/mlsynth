#!/usr/bin/env Rscript
# Goh & Yu (2022) Section 6, Tables 3 and 4 -- the authors' own script with two
# substitutions and nothing else changed:
#
#   invgamma::rinvgamma(1, shape=a, rate=b)  ->  1/rgamma(1, shape=a, rate=b)
#   HDInterval::hdi(x, credMass)             ->  shortest interval containing
#                                                credMass of the sorted draws
#
# Both packages are unavailable here; both are one-liners with exact
# equivalents, so the RNG stream and the arithmetic are unchanged.
suppressWarnings(suppressMessages({library(Synth); library(Rsolnp); library(truncnorm)}))

rinvgamma <- function(n, shape, rate) 1 / rgamma(n, shape = shape, rate = rate)
hdi <- function(x, credMass = 0.95) {
  # HDInterval::hdi -- the narrowest window of the sorted draws holding
  # ceiling(credMass * n) of them.
  x <- sort(as.numeric(x)); n <- length(x)
  m <- max(2, ceiling(credMass * n))
  i <- 1:(n - m + 1)
  k <- which.min(x[i + m - 1] - x[i])
  c(lower = x[k], upper = x[k + m - 1])
}

data(basque)
dataprep.out <- dataprep(
  foo = basque,
  predictors = c("school.illit","school.prim","school.med",
                 "school.high","school.post.high","invest"),
  predictors.op = "mean", time.predictors.prior = 1964:1969,
  special.predictors = list(
    list("gdpcap", 1960:1969, "mean"),
    list("sec.agriculture", seq(1961,1969,2), "mean"),
    list("sec.energy", seq(1961,1969,2), "mean"),
    list("sec.industry", seq(1961,1969,2), "mean"),
    list("sec.construction", seq(1961,1969,2), "mean"),
    list("sec.services.venta", seq(1961,1969,2), "mean"),
    list("sec.services.nonventa", seq(1961,1969,2), "mean"),
    list("popdens", 1969, "mean")),
  dependent = "gdpcap", unit.variable = "regionno",
  unit.names.variable = "regionname", time.variable = "year",
  treatment.identifier = 17, controls.identifier = c(2:16, 18),
  time.optimize.ssr = 1960:1969, time.plot = 1955:1997)

dataprep.out$X1["school.high",] <- dataprep.out$X1["school.high",] +
  dataprep.out$X1["school.post.high",]
dataprep.out$X1 <- as.matrix(dataprep.out$X1[
  -which(rownames(dataprep.out$X1) == "school.post.high"),])
dataprep.out$X0["school.high",] <- dataprep.out$X0["school.high",] +
  dataprep.out$X0["school.post.high",]
dataprep.out$X0 <- dataprep.out$X0[
  -which(rownames(dataprep.out$X0) == "school.post.high"),]
lowest <- which(rownames(dataprep.out$X0) == "school.illit")
highest <- which(rownames(dataprep.out$X0) == "school.high")
dataprep.out$X1[lowest:highest,] <- (100 * dataprep.out$X1[lowest:highest,]) /
  sum(dataprep.out$X1[lowest:highest,])
dataprep.out$X0[lowest:highest,] <- 100 * scale(
  dataprep.out$X0[lowest:highest,], center = FALSE,
  scale = colSums(dataprep.out$X0[lowest:highest,]))

synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS", verbose = FALSE)
synth.tables <- synth.tab(dataprep.res = dataprep.out, synth.res = synth.out)
hat.w.scm1 <- synth.tables$tab.w[,1]
obs.Y.1t <- dataprep.out$Y1
hat.Y.1t.scm1 <- (dataprep.out$Y0) %*% hat.w.scm1

set.seed(12345)
Year <- 1955:1997; p <- 12; N <- 17; T0 <- length(1960:1969)
X1 <- c((dataprep.out$Y1)[which(Year==1960):which(Year==1969)],
        as.numeric(dataprep.out$X1)[-6])
X0 <- rbind(cbind(rep(1,T0), (dataprep.out$Y0)[which(Year==1960):which(Year==1969),]),
            cbind(rep(0,p), (dataprep.out$X0)[-6,]))

c0 <- .5; d0 <- .5; inv.nu <- 1; prob0 <- 0.5
hat.xi <- rep(1,p); xi.all <- c(rep(1,T0), hat.xi)
LS2 <- function(w) inv.nu * sum(xi.all * as.numeric((X1 - as.numeric(X0 %*% w))^2))
sum.constrant2 <- function(w) sum(w[-1])
hat.w.MAP <- rep(0,N); rep.r <- 1000
for (r in 1:rep.r) {
  hat.w.MAP0 <- hat.w.MAP
  invisible(capture.output(hat.w.MAP <- solnp(rep(1/N,N), fun=LS2,
    eqfun=sum.constrant2, eqB=1, LB=c(-Inf, rep(0,N-1)))$pars))
  if (r == 1) hat.w.MAP.naive <- hat.w.MAP
  if (max(abs(hat.w.MAP0 - hat.w.MAP)) < (0.1)^4) break
  Gibbs.rep <- 15000
  MC.nu <- rep(0,Gibbs.rep); MC.xi <- matrix(0,Gibbs.rep,T0+p)
  for (i in 1:Gibbs.rep) {
    nu.a <- (T0)/2 + c0
    nu.b <- 0.5*sum(xi.all*(as.numeric(X1 - X0 %*% hat.w.MAP)^2)) + d0
    hat.nu <- rinvgamma(1, shape=nu.a, rate=nu.b)
    phi.Z <- dnorm(X1[-(1:T0)], as.numeric(X0[-(1:T0),] %*% hat.w.MAP), sqrt(hat.nu))
    xi.prob <- prob0/((1-prob0)/phi.Z + prob0)
    hat.xi <- rbinom(p,1,xi.prob); xi.all <- c(rep(1,T0), hat.xi)
    MC.xi[i,] <- xi.all; MC.nu[i] <- hat.nu
  }
  xi.all <- apply(MC.xi[-(1:5000),],2,mean); inv.nu <- mean(1/MC.nu[-(1:5000)])
  if (r == rep.r) cat("Algorithm does NOT reach the convergence\n")
}
cat(sprintf("EM iterations: %d\n", r))
hat.Y.1t.MAP <- (dataprep.out$Y0) %*% hat.w.MAP[-1] + hat.w.MAP[1]*rep(1,length(Year))

MC.size <- 20000; burnin.size <- 5000; T <- length(Year)
MCMC.omega <- hat.w.MAP; MCMC.nu <- hat.nu; MCMC.xi <- hat.xi
xi.all <- c(rep(1,T0), hat.xi)
MCMC.hat.Y.Bayes <- matrix(0,MC.size,T); MCMC.hat.Y.Bayes.v2 <- matrix(0,MC.size,T)
active.donor <- which(round(hat.w.MAP,5) != 0)
MCMC.omega[-active.donor] <- 0
MCMC.omega[(active.donor[-1])] <- hat.w.MAP[(active.donor[-1])]/sum(hat.w.MAP[(active.donor[-1])])
M <- max(active.donor); D1 <- dim(X0)[1]; D2 <- dim(X0)[2]
for (MCMC.goh in 1:MC.size) {
  sig2.i <- 1/sum((inv.nu*xi.all)*(X0[,1]^2))
  mu.i <- sum(X0[,1]*((inv.nu*xi.all)*(X1 - as.numeric(X0[,-1] %*% MCMC.omega[-1]))))*sig2.i
  MCMC.omega[1] <- rnorm(1, mean=mu.i, sd=sqrt(sig2.i))
  for (i in active.donor[-1]) {
    if (i == M) { MCMC.omega[i] <- 1 - sum(MCMC.omega[-c(1,i)]) } else {
      X.i.s <- X0[,i] - X0[,M]
      sig2.i <- 1/sum((inv.nu*xi.all)*(X.i.s^2))
      mu.i <- sum(X.i.s*((inv.nu*xi.all)*(X1 - X0[,M] - MCMC.omega[1]*X0[,1] -
        as.numeric((X0[,-c(1,i,M)] - matrix(X0[,M],D1,(D2-3))) %*% MCMC.omega[-c(1,i,M)]))))*sig2.i
      UB.i <- 1 - sum(MCMC.omega[-c(1,i,M)])
      MCMC.omega[i] <- rtruncnorm(1, a=0, b=UB.i, mean=mu.i, sd=sqrt(sig2.i))
    }
  }
  nu.a <- (T0)/2 + c0
  nu.b <- 0.5*sum(xi.all*(as.numeric(X1 - X0 %*% MCMC.omega)^2)) + d0
  MCMC.nu <- rinvgamma(1, shape=nu.a, rate=nu.b); inv.nu <- 1/MCMC.nu
  phi.Z <- dnorm(X1[-(1:T0)], as.numeric(X0[-(1:T0),] %*% MCMC.omega), sqrt(MCMC.nu))
  xi.prob <- prob0/((1-prob0)/phi.Z + prob0)
  MCMC.xi <- rbinom(p,1,xi.prob); xi.all <- c(rep(1,T0), MCMC.xi)
  if (is.na(sum(MCMC.xi))) break
  MCMC.hat.Y.Bayes[MCMC.goh,] <- (dataprep.out$Y0) %*% MCMC.omega[-1] + rep(MCMC.omega[1],T)
  MCMC.hat.Y.Bayes.v2[MCMC.goh,] <- rnorm(T, MCMC.hat.Y.Bayes[MCMC.goh,], sqrt(MCMC.nu))
}
MCMC.hat.Y.Bayes.v2[,which(Year==1960):which(Year==1969)] <-
  MCMC.hat.Y.Bayes[,which(Year==1960):which(Year==1969)]

Trt.period <- which(Year > 1980 & Year < 1995)
MCMC.theta <- matrix(obs.Y.1t[Trt.period], dim(MCMC.hat.Y.Bayes)[1],
                     length(Trt.period), byrow=TRUE) - MCMC.hat.Y.Bayes[,Trt.period]
MCMC.ATE <- apply(MCMC.theta[-(1:burnin.size),],1,mean)
Bayes.ATE.HPD95 <- hdi(MCMC.ATE, credMass=0.95)
ADH_SCM.ATE <- mean((obs.Y.1t - hat.Y.1t.scm1)[Trt.period])
Bayes_SCM.ATE <- mean((obs.Y.1t - hat.Y.1t.MAP)[Trt.period])

emit <- function(k,v) cat(sprintf("%s=%.10g\n", k, v))
emit("bayes_ate", Bayes_SCM.ATE); emit("adh_ate", ADH_SCM.ATE)
emit("bayes_hpd_lo", Bayes.ATE.HPD95[1]); emit("bayes_hpd_hi", Bayes.ATE.HPD95[2])
nm <- gsub("[^A-Za-z]", "", c("Intercept", synth.tables$tab.w[,2]))
for (i in 1:N) { emit(paste0("w_bayes_", nm[i]), hat.w.MAP[i])
                 emit(paste0("w_adh_", nm[i]), c(0, synth.tables$tab.w[,1])[i]) }
emit("pct_loss_bayes", 100*Bayes_SCM.ATE/mean(obs.Y.1t[Trt.period]))
emit("pct_loss_adh", 100*ADH_SCM.ATE/mean(obs.Y.1t[Trt.period]))
emit("n_active_donors", sum(round(hat.w.MAP[-1],5) != 0))
