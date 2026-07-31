#!/usr/bin/env Rscript
# Reference capture for the de Brabander, Juodis & Miyazato Szini (2025)
# Monte Carlo (Econometric Reviews, Section 5; Tables 9-12).
#
# Ports the authors' "Simulations/simulations_no covariates.R" DGP onto the
# committed Brexit panel, and dumps -- per replication -- the simulated outcome
# matrix together with each estimator's weights and estimation error. The Python
# case reads those outcome matrices, re-runs mlsynth on them, and compares the
# errors value-for-value. Cross-validating the simulator itself is what makes
# this a real check: matching Monte Carlo aggregates alone would confound a
# wrong DGP with a wrong estimator.
#
# Two deliberate departures from the paper's prose, both following the authors'
# code because the code is what produced the printed tables:
#
#   * the common factor is gamma * (1 - 1/t), not the paper's t*gamma/T0;
#   * the idiosyncratic errors are drawn with rnorm(sd = sigma^2), so
#     sigma = 0.25 means a standard deviation of 0.0625. That reading is
#     checkable: the published sigma = 0.25 RMSEs (~0.07) are 0.0625 times the
#     sigma = 1 ones (~1.11), which the sd = sigma reading cannot produce.
#
# The treated unit's factor loading is the second largest of the draws (the
# authors' mu_type = "second"), which keeps it inside the donors' convex hull.
#
# Usage:  Rscript benchmarks/reference/brabander_mc/reference.R
# Needs:  synthdid.
#
# Only SC, DSC and SDID are captured. MASC is not: the authors' masc package
# needs Gurobi. ASCM is not either: augsynth 0.2.0 warns that "tuning ridge
# regression when number of units (24) is almost equal to the number of time
# periods (24) is often unstable" and then errors out on this design, so there
# is no reference to capture. Both are checked against the published Monte
# Carlo aggregates instead, and the case says so. mlsynth's ridge ASCM is
# separately cross-validated against a live augsynth run in ascm_kansas, on a
# panel where augsynth does run.

suppressMessages(library(synthdid))

here <- file.path("benchmarks", "reference", "brabander_mc")
panel <- read.csv(file.path("benchmarks", "reference", "brabander_sdid_match",
                            "brexit_panel.csv"), stringsAsFactors = FALSE)

COVS <- c("real_con", "real_inv", "real_exp", "real_imp", "lab_prod",
          "tot_emp_pc")
UK <- "United Kingdom"
T0 <- 24; N0 <- 23; PERIODS <- T0 + 1; UNITS <- N0 + 1

## ---- calibration ----------------------------------------------------------
## The authors regress the outcome on unit-covariate x period-identity
## interactions plus period dummies. That design is block diagonal by period,
## so it is PERIODS independent regressions of the units' outcomes on their
## covariate means with an intercept.
countries <- unique(panel$country)
order <- c(UK, setdiff(countries, UK))

wide <- reshape(panel[, c("country", "t", "y")], idvar = "t",
                timevar = "country", direction = "wide")
wide <- wide[order(wide$t), ]
Ycols <- paste0("y.", order)
Y <- as.matrix(wide[wide$t >= 202 & wide$t <= 226, Ycols])   # (25, 24)

covwin <- panel[panel$t >= 202 & panel$t <= 225, ]
Cmat <- sapply(order, function(cn)
  colMeans(covwin[covwin$country == cn, COVS]))              # (6, 24)

X <- cbind(1, t(Cmat))                                       # (24, 7)
coef <- qr.solve(t(X) %*% X, t(X) %*% t(Y))                  # (7, 25)
delta <- coef[1, ]
beta <- coef[-1, , drop = FALSE]                             # (6, 25)
systematic <- matrix(delta, PERIODS, UNITS) + t(beta) %*% Cmat

write.csv(data.frame(period = seq_len(PERIODS), delta = delta),
          file.path(here, "calibration_delta.csv"), row.names = FALSE)
write.csv(as.data.frame(t(beta)), file.path(here, "calibration_beta.csv"),
          row.names = FALSE)
write.csv(as.data.frame(Cmat), file.path(here, "calibration_covariates.csv"),
          row.names = TRUE)

## ---- the DGP --------------------------------------------------------------
simulate_one <- function(S, gamma, sigma) {
  set.seed(S)
  tt <- seq_len(PERIODS)
  gamma_t <- gamma * (1 - 1 / tt)
  mu_initial <- runif(UNITS - 1, min = 0, max = 1)
  n <- length(mu_initial)
  mu <- c(sort(mu_initial, partial = n - 1)[n - 1], mu_initial)
  eps <- matrix(rnorm(PERIODS * UNITS, 0, sigma^2), nrow = PERIODS)
  outcome <- systematic + outer(gamma_t, mu) + eps
  list(outcome = outcome, mu = mu, gamma_t = gamma_t, eps = eps)
}

## ---- estimators -----------------------------------------------------------
estimate <- function(outcome) {
  Y1 <- t(outcome[, 1, drop = FALSE])
  Y0 <- t(outcome[, -1, drop = FALSE])
  Ysdid <- rbind(Y0, Y1)
  Z1 <- as.matrix(outcome[1:T0, 1])
  Z0 <- outcome[1:T0, -1]

  sc <- synthdid_estimate(Ysdid, N0, T0, zeta.omega = 0, zeta.lambda = 0,
                          omega.intercept = FALSE, lambda.intercept = FALSE)
  w_sc <- attr(sc, "weights")$omega
  tau_sc <- outcome[PERIODS, 1] - outcome[PERIODS, -1] %*% w_sc

  dsc <- synthdid_estimate(Ysdid, N0, T0, zeta.omega = 0, zeta.lambda = 0,
                           omega.intercept = TRUE, lambda.intercept = TRUE)
  w_dsc <- attr(dsc, "weights")$omega
  lam <- attr(dsc, "weights")$lambda
  tau_dsc <- outcome[PERIODS, 1] - outcome[PERIODS, -1] %*% w_dsc -
    mean(Z1 - Z0 %*% w_dsc)
  tau_sdid <- outcome[PERIODS, 1] - outcome[PERIODS, -1] %*% w_dsc -
    lam %*% (Z1 - Z0 %*% w_dsc)

  list(tau = c(SC = as.numeric(tau_sc), DSC = as.numeric(tau_dsc),
               SDID = as.numeric(tau_sdid)),
       omega = cbind(SC = as.numeric(w_sc), DSC = as.numeric(w_dsc)),
       lambda = as.numeric(lam))
}

## ---- capture --------------------------------------------------------------
SETTINGS <- list(c(gamma = 1.0, sigma = 1.0), c(gamma = 0.0, sigma = 1.0),
                 c(gamma = 1.5, sigma = 1.0), c(gamma = 1.0, sigma = 0.25))
N_DRAWS <- 12

rows <- list(); k <- 0
for (st in SETTINGS) {
  g <- unname(st["gamma"]); s <- unname(st["sigma"])
  tag <- sprintf("g%s_s%s", g, s)
  for (S in seq_len(N_DRAWS)) {
    sim <- simulate_one(S, g, s)
    est <- tryCatch(estimate(sim$outcome), error = function(e) NULL)
    if (is.null(est)) next
    write.csv(as.data.frame(sim$outcome),
              file.path(here, sprintf("outcome_%s_S%02d.csv", tag, S)),
              row.names = FALSE)
    k <- k + 1
    rows[[k]] <- data.frame(setting = tag, gamma = g, sigma = s, seed = S,
                            method = names(est$tau), tau = as.numeric(est$tau))
  }
  cat("done", tag, "\n")
}
write.csv(do.call(rbind, rows), file.path(here, "tau.csv"), row.names = FALSE)

writeLines(c(paste("R:", paste(R.version$major, R.version$minor, sep = ".")),
             paste("synthdid:", as.character(packageVersion("synthdid"))),
             paste("draws_per_setting:", N_DRAWS),
             "note: masc not captured (its solver needs Gurobi)",
             "note: augsynth not captured (errors on 24 units x 24 pre-periods)"),
           file.path(here, "versions.txt"))
cat("wrote", k, "replications\n")
