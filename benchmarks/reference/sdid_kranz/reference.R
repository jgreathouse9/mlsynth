#!/usr/bin/env Rscript
# Gold for the Kranz (2022) two-step SDID adjustment, from skranz/xsynthdid.
#
#   Rscript benchmarks/reference/sdid_kranz/reference.R
#
# The fixture is the package README's own worked example: xsdid.mc(N=30, T=30,
# tau=50) under set.seed(1), where the true effect is 50, plain SDID gives
# 39.760 and the adjusted estimate gives 49.398. Using the author's published
# example means the endpoint is checkable against a number he printed, not only
# against a number this script produced.
#
# Two artefacts are captured, and the first matters more:
#
#   panel.csv       the panel plus y.adj, so mlsynth can be checked at the SEAM
#                   -- element-wise on the adjusted outcome -- before the SDID
#                   solve is involved at all. A wrong regression and a wrong
#                   solve both move the endpoint; only this separates them.
#   gold.txt        the SDID estimates and the fitted covariate coefficient.
#
# Exporting the panel also means both sides read one file rather than each
# regenerating xsdid.mc from its own RNG, which R and numpy will not agree on.
#
# Each estimate is captured TWICE, at synthdid's default stopping rule and at a
# tightened one, because on this panel they are not the same number.
# synthdid_estimate solves the weight programs by projected gradient and stops
# when the objective decrease falls below min.decrease = 1e-5 * noise.level. On
# the adjusted outcome the unit-weight ridge (zeta.omega = 218.6) dominates the
# fit, so omega is pushed to within 4e-6 of the uniform simplex point and the
# surface there is nearly flat -- the early stop lands 0.0052 away in the
# estimate while the objective is only 6.4e-5 higher. Running to convergence
# removes it. Both are worth having: the default is the number the README
# prints, and the converged one is the actual argmin of eq. (4), which is what
# an exact solver should be checked against.
suppressMessages({library(xsynthdid); library(synthdid); library(fixest)})
OUT <- file.path("benchmarks", "reference", "sdid_kranz")

set.seed(1)
dat <- xsdid.mc(N = 30, T = 30, tau = 50, return.data = TRUE)

dat$y.adj <- adjust.outcome.for.x(dat, unit = "i", time = "t",
                                  outcome = "y", treatment = "treat_exp", x = "x")

pm  <- panel.matrices(dat, unit="i", time="t", outcome="y",     treatment="treat_exp")
pma <- panel.matrices(dat, unit="i", time="t", outcome="y.adj", treatment="treat_exp")
tau_raw <- synthdid_estimate(pm$Y,  pm$N0,  pm$T0)
tau_adj <- synthdid_estimate(pma$Y, pma$N0, pma$T0)

# the same two estimates, run to convergence rather than to synthdid's default
# early stop (see the header note)
CONV <- list(min.decrease = 1e-12, max.iter = 2e5)
tau_raw_c <- synthdid_estimate(pm$Y,  pm$N0,  pm$T0,  min.decrease=CONV$min.decrease,
                               max.iter=CONV$max.iter)
tau_adj_c <- synthdid_estimate(pma$Y, pma$N0, pma$T0, min.decrease=CONV$min.decrease,
                               max.iter=CONV$max.iter)

# the regression the adjustment rests on, refit here so its coefficient is
# pinned independently of the adjusted outcome it produces
reg <- feols(y ~ x | i + t, data = dat[as.integer(dat$treat_exp) == 0, ])

write.csv(dat[, c("i","t","treat_exp","x","y","y.adj")],
          file.path(OUT, "panel.csv"), row.names = FALSE)
writeLines(c(
  sprintf("tau_unadjusted: %.10f", as.numeric(tau_raw)),
  sprintf("tau_adjusted:   %.10f", as.numeric(tau_adj)),
  sprintf("tau_unadjusted_converged: %.10f", as.numeric(tau_raw_c)),
  sprintf("tau_adjusted_converged:   %.10f", as.numeric(tau_adj_c)),
  sprintf("beta_x:         %.10f", unname(coef(reg)["x"])),
  sprintf("n_rows:         %d", nrow(dat)),
  sprintf("n_units:        %d", length(unique(dat$i))),
  sprintf("n_periods:      %d", length(unique(dat$t))),
  sprintf("N0:             %d", pm$N0),
  sprintf("T0:             %d", pm$T0),
  sprintf("true_tau:       %d", 50L),
  sprintf("xsynthdid:      %s", as.character(packageVersion("xsynthdid"))),
  sprintf("synthdid:       %s", as.character(packageVersion("synthdid"))),
  sprintf("R:              %s.%s", R.version$major, R.version$minor)),
  file.path(OUT, "gold.txt"))
cat(readLines(file.path(OUT, "gold.txt")), sep = "\n")
