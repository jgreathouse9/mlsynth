## Gold standard for the mlsynth DM-LFM port.
##
## Reproduces the DM-LFM half of Pang, Liu & Xu (2022) replication script
## 2_ex_adh2015.R verbatim, at several seeds, and writes the posterior
## summaries the Python port must match.
##
## Spec, from 2_ex_adh2015.R lines 118-160:
##   - six TIME-INVARIANT covariates, each the unit's mean over ALL 44 years
##     (1960-2003, post-treatment included): pgdp, trade, inflation, industry,
##     schooling, invest. `invest` pools invest60/70/80.
##   - Xname = Aname (constant coefficients AND time-varying coefficients),
##     Zname = NULL (no unit-varying coefficients, to match the SCM)
##   - re = "time", r = 10, ar1 default TRUE
##   - niter = 25000, burn = 5000
##   - xlasso = zlasso = alasso = 0 (flat priors), flasso = 1 (factor selection)
##
## Run from the `replication/` directory of the replication package.

library(haven)
library(pblasso)
source("code/summary_function.R")

SEEDS <- c(1234, 1, 2, 3, 4)
OUT <- "gold"
dir.create(OUT, showWarnings = FALSE)

## ---- data, exactly as the authors build it ----
data <- as.data.frame(read_dta("data/repgermany.dta"))
data <- data[order(data$index, data$year), ]
D <- rep(0, nrow(data))
D[which(data$index == 7 & data$year >= 1990)] <- 1
data$D <- D
data.old <- data
data$pgdp <- data$trade <- data$inflation <-
  data$industry <- data$schooling <- data$invest <- NA
for (i in unique(data$index)) {
  pos <- which(data$index == i)
  s <- data.old[pos, ]
  data[pos, "pgdp"]      <- mean(s[, "gdp"], na.rm = TRUE)
  data[pos, "trade"]     <- mean(s[, "trade"], na.rm = TRUE)
  data[pos, "inflation"] <- mean(s[, "infrate"], na.rm = TRUE)
  data[pos, "industry"]  <- mean(s[, "industry"], na.rm = TRUE)
  data[pos, "schooling"] <- mean(s[, "schooling"], na.rm = TRUE)
  data[pos, "invest"]    <- mean(unlist(c(s[, c("invest60", "invest70", "invest80")])),
                                 na.rm = TRUE)
}
COV <- c("pgdp", "trade", "inflation", "industry", "schooling", "invest")
YEARS <- sort(unique(data$year))
write.csv(unique(data[, c("index", "country", COV)]),
          file.path(OUT, "gold_covariates.csv"), row.names = FALSE)

att_rows <- ct_rows <- sum_rows <- wg_rows <- list()

for (sd in SEEDS) {
  set.seed(sd)
  t0 <- Sys.time()
  fit <- pblasso(data = data, index = c("index", "year"),
                 Yname = "gdp", Dname = "D",
                 Xname = COV, Zname = NULL, Aname = COV,
                 re = "time", r = 10, niter = 25000, burn = 5000,
                 xlasso = 0, zlasso = 0, alasso = 0, flasso = 1)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  e <- effSummary(fit, usr.id = NULL, cumu = FALSE, rela.period = FALSE)
  eff <- as.data.frame(e$est.eff)
  eff$year <- YEARS
  eff$seed <- sd
  att_rows[[length(att_rows) + 1]] <-
    eff[, c("seed", "year", "estimated_ATT",
            "estimated_ATT_ci_l", "estimated_ATT_ci_u")]

  ## observed vs posterior-mean counterfactual
  yct <- rowMeans(fit$yct)                       # posterior mean of Y(0) draws
  ct_rows[[length(ct_rows) + 1]] <- data.frame(
    seed = sd, year = YEARS,
    observed = data$gdp[data$index == 7],
    counterfactual = yct)

  a <- e$est.avg
  pre <- eff$estimated_ATT[eff$year < 1990]
  sum_rows[[length(sum_rows) + 1]] <- data.frame(
    seed = sd,
    att = mean(a), att_median = median(a),
    ci_l = quantile(a, .025), ci_u = quantile(a, .975),
    pre_gap_mean = mean(pre), pre_gap_maxabs = max(abs(pre)),
    sigma2 = mean(fit$sigma2), seconds = secs)

  ## omega_gamma: the factor-loading scales that drive factor selection
  wg <- abs(fit$wg)                              # r x ndraws
  wg_rows[[length(wg_rows) + 1]] <- data.frame(
    seed = sd, factor = seq_len(nrow(wg)),
    mean_abs_omega = rowMeans(wg),
    q10 = apply(wg, 1, quantile, .10),
    q90 = apply(wg, 1, quantile, .90))

  cat(sprintf("seed %5d  ATT %8.1f [%8.1f, %8.1f]  pre|gap|max %6.1f  %.0fs\n",
              sd, mean(a), quantile(a, .025), quantile(a, .975),
              max(abs(pre)), secs))
}

write.csv(do.call(rbind, att_rows), file.path(OUT, "gold_att_by_year.csv"), row.names = FALSE)
write.csv(do.call(rbind, ct_rows), file.path(OUT, "gold_counterfactual.csv"), row.names = FALSE)
write.csv(do.call(rbind, sum_rows), file.path(OUT, "gold_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, wg_rows), file.path(OUT, "gold_omega_gamma.csv"), row.names = FALSE)

s <- do.call(rbind, sum_rows)
cat("\n-- across seeds --\n")
cat(sprintf("ATT      mean %.1f  sd %.1f  range [%.1f, %.1f]\n",
            mean(s$att), sd(s$att), min(s$att), max(s$att)))
cat(sprintf("pre|gap| max across seeds %.1f\n", max(s$pre_gap_maxabs)))
cat("wrote", OUT, "\n")
