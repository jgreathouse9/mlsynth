#!/usr/bin/env Rscript
# Seam gold for SDID covariate matching, from the authors' own R code.
#
#   Rscript benchmarks/reference/brabander_sdid_match/reference.R
#
# de Brabander, E., Juodis, A., & Miyazato Szini, G. (2025), "On the use of
# synthetic difference-in-differences approach with (-out) covariates: The case
# study of Brexit referendum", Econometric Reviews 44(10):1617-1646.
#
# WHAT THIS CAPTURES, AND WHY IT IS THE SEAM
# ------------------------------------------
# Their SDID-with-covariates is assembled from two separate fits (see
# "Treatment effects - 2016Q3 mean covariates - no penalty.R"):
#
#   omega  <- synth(X1 = [demeaned pre-outcomes; covariate means],
#                   X0 = same for donors,
#                   Z1 = demeaned pre-outcomes,      # outer objective
#                   Z0 = same for donors)$solution.w
#   lambda <- synthdid_estimate(Y, N0, T0, zeta.omega = 0, zeta.lambda = 0, ...)
#
# So the covariates enter only through omega, and only through Synth's nested
# V search. omega and the fitted V are therefore the quantities to compare --
# not the ATT, which mixes them with the time weights and the bias correction.
# A disagreement in the ATT alone cannot say which half moved.
#
# Three matching schemes are captured, because the number of demeaned outcome
# rows in X is what decides whether the covariates can bind at all (Kaul,
# Klossner, Pfeifer & Schieler 2022; the paper adopts the result on p. 1629):
#
#   all   rows 1:86 of the pre-period      -- covariates provably irrelevant
#   half  rows 44:86                       -- the paper's "half" specification
#   one   row 86 only                      -- the last pre-treatment period
#
# Note the outer objective Z is the FULL demeaned pre-period in every case; only
# X changes. That asymmetry is easy to miss and is the first thing to check if
# the Python side disagrees.
#
# Data: basedata/brexit_oecd_nov2018.mat, the authors' file unmodified. The
# transformations below (log GDP, covariate ratios to GDP, the labour
# productivity log-difference, the twelve dropped countries) are theirs and are
# reproduced here rather than in Python on purpose -- rebuilding the inputs in
# the port is what produced a sign flip in an earlier replication.
suppressMessages({library(R.matlab); library(Synth)})

OUT  <- file.path("benchmarks", "reference", "brabander_sdid_match")
DATA <- file.path("basedata", "brexit_oecd_nov2018.mat")

d <- readMat(DATA)

country_names <- c('Australia','Austria','Belgium','Canada','Chile',
  'Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece',
  'Hungary','Iceland','Ireland','Israel','Italy','Japan','Korea','Latvia',
  'Lithuania','Luxembourg','Mexico','Netherlands','New Zealand','Norway',
  'Poland','Portugal','Slovak Republic','Slovenia','Spain','Sweden',
  'Switzerland','Turkey','United Kingdom','United States')

y_long <- log(d[["real.gdp"]])
gdp_raw <- d[["real.gdp.raw"]]
con  <- d[["real.con.raw"]] / gdp_raw
inv  <- d[["real.inv.raw"]] / gdp_raw
exp_ <- d[["real.exp.raw"]] / gdp_raw
imp  <- d[["real.imp.raw"]] / gdp_raw
lab  <- rbind(rep(NaN, 36), 100 * diff(log(d[["lab.prod.raw"]])))
emp  <- d[["tot.emp.pc.raw"]]

drop <- c(5,6,7,8,12,16,20,21,23,27,30,34)
country_names <- country_names[-drop]
y_long <- y_long[, -drop]
con <- con[, -drop]; inv <- inv[, -drop]; exp_ <- exp_[, -drop]
imp <- imp[, -drop]; lab <- lab[, -drop]; emp <- emp[, -drop]

# Treatment at 2016Q3, so the pre-period is rows 141:226. The UK is column 23
# of the surviving 24.
PRE <- 141:226
TREATED_COL <- 23
Ypre <- y_long[PRE, ]
Ypre_dm <- apply(Ypre, 2, function(x) x - mean(x))   # each unit by its own mean

covmeans <- rbind(colMeans(con[PRE, ]), colMeans(inv[PRE, ]),
                  colMeans(exp_[PRE, ]), colMeans(imp[PRE, ]),
                  colMeans(lab[PRE, ]), colMeans(emp[PRE, ]))
rownames(covmeans) <- c("real_con", "real_inv", "real_exp", "real_imp",
                        "lab_prod", "tot_emp_pc")

donors <- setdiff(seq_len(ncol(Ypre)), TREATED_COL)
Z1 <- as.matrix(Ypre_dm[, TREATED_COL])
Z0 <- Ypre_dm[, donors]

SCHEMES <- list(all = 1:86, half = 44:86, one = 86)

summ <- NULL
for (nm in names(SCHEMES)) {
  rows <- SCHEMES[[nm]]
  X1 <- as.matrix(c(Ypre_dm[rows, TREATED_COL], covmeans[, TREATED_COL]))
  # drop = FALSE: with the one-period scheme a single row would collapse to a
  # vector and as.matrix would turn it into a column, silently transposing X0.
  X0 <- rbind(Ypre_dm[rows, donors, drop = FALSE], covmeans[, donors])
  fit <- synth(X1 = X1, X0 = X0, Z0 = Z0, Z1 = Z1)

  w <- as.numeric(fit$solution.w)
  v <- as.numeric(fit$solution.v)
  write.csv(data.frame(donor = country_names[donors], weight = w),
            file.path(OUT, paste0("omega_", nm, ".csv")), row.names = FALSE)
  write.csv(data.frame(predictor = seq_along(v), v = v),
            file.path(OUT, paste0("v_", nm, ".csv")), row.names = FALSE)
  write.csv(data.frame(row = rows), file.path(OUT, paste0("rows_", nm, ".csv")),
            row.names = FALSE)

  gap <- as.numeric(Z1 - Z0 %*% w)
  summ <- rbind(summ, data.frame(
    scheme = nm, n_outcome_rows = length(rows), n_predictors = length(v),
    n_donors = length(donors), loss_v = as.numeric(fit$loss.v),
    loss_w = as.numeric(fit$loss.w), outer_ssr = sum(gap^2),
    max_weight = max(w), n_positive = sum(w > 1e-8),
    v_on_covariates = sum(tail(v, 6)), stringsAsFactors = FALSE))
  cat(sprintf("%-5s rows=%2d preds=%3d  loss.v=%.6g  cov share of V=%.4f  nnz(w)=%d\n",
              nm, length(rows), length(v), fit$loss.v, sum(tail(v, 6)), sum(w > 1e-8)))
}
write.csv(summ, file.path(OUT, "summary.csv"), row.names = FALSE)

# The inputs themselves, so the Python side can be checked against the exact
# matrices Synth saw rather than rebuilding them and hoping.
write.csv(as.data.frame(Ypre_dm), file.path(OUT, "Ypre_demeaned.csv"), row.names = FALSE)
write.csv(as.data.frame(covmeans), file.path(OUT, "covariate_means.csv"))
writeLines(country_names, file.path(OUT, "country_names.txt"))

# A long-format panel for the estimator-level tests: raw log GDP plus the six
# covariates, pre- and post-treatment, with the UK treated from 2016Q3 (index
# 227). Exported here rather than rebuilt in Python for the same reason as the
# matrices above. The covariates are the same per-period series the pre-period
# means are taken from, so a test can use either.
POST <- 227:240
ALL <- c(PRE, POST)
long <- do.call(rbind, lapply(seq_along(country_names), function(j) {
  data.frame(country = country_names[j], t = ALL,
             y = y_long[ALL, j], real_con = con[ALL, j], real_inv = inv[ALL, j],
             real_exp = exp_[ALL, j], real_imp = imp[ALL, j],
             lab_prod = lab[ALL, j], tot_emp_pc = emp[ALL, j],
             treat = as.integer(j == TREATED_COL & ALL >= 227),
             stringsAsFactors = FALSE)
}))
stopifnot(!any(is.na(long)))
write.csv(long, file.path(OUT, "brexit_panel.csv"), row.names = FALSE)
cat(sprintf("panel: %d rows, %d units, %d periods, %d treated rows\n",
            nrow(long), length(unique(long$country)), length(ALL), sum(long$treat)))

writeLines(c(sprintf("Synth: %s", as.character(packageVersion("Synth"))),
             sprintf("R: %s.%s", R.version$major, R.version$minor),
             sprintf("treated: %s", country_names[TREATED_COL]),
             sprintf("n_donors: %d", length(donors)),
             sprintf("pre_periods: %d", length(PRE))),
           file.path(OUT, "versions.txt"))
cat("\n", readLines(file.path(OUT, "versions.txt")), sep = "\n")
