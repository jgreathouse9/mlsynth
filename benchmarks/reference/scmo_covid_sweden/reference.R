# Reference run for the `scmo_covid_sweden` benchmark case.
#
# Runs the numeric core of Tian, Lee & Panchenko's own COVID_analysis.R (Online
# Appendix B.3: Sweden's light-touch non-pharmaceutical interventions) and emits
# every quantity the appendix reports through a figure. The paper prints the
# donor weights (Table B.3) and a handful of percentages in the text; the
# aggregate treatment effects (Figure B.5), the per-period and aggregate
# p-values (Figures B.6 and B.7) and the six robustness charts (Figures B.8 to
# B.13) are drawn, not tabulated. Their code computes all of them, so this
# script keeps the computation, drops every plotting call, and prints the
# objects the plots were built from.
#
# What is kept verbatim from their script: the fn_W quadratic program
# (quadprog::solve.QP), the per-domain matching matrix (demeaned per unit and
# per outcome, standardized by cross-unit SD, weighted by the diagonal V with
# 1/#pre_k on each column of outcome k), the level-shifted synthetic, the
# one-sided truncation of the post-treatment gaps, the eta = 0.01 guard applied
# after standardizing by the average post-period cross-sectional SD, and the
# p-value as the treated unit's rank among the sorted RMSPE ratios.
#
# Data are vendored alongside this script (see NOTICE): covid_panel.csv is the
# authors' own assembled panel (Data_COVID/data.csv of their replication
# package) cut to the 27 countries, the window 2019-01-01 to 2020-09-30, and the
# twelve outcomes the three domains use. It is written from the same Parquet the
# Python case reads, so both sides provably see the same numbers.
#
# Run from the repository root:  Rscript benchmarks/reference/scmo_covid_sweden/reference.R
suppressMessages(library(quadprog))

fn_W <- function(Zi, ZJ, V) {
  J <- nrow(ZJ)
  Dmat <- ZJ %*% V %*% t(ZJ) + (10^-7) * diag(J)
  dvec <- ZJ %*% V %*% Zi
  Amat <- cbind(cbind(rep(1, J)), diag(1, J, J))
  bvec <- c(1, rep(0, J))
  solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
}

rmse <- function(x) sqrt(mean(x^2))

base <- "benchmarks/reference/scmo_covid_sweden"
data <- read.csv(file.path(base, "covid_panel.csv"), check.names = FALSE)
data$date <- as.Date(data$date)

start_date <- as.Date("2019-01-01")
day_back <- as.Date("2019-10-01")           # the backdating cutoff
day_treat_vec <- as.Date(c("2020-03-28", "2020-02-15", "2020-02-15"))

codes0 <- c("AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST", "FIN", "FRA",
            "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "NLD", "NOR",
            "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "CHE", "GBR")

outcomes <- list(
  c("covid_cases", "covid_deaths", "deaths_TOTAL"),
  c("labour_employ", "labour_absence", "labour_hours"),
  c("gdp_a", "import", "export", "industrial", "retail_both", "CPI"))
domains <- c("health", "labour", "economic")

vals <- new.env(parent = emptyenv())        # key -> value, printed at the end
put <- function(key, value) assign(key, as.numeric(value), envir = vals)

weights_out <- list()

for (s in 1:3) {

  day_treat <- day_treat_vec[s]
  codes <- codes0
  if (s == 1) codes <- setdiff(codes0, "IRL")     # no weekly deaths
  if (s == 2) codes <- setdiff(codes0, "DEU")     # no post-treatment absence/hours
  N <- length(codes)
  dom <- domains[s]

  # --- the three matching matrices: backdated, not demeaned, benchmark -----
  W <- c()
  for (t in 1:3) {
    Z <- c()
    for (i in codes) {
      cut <- c(day_back, day_treat, day_treat)[t]
      values <- data[data$code == i & data$date >= start_date & data$date <= cut,
                     outcomes[[s]], drop = FALSE]
      if (t != 2) {
        for (col in 1:ncol(values)) {
          values[, col] <- values[, col] - mean(values[, col], na.rm = TRUE)
        }
      }
      Z <- rbind(Z, unlist(values))
    }
    Z <- Z[, !apply(Z, 2, anyNA)]
    Z <- Z[, apply(Z, 2, sd) != 0]
    Z <- scale(Z, center = FALSE, scale = apply(Z, 2, sd))

    V <- c()
    for (l in outcomes[[s]]) {
      count <- sum(sapply(colnames(Z), substr, 1, nchar(l)) == l)
      V <- c(V, rep(1 / count, count))
    }
    W <- cbind(W, fn_W(cbind(Z[codes == "SWE", ]), Z[codes != "SWE", ], diag(V, ncol(Z))))
  }
  W_back <- W[, 1]
  W_ND <- W[, 2]
  W_SWE <- W[, 3]
  put(paste0("ncol_Z_", dom), ncol(Z))
  donors <- setdiff(codes, "SWE")
  for (j in seq_along(donors)) weights_out[[length(weights_out) + 1]] <-
    c(paste0(dom, "/", donors[j]), W_SWE[j])

  # --- permutation weights: every donor in the treated seat ----------------
  W_mat <- c()
  for (i in donors) {
    W_mat <- cbind(W_mat, fn_W(cbind(Z[codes == i, ]), Z[codes != i, ], diag(V, ncol(Z))))
  }

  # --- leave-one-unit-out: drop each donor that carries weight -------------
  LOO <- donors[round(W_SWE, 2) >= .01]
  W_LOO <- matrix(0, N - 1, length(LOO))
  for (k in LOO) {
    W_LOO[donors != k, which(LOO == k)] <-
      fn_W(cbind(Z[codes == "SWE", ]), Z[!codes %in% c("SWE", k), ], diag(V, ncol(Z)))
  }
  put(paste0("n_loo_", dom), length(LOO))

  preloss_agg <- c()
  postloss_agg <- c()
  gaps_agg <- gap_std_agg <- vector("list", length(outcomes[[s]]))

  for (y in outcomes[[s]]) {

    Y <- c()
    for (i in codes) Y <- rbind(Y, c(unlist(data[data$code == i & data$date >= start_date, y])))
    dates <- unique(data$date[data$date >= start_date])[!apply(Y, 2, anyNA)]
    Y <- cbind(Y[, !apply(Y, 2, anyNA)])
    Yi <- Y[codes == "SWE", ]
    YJ <- Y[codes != "SWE", ]
    pre <- dates <= day_treat
    q2 <- dates >= as.Date("2020-04-01") & dates < as.Date("2020-07-01")

    # benchmark synthetic, level-shifted onto Sweden's pre-treatment mean
    synthY1 <- t(YJ) %*% W_SWE
    synthY1 <- synthY1 - mean(synthY1[pre]) + mean(Yi[pre])

    # single-outcome SC (Figure B.12)
    Zs <- Y[, pre]
    Zs <- Zs[, !apply(Zs, 2, anyNA)]
    Zs <- Zs[, apply(Zs, 2, sd) != 0]
    for (i in 1:N) Zs[i, ] <- Zs[i, ] - mean(Zs[i, ])
    Zs <- scale(Zs, center = FALSE, scale = apply(Zs, 2, sd))
    Vs <- rep(1 / ncol(Zs), ncol(Zs))
    Ws <- fn_W(cbind(Zs[codes == "SWE", ]), Zs[codes != "SWE", ], diag(Vs, ncol(Zs)))
    synthSG <- t(YJ) %*% Ws
    synthSG <- synthSG - mean(synthSG[pre]) + mean(Yi[pre])

    # no demeaning (Figure B.13) -- matched in levels, and not shifted
    synthND <- t(YJ) %*% W_ND

    # backdating (Figure B.9)
    synthBD <- t(YJ) %*% W_back
    back <- dates <= day_back
    if (sum(back) > 1) synthBD <- synthBD - mean(synthBD[back]) + mean(Yi[back])

    # leave-one-unit-out and leave-one-outcome-out bands (Figures B.10, B.11)
    loo_pre <- loo_q2 <- c()
    for (k in LOO) {
      synthLOO <- t(YJ) %*% W_LOO[, which(LOO == k)]
      synthLOO <- synthLOO - mean(synthLOO[pre]) + mean(Yi[pre])
      loo_pre <- c(loo_pre, mean(synthLOO[pre]))
      loo_q2 <- c(loo_q2, mean(synthLOO[q2]))
    }
    looo_pre <- looo_q2 <- c()
    for (k in 1:length(outcomes[[s]])) {
      Zk <- c()
      for (i in codes) {
        values <- data[data$code == i & data$date >= start_date & data$date <= day_treat,
                       outcomes[[s]][-k], drop = FALSE]
        for (col in 1:ncol(values)) {
          values[, col] <- values[, col] - mean(values[, col], na.rm = TRUE)
        }
        Zk <- rbind(Zk, unlist(values))
      }
      Zk <- Zk[, !apply(Zk, 2, anyNA)]
      Zk <- Zk[, apply(Zk, 2, sd) != 0]
      Zk <- scale(Zk, center = FALSE, scale = apply(Zk, 2, sd))
      Vk <- c()
      for (l in outcomes[[s]][-k]) {
        count <- sum(sapply(colnames(Zk), substr, 1, nchar(l)) == l)
        Vk <- c(Vk, rep(1 / count, count))
      }
      Wk <- fn_W(cbind(Zk[codes == "SWE", ]), Zk[codes != "SWE", ], diag(Vk, ncol(Zk)))
      synthLOOO <- t(YJ) %*% Wk
      synthLOOO <- synthLOOO - mean(synthLOOO[pre]) + mean(Yi[pre])
      looo_pre <- c(looo_pre, mean(synthLOOO[pre]))
      looo_q2 <- c(looo_q2, mean(synthLOOO[q2]))
    }

    # the radar charts are these window means
    put(paste0("sweden_pre_", y), mean(Yi[pre]))
    put(paste0("sweden_q2_", y), mean(Yi[q2]))
    for (nm in c("benchmark", "single", "nodemean", "backdate")) {
      series <- switch(nm, benchmark = synthY1, single = synthSG,
                       nodemean = synthND, backdate = synthBD)
      put(paste0("synth_", nm, "_pre_", y), mean(series[pre]))
      put(paste0("synth_", nm, "_q2_", y), mean(series[q2]))
    }
    put(paste0("synth_loo_min_q2_", y), min(loo_q2))
    put(paste0("synth_loo_max_q2_", y), max(loo_q2))
    put(paste0("synth_looo_min_q2_", y), min(looo_q2))
    put(paste0("synth_looo_max_q2_", y), max(looo_q2))
    put(paste0("synth_loo_min_pre_", y), min(loo_pre))
    put(paste0("synth_loo_max_pre_", y), max(loo_pre))
    put(paste0("synth_looo_min_pre_", y), min(looo_pre))
    put(paste0("synth_looo_max_pre_", y), max(looo_pre))

    # --- gaps, for Sweden and for every donor in the treated seat ----------
    gaps <- c(synthY1 - Yi)
    for (k in donors) {
      sY1 <- t(Y[codes != k, ]) %*% W_mat[, donors == k]
      sY1 <- sY1 - mean(sY1[pre]) + mean(Y[codes == k, ][pre])
      gaps <- cbind(gaps, sY1 - Y[codes == k, ])
    }
    colnames(gaps) <- c("SWE*", donors)
    rownames(gaps) <- as.character(dates)
    if (y %in% c("labour_absence")) gaps <- -gaps

    T0 <- sum(pre)
    TT <- length(dates)
    sigma <- mean(apply(Y[, !pre], 2, sd))     # average post-period cross-sectional SD

    gap1 <- -gaps[, 1][!pre]                   # observed minus synthetic
    gap_std <- gap1 / sigma
    gap_percent <- gap1 / Yi[!pre] * 100
    names(gap1) <- names(gap_percent) <- names(gap_std) <- as.character(dates[!pre])
    gap_std_agg[[which(outcomes[[s]] == y)]] <- gap_std
    for (d in names(gap1)) {
      put(paste0("gap_", y, "_", d), gap1[[d]])
      put(paste0("gappct_", y, "_", d), gap_percent[[d]])
    }
    if (y == "deaths_TOTAL") {                 # their own sum(gap_list[[3]][...])
      put("cumgap_deaths_TOTAL_to_2020-07-26",
          sum(gap1[names(gap1) <= "2020-07-26"]))
    }

    # one-sided inference: keep only the side the paper tests, standardize,
    # then the eta guard (0.01 after standardizing == 0.01 * sigma before)
    gaps[(T0 + 1):TT, ][gaps[(T0 + 1):TT, ] > 0] <- 0
    gaps[(T0 + 1):TT, ] <- abs(gaps[(T0 + 1):TT, ])
    gaps <- gaps / sigma
    gaps_agg[[which(outcomes[[s]] == y)]] <- gaps[!pre, ]

    eta <- 0.01
    postloss <- apply(cbind(gaps[(T0 + 1):TT, ]), 2, rmse) + eta
    preloss <- apply(cbind(gaps[1:T0, ]), 2, rmse) + eta
    postloss_agg <- cbind(postloss_agg, postloss)
    preloss_agg <- cbind(preloss_agg, preloss)

    ratio <- sort(postloss / preloss)
    pvalue <- 1 - (which(names(ratio) == "SWE*") - 1) / N
    put(paste0("p_", y), pvalue)
    put(paste0("ratio_", y), (postloss / preloss)[["SWE*"]])

    # p-value in each post-treatment period
    sig <- 0
    first_sig <- NA
    for (t in (T0 + 1):TT) {
      r <- sort((gaps[t, ] + eta) / preloss)
      p <- 1 - (which(names(r) == "SWE*") - 1) / N
      put(paste0("p_", y, "_", as.character(dates[t])), p)
      if (p <= 3 / N + 1e-9) {
        sig <- sig + 1
        if (is.na(first_sig)) first_sig <- as.numeric(dates[t] - day_treat)
      }
    }
    put(paste0("nsig_", y), sig)
    # days after the treatment date, or -1 if never significant
    put(paste0("firstsig_days_", y), ifelse(is.na(first_sig), -1, first_sig))
  }

  # --- the aggregate index and its p-values (Figures B.5 and B.7) ----------
  # Their script aggregates the public-health domain on the weekly all-cause
  # death dates and the other two on three quarter marks. (Their date labels are
  # R's numeric representation, because `rownames(m) <- Dates` coerces through
  # the underlying number while `names(x) <- Dates` does not; the labels here are
  # ISO throughout, and the windows below are built to match.)
  if (s == 1) {
    agg_dates <- as.Date(rownames(gaps_agg[[which(outcomes[[s]] == "deaths_TOTAL")]]))
  } else {
    agg_dates <- as.Date(paste0("2020-", c(3, 6, 9), "-16"))
  }
  dates1 <- c(day_treat, agg_dates)

  taus <- c()
  pvalues <- c()
  for (t in 2:length(dates1)) {
    # the window spans both endpoints, as their `dates1[t-1]:dates1[t]` does
    window <- as.character(seq(dates1[t - 1], dates1[t], by = "day"))
    gap1s <- c()
    for (i in 1:length(outcomes[[s]])) {
      gap1s <- c(gap1s, mean(gap_std_agg[[i]][
        names(gap_std_agg[[i]]) %in% window]))
    }
    taus <- c(taus, mean(gap1s))

    g <- c()
    for (i in 1:length(outcomes[[s]])) {
      keep <- rownames(gaps_agg[[i]]) %in% window
      g <- cbind(g, colMeans(rbind(gaps_agg[[i]][keep, ])))
    }
    r <- sort(rowMeans(g) / rowMeans(preloss_agg))
    pvalues <- c(pvalues, 1 - (which(names(r) == "SWE*") - 1) / N)
  }
  r <- sort(rowMeans(postloss_agg) / rowMeans(preloss_agg))
  put(paste0("p_agg_", dom), 1 - (which(names(r) == "SWE*") - 1) / N)
  put(paste0("tau_agg_", dom), mean(taus))
  for (t in seq_along(taus)) {
    label <- as.character(dates1[t + 1])
    put(paste0("tau_", dom, "_", label), taus[t])
    put(paste0("pagg_", dom, "_", label), pvalues[t])
  }
  put(paste0("alpha_", dom), 3 / N)
  put(paste0("n_units_", dom), N)
}

cat("== REFERENCE VALUES ==\n")
for (key in sort(ls(vals))) cat(sprintf("%s\t%.6f\n", key, get(key, envir = vals)))
for (w in weights_out) cat(sprintf("weight\t%s\t%.6f\n", w[1], as.numeric(w[2])))
cat("== SESSION INFO ==\n")
print(sessionInfo())
