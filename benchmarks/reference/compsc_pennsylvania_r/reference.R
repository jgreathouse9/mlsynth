# csc_replication.R -- (c) O. Boussim, author of "Compositional Synthetic
# Controls" (arXiv:2607.16991). Provided by the author and reproduced here
# unmodified above the capture block at the end, solely so mlsynth's COMPSC
# estimator can be checked against it. Authorship of the method and of this
# script is his; mlsynth claims neither. The two additive changes below are
# marked, and nothing in the estimation path is altered.
#
# The script as supplied reads "data_energy.csv" from the working directory and
# prints its tables rounded to three decimals. Two changes, both additive: the
# input path comes from argv so the bundle can run it against the panel mlsynth
# ships, and a "== REFERENCE VALUES ==" block is emitted at full precision so a
# benchmark can pin the numbers instead of the printout.

.args <- commandArgs(trailingOnly = TRUE)
.data_path <- if (length(.args) >= 1) .args[1] else "data_energy.csv"

################################################################################
## Compositional Synthetic Controls: Replication Package
## Application: Pennsylvania's Alternative Energy Portfolio Standard (AEPS)
##
## Data:    U.S. EIA state-level net generation, 1990-2023
## Categories: Natural gas; Coal & oil (fossil); Renewables (excl. nuclear)
## Treated: Pennsylvania (PA), T0 = 2003  (Act 213 of 2004)
## Donor pool: 42 states (excl. structural zeros and policy-contaminated units)
##
## Required: R >= 4.0, quadprog
################################################################################

library(quadprog)

cat("================================================================\n")
cat("  Compositional Synthetic Controls -- Replication\n")
cat("================================================================\n\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

dat <- read.csv(.data_path, stringsAsFactors = FALSE)

# Three categories: gas, fossil (coal & oil), renewables -- nuclear excluded
dat$total3  <- dat$gas + dat$fossil + dat$renewables
dat$sh_gas  <- dat$gas       / dat$total3
dat$sh_fos  <- dat$fossil    / dat$total3
dat$sh_ren  <- dat$renewables / dat$total3

states <- sort(unique(dat$state))
years  <- sort(unique(dat$year))

# ══════════════════════════════════════════════════════════════════════════════
# 2. DONOR POOL
# ══════════════════════════════════════════════════════════════════════════════

# (a) Exclude states with non-positive generation in any category, any year
#     (structural zeros: log-odds undefined)
struct_zero <- c()
for (s in states) {
  d <- dat[dat$state == s, ]
  if (any(d$gas <= 0) | any(d$fossil <= 0) | any(d$renewables <= 0))
    struct_zero <- c(struct_zero, s)
}

# (b) Exclude states that adopted comparable alternative energy portfolio
#     standards during the sample (policy contamination):
#       OH -- Alternative Energy Portfolio Standard, SB 221, 2008
#       WV -- Alternative and Renewable Energy Portfolio Standard, HB 103, 2009
# (c) Exclude compositional outliers with near-structural zeros:
#       VT -- gas and fossil shares each below 0.1% throughout the sample
policy_excl <- c("OH", "WV", "VT")

all_excluded <- unique(c(struct_zero, policy_excl))
eligible     <- setdiff(states, c("PA", all_excluded))

cat("Structural-zero states excluded:", paste(struct_zero, collapse = ", "), "\n")
cat("Policy/outlier states excluded: ", paste(policy_excl, collapse = ", "), "\n")
cat("Donor pool size: ", length(eligible), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOG-ODDS TRANSFORMATION (ALR, baseline = renewables)
# ══════════════════════════════════════════════════════════════════════════════

T0     <- 2003   # last pre-treatment year
T_all  <- years
n_pre  <- sum(years <= T0)   # 14
n_post <- sum(years >  T0)   # 20
p      <- 3                  # categories
pminus1 <- p - 1             # 2 log-odds per period

all_units <- c("PA", eligible)
J <- length(all_units)       # 1 treated + 42 donors = 43

# Arrays: shares[k,t,j], logodds[k,t,j]
logodds <- array(NA, dim = c(pminus1, length(T_all), J))
shares  <- array(NA, dim = c(p,       length(T_all), J))

for (jj in seq_along(all_units)) {
  d <- dat[dat$state == all_units[jj], ]
  d <- d[order(d$year), ]
  for (tt in seq_along(T_all)) {
    r <- d[d$year == T_all[tt], ]
    shares[1, tt, jj] <- r$sh_gas
    shares[2, tt, jj] <- r$sh_fos
    shares[3, tt, jj] <- r$sh_ren
    logodds[1, tt, jj] <- log(r$sh_gas / r$sh_ren)
    logodds[2, tt, jj] <- log(r$sh_fos / r$sh_ren)
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. CSC WEIGHT ESTIMATION -- STACKED QP
# ══════════════════════════════════════════════════════════════════════════════
#
#   Following Sun, Ben-Michael & Feller (2025), stack the (p-1) log-odds
#   equations across categories and pre-treatment periods, yielding an
#   effective sample of N = (p-1) * T0.
#
#   min_{w in W} sum_{t<=T0} ||ell(pi_{1,t}) - sum_j w_j ell(pi_{j,t})||^2
#
#   = min_{w in W} sum_{t<=T0} sum_{k=1}^{p-1}
#       ( ell_k(pi_{1,t}) - sum_j w_j ell_k(pi_{j,t}) )^2
#
#   This is a QP: min 0.5 w'Hw - f'w  s.t. w >= 0, 1'w = 1

csc_weights <- function(treated_idx, donor_indices, lo, pre_t) {
  nd    <- length(donor_indices)
  K     <- dim(lo)[1]          # p-1
  n_pre <- length(pre_t)
  N     <- K * n_pre           # stacked sample size

  # Build design matrix X (N x nd) and response Y (N x 1)
  Y <- numeric(N)
  X <- matrix(0, nrow = N, ncol = nd)
  ri <- 0L
  for (kk in seq_len(K)) {
    for (tt in pre_t) {
      ri <- ri + 1L
      Y[ri] <- lo[kk, tt, treated_idx]
      for (dd in seq_along(donor_indices))
        X[ri, dd] <- lo[kk, tt, donor_indices[dd]]
    }
  }

  # QP matrices
  H <- crossprod(X)
  f <- crossprod(X, Y)
  H <- H + 1e-8 * diag(nd)     # numerical regularisation

  # Constraints: 1'w = 1 (equality, meq = 1), w >= 0
  Amat <- cbind(rep(1, nd), diag(nd))
  bvec <- c(1, rep(0, nd))

  sol <- solve.QP(Dmat = H, dvec = f, Amat = Amat, bvec = bvec, meq = 1)
  w   <- sol$solution
  w[w < 1e-6] <- 0
  w <- w / sum(w)
  return(w)
}

pre_idx   <- which(T_all <= T0)
post_idx  <- which(T_all >  T0)
donor_idx <- 2:J

w_hat <- csc_weights(1, donor_idx, logodds, pre_idx)

cat("\n--- Donor Weights ---\n")
wdf <- data.frame(Donor = all_units[donor_idx], Weight = round(w_hat, 3))
wdf <- wdf[wdf$Weight > 0, ]
wdf <- wdf[order(-wdf$Weight), ]
print(wdf, row.names = FALSE)
cat("Active donors:", nrow(wdf), "\n")
cat("Stacked sample: (p-1)*T0 =", pminus1, "*", n_pre, "=", pminus1 * n_pre, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. COUNTERFACTUAL SHARES (inverse ALR = softmax)
# ══════════════════════════════════════════════════════════════════════════════

synth_lo <- array(0, dim = c(pminus1, length(T_all)))
for (kk in seq_len(pminus1))
  for (tt in seq_along(T_all))
    synth_lo[kk, tt] <- sum(w_hat * logodds[kk, tt, donor_idx])

synth_sh <- matrix(0, nrow = p, ncol = length(T_all))
for (tt in seq_along(T_all)) {
  e1 <- exp(synth_lo[1, tt])
  e2 <- exp(synth_lo[2, tt])
  denom <- 1 + e1 + e2
  synth_sh[1, tt] <- e1 / denom    # gas
  synth_sh[2, tt] <- e2 / denom    # fossil
  synth_sh[3, tt] <- 1  / denom    # renewables
}

cat("\n--- Pre-treatment Balance ---\n")
cat(sprintf("                  PA      Synthetic\n"))
cat(sprintf("Natural gas:   %5.1f%%      %5.1f%%\n",
            mean(shares[1, pre_idx, 1]) * 100, mean(synth_sh[1, pre_idx]) * 100))
cat(sprintf("Coal & oil:    %5.1f%%      %5.1f%%\n",
            mean(shares[2, pre_idx, 1]) * 100, mean(synth_sh[2, pre_idx]) * 100))
cat(sprintf("Renewables:    %5.1f%%      %5.1f%%\n",
            mean(shares[3, pre_idx, 1]) * 100, mean(synth_sh[3, pre_idx]) * 100))

# ══════════════════════════════════════════════════════════════════════════════
# 6. AITCHISON DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

aitchison_dist <- function(pi1, pi2) {
  p <- length(pi1)
  ss <- 0
  for (i in 1:p) for (j in 1:p)
    ss <- ss + (log(pi1[i] / pi1[j]) - log(pi2[i] / pi2[j]))^2
  sqrt(ss / (2 * p))
}

# Pre-treatment fit
ait_pre <- sapply(seq_along(pre_idx), function(i)
  aitchison_dist(shares[, pre_idx[i], 1], synth_sh[, pre_idx[i]]))
ait_rmspe_pre <- sqrt(mean(ait_pre^2))

euc_pre <- sapply(seq_along(pre_idx), function(i)
  sqrt(sum((shares[, pre_idx[i], 1] - synth_sh[, pre_idx[i]])^2)))
euc_rmspe_pre <- sqrt(mean(euc_pre^2)) * 100

cat(sprintf("\nAitchison pre-treatment RMSPE: %.3f\n", ait_rmspe_pre))
cat(sprintf("Euclidean share RMSPE (pp):    %.2f\n", euc_rmspe_pre))

# ══════════════════════════════════════════════════════════════════════════════
# 7. TREATMENT EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

cat("\n--- Estimated Compositional Treatment Effects ---\n")
cat(sprintf("%4s %8s %8s %8s %12s %12s %8s\n",
            "Year", "ATT_gas", "ATT_fos", "ATT_ren", "LRTE_g/f", "LRTE_g/r", "delta_A"))
cat(paste(rep("-", 72), collapse = ""), "\n")

results <- data.frame()
for (tt in post_idx) {
  yr  <- T_all[tt]
  obs <- shares[, tt, 1]
  cf  <- synth_sh[, tt]

  att     <- (obs - cf) * 100
  lrte_gf <- log(obs[1] / obs[2]) - log(cf[1] / cf[2])
  lrte_gr <- log(obs[1] / obs[3]) - log(cf[1] / cf[3])
  d_a     <- aitchison_dist(obs, cf)

  results <- rbind(results, data.frame(
    year = yr, att_gas = att[1], att_fos = att[2], att_ren = att[3],
    lrte_gf = lrte_gf, lrte_gr = lrte_gr, delta_A = d_a))

  if (yr %% 2 == 0 | yr == 2023)
    cat(sprintf("%4d %+7.1f  %+7.1f  %+7.1f  %+10.2f  %+10.2f  %7.2f\n",
                yr, att[1], att[2], att[3], lrte_gf, lrte_gr, d_a))
}

# ══════════════════════════════════════════════════════════════════════════════
# 8. PLACEBO INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

cat("\n--- Placebo Permutation Test ---\n")
cat("Running", length(donor_idx), "placebo estimations...\n")

# PA RMSPE ratio
ait_post_pa <- sapply(seq_along(post_idx), function(i)
  aitchison_dist(shares[, post_idx[i], 1], synth_sh[, post_idx[i]]))
rmspe_pre_pa  <- sqrt(mean(ait_pre^2))
rmspe_post_pa <- sqrt(mean(ait_post_pa^2))
R_pa <- rmspe_post_pa / rmspe_pre_pa

# Placebo for each donor
n_donors      <- length(donor_idx)
R_placebo     <- numeric(n_donors)
rmspe_pre_plc <- numeric(n_donors)
names(R_placebo)     <- all_units[donor_idx]
names(rmspe_pre_plc) <- all_units[donor_idx]

for (dd in seq_along(donor_idx)) {
  j <- donor_idx[dd]
  plc_donors <- setdiff(donor_idx, j)

  w_p <- tryCatch(
    csc_weights(j, plc_donors, logodds, pre_idx),
    error = function(e) rep(1 / length(plc_donors), length(plc_donors))
  )

  # Synthetic log-odds for unit j
  slo <- array(0, dim = c(pminus1, length(T_all)))
  for (kk in seq_len(pminus1))
    for (tt in seq_along(T_all))
      slo[kk, tt] <- sum(w_p * logodds[kk, tt, plc_donors])

  ssh <- matrix(0, p, length(T_all))
  for (tt in seq_along(T_all)) {
    e1 <- exp(slo[1, tt]); e2 <- exp(slo[2, tt]); denom <- 1 + e1 + e2
    ssh[1, tt] <- e1 / denom; ssh[2, tt] <- e2 / denom; ssh[3, tt] <- 1 / denom
  }

  ap <- sapply(seq_along(pre_idx),  function(i) aitchison_dist(shares[, pre_idx[i],  j], ssh[, pre_idx[i]]))
  aq <- sapply(seq_along(post_idx), function(i) aitchison_dist(shares[, post_idx[i], j], ssh[, post_idx[i]]))

  rmspe_pre_plc[dd] <- sqrt(mean(ap^2))
  R_placebo[dd]     <- sqrt(mean(aq^2)) / rmspe_pre_plc[dd]
}

# Screen: retain donors with pre-RMSPE <= 5 * PA's
m_screen <- 5
retained <- rmspe_pre_plc <= m_screen * rmspe_pre_pa
M        <- sum(retained) + 1   # +1 for PA

cat(sprintf("PA: RMSPE_pre = %.3f, RMSPE_post = %.3f, R = %.2f\n",
            rmspe_pre_pa, rmspe_post_pa, R_pa))
cat(sprintf("Donors retained (pre-RMSPE <= %gx PA): %d / %d\n",
            m_screen, sum(retained), n_donors))
cat(sprintf("Total M (incl PA): %d\n", M))

n_exceed <- sum(R_placebo[retained] >= R_pa) + 1  # +1 for PA
p_value  <- n_exceed / M

cat("\nTop placebo ratios (retained):\n")
R_ret <- sort(R_placebo[retained], decreasing = TRUE)
print(head(R_ret, 6))

cat(sprintf("\nExact p-value: %d / %d = %.3f\n", n_exceed, M, p_value))

# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

cat("\n================================================================\n")
cat("                    REPLICATION SUMMARY\n")
cat("================================================================\n")
cat(sprintf("Treated unit:          Pennsylvania\n"))
cat(sprintf("Treatment date:        T0 = %d  (Act 213 of 2004)\n", T0))
cat(sprintf("Pre-treatment:         %d periods  (1990-%d)\n", n_pre, T0))
cat(sprintf("Post-treatment:        %d periods  (%d-2023)\n", n_post, T0 + 1))
cat(sprintf("Donor pool:            %d states\n", length(eligible)))
cat(sprintf("Stacked sample:        (p-1)*T0 = %d\n", pminus1 * n_pre))
cat(sprintf("Donors with wt > 0:    %d\n", sum(w_hat > 0)))
cat(sprintf("Aitchison pre-RMSPE:   %.3f\n", ait_rmspe_pre))
cat(sprintf("Share pre-RMSPE (pp):  %.2f\n", euc_rmspe_pre))
cat(sprintf("Gas ATT in 2022:       +%.1f pp\n", results$att_gas[results$year == 2022]))
cat(sprintf("LRTE gas/fos 2022:     %.2f\n", results$lrte_gf[results$year == 2022]))
cat(sprintf("PA RMSPE ratio:        %.2f\n", R_pa))
cat(sprintf("Placebo p-value:       %d/%d = %.3f\n", n_exceed, M, p_value))
cat("================================================================\n")

# ══════════════════════════════════════════════════════════════════════════════
# 10. MACHINE-READABLE CAPTURE  (appended; nothing above is modified)
# ══════════════════════════════════════════════════════════════════════════════

cat("\n== REFERENCE VALUES ==\n")
for (i in seq_along(donor_idx)) {
  w <- w_hat[i]
  if (w > 0) cat(sprintf("weight\t%s\t%.10f\n", all_units[donor_idx[i]], w))
}
cat(sprintf("n_donors\t%d\n", length(eligible)))
cat(sprintf("n_active_donors\t%d\n", sum(w_hat > 0)))
cat(sprintf("aitchison_rmspe_pre\t%.10f\n", ait_rmspe_pre))
cat(sprintf("share_rmspe_pre_pp\t%.10f\n", euc_rmspe_pre))
for (nm in c("gas", "fos", "ren"))
  cat(sprintf("pre_share_%s_pct\t%.10f\n", nm,
              mean(shares[match(nm, c("gas", "fos", "ren")), pre_idx, 1]) * 100))
for (i in seq_len(nrow(results))) {
  yr <- results$year[i]
  cat(sprintf("att_gas_%d\t%.10f\n", yr, results$att_gas[i]))
  cat(sprintf("att_fos_%d\t%.10f\n", yr, results$att_fos[i]))
  cat(sprintf("att_ren_%d\t%.10f\n", yr, results$att_ren[i]))
  cat(sprintf("lrte_gf_%d\t%.10f\n", yr, results$lrte_gf[i]))
  cat(sprintf("lrte_gr_%d\t%.10f\n", yr, results$lrte_gr[i]))
  cat(sprintf("delta_A_%d\t%.10f\n", yr, results$delta_A[i]))
}
cat(sprintf("rmspe_pre_pa\t%.10f\n", rmspe_pre_pa))
cat(sprintf("rmspe_post_pa\t%.10f\n", rmspe_post_pa))
cat(sprintf("rmspe_ratio_pa\t%.10f\n", R_pa))
cat(sprintf("placebo_n_retained\t%d\n", M))
cat(sprintf("placebo_n_exceed\t%d\n", n_exceed))
cat(sprintf("placebo_p_value\t%.10f\n", p_value))
cat("== SESSION ==\n")
print(sessionInfo())
