# Reference run for the `scta_texas_sb8` benchmark case.
#
# The authors' own replication package for Sun, Ben-Michael & Feller (2024),
# "Temporal Aggregation for the Synthetic Control Method" (AEA P&P 114:614-617),
# reduced to the numbers the Python case pins against. The construction is
# transcribed from their `time_aggregation.rmd` and run through their library,
# `augsynth` 0.2.0, so this is the authors' estimator on the authors' data.
#
# What the .rmd does, and what is reproduced here:
#
#   combined_dat   monthly live births per state, annualised (x 12); Texas
#                  treated from year_mth >= 2022.25, i.e. April 2022.
#   yearly_births  the state-year mean of the annualised series, kept for the
#                  six whole pre-treatment calendar years 2016-2021.
#   dat_w_yearly   the two stacked: every monthly row, plus the six yearly rows
#                  carried at year_mth = -year so they sort ahead of the months.
#   V              diag(rep(12 * year_wt, 6), rep(1, 75)).
#   fit            augsynth(..., V = V, fixedeff = TRUE): simplex synthetic
#                  control on the stacked design, unit fixed effects.
#
# Three details are preserved deliberately, because changing any of them would
# stop this being the authors' fit:
#
#   * `progfun = "none"` is a misspelling of augsynth's `progfunc`, so it lands
#     in `...` and is discarded. augsynth's default is progfunc = "None", so the
#     fit is the plain simplex either way and the typo changes nothing. It is
#     kept so the call is line-for-line the .rmd's.
#   * the yearly rows sort as -2021 < -2020 < ... < -2016, so the aggregate
#     block runs backwards in time. Every aggregate row carries the same V
#     weight, so the ordering does not affect the fit.
#   * the .rmd calls `summary(syn, inf = F)`. That argument belongs to an older
#     augsynth signature; 0.2.0 spells the same thing `inf_type = "none"`, which
#     is what is called here. Point estimates are unaffected -- the argument
#     only selects which inference procedure runs, and both settings run none.
#
# Two conventions of augsynth's internals are measured here instead of assumed,
# because the Python case pins the correspondence they imply:
#
#   * V enters as `X_c <- X_c %*% V` (augsynth:::fit_ridgeaug_formatted), which
#     scales each matching column by V and therefore weights the least-squares
#     objective by V squared. The paper writes the objective as
#     (a - B g)' V (a - B g). So augsynth's `year_wt` and a V-linear
#     implementation's `nu` are not the same knob; `ssr_v_*` and `ssr_v2_*`
#     below report the objective under each reading at augsynth's own weights,
#     so the mapping is established from the numbers.
#   * `fixedeff = TRUE` demeans by `rowMeans(wide_data$X)`
#     (augsynth:::demean_data) -- the mean over all 81 stacked pre-treatment
#     columns, six aggregates plus 75 months -- not the mean over the 75
#     disaggregated months alone. `unit_mean_*` below reports both for Texas so
#     the residual difference is attributable.
#
# Data: basedata/texas_sb8_births.csv -- the package's compileddata.csv verbatim.
# CDC WONDER Natality 2016-2022 (expanded), public domain, compiled by Bell,
# Stuart & Gemmill (2023, JAMA 330(3):281-282) and redistributed by the authors,
# who license users to download, copy and modify it.

suppressMessages({
  library(dplyr)
  library(augsynth)
})

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "basedata/texas_sb8_births.csv"

MONTHS <- c("January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December")

raw <- read.csv(path, stringsAsFactors = FALSE)
raw$month_num <- match(raw$month, MONTHS)

combined_dat <- raw %>%
  mutate(year_mth = year + (month_num - 1) / 12,
         annualized_all_births = all_births * 12) %>%
  mutate(treated = 1 * (year_mth >= 2022.25 & state == "Texas")) %>%
  select(state, year, year_mth, treated, annualized_all_births) %>%
  arrange(state, year_mth)

yearly_births <- combined_dat %>%
  group_by(state, year) %>%
  summarise(annualized_all_births = mean(annualized_all_births),
            .groups = "drop") %>%
  mutate(treated = 1 * (year >= 2022 & state == "Texas"))

dat_w_yearly <- bind_rows(
  combined_dat %>% select(state, year_mth, treated, annualized_all_births),
  yearly_births %>% filter(year < 2022) %>%
    mutate(year_mth = -year) %>%
    select(state, year_mth, treated, annualized_all_births)
) %>% arrange(year_mth)

syn_no_agg   <- suppressMessages(augsynth(annualized_all_births ~ treated, state,
                                          year_mth, combined_dat,
                                          progfunc = "ridge"))
syn_year_agg <- suppressMessages(augsynth(annualized_all_births ~ treated, state,
                                          year, yearly_births,
                                          progfun = "ridge"))
n_pre_months <- ncol(syn_no_agg$data$X)
n_pre_yrs    <- ncol(syn_year_agg$data$X)

fit_year <- function(year_wt) {
  V <- diag(c(rep(12 * year_wt, n_pre_yrs), rep(1, n_pre_months)))
  syn <- suppressMessages(augsynth(annualized_all_births ~ treated, state,
                                   year_mth, dat_w_yearly, progfun = "none",
                                   V = V, fixedeff = TRUE))
  list(syn = syn, att = summary(syn, inf_type = "none")$att, wt = year_wt)
}

# The pre-treatment objective at augsynth's own weights, on the design augsynth
# itself fits (its rowMeans demeaning), under each reading of V.
objective <- function(syn, year_wt, square) {
  X <- syn$data$X
  Xd <- X - rowMeans(X)                             # augsynth's fixed effect
  trt <- syn$data$trt == 1
  a <- as.numeric(Xd[trt, ])
  B <- t(Xd[!trt, , drop = FALSE])
  w <- as.numeric(syn$weights)
  vv <- if (square) (12 * year_wt)^2 else 12 * year_wt
  v <- c(rep(vv, n_pre_yrs), rep(1, n_pre_months))
  sum(v * as.numeric(a - B %*% w)^2)
}

GRID <- c(0, 0.5, 1, 2, 30)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("n_states\t%d\n", nrow(syn_no_agg$data$X)))
cat(sprintf("n_pre_months\t%d\n", n_pre_months))
cat(sprintf("n_pre_yrs\t%d\n", n_pre_yrs))
cat(sprintf("n_post\t%d\n", ncol(syn_no_agg$data$y)))

# Texas's fixed effect under each demeaning basis.
f1 <- fit_year(1)
Xf <- f1$syn$data$X
tx <- which(f1$syn$data$trt == 1)
cat(sprintf("unit_mean_stacked_tx\t%.6f\n", mean(Xf[tx, ])))
cat(sprintf("unit_mean_monthly_tx\t%.6f\n",
            mean(Xf[tx, (n_pre_yrs + 1):(n_pre_yrs + n_pre_months)])))

for (wt in GRID) {
  f <- fit_year(wt)
  att <- f$att
  post    <- att$Time >= 2022.25 & att$Time >= 0
  pre_mth <- att$Time <  2022.25 & att$Time >= 0
  pre_yr  <- att$Time <  0

  tag <- sub("\\.", "p", as.character(wt))
  cat(sprintf("est_%s\t%.6f\n",      tag, mean(att$Estimate[post])))
  cat(sprintf("rmse_mth_%s\t%.6f\n", tag, sqrt(mean(att$Estimate[pre_mth]^2))))
  cat(sprintf("rmse_yr_%s\t%.6f\n",  tag, sqrt(mean(att$Estimate[pre_yr]^2))))
  cat(sprintf("ssr_v_%s\t%.6f\n",    tag, objective(f$syn, wt, FALSE)))
  cat(sprintf("ssr_v2_%s\t%.6f\n",   tag, objective(f$syn, wt, TRUE)))

  w  <- as.numeric(f$syn$weights)
  nm <- rownames(f$syn$weights)
  for (i in seq_along(w)) {
    if (abs(w[i]) > 1e-8) {
      cat(sprintf("weight\t%s@%s\t%.8f\n", tag, nm[i], w[i]))
    }
  }
}

cat("== SESSION INFO ==\n")
print(sessionInfo())
