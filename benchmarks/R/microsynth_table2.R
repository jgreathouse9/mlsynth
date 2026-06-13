# Reference generator for the microsynth Table 2 reproduction (JSS v97i02,
# Robbins & Davenport 2021) -- the main multi-outcome analysis ``sea1`` on the
# Seattle DMI data: joint match on i_felony, i_misdemea, i_drugs, any_crime.
#
# Reproduces the "top panel" of the package's summary(sea1) results table:
# Trt / Con / Pct.Chng per outcome (cumulative over the post window), plus the
# permutation p-values (test = "lower"). The Linear (Taylor-linearization)
# p-values/CIs in the JSS top panel are a survey-methodology variance estimate
# not ported to mlsynth; mlsynth's inference is the placebo permutation, so the
# Perm.pVal column is the comparison target.
#
# Install route (CRAN is firewalled here): see microsynth_seattle.R.
# Run:  Rscript benchmarks/R/microsynth_table2.R

suppressMessages(library(microsynth))
cat("microsynth version:", as.character(packageVersion("microsynth")), "\n")
data(seattledmi)

cov.var   <- c("TotalPop", "BLACK", "HISPANIC", "Males_1521", "HOUSEHOLDS",
               "FAMILYHOUS", "FEMALE_HOU", "RENTER_HOU", "VACANT_HOU")
match.out <- c("i_felony", "i_misdemea", "i_drugs", "any_crime")

set.seed(99199)
sea1 <- microsynth(
  seattledmi,
  idvar = "ID", timevar = "time", intvar = "Intervention",
  start.pre = 1, end.pre = 12, end.post = 16,
  match.out = match.out, match.covar = cov.var,
  result.var = match.out, omnibus.var = match.out,
  test = "lower", perm = 250, jack = TRUE, n.cores = 1
)

cat("=== summary(sea1) Results (end.post = 16) ===\n")
print(round(sea1$Results[["16"]], 4))

# Cumulative Trt / Con / Pct.Chng directly from the per-period Plot.Stats.
ps <- sea1$Plot.Stats
post <- as.character(13:16)
cat("\n=== cumulative point estimates (post 13:16) ===\n")
for (o in match.out) {
  trt <- sum(ps$Treatment[o, post])
  con <- sum(ps$Control[o, post])
  cat(sprintf("%-11s Trt=%4.0f Con=%8.2f Pct=%6.1f%%\n",
              o, trt, con, 100 * (trt - con) / con))
}
