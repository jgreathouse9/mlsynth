# Reference generator for the microsynth_seattle cross-validation benchmark.
#
# Produces the R `microsynth` package's per-period treatment effects on the
# canonical Seattle Drug Market Intervention example, single-outcome
# (any_crime) configuration -- so the constraint set matches mlsynth's
# one-outcome MicroSynth(weight_method="panel") exactly. The printed
# Difference[any_crime, 13:16] and weight totals are baked into
# benchmarks/cases/microsynth_seattle.py.
#
# microsynth does not install from CRAN in the CI/sandbox network (CRAN-over-
# HTTPS is firewalled). Install route that works here (Ubuntu 24.04, R 4.3.3):
#
#   apt-get install -y r-base r-base-dev \
#     r-cran-survey r-cran-kernlab r-cran-pracma r-cran-mass
#   git clone --depth 1 https://github.com/cran/LowRankQP && R CMD INSTALL LowRankQP
#   git clone --depth 1 https://github.com/cran/microsynth && R CMD INSTALL microsynth
#
# Run:  Rscript benchmarks/R/microsynth_seattle.R

suppressMessages(library(microsynth))
cat("microsynth version:", as.character(packageVersion("microsynth")), "\n")
data(seattledmi)

cov.var <- c("TotalPop", "BLACK", "HISPANIC", "Males_1521", "HOUSEHOLDS",
             "FAMILYHOUS", "FEMALE_HOU", "RENTER_HOU", "VACANT_HOU")

set.seed(99199)
sea <- microsynth(
  seattledmi,
  idvar = "ID", timevar = "time", intvar = "Intervention",
  start.pre = 1, end.pre = 12, end.post = 16,
  match.out = c("any_crime"),          # single outcome -> matches mlsynth's setup
  match.covar = cov.var,
  result.var = c("any_crime"),
  perm = 0, test = "lower", n.cores = 1,
  check.feas = FALSE, use.survey = FALSE
)

ps <- sea$Plot.Stats
# Single-outcome Plot.Stats$Difference is a 1 x 1 x 16 array (outcome x group x
# time); flatten to the length-16 per-period effect vector.
diff_any <- as.vector(ps$Difference)               # Treatment - synthetic Control
cat("=== per-period effects (Difference[any_crime, ]) ===\n")
print(round(diff_any, 4))
cat("post effects 13:16:\n"); print(round(diff_any[13:16], 4))
cat("ATT (mean of 13:16):", round(mean(diff_any[13:16]), 4), "\n")

# weight totals: treated units carry weight 1; control weights sum to the
# treated count, so the full vector sums to 2 * n_treated.
id13   <- seattledmi$ID[seattledmi$time == 13]
trt13  <- seattledmi$Intervention[seattledmi$time == 13]
ids    <- seattledmi$ID[seattledmi$time == 1]
w      <- sea$w$Weights
ctrl_w <- w[trt13[match(ids, id13)] == 0]
cat("n treated:", sum(trt13), "\n")
cat("control weight sum (== treated count):", round(sum(ctrl_w), 4), "\n")
cat("control weights nonzero (>1e-6):", sum(ctrl_w > 1e-6), "\n")
