# Reference run for the `nsc_prop99` benchmark case.
#
# Sources Tian's (2023) own NSC implementation -- vendored verbatim at
# benchmarks/R/nsc_tian2023_reference.R -- and runs it on the Abadie-Diamond-
# Hainmueller Proposition 99 panel (basedata/smoking_data.csv) at the Table-2
# selected penalty (a = 0.3, b = 0.7). With a and b fixed the estimator is
# deterministic: the stochastic leave-one-out CV that selects the penalty is
# bypassed, and the confidence-interval resampling is disabled (CI = FALSE), so
# the donor weights and the effect path are an exact, reproducible function of
# the data. These are the genuine NSC.R outputs the Python case pins against --
# not numbers transcribed from the paper's Table 2.
#
# Run from the repository root:  Rscript benchmarks/reference/nsc_prop99/reference.R
suppressMessages(library(quadprog))
source("benchmarks/R/nsc_tian2023_reference.R")

d <- read.csv("basedata/smoking_data.csv", check.names = FALSE)
d$treat <- as.integer(d[["Proposition 99"]] %in% c("True", "TRUE", "1", TRUE))
d <- d[, c("state", "year", "cigsale", "treat")]

res <- NSC(d, ID = 1, Time = 2, Outcome = 3, Treatment = 4,
           a = 0.3, b = 0.7, CI = FALSE)

# The weight vector is ordered like the non-treated units after NSC()'s
# order(ID, Time) sort -- i.e. the donor states alphabetically, California out.
states <- unique(d[order(d$state, d$year), "state"])
treated_name <- unique(d$state[d$treat == 1])
donors <- states[states != treated_name]
w <- as.numeric(res$weights)

years <- sort(unique(d$year))
ite <- as.numeric(res$ITE)
names(ite) <- as.character(years)
# Proposition 99 took effect in 1989; the post-period average effect.
att_mean_post <- mean(ite[as.character(years[years >= 1989])])

cat("== REFERENCE VALUES ==\n")
for (i in seq_along(donors)) {
  cat(sprintf("weight\t%s\t%.6f\n", donors[i], w[i]))
}
cat(sprintf("ite_1990\t%.6f\n", ite[["1990"]]))
cat(sprintf("ite_1995\t%.6f\n", ite[["1995"]]))
cat(sprintf("ite_2000\t%.6f\n", ite[["2000"]]))
cat(sprintf("att_mean_post\t%.6f\n", att_mean_post))
cat(sprintf("n_donors\t%d\n", length(donors)))
cat("== SESSION INFO ==\n")
print(sessionInfo())
