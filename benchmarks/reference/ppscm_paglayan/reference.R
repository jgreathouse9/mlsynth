# Reference run for the `ppscm_paglayan` benchmark case.
#
# Runs augsynth's partially-pooled multisynth (Ben-Michael, Feller & Rothstein
# 2021) on the Paglayan (2018) public-sector collective-bargaining study and
# records the average event-study effect path (time since treatment 0..10) for
# both the basic and time-cohort aggregations, plus the per-period standard
# errors under augsynth's jackknife (delete-one) and bootstrap (Mammen wild
# multiplier) inference. Output is a stable parseable block followed by
# sessionInfo() for provenance.
#
# The jackknife SE and the point-estimate paths are deterministic; the bootstrap
# SE is a stochastic draw (seeded here) -- mlsynth's bootstrap is a separate
# draw, so those agree only up to Monte-Carlo noise.
#
# Run from the repository root:  Rscript benchmarks/reference/ppscm_paglayan/reference.R
suppressMessages(library(augsynth))

d <- read.csv("basedata/Teachingaugsynth.scv")
d <- d[!d$State %in% c("DC", "WI"), ]
d <- d[d$year >= 1959 & d$year <= 1997, ]
d$cbr <- as.integer(d$year >= ifelse(is.na(d$YearCBrequired), Inf, d$YearCBrequired))

avg_path <- function(m, inf) {
  s <- summary(m, inf_type = inf)
  a <- s$att[s$att$Level == "Average" & s$att$Time %in% 0:10, ]
  a <- a[order(a$Time), ]
  list(est = a$Estimate, se = a$Std.Error)
}

m   <- multisynth(lnppexpend ~ cbr, State, year, data = d)
mtc <- multisynth(lnppexpend ~ cbr, State, year, data = d, time_cohort = TRUE)
jk  <- avg_path(m, "jackknife")
tc  <- avg_path(mtc, "jackknife")
set.seed(1)
bt  <- avg_path(m, "bootstrap")

cat("== REFERENCE VALUES ==\n")
cat(sprintf("nu\t%.6f\n", m$nu))
cat(sprintf("global_l2\t%.6f\n", m$global_l2))
cat(sprintf("att\t%.6f\n", mean(jk$est)))           # overall ATT = post-period path mean
cat(sprintf("tc_nu\t%.6f\n", mtc$nu))
cat(sprintf("tc_att\t%.6f\n", mean(tc$est)))
for (k in 0:10) cat(sprintf("tau_%02d\t%.6f\n", k, jk$est[k + 1]))
for (k in 0:10) cat(sprintf("tau_tc_%02d\t%.6f\n", k, tc$est[k + 1]))
for (k in 0:10) cat(sprintf("jack_se_%02d\t%.6f\n", k, jk$se[k + 1]))
for (k in 0:10) cat(sprintf("boot_se_%02d\t%.6f\n", k, bt$se[k + 1]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
