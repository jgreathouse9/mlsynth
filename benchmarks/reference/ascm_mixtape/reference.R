# Reference run for the `ascm_mixtape` benchmark case.
#
# Runs the augsynth R package (Ben-Michael, Feller & Rothstein 2021) on the two
# Augmented SCM examples in Cunningham's Mixtape teaching material, and prints
# the average ATT, the pre-treatment L2 imbalance, and the jackknife+ interval
# for each, in a stable parseable format, followed by sessionInfo().
#
# The two studies are chosen because they sit at opposite ends of the fit
# spectrum, which is what makes the pair informative:
#
#   Prop 99  -- California cigarette sales, treated 1989. 38 donors and 19
#               pre-periods, and no convex combination of the donors reaches
#               California, so the simplex fit leaves visible imbalance and the
#               ridge augmentation has real work to do.
#   Texas    -- black male prison population, treated 1993. 50 donors and only
#               8 pre-periods, so donors outnumber pre-periods six to one and
#               the simplex fit is near-exact. Ridge augmentation on an
#               already-interpolating fit is the degenerate end of the method,
#               and it is where an implementation that shrinks toward the wrong
#               target will differ from the package without looking wrong.
#
# The panels are read from the vendored CSVs the Python case reads, not from the
# Mixtape .dta URLs, so both sides see byte-identical inputs. `smoking_data.csv`
# is `synth_smoking.dta` with the state labels decoded (California is state 3
# there) and agrees with it to 1.2e-05, which is float32 storage in the .dta.
#
# Values print with %.12g and not %.6f. The two outcomes differ by three orders
# of magnitude, so a fixed number of decimal places carries twelve significant
# digits on the Texas ATT and five on a scaled imbalance of 0.0457 -- and the
# rounding at five digits is then larger than the solver difference the case is
# trying to measure. Significant digits keep the reference's precision uniform
# across quantities whatever their scale.
#
# Run from the repository root:  Rscript benchmarks/reference/ascm_mixtape/reference.R
suppressMessages(library(augsynth))

fit <- function(data, form, unit_col, time_col, tag) {
  a <- augsynth(form, unit = !!rlang::sym(unit_col), time = !!rlang::sym(time_col),
                data = data, progfunc = "ridge", scm = TRUE)
  s <- summary(a, inf_type = "jackknife+")
  cat(sprintf("att_%s\t%.12g\n", tag, s$average_att$Estimate))
  cat(sprintf("l2_%s\t%.12g\n", tag, a$l2_imbalance))
  cat(sprintf("scaled_l2_%s\t%.12g\n", tag, a$scaled_l2_imbalance))
  cat(sprintf("att_lower_%s\t%.12g\n", tag, s$average_att$lower_bound))
  cat(sprintf("att_upper_%s\t%.12g\n", tag, s$average_att$upper_bound))
  # every donor weight, so the case can pin the fit and not only its summary
  w <- a$weights[, 1]
  for (nm in names(w)) cat(sprintf("weight\t%s.%s\t%.12g\n", tag, nm, w[[nm]]))
}

smoking <- read.csv("basedata/smoking_data.csv", check.names = FALSE)
smoking$treated <- as.numeric(smoking$state == "California" & smoking$year >= 1989)

texas <- read.csv("basedata/texas_bmprison.csv")
texas$treated <- as.numeric(texas$state == "Texas" & texas$year >= 1993)

cat("== REFERENCE VALUES ==\n")
fit(smoking, cigsale ~ treated, "state", "year", "prop99")
fit(texas, bmprison ~ treated, "statefip", "year", "texas")
cat("== SESSION INFO ==\n")
print(sessionInfo())
