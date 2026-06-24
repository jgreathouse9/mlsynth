# Reference run for the `ascm_kansas` benchmark case.
#
# Runs the augsynth R package (Ben-Michael, Feller & Rothstein 2021) on its own
# Kansas tax-cut study and prints the Augmented SCM "ladder" -- classic SCM,
# ridge ASCM, ridge ASCM with covariates, and the residualized covariate variant
# -- as the average ATT on quarterly log GDP per capita and the pre-fit L2
# imbalance, in a stable parseable format, followed by sessionInfo() for
# provenance. These are the genuine package outputs the Python case pins against.
#
# Run from the repository root:  Rscript benchmarks/reference/ascm_kansas/reference.R
suppressMessages(library(augsynth))

d <- read.csv("basedata/kansas_ascm.csv")
covf <- lngdpcapita ~ treated | lngdpcapita + log(revstatecapita) + log(revlocalcapita) +
        log(avgwklywagecapita) + estabscapita + emplvlcapita

fit <- function(form, progfunc, fixedeff = FALSE) {
  a <- augsynth(form, unit = fips, time = year_qtr, data = d,
                progfunc = progfunc, scm = TRUE, fixedeff = fixedeff)
  c(att = summary(a)$average_att$Estimate, l2 = a$l2_imbalance)
}

specs <- list(
  scm          = fit(lngdpcapita ~ treated, "None"),
  ridge        = fit(lngdpcapita ~ treated, "Ridge"),
  covariate    = fit(covf, "Ridge"),
  residualized = fit(covf, "Ridge", fixedeff = TRUE))

cat("== REFERENCE VALUES ==\n")
for (nm in names(specs)) {
  cat(sprintf("att_%s\t%.6f\n", nm, specs[[nm]]["att"]))
  cat(sprintf("l2_%s\t%.6f\n", nm, specs[[nm]]["l2"]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
