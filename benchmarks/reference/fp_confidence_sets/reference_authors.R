# Reference generator for the PUBLISHED Firpo-Possebom confidence set.
#
# This is the authors' own driver (california_beta_testing_2018-08-12.R from the
# journal supplement) with three changes: the working directory and the parallel
# backend are dropped, paths come from the command line, and the sensitivity
# sweep is run after the headline call.
#
# Install:
#   apt-get install -y r-cran-kernlab r-cran-optimx r-cran-rgenoud r-cran-ggplot2
#   git clone --depth 1 https://github.com/j-hai/Synth && R CMD INSTALL Synth
#
# Run:
#   Rscript reference_published.R <supplement_dir> <outdir>
# where <supplement_dir> holds function_SCM-CS_v07.R and smoking_dataset.csv
# from the article's supplementary material (neither is redistributed here).

args <- commandArgs(trailingOnly = TRUE)
suppdir <- args[1]
outdir <- ifelse(length(args) >= 2, args[2], ".")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

library("Synth")
source(file.path(suppdir, "function_SCM-CS_v07.R"))
cat("Synth version:", as.character(packageVersion("Synth")), "\n")

# --- the authors' data preparation, verbatim ---------------------------------
smoking <- read.csv(file = file.path(suppdir, "smoking_dataset.csv"))
smoking <- smoking[order(smoking$year, smoking$state), ]
stateid <- as.numeric(rep(1:39, 31))
smoking <- cbind(smoking, stateid)
smoking$state <- rep(levels(as.factor(smoking$state)), 31)
californiaid <- 3

# --- the authors' placebo loop, serialised ------------------------------------
ycache <- file.path(outdir, "Ymat_authors.csv")
wcache <- file.path(outdir, "weightsmat_authors.csv")
results <- NULL
for (j in if (file.exists(ycache) && file.exists(wcache)) c() else 1:39) {
  controlunits <- setdiff(1:39, j)
  dataprep.out <- dataprep(
    foo = smoking,
    predictors = c("lnincome", "beer", "age15to24", "retprice"),
    predictors.op = "mean",
    time.predictors.prior = seq(from = 1980, to = 1988, by = 1),
    special.predictors = list(
      list("cigsale", seq(from = 1975, to = 1975, by = 1), "mean"),
      list("cigsale", seq(from = 1980, to = 1980, by = 1), "mean"),
      list("cigsale", seq(from = 1988, to = 1988, by = 1), "mean")),
    dependent = "cigsale",
    unit.variable = "stateid",
    unit.names.variable = "state",
    time.variable = "year",
    treatment.identifier = j,
    controls.identifier = controlunits,
    time.optimize.ssr = seq(from = 1970, to = 1988, by = 1),
    time.plot = seq(from = 1970, to = 2000, by = 1))
  synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")
  results <- rbind(results, c(dataprep.out$Y1plot, NA, synth.out$solution.w))
  cat("fitted unit", j, "\n")
}
if (is.null(results)) {
  cat("reusing the cached placebo fits\n")
  Ymat <- as.matrix(read.csv(ycache))
  weightsmat <- as.matrix(read.csv(wcache))
  dimnames(Ymat) <- NULL
  dimnames(weightsmat) <- NULL
} else {
  Ymat <- t(results[, 1:31])
  weightsmat <- t(results[, 33:70])
  write.csv(Ymat, ycache, row.names = FALSE)
  write.csv(weightsmat, wcache, row.names = FALSE)
}

# --- the authors' headline options -------------------------------------------
treated <- californiaid
T0 <- 19
precision <- 30
significance <- 4 / 39
v <- matrix(0, 1, 39)

rows <- NULL
for (type in c("constant", "linear")) {
  b <- SCM.CS(Ymat, weightsmat, treated, T0, 0, v, precision, type,
              significance, FALSE)
  u <- as.vector(b$u); l <- as.vector(b$l); n <- length(u)
  d <- if (type == "linear") (n - T0) else 1
  rows <- rbind(rows, data.frame(kind = type, phi = 0,
                                 lower = l[n] / d, upper = u[n] / d))
  cat(sprintf("%-8s phi=0   [%.15f, %.15f]\n", type, l[n] / d, u[n] / d))
}

# --- the sensitivity sweep, v marking the treated unit ------------------------
v_treated <- matrix(0, 1, 39)
v_treated[1, californiaid] <- 1
for (p in c(0.5, 1.0, 2.0)) {
  b <- tryCatch(SCM.CS(Ymat, weightsmat, treated, T0, p, v_treated, precision,
                       "linear", significance, FALSE),
                error = function(e) NULL)
  if (is.null(b)) {
    lo <- NA; hi <- NA
  } else {
    u <- as.vector(b$u); l <- as.vector(b$l); n <- length(u); d <- n - T0
    lo <- l[n] / d; hi <- u[n] / d
  }
  rows <- rbind(rows, data.frame(kind = "linear", phi = p, lower = lo, upper = hi))
  cat(sprintf("linear   phi=%.1f [%.15f, %.15f]\n", p, lo, hi))
}

write.csv(rows, file.path(outdir, "gold_bounds_authors.csv"), row.names = FALSE)
