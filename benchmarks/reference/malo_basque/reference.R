# Reference run for the `malo_basque` benchmark case.
#
# Runs Malo, Eskelinen, Zhou & Kuosmanen's bilevel synthetic-control solver
# `scm.corner` (SCM-Debug, github.com/Xun90/SCM-Debug -- a constrained outcome
# QP for the donor weights W, then an LP for the predictor weights V) on the
# Abadie-Gardeazabal (2003) Basque study, outcome fit over 1960-1969 (MSCMT's
# times.dep), and prints the bilevel-optimum donor weights in a parseable block.
# These are the genuine solver outputs the Python case pins against, not
# transcribed constants.
#
# scm.corner.R is vendored next to this script (MIT, see NOTICE); the upstream
# fetch is documented in benchmarks/R/install_scmcorner.sh. The step-1 QP is
# solved twice (kernlab::ipop and LowRankQP); we report the LowRankQP corner
# solution (column 2), matching benchmarks/R/scmcorner_basque.R.
#
# Run from the repository root:  Rscript benchmarks/reference/malo_basque/reference.R
suppressMessages({library(kernlab); library(LowRankQP); library(lpSolve)})

args <- commandArgs(trailingOnly = FALSE)
self <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(self)) dirname(normalizePath(self)) else "benchmarks/reference/malo_basque"
source(file.path(here, "scm.corner.R"))

d <- read.csv("basedata/basque_mscmt.csv", check.names = FALSE)
d <- d[d$regionname != "Spain (Espana)", ]
treated <- "Basque Country (Pais Vasco)"
donors <- setdiff(unique(d$regionname), treated)

# Outcome (gdpcap) over the fit window 1960-1969 -- MSCMT's times.dep.
yrs <- 1960:1969
wide <- reshape(d[d$year %in% yrs, c("regionname", "year", "gdpcap")],
                idvar = "year", timevar = "regionname", direction = "wide")
wide <- wide[order(wide$year), ]
colnames(wide) <- sub("^gdpcap\\.", "", colnames(wide))
Y1pre <- as.matrix(wide[[treated]])
Y0pre <- as.matrix(wide[, donors])

# Predictors: AG means over the MSCMT vignette windows (X feeds the step-2 V LP;
# the donor weights W are the step-1 corner QP solution).
predspec <- list(
  c("school.illit", 1964, 1969), c("school.prim", 1964, 1969), c("school.med", 1964, 1969),
  c("school.higher", 1964, 1969), c("invest", 1964, 1969), c("gdpcap", 1960, 1969),
  c("sec.agriculture", 1961, 1969), c("sec.energy", 1961, 1969), c("sec.industry", 1961, 1969),
  c("sec.construction", 1961, 1969), c("sec.services.venta", 1961, 1969),
  c("sec.services.nonventa", 1961, 1969), c("popdens", 1969, 1969))
mk <- function(v, a, b) {
  sub <- d[d$year >= as.numeric(a) & d$year <= as.numeric(b), ]
  sapply(c(treated, donors), function(u) mean(sub[sub$regionname == u, v], na.rm = TRUE))
}
P <- t(sapply(predspec, function(s) mk(s[1], s[2], s[3])))
X1 <- as.matrix(P[, treated]); X0 <- P[, donors]

res <- scm.corner(Y1pre, Y0pre, X1, X0)
W <- res$W[, 2]            # LowRankQP corner solution
names(W) <- donors

cat("== REFERENCE VALUES ==\n")
for (donor in c("Madrid (Comunidad De)", "Baleares (Islas)", "Rioja (La)")) {
  cat(sprintf("weight\t%s\t%.6f\n", donor, W[[donor]]))
}
cat("== SESSION INFO ==\n")
print(sessionInfo())
