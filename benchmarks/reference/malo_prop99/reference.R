# Reference run for the `malo_prop99` benchmark case.
#
# Runs Malo, Eskelinen, Zhou & Kuosmanen's bilevel synthetic-control solver
# `scm.corner` (SCM-Debug, github.com/Xun90/SCM-Debug -- a constrained outcome
# QP for the donor weights W, then an LP for the predictor weights V) on the
# Abadie-Diamond-Hainmueller (2010) California Proposition 99 panel, outcome fit
# over the pre-period 1970-1988 (treatment takes effect 1989), and prints the
# bilevel-optimum donor weights and the upper-level loss L_V in a parseable
# block. These are the genuine solver outputs the Python case pins against --
# i.e. Malo et al. (2024) Table 1's Prop-99 "Optimum" -- not constants
# transcribed from the paper.
#
# scm.corner.R is vendored in the malo_basque bundle (MIT, see that dir's
# NOTICE); we reuse it here rather than re-fetching -- the upstream fetch is
# documented in benchmarks/R/install_scmcorner.sh. The step-1 QP is solved twice
# (kernlab::ipop and LowRankQP); we report the LowRankQP corner solution
# (column 2), matching benchmarks/reference/malo_basque/reference.R.
#
# Run from the repository root:  Rscript benchmarks/reference/malo_prop99/reference.R
suppressMessages({library(kernlab); library(LowRankQP); library(lpSolve)})

# Reuse the vendored solver from the malo_basque bundle (no re-fetch).
source("benchmarks/reference/malo_basque/scm.corner.R")

d <- read.csv("basedata/augmented_cali_long.csv", check.names = FALSE)
treated <- "California"
donors <- setdiff(unique(d$state), treated)

# Outcome (cigsale) over the pre-period 1970-1988 -- ADH's Prop-99 fit window
# (Proposition 99 takes effect in 1989).
yrs <- 1970:1988
wide <- reshape(d[d$year %in% yrs, c("state", "year", "cigsale")],
                idvar = "year", timevar = "state", direction = "wide")
wide <- wide[order(wide$year), ]
colnames(wide) <- sub("^cigsale\\.", "", colnames(wide))
Y1pre <- as.matrix(wide[[treated]])
Y0pre <- as.matrix(wide[, donors])

# Lagged cigsale levels used as predictors (cig1975/cig1980/cig1988).
for (L in c(1975, 1980, 1988)) {
  m <- d[d$year == L, c("state", "cigsale")]
  d[[paste0("cig", L)]] <- m$cigsale[match(d$state, m$state)]
}

# Predictors: ADH means over the case's covariate windows (X feeds the step-2 V
# LP; the donor weights W are the step-1 corner QP solution).
predspec <- list(
  c("loginc", 1980, 1988), c("p_cig", 1980, 1988), c("pct15-24", 1980, 1988),
  c("pc_beer", 1984, 1988), c("cig1975", 1975, 1975),
  c("cig1980", 1980, 1980), c("cig1988", 1988, 1988))
mk <- function(v, a, b) {
  sub <- d[d$year >= as.numeric(a) & d$year <= as.numeric(b), ]
  sapply(c(treated, donors), function(u) mean(sub[sub$state == u, v], na.rm = TRUE))
}
P <- t(sapply(predspec, function(s) mk(s[1], s[2], s[3])))
X1 <- as.matrix(P[, treated]); X0 <- P[, donors]

res <- scm.corner(Y1pre, Y0pre, X1, X0)
W <- res$W[, 2]            # LowRankQP corner solution
names(W) <- donors
L_V <- res$Lv[2]          # upper-level loss at the corner (LowRankQP column)

cat("== REFERENCE VALUES ==\n")
# The non-zero donors Malo et al. Table 1 reports as the Prop-99 "Optimum".
for (donor in c("Utah", "Montana", "Nevada", "Connecticut",
                "New Hampshire", "Colorado")) {
  cat(sprintf("weight\t%s\t%.6f\n", donor, W[[donor]]))
}
cat(sprintf("L_V\t%.6f\n", L_V))
cat(sprintf("n_positive_donors\t%d\n", sum(W > 1e-3)))
cat("== SESSION INFO ==\n")
print(sessionInfo())
