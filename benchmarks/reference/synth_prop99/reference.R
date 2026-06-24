# Reference run for the `synth_prop99` benchmark case.
#
# Runs the original CRAN `Synth` solver (Abadie-Diamond-Hainmueller) on the
# California Proposition 99 tobacco panel under outcome-only matching -- the
# pre-1989 cigarette-sales path serves as both the predictors (X) and the
# dependent fit (Z) -- and prints the donor weights, the attained pre-period
# sum of squared residuals, and the post-period average treatment effect, in a
# stable, parseable format, followed by sessionInfo() for provenance.
#
# This is the genuine reference implementation the Python case cross-validates
# against; its captured stdout and parsed values are committed alongside it so
# every pinned number traces to an inspectable run.
#
# Run from the repository root:  Rscript benchmarks/reference/synth_prop99/reference.R
suppressMessages(library(Synth))

panel <- read.csv("basedata/california_panel.csv")
int_year <- 1989
years <- sort(unique(panel$year))
T0 <- sum(years < int_year)                       # 1970-1988 = 19 pre periods

wide <- reshape(panel[, c("state", "year", "cigsale")],
                idvar = "state", timevar = "year", direction = "wide")
rownames(wide) <- wide$state
Ymat <- as.matrix(wide[, -1])                      # states x periods
donors <- setdiff(rownames(Ymat), "California")
Y0 <- Ymat[donors, , drop = FALSE]                 # donor outcomes
y1 <- Ymat["California", ]                         # treated outcome

sc <- Synth::synth(
  X1 = matrix(y1[1:T0], ncol = 1), X0 = t(Y0[, 1:T0, drop = FALSE]),
  Z1 = matrix(y1[1:T0], ncol = 1), Z0 = t(Y0[, 1:T0, drop = FALSE]),
  verbose = FALSE)
w <- as.numeric(sc$solution.w)
cf <- as.numeric(t(w) %*% Y0)                      # synthetic California
pre_ssr <- sum((y1[1:T0] - cf[1:T0])^2)
att <- mean((y1 - cf)[(T0 + 1):length(y1)])

cat("== REFERENCE VALUES ==\n")
cat(sprintf("synth_pre_ssr\t%.6f\n", pre_ssr))
cat(sprintf("synth_att\t%.6f\n", att))
ord <- order(-w)
for (i in ord) cat(sprintf("weight\t%s\t%.6f\n", donors[i], w[i]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
