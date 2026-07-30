#!/usr/bin/env Rscript
# Gold for mlsynth's empirical quantile convention, against R's stats::quantile.
#
#   Rscript benchmarks/reference/dsc_quantile/reference.R
#
# Two types matter here and they are not the same choice:
#
#   type 1  the inverse of the empirical CDF -- a step function returning actual
#           order statistics. This is the estimator Gunsilius (2023) writes down
#           for DSC, and what mlsynth's empirical_quantile implements via
#           numpy's method="inverted_cdf".
#   type 7  R's default, linear interpolation between order statistics. This is
#           what the reference implementation DiSCos actually uses (qtype = 7).
#
# So the paper and its own reference package differ, and a port can match one or
# the other but not both. Capturing both lets the tests pin which convention
# mlsynth implements AND measure the distance to the other, instead of leaving
# the difference as an unexamined source of disagreement (issue #304).
#
# The sample is deliberately small, odd-length and irregularly spaced so the two
# types visibly diverge; with a large smooth sample they nearly coincide, which
# would make the test pass for the wrong reason.

set.seed(11)
x <- c(0.0, 0.5, 1.25, 2.0, 3.5, 3.5, 7.0, 11.25, 19.0, 42.0, 100.5)
q <- c(0.001, 0.01, 0.05, 0.1, 0.25, 1/3, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99, 0.999)

out <- data.frame(
  q      = q,
  type1  = as.numeric(quantile(x, probs = q, type = 1, names = FALSE)),
  type7  = as.numeric(quantile(x, probs = q, type = 7, names = FALSE))
)
write.csv(out, "benchmarks/reference/dsc_quantile/gold_quantiles.csv", row.names = FALSE)
writeLines(c(sprintf("R: %s.%s", R.version$major, R.version$minor),
             sprintf("sample: %s", paste(x, collapse = ",")),
             sprintf("n: %d", length(x))),
           "benchmarks/reference/dsc_quantile/provenance.txt")
cat("wrote gold for", length(q), "quantiles\n")
print(out)
