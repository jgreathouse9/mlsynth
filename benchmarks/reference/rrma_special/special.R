# Reference values for benchmarks/studies/rrma_ablation/special.py, from R itself.
#
# qnorm, qchisq, pchisq (both tails) on fixed grids, and cor.test(method =
# "kendall") p-values on input vectors written by capture.py (one TSV per case,
# two columns). Usage: Rscript special.R <dir with kendall_*.tsv>. Prints
# "kind<TAB>values" lines; capture.py parses them.
args <- commandArgs(TRUE)
emit <- function(...) cat(paste(..., sep = "\t"), "\n", sep = "")
f <- function(v) format(v, digits = 17)
for (p in c(1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 0.001, 0.01, 0.025,
            0.05, 0.1, 0.2, 0.3, 0.4, 0.42, 0.43, 0.45, 0.55, 0.6, 0.75, 0.9, 0.95,
            0.975, 0.99, 0.999, 1 - 1e-5, 1 - 1e-10, 1 - 1e-15))
  emit("qnorm", f(p), f(qnorm(p)))
for (df in c(1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 50, 80))
  for (p in c(1e-12, 1e-8, 1e-4, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975,
              0.99, 1 - 1e-6, 1 - 1e-10))
    emit("qchisq", f(p), f(df), f(qchisq(p, df)))
# the arguments cov.rob passes: qchisq(qn / n, p) with qn = floor((n + p + 1) / 2)
for (n in c(10, 15, 20, 30, 45, 59, 60, 90, 119, 120))
  for (df in 1:20) {
    qn <- floor((n + df + 1) / 2)
    if (qn >= df + 1 && qn <= n - 1) emit("qchisq", f(qn / n), f(df), f(qchisq(qn / n, df)))
  }
for (df in c(1, 2, 3, 5, 8, 15, 30, 80))
  for (x in c(1e-6, 0.01, 0.5, 1, 2, 5, 10, 20, 50, 100, 200))
    emit("pchisq", f(x), f(df), f(pchisq(x, df)), f(pchisq(x, df, lower.tail = FALSE)))
files <- sort(list.files(args[1], pattern = "^kendall_.*\\.tsv$", full.names = TRUE))
for (file in files) {
  d <- as.matrix(read.table(file))
  r <- suppressWarnings(cor.test(d[, 1], d[, 2], method = "kendall"))
  emit("kendall", basename(file), f(r$p.value), names(r$statistic))
}
emit("versions", R.version.string)
