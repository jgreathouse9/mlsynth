#!/usr/bin/env Rscript
# Live-augsynth gold for the 30 stratified Song et al. cells.
#
#   Rscript benchmarks/reference/song_ml_ascm/reference.R
#   (install the reference first: bash benchmarks/R/install_augsynth.sh)
#
# Why this exists alongside the authors' published main_result.csv.
#
# mlsynth reproduces a LIVE augsynth 0.2.0 run on these cells to ~1e-7, but the
# published CSV differs from that live run on a substantial minority of cells --
# concentrated in the 2016 heating year, where the gap reaches 1.45 on an ATT of
# ~25 (about 6 percent). The disagreement is not mlsynth against augsynth; it is
# the published artifact against the pinned package. The authors ran whatever
# augsynth was current in 2022-2023 and its ridge cross-validation has changed
# since -- among other things it had, and augsynth still has, the fold and
# standard-error conventions documented in docs/replications/ascm_ridge_cv.rst.
#
# `agents_tests.md` step 0 says to confirm the reference implements the same
# version of the spec before comparing bit-for-bit. Two references are needed
# here because they are two different things: this file is the cross-validation
# target (tight), and main_result.csv is the Path-A target (loose, and the drift
# is the finding rather than a defect).

suppressMessages({library(dplyr); library(augsynth)})
OUT <- file.path("benchmarks", "reference", "song_ml_ascm")

cells <- read.csv(file.path(OUT, "strata_cells.csv"), check.names = FALSE)
cells$date <- as.Date(cells$date)

rows <- list()
for (k in unique(paste(cells$grp, cells$yr, cells$pol, sep = "|"))) {
  p <- strsplit(k, "\\|")[[1]]
  s <- cells %>% filter(grp == p[1], yr == as.numeric(p[2]), pol == p[3])
  a <- augsynth(y ~ treatment, ID, date, s, progfunc = "Ridge", scm = TRUE)
  rows[[length(rows) + 1]] <- data.frame(
    group = p[1], year = as.numeric(p[2]), pollutant = p[3],
    att = summary(a)$average_att$Estimate,
    lambda = a$lambda, l2 = a$l2_imbalance,
    scaled_l2 = a$scaled_l2_imbalance, stringsAsFactors = FALSE)
  cat(sprintf("  %-22s %4s %-8s att=%.8f\n", p[1], p[2], p[3],
              summary(a)$average_att$Estimate))
}
write.csv(do.call(rbind, rows), file.path(OUT, "gold_live_augsynth.csv"),
          row.names = FALSE)
writeLines(c(sprintf("augsynth: %s", as.character(packageVersion("augsynth"))),
             "commit: 7a90ea48877fae7925a72cb50bc03a315bc7c042",
             sprintf("R: %s.%s", R.version$major, R.version$minor),
             sprintf("cells: %d", length(rows))),
           file.path(OUT, "provenance_live.txt"))
cat("wrote live gold to", OUT, "\n")
