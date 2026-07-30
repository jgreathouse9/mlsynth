#!/usr/bin/env Rscript
# Gold for mlsynth's ridge lambda cross-validation, from augsynth itself.
#
#   Rscript benchmarks/reference/ascm_ridge_cv/reference.R
#   (install the reference first: bash benchmarks/R/install_augsynth.sh)
#
# Dumps the lambda grid, the CV error curve, its standard error, and the selected
# penalty, for TWO panels chosen to differ in the way that matters:
#
#   kansas  T0 = 89 pre-periods > J = 49 donors. The penalty agrees between
#           implementations here even when the CV loop does not, so this panel
#           alone cannot detect a fold-count or standard-error defect -- it is
#           the negative control.
#   song    T0 = 25 < J = 37, and the final pre-period sits on the seasonal ramp
#           into the heating season, so its held-out error is ~9x the mean of the
#           rest. Any disagreement about whether that period is a CV fold moves
#           the whole curve and changes the selected penalty.
#
# Provenance for the song slice: Song et al. (2023), Environ. Sci. Technol.
# 57:17707-17717, open access (CC-BY 4.0); deweathered PM2.5 for the "2+26
# cities" group and the 37 southern control cities used by their own
# main_result.R, from github.com/songnku/ML-ASCM (RegionAverageAll.xlsx).

suppressMessages({library(dplyr); library(augsynth)})
OUT <- file.path("benchmarks", "reference", "ascm_ridge_cv")

dump_curve <- function(a, slug, note) {
  write.csv(data.frame(lambda = a$lambdas, err = a$lambda_errors,
                       se = a$lambda_errors_se),
            file.path(OUT, sprintf("gold_curve_%s.csv", slug)), row.names = FALSE)
  writeLines(c(sprintf("lambda: %.17g", a$lambda),
               sprintf("n_grid: %d", length(a$lambdas)),
               sprintf("T0: %d", ncol(a$data$X)),
               sprintf("J: %d", sum(a$data$trt == 0)),
               sprintf("note: %s", note)),
             file.path(OUT, sprintf("gold_selected_%s.txt", slug)))
  cat(sprintf("%-6s lambda = %-22.17g T0 = %-4d J = %d\n", slug, a$lambda,
              ncol(a$data$X), sum(a$data$trt == 0)))
}

data(kansas)
dump_curve(augsynth(lngdpcapita ~ treated, fips, year_qtr, kansas,
                    progfunc = "Ridge", scm = TRUE),
           "kansas", "T0 > J; negative control -- penalty agrees regardless")

s <- read.csv(file.path(OUT, "song_pm25_cell.csv"), check.names = FALSE)
s$date <- as.Date(s$date)
s$treatment <- 0
s$treatment[s$date >= as.Date("2015-10-23") & s$ID == "2+26 cities"] <- 1
dump_curve(augsynth(PM2.5wn ~ treatment, ID, date, s,
                    progfunc = "Ridge", scm = TRUE),
           "song", "T0 < J; final pre-period is a ~9x outlier fold")

cat("wrote gold to", OUT, "\n")
