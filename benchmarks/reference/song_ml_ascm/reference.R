#!/usr/bin/env Rscript
# Live-augsynth gold for the 30 stratified Song et al. cells.
#
#   Rscript benchmarks/reference/song_ml_ascm/reference.R
#   (install the references first: bash benchmarks/R/install_augsynth.sh
#    and  git clone --depth 1 https://github.com/cran/nanoparquet &&
#         R CMD INSTALL nanoparquet)
#
# nanoparquet is a zero-dependency parquet reader, so this script reads the SAME
# vendored panel the Python case reads -- basedata/song_ml_ascm_china.parquet --
# and slices it here. No CSV detour, following the convention already used by
# benchmarks/R/microsynth_baltimore.R and benchmarks/reference/scd_cps.
#
# That is not a disk-space point. It removes a class of harness bug: an earlier
# comparison in this project silently ran 33 donors on the R side and 37 on the
# Python side because each read its own copy of the inputs. Two sides, one file,
# and that cannot happen. The window and treatment-date arithmetic below is
# transcribed from the authors' main_result.R and mirrors
# benchmarks/cases/song_ml_ascm.py's WINDOWS / INTERVENTION_MMDD / _cell(); the
# case pins agreement against this gold, so a drift between the two shows up as
# a failing row rather than as a silent difference in what was compared.
#
# Why this exists alongside the authors' published main_result.csv.
#
# mlsynth reproduces a LIVE augsynth 0.2.0 run on these cells to ~1e-6, but the
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

suppressMessages({library(dplyr); library(augsynth); library(nanoparquet)})
OUT <- file.path("benchmarks", "reference", "song_ml_ascm")
PANEL <- file.path("basedata", "song_ml_ascm_china.parquet")

# The authors' design, from main_result.R: heating years run 1 May to 30 April,
# treatment starts 23 October of the starting year.
WINDOWS <- list(
  "2014" = c("2014-05-01", "2015-04-30"), "2015" = c("2015-05-01", "2016-04-30"),
  "2016" = c("2016-05-01", "2017-04-30"), "2017" = c("2017-05-01", "2018-04-30"),
  "2018" = c("2018-05-01", "2019-04-30"), "2019" = c("2019-05-01", "2020-04-30"),
  "2020" = c("2020-05-01", "2021-04-30"), "2021" = c("2021-05-01", "2021-12-31"))
INTERVENTION_MMDD <- "10-23"

GROUPS <- c("2+26 cities", "Other northern cities", "Alternative", "Northern",
            "South mixing", "Southern", "Southern control", "China")
POLLUTANTS <- c("SO2wn", "NO2wn", "PM2.5wn", "PM10wn", "O3_8hwn", "COwn",
                "SO2", "NO2", "PM2.5", "PM10", "O3_8h", "CO", "O3", "O3wn",
                "Ox", "Oxwn")

# The same 30 strata as benchmarks/cases/song_ml_ascm.py's strata(): each
# dimension swept independently, de-duplicated in first-seen order.
strata <- function() {
  cells <- c(lapply(GROUPS, function(g) list(g, 2015, "PM2.5wn")),
             lapply(POLLUTANTS, function(p) list("2+26 cities", 2015, p)),
             lapply(names(WINDOWS), function(y) list("2+26 cities",
                                                    as.numeric(y), "PM2.5wn")))
  keys <- vapply(cells, function(c) paste(c[[1]], c[[2]], c[[3]], sep = "|"), "")
  cells[!duplicated(keys)]
}

panel <- as.data.frame(read_parquet(PANEL))
panel$date <- as.Date(panel$date)
donors <- readLines(file.path(OUT, "donor_pool.txt"))
donors <- donors[nzchar(trimws(donors))]

rows <- list()
for (cell in strata()) {
  grp <- cell[[1]]; yr <- cell[[2]]; pol <- cell[[3]]
  w <- WINDOWS[[as.character(yr)]]
  cut <- as.Date(sprintf("%d-%s", yr, INTERVENTION_MMDD))
  s <- panel %>%
    filter(date >= as.Date(w[1]), date <= as.Date(w[2]),
           ID %in% c(grp, donors)) %>%
    select(ID, date, y = all_of(pol)) %>%
    mutate(treatment = as.integer(date >= cut & ID == grp))

  a <- augsynth(y ~ treatment, ID, date, s, progfunc = "Ridge", scm = TRUE)
  rows[[length(rows) + 1]] <- data.frame(
    group = grp, year = yr, pollutant = pol,
    att = summary(a)$average_att$Estimate,
    lambda = a$lambda, l2 = a$l2_imbalance,
    scaled_l2 = a$scaled_l2_imbalance, stringsAsFactors = FALSE)
  cat(sprintf("  %-22s %4s %-8s att=%.8f\n", grp, yr, pol,
              summary(a)$average_att$Estimate))
}
write.csv(do.call(rbind, rows), file.path(OUT, "gold_live_augsynth.csv"),
          row.names = FALSE)
writeLines(c(sprintf("augsynth: %s", as.character(packageVersion("augsynth"))),
             "commit: 7a90ea48877fae7925a72cb50bc03a315bc7c042",
             sprintf("nanoparquet: %s",
                     as.character(packageVersion("nanoparquet"))),
             sprintf("R: %s.%s", R.version$major, R.version$minor),
             sprintf("panel: %s", PANEL),
             sprintf("cells: %d", length(rows))),
           file.path(OUT, "provenance_live.txt"))
cat("wrote live gold to", OUT, "\n")
