#!/usr/bin/env Rscript
# Live-augsynth gold for Bilgel (2022) Table 3, column 1.
#
#   Rscript benchmarks/reference/bilgel_turkey_lockdown/reference.R
#   (install the reference first: bash benchmarks/R/install_augsynth.sh)
#
# Reads the SAME vendored panel the Python case reads --
# basedata/bilgel_turkey_lockdown.parquet -- through nanoparquet, and slices it
# here. No CSV detour. That removes a class of harness bug this project has hit
# before: a comparison ran 33 donors on the R side and 37 on the Python side
# because each read its own copy of the inputs. Two sides, one file.
#
# The call is transcribed from the author's own augsynth_replication_final.R:
#
#     set.seed(1234); multisynth(retail~lockdown, id, eday, nu=0.5,
#                                data=augsynth_retail_t0c1)
#
# including the unit key. He keys on `id`, the numeric province code, not on the
# province name, and this script does the same so both sides of the comparison
# order units identically.
#
# Emitted:
#   gold_live_augsynth.csv       one row per outcome: the overall ATT and its SE
#   gold_live_trajectory.csv     the average ATT at every event time
#
# The trajectory is the sharper of the two. The overall ATT is six numbers; the
# trajectory is the whole path the estimator produces, and the point estimate is
# a deterministic QP, so it is comparable to solver tolerance and not to the
# bootstrap's.

suppressMessages({
  library(augsynth)
  library(nanoparquet)
})

OUTCOMES <- c("retail", "grocery", "park", "transit", "workplace", "residential")
PANEL <- "basedata/bilgel_turkey_lockdown.parquet"
OUT <- "benchmarks/reference/bilgel_turkey_lockdown"

stopifnot(file.exists(PANEL))
full <- read_parquet(PANEL)

scalars <- list()
paths <- list()

for (oc in OUTCOMES) {
  d <- full[full$outcome == oc, ]
  d <- d[order(d$id, d$eday), ]
  d$id <- as.numeric(d$id)
  d$eday <- as.numeric(d$eday)
  d$lockdown <- as.numeric(d$lockdown)
  d$mobility <- as.numeric(d$mobility)

  set.seed(1234)
  ms <- multisynth(mobility ~ lockdown, id, eday, nu = 0.5, data = d)
  s <- summary(ms)

  # Overall ATT: the Average level at the NA event time.
  overall <- subset(s$att, Level == "Average" & is.na(Time))
  stopifnot(nrow(overall) == 1)

  scalars[[oc]] <- data.frame(
    outcome = oc,
    att = overall$Estimate,
    se = overall$Std.Error,
    n_units = length(unique(d$id)),
    n_treated = sum(tapply(d$lockdown, d$id, max)),
    stringsAsFactors = FALSE
  )

  avg <- subset(s$att, Level == "Average" & !is.na(Time))
  paths[[oc]] <- data.frame(
    outcome = oc,
    time = avg$Time,
    estimate = avg$Estimate,
    stringsAsFactors = FALSE
  )

  cat(sprintf("%-12s att=%.10f se=%.10f  units=%d treated=%d\n",
              oc, overall$Estimate, overall$Std.Error,
              length(unique(d$id)), sum(tapply(d$lockdown, d$id, max))))
}

write.csv(do.call(rbind, scalars), file.path(OUT, "gold_live_augsynth.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, paths), file.path(OUT, "gold_live_trajectory.csv"),
          row.names = FALSE)

writeLines(c(
  paste("augsynth:", as.character(packageVersion("augsynth"))),
  "commit: 7a90ea48877fae7925a72cb50bc03a315bc7c042",
  paste("nanoparquet:", as.character(packageVersion("nanoparquet"))),
  paste("R:", paste0(R.version$major, ".", R.version$minor)),
  paste("panel:", PANEL),
  paste("outcomes:", length(OUTCOMES)),
  "call: multisynth(mobility ~ lockdown, id, eday, nu = 0.5), set.seed(1234)"
), file.path(OUT, "provenance_live.txt"))

cat("wrote gold to", OUT, "\n")
