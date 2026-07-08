# Live reference for benchmarks/cases/propsc_spain.py.
#
# Runs the authors' R package propsdid (common-weights SDID, porp_dat=TRUE) on
# the Spain "Just Transition" panel and emits one prefixed line per party for
# the ATT and its jackknife SE, so the Python case can diff mlsynth's PROPSC
# .fit() against it cell-by-cell. Reproduces Table 2 (common weights) of
# Bogatyrev & Stoetzer (2026).
#
# Usage: Rscript benchmarks/R/propsdid_spain.R [path/to/spain_propsc.csv]
# The propsdid core is source()d from the pinned cache populated by
# benchmarks/R/install_propsdid.sh.

cache <- "benchmarks/reference/.cache/propsdid/R"
for (f in c("solver.R", "utils.R", "synthdid.R", "vcov_multivar.R")) {
  source(file.path(cache, f))
}

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "basedata/spain_propsc.csv"
d <- read.csv(data_path, stringsAsFactors = FALSE)

parties <- c("psoe", "pp", "podem", "cs", "vox", "others")

# Long form: one row per (munid, year, party); NA outcomes -> 0 (as in the paper's
# balanced-panel construction, already applied in spain_propsc.csv).
long <- do.call(rbind, lapply(seq_along(parties), function(k) {
  v <- d[[parties[k]]]; v[is.na(v)] <- 0
  data.frame(unit = d$munid, time = d$year, category = k, value = v,
             treat = d$coalXpost, stringsAsFactors = FALSE)
}))

pa <- panel.array(long, unit = "unit", time = "time", category = "category",
                  outcome = "value", treatment = "treat")

est <- sc_estimate(pa$Y, pa$N0, pa$T0, porp_dat = TRUE, method = "sdid")
se  <- sqrt(vcov.synthdid_estimate_multi(est, method = "jackknife"))

for (k in seq_along(parties)) {
  cat(sprintf("ATT_%s %.10f\n", parties[k], as.numeric(est)[k]))
  cat(sprintf("SE_%s %.10f\n", parties[k], as.numeric(se)[k]))
}
cat(sprintf("SUM %.3e\n", sum(as.numeric(est))))
