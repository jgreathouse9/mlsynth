#!/usr/bin/env Rscript
# Live-fect gold for Xu (2017) Table 2, columns (3) and (4).
#
#   Rscript benchmarks/reference/gsynth_xu_turnout/reference.R
#   (install the reference first: bash benchmarks/R/install_fect.sh)
#
# Reads the SAME vendored panel the Python case reads --
# basedata/xu_edr_turnout.parquet -- through nanoparquet. No CSV detour. That
# removes a class of harness bug this project has hit before: a comparison ran
# 33 donors on the R side and 37 on the Python side because each read its own
# copy of the inputs. Two sides, one file.
#
# The reference is fect's `method = "gsynth"` path, which is what the gsynth
# package itself now dispatches to (gsynth 1.4.0 is a shell over fect).
#
# Three families are emitted, and the grid is the sharp one:
#
#   gold_grid.csv        r = 0..5 x {no covariates, covariates}: the ATT and,
#                        where applicable, the covariate coefficients. Two
#                        matching numbers could be luck; twelve fits could not.
#   gold_path.csv        the ATT at every period since adoption, per cell of
#                        that grid. fect numbers the first treated period 1;
#                        mlsynth numbers it 0, and the Python side shifts.
#   gold_cv.csv          the Algorithm 1 criterion by rank. fect reaches it as
#                        cv.method = "loo"; its DEFAULT (rolling, since v2.3.0)
#                        is a different rule that selects a different rank on
#                        this panel, which is the finding this case pins.
#
# No bootstrap is run here. Column (3)'s standard error comes from a parametric
# bootstrap, so reproducing it to the digit would mean reproducing R's RNG
# stream and not the estimator; the Python side compares against the published
# 2.27 at a tolerance that catches a broken routine.

suppressMessages({
  library(fect)
  library(nanoparquet)
})

PANEL <- "basedata/xu_edr_turnout.parquet"
OUT <- "benchmarks/reference/gsynth_xu_turnout"
R_GRID <- 0:5
SEED <- 02139

stopifnot(file.exists(PANEL))
d <- read_parquet(PANEL)
d$abb <- as.character(d$abb)
d$year <- as.numeric(d$year)
for (col in c("turnout", "policy_edr", "policy_mail_in", "policy_motor")) {
  d[[col]] <- as.numeric(d[[col]])
}
d <- d[order(d$abb, d$year), ]

FORMULA <- list(
  nocov = turnout ~ policy_edr,
  cov   = turnout ~ policy_edr + policy_mail_in + policy_motor
)

rows <- list()
paths <- list()

for (spec in names(FORMULA)) {
  for (r in R_GRID) {
    set.seed(SEED)
    g <- fect(FORMULA[[spec]], data = d, index = c("abb", "year"),
              se = FALSE, method = "gsynth", r = r, CV = FALSE,
              force = "two-way", seed = SEED)

    beta <- if (spec == "cov") paste(sprintf("%.10f", g$beta), collapse = ";") else ""
    rows[[length(rows) + 1]] <- data.frame(
      spec = spec, r = r, att = g$att.avg, beta = beta,
      stringsAsFactors = FALSE)

    # g$time is fect's event-time vector for g$att: the number of periods since
    # onset, with 1 the first treated period. It is NOT names(g$att), which is
    # empty here -- reading the labels off the names silently yields 1..33 and
    # a comparison that merges the wrong rows together.
    stopifnot(length(g$time) == length(g$att))
    paths[[length(paths) + 1]] <- data.frame(
      spec = spec, r = r, time = as.numeric(g$time), att = as.numeric(g$att),
      stringsAsFactors = FALSE)

    cat(sprintf("%-6s r=%d  ATT=%12.8f  %s\n", spec, r, g$att.avg, beta))
  }
}

# Algorithm 1: fect reaches it as cv.method = "loo". The package default is
# cv.method = "rolling", which masks a random tenth of the control units over
# twenty folds and selects a seed-dependent rank on this panel.
cvs <- list()
for (spec in names(FORMULA)) {
  set.seed(SEED)
  g <- fect(FORMULA[[spec]], data = d, index = c("abb", "year"), se = FALSE,
            method = "gsynth", r = c(0, max(R_GRID)), CV = TRUE,
            force = "two-way", cv.method = "loo", seed = SEED)
  tab <- as.data.frame(g$CV.out[, c("r", "MSPE")])
  cvs[[length(cvs) + 1]] <- data.frame(
    spec = spec, r = tab$r, mspe = tab$MSPE, r_cv = g$r.cv,
    stringsAsFactors = FALSE)
  cat(sprintf("CV %-6s r*=%d\n", spec, g$r.cv))
}

write.csv(do.call(rbind, rows), file.path(OUT, "gold_grid.csv"), row.names = FALSE)
write.csv(do.call(rbind, paths), file.path(OUT, "gold_path.csv"), row.names = FALSE)
write.csv(do.call(rbind, cvs), file.path(OUT, "gold_cv.csv"), row.names = FALSE)

writeLines(c(
  paste("fect:", as.character(packageVersion("fect"))),
  "commit: eccc9731ad74d049b22a3e769393c6a6c47b33bd",
  paste("nanoparquet:", as.character(packageVersion("nanoparquet"))),
  paste("R:", paste0(R.version$major, ".", R.version$minor)),
  paste("panel:", PANEL),
  paste("ranks:", paste(range(R_GRID), collapse = "-")),
  "call: fect(method='gsynth', force='two-way', CV=FALSE, se=FALSE)",
  "cv:   fect(method='gsynth', CV=TRUE, cv.method='loo')"
), file.path(OUT, "provenance.txt"))

cat("wrote gold to", OUT, "\n")
