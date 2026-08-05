#!/usr/bin/env Rscript
# gsynth 1.2.1 gold for the Lang et al. (2026) age-verification panel.
#
#   Rscript benchmarks/reference/gsynth_av_laws/reference.R
#   (install the reference first: bash benchmarks/R/install_gsynth_121.sh)
#
# Reads the SAME vendored panel the Python case reads --
# basedata/lang_av_laws.parquet -- through nanoparquet. No CSV detour, so the
# two sides cannot run on different inputs.
#
# gsynth 1.2.1 is the version the paper used, and the last release before the
# package became a shell over fect. The call follows their
# 03_preregistered_hypotheses.R:
#
#     gsynth(hits ~ post_treat, data = ..., index = c("state","int_date"),
#            se = TRUE, seed = 12345, inference = "parametric", force = "none")
#
# with se = FALSE here, because the point estimates are what this case compares
# and the bootstrap would only add an RNG stream neither side can share.
#
# Emitted:
#   gold_grid.csv   one row per (outcome, force, r): the overall ATT and the
#                   mean over the paper's estimand window
#   gold_cv.csv     the Algorithm 1 criterion by rank, per (outcome, force)
#
# The grid is the sharp comparison: four outcomes x four force settings x six
# ranks is 96 fits, against which two matching headline numbers would be luck.
#
# gsynth's relative-time index runs from names(m$att); period 0 is the last
# pre-treatment period and 1 is the first treated one. The paper's ATT is
# cumuEff(period = c(0, 12)) / 13, which is the mean over periods 0..12 in that
# numbering, so `att_paper` below computes exactly that.
#
# The index is also truncated below: it starts at the earliest horizon where
# every treated unit is still observed (-53 here, the first adopter having 54
# pre-periods), so `mae_pre` is a balanced-window average. mlsynth's event study
# reports the full range instead, down to -145 where one late adopter carries
# the average alone, and the Python side restricts to the same block.

suppressMessages({
  library(gsynth)
  library(nanoparquet)
})

PANEL <- "basedata/lang_av_laws.parquet"
OUT <- "benchmarks/reference/gsynth_av_laws"
OUTCOMES <- c("pornhub", "xvideos", "vpn", "porn")
FORCES <- c("none", "unit", "time", "two-way")
R_GRID <- 0:5
SEED <- 12345

stopifnot(file.exists(PANEL))
full <- read_parquet(PANEL)

rows <- list()
cvs <- list()

for (oc in OUTCOMES) {
  d <- full[full$outcome == oc, ]
  d <- data.frame(
    state = as.character(d$state),
    int_date = as.numeric(d$int_date),
    hits = as.numeric(d$hits),
    post_treat = as.numeric(d$post_treat),
    stringsAsFactors = FALSE
  )
  d <- d[order(d$state, d$int_date), ]

  for (f in FORCES) {
    for (r in R_GRID) {
      set.seed(SEED)
      m <- try(gsynth(hits ~ post_treat, data = d, index = c("state", "int_date"),
                      se = FALSE, seed = SEED, force = f, r = r, CV = FALSE),
               silent = TRUE)
      if (inherits(m, "try-error")) {
        cat(sprintf("%-8s %-8s r=%d  ERROR\n", oc, f, r))
        next
      }
      e <- m$att
      tn <- as.numeric(names(e))
      rows[[length(rows) + 1]] <- data.frame(
        outcome = oc, force = f, r = r,
        att = m$att.avg,
        att_paper = mean(e[tn >= 0 & tn <= 12], na.rm = TRUE),
        mae_pre = mean(abs(e[tn < 0]), na.rm = TRUE),
        stringsAsFactors = FALSE)
    }

    # Algorithm 1, which gsynth 1.2.1 runs as its own cross-validation.
    set.seed(SEED)
    g <- try(gsynth(hits ~ post_treat, data = d, index = c("state", "int_date"),
                    se = FALSE, seed = SEED, force = f, r = c(0, max(R_GRID)),
                    CV = TRUE), silent = TRUE)
    if (!inherits(g, "try-error")) {
      tab <- as.data.frame(g$CV.out[, c("r", "MSPE")])
      cvs[[length(cvs) + 1]] <- data.frame(
        outcome = oc, force = f, r = tab$r, mspe = tab$MSPE, r_cv = g$r.cv,
        stringsAsFactors = FALSE)
      cat(sprintf("%-8s %-8s r.cv=%d\n", oc, f, g$r.cv))
    }
  }
}

write.csv(do.call(rbind, rows), file.path(OUT, "gold_grid.csv"), row.names = FALSE)
write.csv(do.call(rbind, cvs), file.path(OUT, "gold_cv.csv"), row.names = FALSE)

writeLines(c(
  paste("gsynth:", as.character(packageVersion("gsynth"))),
  "commit: e4525ad7d01e0cb8edf4adc1d2a403007ff341d5 (tag 1.2.1)",
  paste("nanoparquet:", as.character(packageVersion("nanoparquet"))),
  paste("R:", paste0(R.version$major, ".", R.version$minor)),
  paste("panel:", PANEL),
  paste("outcomes:", paste(OUTCOMES, collapse = ",")),
  paste("forces:", paste(FORCES, collapse = ",")),
  paste("ranks:", paste(range(R_GRID), collapse = "-")),
  "call: gsynth(hits ~ post_treat, se=FALSE, seed=12345, CV=FALSE)",
  "cv:   gsynth(..., CV=TRUE, r=c(0,5))"
), file.path(OUT, "provenance.txt"))

cat("wrote", nrow(do.call(rbind, rows)), "grid rows to", OUT, "\n")
