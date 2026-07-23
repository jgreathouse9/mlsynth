# Reference generator for the microsynth_baltimore cross-validation benchmark.
#
# Produces the R `microsynth` package's per-period synthetic-control totals and
# treatment effects on the Baltimore City Intelligence Center (BCIC) evaluation
# of Lawrence, Peterson & March (2026, J. Criminal Justice 102:102572), across
# the four treated police districts (Central, Eastern, Southwestern, Western)
# and the eight headline outcomes (person / property / shooting / all-crime, in
# each of the {total, outdoor} panels).
#
# Two configurations are dumped per (district, panel, outcome):
#   (A) match.out = c(outcome)      -- the FULL pre-period trajectory match, the
#       constraint set mlsynth's MicroSynth(weight_method="panel",
#       outcome_lag_periods=<full pre-window>) uses. This is the apples-to-apples
#       cross-check of the two solvers (R LowRankQP vs mlsynth's ridge-selected
#       max-ESS QP).
#   (B) match.out.min = c(outcome), match.out = FALSE -- the paper's own
#       configuration (aggregate pre-period outcome match), used to reproduce the
#       authors' Appendix Tables A1-A4 and to separate a config difference from a
#       solver difference on the sparse (shooting) outcomes.
#
# microsynth does not install from CRAN in the CI/sandbox network (CRAN-over-
# HTTPS is firewalled). Install route (Ubuntu 24.04, R 4.3.3):
#
#   apt-get install -y r-base r-base-dev \
#     r-cran-survey r-cran-kernlab r-cran-pracma r-cran-mass r-cran-matrix
#   git clone --depth 1 https://github.com/cran/LowRankQP   && R CMD INSTALL LowRankQP
#   git clone --depth 1 https://github.com/cran/microsynth  && R CMD INSTALL microsynth
#   git clone --depth 1 https://github.com/cran/nanoparquet && R CMD INSTALL nanoparquet
#
# (git clone works where curl of codeload.github.com is proxy-gated to the
# session's enabled repo -- the mlsynth-sandbox convention documented in
# install_propsdid.sh.) nanoparquet is a zero-dependency parquet reader, so the
# script reads the SAME parquet panels the benchmark ships -- no CSV detour.
#
# Run:  Rscript benchmarks/R/microsynth_baltimore.R <datadir> <outfile.csv>
#   <datadir> holds BCIC_<District>_<All|Outside> Crime.parquet (default: the
#   shipped basedata/bcic_baltimore).

suppressMessages(library(microsynth))
suppressMessages(library(nanoparquet))
cat("microsynth version:", as.character(packageVersion("microsynth")), "\n")

args <- commandArgs(trailingOnly = TRUE)
datadir <- ifelse(length(args) >= 1, args[1], "basedata/bcic_baltimore")
outfile <- ifelse(length(args) >= 2, args[2], file.path(datadir, "microsynth_baltimore_ref.csv"))
# arg 3: comma-separated configs to run. "A" = full-trajectory match.out (the
# mlsynth-mirror cross-check); "B" = the paper's aggregate match.out.min. Default
# both.
configs <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else c("A", "B")

cov.var <- c("block_size", "total_pop",
             "white_percent", "black_percent", "hisp_percent", "othrace_percent",
             "age15_29_percent", "poverty_percent", "unemploy_percent",
             "med_house_income", "vacant_percent",
             "commercial_percent", "detached_res_percent", "indust_percent",
             "open_percent", "multifamily_percent", "spec_purp_percent",
             "dual_occ_percent", "own_occ_percent", "rent_occ_percent",
             "vacantlot_percent")

# district -> (file stem for the {total, outdoor} panels, pre-period length)
districts <- list(
  Central      = list(pre = 27),
  Eastern      = list(pre = 62),
  Southwestern = list(pre = 42),
  Western      = list(pre = 62)
)
panels   <- c(Total = "All Crime", Outdoor = "Outside Crime")
outcomes <- c("allcrime", "person", "property", "shooting")

# Incremental writer: append each fit's per-period rows to `outfile` as they
# finish, so a long capture is recoverable and monitorable mid-run.
.wrote_header <- FALSE
write_fit <- function(df) {
  write.table(df, outfile, sep = ",", row.names = FALSE,
              col.names = !.wrote_header, append = .wrote_header)
  .wrote_header <<- TRUE
}

run_one <- function(dat, outcome, pre, npost, config) {
  # config "A": full-trajectory match.out; "B": paper's aggregate match.out.min
  common <- list(dat, idvar = "masterid", timevar = "timeperiod", intvar = "intvar",
                 start.pre = 1, end.pre = pre, end.post = pre + npost,
                 result.var = outcome, perm = 0, test = "lower", n.cores = 1,
                 check.feas = FALSE, use.survey = FALSE)
  if (config == "A") {
    m <- do.call(microsynth, c(common, list(match.out = c(outcome), match.covar = cov.var)))
  } else {
    m <- do.call(microsynth, c(common, list(match.out = FALSE, match.out.min = c(outcome),
                                             match.covar = FALSE, match.covar.min = cov.var)))
  }
  ps <- m$Plot.Stats
  # Plot.Stats$Treatment / $Control / $Difference are (outcome x group x time).
  trt  <- as.vector(ps$Treatment)
  ctrl <- as.vector(ps$Control)
  diff <- as.vector(ps$Difference)
  list(trt = trt, ctrl = ctrl, diff = diff)
}

for (dname in names(districts)) {
  pre <- districts[[dname]]$pre
  npost <- pre  # each district has an equal pre/post window in these panels
  for (pkey in names(panels)) {
    fp <- file.path(datadir, sprintf("BCIC_%s_%s.parquet", dname, panels[[pkey]]))
    if (!file.exists(fp)) { cat("MISSING:", fp, "\n"); next }
    dat <- as.data.frame(read_parquet(fp))
    for (oc in outcomes) {
      for (cfg in configs) {
        res <- tryCatch(run_one(dat, oc, pre, npost, cfg), error = function(e) {
          cat("ERROR", dname, pkey, oc, cfg, ":", conditionMessage(e), "\n"); NULL })
        if (is.null(res)) next
        Tn <- length(res$diff)
        write_fit(data.frame(
          district = dname, panel = pkey, outcome = oc, config = cfg,
          period = seq_len(Tn), treat = res$trt, control = res$ctrl,
          difference = res$diff, stringsAsFactors = FALSE))
        cat(sprintf("done %-13s %-7s %-9s cfg=%s (%d periods)\n", dname, pkey, oc, cfg, Tn))
      }
    }
  }
}

cat("wrote reference ->", outfile, "\n")
