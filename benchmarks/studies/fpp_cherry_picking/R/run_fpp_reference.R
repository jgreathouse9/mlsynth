# Run one of Ferman-Pinto-Possebom's replication scripts and capture every
# dataprep() and synth() return value, so a Python port can be compared at each
# seam (predictor matrices, V, W, losses), not only at the final p-values.
#
# Usage: Rscript --vanilla run_fpp_reference.R <script.R> <workdir> <out.rds>
# <workdir> must already hold the data file the script reads by relative path.
# The FPP scripts begin with rm(list = ls(all = TRUE)), which clears globalenv,
# so everything this driver keeps lives in a function frame or in an
# environment attached to the search path.

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  script <- normalizePath(args[1])
  workdir <- normalizePath(args[2])
  out <- file.path(normalizePath(dirname(args[3])), basename(args[3]))
  suppressMessages(library(Synth))
  store <- attach(NULL, name = "fpp_trace")
  assign("synth_calls", list(), envir = store)
  assign("prep_calls", list(), envir = store)
  assign("t0", Sys.time(), envir = store)
  trace("synth", exit = quote({
    e <- as.environment("fpp_trace")
    rv <- returnValue(NULL)  # NULL when synth() exits by error
    calls <- get("synth_calls", envir = e)
    calls[[length(calls) + 1L]] <- list(
      treat = data.prep.obj$tag$treatment.identifier,
      controls = data.prep.obj$tag$controls.identifier,
      failed = is.null(rv),
      w = as.numeric(rv$solution.w), v = as.numeric(rv$solution.v),
      loss.v = as.numeric(rv$loss.v), loss.w = as.numeric(rv$loss.w))
    assign("synth_calls", calls, envir = e)
    n <- length(calls)
    if (n %% 34L == 0L)
      message(sprintf("synth calls: %d  elapsed %.1f min", n,
                      as.numeric(difftime(Sys.time(), get("t0", envir = e), units = "mins"))))
  }), print = FALSE)
  trace("dataprep", exit = quote({
    e <- as.environment("fpp_trace")
    rv <- returnValue()
    preps <- get("prep_calls", envir = e)
    preps[[length(preps) + 1L]] <- list(
      X1 = rv$X1, X0 = rv$X0, Z1 = rv$Z1, Z0 = rv$Z0,
      Y1plot = rv$Y1plot, Y0plot = rv$Y0plot, tag = rv$tag)
    assign("prep_calls", preps, envir = e)
  }), print = FALSE)
  setwd(workdir)
  pdf(file.path(workdir, "Rplots.pdf"))
  source(script, echo = FALSE)
  dev.off()
  g <- globalenv()
  kept <- c(intersect(c("results", "te", "rmspe_matrix", "p.average"), ls(g)),
            grep("^(spec|p\\.spec)\\.", ls(g), value = TRUE))
  saveRDS(list(objects = mget(kept, envir = g),
               gaps = mget(grep("^gaps[0-9]", ls(g), value = TRUE), envir = g),
               synth_calls = get("synth_calls", envir = store),
               prep_calls = get("prep_calls", envir = store),
               session = list(R = R.version.string,
                              Synth = as.character(packageVersion("Synth")),
                              optimx = as.character(packageVersion("optimx")),
                              kernlab = as.character(packageVersion("kernlab")))),
          out)
  message("saved ", out)
}
main()
