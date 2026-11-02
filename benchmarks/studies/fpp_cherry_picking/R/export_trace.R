# Re-emit run_fpp_reference.R captures as one plain-JSON trace for Python.
#
# Usage: Rscript --vanilla export_trace.R <out.json> <treated_id> <in.rds> [<in.rds> ...]
#
# Several captures (one per specification, in specification order) are
# concatenated; the matrices each run fills one row or column of (results,
# rmspe_matrix) are merged cell by cell. Every synth() call is kept (V, W, both
# losses, treated and control identifiers, whether it failed), because the
# placebo distribution needs every unit's fit. dataprep() returns are kept only
# for the actually treated unit, which is enough to check the predictor
# matrices seam by seam.
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
treated <- as.numeric(args[2])
xs <- lapply(args[-(1:2)], readRDS)

calls <- do.call(c, lapply(seq_along(xs), function(k) lapply(
  xs[[k]]$synth_calls, function(c) c(c, part = k))))
calls <- lapply(calls, function(c) list(
  part = c$part, treat = as.numeric(c$treat), controls = as.numeric(c$controls),
  failed = isTRUE(c$failed), w = c$w, v = c$v, loss_v = c$loss.v, loss_w = c$loss.w))

mat <- function(m) list(values = unname(as.matrix(m)),
                        rows = rownames(m), cols = colnames(m))
preps <- do.call(c, lapply(xs, `[[`, "prep_calls"))
preps <- Filter(function(p) as.numeric(p$tag$treatment.identifier) == treated, preps)
preps <- lapply(preps, function(p) list(
  X1 = mat(p$X1), X0 = mat(p$X0), Z1 = mat(p$Z1), Z0 = mat(p$Z0),
  controls = as.numeric(p$tag$controls.identifier)))

merged <- function(name) {
  ms <- Filter(Negate(is.null), lapply(xs, function(x) x$objects[[name]]))
  if (!length(ms)) return(NULL)
  m <- ms[[1]]
  for (o in ms[-1]) {
    if (is.matrix(m)) { fill <- !is.na(o); m[fill] <- o[fill] } else m <- o
  }
  if (is.matrix(m)) mat(m) else unname(m)
}
names_all <- unique(unlist(lapply(xs, function(x) names(x$objects))))
objects <- setNames(lapply(names_all, merged), names_all)

write_json(list(session = xs[[1]]$session, objects = objects, synth_calls = calls,
                treated_preps = preps),
           out, digits = NA, auto_unbox = TRUE, null = "null", na = "null")
message(sprintf("wrote %s: %d synth calls (%d failed), %d treated preps", out,
                length(calls), sum(vapply(calls, `[[`, logical(1), "failed")),
                length(preps)))
