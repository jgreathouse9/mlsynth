# Gold alignments for mlsynth's DTW kernel, from R `dtw`.
# Usage: Rscript benchmarks/reference/dtwsc_dtw/gen_steppattern_gold.R
suppressMessages(library(dtw))
set.seed(20240613)
pats <- c("symmetricP1","asymmetricP2","symmetricP2","asymmetricP1","typeIc","mori2006")
rows <- list()
for (nm in pats) {
  sp <- get(nm, envir = asNamespace("dtw"))
  for (trial in 1:30) {
    n <- sample(2:25, 1); m <- sample(2:25, 1)
    q <- round(rnorm(n), 6); r <- round(rnorm(m), 6)
    for (oe in c(FALSE, TRUE)) {
      res <- tryCatch(
        dtw(q, r, step.pattern = sp, open.end = oe, keep.internals = FALSE),
        error = function(e) NULL)
      if (is.null(res)) {
        rows[[length(rows)+1]] <- data.frame(
          pattern = nm, trial = trial, open_end = oe,
          query = paste(format(q, digits=17), collapse=","),
          reference = paste(format(r, digits=17), collapse=","),
          status = "error", distance = NA, normalized = NA,
          index1 = "", index2 = "", stringsAsFactors = FALSE)
      } else {
        rows[[length(rows)+1]] <- data.frame(
          pattern = nm, trial = trial, open_end = oe,
          query = paste(format(q, digits=17), collapse=","),
          reference = paste(format(r, digits=17), collapse=","),
          status = "ok",
          distance = format(res$distance, digits=17),
          normalized = format(res$normalizedDistance, digits=17),
          index1 = paste(res$index1, collapse=","),
          index2 = paste(res$index2, collapse=","), stringsAsFactors = FALSE)
      }
    }
  }
}
out <- do.call(rbind, rows)
write.csv(out, "benchmarks/reference/dtwsc_dtw/gold_steppatterns.csv", row.names = FALSE)
cat("wrote", nrow(out), "cases\n")

# --- warp() and RefTooShort under each pattern -------------------------------
set.seed(20240614)
RefTooShort <- function(query, reference, step.pattern) {
  a <- tryCatch(dtw(reference, query, step.pattern = step.pattern,
                    open.end = TRUE), error = function(e) NULL)
  if (is.null(a)) return(FALSE)
  if (length(unique(a$index2)) == 1) return(TRUE)
  round(max(warp(a, index.reference = FALSE))) < length(query)
}
rows2 <- list()
for (nm in pats) {
  sp <- get(nm, envir = asNamespace("dtw"))
  for (trial in 1:25) {
    n <- sample(3:20, 1); m <- sample(3:20, 1)
    q <- round(rnorm(n), 6); r <- round(rnorm(m), 6)
    a <- tryCatch(dtw(q, r, step.pattern = sp), error = function(e) NULL)
    wq <- if (is.null(a)) "" else paste(format(warp(a, index.reference = FALSE),
                                               digits = 17), collapse = ",")
    wr <- if (is.null(a)) "" else paste(format(warp(a, index.reference = TRUE),
                                               digits = 17), collapse = ",")
    rows2[[length(rows2)+1]] <- data.frame(
      pattern = nm, trial = trial,
      query = paste(format(q, digits=17), collapse=","),
      reference = paste(format(r, digits=17), collapse=","),
      status = if (is.null(a)) "error" else "ok",
      warp_query = wq, warp_reference = wr,
      ref_too_short = RefTooShort(q, r, sp), stringsAsFactors = FALSE)
  }
}
write.csv(do.call(rbind, rows2),
          "benchmarks/reference/dtwsc_dtw/gold_warp.csv", row.names = FALSE)
cat("wrote warp gold\n")
