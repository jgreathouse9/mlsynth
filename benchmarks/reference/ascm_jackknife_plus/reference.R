#!/usr/bin/env Rscript
# Gold for mlsynth's jackknife+ port, from augsynth itself.
#
#   Rscript benchmarks/reference/ascm_jackknife_plus/reference.R
#
# Install the reference first: bash benchmarks/R/install_augsynth.sh
# (augsynth 0.2.0 @ 7a90ea48877fae7925a72cb50bc03a315bc7c042).
#
# This dumps the SEAM, not only the endpoint. `agents_tests.md` is explicit that
# the breakthroughs come from pinning intermediates: the final interval only says
# *that* a port disagrees, the per-drop error and prediction say *where*. So for
# every held-out pre-period we record the refit's prediction path and its
# held-out error, alongside the assembled bounds.
#
# Both settings of `conservative` are generated. The reference has an option, so
# a port validated under only one setting is validated under neither -- the two
# branches assemble the bounds from different rows of the jackknife distribution.

suppressMessages({
  library(augsynth)
  library(dplyr)
})

OUT <- file.path("benchmarks", "reference", "ascm_jackknife_plus")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

data(kansas)
form <- lngdpcapita ~ treated

# Ridge ASCM, the specification Song et al. and the augsynth README both use.
asyn <- augsynth(form, fips, year_qtr, kansas, progfunc = "Ridge", scm = TRUE)

# --- Sanity: the fit itself must match the README before its inference matters.
cat(sprintf("ridge ASCM average ATT = %.8f  (README: -0.04006291)\n",
            summary(asyn)$average_att$Estimate))

# --- Seam 1: the assembled jackknife+ output, both settings -------------------
for (cons in c(FALSE, TRUE)) {
  jk <- augsynth:::time_jackknife_plus(asyn, alpha = 0.05, conservative = cons)
  n <- length(jk$att)
  write.csv(
    data.frame(idx = seq_len(n), att = jk$att,
               heldout_att = jk$heldout_att[seq_len(n)],
               lb = jk$lb, ub = jk$ub, alpha = jk$alpha,
               conservative = cons),
    file.path(OUT, sprintf("gold_jackknife_plus_conservative_%s.csv",
                           tolower(as.character(cons)))),
    row.names = FALSE)
}

# --- Seam 2: the per-drop intermediates --------------------------------------
# Re-walks augsynth's own loop using its own internals, so the dumped numbers are
# the reference's, not a re-derivation of them.
wide_data <- asyn$data
synth_data <- asyn$data$synth_data
Z <- wide_data$Z
t0 <- dim(synth_data$Z0)[1]
t_final <- dim(synth_data$Y0plot)[1]

rows <- list()
for (tdrop in seq_len(t0)) {
  new_data <- augsynth:::drop_time_t(wide_data, Z, tdrop)
  new_ascm <- do.call(augsynth:::fit_augsynth_internal,
    c(list(wide = new_data$wide, synth_data = new_data$synth_data,
           Z = new_data$Z, progfunc = asyn$progfunc, scm = asyn$scm,
           fixedeff = asyn$fixedeff), asyn$extra_args))
  pred <- predict(new_ascm, att = FALSE)
  est <- pred[(t0 + 1):t_final]
  est <- c(est, mean(est))
  err <- c(colMeans(wide_data$X[wide_data$trt == 1, tdrop, drop = FALSE]) -
             pred[t0])
  rows[[length(rows) + 1]] <- data.frame(
    tdrop = tdrop, err = err, pos = seq_along(est), est = est)
}
write.csv(do.call(rbind, rows), file.path(OUT, "gold_perdrop.csv"),
          row.names = FALSE)

# --- Seam 3: the observed treated path the bounds are reflected through -------
# `lb = y1 - ub` and `ub = y1 - lb`; getting y1 wrong flips the interval about
# the wrong centre, which is easy to do and hard to spot end-to-end.
att_full <- predict(asyn, att = TRUE)
y1 <- predict(asyn, att = FALSE) + att_full
write.csv(data.frame(idx = seq_along(y1), y1 = y1, att = att_full,
                     cf = predict(asyn, att = FALSE)),
          file.path(OUT, "gold_treated_path.csv"), row.names = FALSE)

# --- Provenance --------------------------------------------------------------
writeLines(c(
  sprintf("augsynth: %s", as.character(packageVersion("augsynth"))),
  sprintf("R: %s", paste(R.version$major, R.version$minor, sep = ".")),
  sprintf("t0: %d", t0),
  sprintf("t_final: %d", t_final),
  sprintf("n_units: %d", nrow(wide_data$X)),
  "panel: augsynth::kansas, lngdpcapita ~ treated, progfunc=Ridge, scm=TRUE"
), file.path(OUT, "provenance.txt"))

cat("wrote gold to", OUT, "\n")
