# Reference dump for the DTWSC (Cao & Chadefaux 2025) cross-validation.
#
# Runs the authors' R package `dsc` (github.com/conflictlab/dsc) on Synth's
# `basque` panel and dumps (a) the rescaled + Savitzky-Golay-filtered panel that
# feeds TFDTW, (b) every donor's TFDTW internals -- cutoff, weight.a, avg.weight
# and the warped series -- and (c) the Synth fit for both the warped (DSC) and
# unwarped (standard SC) donor sets.
#
# COMMIT-PINNED so the cross-check runs the same reference every time:
#   conflictlab/dsc   main   b1cd241518329ac2bc8cfe21a871a798ac14d74f
#
# The package itself is NOT installed: `furrr` fails to build against the R
# available in this sandbox, and `dsc()` only needs it when parallel = TRUE. The
# four R/*.R files are source()d directly instead, and the body of dsc() is
# inlined here so the per-donor internals can be captured. See
# benchmarks/R/install_dtwsc.sh for the dependency chain.
#
# Outputs (written to the working directory):
#   gold_preproc.csv  id, unit, time, value (filtered), value_raw   -- %.17g
#   gold_tfdtw.csv    "<unit>;<quantity>;<comma-separated values>"  -- %.17g
#   gold_fit.csv      time, value, synth_dsc, synth_sc
#
# Numbers this produces (Basque, 1955-1997, treat 1970, Spain dropped, 16 donors):
#   Standard SC   pre-RMSE 0.0886   ATT -0.6027
#   DSC (R)       pre-RMSE 0.0705   ATT -0.5579   (ATT over 1971-1996; three
#                                                  donors' warped series end one
#                                                  period short, so 1997 is NA)
#
# Values are dumped at %.17g deliberately. At the R default (15 significant
# digits) weight.a appears to disagree with a correct port at 3e-12, which is
# the dump precision and not a real difference.
#
# Set DSC_REPO to the cloned reference package (default ../../../dsc_repo).

DSC_REPO <- Sys.getenv("DSC_REPO", unset = "dsc_repo")

suppressMessages({library(dplyr); library(purrr); library(magrittr); library(tibble)
  library(Matrix); library(zoo); library(signal); library(forecast); library(dtw)
  library(Synth); library(reshape2); library(ggplot2); library(rlang)})
for (f in c("misc","TFDTW","synth","dsc")) source(file.path(DSC_REPO, "R", paste0(f, ".R")))

start.time <- 1955; end.time <- 1997; treat.time <- 1970
dependent <- "Basque Country (Pais Vasco)"
k <- 4; filter.width <- 5; buffer <- 0; n.burn <- 3

data(basque, package = "Synth")
d <- basque; colnames(d)[1:4] <- c("id","unit","time","value")
d$invest_ratio <- d$invest / d$value; d$value_raw <- d$value
d <- d %>% dplyr::filter(unit != "Spain (Espana)")

# --- rescale, exactly as dsc() ---
rs <- d %>% dplyr::filter(time <= treat.time) %>% group_by(unit) %>%
  summarize(value.min = min(value), value.max = max(value)) %>% ungroup()
mean.diff <- mean(rs$value.max - rs$value.min)
rs <- rs %>% mutate(multiplier = mean.diff/(value.max - value.min))
d <- left_join(d, rs, by = "unit") %>%
  mutate(value.bak = value_raw, value_raw = (value_raw - value.min)*multiplier, value = value_raw)
# --- Savitzky-Golay preprocessing ---
d <- preprocessing(d, filter.width)
{ dd <- d %>% select(id, unit, time, value, value_raw); dd$value <- sprintf("%.17g", dd$value); dd$value_raw <- sprintf("%.17g", dd$value_raw); write.csv(dd, "gold_preproc.csv", row.names = FALSE) }

t.treat <- (treat.time - start.time) + 1
yv <- d %>% dplyr::filter(unit == dependent & time <= end.time) %>% select(value_raw, value)
y.raw <- yv$value_raw; y.processed <- yv$value
x.list <- d %>% dplyr::filter(unit != dependent & time <= end.time) %>%
  select(unit, time, value, value_raw) %>% group_by(unit) %>% group_split(.keep = TRUE)

out <- list(); warped <- list()
for (item in x.list) {
  unit <- item[[1,1]]; x.raw <- item[["value_raw"]]
  res <- TFDTW(x = item[["value"]], y = y.processed, k = k, t.treat = t.treat,
               buffer = buffer, norm.method = "t", match.method = "fixed",
               step.pattern1 = dtw::symmetricP1, step.pattern2 = dtw::asymmetricP2,
               n.burn = n.burn, ma = 3, ma.na = "original",
               dist.quant = 1, n.IQR = 3, window.type = "none",
               default.margin = 3, n.q = 1, n.r = 1)
  x.warped <- c(warpWITHweight(x.raw[1:res$cutoff], res$weight.a)[1:t.treat],
                warpWITHweight(x.raw[res$cutoff:length(x.raw)], res$avg.weight)[-1])
  out[[unit]] <- list(cutoff = res$cutoff, weight.a = res$weight.a, avg.weight = res$avg.weight)
  warped[[unit]] <- x.warped[1:length(y.raw)]
  cat(sprintf("%-32s cutoff=%3d  len(weight.a)=%3d  len(avg.weight)=%3d\n",
              unit, res$cutoff, length(res$weight.a), length(res$avg.weight)))
}
# dump per-donor internals
con <- file("gold_tfdtw.csv","w")
for (u in names(out)) {
  writeLines(paste(u, "cutoff", out[[u]]$cutoff, sep=";"), con)
  writeLines(paste(u, "weight.a", paste(sprintf("%.17g", out[[u]]$weight.a), collapse=","), sep=";"), con)
  writeLines(paste(u, "avg.weight", paste(sprintf("%.17g", out[[u]]$avg.weight), collapse=","), sep=";"), con)
  writeLines(paste(u, "warped", paste(sprintf("%.17g", warped[[u]]), collapse=","), sep=";"), con)
}
close(con)

special_preds <- expression(list(
  list(dep.var, 1960:1969, c("mean")), list("invest_ratio", 1964:1969, c("mean")),
  list("popdens", 1969, c("mean")), list("sec.agriculture", 1961:1969, c("mean")),
  list("sec.energy", 1961:1969, c("mean")), list("sec.industry", 1961:1969, c("mean")),
  list("sec.construction", 1961:1969, c("mean")), list("sec.services.venta", 1961:1969, c("mean")),
  list("sec.services.nonventa", 1961:1969, c("mean")), list("school.illit", 1964:1969, c("mean")),
  list("school.prim", 1964:1969, c("mean")), list("school.med", 1964:1969, c("mean")),
  list("school.high", 1964:1969, c("mean")), list("school.post.high", 1964:1969, c("mean"))))

df.synth <- lapply(names(warped), function(u)
    data.frame(time = start.time:(start.time+length(y.raw)-1), unit = u,
               value_warped = warped[[u]], stringsAsFactors = FALSE)) %>%
  bind_rows() %>%
  add_row(time = seq(start.time, length(y.raw)+start.time-1), unit = dependent, value_warped = y.raw)
df.synth <- right_join(d, df.synth, by = c("unit","time"))
dep.id <- df.synth[["id"]][which.max(df.synth$unit == dependent)]

fitq <- function(r) {
  pre <- 1:t.treat; post <- (t.treat+1):length(r$value)
  c(pre.rmse = sqrt(mean((r$value[pre]-r$synthetic[pre])^2)),
    att = mean(r$value[post]-r$synthetic[post]))
}
sink("/dev/null")
r.dsc <- do.synth(df.synth, "value_warped", dep.id, NULL, special_preds, 1955:1969, 1955:1969)
r.sc  <- do.synth(df.synth, "value_raw",    dep.id, NULL, special_preds, 1955:1969, 1955:1969)
sink()
q1 <- fitq(r.dsc); q2 <- fitq(r.sc)
cat(sprintf("\n%-14s %10s %10s\n", "method", "pre-RMSE", "post-ATT"))
cat(sprintf("%-14s %10.4f %10.4f\n", "Standard SC", q2[1], q2[2]))
cat(sprintf("%-14s %10.4f %10.4f\n", "DSC (R gold)", q1[1], q1[2]))
write.csv(data.frame(time = start.time:end.time, value = r.dsc$value,
                     synth_dsc = r.dsc$synthetic, synth_sc = r.sc$synthetic),
          "gold_fit.csv", row.names = FALSE)
