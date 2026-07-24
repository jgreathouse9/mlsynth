#!/usr/bin/env Rscript
# Live reference for the drosc_basque cross-validation.
#
# Runs the AUTHORS' own DRoSC (Koo & Guo 2026) on the Basque study and prints the
# deterministic worst-case point estimand tau(lambda) and the lambda=0 donor
# weights as JSON. It sources the authors' unmodified src/helpers.R directly from
# their repository (github.com/taehyeonkoo/DRoSC), cloned once into a cache dir
# and reused thereafter -- so this is their live code, not a transcription.
#
# Inference=FALSE, so no random seed / cond flag is involved. Solver: limSolve::lsei
# (the same QP family the authors use). The Python case (benchmarks/cases/
# drosc_basque.py) BenchmarkSkipped()s when Rscript / limSolve / the clone are
# unavailable, so a missing toolchain never reds the suite.
#
#   Rscript reference.R <basque_jasa.csv>

suppressMessages({library(limSolve); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
data_csv <- if (length(args) >= 1) args[1] else "basedata/basque_jasa.csv"

cache <- Sys.getenv("DROSC_CACHE", "")
if (!nzchar(cache)) cache <- file.path(tempdir(), "drosc_cache")
repo <- file.path(cache, "DRoSC")
if (!file.exists(file.path(repo, "src", "helpers.R"))) {
  dir.create(cache, showWarnings = FALSE, recursive = TRUE)
  st <- system(paste("git clone --depth 1 https://github.com/taehyeonkoo/DRoSC",
                     shQuote(repo)), ignore.stdout = TRUE, ignore.stderr = TRUE)
  if (st != 0 || !file.exists(file.path(repo, "src", "helpers.R")))
    stop("could not clone github.com/taehyeonkoo/DRoSC")
}
source(file.path(repo, "src", "helpers.R"))   # authors' sc() + DRoSC()
cond <- FALSE                                  # unused for Inference=FALSE

d <- read.csv(data_csv)
d <- d[d$regionname != "Spain (Espana)", ]
T0 <- 15; T1 <- 28
Y <- d[d$regionname == "Basque Country (Pais Vasco)", "gdpcap"]
regions <- unique(d[d$regionname != "Basque Country (Pais Vasco)", "regionname"])
X <- matrix(d[d$regionname != "Basque Country (Pais Vasco)", "gdpcap"],
            nrow = T0 + T1, byrow = FALSE)
Y0 <- Y[1:T0]; Y1 <- Y[-(1:T0)]; X0 <- X[1:T0, ]; X1 <- X[-(1:T0), ]

SC <- sc(Y0, X0)
values <- list(tau_SC = as.numeric(mean(Y1 - X1 %*% SC$w.hat)))
w0 <- NULL
for (l in c(0, 0.015, 0.03, 0.045, 0.06)) {
  f <- DRoSC(Y0, Y1, X0, X1, lambda = l, Inference = FALSE)
  key <- if (l %in% c(0, 0.03, 0.06)) sprintf("tau_lam%.2f", l) else sprintf("tau_lam%g", l)
  values[[key]] <- as.numeric(f$tauHat)
  if (l == 0) w0 <- setNames(as.numeric(f$betaHat), regions)
}
weights <- setNames(as.numeric(w0), regions)
cat(toJSON(list(values = values, weights = as.list(weights)),
           auto_unbox = TRUE, digits = 10))
