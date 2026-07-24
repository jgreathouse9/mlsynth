# Reference generator for the drosc_basque cross-validation benchmark.
#
# Runs the authors' DRoSC() (Koo & Guo 2026, helpers.R) on the Basque Country
# study and dumps the deterministic worst-case point estimand tau(lambda) and the
# lambda=0 donor weights, for cross-validation against mlsynth's DROSC.
#
# helpers.R is cloned from the authors' repo (github.com/taehyeonkoo/DRoSC) at
# run time. Two patches are applied: `cond <- FALSE` (the repo leaves the
# conditional/unconditional inference flag undefined) and the hardcoded source()
# paths are replaced. Only the point estimand (Inference = FALSE) is used, so no
# random seed is involved.
#
# CRAN is firewalled in CI; install limSolve from the GitHub CRAN mirror:
#   apt-get install -y r-cran-lpsolve r-cran-intervals r-cran-quadprog
#   git clone --depth 1 https://github.com/cran/limSolve && R CMD INSTALL limSolve
#   git clone --depth 1 https://github.com/cran/Synth    && R CMD INSTALL Synth
#
#   Rscript benchmarks/R/drosc_basque.R <basque_jasa.csv> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
data_csv <- if (length(args) >= 1) args[1] else "basedata/basque_jasa.csv"
out_dir  <- if (length(args) >= 2) args[2] else "benchmarks/reference/drosc_basque"

suppressMessages({library(limSolve); library(jsonlite)})

# clone + source the authors' helpers.R (idempotent)
repo <- file.path(tempdir(), "DRoSC")
if (!dir.exists(repo)) system(paste("git clone --depth 1 https://github.com/taehyeonkoo/DRoSC",
                                     shQuote(repo)), ignore.stdout = TRUE, ignore.stderr = TRUE)
source(file.path(repo, "src", "helpers.R"))
cond <- FALSE

d <- read.csv(data_csv)
d <- d[d$regionname != "Spain (Espana)", ]
T0 <- 15; T1 <- 28
Y <- d[d$regionname == "Basque Country (Pais Vasco)", "gdpcap"]
regions <- unique(d[d$regionname != "Basque Country (Pais Vasco)", "regionname"])
X <- matrix(d[d$regionname != "Basque Country (Pais Vasco)", "gdpcap"],
            nrow = T0 + T1, byrow = FALSE)
colnames(X) <- regions
Y0 <- Y[1:T0]; Y1 <- Y[-(1:T0)]; X0 <- X[1:T0, ]; X1 <- X[-(1:T0), ]

SC <- sc(Y0, X0)
res <- list(
  reference = "Koo & Guo (2026) DRoSC helpers.R (limSolve::lsei), Basque, T0=15 T1=28 N=16",
  donor_names = as.character(regions),
  values = list(tau_SC = as.numeric(mean(Y1 - X1 %*% SC$w.hat)))
)
w0 <- NULL
for (l in c(0, 0.015, 0.03, 0.045, 0.06)) {
  f <- DRoSC(Y0, Y1, X0, X1, lambda = l, Inference = FALSE)
  key <- if (l %in% c(0, 0.03, 0.06)) sprintf("tau_lam%.2f", l) else sprintf("tau_lam%g", l)
  res$values[[key]] <- as.numeric(f$tauHat)
  if (l == 0) w0 <- setNames(as.numeric(f$betaHat), regions)
}
for (frag in c("Madrid", "Baleares", "Cataluna", "Asturias")) {
  res$values[[sprintf("w_%s_lam0", frag)]] <- as.numeric(w0[grepl(frag, names(w0))][1])
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(res, auto_unbox = TRUE, digits = 10), file.path(out_dir, "reference.json"))
cat("wrote", file.path(out_dir, "reference.json"), "\n")
for (k in names(res$values)) cat(sprintf("  %-16s %.6f\n", k, res$values[[k]]))
