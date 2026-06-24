# Reference run for the `orthsc_carbontax` benchmark case.
#
# Runs Joseph Patrick Fry's Orthogonalized Synthetic Control R code -- the
# genuine author implementation from
#   github.com/JosephPatrickFry/OrthogonalizedSyntheticControl
# -- on Andersson (2019)'s Swedish carbon-tax panel and prints the orthogonalized
# ATT, the fixed-smoothing t-test p-value, the Sun (2013) smoothing parameter K,
# and the 95% confidence interval. These are the genuine outputs the Python case
# pins against (`mlsynth`'s NumPy/cvxpy port reproduces them to the digit), not
# transcribed constants.
#
# Fry's repo carries no licence, so its R scripts are not vendored; this script
# fetches them on demand at a pinned commit into benchmarks/reference/.cache
# (git clone is proxy-blocked in the sandbox, so the codeload tarball is the
# fallback fetch, the same one benchmarks/reference/_fetch.py uses).  See
# benchmarks/R/install_orthsc.sh for the manual fetch / dependency notes.
#
# Run from the repository root:  Rscript benchmarks/reference/orthsc_carbontax/reference.R
suppressMessages({library(foreign)})

COMMIT <- "3b3868404f6cc2deefdd9ceb4dd2911c67f36177"
REPO   <- "JosephPatrickFry/OrthogonalizedSyntheticControl"
CACHE  <- file.path("benchmarks", "reference", ".cache",
                    "OrthogonalizedSyntheticControl")
DATA   <- file.path("basedata", "carbontax_fullsample_data.dta.txt")

# --- ensure Fry's R scripts are present (codeload tarball at the pinned commit) ---
if (!file.exists(file.path(CACHE, "OrthogonalizedSCE.R"))) {
  dir.create(dirname(CACHE), recursive = TRUE, showWarnings = FALSE)
  tarball <- tempfile(fileext = ".tar.gz")
  url <- sprintf("https://codeload.github.com/%s/tar.gz/%s", REPO, COMMIT)
  if (download.file(url, tarball, quiet = TRUE) != 0)
    stop("could not download Fry OrthogonalizedSyntheticControl tarball")
  exdir <- tempfile(); dir.create(exdir)
  untar(tarball, exdir = exdir)
  top <- list.dirs(exdir, recursive = FALSE)[1]
  dir.create(CACHE, recursive = TRUE, showWarnings = FALSE)
  for (f in list.files(top, full.names = TRUE))
    file.copy(f, file.path(CACHE, basename(f)), overwrite = TRUE)
}

# --- dependencies: corpcor + comprehenr into a local lib if not already present ---
userlib <- Sys.getenv("R_LIBS_USER")
if (nzchar(userlib)) .libPaths(c(userlib, .libPaths()))
for (pkg in c("corpcor", "comprehenr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop(sprintf("R package '%s' is required; see benchmarks/R/install_orthsc.sh", pkg))
}

# --- the carbon-tax panel: same controls / instruments / pre-period as the case ---
ct <- read.dta(DATA); ct$country <- as.character(ct$country)
controls <- c("Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
              "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
              "Switzerland", "United States")
instrs <- c("Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
            "United Kingdom")
years <- sort(unique(ct$year)); pre <- years < 1990; post <- years >= 1990
wide <- function(u) { s <- ct[ct$country == u, ]; s$CO2_transport_capita[match(years, s$year)] }

PreY0  <- matrix(wide("Sweden")[pre],  nrow = 1)
PostY0 <- matrix(wide("Sweden")[post], nrow = 1)
PreYJ  <- t(sapply(controls, function(u) wide(u)[pre]))
PostYJ <- t(sapply(controls, function(u) wide(u)[post]))
Z      <- t(sapply(instrs,   function(u) wide(u)[pre]))

# Fry's OrthoganilzedSCE / EstimateDelta read T0 and T1 from the calling frame
# (the constant-instrument row and the lasso tuning use them before they are
# locally assigned), so define them in the global env as the author's scripts
# expect.  T0 = number of pre-treatment periods, T1 = number of post periods.
T0 <- sum(pre); T1 <- sum(post)

# Fry's functions source() their siblings (RegularizedEstimate.R / SeriesHAC.R)
# by relative path at *call* time, so run with the cache dir as the working
# directory; the data is already loaded above by absolute-from-root path.
old <- getwd(); setwd(CACHE)
source("OrthogonalizedSCE.R"); source("RegularizedEstimate.R"); source("SeriesHAC.R")
res <- OrthoganilzedSCE(PreY0 = PreY0, PreYJ = PreYJ, Z = Z,
                        PostY0 = PostY0, PostYJ = PostYJ,
                        alpha = 0.05, beta0 = 0, includeConstant = TRUE)
setwd(old)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("att\t%.6f\n",         res$beta))
cat(sprintf("p_value\t%.6f\n",     res$pvalue))
cat(sprintf("smoothing_K\t%.6f\n", res$df))
cat(sprintf("ci_lower\t%.6f\n",    res$CI[1]))
cat(sprintf("ci_upper\t%.6f\n",    res$CI[2]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
