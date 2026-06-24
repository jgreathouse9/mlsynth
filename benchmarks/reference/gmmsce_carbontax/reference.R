# Reference run for the `gmmsce_carbontax` benchmark case.
#
# Runs Joseph Patrick Fry's GMM Synthetic Control Estimator -- the genuine
# author implementation GMM-SCE.R from
#   github.com/JosephPatrickFry/OrthogonalizedSyntheticControl
# -- on Andersson (2019)'s Swedish carbon-tax panel, with the same control pool
# and instrument set as the orthsc_carbontax case, and prints the GMM control
# weights and the over-identification J-statistic. These are the genuine outputs
# the Python case pins against (mlsynth's NumPy/cvxpy port reproduces them to
# LowRankQP's interior-point tolerance and attains an objective at least as low),
# not transcribed constants.
#
# Fry's GMMSC() loads several packages it never uses (matlib, kernlab, cointReg,
# corpcor, sandwich); only LowRankQP is exercised. To run the author's algorithm
# verbatim without those unused installs, the script reads GMM-SCE.R and drops
# the unused library() lines before evaluating it -- the GMMSC body is unchanged.
#
# Run from the repository root:  Rscript benchmarks/reference/gmmsce_carbontax/reference.R
suppressMessages({library(foreign); library(LowRankQP)})

COMMIT <- "3b3868404f6cc2deefdd9ceb4dd2911c67f36177"
REPO   <- "JosephPatrickFry/OrthogonalizedSyntheticControl"
CACHE  <- file.path("benchmarks", "reference", ".cache",
                    "OrthogonalizedSyntheticControl")
DATA   <- file.path("basedata", "carbontax_fullsample_data.dta.txt")

# --- ensure Fry's GMM-SCE.R is present (codeload tarball at the pinned commit) ---
if (!file.exists(file.path(CACHE, "GMM-SCE.R"))) {
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

# --- define GMMSC from the author's GMM-SCE.R, dropping only unused imports ----
src <- readLines(file.path(CACHE, "GMM-SCE.R"))
src <- src[!grepl("library\\((matlib|kernlab|cointReg|corpcor|sandwich)\\)", src)]
eval(parse(text = src))

# --- the carbon-tax panel: same controls / instruments / pre-period as the case ---
ct <- read.dta(DATA); ct$country <- as.character(ct$country)
controls <- c("Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
              "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
              "Switzerland", "United States")
instrs <- c("Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
            "United Kingdom")
years <- sort(unique(ct$year)); pre <- years < 1990
wide <- function(u) { s <- ct[ct$country == u, ]; s$CO2_transport_capita[match(years, s$year)] }

Y0 <- matrix(wide("Sweden")[pre], ncol = 1)              # (T0, 1)
YJ <- sapply(controls, function(u) wide(u)[pre])         # (T0, J)
YK <- sapply(instrs,   function(u) wide(u)[pre])         # (T0, K)

res <- GMMSC(Y0 = Y0, YJ = YJ, YK = YK, meanfit = TRUE)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("jstatistic\t%.8f\n", res$Jstatistic))
for (j in seq_along(controls))
  cat(sprintf("weight\t%s\t%.8f\n", controls[j], res$Weights[j]))
cat("== SESSION INFO ==\n")
print(sessionInfo())
