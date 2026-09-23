# Reference run for the `lpca_kansas` benchmark case.
#
# Runs Feng's own replication code -- `knn.index` / `findknn` / `lpca` /
# `sel.r` from Application/Feng_LPCA_app_suppfuns.R in
# yingjieum/Replication_NonlinearFactorModel_2023, pinned at ca34fba -- on the
# 2012 Kansas tax-cut panel, and records the local-PCA counterfactual it
# produces. These are the author's own outputs, not constants transcribed from
# the paper.
#
# The panel is read from basedata/kansas_taxcut.csv so that R and mlsynth
# consume identical bytes. Feng ships the same data as Application/kansas.rda;
# the run asserts the two agree before using the CSV, so the cross-check cannot
# silently drift onto a different vintage.
#
# The synthetic-control arm of Feng's script is deliberately not run here.
# mlsynth's LPCA implements local PCA; the SC comparison is a separate
# estimator, and the paper's first version reported it with a centring defect
# (see the case docstring).
#
# Install: benchmarks/R/install_feng_lpca.sh
# Run from the repository root:
#   Rscript benchmarks/reference/lpca_kansas/reference.R
suppressMessages({library(Rfast); library(irlba)})

REPL <- Sys.getenv("LPCA_REFERENCE_DIR", "/tmp/feng_lpca")
source(file.path(REPL, "replication", "Application", "Feng_LPCA_app_suppfuns.R"))

# ---------------------------------------------------------------- the data --
csv <- read.csv("basedata/kansas_taxcut.csv")

# Confirm the CSV is Feng's kansas.rda before relying on it.
rda <- file.path(REPL, "replication", "Application", "kansas.rda")
agree <- NA
if (file.exists(rda)) {
  load(rda)
  # Sort both sides on the same key: the rda carries `abb` as well as `state`,
  # and ordering one frame by abbreviation against the other by full name would
  # compare different rows and report a spurious mismatch.
  k <- kansas[, c("state", "lngdpcapita", "year_qtr", "treated")]
  k <- k[order(k$state, k$year_qtr), ]
  c2 <- csv[order(csv$state, csv$year_qtr), ]
  agree <- (nrow(k) == nrow(c2)) &&
    max(abs(k$lngdpcapita - c2$lngdpcapita)) < 1e-10 &&
    all(k$treated == c2$treated) &&
    max(abs(k$year_qtr - c2$year_qtr)) < 1e-12
}

# ------------------------------------------- Feng_LPCA_app.R, lines 18-44 --
x <- as.data.frame(csv[, c("state", "lngdpcapita", "year_qtr")])
kansas.ts <- as.matrix(x[x[, 1] == "Kansas", -1])
kansas.ts[, 1] <- c(NA, diff(kansas.ts[, 1])) * 100
kansas.ts <- kansas.ts[-1, ]

treated.ind <- csv$treated[csv$state == "Kansas"]
T0 <- which.max(treated.ind) - 1

x[csv$treated == 1, 2] <- NA
x <- reshape(x, idvar = "state", timevar = "year_qtr", direction = "wide")
name <- x[, 1]
id <- which(name == "Kansas")
x <- as.matrix(x[, -1])
x <- t(apply(x, 1, diff) * 100)

col.mean <- colMeans(x, na.rm = TRUE)
x <- sweep(x, 2, col.mean)
n <- nrow(x); p <- ncol(x); K <- round(n^(2 / 3)); p1 <- 40; nlam <- 3
x[id, T0:p] <- 0

fullkmat  <- knn.index(A = x[, 1:p1], Kseq = K)
mean.pred <- lpca(A = x[, (p1 + 1):p], index = fullkmat[, id], nlam = nlam,
                  n = n, K = K, i = id)
mean.lpca <- mean.pred[1:(p - p1)] + col.mean[(p1 + 1):p]

post <- (T0 - p1):(p - p1)
att <- mean(kansas.ts[T0:p]) - mean(mean.lpca[post])
count <- sum(mean.lpca[post] - kansas.ts[T0:p, 1] > 0)

# The neighbourhood and the rank Feng's own rule lands on, so the Python case
# pins the intermediate objects and not only the headline.
nbr <- name[fullkmat[, id]]
blk <- x[fullkmat[, id], (p1 + 1):p]
sv  <- irlba(blk, nu = nlam, nv = nlam)$d
rank <- sel.r(sv, K)

cat("== REFERENCE VALUES ==\n")
cat(sprintf("csv_matches_rda\t%d\n", as.integer(isTRUE(agree))))
cat(sprintf("n_units\t%d\n", n))
cat(sprintf("n_periods_differenced\t%d\n", p))
cat(sprintf("K\t%d\n", K))
cat(sprintf("match_periods\t%d\n", p1))
cat(sprintf("neighbourhood_size\t%d\n", length(nbr)))
cat(sprintf("local_rank\t%d\n", rank))
cat(sprintf("lpca_att\t%.10f\n", att))
cat(sprintf("post_periods_observed_below\t%d\n", count))
cat(sprintf("observed_post_mean\t%.10f\n", mean(kansas.ts[T0:p])))
cat(sprintf("counterfactual_post_mean\t%.10f\n", mean(mean.lpca[post])))
cat(sprintf("pre_period_rmse\t%.10f\n",
            sqrt(mean((kansas.ts[(p1 + 1):(T0 - 1), 1] -
                       mean.lpca[1:(T0 - p1 - 1)])^2))))
for (i in seq_along(post)) {
  cat(sprintf("weight\tcf_post_%02d\t%.10f\n", i, mean.lpca[post][i]))
}
for (nm in nbr) cat(sprintf("weight\tneighbour_%s\t%.1f\n", gsub(" ", "_", nm), 1))
cat("== SESSION INFO ==\n")
print(sessionInfo())
