## Reference fits for benchmarks/cases/fgrc_grc_crossval.py
##
## Michio Yamamoto's own `grc` package (https://github.com/michioyamamoto/grc),
## whose OptimGRC_C kernel mlsynth's `fgrc.optim_grc` transcribes. R generates
## the panel with the authors' GRC.Rd DGP and their RNG, draws the ALS start,
## and fits; Python reads the panel and the start and estimates from them, so
## the comparison is not confounded by two random number streams landing in
## two local optima.
##
## Usage:
##   R CMD INSTALL --library=<lib> <clone of michioyamamoto/grc>
##   Rscript benchmarks/R/fgrc_grc_crossval.R <lib> <outdir>

args   <- commandArgs(trailingOnly = TRUE)
libloc <- args[1]
outdir <- args[2]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
suppressMessages(library(grc, lib.loc = libloc))

## The authors' GRC.Rd example: 300 subjects, 5 variables, three clusters of
## 100 planted in variable 1; scale() gives the four noise variables the same
## spread as the one that carries the grouping.
make_panel <- function(seed) {
  set.seed(seed)
  X1 <- rnorm(100, -10, 1); X2 <- rnorm(100, 0, 1); X3 <- rnorm(100, 10, 1)
  scale(cbind(c(X1, X2, X3), matrix(rnorm(300 * 4), 300, 4)))
}

CASES <- list(list(seed = 0, c1 = 1, c2 = 0, k = 3),
              list(seed = 1, c1 = 2, c2 = 0, k = 3),
              list(seed = 2, c1 = 2, c2 = 1, k = 3))

rows <- list()
for (cs in CASES) {
  X   <- make_panel(cs$seed)
  ncomp <- cs$c1 + cs$c2
  A0  <- qr.Q(qr(matrix(rnorm(ncol(X) * ncomp), ncol(X), ncomp)))
  ## GRC's inner k-means draws from the stream, so the fit depends on where
  ## the stream sits when it is called. Left to follow the panel and start
  ## draws, the same case gave 1111.06 in 14 iterations on one run and
  ## 1109.77 in 7 on the next. Reseeding immediately before each fit pins it.
  set.seed(1000 + cs$seed)
  res <- GRC(X, cs$c1, cs$c2, cs$k, N.random = 1, nstart = 100, A.first = A0)
  tag <- sprintf("s%d_c%d_%d_k%d", cs$seed, cs$c1, cs$c2, cs$k)
  write.csv(X,  file.path(outdir, sprintf("%s_X.csv",  tag)), row.names = FALSE)
  write.csv(A0, file.path(outdir, sprintf("%s_A0.csv", tag)), row.names = FALSE)
  write.csv(res$A, file.path(outdir, sprintf("%s_A.csv", tag)), row.names = FALSE)
  write.csv(data.frame(cluster = res$cluster),
            file.path(outdir, sprintf("%s_cl.csv", tag)), row.names = FALSE)
  rows[[length(rows) + 1]] <- data.frame(
    tag = tag, seed = cs$seed, c1 = cs$c1, c2 = cs$c2, k = cs$k,
    lossfunc = res$lossfunc, n_ite = res$n.ite)
}
write.csv(do.call(rbind, rows), file.path(outdir, "summary.csv"), row.names = FALSE)
cat(sprintf("wrote %d reference fits to %s\n", length(CASES), outdir))
