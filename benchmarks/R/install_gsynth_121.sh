#!/usr/bin/env bash
# Install gsynth 1.2.1 -- the version Lang et al. (2026) used -- for
# benchmarks/cases/gsynth_av_laws.py.
#
# This is deliberately NOT the current gsynth. As of 1.4.0 the package is a thin
# shell over fect, so installing master would silently give a different codebase
# than the paper ran. The `1.2.1` tag is the last release before that merge.
#
# The two generations agree: at matched rank and force, gsynth 1.2.1 and
# fect 2.4.5 return the same numbers to fect's print precision on this panel
# (pornhub -35.185 both, xvideos 28.403 both, vpn 11.435 both, porn 3.018 both).
# Either can serve as the reference; this one is pinned because it is what the
# paper ran, so a disagreement is about mlsynth and not about a version gap.
#
# COMMIT-PINNED (frozen 2026-08-05):
#
#   gsynth      1.2.1    e4525ad7d01e0cb8edf4adc1d2a403007ff341d5  (xuyiqing/gsynth, tag 1.2.1)
#   lfe         3.1.1    367ab39d53fb3c68ba8a1b746d95728e8116b106  (cran/lfe)
#   nanoparquet 0.5.1    4b1627a63513175950304c9d5365df2977cbbb49  (cran/nanoparquet)
#
# gsynth 1.2.1 imports lfe, which is not in apt and needs xtable and sandwich
# first; both of those are. Install output is not redirected, because a silenced
# `R CMD INSTALL` hides a dependency-order failure behind a zero exit code from
# the surrounding pipeline, and the check at the bottom is a `library()` call
# for the same reason.
set -euo pipefail

GSYNTH_SHA=e4525ad7d01e0cb8edf4adc1d2a403007ff341d5
LFE_SHA=367ab39d53fb3c68ba8a1b746d95728e8116b106
NANOPARQUET_SHA=4b1627a63513175950304c9d5365df2977cbbb49

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential gfortran \
  r-cran-rcpp r-cran-rcpparmadillo r-cran-ggplot2 r-cran-ggally \
  r-cran-doparallel r-cran-foreach r-cran-abind r-cran-mass r-cran-gridextra \
  r-cran-mvtnorm r-cran-future r-cran-dorng r-cran-xtable r-cran-sandwich \
  r-cran-matrix r-cran-formula r-cran-nlme

# Compile a GitHub repo at a pinned commit:  inst <owner/repo> <sha> <dirslug>
#
# Cloned, not fetched as a tarball: the sandbox allows git-over-HTTPS to
# github.com but blocks codeload.github.com, which the archive URLs redirect to.
# The clone is full, not --depth 1, because a shallow clone cannot check out an
# arbitrary SHA.
inst() {
  cd /tmp
  rm -rf "$3"
  rm -rf /usr/local/lib/R/site-library/00LOCK-"$3"
  git clone --quiet "https://github.com/$1" "$3"
  git -C "$3" checkout --quiet "$2"
  R CMD INSTALL --no-docs --no-help "$3"
}
inst cran/lfe        "$LFE_SHA"         lfe           # gsynth 1.2.1 imports it
inst cran/nanoparquet "$NANOPARQUET_SHA" nanoparquet  # reads basedata/*.parquet
inst xuyiqing/gsynth "$GSYNTH_SHA"      gsynth

Rscript -e 'suppressMessages({library(gsynth); library(nanoparquet)}); v <- as.character(packageVersion("gsynth")); if (v != "1.2.1") stop("expected gsynth 1.2.1, got ", v); cat("gsynth", v, "/ nanoparquet", as.character(packageVersion("nanoparquet")), "OK\n")'
