#!/usr/bin/env bash
# Obtain Joseph Patrick Fry's Orthogonalized Synthetic Control R code for the
# orthsc_carbontax reference run (benchmarks/reference/orthsc_carbontax/reference.R).
#
# Fry's repo (github.com/JosephPatrickFry/OrthogonalizedSyntheticControl) is a set
# of loose R scripts -- OrthogonalizedSCE.R (the orthogonalized ATT), plus
# RegularizedEstimate.R (the delta/eta lasso weights) and SeriesHAC.R (the Sun
# 2013 fixed-smoothing variance) that it source()s by relative path. It carries
# NO LICENCE, so the scripts are not vendored into mlsynth; reference.R fetches
# them on demand at a pinned commit into benchmarks/reference/.cache (gitignored).
# This script documents that fetch and installs the two missing R dependencies.
#
# COMMIT-PINNED so the cross-check runs the SAME reference every time. To refresh,
# bump COMMIT here and in reference.R, regenerate the bundle, and re-pin EXPECTED
# in benchmarks/cases/orthsc_carbontax.py.
#
#   OrthogonalizedSyntheticControl   main   3b3868404f6cc2deefdd9ceb4dd2911c67f36177
#
# git clone is proxy-blocked in the sandbox (HTTP 403); the codeload tarball of
# the same pinned commit is the reliable fetch (the fallback _fetch.py uses).
set -euo pipefail

COMMIT=3b3868404f6cc2deefdd9ceb4dd2911c67f36177
REPO=JosephPatrickFry/OrthogonalizedSyntheticControl
CACHE="benchmarks/reference/.cache/OrthogonalizedSyntheticControl"

# 1. R dependencies. limSolve (lasso weight programs) and pracma are usually
#    present; corpcor (pseudoinverse in the smoothing-parameter search) and
#    comprehenr are not on CRAN-free sandboxes, so build them from the CRAN
#    GitHub mirror into the user library. reference.R prepends $R_LIBS_USER.
export R_LIBS_USER="${R_LIBS_USER:-$HOME/.R/libs}"
mkdir -p "$R_LIBS_USER"
for pkg in limSolve pracma corpcor comprehenr; do
  if ! Rscript -e "quit(status = !requireNamespace('$pkg', quietly = TRUE))"; then
    curl -sL "https://codeload.github.com/cran/${pkg}/tar.gz/refs/heads/master" -o "${pkg}.tar.gz"
    tar xzf "${pkg}.tar.gz"
    R CMD INSTALL -l "$R_LIBS_USER" "${pkg}-master"
  fi
done

# 2. Fetch Fry's scripts at the pinned commit (reference.R does this itself; this
#    mirrors it so the cache can be warmed manually).
if [ ! -f "${CACHE}/OrthogonalizedSCE.R" ]; then
  mkdir -p "$(dirname "$CACHE")"
  curl -sL "https://codeload.github.com/${REPO}/tar.gz/${COMMIT}" -o orthsc.tar.gz
  tar xzf orthsc.tar.gz
  mv "OrthogonalizedSyntheticControl-${COMMIT}" "$CACHE"
fi

Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); for (p in c("limSolve","pracma","corpcor","comprehenr")) stopifnot(requireNamespace(p, quietly = TRUE)); cat("ORTHSC reference deps OK\n")'
