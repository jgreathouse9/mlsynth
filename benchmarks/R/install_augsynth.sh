#!/usr/bin/env bash
# Install the augsynth reference used by the augsynth-based cross-checks
# (e.g. benchmarks/cases/ppscm_paglayan.py).
#
# These cross-checks need only augsynth's ridge Augmented SCM -- NOT the heavy
# MarketMatching -> bsts -> Boom chain. Verified on Ubuntu + R 4.3.x
# in a sandbox where CRAN is blocked but apt and GitHub are open: apt for the
# prebuilt majority, compile the non-apt leaves from the GitHub cran mirror.
#
# COMMIT-PINNED (frozen 2026-06-12) so the bit-for-bit cross-check runs the SAME
# reference code every time -- augsynth's master is active dev, and an unpinned
# tip is exactly the version drift that makes a pinned vignette ATT go stale.
# To refresh the reference, bump the SHAs below and re-pin the expected numbers
# in the relevant benchmarks/cases/*.py.
#
#   augsynth   0.2.0     7a90ea48877fae7925a72cb50bc03a315bc7c042  (ebenmichael/augsynth)
#   osqp       1.0.0     260dc73e1e3d07ccb7dbff85b62eaaf483672394  (cran/osqp)
#   S7         0.2.2     33c8f3212c62cd2ebec79cd61d1315e9acc84128  (cran/S7)
#   LiblineaR  2.10.24   07cca10ee74e2442a8726173bd52360c323ad07e  (cran/LiblineaR)
#   nanoparquet 0.5.1    4b1627a63513175950304c9d5365df2977cbbb49  (cran/nanoparquet)
set -euo pipefail

AUGSYNTH_SHA=7a90ea48877fae7925a72cb50bc03a315bc7c042
OSQP_SHA=260dc73e1e3d07ccb7dbff85b62eaaf483672394
S7_SHA=33c8f3212c62cd2ebec79cd61d1315e9acc84128
LIBLINEAR_SHA=07cca10ee74e2442a8726173bd52360c323ad07e
# Zero-dependency parquet reader, so the Song reference reads the SAME vendored
# panel the Python case does instead of a committed CSV export of it.
NANOPARQUET_SHA=4b1627a63513175950304c9d5365df2977cbbb49

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran \
  r-cran-dplyr r-cran-tidyr r-cran-magrittr r-cran-ggplot2 r-cran-formula \
  r-cran-rlang r-cran-purrr r-cran-fnn r-cran-rcpp r-cran-r6 \
  r-cran-doparallel r-cran-foreach r-cran-gridextra r-cran-lifecycle \
  r-cran-stringr r-cran-tibble r-cran-rcpparmadillo r-cran-rcppeigen r-cran-bh \
  r-cran-glmnet r-cran-mass r-cran-matrix r-cran-cli

# Compile a GitHub repo at a pinned commit:  inst <owner/repo> <sha> <dirslug>
#
# Fetches by `git clone`, NOT by tarball. The sandbox allows git-over-HTTPS to
# github.com but blocks `codeload.github.com` (403), and
# `github.com/<o>/<r>/archive/<sha>.tar.gz` redirects to codeload, so it is
# blocked too. An earlier revision of this script used curl against codeload and
# failed with "gzip: stdin: not in gzip format" -- that is the 403 JSON body
# being handed to tar, not a corrupt download.
#
# The clone is deliberately full, not --depth 1: a shallow clone cannot check
# out an arbitrary SHA.
inst() {
  cd /tmp
  rm -rf "$3"
  git clone --quiet "https://github.com/$1" "$3"
  git -C "$3" checkout --quiet "$2"
  R CMD INSTALL --no-docs --no-help "$3"
}
inst cran/S7        "$S7_SHA"        S7          # newer osqp needs it
inst cran/LiblineaR "$LIBLINEAR_SHA" LiblineaR   # bundles liblinear C++
inst cran/osqp      "$OSQP_SHA"      osqp        # the SCM QP solver
inst cran/nanoparquet "$NANOPARQUET_SHA" nanoparquet  # reads basedata/*.parquet
inst ebenmichael/augsynth "$AUGSYNTH_SHA" augsynth

Rscript -e 'suppressMessages({library(augsynth); library(nanoparquet)}); cat("augsynth", as.character(packageVersion("augsynth")), "/ nanoparquet", as.character(packageVersion("nanoparquet")), "OK\n")'
