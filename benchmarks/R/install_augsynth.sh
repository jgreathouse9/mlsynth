#!/usr/bin/env bash
# Install the augsynth reference for benchmarks/cases/geolift_augsynth_ref.py.
#
# GeoLift's GeoLift() is augsynth's ridge Augmented SCM with fixed effects under
# the hood, so the GeoLift cross-check only needs augsynth -- NOT the heavy
# GeoLift -> MarketMatching -> bsts -> Boom chain. Verified on Ubuntu + R 4.3.x
# in a sandbox where CRAN is blocked but apt and GitHub are open: apt for the
# prebuilt majority, compile the non-apt leaves from the GitHub cran mirror.
#
# COMMIT-PINNED (frozen 2026-06-12) so the bit-for-bit cross-check runs the SAME
# reference code every time -- augsynth's master is active dev, and an unpinned
# tip is exactly the version drift that made GeoLift's vignette ATT (155.556) go
# stale. To refresh the reference, bump the SHAs below and re-pin the expected
# numbers in benchmarks/cases/geolift_augsynth_ref.py.
#
#   augsynth   0.2.0     7a90ea48877fae7925a72cb50bc03a315bc7c042  (ebenmichael/augsynth)
#   osqp       1.0.0     260dc73e1e3d07ccb7dbff85b62eaaf483672394  (cran/osqp)
#   S7         0.2.2     33c8f3212c62cd2ebec79cd61d1315e9acc84128  (cran/S7)
#   LiblineaR  2.10.24   07cca10ee74e2442a8726173bd52360c323ad07e  (cran/LiblineaR)
set -euo pipefail

AUGSYNTH_SHA=7a90ea48877fae7925a72cb50bc03a315bc7c042
OSQP_SHA=260dc73e1e3d07ccb7dbff85b62eaaf483672394
S7_SHA=33c8f3212c62cd2ebec79cd61d1315e9acc84128
LIBLINEAR_SHA=07cca10ee74e2442a8726173bd52360c323ad07e

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran \
  r-cran-dplyr r-cran-tidyr r-cran-magrittr r-cran-ggplot2 r-cran-formula \
  r-cran-rlang r-cran-purrr r-cran-fnn r-cran-rcpp r-cran-r6 \
  r-cran-doparallel r-cran-foreach r-cran-gridextra r-cran-lifecycle \
  r-cran-stringr r-cran-tibble r-cran-rcpparmadillo r-cran-rcppeigen r-cran-bh \
  r-cran-glmnet r-cran-mass r-cran-matrix

# Compile a GitHub repo at a pinned commit:  inst <owner/repo> <sha> <dirslug>
inst() {
  cd /tmp
  curl -sL "https://codeload.github.com/$1/tar.gz/$2" -o "$3.tgz"
  tar xzf "$3.tgz"
  R CMD INSTALL --no-docs --no-help "$(basename "$1")-$2"
}
inst cran/S7        "$S7_SHA"        S7          # newer osqp needs it
inst cran/LiblineaR "$LIBLINEAR_SHA" LiblineaR   # bundles liblinear C++
inst cran/osqp      "$OSQP_SHA"      osqp        # the SCM QP solver
inst ebenmichael/augsynth "$AUGSYNTH_SHA" augsynth

Rscript -e 'suppressMessages(library(augsynth)); cat("augsynth", as.character(packageVersion("augsynth")), "OK\n")'
