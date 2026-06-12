#!/usr/bin/env bash
# Install the augsynth reference for benchmarks/cases/geolift_augsynth_ref.py.
#
# GeoLift's GeoLift() is augsynth's ridge Augmented SCM with fixed effects under
# the hood, so the GeoLift cross-check only needs augsynth -- NOT the heavy
# GeoLift -> MarketMatching -> bsts -> Boom chain. Verified on Ubuntu + R 4.3.x
# in a sandbox where CRAN is blocked but apt and GitHub are open: apt for the
# prebuilt majority, compile the non-apt leaves from the GitHub cran mirror.
set -euo pipefail

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran \
  r-cran-dplyr r-cran-tidyr r-cran-magrittr r-cran-ggplot2 r-cran-formula \
  r-cran-rlang r-cran-purrr r-cran-fnn r-cran-rcpp r-cran-r6 \
  r-cran-doparallel r-cran-foreach r-cran-gridextra r-cran-lifecycle \
  r-cran-stringr r-cran-tibble r-cran-rcpparmadillo r-cran-rcppeigen r-cran-bh \
  r-cran-glmnet r-cran-mass r-cran-matrix

inst() {  # compile a CRAN package from the GitHub cran mirror (apt-blocked deps)
  cd /tmp
  curl -sL "https://codeload.github.com/cran/$1/tar.gz/refs/heads/master" -o "$1.tgz"
  tar xzf "$1.tgz"
  R CMD INSTALL --no-docs --no-help "$1-master"
}
inst S7            # newer osqp needs it
inst LiblineaR     # bundles liblinear C++
inst osqp          # the SCM QP solver

cd /tmp
curl -sL "https://codeload.github.com/ebenmichael/augsynth/tar.gz/refs/heads/master" \
  -o augsynth.tgz && tar xzf augsynth.tgz
R CMD INSTALL --no-docs --no-help augsynth-master

Rscript -e 'suppressMessages(library(augsynth)); cat("augsynth OK\n")'
