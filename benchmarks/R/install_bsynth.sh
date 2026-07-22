#!/usr/bin/env bash
# Install the authors' bsynth R package (Martinez & Vives-i-Bastida 2024,
# github.com/ignacio82/bsynth) for the mvbbsc_bsynth_ref reference run.
#
# bsynth fits the Bayesian synthetic control in Stan via rstan. rstan and its
# heavy transitive stack (StanHeaders, RcppEigen, BH, Boom-free) build slowly
# from source, so we take the prebuilt rstan from apt (Ubuntu universe) and
# compile only the two GitHub-only leaves: vizdraws (a Suggests bsynth loads at
# fit time) and bsynth itself. Verified on Ubuntu + R 4.3.x in a sandbox where
# CRAN is blocked but apt and `git clone` over HTTPS are open.
#
# COMMIT-PINNED (frozen 2026-07-22) so the cross-check runs the SAME reference
# code every time. To refresh, bump the SHAs and re-pin the expected numbers in
# benchmarks/cases/mvbbsc_germany.py and docs/replications/mvbbsc.rst.
#
#   bsynth     22d960f7496ba57f3d30740097ce6dfaac70d1d5  (2024-06-27, ignacio82/bsynth)
#   vizdraws   1ca4d75fda9f8438b77ba65f47a4dc6e6aff0a3c  (ignacio82/vizdraws)
set -euo pipefail

BSYNTH_SHA=22d960f7496ba57f3d30740097ce6dfaac70d1d5
VIZDRAWS_SHA=1ca4d75fda9f8438b77ba65f47a4dc6e6aff0a3c

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# Prebuilt rstan + the CRAN deps bsynth Imports, from apt (avoids compiling the
# whole Stan toolchain). dplyr/tidyr/ggplot2/scales/glue/tibble/purrr are pulled
# in transitively; list the direct Imports explicitly.
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential gfortran git \
  r-cran-rstan r-cran-rcpp r-cran-dplyr r-cran-tidyr r-cran-ggplot2 \
  r-cran-scales r-cran-glue r-cran-tibble r-cran-purrr r-cran-magrittr \
  r-cran-r6 r-cran-jsonlite r-cran-htmlwidgets r-cran-rlang

clone_install() {  # $1 = owner/repo, $2 = pinned SHA
  local repo="$1" sha="$2" name
  name="$(basename "$repo")"
  git clone --quiet "https://github.com/$repo.git" "$name"
  ( cd "$name" && git checkout --quiet "$sha" )
  R CMD INSTALL --no-help --no-docs "$name"
}

# vizdraws is a bsynth Suggests it loads when building plot data; install it
# first so the bsynth install and the reference run both find it.
clone_install ignacio82/vizdraws "$VIZDRAWS_SHA"
clone_install ignacio82/bsynth   "$BSYNTH_SHA"

Rscript -e 'suppressMessages(library(bsynth)); cat("bsynth", as.character(packageVersion("bsynth")), "OK\n")'
