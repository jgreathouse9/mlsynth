#!/usr/bin/env bash
# Install the R reference for benchmarks/cases/vanillasc_olympics.py.
#
#   bash benchmarks/R/install_tidysynth_olympics.sh
#
# Yoneoka et al. (2022) ran tidysynth under R 4.0.5. Their script asks for
# `quadopt = "LowRankQP"`, which is not tidysynth's default (`ipop`), so
# LowRankQP is installed here as well -- without it the reference silently
# solves a different inner problem than the paper did.
#
# COMMIT-PINNED (frozen 2026-08-05):
#
#   tidysynth  0.2.0    58eda583ea28645a3531cd769b7852ff504656df  (edunford/tidysynth)
#   Synth      1.1-10   ff03e7fdac238e1d3303d48c1a077cc7f8766a99  (cran/Synth)
#   LowRankQP  1.0.6    dfa675f0598950548985f97a7b45229f65aa39b5  (cran/LowRankQP)
#
# tidysynth 0.2.0 is not the version the paper ran -- the authors report R 4.0.5
# and do not name a tidysynth version, and 0.2.0 does not reproduce their
# committed donor weights (see the case docstring). It is pinned as the current
# reference, and the drift from the authors' own artifact is a recorded row.
#
# Cloned, not fetched as a tarball: the sandbox allows git-over-HTTPS to
# github.com but blocks codeload.github.com, which the archive URLs redirect to.
# CRAN itself is unreachable from some sandboxes, so the CRAN packages come from
# the read-only github.com/cran mirrors, which carry identical sources.
#
# Install output is not redirected, because a silenced `R CMD INSTALL` hides a
# dependency-order failure behind a zero exit code from the surrounding
# pipeline, and the check at the bottom is a `library()` call for the same
# reason.
set -euo pipefail

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential gfortran \
  r-cran-dplyr r-cran-tidyr r-cran-purrr r-cran-rlang r-cran-magrittr \
  r-cran-tibble r-cran-ggplot2 r-cran-stringr r-cran-readr r-cran-tidyselect \
  r-cran-kernlab r-cran-optimx r-cran-rgenoud r-cran-robustbase \
  r-cran-matrix r-cran-nlme

# Compile a GitHub repo at a pinned ref:  inst <owner/repo> <ref> <dirslug>
#
# The clone is full, not --depth 1, because a shallow clone cannot check out an
# arbitrary SHA.
inst() {
  cd /tmp
  rm -rf "$3"
  rm -rf /usr/local/lib/R/site-library/00LOCK-"$3"
  git clone --quiet "https://github.com/$1" "$3"
  git -C "$3" checkout --quiet "$2"
  echo "[$3] $(git -C "$3" rev-parse HEAD)"
  R CMD INSTALL --no-docs --no-help "$3"
}
inst cran/LowRankQP     dfa675f0598950548985f97a7b45229f65aa39b5 LowRankQP  # the paper's quadopt, not the default
inst cran/Synth         ff03e7fdac238e1d3303d48c1a077cc7f8766a99 Synth      # tidysynth imports it
inst edunford/tidysynth 58eda583ea28645a3531cd769b7852ff504656df tidysynth

Rscript -e 'suppressMessages({library(tidysynth); library(LowRankQP)}); cat("tidysynth", as.character(packageVersion("tidysynth")), "/ LowRankQP", as.character(packageVersion("LowRankQP")), "OK\n")'
