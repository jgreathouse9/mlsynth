#!/usr/bin/env bash
# Dependency chain for the DTWSC (Cao & Chadefaux 2025) reference run,
# benchmarks/reference/dtwsc_basque/reference.R.
#
# The reference is the authors' R package `dsc`
# (github.com/conflictlab/dsc, MIT, pinned at b1cd241518329ac2bc8cfe21a871a798ac14d74f).
# It is NOT vendored -- this script clones it and installs what it needs.
#
# Verified end to end on Ubuntu noble + R 4.3.3 in the sandbox. Two things differ
# from the augsynth recipe and cost time if rediscovered:
#
#   1. `furrr` will NOT build here (lazy-load failure against the apt-shipped
#      purrr/rlang). It is only needed for dsc(parallel = TRUE), so reference.R
#      source()s the four R/*.R files rather than installing the package. Do not
#      chase the furrr build.
#   2. `dtw` needs `proxy`, and `globals` needs `codetools` -- neither is pulled
#      in automatically, and both failures are reported one level up as a missing
#      `dtw` / `future`.
#
# CRAN itself is proxy-blocked (403); the github.com/cran/<pkg> mirrors are open.
set -euo pipefail

COMMIT=b1cd241518329ac2bc8cfe21a871a798ac14d74f
CACHE="${CACHE:-benchmarks/reference/.cache}"

echo "== apt: R and the packages Ubuntu ships =="
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  r-base-core r-base-dev \
  r-cran-zoo r-cran-matrix r-cran-dplyr r-cran-purrr r-cran-magrittr \
  r-cran-tibble r-cran-rlang r-cran-ggplot2 r-cran-reshape2 r-cran-forecast \
  r-cran-kernlab r-cran-rgenoud r-cran-optimx r-cran-nloptr r-cran-pracma \
  r-cran-quadprog r-cran-lhs r-cran-tseries r-cran-lmtest r-cran-codetools

echo "== source: what apt does not ship (order matters) =="
mkdir -p "$CACHE"
# proxy before dtw; codetools (apt, above) before globals; globals before future.
for pkg in proxy dtw signal Synth listenv parallelly globals future; do
  [ -d "$CACHE/src_$pkg" ] || git clone -q --depth 1 "https://github.com/cran/$pkg.git" "$CACHE/src_$pkg"
  R CMD INSTALL "$CACHE/src_$pkg" >/dev/null 2>&1 && echo "  $pkg ok" || echo "  $pkg FAILED"
done

echo "== reference package (pinned, not vendored) =="
if [ ! -d "$CACHE/dsc_repo" ]; then
  git clone -q "https://github.com/conflictlab/dsc.git" "$CACHE/dsc_repo"
fi
git -C "$CACHE/dsc_repo" checkout -q "$COMMIT"

echo "== python side =="
pip install -q dtw-python

Rscript -e 'for (p in c("dtw","signal","Synth","forecast","zoo","Matrix","dplyr"))
  cat(sprintf("  %-10s %s\n", p, requireNamespace(p, quietly = TRUE)))'
echo
echo "Run:  DSC_REPO=$CACHE/dsc_repo Rscript benchmarks/reference/dtwsc_basque/reference.R"
