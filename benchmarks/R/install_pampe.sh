#!/usr/bin/env bash
# Install the pampe R package (Vega-Bayo; CRAN "pampe") for the
# pda_hcw_hongkong reference run. pampe is the canonical implementation of the
# panel-data approach of Hsiao, Ching & Wan (2012): leaps::regsubsets
# best-subset search + AICc model-size choice + OLS.
#
# pampe declares a single hard dependency, leaps (Depends: leaps), and only
# Enhances: xtable -- nothing heavy to drop, so unlike install_scinference.sh
# we do not trim DESCRIPTION. leaps ships with most R installs / is on Debian as
# r-cran-leaps; we install it from the codeload tarball if it is missing.
#
# git clone may be proxy-blocked (403); the codeload tarball is the reliable
# fetch (the same fallback benchmarks/reference/_fetch.py uses).
set -euo pipefail
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# dependency: leaps (Furnival-Wilson best-subset Fortran; no further deps)
Rscript -e 'if (!"leaps" %in% rownames(installed.packages())) quit(status=1)' \
  || { curl -sL -o leaps.tar.gz https://codeload.github.com/cran/leaps/tar.gz/refs/heads/master
       tar xzf leaps.tar.gz && R CMD INSTALL leaps-master; }

curl -sL -o pampe.tar.gz https://codeload.github.com/cran/pampe/tar.gz/refs/heads/master
tar xzf pampe.tar.gz
R CMD INSTALL pampe-master
Rscript -e 'suppressMessages(library(pampe)); cat("pampe OK\n")'
