#!/usr/bin/env bash
# Install the GeoLift reference for benchmarks/cases/geolift_marketselection_ref.py.
#
# GeoLiftMarketSelection's power simulation is augsynth under the hood, but the
# GeoLift package hard-imports the MarketMatching -> CausalImpact -> bsts -> Boom
# chain plus gsynth (its NAMESPACE does import(gsynth) and uses MarketMatching),
# so the whole package must be installed to call it. Verified on Ubuntu + R 4.3.x
# where CRAN is blocked but apt and GitHub are open: apt for the prebuilt
# majority, compile the non-apt leaves from the GitHub cran mirror (no CRAN call).
#
# Builds on install_augsynth.sh (augsynth is a GeoLift import). Every source
# package is pinned to a CRAN version tag / commit so the cross-check runs the
# SAME reference every time. The latest Boom requires R >= 4.5, so the chain is
# frozen to the last R-4.3-compatible release set:
#
#   Boom           0.9.13     bsts           0.9.10    BoomSpikeSlab  1.2.6
#   CausalImpact   1.4.1      MarketMatching 1.2.1     lfe            3.1.1
#   gsynth         1.2.1      dtw            1.23-3     panelView      1.3.1
#   directlabels   2026.4.23
#   GeoLift        2.7.5  db34ea4299ff0e28515ebac502b78c076c93c905  (facebookincubator/GeoLift)
set -euo pipefail
export MAKEFLAGS="-j$(nproc)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/install_augsynth.sh"   # augsynth + its deps (GeoLift imports augsynth)

DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-cran-zoo r-cran-xts r-cran-assertthat r-cran-proxy r-cran-quadprog \
  r-cran-xtable r-cran-sandwich r-cran-ggally r-cran-future r-cran-dorng \
  r-cran-iterators r-cran-abind r-cran-mvtnorm r-cran-reshape2 r-cran-scales \
  r-cran-knitr r-cran-progress r-cran-tibble r-cran-data.table r-cran-bh

# inst <owner/repo> <ref> -- compile a GitHub tarball at a pinned tag/commit.
inst() { cd /tmp; curl -sL "https://codeload.github.com/$1/tar.gz/$2" -o pkg.tgz
         tar xzf pkg.tgz; R CMD INSTALL --no-docs --no-help "$(basename "$1")-${2##*/}"; }

inst cran/dtw            1.23-3
inst cran/directlabels   2026.4.23
inst cran/panelView      1.3.1
inst cran/Boom           0.9.13      # the big C++ compile
inst cran/BoomSpikeSlab  1.2.6
inst cran/bsts           0.9.10
inst cran/CausalImpact   1.4.1
inst cran/lfe            3.1.1
inst cran/gsynth         1.2.1
inst cran/MarketMatching 1.2.1
inst facebookincubator/GeoLift db34ea4299ff0e28515ebac502b78c076c93c905

Rscript -e 'suppressMessages(library(GeoLift)); cat("GeoLift", as.character(packageVersion("GeoLift")), "OK\n")'
