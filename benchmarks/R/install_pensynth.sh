#!/usr/bin/env bash
# Install the pensynth reference for benchmarks/cases/pensynth_prop99.py.
#
# The cross-check needs two things:
#   1. the authors' solver functions (wsoll1.R / TZero.R) -- cloned on demand at a
#      pinned commit by benchmarks/reference/clone_pensynth.py, NOT installed here;
#   2. LowRankQP, the low-rank QP solver wsoll1 calls.
#
# LowRankQP was archived from CRAN, so it is compiled from the CRAN GitHub mirror
# at a pinned commit. Verified on Ubuntu + R 4.3.x in a sandbox where CRAN is
# blocked but apt and GitHub are open: apt for r-base/r-base-dev, build the leaf
# from source.
#
# COMMIT-PINNED so the byte-for-byte solver cross-check runs the SAME reference
# every time. To refresh, bump the SHAs here (and _COMMIT in clone_pensynth.py)
# and re-pin EXPECTED in benchmarks/cases/pensynth_prop99.py.
#
#   pensynth    master   3f2ad93a96acd558841275d07cd70576c78d451f  (jeremylhour/pensynth)
#   LowRankQP   1.0.5    dfa675f0598950548985f97a7b45229f65aa39b5  (cran/LowRankQP)
set -euo pipefail

LOWRANKQP_SHA=dfa675f0598950548985f97a7b45229f65aa39b5

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential

# Compile LowRankQP from the CRAN GitHub mirror at its pinned commit.
cd /tmp
curl -sL "https://codeload.github.com/cran/LowRankQP/tar.gz/${LOWRANKQP_SHA}" -o LowRankQP.tgz
tar xzf LowRankQP.tgz
R CMD INSTALL "LowRankQP-${LOWRANKQP_SHA}"

Rscript -e 'suppressMessages(library(LowRankQP)); cat("LowRankQP", as.character(packageVersion("LowRankQP")), "OK\n")'
