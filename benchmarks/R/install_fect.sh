#!/usr/bin/env bash
# Install the fect reference used by benchmarks/cases/gsynth_xu_turnout.py.
#
# fect (Liu, Wang & Xu) is what the gsynth package now dispatches to, so its
# `method = "gsynth"` path is the reference for mlsynth's GSYNTH.
#
# Same approach as install_augsynth.sh, verified on Ubuntu + R 4.3.x in a
# sandbox where CRAN is blocked but apt and GitHub are open: apt for the
# prebuilt majority, compile the non-apt leaves from the GitHub cran mirror.
#
# COMMIT-PINNED (frozen 2026-08-04) so the cross-check runs the SAME reference
# code every time. To refresh, bump the SHAs and re-pin the expected numbers in
# benchmarks/cases/gsynth_xu_turnout.py.
#
#   fect        2.4.5    eccc9731ad74d049b22a3e769393c6a6c47b33bd  (xuyiqing/fect)
#   fixest      0.14.2   166c3b7a4d376af8687e7afabc9c12a00c41fc1c  (cran/fixest)
#   dreamerr    1.5.0    4e32a49f44b053a3a1a1f65a19e6ff5772cdb29b  (cran/dreamerr)
#   stringmagic 1.2.0    2e4d898709004abbb0700cf98c2dca057a971c9b  (cran/stringmagic)
#   future      1.75.0   b200f7cdfa1a82cbc94dcace3893bb1c2a93499a  (cran/future)
#   doFuture    1.3.0    4b1d092ac2c8bfc9e274ef0f154823d14a930481  (cran/doFuture)
#   nanoparquet 0.5.1    4b1627a63513175950304c9d5365df2977cbbb49  (cran/nanoparquet)
#
# Four ordering constraints were found the hard way and are why the sequence
# below is what it is:
#
#   * dreamerr imports stringmagic, so stringmagic installs first;
#   * fixest is a hard NAMESPACE import of fect (importFrom(fixest, feols)),
#     so fect cannot load without it however unused it looks;
#   * doFuture needs future >= 1.49, and apt ships 1.33.1, so future is
#     compiled even though an apt build exists;
#   * a failed install leaves a 00LOCK-<pkg> directory that blocks the retry,
#     so each install clears its own lock first.
#
# Install output is NOT redirected to /dev/null. A silenced `R CMD INSTALL`
# hides a dependency-order failure behind a zero exit code from the surrounding
# pipeline, and the check at the bottom is a `library()` call, not an exit code,
# for the same reason.
set -euo pipefail

FECT_SHA=eccc9731ad74d049b22a3e769393c6a6c47b33bd
FIXEST_SHA=166c3b7a4d376af8687e7afabc9c12a00c41fc1c
DREAMERR_SHA=4e32a49f44b053a3a1a1f65a19e6ff5772cdb29b
STRINGMAGIC_SHA=2e4d898709004abbb0700cf98c2dca057a971c9b
FUTURE_SHA=b200f7cdfa1a82cbc94dcace3893bb1c2a93499a
DOFUTURE_SHA=4b1d092ac2c8bfc9e274ef0f154823d14a930481
NANOPARQUET_SHA=4b1627a63513175950304c9d5365df2977cbbb49

DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran \
  r-cran-rcpp r-cran-rcpparmadillo r-cran-ggplot2 r-cran-ggally \
  r-cran-doparallel r-cran-foreach r-cran-abind r-cran-mass r-cran-gridextra \
  r-cran-mvtnorm r-cran-dplyr r-cran-reshape2 r-cran-rlang r-cran-scales \
  r-cran-future.apply r-cran-parallelly r-cran-dorng r-cran-codetools \
  r-cran-numderiv r-cran-nlme r-cran-sandwich r-cran-data.table

# Compile a GitHub repo at a pinned commit:  inst <owner/repo> <sha> <dirslug>
#
# Cloned, not fetched as a tarball: the sandbox allows git-over-HTTPS to
# github.com but blocks codeload.github.com, which the archive URLs redirect to.
# The clone is full, not --depth 1, because a shallow clone cannot check out an
# arbitrary SHA.
inst() {
  cd /tmp
  rm -rf "$3"
  rm -rf /usr/local/lib/R/site-library/00LOCK-"$3"
  git clone --quiet "https://github.com/$1" "$3"
  git -C "$3" checkout --quiet "$2"
  R CMD INSTALL --no-docs --no-help "$3"
}
inst cran/stringmagic "$STRINGMAGIC_SHA" stringmagic   # dreamerr imports it
inst cran/dreamerr    "$DREAMERR_SHA"    dreamerr      # fixest arg checking
inst cran/fixest      "$FIXEST_SHA"      fixest        # fect NAMESPACE import
inst cran/future      "$FUTURE_SHA"      future        # apt's 1.33.1 is too old
inst cran/doFuture    "$DOFUTURE_SHA"    doFuture
inst cran/nanoparquet "$NANOPARQUET_SHA" nanoparquet   # reads basedata/*.parquet
inst xuyiqing/fect    "$FECT_SHA"        fect

Rscript -e 'suppressMessages({library(fect); library(nanoparquet)}); cat("fect", as.character(packageVersion("fect")), "/ nanoparquet", as.character(packageVersion("nanoparquet")), "OK\n")'
