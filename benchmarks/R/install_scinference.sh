#!/usr/bin/env bash
# Install the authors' scinference R package (Chernozhukov, Wuthrich & Zhu;
# github.com/kwuthrich/scinference) for the cwz_ttest reference run.
#
# scinference declares CVXR in Imports, but CVXR is only used by its
# constrained-lasso estimator -- the synthetic-control t-test (sc.cf) needs only
# limSolve. We drop CVXR from Imports so the (otherwise heavy) CVXR chain is not
# required. git clone may be proxy-blocked; the codeload tarball is the reliable
# fetch (the same fallback benchmarks/reference/_fetch.py uses).
set -euo pipefail
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# dependency: limSolve (uses quadprog, lpSolve, MASS -- usually already present)
Rscript -e 'if (!"limSolve" %in% rownames(installed.packages())) quit(status=1)' \
  || { curl -sL -o limSolve.tar.gz https://codeload.github.com/cran/limSolve/tar.gz/refs/heads/master
       tar xzf limSolve.tar.gz && R CMD INSTALL limSolve-master; }

curl -sL -o scinference.tar.gz https://codeload.github.com/kwuthrich/scinference/tar.gz/refs/heads/main
tar xzf scinference.tar.gz
mv scinference-main scinference
# drop the CVXR dependency (unused by the SC t-test)
perl -0pi -e 's/Imports:\s*\n\s*limSolve,\s*\n\s*CVXR\s*\n/Imports:\n  limSolve\n/' scinference/DESCRIPTION
R CMD INSTALL scinference
Rscript -e 'suppressMessages(library(scinference)); cat("scinference OK\n")'
