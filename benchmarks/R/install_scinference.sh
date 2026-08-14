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
  || { for pkg in lpSolve limSolve; do   # limSolve needs lpSolve
         git clone --quiet --depth 1 "https://github.com/cran/$pkg" "$pkg" \
           && R CMD INSTALL --no-docs --no-help "$pkg"
       done; }

# Fetch at a PINNED SHA. A cross-check must run the same reference code every time,
# and main drifts. Two fetch paths because which one works is environment-dependent:
# the codeload tarball is fastest, but it is 403-blocked behind some egress proxies
# (including the Claude Code sandbox -- see agents/agents_r_environment.md), where a
# plain git clone is the escape hatch.
SCINFERENCE_SHA="${SCINFERENCE_SHA:-567c6889ce0a1d269a62d415b88aa6baf723a3fe}"
if curl -sfL -o scinference.tar.gz \
     "https://codeload.github.com/kwuthrich/scinference/tar.gz/$SCINFERENCE_SHA"; then
  tar xzf scinference.tar.gz
  mv "scinference-$SCINFERENCE_SHA" scinference
else
  echo "codeload unreachable; falling back to git clone" >&2
  git clone --quiet https://github.com/kwuthrich/scinference scinference
  git -C scinference checkout --quiet "$SCINFERENCE_SHA"
fi
# drop the CVXR dependency (unused by the SC t-test)
perl -0pi -e 's/Imports:\s*\n\s*limSolve,\s*\n\s*CVXR\s*\n/Imports:\n  limSolve\n/' scinference/DESCRIPTION
R CMD INSTALL scinference
Rscript -e 'suppressMessages(library(scinference)); cat("scinference OK\n")'
