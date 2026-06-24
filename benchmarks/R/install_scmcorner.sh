#!/usr/bin/env bash
# Obtain Malo, Eskelinen, Zhou & Kuosmanen's bilevel solver `scm.corner` for the
# malo_basque cross-validation reference (github.com/Xun90/SCM-Debug, MIT).
#
# SCM-Debug ships scm.corner as a single R source file (not an installable
# package); the malo_basque bundle vendors it verbatim at
# benchmarks/reference/malo_basque/scm.corner.R (see that dir's NOTICE), so the
# reference is reproducible offline -- this script only documents/refreshes the
# upstream fetch. `git clone` is proxy-blocked in the sandbox (HTTP 403); the
# codeload tarball is the reliable fetch (the same fallback _fetch.py uses).
#
# scm.corner needs only QP/LP solvers already present in the system R:
# kernlab (ipop), LowRankQP, lpSolve, quadprog, limSolve. It is run by
# benchmarks/reference/malo_basque/reference.R (sources the vendored copy) and
# by benchmarks/R/scmcorner_basque.R (sources /tmp/SCM-Debug).
set -euo pipefail

DEST="${1:-benchmarks/reference/malo_basque}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# verify the QP/LP solvers scm.corner depends on are installed
Rscript -e 'pkgs <- c("kernlab","LowRankQP","lpSolve","quadprog","limSolve");
            miss <- pkgs[!pkgs %in% rownames(installed.packages())];
            if (length(miss)) { cat("missing:", paste(miss, collapse=" "), "\n"); quit(status=1) };
            cat("solver deps OK\n")'

# main first, then master, via codeload (git clone is 403 in the sandbox)
for ref in main master; do
  if curl -sfL -o scm.tar.gz "https://codeload.github.com/Xun90/SCM-Debug/tar.gz/refs/heads/${ref}"; then
    tar xzf scm.tar.gz && break
  fi
done
SRC="$(find . -maxdepth 2 -name scm.corner.R | head -n1)"
[ -n "$SRC" ] || { echo "could not fetch scm.corner.R from SCM-Debug"; exit 1; }

cp "$SRC" "$(dirname "$SRC")/LICENSE" "$OLDPWD/$DEST/" 2>/dev/null || cp "$SRC" "$OLDPWD/$DEST/"
echo "vendored scm.corner.R -> $DEST/ (refresh benchmarks/reference/malo_basque/NOTICE provenance if upstream changed)"
