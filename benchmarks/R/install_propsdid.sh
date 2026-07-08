#!/usr/bin/env bash
# Obtain the authors' R package propsdid for the propsc_spain reference run
# (benchmarks/R/propsdid_spain.R), which cross-validates mlsynth's PROPSC
# against Bogatyrev & Stoetzer (2026), "Estimating Treatment Effects on
# Proportions with Synthetic Controls" (Political Analysis).
#
# propsdid (github.com/lstoetze/propsdid) is a GPL (>= 2) fork of synthdid that
# adds the common-weights multivariate estimator. Its estimation core
# (R/{solver,utils,synthdid,vcov_multivar}.R) is base-R only
# (DESCRIPTION Imports: none; NAMESPACE: importFrom(stats, var)), so the
# reference script source()s those four files directly -- no package build, no
# CRAN call. The package is not vendored into mlsynth; it is fetched on demand
# at a pinned commit into benchmarks/reference/.cache (gitignored).
#
# COMMIT-PINNED so the cross-check runs the SAME reference every time. To refresh,
# bump COMMIT here and in propsdid_spain.R, then re-pin _FROZEN_REFERENCE in
# benchmarks/cases/propsc_spain.py from a fresh PROPSDID_LIVE=1 run.
#
#   propsdid   0.0.1   9ec3f65e754af3b915dd884aaed68f7595f527d9
#
# git clone is proxy-blocked in the sandbox (HTTP 403); the codeload tarball of
# the same pinned commit is the reliable fetch.
set -euo pipefail

COMMIT=9ec3f65e754af3b915dd884aaed68f7595f527d9
REPO=lstoetze/propsdid
CACHE="benchmarks/reference/.cache/propsdid"

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not on PATH; install R (e.g. apt-get install -y r-base-core) first." >&2
  exit 1
fi

# Fetch the package source at the pinned commit (benchmarks/cases/propsc_spain.py
# does this itself; this mirrors it so the cache can be warmed manually). Prefer
# the codeload tarball; fall back to a git clone + checkout where codeload is
# proxy-gated (the mlsynth sandbox gates codeload to the enabled repo but allows
# git to public repos).
if [ ! -f "${CACHE}/R/synthdid.R" ]; then
  mkdir -p "$(dirname "$CACHE")"
  if curl -sfL "https://codeload.github.com/${REPO}/tar.gz/${COMMIT}" -o propsdid.tar.gz \
       && tar xzf propsdid.tar.gz; then
    rm -rf "$CACHE"; mv "propsdid-${COMMIT}" "$CACHE"; rm -f propsdid.tar.gz
  else
    rm -f propsdid.tar.gz
    tmp="$(dirname "$CACHE")/propsdid_git"
    rm -rf "$tmp"
    git clone --quiet "https://github.com/${REPO}.git" "$tmp"
    git -C "$tmp" checkout --quiet "$COMMIT"
    rm -rf "$tmp/.git" "$CACHE"; mv "$tmp" "$CACHE"
  fi
fi

Rscript -e 'cache <- "benchmarks/reference/.cache/propsdid/R";
  for (f in c("solver.R","utils.R","synthdid.R","vcov_multivar.R")) source(file.path(cache, f));
  cat("propsdid reference core sourced OK (sc_estimate, panel.array present:",
      exists("sc_estimate") && exists("panel.array"), ")\n")'
