#!/usr/bin/env bash
# Install the authors' MSCMT R package (Becker & Klossner 2018,
# github.com/mabe0033/MSCMT) for the mscmt_basque reference run.
#
# MSCMT's optimisation path (outer.optim="DEoptim") needs all its hard Imports
# built from source plus DEoptim (a Suggests). git clone is proxy-blocked here;
# the codeload tarball is the reliable fetch. CRAN deps come from the cran/<pkg>
# GitHub mirror the same way. None of MSCMT's Imports are dropped -- the inner LP
# (Rglpk -> lpSolveAPI -> lpSolve fallback chain) and Rdpack's doc macros all
# build given a couple of system libraries.
#
# System libraries required (install via your package manager before running):
#   - GLPK  (libglpk-dev)  for the Rglpk LP solver.
# Everything else builds with the system gcc/gfortran toolchain.
set -euo pipefail
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# GLPK system library is needed to compile Rglpk.
if [ ! -e /usr/include/glpk.h ]; then
  apt-get install -y libglpk-dev >/dev/null
fi

fetch_install() {  # $1 = CRAN package name (installed from the cran/ GitHub mirror)
  local pkg="$1"
  Rscript -e "if (\"$pkg\" %in% rownames(installed.packages())) quit(status=1)" \
    && { curl -sL -o "$pkg.tar.gz" \
           "https://codeload.github.com/cran/$pkg/tar.gz/refs/heads/master"
         tar xzf "$pkg.tar.gz"
         # Rdpack's help-prep can hit a fortify buffer-overflow on this toolchain;
         # the runtime macros (used only for docs) install fine without help.
         R CMD INSTALL --no-help --no-docs --no-byte-compile "$pkg-master"; } \
    || true
}

# MSCMT Imports (and transitive CRAN deps) + the DEoptim outer optimiser.
fetch_install rbibutils      # Rdpack dependency
fetch_install Rdpack
fetch_install slam           # Rglpk dependency
fetch_install Rglpk
fetch_install lpSolveAPI
fetch_install DEoptim
# lpSolve, ggplot2, rlang, parallel, stats, utils are typically already present.

curl -sL -o mscmt.tar.gz \
  https://codeload.github.com/mabe0033/MSCMT/tar.gz/refs/heads/master
tar xzf mscmt.tar.gz
R CMD INSTALL --no-help --no-docs MSCMT-master
Rscript -e 'suppressMessages(library(MSCMT)); cat("MSCMT", as.character(packageVersion("MSCMT")), "OK\n")'
