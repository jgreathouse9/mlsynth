#!/usr/bin/env bash
# Install the DiSCos reference (Distributional Synthetic Controls, Gunsilius
# 2023) used to cross-validate mlsynth's DSC / DSCAR ports -- see
# benchmarks/cases/dsc_dube.py.
#
# Same sandbox constraints as install_augsynth.sh: CRAN is blocked, codeload is
# blocked, apt and git-over-HTTPS to github.com are open. So the recipe is again
# "apt for the prebuilt majority, clone cran/<Pkg> for the rest".
#
# What is different from augsynth is the DEPTH of the tree. augsynth needed
# three source builds and a hand-written list. DiSCos needs twelve, and the list
# is not knowable up front: extremeStat alone pulls a chain of extreme-value
# packages (lmomco -> Lmoments, berryFunctions, evir, ismev, extRemes ->
# distillery, Renext) that apt does not carry. Hand-walking that closure is
# error-prone and goes stale, so `need` below resolves it automatically:
# prefer the apt binary, else clone cran/<Pkg>, recurse through that package's
# own Depends/Imports/LinkingTo, then compile.
#
# Budget 15-25 minutes on a cold container -- CVXR, scs and the extreme-value
# chain are substantial C/C++/Fortran builds.
#
# COMMIT-PINNED (frozen 2026-07-30) so the cross-check runs the SAME reference
# every time. To refresh, bump the SHAs and re-pin the expected numbers in the
# relevant benchmarks/cases/*.py.
#
#   DiSCos  0.1.4    ed2b3d948ed591ed2785e1d97acb567225432a60  (Davidvandijcke/DiSCos)
#   CVXR    1.0-15   871a29e5f770a8479cc1aa77cf437e20dd16dc79  (cran/CVXR)
set -euo pipefail

DISCOS_SHA=ed2b3d948ed591ed2785e1d97acb567225432a60
# CVXR MUST be pinned to the 1.0 series and MUST be resolved before anything
# else can pull it in transitively. DiSCos declares a bare `CVXR` with no
# version constraint, so an unpinned resolve takes 1.9.1 -- and that is a wall,
# not an inconvenience:
#
#   * CVXR >= 1.8 requires Matrix >= 1.7 and Rcpp >= 1.1. Noble ships Matrix
#     1.6.5 and Rcpp 1.0.12, so both would have to be source-built too.
#   * it adds `clarabel`, which needs a full Rust toolchain, plus `highs`.
#
# 1.0-15 needs only `scs` beyond what apt provides. Do not bump this hoping it
# "just works"; if a newer CVXR is genuinely required, budget for Rust plus
# source builds of Matrix and Rcpp.
CVXR_SHA=871a29e5f770a8479cc1aa77cf437e20dd16dc79

SRC=/tmp/discos-src
mkdir -p "$SRC"

DEBIAN_FRONTEND=noninteractive apt-get update -qq
# libgmp/libmpfr/libgsl are the system side of gmp, Rmpfr and gsl, which come in
# transitively under CVXR; without them those builds fail at configure.
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran git \
  libgmp-dev libmpfr-dev libgsl-dev \
  r-cran-data.table r-cran-ggplot2 r-cran-pracma r-cran-rdpack r-cran-mass \
  r-cran-matrix r-cran-rcpp r-cran-rcppeigen r-cran-bit64 r-cran-gmp \
  r-cran-rmpfr r-cran-ecosolver r-cran-slam r-cran-cli r-cran-r6 r-cran-gsl \
  r-cran-sparsem r-cran-pbapply r-cran-rcolorbrewer r-cran-evd r-cran-mgcv \
  r-cran-quadprog

have() { Rscript -e "quit(status = !requireNamespace('$1', quietly = TRUE))" \
           >/dev/null 2>&1; }

# A package's own Depends/Imports/LinkingTo, minus base and recommended packages
# (those ship with r-base, so trying to build them is both wasteful and likely
# to fail against the system copies).
deps() {
  Rscript -e '
    d <- read.dcf(file.path(commandArgs(TRUE)[1], "DESCRIPTION"))
    f <- unlist(lapply(c("Depends", "Imports", "LinkingTo"), function(k)
      if (k %in% colnames(d)) strsplit(d[1, k], ",")[[1]] else character()))
    f <- trimws(sub("\\(.*", "", f))
    base <- rownames(installed.packages(priority = c("base", "recommended")))
    cat(setdiff(f[nzchar(f)], c("R", base)), sep = "\n")' "$1" 2>/dev/null
}

# need <Pkg> [sha] -- ensure <Pkg> is importable, building it if necessary.
# Prefers the apt binary (fast, no compile) unless a SHA is given, in which case
# the pin is the point and apt would defeat it.
need() {
  local pkg="$1" sha="${2:-}" deb dir
  have "$pkg" && return 0
  deb="r-cran-$(echo "$pkg" | tr '[:upper:]' '[:lower:]')"
  if [ -z "$sha" ] && apt-cache show "$deb" >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      "$deb" >/dev/null 2>&1 && { have "$pkg" && return 0; }
  fi
  echo "  building $pkg${sha:+ @ ${sha:0:8}}"
  dir="$SRC/$pkg"
  rm -rf "$dir"
  # git clone, NOT a tarball: codeload.github.com is 403 in this sandbox and
  # github.com/<o>/<r>/archive/<sha>.tar.gz redirects to it. A full clone, not
  # --depth 1, because a shallow clone cannot check out an arbitrary SHA.
  git clone --quiet "https://github.com/cran/$pkg" "$dir"
  [ -n "$sha" ] && git -C "$dir" checkout --quiet "$sha"
  while read -r d; do [ -n "$d" ] && need "$d"; done < <(deps "$dir")
  R CMD INSTALL --no-docs --no-help "$dir"
}

# CVXR first, so the resolver cannot reach a newer one transitively (see above).
need CVXR "$CVXR_SHA"

# quadprog is declared in DiSCos' *Suggests*, not Imports, so a correct
# Depends/Imports resolve skips it -- and then `library(DiSCos)` succeeds while
# the first real fit dies with "quadprog needed for this function to work".
# It is a hard runtime requirement of the default (non-mixture) weights solver;
# R/DiSCo_weights_reg.R guards it with requireNamespace. Installed explicitly
# above via apt; asserted here so a change to the apt list cannot silently drop
# it.
need quadprog

# Optional quantile-estimation backends. evmix backs qmethod = "qkden" and
# extremeStat backs qmethod = "extreme"; each is used at exactly one call site.
# Dropping them cuts the build substantially but makes those options unavailable,
# which is not acceptable for a reference meant to cross-check arbitrary options.
for p in data.table ggplot2 pracma Rdpack MASS evmix extremeStat; do
  need "$p"
done

echo "  building DiSCos @ ${DISCOS_SHA:0:8}"
rm -rf "$SRC/DiSCos"
git clone --quiet https://github.com/Davidvandijcke/DiSCos "$SRC/DiSCos"
git -C "$SRC/DiSCos" checkout --quiet "$DISCOS_SHA"
while read -r d; do [ -n "$d" ] && need "$d"; done < <(deps "$SRC/DiSCos")
R CMD INSTALL --no-docs --no-help "$SRC/DiSCos"

# `library(DiSCos)` is NOT a sufficient check -- that is exactly what passes
# while quadprog is missing. Run a real fit, on the default non-mixture path
# that quadprog backs.
Rscript -e '
  suppressMessages({library(DiSCos); library(data.table)})
  data(dube)
  d <- DiSCo(copy(dube), id_col.target = 2, t0 = 2003, G = 100, M = 100,
             num.cores = 1, seed = 1, simplex = TRUE, q_max = 0.9)
  stopifnot(abs(sum(d$weights) - 1) < 1e-6)
  cat("DiSCos", as.character(packageVersion("DiSCos")),
      "/ CVXR", as.character(packageVersion("CVXR")),
      "OK (real fit, sum(weights) =", sprintf("%.6f", sum(d$weights)), ")\n")'
