#!/usr/bin/env bash
# Fetch Abadie & Zhao's SCDesign reference code (github.com/jinglongzhao2/SCDesign)
# for the marex_walmart reference run.
#
# SCDesign is NOT an R package -- it ships as loose R source (the per-figure/table
# "LazyRun" scripts under numbered directories). We do not install it; the
# reference script benchmarks/reference/marex_walmart/reference.R reproduces the
# two functions on the open path -- `Synthetic_Control` (the quadprog
# constrained-least-squares SC weight QP) and `Synthetic_Experiment_Cardinality_
# Constraint` (the paper's constrained / cardinality-K design) -- verbatim from
# SCDesign's "3. Walmart Data Simulations/Walmart_LazyRun.R", plus their
# `permutation.test` and `quantile_blank`. SCDesign's *other* design routine,
# the non-convex `Synthetic_Experiment` MIQP, needs a commercial Gurobi licence
# and is NOT used: the cardinality routine above is the exact design MAREX's
# `m_eq` solves, and it runs on the open quadprog backend.
#
# R-package dependencies on the needed path: quadprog (solve.QP) and Matrix
# (nearPD) -- both base/recommended or trivially installable; no GLPK/Rglpk and
# no Gurobi. (SCDesign's scripts also `library(gurobi, slam, limSolve, gamlr,
# EnvStats, tidyr, dplyr, matrixcalc)`, but none of those are on the cardinality
# + quadprog + permutation path the reference run uses.)
#
# git clone is proxy-blocked (403) in this environment; the codeload tarball is
# the reliable fetch (main confirmed 200). This script documents/vendors the
# source; reference.R is self-contained and does NOT need the tarball present.
set -euo pipefail

# 1. R-package dependencies on the open path.
Rscript -e 'for (p in c("quadprog","Matrix")) if (!p %in% rownames(installed.packages())) {
  install.packages(p, repos="https://cloud.r-project.org") }
  suppressMessages({library(quadprog); library(Matrix)}); cat("quadprog + Matrix OK\n")'

# 2. Fetch SCDesign source (documentation / inspection; reference.R is self-contained).
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT; cd "$TMP"
curl -sL -o scdesign.tar.gz \
  https://codeload.github.com/jinglongzhao2/SCDesign/tar.gz/refs/heads/main
tar xzf scdesign.tar.gz
echo "SCDesign source fetched. The Walmart design lives in:"
ls -d "SCDesign-main/3. Walmart Data Simulations"* || true
echo "Reference functions are reproduced verbatim in"
echo "  benchmarks/reference/marex_walmart/reference.R"
echo "Run it from the repo root: Rscript benchmarks/reference/marex_walmart/reference.R"
