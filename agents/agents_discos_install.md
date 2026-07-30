# Installing DiSCos in this sandbox — an exact runbook

A step-by-step recipe for getting a working [DiSCos](https://github.com/Davidvandijcke/DiSCos)
(Distributional Synthetic Controls, Gunsilius 2023) into the Claude Code remote sandbox,
so it can be used as a live R reference for cross-validating mlsynth's `DSC` / `DSCAR`
ports — see `benchmarks/cases/dsc_dube.py`.

Verified end to end on 2026-07-30, on Ubuntu 24.04 with R 4.3.3. Read
`agents_r_environment.md` first for the general sandbox constraints; this file is the
DiSCos-specific replay, and `benchmarks/R/install_augsynth.sh` is the same pattern at a
third of the depth if you want the shape without the detail.

## TL;DR

```bash
bash benchmarks/R/install_discos.sh
```

Budget 15–25 minutes on a cold container. Twelve packages compile from source, several
with substantial C/C++/Fortran (CVXR, scs, evmix, the extreme-value chain).

## What makes DiSCos harder than augsynth

The sandbox constraints are identical to augsynth's — CRAN blocked, `codeload` blocked,
apt and `git clone` open, so the recipe is still "apt for the prebuilt majority, clone
`cran/<pkg>` for the rest." See `agents_r_environment.md` for that evidence table.

What is different is the depth of the tree. augsynth needed three source builds with a
hand-written list. DiSCos needs twelve, and the list is not obvious up front:
`extremeStat` alone pulls in a chain of extreme-value packages (lmomco → Lmoments,
berryFunctions, evir, ismev, extRemes → distillery, Renext) that apt does not carry.

So `install_discos.sh` does not hand-list dependencies. It resolves the closure
automatically with a `need <Pkg>` function that prefers the apt binary and otherwise
clones `cran/<Pkg>`, recurses through that package's own Depends/Imports/LinkingTo, and
then compiles. The dependency parsing is `read.dcf` in R, and base/recommended packages
are excluded because they ship with `r-base`:

```bash
need() {
  local pkg="$1" sha="${2:-}" deb dir
  have "$pkg" && return 0
  deb="r-cran-$(echo "$pkg" | tr '[:upper:]' '[:lower:]')"
  if [ -z "$sha" ] && apt-cache show "$deb" >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$deb" \
      && { have "$pkg" && return 0; }
  fi
  dir="$SRC/$pkg"; rm -rf "$dir"
  git clone --quiet "https://github.com/cran/$pkg" "$dir"
  [ -n "$sha" ] && git -C "$dir" checkout --quiet "$sha"
  while read -r d; do [ -n "$d" ] && need "$d"; done < <(deps "$dir")
  R CMD INSTALL --no-docs --no-help "$dir"
}
```

Reuse this resolver for any other R reference package. It is the generalisation of the
augsynth recipe and saves the manual closure-walking entirely.

## The two things you have to get right by hand

Everything else the resolver handles. These two it cannot.

### 1. Pin CVXR to the 1.0 series

DiSCos declares a bare `CVXR` with no version constraint, so an unpinned resolve grabs
1.9.1 — and that is a wall:

- CVXR >= 1.8 requires `Matrix >= 1.7` and `Rcpp >= 1.1`. Noble ships Matrix 1.6.5 and
  Rcpp 1.0.12, so both would themselves have to be source-built.
- It adds `clarabel`, which needs a full Rust toolchain, plus `highs`.

CVXR 1.0-15 needs only `scs` beyond what apt provides. Pin it:

```
CVXR  1.0-15  871a29e5f770a8479cc1aa77cf437e20dd16dc79  (cran/CVXR)
```

This is why `need CVXR "$CVXR_SHA"` is called before the loop — so the resolver cannot
reach a newer CVXR as a transitive dependency of anything else.

### 2. Install quadprog, which is mis-declared

This one will bite you. `quadprog` sits in DiSCos' `Suggests`, not `Imports`, so a
correct Depends/Imports resolve skips it — and then `library(DiSCos)` succeeds and the
first real fit dies:

```
Error: quadprog needed for this function to work. Please install it.
```

It is a hard runtime requirement of the weights solver on the default non-mixture path
(`R/DiSCo_weights_reg.R:82` guards it with `requireNamespace`). apt has it as
`r-cran-quadprog`. The script installs it explicitly with a comment saying why.

The general lesson, and the same one augsynth taught: a package that imports cleanly is
not a package that works. Suggests-but-actually-required is common in R, and the only
thing that catches it is running a real fit.

## What ends up where

Pinned commits, frozen 2026-07-30:

| package | version | commit                                     | source                |
|---------|---------|--------------------------------------------|-----------------------|
| DiSCos  | 0.1.4   | `ed2b3d948ed591ed2785e1d97acb567225432a60` | Davidvandijcke/DiSCos |
| CVXR    | 1.0-15  | `871a29e5f770a8479cc1aa77cf437e20dd16dc79` | cran/CVXR             |

Compiled from source by the resolver (12): CVXR, scs, evmix, extremeStat, lmomco,
Lmoments, berryFunctions, evir, ismev, extRemes, distillery, Renext.

Taken prebuilt from apt: data.table, ggplot2, pracma, Rdpack, MASS, quadprog, plus the
transitive closure of each (Matrix, Rcpp, RcppEigen, bit64, gmp, Rmpfr, ECOSolveR, slam,
cli, R6, gsl, SparseM, pbapply, RColorBrewer, evd, fExtremes, mgcv, …). The system
libraries `libgmp-dev`, `libmpfr-dev` and `libgsl-dev` are needed for gmp/Rmpfr/gsl and
are installed up front.

`parallel` and `utils` are in DiSCos' Imports but are base R — nothing to install.

## Which dependency backs which code path

Worth knowing when something breaks, because the failure is usually specific to one
option rather than to the package as a whole:

| dependency  | used by                                        |
|-------------|------------------------------------------------|
| quadprog    | default weights solver (`simplex`/non-mixture) |
| pracma      | `lsqlincon` constrained least-squares weights  |
| CVXR        | `mixture = TRUE` mixture-of-distributions path |
| evmix       | `qmethod = "qkden"` only                       |
| extremeStat | `qmethod = "extreme"` only                     |

`evmix` and `extremeStat` are each used in exactly one call site and back optional
quantile-estimation methods. If you only need the default path and want to cut the build
substantially, you can drop them from the `need` loop — but then `qmethod` is unavailable,
so do not drop them for a reference install meant to cross-check arbitrary options.

## Verifying

`library(DiSCos)` is not a sufficient check — that is exactly what passes while quadprog
is missing. Run a real fit against the bundled Dube panel:

```r
library(DiSCos); library(data.table)
data(dube)                       # 652,870 x 3
disco <- DiSCo(copy(dube), id_col.target = 2, t0 = 2003,
               G = 100, M = 100, num.cores = 1, seed = 1,
               simplex = TRUE, q_max = 0.9)
sum(disco$weights)               # ~1 under simplex = TRUE
DiSCoTEA(disco, agg = "quantileDiff", graph = FALSE)
```

The installer ends with exactly this fit and asserts `sum(weights) ≈ 1`, so a successful
run of the script is itself the check.

Pass `mixture = TRUE` to exercise the CVXR path, and `qmethod = "qkden"` /
`qmethod = "extreme"` to exercise evmix / extremeStat. Use `copy(dube)` — DiSCos takes a
`data.table` and modifies it by reference, so a bare `dube` mutates the bundled dataset
for the rest of the session.

Expected numbers are not pinned here yet. The vignette's own call
(`G = 1000, boots = 1000, permutation = TRUE, CI = TRUE`) is expensive, and its published
weight/QTE values live in figures rather than printed tables — see the note in
`benchmarks/cases/dsc_dube.py`. Capture reference numbers into
`benchmarks/reference/<case>/` when you pin them, per `agents_benchmarking.md`.

## Gotchas

- The container is ephemeral. Re-run the install once per fresh container, and never
  write a benchmark case that assumes R is present — CI has no R.
- `DiSCo()` is genuinely slow: it integrates over full distributions. For iteration use
  `G = 100`, `M = 100`, `CI = FALSE`, `permutation = FALSE`; the vignette's settings are
  a final-run configuration, not a debug loop.
- Do not bump CVXR to 1.9.x hoping it "just works" — see the pinning section. If you do
  need a newer CVXR, budget for Rust plus source builds of Matrix and Rcpp.
