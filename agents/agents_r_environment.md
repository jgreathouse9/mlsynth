# Installing R (and R reference packages) in this sandbox

This note is for a future agent who needs a live R reference to cross-validate a
Python port — the differential-TDD workflow in `agents_tests.md`. It records what
actually works in the Claude Code remote sandbox, where the naive `install.packages()`
path is dead on arrival.

## The one thing to know: CRAN is blocked, apt and GitHub-over-git are open

The sandbox routes outbound HTTPS through an egress proxy with an allowlist. In
practice:

- `install.packages("augsynth")` / any `cran.r-project.org` fetch — BLOCKED (403).
- `codeload.github.com/.../tar.gz/<sha>` (what `remotes::install_github` and the
  `curl -L codeload...` idiom use) — BLOCKED in this session. Do not trust a
  script that reaches for `codeload`; it may have worked in some past
  environment but not here.
- `apt-get install r-base ...` — WORKS (the Ubuntu archive is reachable).
- `git clone https://github.com/<owner>/<repo>` — WORKS. This is the escape hatch
  for any package that isn't packaged for apt.
- CRAN/Bioconductor package DATA (e.g. dataverse-hosted example datasets) — BLOCKED.
  Vendor the data into `basedata/` from a source you can reach, or reconstruct it.

So the recipe is: apt for everything Debian ships prebuilt (this is most of the
dependency tree), then `git clone` at a pinned SHA + `R CMD INSTALL` for the few
leaves apt doesn't carry.

## Step 1 — base R and the prebuilt dependency majority via apt

```bash
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential cmake gfortran \
  r-cran-dplyr r-cran-tidyr r-cran-magrittr r-cran-ggplot2 r-cran-formula \
  r-cran-rlang r-cran-purrr r-cran-fnn r-cran-rcpp r-cran-r6 \
  r-cran-doparallel r-cran-foreach r-cran-gridextra r-cran-lifecycle \
  r-cran-stringr r-cran-tibble r-cran-rcpparmadillo r-cran-rcppeigen r-cran-bh \
  r-cran-glmnet r-cran-mass r-cran-matrix
```

Ubuntu ships hundreds of `r-cran-*` packages. When a `git clone`+`R CMD INSTALL`
fails on a missing dependency, first check whether `apt-cache search r-cran-<dep>`
has it — installing the prebuilt binary is faster and avoids a compile.

If `apt-get update` fails with a 405/403 through the proxy, a third-party PPA is
usually the culprit (deadsnakes / ondrej are commonly pre-added and are NOT on the
allowlist). Disable them and retry:

```bash
for f in /etc/apt/sources.list.d/*.sources; do mv "$f" "$f.disabled"; done
DEBIAN_FRONTEND=noninteractive apt-get update -qq
```

## Step 2 — the non-apt leaves via pinned `git clone`

For anything apt doesn't carry, clone the CRAN GitHub mirror (`cran/<pkg>`) or the
upstream repo at a FROZEN commit and install from source. Pinning the SHA is not
optional: a bit-for-bit cross-check must run the same reference code every time, and
active-dev master branches drift (this is exactly what stales a pinned expected
number). Record every SHA in the benchmark case that depends on it.

```bash
inst() {   # inst <owner/repo> <sha> <dirslug>
  cd /tmp
  git clone --quiet "https://github.com/$1" "$3"
  git -C "$3" checkout --quiet "$2"
  R CMD INSTALL --no-docs --no-help "$3"
}
# order matters: dependencies before dependents
inst cran/S7               33c8f3212c62cd2ebec79cd61d1315e9acc84128 S7
inst cran/LiblineaR        07cca10ee74e2442a8726173bd52360c323ad07e LiblineaR
inst cran/osqp             260dc73e1e3d07ccb7dbff85b62eaaf483672394 osqp
inst ebenmichael/augsynth  7a90ea48877fae7925a72cb50bc03a315bc7c042 augsynth  # 0.2.0

Rscript -e 'suppressMessages(library(augsynth)); cat("augsynth", as.character(packageVersion("augsynth")), "OK\n")'
```

The pinned reference set used by the augsynth cross-checks (PPSCM, ASCM):

| package    | version  | commit                                     | source              |
|------------|----------|--------------------------------------------|---------------------|
| augsynth   | 0.2.0    | `7a90ea48877fae7925a72cb50bc03a315bc7c042` | ebenmichael/augsynth|
| osqp       | 1.0.0    | `260dc73e1e3d07ccb7dbff85b62eaaf483672394` | cran/osqp           |
| S7         | 0.2.2    | `33c8f3212c62cd2ebec79cd61d1315e9acc84128` | cran/S7             |
| LiblineaR  | 2.10.24  | `07cca10ee74e2442a8726173bd52360c323ad07e` | cran/LiblineaR      |

`benchmarks/R/install_augsynth.sh` is the canonical script. Note it currently uses
the `codeload` tarball idiom, which worked in the environment it was authored in but
was blocked in this session — if it fails on the fetch, swap the `curl … codeload …`
body of `inst()` for the `git clone` + `checkout` form above. The SHAs are the same.

## Step 3 — dumping intermediates from the reference (for seam-level TDD)

The whole point of a live R install is to compare the Python port against the
reference *at each seam*, not just at the final number (see `agents_tests.md`,
"Cross-Implementation Differential TDD"). Two techniques that pay off:

Trace an internal function and capture its locals on exit — no need to edit the
package source:

```r
trace(augsynth:::multisynth_qp,
      exit = quote(saveRDS(mget(ls()), "/tmp/qp_locals.rds")),
      print = FALSE)
# ... run the fit that calls it ...
untrace(augsynth:::multisynth_qp)
```

Round-trip R objects with `saveRDS`, never CSV. A `Matrix`/`dgeMatrix`, or a list of
per-cohort blocks, does not survive `write.csv` — `as.matrix(a_list)` silently yields
an array-of-lists. `saveRDS` on the R side and `pyreadr`/`rpy2` (or a tiny R helper
that re-emits plain arrays) on the Python side preserve structure exactly. When you
need a specific intermediate as a plain numeric block, have R write it explicitly:

```r
con <- readRDS("/tmp/qp_locals.rds")
write.table(as.matrix(con$Z_scaled), "/tmp/z_scaled.tsv",
            row.names = FALSE, col.names = FALSE)
```

Then load the `.tsv` on the Python side and diff element-wise against the port's
intermediate. A constant ratio between the two blocks localizes the disagreement to a
single scaling seam — which is how the PPSCM `sdx = sd(X[[1]][is.finite(trt)])` seam
was found (`multi_synth_qp.R:98`). See the PPSCM worked example in `agents_tests.md`.

## Gotchas

- The install compiles C++/Fortran (RcppArmadillo, osqp, LiblineaR). It is slow the
  first time and needs `build-essential cmake gfortran`. Run it once per fresh
  container and cache nothing — the container is ephemeral.
- Example datasets hosted on dataverse (augsynth's `?multisynth` uses one) are behind
  the blocked egress. The Paglayan teacher-pay panel is vendored at
  `basedata/Teachingaugsynth.scv`; prefer vendored data over a live download.
- Keep the R reference out of the Python test path. Live-R cross-checks are captured
  once into `benchmarks/reference/<case>/` (`reference.out` + `reference.json`) and
  the Python tests assert against those frozen numbers — CI has no R.
