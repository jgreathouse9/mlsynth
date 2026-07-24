#!/usr/bin/env bash
# Provision R's limSolve for the drosc_basque LIVE cross-validation
# (benchmarks/reference/drosc_basque/reference.R), which sources the authors'
# own DRoSC (github.com/taehyeonkoo/DRoSC, cloned by reference.R itself) and
# solves the worst-case moment-band program with limSolve::lsei.
#
# limSolve is not in the base R provisioning. CRAN is proxy-blocked in the
# sandbox (HTTP 403), so it is installed from the read-only CRAN mirror on GitHub
# via git clone + R CMD INSTALL, together with its dependencies (quadprog,
# lpSolve). The benchmark case BenchmarkSkipped()s if limSolve is still absent,
# so this provisioner is best-effort and never a hard failure.
set -uo pipefail

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not on PATH; install R first (apt-get install -y r-base-dev)." >&2
  exit 0
fi

# quadprog / lpSolve: prefer the precompiled apt builds; fall back to the mirror.
apt-get install -y r-cran-quadprog r-cran-lpsolve >/dev/null 2>&1 || true

CACHE="benchmarks/reference/.cache/rpkgs"
mkdir -p "$CACHE"
for pkg in lpSolve limSolve; do
  if Rscript -e "q(status = requireNamespace('${pkg}', quietly = TRUE))" >/dev/null 2>&1; then
    continue
  fi
  echo "::group::install ${pkg}"
  if [ ! -d "${CACHE}/${pkg}" ]; then
    git clone --depth 1 "https://github.com/cran/${pkg}.git" "${CACHE}/${pkg}" \
      || echo "clone ${pkg} failed (its cross-check will skip)"
  fi
  [ -d "${CACHE}/${pkg}" ] && R CMD INSTALL "${CACHE}/${pkg}" \
    || echo "R CMD INSTALL ${pkg} failed (its cross-check will skip)"
  echo "::endgroup::"
done

Rscript -e "cat('limSolve:', requireNamespace('limSolve', quietly = TRUE), '\n')" || true
