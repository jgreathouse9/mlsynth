#!/usr/bin/env bash
# Install the R references for benchmarks/cases/vanillasc_xval_references.py.
#
# CRAN itself is unreachable from some sandboxes, so this installs from the
# read-only GitHub mirrors of the CRAN packages instead (github.com/cran/<pkg>),
# which carry identical sources. R itself comes from the distro, not CRAN.
#
#   bash benchmarks/R/install_sc_references.sh
#
# Then capture the reference bundle with:
#   Rscript benchmarks/R/sc_references.R benchmarks/reference/vanillasc_xval_references/reference.json
set -euo pipefail

CACHE="${CACHE:-$(cd "$(dirname "$0")/../.." && pwd)/benchmarks/reference/.cache/sc_references}"
mkdir -p "$CACHE"

# Pinned so a re-capture is reproducible. tidysynth is taken from upstream
# (edunford), the rest from the CRAN mirrors.
declare -A REPOS=(
  [rgenoud]="cran/rgenoud"
  [cvTools]="cran/cvTools"
  [Synth]="cran/Synth"
  [SCtools]="cran/SCtools"
  [tidysynth]="edunford/tidysynth"
)

command -v Rscript >/dev/null || {
  echo "R not found. On Debian/Ubuntu: apt-get install -y r-base-core" >&2
  echo "Most dependencies are packaged too: apt-get install -y r-cran-{ggplot2,dplyr,tidyr,purrr,rlang,magrittr,tibble,stringr,future,furrr,optimx,kernlab}" >&2
  exit 1
}

for pkg in "${!REPOS[@]}"; do
  if Rscript -e "q(status = !requireNamespace('$pkg', quietly=TRUE))" 2>/dev/null; then
    echo "[have] $pkg"
    continue
  fi
  src="$CACHE/$pkg"
  [ -d "$src" ] || git clone -q --depth 1 "https://github.com/${REPOS[$pkg]}.git" "$src"
  echo "[install] $pkg from ${REPOS[$pkg]}"
  R CMD INSTALL "$src" >/dev/null
done

Rscript -e 'for (p in c("Synth","tidysynth","SCtools")) cat(sprintf("%-10s %s\n", p, as.character(packageVersion(p))))'
