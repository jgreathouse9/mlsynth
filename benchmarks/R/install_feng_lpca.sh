#!/usr/bin/env bash
# Install the R toolchain for the `lpca_kansas` and `lpca_mc` reference runs.
#
# The reference here is not a package but the author's own replication scripts:
# yingjieum/Replication_NonlinearFactorModel_2023, which carries Feng's
# `knn.index` / `findknn` / `lpca` / `sel.r` and the Section 5 simulation
# harness. The reference runs source those files directly, so the numbers
# mlsynth is pinned against come from the author's code and not from constants
# transcribed out of the PDF.
#
# Every SHA below is frozen. A bit-for-bit cross-check has to run the same
# reference code every time, and CRAN mirror masters drift.
#
# CRAN itself is blocked through this sandbox's egress proxy (see
# agents/agents_r_environment.md), so apt supplies everything Debian prebuilds
# and the remaining leaves come from the cran/<pkg> GitHub mirrors.
set -euo pipefail

FENG_SHA="ca34fba2b336967c352794a9d5f42cbd375f42b0"   # repo HEAD, 2024-08-01
ZIGG_SHA="c24659f6dffccacaf733744312af1b126b68ea41"      # zigg
ZIGGURAT_SHA="d27fea9f91a6abb2c7e474de1effb98475e4b6bd"  # RcppZiggurat
RFAST_SHA="8dff98abdf8da3af561c1141390c9a134942e4c4"     # Rfast
HDRFA_SHA="eb0e8c9844db647b070998d8dbdcc42b0761f04a"     # HDRFA

DEST="${LPCA_REFERENCE_DIR:-/tmp/feng_lpca}"
mkdir -p "$DEST"

# ---------------------------------------------------------------- base R ----
# The Ubuntu archive is reachable; third-party PPAs (deadsnakes / ondrej) are
# not on the allowlist and break `apt-get update` if enabled.
for f in /etc/apt/sources.list.d/*.sources; do
  case "$f" in *ubuntu.sources) ;; *) [ -e "$f" ] && mv "$f" "$f.disabled" || true ;; esac
done
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  r-base r-base-dev build-essential gfortran libgsl-dev \
  r-cran-irlba r-cran-matrix r-cran-rcpp r-cran-rcpparmadillo \
  r-cran-rcppparallel r-cran-rcppgsl r-cran-quantreg r-cran-pracma

# ------------------------------------------------------- the CRAN leaves ----
# Rfast supplies colMaxs / nth, which Feng's matching step calls; HDRFA supplies
# PCA_FN, the eigenvalue-ratio factor count behind the simulation's global-PCA
# baseline. Order matters: zigg -> RcppZiggurat -> Rfast.
install_pinned() {
  local pkg="$1" sha="$2" tmp
  tmp="$(mktemp -d)"
  git clone -q "https://github.com/cran/${pkg}.git" "$tmp/$pkg"
  git -C "$tmp/$pkg" checkout -q "$sha"
  R CMD INSTALL "$tmp/$pkg"
  rm -rf "$tmp"
}
install_pinned zigg          "$ZIGG_SHA"
install_pinned RcppZiggurat  "$ZIGGURAT_SHA"
install_pinned Rfast         "$RFAST_SHA"
install_pinned HDRFA         "$HDRFA_SHA"

# --------------------------------------------- the author's own scripts ----
rm -rf "$DEST/replication"
git clone -q https://github.com/yingjieum/Replication_NonlinearFactorModel_2023.git \
  "$DEST/replication"
git -C "$DEST/replication" checkout -q "$FENG_SHA"

echo "Feng replication code at $DEST/replication (pinned $FENG_SHA)"
Rscript -e 'for (p in c("Rfast","irlba","HDRFA")) cat(p, as.character(packageVersion(p)), "\n")'
