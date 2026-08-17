#!/usr/bin/env bash
# Install the authors' scinference R package (Chernozhukov, Wuthrich & Zhu;
# github.com/kwuthrich/scinference) for the cwz_ttest and cwz_conformal
# reference runs.
#
# Pinned at v1.0.0 (567c6889ce0a1d269a62d415b88aa6baf723a3fe) -- the tag the
# authors' own replication package installs (code/0_setup.R:
# remotes::install_github("kwuthrich/scinference@v1.0.0")). At the time of
# writing v1.0.0 and main are identical under R/, so the pin costs nothing and
# fixes what a future upstream commit could otherwise move.
#
# scinference declares CVXR in Imports, but CVXR is only used by its
# constrained-lasso estimator -- the SC t-test (sc.cf) and the SC conformal
# path both need only limSolve. We drop CVXR from Imports so the (otherwise
# heavy) CVXR chain is not required. Restore it to reach
# estimation_method = "classo".
#
# Two fetch routes, because which one works depends on the sandbox's egress
# allowlist (see agents/agents_r_environment.md): git clone is tried first and
# the codeload tarball is the fallback. In the environment this was last run in,
# codeload returned 403 and git clone worked; in an earlier one it was the other
# way round.
set -euo pipefail
REF="567c6889ce0a1d269a62d415b88aa6baf723a3fe"   # tag v1.0.0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# dependency: limSolve (uses quadprog, lpSolve, MASS -- apt carries all three as
# r-cran-quadprog / r-cran-lpsolve / r-cran-mass)
Rscript -e 'if (!"limSolve" %in% rownames(installed.packages())) quit(status=1)' \
  || { git clone --quiet https://github.com/cran/limSolve limSolve \
         || { curl -sL -o limSolve.tar.gz https://codeload.github.com/cran/limSolve/tar.gz/refs/heads/master
              tar xzf limSolve.tar.gz && mv limSolve-master limSolve; }
       R CMD INSTALL --no-docs --no-help limSolve; }

if git clone --quiet https://github.com/kwuthrich/scinference scinference; then
  git -C scinference checkout --quiet "$REF"
else
  curl -sL -o scinference.tar.gz "https://codeload.github.com/kwuthrich/scinference/tar.gz/$REF"
  tar xzf scinference.tar.gz
  mv "scinference-$REF" scinference
fi
# drop the CVXR dependency (unused by the SC t-test and the SC conformal path)
perl -0pi -e 's/Imports:\s*\n\s*limSolve,\s*\n\s*CVXR\s*\n/Imports:\n  limSolve\n/' scinference/DESCRIPTION
R CMD INSTALL --no-docs --no-help scinference
Rscript -e 'suppressMessages(library(scinference)); cat("scinference OK\n")'
