#!/usr/bin/env bash
# Check every numerical claim in the BASC West Germany referee report, from
# fresh clones and a fresh install.
#
#   ./run_all.sh              short chains, about 20 minutes
#   ./run_all.sh --full       adds the 25000/25000 chains, about 2 hours
#   ./run_all.sh --verify-only  re-check against the CSVs already here
#
# Fetches the authors' sampler and mlsynth from main, derives the patched
# sampler files from the authors' own code, runs the MCMC diagnostics, and then
# recomputes every claim the report makes and compares it against the stated
# value. Exits non-zero if any claim fails. Rendering the report is optional and
# needs Quarto; the verification does not.
set -euo pipefail
cd "$(dirname "$0")"

FULL=0; RENDER_ONLY=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    --verify-only) RENDER_ONLY=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- prerequisites
say "checking prerequisites"
MISSING=""
for c in python3 git; do have "$c" || MISSING="$MISSING $c"; done
[ -n "$MISSING" ] && { echo "missing:$MISSING" >&2; exit 1; }

have Rscript || {
  echo "Rscript not found. The MCMC diagnostics need R 4.x; install it and re-run." >&2
  echo "Debian/Ubuntu: apt-get install r-base-core r-cran-mass" >&2
  exit 1; }

have quarto || echo "quarto absent: the report will not be rendered. Verification does not need it."
echo "python3 $(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))'), $(Rscript -e 'cat(R.version.string)' 2>/dev/null)"

# ------------------------------------------------------------------ python deps
if [ "$RENDER_ONLY" -eq 0 ]; then
  say "installing python dependencies"
  # PyPI carries mlsynth 1.0.0; the report needs 2.x, so install from source.
  python3 -c 'import mlsynth,sys; sys.exit(0 if mlsynth.__version__.startswith("2.") else 1)' 2>/dev/null \
    || python3 -m pip install --quiet "mlsynth[bayes] @ git+https://github.com/jgreathouse9/mlsynth@main"
  python3 -m pip install --quiet jupyter matplotlib tabulate cvxpy
  python3 -c 'import mlsynth; print("mlsynth", mlsynth.__version__)'
fi

# ------------------------------------------------------------ R prerequisite
if [ "$RENDER_ONLY" -eq 0 ]; then
  Rscript -e 'if (!requireNamespace("MASS", quietly=TRUE)) { cat("MASS missing\n"); quit(status=1) }' \
    || { echo "R package MASS is required (apt-get install r-cran-mass)." >&2; exit 1; }
fi

# ------------------------------------------------------------- authors' sampler
if [ "$RENDER_ONLY" -eq 0 ]; then
  say "fetching the authors' sampler"
  [ -d paper-BASC ] || git clone --depth 1 https://github.com/sll-lee/paper-BASC paper-BASC
  python3 scripts/prepare_sampler.py paper-BASC/BASC_realdata.R
  python3 scripts/export_inputs.py basedata/repgermany.dta
fi

# ------------------------------------------------------------------ diagnostics
run_if_absent() {   # run_if_absent <sentinel csv> <script> <burnin> <sampling> <seed>
  if [ -f "$1" ]; then echo "  present, skipping: $1"; return; fi
  echo "  $2  $3/$4  seed $5"
  Rscript "scripts/$2" "$3" "$4" "$5" >"logs/${2%.R}_$3_$5.log" 2>&1 \
    || { echo "FAILED: $2 (see logs/${2%.R}_$3_$5.log)" >&2; exit 1; }
}

if [ "$RENDER_ONLY" -eq 0 ]; then
  mkdir -p logs
  say "BASC posterior paths (basc_run.R)"
  if [ -f data/basc_counterfactual.csv ]; then echo "  present, skipping"
  else Rscript scripts/basc_run.R >logs/basc_run.log 2>&1 || { echo "FAILED: basc_run.R" >&2; exit 1; }; fi

  say "MCMC diagnostics at 2000/2000, seeds 100 200 300"
  for s in 100 200 300; do
    run_if_absent "gamma1_results_2000_$s.csv" run_gamma1.R 2000 2000 "$s"
    run_if_absent "q_results_2000_$s.csv"      run_q.R      2000 2000 "$s"
    run_if_absent "decomp_2000_$s.csv"         run_decomp.R 2000 2000 "$s"
  done
  run_if_absent "init_results_2000_200.csv" run_init.R 2000 2000 200

  if [ "$FULL" -eq 1 ]; then
    say "long chains at 25000/25000, seed 200 (about 25 minutes each)"
    run_if_absent "gamma1_results_25000_200.csv" run_gamma1.R 25000 25000 200
    run_if_absent "q_results_25000_200.csv"  run_q.R      25000 25000 200
  else
    echo "  (skipping the 25000/25000 chains; pass --full to run them)"
  fi

  say "consolidating into data/"
  python3 scripts/collect_results.py
fi

# ------------------------------------------------------------------ verification
say "checking every claim in the report"
MCMC_FLAG=""
if [ -f data/gamma1_diagnostic.csv ]; then MCMC_FLAG="--with-mcmc"; fi
# verify.py exits non-zero when a claim fails; capture that instead of aborting
STATUS=0
python3 verify.py $MCMC_FLAG --json verification.json || STATUS=$?

# ------------------------------------------------------------- optional render
if have quarto; then
  say "rendering the report"
  quarto render basc_westgermany_review.qmd || echo "render failed; verification above is unaffected"
fi

say "done"
exit $STATUS
