#!/usr/bin/env bash
# Builds the environment the Trimmed Match cross-validation runs google/trimmed_match in:
#   1. clones the package at the pinned commit (clone_trimmed_match.py);
#   2. makes its own virtual environment with versions of the package's era (it was tested on Python 3.7;
#      these are the oldest that install on Python 3.11);
#   3. compiles its C++ core (estimator.cc, geox_data_util.cc, math_util.cc and the pybind11 module) with
#      the system C++ compiler, in place of the package's Bazel build: the core needs only glog's CHECK and
#      absl's optional, for which shim/ holds two small headers;
#   4. runs the package's own Python test suite in that environment, so a build that departs from the
#      reference fails here, not in the benchmark.
#
#   bash benchmarks/reference/trimmed_match/install.sh [PYTHON3.11]
#
# Everything goes under benchmarks/reference/.cache (git-ignored). Needs git or network access to GitHub,
# PyPI, and a C++17 compiler.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
py="${1:-python3.11}"
cache="$here/../.cache"
venv="$cache/trimmed_match_venv"
mkdir -p "$cache"

clone="$(cd "$repo" && python3 -c 'from benchmarks.reference.clone_trimmed_match import ensure_clone; print(ensure_clone())')"
"$py" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet \
  "numpy==1.26.4" "pandas==1.5.3" "scipy==1.11.4" "matplotlib==3.8.4" "seaborn==0.12.2" \
  "Jinja2==3.1.6" "absl-py==2.5.1" "pybind11==2.13.6" "pytest"

ext="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
(cd "$clone" && "${CXX:-c++}" -O2 -std=c++17 -shared -fPIC -I"$here/shim" -I. \
  $("$venv/bin/python" -m pybind11 --includes) \
  trimmed_match/core/estimator.cc trimmed_match/core/geox_data_util.cc trimmed_match/core/math_util.cc \
  trimmed_match/core/python/estimator_ext.cc \
  -o "trimmed_match/core/python/estimator_ext$ext")

# The package's own tests (multi_cell is incomplete at this commit: its tests import a missing module).
(cd "$clone" && MPLBACKEND=Agg "$venv/bin/python" -m pytest -q -p no:cacheprovider trimmed_match \
  --ignore=trimmed_match/design/multi_cell)
echo "trimmed_match reference environment: $venv/bin/python (package at $clone)"
