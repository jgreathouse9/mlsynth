#!/usr/bin/env bash
# Runs a script of this directory inside the trimmed_match reference environment (install.sh), with the
# pinned clone importable; prints what the script prints (JSON).
#   bash benchmarks/reference/trimmed_match/run.sh pairs.py response
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cache="$here/../.cache"
py="${MLSYNTH_TRIMMED_MATCH_PYTHON:-$cache/trimmed_match_venv/bin/python}"
clone="$cache/trimmed_match"
if [ ! -x "$py" ] || [ ! -d "$clone/trimmed_match" ]; then
  echo "trimmed_match reference environment missing: run benchmarks/reference/trimmed_match/install.sh" >&2
  exit 3
fi
script="$1"; shift
cd "$clone" && MPLBACKEND=Agg PYTHONPATH="$clone" "$py" "$here/$script" "$clone" "$@"
