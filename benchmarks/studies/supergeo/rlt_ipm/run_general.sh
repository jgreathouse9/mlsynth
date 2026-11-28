#!/usr/bin/env bash
# Step 1 after the stall RCA: does the working configuration (certified facial reduction, per-entry shift, centring
# guard, no removal before convergence) hold beyond the 24-market seed-0 arm? For one arm (n, panel seed):
#   1. the full LP's value by HiGHS interior point (the reference);
#   2. the certified facial reduction (rca_facial2.py);
#   3. sparse_ipm_rca.py with the working configuration on the face, and the same fixes on the unreduced LP.
# Each step logs to its own file in OUT; one summary line per step is appended to OUT/summary.log as it ends.
# usage: run_general.sh <worktree> <python> <OUT> <n> <seed>
set -u
WT=$1; PY=$2; OUT=$3; N=$4; SEED=$5
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; cd "$OUT"
TAG="n${N}_s${SEED}"
say() { echo "$(date -u +%H:%M:%S) $TAG $*" >> summary.log; }
say "start"
PANEL_SEED=$SEED "$PY" -I -u "$HERE/rlt_scale.py" "$WT" "$N" ipm 5400 > "ref_$TAG.log" 2>&1
REF=$(grep -o "objective [0-9.]*" "ref_$TAG.log" | head -1 | awk '{print $2}')
say "reference $(grep -E 'ipm \{' ref_$TAG.log | sed 's/^\[[^]]*\] //')"
if [ -z "$REF" ]; then say "no reference, stopping"; exit 1; fi
PANEL_SEED=$SEED HIGHS_LOG="fr_highs_$TAG.log" "$PY" -I -u "$HERE/rca_facial2.py" "$WT" "$N" "fr_$TAG.npz" > "fr_$TAG.log" 2>&1
say "facial reduction: $(grep -E 'certified zeros|strict feasibility|rows:' fr_$TAG.log | tr '\n' ' ')"
PANEL_SEED=$SEED FR="fr_$TAG.npz" SHIFT=diagrel GUARD=1 LEAVE=off "$PY" -I -u "$HERE/sparse_ipm_rca.py" "$WT" "$N" "$REF" \
    "face_$TAG.log" 0.5 0.1 0.001 0.25 200 > /dev/null 2>&1
say "face, fixes, no removal: $(grep -E 'Solution status|Bound B|Time ' face_$TAG.log | tr -s ' ' | tr '\n' ' ')"
PANEL_SEED=$SEED SHIFT=diagrel GUARD=1 LEAVE=off "$PY" -I -u "$HERE/sparse_ipm_rca.py" "$WT" "$N" "$REF" \
    "full_$TAG.log" 0.5 0.1 0.001 0.25 200 > /dev/null 2>&1
say "unreduced, fixes, no removal: $(grep -E 'Solution status|Bound B|Time ' full_$TAG.log | tr -s ' ' | tr '\n' ' ')"
say "done"
