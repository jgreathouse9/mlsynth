#!/usr/bin/env bash
# Step 2, criterion 3: does the support stay small as pairs per row grow? 24 markets, seed 0, one LP per
# (GROUP_GAP_FACTOR, PAIR_GAP_FACTOR): the HiGHS reference, then sparse_ipm_rca.py with the fixes on the unreduced LP
# and no removal (so the peak support is every pair ever priced in). One summary line per LP in OUT/scaling.log.
# usage: run_scaling.sh <worktree> <python> <OUT> <group_factor> <pair_factor> [<group_factor> <pair_factor> ...]
set -u
WT=$1; PY=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; cd "$OUT"
while [ $# -ge 2 ]; do
  G=$1; F=$2; shift 2; TAG="g${G}_p${F}"
  GROUP_GAP_FACTOR=$G PAIR_GAP_FACTOR=$F "$PY" -I -u "$HERE/rlt_scale.py" "$WT" 24 ipm 5400 > "sref_$TAG.log" 2>&1
  REF=$(grep -o "objective [0-9.]*" "sref_$TAG.log" | head -1 | awk '{print $2}')
  SIZE=$(grep -E "pairs that fit|LP:" "sref_$TAG.log" | sed 's/^\[[^]]*\] //' | tr '\n' ' ')
  GROUP_GAP_FACTOR=$G PAIR_GAP_FACTOR=$F SHIFT=diagrel GUARD=1 LEAVE=off "$PY" -I -u "$HERE/sparse_ipm_rca.py" "$WT" 24 \
      "$REF" "scal_$TAG.log" 0.5 0.1 0.001 0.25 200 > /dev/null 2>&1
  echo "$(date -u +%H:%M:%S) $TAG | $SIZE| reference $REF | $(grep -E 'Solution status|Bound B|Support|Time ' scal_$TAG.log | tr -s ' ' | tr '\n' ' ')" >> scaling.log
done
