#!/bin/sh
# Both of Kathleen T. Li's released implementations of Forward DiD, run live on
# her released Hong Kong GDP panel, and merged into one reference block.
#
# R gives the point estimates; MATLAB (under Octave) gives those and the
# inference her R script never computes -- the standardized ATT, the standard
# error, the two-sided p-value and the 95 percent interval, for both the forward
# and the conventional fit. Running both also puts her two implementations
# against each other, which is a check nobody else makes.
#
# The generator parses one "== REFERENCE VALUES ==" block and stops at the next
# "== " line, so the two scripts' own markers are stripped and a single block is
# emitted, followed by a single session-info block carrying both version dumps.
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"

R_OUT=$(Rscript benchmarks/reference/fdid_hongkong/reference.R)
M_OUT=$(octave-cli --no-gui benchmarks/octave/fdid_hongkong.m \
          --data benchmarks/reference/fdid_hongkong/GDP.csv)

echo "== REFERENCE VALUES =="
echo "$R_OUT" | sed -n '/^== REFERENCE VALUES ==$/,/^== /p' | grep -v '^== '
echo "$M_OUT" | sed -n '/^== MATLAB REFERENCE VALUES ==$/,/^== /p' | grep -v '^== '
echo "== SESSION INFO =="
echo "$R_OUT" | sed -n '/^== SESSION INFO ==$/,$p' | grep -v '^== SESSION INFO =='
echo "GNU Octave $(echo "$M_OUT" | tail -1)"
