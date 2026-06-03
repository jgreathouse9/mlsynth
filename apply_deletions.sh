#!/usr/bin/env bash
# Removes the three dead duplicate modules (a zip cannot delete files for you).
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
rm -fv "mlsynth/utils/ppscm_helpers/optimization.py"
rm -fv "mlsynth/utils/proximal_helpers/estimation.py"
rm -fv "mlsynth/utils/spcd_helpers/optimization.py"
echo "Dead modules removed."
