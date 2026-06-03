#!/usr/bin/env bash
# Remove the 6 orphaned doc stub pages (a zip cannot delete files for you).
set -e; cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
rm -fv "docs/est.rst"
rm -fv "docs/exp.rst"
rm -fv "docs/opthelpers.rst"
rm -fv "docs/optutils.rst"
rm -fv "docs/selector.rst"
rm -fv "docs/spill.rst"
echo "Stale doc pages removed. Rebuild: python -m sphinx -b html docs _build/html"
