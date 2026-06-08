"""Tolerance-based comparison and reporting for benchmark cases."""
from __future__ import annotations

from typing import Dict, Tuple


class BenchmarkSkipped(Exception):
    """Raised by a case's ``run()`` when an optional dependency or reference
    implementation is unavailable (e.g. ``causaltensor`` not installed, or the
    reference repo could not be cloned). The driver reports it as ``SKIP`` --
    distinct from a numerical ``FAIL`` -- so a missing optional toolchain never
    turns the suite red.
    """


def compare(got: Dict[str, float],
            expected: Dict[str, Tuple[float, float]]) -> Tuple[bool, str]:
    """Compare a case's numbers to (expected, abs_tol). Returns (ok, report)."""
    rows, ok_all = [], True
    for key, (exp, tol) in expected.items():
        if key not in got:
            rows.append(f"  MISSING  {key}: expected {exp}"); ok_all = False; continue
        diff = abs(got[key] - exp)
        ok = diff <= tol
        ok_all &= ok
        rows.append(f"  {'PASS' if ok else 'FAIL'}  {key}: "
                    f"got {got[key]:.4g}  exp {exp:.4g}  |Δ|={diff:.4g} (tol {tol:g})")
    return ok_all, "\n".join(rows)
